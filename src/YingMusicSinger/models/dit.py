"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from src.YingMusicSinger.models.modules import (
    AdaLayerNorm_Final,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    TimestepGuidanceEmbedding,
    get_pos_embed_indices,
    precompute_freqs_cis,
)


# Text embedding


class TextEmbedding(nn.Module):
    def __init__(
        self,
        text_num_embeds,
        text_dim,
        mask_padding=False,
        average_upsampling=False,
        conv_layers=0,
        conv_mult=2,
    ):
        super().__init__()
        self.text_embed = nn.Embedding(
            text_num_embeds + 1, text_dim
        )  # index 0 reserved as filler token

        self.mask_padding = mask_padding
        self.average_upsampling = average_upsampling  # ZipVoice-style late average upsampling (after text encoder)
        if average_upsampling:
            assert mask_padding, (
                "text_embedding_average_upsampling requires text_mask_padding to be True"
            )

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24kHz audio
            self.register_buffer(
                "freqs_cis",
                precompute_freqs_cis(text_dim, self.precompute_max_pos),
                persistent=False,
            )
            self.text_blocks = nn.Sequential(
                *[
                    ConvNeXtV2Block(text_dim, text_dim * conv_mult)
                    for _ in range(conv_layers)
                ]
            )
        else:
            self.extra_modeling = False

        print(
            f"[info] TextEmbedding: mask_padding={mask_padding}, average_upsampling={average_upsampling}, conv_layers={conv_layers}"
        )

    def average_upsample_text_by_mask(self, text, text_mask, audio_mask):
        batch, text_len, text_dim = text.shape

        if audio_mask is None:
            audio_mask = torch.ones_like(text_mask, dtype=torch.bool)
        valid_mask = audio_mask & text_mask
        audio_lens = audio_mask.sum(dim=1)  # [batch]
        valid_lens = valid_mask.sum(dim=1)  # [batch]

        upsampled_text = torch.zeros_like(text)

        for i in range(batch):
            audio_len = audio_lens[i].item()
            valid_len = valid_lens[i].item()

            if valid_len == 0:
                continue

            valid_ind = torch.where(valid_mask[i])[0]
            valid_data = text[i, valid_ind, :]  # [valid_len, text_dim]

            base_repeat = audio_len // valid_len
            remainder = audio_len % valid_len

            indices = []
            for j in range(valid_len):
                repeat_count = base_repeat + (1 if j >= valid_len - remainder else 0)
                indices.extend([j] * repeat_count)

            indices = torch.tensor(
                indices[:audio_len], device=text.device, dtype=torch.long
            )
            upsampled = valid_data[indices]  # [audio_len, text_dim]

            upsampled_text[i, :audio_len, :] = upsampled

        return upsampled_text

    def forward(
        self,
        text: int["b nt"],
        seq_len,
        drop_text=False,
        audio_mask: bool["b n"] | None = None,
    ):  # noqa: F722
        # Text tokens start from 0; shift by 1 so that 0 is never a valid token
        text = text + 1
        # Note: 1 is used as the PAD token
        text = text[
            :, :seq_len
        ]  # Truncate if text tokens exceed mel spectrogram length
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=1)

        if self.mask_padding:
            text_mask = text == 1
        else:
            text_mask = torch.zeros(
                (batch, seq_len), device=text.device, dtype=torch.bool
            )

        if drop_text:  # CFG for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # Optional extra modeling
        if self.extra_modeling:
            # Sinusoidal positional embedding
            batch_start = torch.zeros((batch,), device=text.device, dtype=torch.long)
            pos_idx = get_pos_embed_indices(
                batch_start, seq_len, max_pos=self.precompute_max_pos
            )
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # ConvNeXtV2 blocks
            if self.mask_padding:
                text = text.masked_fill(
                    text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0
                )
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(
                        text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0
                    )
            else:
                text = self.text_blocks(text)

        if self.average_upsampling:
            text = self.average_upsample_text_by_mask(text, ~text_mask, audio_mask)

        return text, text_mask


# Noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim, midi_dim=128):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim + midi_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)
        self.midi_proj = nn.Linear(128, 128)

    def forward(
        self,
        x: float["b n d"],  # noqa: F722
        cond: float["b n d"],  # noqa: F722
        text_embed: float["b n d"],  # noqa: F722
        midi,
        drop_audio_cond=False,
        drop_midi=False,
    ):
        if drop_audio_cond:  # CFG for conditioning audio
            cond = torch.zeros_like(cond)

        midi = self.midi_proj(midi)

        if drop_midi:  # CFG for melody
            midi = torch.zeros_like(midi)

        x = self.proj(torch.cat((x, cond, text_embed, midi), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using DiT blocks


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        n_f0_bins=512,
        text_mask_padding=True,
        text_embedding_average_upsampling=False,
        qk_norm=None,
        conv_layers=0,
        pe_attn_head=None,
        attn_backend="torch",  # "torch" | "flash_attn"
        attn_mask_enabled=False,
        long_skip_connection=False,
        checkpoint_activations=False,
        use_guidance_scale_embed: bool = False,
        guidance_scale_embed_dim: int = 192,
    ):
        super().__init__()

        self.time_embed = TimestepGuidanceEmbedding(
            dim,
            use_guidance_scale_embed=use_guidance_scale_embed,
            guidance_scale_embed_dim=guidance_scale_embed_dim,
        )
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed_p = TextEmbedding(
            text_num_embeds,
            text_dim,
            mask_padding=text_mask_padding,
            average_upsampling=text_embedding_average_upsampling,
            conv_layers=conv_layers,
        )
        self.text_cond, self.text_uncond = None, None  # text cache
        self.input_embed_with_midi = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)
        self.use_guidance_scale_embed = use_guidance_scale_embed

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = (
            nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        )

        self.norm_out = AdaLayerNorm_Final(dim)  # Final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out AdaLN layers in DiT blocks
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

        nn.init.zeros_(self.input_embed_with_midi.midi_proj.weight)
        nn.init.zeros_(self.input_embed_with_midi.midi_proj.bias)

    def ckpt_wrapper(self, module):
        # Ref: https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def get_input_embed(
        self,
        x,  # b n d
        cond,  # b n d
        text,  # b nt
        midi,  # b n
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        drop_midi: bool = False,
        cache: bool = True,
        audio_mask: bool["b n"] | None = None,  # noqa: F722
    ):
        seq_len = x.shape[1]

        if cache:
            if drop_text:
                if self.text_uncond is None:
                    self.text_uncond, _ = self.text_embed_p(
                        text, seq_len, drop_text=True, audio_mask=audio_mask
                    )
                text_embed = self.text_uncond
            else:
                if self.text_cond is None:
                    self.text_cond, _ = self.text_embed_p(
                        text, seq_len, drop_text=False, audio_mask=audio_mask
                    )
                text_embed = self.text_cond
        else:
            text_embed, text_mask = self.text_embed_p(
                text, seq_len, drop_text=drop_text, audio_mask=audio_mask
            )

        if midi is None:
            midi = torch.zeros(
                (x.size(0), x.size(1)), device=x.device, dtype=torch.long
            )

        x = self.input_embed_with_midi(
            x,
            cond,
            text_embed,
            midi,
            drop_audio_cond=drop_audio_cond,
            drop_midi=drop_midi,
        )

        return x, None

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    def forward(
        self,
        x: float["b n d"],  # Noised input audio  # noqa: F722
        cond: float["b n d"],  # Masked conditioning audio  # noqa: F722
        text: int["b nt"],  # Text tokens  # noqa: F722
        time: float["b"] | float[""],  # Timestep  # noqa: F821 F722
        midi: float["b n"] | None = None,  # Melody latent  # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        drop_audio_cond: bool = False,  # CFG for conditioning audio
        drop_text: bool = False,  # CFG for text
        drop_midi: bool = False,  # CFG for melody
        cfg_infer: bool = False,  # CFG inference: pack cond & uncond forward
        cache: bool = False,
        guidance_scale=None,
        cfg_infer_ids=None,  # tuple(bool): (x_cond, x_uncond, x_uncond_cc, x_drop_all_cond)
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # Timestep embedding (with optional distillation guidance scale)
        t = self.time_embed(time, guidance_scale=guidance_scale)

        if cfg_infer:  # Pack cond & uncond forward: b n d -> Kb n d
            x_cond, x_uncond, x_uncond_cc, x_drop_all_cond = None, None, None, None
            if cfg_infer_ids is None or cfg_infer_ids[0]:
                x_cond, _ = self.get_input_embed(
                    x,
                    cond,
                    text,
                    midi,
                    drop_audio_cond=False,
                    drop_text=False,
                    drop_midi=False,
                    cache=cache,
                    audio_mask=mask,
                )
            if cfg_infer_ids is None or cfg_infer_ids[1]:
                x_uncond, _ = self.get_input_embed(
                    x,
                    cond,
                    text,
                    midi,
                    drop_audio_cond=True,
                    drop_text=False,
                    drop_midi=False,
                    cache=cache,
                    audio_mask=mask,
                )
            if cfg_infer_ids is None or cfg_infer_ids[2]:
                x_uncond_cc, _ = self.get_input_embed(
                    x,
                    cond,
                    text,
                    midi,
                    drop_audio_cond=False,
                    drop_text=True,
                    drop_midi=True,
                    cache=cache,
                    audio_mask=mask,
                )
            if cfg_infer_ids is None or cfg_infer_ids[3]:
                x_drop_all_cond, _ = self.get_input_embed(
                    x,
                    cond,
                    text,
                    midi,
                    drop_audio_cond=True,
                    drop_text=True,
                    drop_midi=True,
                    cache=cache,
                    audio_mask=mask,
                )

            # Concatenate only non-None tensors
            x_list = [
                xi
                for xi in [x_cond, x_uncond, x_uncond_cc, x_drop_all_cond]
                if xi is not None
            ]
            x = torch.cat(x_list, dim=0)
            t = torch.cat([t] * len(x_list), dim=0)
            mask = torch.cat([mask] * len(x_list), dim=0) if mask is not None else None
        else:
            x, text_inner_sim_matrix = self.get_input_embed(
                x,
                cond,
                text,
                midi,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                drop_midi=drop_midi,
                cache=cache,
                audio_mask=mask,
            )

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        # Mask is all zeros during inference
        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False
                )
            else:
                x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output, text_inner_sim_matrix if not cfg_infer else None
