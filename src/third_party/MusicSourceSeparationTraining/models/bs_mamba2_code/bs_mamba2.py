if __name__ == "__main__":
    import os

    gpu_use = "0"
    print("GPU use: {}".format(gpu_use))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


import sys

sys.path.append("..")
from functools import partial

import torch
import torch.nn.functional as F
from beartype import beartype
from beartype.typing import Tuple
from einops import pack, rearrange, unpack
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.ops.triton.layer_norm import RMSNorm as MRMSNorm
from models.bs_mamba2_code.attend_mamba import Attend
from rotary_embedding_torch import RotaryEmbedding
from torch import nn
from torch.nn import Module, ModuleList

# helper functions


def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class Rearrange(Module):
    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern

    def forward(self, x):
        return rearrange(x, self.pattern)


class FeedForward(Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MambaBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        n_layer=1,
        bidirectional=False,
        mamba_type=Mamba,
        mlp_cls=nn.Identity,
    ):
        super(MambaBlock, self).__init__()
        self.forward_blocks = nn.ModuleList([])
        for i in range(n_layer):
            self.forward_blocks.append(
                Block(
                    in_channels,
                    mixer_cls=partial(mamba_type, layer_idx=i, expand=4),
                    norm_cls=partial(MRMSNorm, eps=1e-5),
                    fused_add_norm=False,
                    mlp_cls=mlp_cls,
                )
            )
        if bidirectional:
            self.backward_blocks = nn.ModuleList([])
            for i in range(n_layer):
                self.backward_blocks.append(
                    Block(
                        in_channels,
                        mixer_cls=partial(mamba_type, layer_idx=i, expand=4),
                        norm_cls=partial(MRMSNorm, eps=1e-5),
                        fused_add_norm=False,
                        mlp_cls=nn.Identity,
                    )
                )
        self.linear = nn.ConvTranspose1d(
            in_channels=in_channels * 2,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
        )
        self.apply(partial(_init_weights, n_layer=n_layer))

    def forward(self, input):
        for_residual = None
        forward_f = input.clone()
        for block in self.forward_blocks:
            forward_f, for_residual = block(
                forward_f, for_residual, inference_params=None
            )
        residual = (forward_f + for_residual) if for_residual is not None else forward_f

        if self.backward_blocks is not None:
            back_residual = None
            backward_f = torch.flip(input, [1])
            for block in self.backward_blocks:
                backward_f, back_residual = block(
                    backward_f, back_residual, inference_params=None
                )
            back_residual = (
                (backward_f + back_residual)
                if back_residual is not None
                else backward_f
            )

            back_residual = torch.flip(back_residual, [1])
            residual = torch.cat([residual, back_residual], -1)
        residual = self.linear(residual.transpose(1, 2)).transpose(1, 2).contiguous()

        return residual + input


class Attention(Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        rotary_embed=None,
        flash=True,
        idx=None,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        self.attend = Attend(flash=flash, dropout=dropout, idx=idx)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False), nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)

        q, k, v = rearrange(
            self.to_qkv(x), "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads
        )

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        gates = self.to_gates(x)
        out = out * rearrange(gates, "b n h -> b h n 1").sigmoid()

        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


class Transformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
        norm_output=True,
        rotary_embed=None,
        flash_attn=True,
        idx=None,
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(
                ModuleList(
                    [
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            rotary_embed=rotary_embed,
                            flash=flash_attn,
                            idx=idx,
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


# bandsplit module
class BandSplit(Module):
    @beartype
    def __init__(self, dim, dim_inputs: Tuple[int, ...]):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(RMSNorm(dim_in), nn.Linear(dim_in, dim))

            self.to_features.append(net)

    def forward(self, x):
        x = x.split(self.dim_inputs, dim=-1)

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        return torch.stack(outs, dim=-2)


def MLP(dim_in, dim_out, dim_hidden=None, depth=1, activation=nn.Tanh):
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)


class MaskEstimator(Module):
    @beartype
    def __init__(self, dim, dim_inputs: Tuple[int, ...], depth, mlp_expansion_factor=4):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            net = []

            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth), nn.GLU(dim=-1)
            )
            self.to_freqs.append(mlp)

    def forward(self, x):
        x = x.unbind(dim=-2)

        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim=-1)


DEFAULT_FREQS_PER_BANDS = (
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    12,
    12,
    12,
    12,
    12,
    12,
    12,
    12,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    48,
    48,
    48,
    48,
    48,
    48,
    48,
    48,
    128,
    129,
)


class BSMamba2Model(Module):
    @beartype
    def __init__(
        self,
        dim,
        *,
        depth,
        stereo=False,
        num_stems=1,
        time_module_depth=1,
        freq_module_depth=1,
        freqs_per_bands=DEFAULT_FREQS_PER_BANDS,
        # in the paper, they divide into ~60 bands, test with 1 for starters
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        flash_attn=True,
        stft_n_fft=2048,
        stft_hop_length=512,
        stft_win_length=2048,
        stft_normalized=False,
        mask_estimator_depth=2,
        multi_stft_resolution_loss_weight=1.0,
        multi_stft_resolutions_window_sizes: Tuple[int, ...] = (
            4096,
            2048,
            1024,
            512,
            256,
        ),
        multi_stft_hop_size=147,
        multi_stft_normalized=False,
        module_type="transformer",
        mamba_gmlp=False,
        audio_cfg=None,
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.dim = dim
        self.stft_n_fft = stft_n_fft
        self.stft_hop_length = stft_hop_length
        self.stft_win_length = stft_win_length
        self.stft_normalized = stft_normalized
        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = torch.hann_window
        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size, normalized=multi_stft_normalized
        )

        self.layers = ModuleList([])

        self.stft_window_fn = partial(torch.hann_window, 2048)

        if module_type == "transformer" or module_type == "mamba2-roformer":
            transformer_kwargs = dict(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                flash_attn=flash_attn,
                norm_output=False,
            )

            time_rotary_embed = RotaryEmbedding(dim=dim_head)
            freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        for i in range(depth):
            if module_type == "transformer":
                self.layers.append(
                    nn.ModuleList(
                        [
                            Transformer(
                                depth=time_module_depth,
                                rotary_embed=time_rotary_embed,
                                **transformer_kwargs,
                                idx=i,
                            ),
                            Transformer(
                                depth=freq_module_depth,
                                rotary_embed=freq_rotary_embed,
                                **transformer_kwargs,
                                idx=i,
                            ),
                        ]
                    )
                )
            elif module_type == "mamba":
                if mamba_gmlp:
                    mlp_cls = partial(GatedMLP, out_features=dim)
                else:
                    mlp_cls = nn.Identity

                self.layers.append(
                    nn.ModuleList(
                        [
                            MambaBlock(
                                in_channels=dim,
                                n_layer=1,
                                bidirectional=True,
                                mamba_type=Mamba,
                                mlp_cls=mlp_cls,
                            ),
                            MambaBlock(
                                in_channels=dim,
                                n_layer=1,
                                bidirectional=True,
                                mamba_type=Mamba,
                                mlp_cls=mlp_cls,
                            ),
                        ]
                    )
                )
            elif module_type == "mamba2":
                if mamba_gmlp:
                    mlp_cls = partial(GatedMLP, out_features=dim)
                else:
                    mlp_cls = nn.Identity

                self.layers.append(
                    nn.ModuleList(
                        [
                            # just make sure d_model * expand / headdim = multiple of 8
                            MambaBlock(
                                in_channels=dim,
                                n_layer=1,
                                bidirectional=True,
                                mamba_type=Mamba2,
                                mlp_cls=mlp_cls,
                            ),
                            MambaBlock(
                                in_channels=dim,
                                n_layer=1,
                                bidirectional=True,
                                mamba_type=Mamba2,
                                mlp_cls=mlp_cls,
                            ),
                        ]
                    )
                )
            elif module_type == "mamba2-roformer":
                if mamba_gmlp:
                    mlp_cls = partial(GatedMLP, out_features=dim)
                else:
                    mlp_cls = nn.Identity

                self.layers.append(
                    nn.ModuleList(
                        [
                            # just make sure d_model * expand / headdim = multiple of 8
                            MambaBlock(
                                in_channels=dim,
                                n_layer=1,
                                bidirectional=True,
                                mamba_type=Mamba2,
                                mlp_cls=mlp_cls,
                            ),
                            Transformer(
                                depth=freq_module_depth,
                                rotary_embed=freq_rotary_embed,
                                **transformer_kwargs,
                                idx=i,
                            ),
                        ]
                    )
                )
            else:
                print("module_type must be mamba or transformer")
                exit()

        self.final_norm = RMSNorm(dim)

        assert len(freqs_per_bands) > 1

        freqs_per_bands_with_complex = tuple(
            2 * f * self.audio_channels for f in freqs_per_bands
        )

        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)

        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
            )

            self.mask_estimators.append(mask_estimator)

    def forward(
        self,
        x: torch.Tensor,
        target=None,
        active_stem_ids=None,
        return_loss_breakdown=False,
    ):
        """
        Input and output are T-F complex-valued features.
        Input shape: batch_size, n_channels, freq, time]
        Output shape: batch_size, n_channels, freq, time]
        """

        raw_audio_length = x.shape[-1]
        device = x.device

        if x.ndim == 2:
            x = rearrange(x, "b t -> b 1 t")

        x, batch_audio_channel_packed_shape = pack_one(x, "* t")

        self.stft_kwargs = dict(
            n_fft=self.stft_n_fft,
            hop_length=self.stft_hop_length,
            win_length=self.stft_win_length,
            normalized=self.stft_normalized,
            center=True,
        )

        stft_window = self.stft_window_fn(device=device)

        x = torch.stft(x, **self.stft_kwargs, window=stft_window, return_complex=True)

        x = unpack_one(x, batch_audio_channel_packed_shape, "* f t")

        # x: b c f t
        stft_repr = torch.view_as_real(x)

        # x: b c f t 2
        stft_repr = rearrange(stft_repr, "b c f t p -> b (f c) t p")

        x = rearrange(stft_repr, "b f t p -> b t (f p)")
        x = self.band_split(x)

        # axial / hierarchical attention
        for time_module, freq_module in self.layers:
            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            x = time_module(x)

            (x,) = unpack(x, ps, "* t d")
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")

            x = freq_module(x)

            (x,) = unpack(x, ps, "* f d")

        x = self.final_norm(x)

        if active_stem_ids is None:
            heads = self.mask_estimators
            stem_ids = list(range(len(self.mask_estimators)))
        else:
            heads = [self.mask_estimators[i] for i in active_stem_ids]
            stem_ids = active_stem_ids

        num_stems = len(heads)

        mask = torch.stack([fn(x) for fn in heads], dim=1)

        mask = rearrange(mask, "b n t (f p) -> b n f t p", p=2)

        stft_repr = rearrange(stft_repr, "b f t p -> b 1 f t p")

        stft_repr = torch.view_as_complex(stft_repr)
        mask = torch.view_as_complex(mask.to(torch.float32))

        stft_repr = stft_repr * mask

        stft_repr = rearrange(
            stft_repr, "b n (f s) t -> (b n s) f t", s=self.audio_channels
        )

        recon_audio = torch.istft(
            stft_repr,
            **self.stft_kwargs,
            window=stft_window,
            return_complex=False,
            length=raw_audio_length,
        )

        recon_audio = rearrange(
            recon_audio, "(b n s) t -> b n s t", s=self.audio_channels, n=num_stems
        )

        if not exists(target):
            return recon_audio

        if target.ndim == 2:
            target = rearrange(target, "... t -> ... 1 t")

        target = target[
            ..., : recon_audio.shape[-1]
        ]  # protect against lost length on istft

        target_sel = target[:, stem_ids]
        loss = F.l1_loss(recon_audio, target_sel)

        multi_stft_resolution_loss = 0.0

        for window_size in self.multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(
                    window_size, self.multi_stft_n_fft
                ),  # not sure what n_fft is across multi resolution stft
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )

            recon_Y = torch.stft(
                rearrange(recon_audio, "b n s t -> (b n s) t"), **res_stft_kwargs
            )
            target_Y = torch.stft(
                rearrange(target_sel, "b n s t -> (b n s) t"), **res_stft_kwargs
            )

            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(
                recon_Y, target_Y
            )

        weighted_multi_resolution_loss = (
            multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight
        )

        total_loss = loss + weighted_multi_resolution_loss

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (loss, multi_stft_resolution_loss)


if __name__ == "__main__":
    batch_size, n_channels, freq, time = 1, 2, 1025, 800
    in_features = torch.rand(
        batch_size, n_channels, freq, time, dtype=torch.cfloat
    ).cuda()
    cfg = {
        "depth": 2,
        "dim": 256,
        "stereo": True,
        "ff_dropout": 0.1,
        "attn_dropout": 0.1,
        "flash_attn": True,
        "module_type": "transformer",
        "dim_head": 48,
        "num_stems": 3,
    }
    model = BSModel(**cfg).cuda()
    print(
        f"Total number of parameters: {sum([p.numel() for p in model.mask_estimators.parameters()])}"
    )

    with torch.cuda.amp.autocast():
        out_features = model(in_features)

    print(f"In shape: {in_features.shape}\nOut shape: {out_features.shape}")
    print(f"In dtype: {in_features.dtype}\nOut dtype: {out_features.dtype}")
    exit()
