from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from src.YingMusicSinger.melody.midi_extractor import MIDIExtractor
from src.YingMusicSinger.utils.common import (
    default,
    exists,
    get_epss_timesteps,
    lens_to_mask,
)


def interpolation_midi_continuous(midi_p, bound_p, total_len):
    """Temporally interpolate 3D melody latent to match target length."""
    if midi_p.shape[1] != total_len:
        midi = (
            F.interpolate(
                midi_p.clone().detach().transpose(1, 2),
                size=total_len,
                mode="linear",
                align_corners=False,
            )
            .transpose(1, 2)
            .clone()
            .detach()
        )
        if bound_p is not None:
            midi_bound = (
                F.interpolate(
                    bound_p.clone().detach().transpose(1, 2),
                    size=total_len,
                    mode="linear",
                    align_corners=False,
                )
                .transpose(1, 2)
                .clone()
                .detach()
            )
    else:
        midi = midi_p.clone().detach()
        if bound_p is not None:
            midi_bound = bound_p.clone().detach()
    if bound_p is not None:
        return midi, midi_bound
    else:
        return midi


def interpolation_midi_continuous_2_dim(midi_p, bound_p, total_len):
    """Temporally interpolate 2D melody latent to match target length."""
    assert len(midi_p.shape) == 2

    if midi_p.shape[1] != total_len:
        midi = (
            F.interpolate(
                midi_p.unsqueeze(2).clone().detach().transpose(1, 2),
                size=total_len,
                mode="linear",
                align_corners=False,
            )
            .transpose(1, 2)
            .clone()
            .detach()
        )
        if bound_p:
            midi_bound = (
                F.interpolate(
                    bound_p.unsqueeze(2).clone().detach().transpose(1, 2),
                    size=total_len,
                    mode="linear",
                    align_corners=False,
                )
                .transpose(1, 2)
                .clone()
                .detach()
            )
    else:
        midi = midi_p.clone().detach()
        if bound_p:
            midi_bound = bound_p.clone().detach()
    if bound_p:
        return midi.squeeze(2), midi_bound.squeeze(2)
    else:
        return midi.squeeze(2)


class Singer(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        is_tts_pretrain,
        melody_input_source,
        cka_disabled,
        distill_stage,
        use_guidance_scale_embed,
        sigma=0.0,
        odeint_kwargs: dict = dict(method="euler"),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        extra_parameters=None,
    ):
        super().__init__()

        self.is_tts_pretrain = is_tts_pretrain

        if distill_stage is None:
            self.distill_stage = 0
        else:
            self.distill_stage = int(distill_stage)

        self.use_guidance_scale_embed = use_guidance_scale_embed

        assert melody_input_source in {
            "student_model",
            "some_pretrain",
            "some_pretrain_fuzzdisturb",
            "some_pretrain_postprocess_embedding",
            "none",
        }
        from src.YingMusicSinger.melody.SmoothMelody import MIDIFuzzDisturb

        if melody_input_source == "some_pretrain_fuzzdisturb":
            self.smoothMelody_MIDIFuzzDisturb = MIDIFuzzDisturb(
                dim=extra_parameters.some_pretrain_fuzzdisturb.dim,
                drop_prob=extra_parameters.some_pretrain_fuzzdisturb.drop_prob,
                noise_scale=extra_parameters.some_pretrain_fuzzdisturb.noise_scale,
                blur_kernel=extra_parameters.some_pretrain_fuzzdisturb.blur_kernel,
                drop_type=extra_parameters.some_pretrain_fuzzdisturb.drop_type,
            )
        from src.YingMusicSinger.melody.SmoothMelody import MIDIDigitalEmbedding

        if melody_input_source == "some_pretrain_postprocess_embedding":
            self.smoothMelody_MIDIDigitalEmbedding = MIDIDigitalEmbedding(
                embed_dim=extra_parameters.some_pretrain_postprocess_embedding.embed_dim,
                num_classes=extra_parameters.some_pretrain_postprocess_embedding.num_classes,
                mark_distinguish_scale=extra_parameters.some_pretrain_postprocess_embedding.mark_distinguish_scale,
            )

        self.melody_input_source = melody_input_source
        self.cka_disabled = cka_disabled

        self.frac_lengths_mask = frac_lengths_mask

        num_channels = default(num_channels, mel_spec_kwargs.n_mel_channels)
        self.num_channels = num_channels

        # Classifier-free guidance drop probabilities
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # Transformer backbone
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # Conditional flow matching
        self.sigma = sigma
        self.odeint_kwargs = odeint_kwargs

        # Melody extractor
        self.midi_extractor = MIDIExtractor(in_dim=num_channels)

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"] | None = None,  # noqa: F821
        *,
        midi_in: float["b n d"] | None = None,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,  # Maximum total length (including ICL prompt), ~190s
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        t_shift=1.0,  # Sampling timestep shift (ZipVoice-style)
        guidance_scale=None,
        edit_mask=None,
        midi_p=None,
        bound_p=None,
        enable_melody_control=True,
    ):
        self.eval()

        assert isinstance(cond, torch.Tensor)
        assert not edit_mask, "edit_mask is not supported in this mode"
        assert not duplicate_test, "duplicate_test is not supported in this mode"

        if self.melody_input_source == "student_model":
            assert midi_p is None and bound_p is None
        elif self.melody_input_source in {
            "some_pretrain",
            "some_pretrain_fuzzdisturb",
            "some_pretrain_postprocess_embedding",
        }:
            assert midi_p is not None and bound_p is not None
        elif self.melody_input_source == "none":
            assert midi_p is None and bound_p is None
        else:
            raise ValueError(
                f"Unsupported melody_input_source: {self.melody_input_source}"
            )

        # duration is the total latent sequence length
        assert duration

        cond = cond.to(next(self.parameters()).dtype)

        # Extract or interpolate melody representation
        if self.melody_input_source == "student_model":
            midi, midi_bound = self.midi_extractor(midi_in)

        elif self.melody_input_source == "some_pretrain":
            midi, midi_bound = interpolation_midi_continuous(
                midi_p=midi_p, bound_p=bound_p, total_len=text.shape[1]
            )
        elif self.melody_input_source == "some_pretrain_fuzzdisturb":
            midi, midi_bound = interpolation_midi_continuous(
                midi_p=midi_p, bound_p=bound_p, total_len=text.shape[1]
            )
            midi = self.smoothMelody_MIDIFuzzDisturb(midi)

        elif self.melody_input_source == "some_pretrain_postprocess_embedding":
            midi_after_postprocess, _ = self.midi_extractor.postprocess(
                midi=midi_p, bounds=bound_p, with_expand=True
            )
            midi = interpolation_midi_continuous_2_dim(
                midi_p=midi_after_postprocess, bound_p=None, total_len=text.shape[1]
            )
            midi = self.smoothMelody_MIDIDigitalEmbedding(midi)
            midi_bound = None

        elif self.melody_input_source == "none":
            midi = torch.zeros(
                text.shape[0], text.shape[1], 128, dtype=cond.dtype, device=text.device
            )
            midi_bound = None
        else:
            raise NotImplementedError()

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        assert isinstance(text, torch.Tensor)

        cond_mask = lens_to_mask(lens)

        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        # Duration must be at least max(text_len, audio_prompt_len) + 1
        duration = torch.maximum(
            torch.maximum((text != 0).sum(dim=-1), lens) + 1, duration
        )
        duration = duration.clamp(max=max_duration)

        max_duration = duration.amax()

        # Duplicate test: interpolate between noise and conditioning
        if duplicate_test:
            test_cond = F.pad(
                cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0
            )

        # Zero-pad conditioning latent to max_duration
        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)

        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(
            cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False
        )
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

        assert max_duration == midi.shape[1]

        # Zero out melody in prompt region; optionally disable melody control entirely
        if enable_melody_control:
            midi = torch.where(cond_mask, torch.zeros_like(midi), midi)
        else:
            midi = torch.zeros_like(midi)

        if self.is_tts_pretrain:
            midi = torch.zeros_like(midi)

        # For batched inference, explicit mask prevents causal attention fallback
        if batch > 1:
            mask = lens_to_mask(duration)
        else:
            mask = None

        # ODE velocity function
        def fn(t, x):
            if cfg_strength < 1e-5:
                # No classifier-free guidance
                pred, _ = self.transformer(
                    x=x,
                    cond=step_cond,
                    text=text,
                    midi=midi,
                    time=t,
                    mask=mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    drop_midi=not enable_melody_control,
                    cache=False,
                )
                return pred
            else:
                if self.use_guidance_scale_embed:
                    # Distilled model with built-in CFG
                    assert enable_melody_control
                    pred_cfg, _ = self.transformer(
                        x=x,
                        cond=step_cond,
                        text=text,
                        midi=midi,
                        time=t,
                        mask=mask,
                        drop_audio_cond=False,
                        drop_text=False,
                        drop_midi=not enable_melody_control,
                        cache=False,
                        guidance_scale=torch.tensor([guidance_scale], device=device),
                    )
                    print(
                        f"CFG 参数调节无作用！ 蒸馏之后的，输入CFG为 guidance_scale={guidance_scale}"
                    )
                    return pred_cfg
                else:
                    # Standard CFG: cond + uncond forward
                    # BUG If enable_melody_control is False, there might be a slight issue here
                    assert guidance_scale is not None
                    pred_cfg, _ = self.transformer(
                        x=x,
                        cond=step_cond,
                        text=text,
                        midi=midi,
                        time=t,
                        mask=mask,
                        cfg_infer=True,
                        cache=False,
                        cfg_infer_ids=(True, False, False, True),
                    )

                    pred, pred_drop_all_cond = torch.chunk(pred_cfg, 2, dim=0)
                    return pred + (pred - pred_drop_all_cond) * float(guidance_scale)

        # Generate initial noise (per-sample seeding for batch consistency)
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(
                torch.randn(
                    dur, self.num_channels, device=self.device, dtype=step_cond.dtype
                )
            )
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        # Build timestep schedule
        assert not use_epss and sway_sampling_coef is None, (
            "Use timestep shift instead of the strategy in F5"
        )
        if t_start == 0 and use_epss:
            # Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(steps, device=self.device, dtype=step_cond.dtype)
        else:
            t = torch.linspace(
                t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype
            )

        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        # Apply timestep shift
        t = t_shift * t / (1 + (t_shift - 1) * t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory
