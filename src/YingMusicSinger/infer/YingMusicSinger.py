import hydra
import torch
import torch.nn as nn
import torchaudio
from einops import rearrange
from ema_pytorch import EMA
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf

from src.YingMusicSinger.melody.midi_extractor import MIDIExtractor
from src.YingMusicSinger.models.model import Singer
from src.YingMusicSinger.utils.cnen_tokenizer import CNENTokenizer
from src.YingMusicSinger.utils.lrc_align import (
    align_lrc_put_to_front,
    align_lrc_sentence_level,
)
from src.YingMusicSinger.utils.mel_spectrogram import MelodySpectrogram
from src.YingMusicSinger.utils.stable_audio_tools.vae_copysyn import StableAudioInfer
from src.YingMusicSinger.utils.smooth_ending import smooth_ending

class YingMusicSinger(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        model_cfg_path,
        ckpt_path=None,
        vae_config_path=None,
        vae_ckpt_path=None,
        midi_teacher_ckpt_path=None,
        is_distilled=False,
        use_ema=True,
    ):
        super().__init__()
        self.cfg = OmegaConf.load(model_cfg_path)
        model_cls = hydra.utils.get_class(
            f"src.YingMusicSinger.models.{self.cfg.model.backbone}"
        )
        self.melody_input_source = self.cfg.model.melody_input_source
        self.is_tts_pretrain = self.cfg.model.is_tts_pretrain

        self.model = Singer(
            transformer=model_cls(
                **self.cfg.model.arch,
                text_num_embeds=self.cfg.datasets_cfg.text_num_embeds,
                mel_dim=self.cfg.model.mel_spec.n_mel_channels,
                use_guidance_scale_embed=is_distilled,
            ),
            mel_spec_kwargs=self.cfg.model.mel_spec,
            is_tts_pretrain=self.is_tts_pretrain,
            melody_input_source=self.melody_input_source,
            cka_disabled=self.cfg.model.cka_disabled,
            num_channels=None,
            extra_parameters=self.cfg.extra_parameters,
            distill_stage=1,
            use_guidance_scale_embed=is_distilled,
        )

        self.vae = StableAudioInfer(
            model_config_path=vae_config_path,
            model_ckpt_path=vae_ckpt_path,
        )

        self._need_midi = self.melody_input_source in {
            "some_pretrain",
            "some_pretrain_fuzzdisturb",
            "some_pretrain_postprocess_embedding",
        }
        self.midi_teacher = None
        if self._need_midi:
            self.midi_teacher = MIDIExtractor()
            if midi_teacher_ckpt_path is not None:
                self.midi_teacher._load_form_ckpt(midi_teacher_ckpt_path)
            for p in self.midi_teacher.parameters():
                p.requires_grad = False

            self.melody_spectrogram_extract = MelodySpectrogram()

        self.vae_frame_rate = 44100 / 2048

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if use_ema:
                ema_model = EMA(self.model, include_online_model=False)
                ema_model.load_state_dict(ckpt["ema_model_state_dict"])

                self.model = ema_model.ema_model
            else:
                self.model.load_state_dict(ckpt["model_state_dict"])

        self.cnen_tokenizer = CNENTokenizer()
        self.rear_silent_time = 1.0

    @property
    def device(self):
        return next(self.parameters()).device

    def prepare_input(
        self,
        ref_audio_path,
        melody_audio_path,
        ref_text,
        target_text,
        sil_len_to_end,
        lrc_align_mode,
    ):
        ref_audio, ref_audio_sr = torchaudio.load(ref_audio_path)
        silence = torch.zeros(ref_audio.shape[0], int(ref_audio_sr * sil_len_to_end))
        ref_wav = torch.cat([ref_audio, silence], dim=1)
        ref_latent = self.vae.encode_audio(ref_wav, in_sr=ref_audio_sr).transpose(
            1, 2
        )  # [B, T, D]


        melody_audio, melody_sr = torchaudio.load(melody_audio_path)
        silence = torch.zeros(melody_audio.shape[0], int(melody_sr * self.rear_silent_time))
        melody_wav = torch.cat([melody_audio, silence], dim=1)
        melody_latent = self.vae.encode_audio(melody_wav, in_sr=melody_sr).transpose(
            1, 2
        )  # [B, T, D]

        midi_in = torch.cat([ref_latent, melody_latent], dim=1)
        if self.is_tts_pretrain:
            midi_in = torch.zeros_like(midi_in)

        ref_latent_len = ref_latent.shape[1]
        total_len = int(ref_latent.shape[1] + melody_latent.shape[1])

        if self._need_midi:
            ref_mel = self.melody_spectrogram_extract(audio=ref_wav, sr=ref_audio_sr)
            melody_mel = self.melody_spectrogram_extract(audio=melody_wav, sr=melody_sr)
            melody_mel_spec = torch.cat([ref_mel, melody_mel], dim=2)
        else:
            raise NotImplementedError()

        assert isinstance(ref_text, str) and isinstance(target_text, str)
        text_list = [ref_text] + [target_text]

        if lrc_align_mode == "put_to_front":
            lrc_token, _ = align_lrc_put_to_front(
                tokenizer=self.cnen_tokenizer,
                lrc_start_times=None,
                lrc_lines=text_list,
                total_lens=total_len,
            )
        elif lrc_align_mode == "sentence_level":
            lrc_token, _ = align_lrc_sentence_level(
                tokenizer=self.cnen_tokenizer,
                lrc_start_times=[0.0, ref_latent_len / self.vae_frame_rate],
                lrc_lines=text_list,
                total_lens=total_len,
                vae_frame_rate=self.vae_frame_rate,
            )
        else:
            raise ValueError(f"Unsupported lrc_align_mode: {lrc_align_mode}")

        text_tokens = (
            torch.tensor(lrc_token, dtype=torch.int64).unsqueeze(0).to(self.device)
        )

        midi_p, bound_p = None, None
        if self._need_midi:
            with torch.no_grad():
                midi_p, bound_p = self.midi_teacher(melody_mel_spec.transpose(1, 2))

        return (
            ref_latent,
            ref_latent_len,
            text_tokens,
            total_len,
            midi_in,
            midi_p,
            bound_p,
        )

    def forward(
        self,
        ref_audio_path,
        melody_audio_path,
        ref_text,
        target_text,
        lrc_align_mode: str = "sentence_level",
        sil_len_to_end: float = 0.5,
        t_shift: float = 0.5,
        nfe_step: int = 32,
        cfg_strength: float = 3.0,
        seed: int = 666,
        is_tts_pretrain: bool = False,
    ):
        """
        Args:
            ref_audio_path:    Path to the reference audio (for timbre)
            melody_audio_path: Path to the melody reference audio (provides target duration and melody information)
            ref_text:          Text corresponding to the reference audio
            target_text:       Target text to be synthesized
            lrc_align_mode:    Lyric alignment mode "sentence_level" | "put_to_front"
            sil_len_to_end:    Duration of silence appended to the end of the reference audio (seconds)
            t_shift:           Sampling time offset
            nfe_step:          ODE sampling steps
            cfg_strength:      CFG strength
            seed:              Random seed
            is_tts_pretrain:   If True, melody is not provided (TTS mode)
        """
        ref_latent, ref_latent_len, text_tokens, total_len, midi_in, midi_p, bound_p = (
            self.prepare_input(
                ref_audio_path=ref_audio_path,
                melody_audio_path=melody_audio_path,
                ref_text=ref_text,
                target_text=target_text,
                sil_len_to_end=sil_len_to_end,
                lrc_align_mode=lrc_align_mode,
            )
        )

        assert midi_p is not None and bound_p is not None
        with torch.inference_mode():
            generated_latent, _ = self.model.sample(
                cond=ref_latent,
                midi_in=midi_in,
                text=text_tokens,
                duration=total_len,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=None,
                use_epss=False,
                seed=seed,
                midi_p=midi_p,
                t_shift=t_shift,
                bound_p=bound_p,
                guidance_scale=cfg_strength,
            )
        generated_latent = generated_latent.to(torch.float32)
        generated_latent = generated_latent[:, ref_latent_len: -int(self.vae_frame_rate*self.rear_silent_time), :]
        generated_latent = generated_latent.permute(0, 2, 1)  # [B, D, T]

        generated_audio = self.vae.decode_audio(generated_latent)
        audio = rearrange(generated_audio, "b d n -> d (b n)")

        audio = audio.to(torch.float32).cpu()
        audio = smooth_ending(audio, 44100)
        return audio, 44100


if __name__ == "__main__":
    # === Export to HuggingFace safetensors (optional) ===
    # model = YingMusicSinger(
    #     model_cfg_path="src/YingMusicSinger/config/YingMusic_Singer.yaml",
    #     ckpt_path="ckpts/YingMusicSinger_model.pt",
    #     vae_config_path="src/YingMusicSinger/config/stable_audio_2_0_vae_20hz_official.json",
    #     vae_ckpt_path="ckpts/stable_audio_2_0_vae_20hz_official.ckpt",
    #     midi_teacher_ckpt_path="ckpts/model_ckpt_steps_100000_simplified.ckpt",
    # )
    # model.save_pretrained("path/to/save")

    # === Inference Example ===
    model = YingMusicSinger.from_pretrained("ASLP-lab/YingMusic-Singer")
    model.to("cuda:0")
    model.eval()

    waveform, sample_rate = model(
        ref_audio_path="path/to/ref_audio",  # Timbre reference audio
        melody_audio_path="path/to/melody_audio",  # Melody-providing singing clip
        ref_text="oh the reason i hold on",  # Lyrics corresponding to ref_audio
        target_text="oldest book broken watch|bare feet in grassy spot",  # Modified target lyrics
        seed=42,
    )

    torchaudio.save("output.wav", waveform, sample_rate=sample_rate)
    print("Saved to output.wav")
