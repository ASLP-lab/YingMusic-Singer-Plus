import json

import torch
import torchaudio.transforms as T
from torch import nn

from .autoencoders import create_autoencoder_from_config
from .utils import load_ckpt_state_dict


class PadCrop(nn.Module):
    def __init__(self, n_samples, randomize=True):
        super().__init__()
        self.n_samples = n_samples
        self.randomize = randomize

    def __call__(self, signal):
        n, s = signal.shape
        start = (
            0
            if (not self.randomize)
            else torch.randint(0, max(0, s - self.n_samples) + 1, []).item()
        )
        end = start + self.n_samples
        output = signal.new_zeros([n, self.n_samples])
        output[:, : min(s, self.n_samples)] = signal[:, start:end]
        return output


def set_audio_channels(audio, target_channels):
    if target_channels == 1:
        audio = audio.mean(1, keepdim=True)
    elif target_channels == 2:
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2, :]
    return audio


def prepare_audio(audio, in_sr, target_sr, target_length, target_channels, device):
    audio = audio.to(device)

    if in_sr != target_sr:
        resample_tf = T.Resample(in_sr, target_sr).to(device)
        audio = resample_tf(audio)

    assert target_length is None
    if target_length is None:
        target_length = audio.shape[-1]

    audio = PadCrop(target_length, randomize=False)(audio)

    # Add batch dimension
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)

    audio = set_audio_channels(audio, target_channels)

    return audio


class StableAudioInfer(nn.Module):
    def __init__(self, model_config_path, model_ckpt_path=None):
        super().__init__()

        with open(model_config_path) as f:
            self.model_config = json.load(f)

        self.model = create_autoencoder_from_config(self.model_config)
        if model_ckpt_path is not None:
            self.model.load_state_dict(load_ckpt_state_dict(model_ckpt_path))

        self.sample_rate = self.model_config["sample_rate"]
        self.sample_size = self.model_config["sample_size"]
        self.io_channels = self.model.io_channels
        self.sample_size = 24576

    @property
    def device(self):
        return next(self.parameters()).device

    def normalize_audio(self, y, target_dbfs=0):
        """Normalize audio to a specific dBFS level."""
        max_amplitude = torch.max(torch.abs(y))
        target_amplitude = 10.0 ** (target_dbfs / 20.0)
        scale_factor = target_amplitude / max_amplitude
        return y * scale_factor

    def encode_audio(self, input_audio, in_sr):
        """Encode audio waveform into VAE latent representation.

        Args:
            input_audio: Input audio tensor.
            in_sr: Input sample rate.

        Returns:
            Latent tensor from the VAE encoder.
        """
        input_audio = prepare_audio(
            input_audio,
            in_sr=in_sr,
            target_sr=self.model.sample_rate,
            target_length=None,  # Determined after resampling
            target_channels=self.io_channels,
            device=self.device,
        )
        input_audio = self.normalize_audio(input_audio, -6)

        with torch.no_grad():
            # Use chunked encoding for long audio to save memory
            if input_audio.shape[-1] > (128 + 10) * self.model.sample_rate:
                latent = self.model.encode_audio(input_audio, chunked=True)
            else:
                latent = self.model.encode_audio(input_audio, chunked=False)

        return latent

    def decode_audio(self, latent):
        """Decode VAE latent back to audio waveform.

        Args:
            latent: Latent tensor.

        Returns:
            Decoded audio tensor.
        """
        with torch.no_grad():
            # Use chunked decoding for long latents to save memory
            if latent.shape[-1] > 128 + 10:
                output = self.model.decode_audio(latent, chunked=True)
            else:
                output = self.model.decode_audio(latent, chunked=False)
        return output

    def forward(self, func_type, x, sr=None):
        x = x.to(next(self.parameters()).device)
        if func_type == "encode":
            assert sr is not None, "sr is required for encoding"
            return self.encode_audio(input_audio=x, in_sr=sr)
        elif func_type == "decode":
            return self.decode_audio(x)
        else:
            raise ValueError(f"Unknown func_type: {func_type}")


if __name__ == "__main__":
    import torchaudio

    device = "cuda"
    vae_model = StableAudioInfer(
        model_config_path="config/stable_audio_2_0_vae_20hz_official.json",
        model_ckpt_path="ckpts/stable_audio_2_0_vae_20hz_official.ckpt",
    )
    vae_model = vae_model.eval().to(device)

    input_audio, in_sr = torchaudio.load("path/to/input.wav")
    latent = vae_model(func_type="encode", x=input_audio, sr=in_sr)

    output_audio = vae_model(func_type="decode", x=latent, sr=None)
    output_audio = output_audio.squeeze(0).cpu()
    torchaudio.save("output.wav", output_audio, sample_rate=44100)
