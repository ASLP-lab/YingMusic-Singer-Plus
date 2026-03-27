import torchaudio
import torch

def smooth_ending(waveform, sr, fade_duration_ms=150):
    """
    Apply a very short fade-out to the end of an audio waveform to avoid abrupt cutoff clicks.

    Args:
        waveform: (channels, samples) tensor
        sr: sample rate
        fade_duration_ms: fade-out duration in milliseconds, default 50ms, minimal impact on original content
    """
    fade_samples = int(sr * fade_duration_ms / 1000)
    fade_samples = min(fade_samples, waveform.shape[-1])  # prevent exceeding audio length

    # Half-cosine fade-out curve for a smoother perceptual result
    fade = (1 + torch.cos(torch.linspace(0, torch.pi, fade_samples))) / 2

    waveform[..., -fade_samples:] *= fade

    return waveform

if __name__ == "__main__":
    # Usage example
    waveform, sr = torchaudio.load("")
    waveform = smooth_ending(waveform, sr, fade_duration_ms=150)
    torchaudio.save("", waveform, sr)