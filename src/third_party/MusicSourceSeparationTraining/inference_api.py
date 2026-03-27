import os
import sys
import warnings

import numpy as np
import torch
import torchaudio

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.audio_utils import denormalize_audio, normalize_audio
from utils.model_utils import demix, load_start_checkpoint
from utils.settings import get_model_from_config

warnings.filterwarnings("ignore")


class Separator:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        model_type: str = "mel_band_roformer",
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model_type = model_type

        torch.backends.cudnn.benchmark = True
        self.model, self.config = get_model_from_config(model_type, config_path)

        if "model_type" in self.config.training:
            self.model_type = self.config.training.model_type

        from argparse import Namespace

        fake_args = Namespace(
            model_type=self.model_type,
            config_path=config_path,
            start_check_point=checkpoint_path,
            device="auto",
            output_dir="./output",
            use_tta=False,
            extract_instrumental=True,
            pcm_type="FLOAT",
            lora_checkpoint_loralib="",
            draw_spectro=False,
        )
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        load_start_checkpoint(fake_args, self.model, ckpt, type_="inference")

        self.model = self.model.to(self.device)
        self.model.eval()
        self.sample_rate = getattr(self.config.audio, "sample_rate", 44100)

    def separate(self, wav: torch.Tensor, sr: int):
        """
        Args:
            wav: Waveform returned by torchaudio.load, shape (channels, samples)
            sr:  Sample rate
        Returns:
            vocal_wav:  np.ndarray, shape (channels, samples)
            inst_wav:   np.ndarray, shape (channels, samples)
            sr:         int, output sample rate
        """
        # Resample if needed
        if sr != self.sample_rate:
            wav = torchaudio.transforms.Resample(sr, self.sample_rate)(wav)
            sr = self.sample_rate

        mix = wav.numpy()

        # Convert mono to stereo
        if mix.shape[0] == 1 and getattr(self.config.audio, "num_channels", 1) == 2:
            mix = np.concatenate([mix, mix], axis=0)

        mix_orig = mix.copy()

        # Normalize
        norm_params = None
        if getattr(self.config.inference, "normalize", False):
            mix, norm_params = normalize_audio(mix)

        # Separate
        waveforms = demix(
            self.config,
            self.model,
            mix,
            self.device,
            model_type=self.model_type,
            pbar=True,
        )

        # Extract vocals
        vocal_wav = waveforms.get("vocals", list(waveforms.values())[0])
        if norm_params is not None:
            vocal_wav = denormalize_audio(vocal_wav, norm_params)

        # Instrumental = original mix - vocals
        inst_wav = mix_orig - vocal_wav

        return vocal_wav, inst_wav, sr


# ---- Example Usage ----
if __name__ == "__main__":
    sep = Separator(
        config_path="ckpts/config_vocals_mel_band_roformer_kj.yaml",
        checkpoint_path="ckpts/MelBandRoformer.ckpt",
        device="cuda:0",
    )

    wav, sr = torchaudio.load("path/to/input.mp3")
    vocal_wav, inst_wav, sr = sep.separate(wav, sr)

    torchaudio.save("output_vocals.wav", torch.from_numpy(vocal_wav), sr)
    torchaudio.save("output_instrumental.wav", torch.from_numpy(inst_wav), sr)
