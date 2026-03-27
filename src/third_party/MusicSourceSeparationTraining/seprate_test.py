# coding: utf-8
"""
MelBand RoFormer 音频分离器 —— 类封装版本
基于 ZFTurbo 的 Music-Source-Separation-Training 推理脚本改写。

用法示例:
    from separator import MelBandSeparator

    sep = MelBandSeparator(
        model_type="mel_band_roformer",
        config_path="ckpts/config_vocals_mel_band_roformer_kj.yaml",
        checkpoint_path="ckpts/MelBandRoformer.ckpt",
        device="cuda:0",          # 或 "cpu" / "mps"
    )

    results = sep.separate("song.wav", output_dir="output/")
    # results: dict[str, str]  —— {instrument_name: output_file_path, ...}
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio

# ---------------------------------------------------------------------------
# 让 utils 模块可以被正确导入（兼容嵌入式 Python 等场景）
# ---------------------------------------------------------------------------
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.append(_CURRENT_DIR)

import warnings

from utils.audio_utils import denormalize_audio, draw_spectrogram, normalize_audio
from utils.model_utils import (
    apply_tta,
    demix,
    load_start_checkpoint,
    prefer_target_instrument,
)
from utils.settings import get_model_from_config

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 数据类：分离结果
# ---------------------------------------------------------------------------
@dataclass
class SeparationResult:
    """单个音频文件的分离结果。"""

    instrument: str
    audio: np.ndarray  # shape: (channels, samples)
    sample_rate: int
    output_path: Optional[str] = None


# ---------------------------------------------------------------------------
# 核心类
# ---------------------------------------------------------------------------
class MelBandSeparator:
    """
    基于 MelBand RoFormer 的音频源分离器。

    Parameters
    ----------
    model_type : str
        模型类型，例如 ``"mel_band_roformer"``、``"bs_roformer"``、``"mdx23c"`` 等。
    config_path : str
        模型配置文件 (.yaml) 路径。
    checkpoint_path : str
        模型权重文件 (.ckpt) 路径。
    device : str | torch.device
        推理设备，例如 ``"cuda:0"``、``"cpu"``、``"mps"``。
        默认为 ``"auto"``，会自动选择可用的 GPU > MPS > CPU。
    device_ids : list[int] | None
        多 GPU DataParallel 时使用的 GPU id 列表。
        仅当 device 为 cuda 且提供多个 id 时生效。
    use_tta : bool
        是否启用 Test-Time Augmentation（推理时数据增强），可略微提升质量但更慢。
    extract_instrumental : bool
        是否额外提取伴奏轨（原始混音 - 人声）。
    pcm_type : str
        输出音频的 PCM 子类型，例如 ``"FLOAT"``、``"PCM_16"``、``"PCM_24"``。
    """

    def __init__(
        self,
        args,
        model_type: str,
        config_path: str,
        checkpoint_path: str,
        device: str = "auto",
        device_ids: list[int] | None = None,
        use_tta: bool = False,
        extract_instrumental: bool = False,
        pcm_type: str = "FLOAT",
    ) -> None:
        self.model_type = model_type
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.use_tta = use_tta
        self.extract_instrumental = extract_instrumental
        self.pcm_type = pcm_type

        # ---- 选择设备 ----
        self.device = self._resolve_device(device)
        self.device_ids = device_ids
        print(f"[MelBandSeparator] Using device: {self.device}")

        # ---- 加载模型 ----
        t0 = time.time()
        torch.backends.cudnn.benchmark = True

        self.model, self.config = get_model_from_config(model_type, config_path)

        # 覆盖 model_type（部分 config 里会自带）
        if "model_type" in self.config.training:
            self.model_type = self.config.training.model_type

        # 加载权重
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        load_start_checkpoint(args, self.model, checkpoint, type_="inference")

        # 多 GPU
        if (
            device_ids is not None
            and len(device_ids) > 1
            and "cuda" in str(self.device)
        ):
            self.model = nn.DataParallel(self.model, device_ids=device_ids)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.sample_rate: int = getattr(self.config.audio, "sample_rate", 44100)
        self.instruments: list[str] = prefer_target_instrument(self.config)[:]

        print(f"[MelBandSeparator] Instruments: {self.instruments}")
        print(f"[MelBandSeparator] Model loaded in {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 公共方法
    # ------------------------------------------------------------------
    def separate(
        self,
        mix_audio,
        mix_audio_sr,
        output_dir: str | None = None,
        draw_spectro: bool = False,
    ) -> list[SeparationResult]:
        """
        对单个音频文件进行源分离。

        Parameters
        ----------
        audio_path : str
            输入音频文件路径（支持 wav / flac / mp3 等 torchaudio 能读取的格式）。
        output_dir : str | None
            输出目录。为 ``None`` 时不写文件，仅返回内存中的结果。
        draw_spectro : bool
            是否保存频谱图（需要 output_dir 不为 None）。

        Returns
        -------
        list[SeparationResult]
            每个乐器/人声轨对应一个 ``SeparationResult``。
        """
        # ---- 1. 使用 torchaudio 读取音频 ----
        mix = mix_audio
        sr = mix_audio_sr
        # 重采样到模型要求的采样率
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sample_rate
            )
            mix = resampler(mix)
            sr = self.sample_rate

        # 转 numpy: (channels, samples)
        mix: np.ndarray = mix.numpy()

        # 单声道 → 立体声（如果模型需要）
        if mix.shape[0] == 1:
            num_channels = getattr(self.config.audio, "num_channels", 1)
            if num_channels == 2:
                print("[MelBandSeparator] Converting mono to stereo...")
                mix = np.concatenate([mix, mix], axis=0)

        mix_orig = mix.copy()

        # ---- 2. 归一化 ----
        norm_params = None
        if getattr(self.config.inference, "normalize", False):
            mix, norm_params = normalize_audio(mix)

        # ---- 3. 分离 ----
        waveforms = demix(
            self.config,
            self.model,
            mix,
            self.device,
            model_type=self.model_type,
            pbar=True,
        )

        # ---- 4. TTA ----
        if self.use_tta:
            waveforms = apply_tta(
                self.config,
                self.model,
                mix,
                waveforms,
                self.device,
                self.model_type,
            )

        # ---- 5. 伴奏提取 ----
        instruments = self.instruments[:]
        if self.extract_instrumental:
            instr_key = "vocals" if "vocals" in instruments else instruments[0]
            waveforms["instrumental"] = mix_orig - waveforms[instr_key]
            if "instrumental" not in instruments:
                instruments.append("instrumental")

        # ---- 6. 反归一化 & 收集结果 ----
        results: list[SeparationResult] = []
        file_stem = 111

        for instr in instruments:
            estimates = waveforms[instr]
            if norm_params is not None:
                estimates = denormalize_audio(estimates, norm_params)

            result = SeparationResult(
                instrument=instr,
                audio=estimates,
                sample_rate=sr,
            )

            # ---- 写文件 ----
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)

                peak = float(np.abs(estimates).max())
                codec = "flac" if (peak <= 1.0 and self.pcm_type != "FLOAT") else "wav"

                out_path = os.path.join(output_dir, f"{file_stem}_{instr}.{codec}")
                sf.write(out_path, estimates.T, sr, subtype=self.pcm_type)
                result.output_path = out_path
                print(f"[MelBandSeparator] Saved: {out_path}")

                # out_path = os.path.join(output_dir, f"_inst.wav")
                # sf.write(out_path, estimates.T, sr, subtype=self.pcm_type)
                # result.output_path = out_path
                # print(f"[MelBandSeparator] Saved: {out_path}")

                if draw_spectro:
                    img_path = os.path.join(output_dir, f"{file_stem}_{instr}.jpg")
                    draw_spectrogram(estimates.T, sr, 1, img_path)
                    print(f"[MelBandSeparator] Spectrogram: {img_path}")

            results.append(result)

        return results

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """自动选择设备。"""
        if device != "auto":
            return torch.device(device)

        if torch.cuda.is_available():
            print("[MelBandSeparator] CUDA detected.")
            return torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[MelBandSeparator] MPS detected.")
            return torch.device("mps")
        else:
            return torch.device("cpu")


# ---------------------------------------------------------------------------
# CLI 入口（可选）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MelBand RoFormer Separator")
    parser.add_argument("--model_type", default="mel_band_roformer")
    parser.add_argument("--config_path", required=True, help="Path to config YAML")
    parser.add_argument(
        "--start_check_point", required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--device", default="auto", help="Device: auto / cpu / cuda:0 / mps"
    )
    parser.add_argument("--output_dir", default="./output", help="Output directory")
    parser.add_argument("--use_tta", action="store_true", help="Enable TTA")
    parser.add_argument("--extract_instrumental", action="store_true")
    parser.add_argument(
        "--pcm_type", default="FLOAT", choices=["FLOAT", "PCM_16", "PCM_24"]
    )
    # parser.add_argument("--model_type", type=str, default='mdx23c',
    #                     help="One of bandit, bandit_v2, bs_roformer, htdemucs, mdx23c, mel_band_roformer,"
    #                          " scnet, scnet_unofficial, segm_models, swin_upernet, torchseg")
    # parser.add_argument("--config_path", type=str, help="path to config file")
    # parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to valid weights")
    # parser.add_argument("--input_folder", type=str, help="folder with mixtures to process")
    # parser.add_argument("--store_dir", type=str, default="", help="path to store results as wav file")
    # parser.add_argument("--draw_spectro", type=float, default=0,
    #                     help="Code will generate spectrograms for resulted stems."
    #                          " Value defines for how many seconds os track spectrogram will be generated.")
    # parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='list of gpu ids')
    # parser.add_argument(
    #     "--extract_instrumental",
    #     action="store_true",
    #     help="invert vocals to get instrumental if provided",
    # )
    # parser.add_argument("--disable_detailed_pbar", action='store_true', help="disable detailed progress bar")
    # parser.add_argument("--force_cpu", action='store_true', help="Force the use of CPU even if CUDA is available")
    # parser.add_argument("--flac_file", action='store_true', help="Output flac file instead of wav")
    # parser.add_argument("--pcm_type", type=str, choices=['PCM_16', 'PCM_24', 'FLOAT'], default='FLOAT',
    #                     help="PCM type for FLAC files (PCM_16 or PCM_24)")
    # parser.add_argument("--use_tta", action='store_true',
    #                     help="Flag adds test time augmentation during inference (polarity and channel inverse)."
    #                     "While this triples the runtime, it reduces noise and slightly improves prediction quality.")
    # parser.add_argument("--lora_checkpoint_peft", type=str, default='', help="Initial checkpoint to LoRA weights")
    # parser.add_argument("--filename_template", type=str, default='{file_name}/{instr}',
    #                     help="Output filename template, without extension, using '/' for subdirectories. Default: '{file_name}/{instr}'")
    parser.add_argument(
        "--lora_checkpoint_loralib",
        type=str,
        default="",
        help="Initial checkpoint to LoRA weights",
    )
    parser.add_argument("--draw_spectro", action="store_true")

    cli_args = parser.parse_args()

    sep = MelBandSeparator(
        cli_args,
        model_type=cli_args.model_type,
        config_path=cli_args.config_path,
        checkpoint_path=cli_args.start_check_point,
        device=cli_args.device,
        use_tta=cli_args.use_tta,
        extract_instrumental=cli_args.extract_instrumental,
        pcm_type=cli_args.pcm_type,
    )

    mix, sr = torchaudio.load(
        "/user-fs/chenzihao/aslp_music/haochunbo/final/张韶轩-隐形的翅膀.mp3"
    )  # (channels, samples)

    results = sep.separate(
        mix_audio=mix,
        mix_audio_sr=sr,
        output_dir="/user-fs/chenzihao/aslp_music/haochunbo/final",
        draw_spectro=False,
    )

    for r in results:
        print(f"  {r.instrument}: {r.output_path or '(in memory only)'}")
"""
PYTHONPATH=. python /user-fs/chenzihao/aslp_music/haochunbo/final/YingMusic-Singer/src/third_party/Music-Source-Separation-Training/seprate_test.py --config_path ckpts/config_vocals_mel_band_roformer_kj.yaml --start_check_point ckpts/MelBandRoformer.ckpt --extract_instrumental
"""
