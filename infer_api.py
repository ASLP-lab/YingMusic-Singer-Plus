"""
YingMusic Singer - Command Line Inference
==========================================
Single-sample inference script, replacing the Gradio Web UI.

Usage:
    # melody control
    python infer.py \
        --ref_audio examples/hf_space/melody_control/melody_control_ZH_02_timbre.wav \
        --melody_audio examples/hf_space/melody_control/melody_control_ZH_02_melody.wav \
        --ref_text "就让你|在别人怀里|快乐" \
        --target_text "Missing you in my mind|missing you in my heart" \
        --output output/melody_control_zh_missing_you.wav

    # sing edit
    python infer.py \
        --ref_audio examples/hf_space/lyric_edit/SingEdit_ZH_02.wav \
        --melody_audio examples/hf_space/lyric_edit/SingEdit_ZH_02.wav \
        --ref_text "歌声是翅膀|唱出了希望|所有的付出只因爱的力量|和你一样" \
        --target_text "火锅是梦想|煮出了欢畅|全部的辛劳全因肉的力量|与汤一样" \
        --output output/lyric_edit_zh_hotpot_dream.wav

    # Enable vocal separation + accompaniment mixing simultaneously
    python infer.py \
        --ref_audio examples/hf_space/lyric_edit/SingEdit_EN_01.wav \
        --melody_audio examples/hf_space/lyric_edit/SingEdit_EN_01.wav \
        --ref_text "can you tell my heart is speaking|my eyes will give you clues" \
        --target_text "can you spot the moon is grinning|my lips will show you hints" \
        --separate_vocals \
        --mix_accompaniment \
        --output output/lyric_edit_en_moon_grinning.wav
"""

import argparse
import os
import random
import tempfile

import torch
import torchaudio

from initialization import download_files


# ---------------------------------------------------------------------------
# Model loading (lazy singleton)
# ---------------------------------------------------------------------------
_model = None
_separator = None


def get_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def get_model():
    global _model
    if _model is None:
        download_files(task="infer")
        from src.YingMusicSinger.infer.YingMusicSinger import YingMusicSinger
        _model = YingMusicSinger.from_pretrained("ASLP-lab/YingMusic-Singer")
        _model = _model.to(get_device())
    _model.eval()
    return _model


def get_separator():
    global _separator
    if _separator is None:
        download_files(task="infer")
        from src.third_party.MusicSourceSeparationTraining.inference_api import Separator
        _separator = Separator(
            config_path="ckpts/config_vocals_mel_band_roformer_kj.yaml",
            checkpoint_path="ckpts/MelBandRoformer.ckpt",
        )
    return _separator


# ---------------------------------------------------------------------------
# Vocal separation
# ---------------------------------------------------------------------------
def separate_vocals(audio_path: str) -> tuple:
    """
    Separate vocals and accompaniment, returns (vocals_path, accompaniment_path).
    """
    separator = get_separator()
    wav, sr = torchaudio.load(audio_path)
    vocal_wav, inst_wav, out_sr = separator.separate(wav, sr)

    tmp_dir = tempfile.mkdtemp()
    vocals_path = os.path.join(tmp_dir, "vocals.wav")
    accomp_path = os.path.join(tmp_dir, "accompaniment.wav")
    torchaudio.save(vocals_path, torch.from_numpy(vocal_wav), out_sr)
    torchaudio.save(accomp_path, torch.from_numpy(inst_wav), out_sr)
    return vocals_path, accomp_path


# ---------------------------------------------------------------------------
# Mix vocals + accompaniment
# ---------------------------------------------------------------------------
def mix_vocal_and_accompaniment(vocal_path: str, accomp_path: str, vocal_gain: float = 1.0) -> str:
    vocal_wav, vocal_sr = torchaudio.load(vocal_path)
    accomp_wav, accomp_sr = torchaudio.load(accomp_path)

    if accomp_sr != vocal_sr:
        accomp_wav = torchaudio.functional.resample(accomp_wav, accomp_sr, vocal_sr)

    if vocal_wav.shape[0] != accomp_wav.shape[0]:
        if vocal_wav.shape[0] == 1:
            vocal_wav = vocal_wav.expand(accomp_wav.shape[0], -1)
        else:
            accomp_wav = accomp_wav.expand(vocal_wav.shape[0], -1)

    min_len = min(vocal_wav.shape[1], accomp_wav.shape[1])
    mixed = vocal_wav[:, :min_len] * vocal_gain + accomp_wav[:, :min_len]

    peak = mixed.abs().max()
    if peak > 1.0:
        mixed = mixed / peak

    out_path = os.path.join(tempfile.mkdtemp(), "mixed_output.wav")
    torchaudio.save(out_path, mixed, sample_rate=vocal_sr)
    return out_path


# ---------------------------------------------------------------------------
# Main inference pipeline
# ---------------------------------------------------------------------------
def synthesize(args):
    actual_seed = args.seed if args.seed >= 0 else random.randint(0, 2**31 - 1)
    print(f"[INFO] Using seed: {actual_seed}")

    actual_ref_path = args.ref_audio
    actual_melody_path = args.melody_audio
    melody_accomp_path = None

    # Step 1: Vocal separation (optional)
    if args.separate_vocals:
        print("[INFO] Separating vocals from reference audio...")
        actual_ref_path, _ = separate_vocals(args.ref_audio)

        print("[INFO] Separating vocals from melody audio...")
        actual_melody_path, melody_accomp_path = separate_vocals(args.melody_audio)

    # Step 2: Model inference
    print("[INFO] Loading model...")
    model = get_model()

    print("[INFO] Running synthesis...")
    audio_tensor, sr = model(
        ref_audio_path=actual_ref_path,
        melody_audio_path=actual_melody_path,
        ref_text=args.ref_text.strip(),
        target_text=args.target_text.strip(),
        lrc_align_mode="sentence_level",
        sil_len_to_end=args.sil_len_to_end,
        t_shift=args.t_shift,
        nfe_step=args.nfe_step,
        cfg_strength=args.cfg_strength,
        seed=actual_seed,
    )

    vocal_out_path = os.path.join(tempfile.mkdtemp(), "vocal_output.wav")
    torchaudio.save(vocal_out_path, audio_tensor.to("cpu"), sample_rate=sr)

    # Step 3: Mix accompaniment (optional)
    if args.separate_vocals and args.mix_accompaniment and melody_accomp_path is not None:
        print("[INFO] Mixing vocals with accompaniment...")
        final_path = mix_vocal_and_accompaniment(vocal_out_path, melody_accomp_path)
    else:
        final_path = vocal_out_path

    # Write to specified output path
    out_wav, out_sr = torchaudio.load(final_path)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torchaudio.save(args.output, out_wav, sample_rate=out_sr)
    print(f"[INFO] Saved to: {args.output}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="YingMusic Singer - Single sample command line inference"
    )

    # Required
    parser.add_argument("--ref_audio", required=True,
                        help="Reference audio path")
    parser.add_argument("--melody_audio", required=True,
                        help="Melody audio path")
    parser.add_argument("--ref_text", required=True,
                        help="Reference lyrics, use | to separate phrases")
    parser.add_argument("--target_text", required=True,
                        help="Target lyrics, use | to separate phrases")

    # Output
    parser.add_argument("--output", default="output.wav",
                        help="Output wav path (default: output.wav)")

    # Optional flags
    parser.add_argument("--separate_vocals", action="store_true",
                        help="Separate vocals before synthesis")
    parser.add_argument("--mix_accompaniment", action="store_true",
                        help="Mix accompaniment into output (requires --separate_vocals)")

    # Advanced params
    parser.add_argument("--nfe_step", type=int, default=32,
                        help="NFE steps (default: 32)")
    parser.add_argument("--cfg_strength", type=float, default=3.0,
                        help="CFG strength (default: 3.0)")
    parser.add_argument("--t_shift", type=float, default=0.5,
                        help="t-shift (default: 0.5)")
    parser.add_argument("--sil_len_to_end", type=float, default=0.5,
                        help="Silence padding in seconds (default: 0.5)")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Random seed, -1 for random (default: -1)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    synthesize(args)