"""
YingMusic-Singer Initialization Script

Downloads required checkpoints from HuggingFace based on task type.

Usage:
    python initialization.py --task infer
    python initialization.py --task train
"""

import argparse
import os

from huggingface_hub import hf_hub_download

REPO_ID = "ASLP-lab/YingMusic-Singer"
CKPT_DIR = "ckpts"

# Files required for each task
INFER_FILES = [
    "ckpts/MelBandRoformer.ckpt",
    "ckpts/config_vocals_mel_band_roformer_kj.yaml",
]

TRAIN_EXTRA_FILES = [
    "ckpts/YingMusicSinger_model.pt",
    "ckpts/model_ckpt_steps_100000_simplified.ckpt",
    "ckpts/stable_audio_2_0_vae_20hz_official.ckpt",
]

TASK_FILES = {
    "infer": INFER_FILES,
    "train": INFER_FILES + TRAIN_EXTRA_FILES,
}


def download_files(task: str):
    files = TASK_FILES[task]
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"Task: {task} | Downloading {len(files)} file(s) to {CKPT_DIR}/")
    for remote_path in files:
        filename = os.path.basename(remote_path)
        local_path = os.path.join(CKPT_DIR, filename)

        if os.path.exists(local_path):
            print(f"  [skip] {filename} already exists")
            continue

        print(f"  [download] {filename} ...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_path,
            local_dir=".",
        )
        print(f"  [done] {filename}")

    print("All downloads complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download YingMusic-Singer checkpoints"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(TASK_FILES.keys()),
        help="Task type: 'infer' for inference, 'train' for training",
    )
    args = parser.parse_args()
    download_files(args.task)
