"""
YingMusicSinger Batch Inference Script
Supports multi-GPU multi-process execution with progress bar display.
Input supports JSONL files or the LyricEditBench dataset.

Usage:
    # JSONL input, 4 GPUs
    python batch_infer.py \
        --input_type jsonl \
        --input_path /path/to/input.jsonl \
        --output_dir /path/to/output \
        --ckpt_path /path/to/ckpts \
        --num_gpus 4

    # LyricEditBench input
    python batch_infer.py \
        --input_type lyric_edit_bench \
        --output_dir /path/to/output \
        --ckpt_path /path/to/ckpts \
        --num_gpus 4
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torchaudio
from datasets import Audio, Dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_dataset_from_local(gtsinger_root: str):
    """
    Build LyricEditBench dataset using your local GTSinger directory.

    Args:
        gtsinger_root: Root directory of your local GTSinger dataset.
    """
    # Download the inherited metadata from HuggingFace
    json_path = hf_hub_download(
        repo_id="ASLP-lab/LyricEditBench",
        filename="GTSinger_Inherited.json",
        repo_type="dataset",
    )

    with open(json_path, "r") as f:
        data = json.load(f)

    gtsinger_root = str(Path(gtsinger_root).resolve())

    # Prepend local root to relative paths
    for item in data:
        item["melody_ref_path"] = os.path.join(gtsinger_root, item["melody_ref_path"])
        item["timbre_ref_path"] = os.path.join(gtsinger_root, item["timbre_ref_path"])
        # Set audio fields to the resolved file paths
        item["melody_ref_audio"] = item["melody_ref_path"]
        item["timbre_ref_audio"] = item["timbre_ref_path"]

    # Build HuggingFace Dataset with Audio features
    ds = Dataset.from_list(data)
    ds = ds.cast_column("melody_ref_audio", Audio())
    ds = ds.cast_column("timbre_ref_audio", Audio())

    return ds


def load_subset(data: list, subset_id: str) -> list:
    """Filter dataset by a subset ID list."""
    subset_path = hf_hub_download(
        repo_id="ASLP-lab/LyricEditBench",
        filename=f"id_lists/{subset_id}.txt",
        repo_type="dataset",
    )

    with open(subset_path, "r") as f:
        id_set = set(line.strip() for line in f if line.strip())

    return [item for item in data if item["id"] in id_set]


def load_lyric_edit_bench(input_type) -> list[dict]:
    # If you have GTSinger downloaded locally, use this:
    ds_full = build_dataset_from_local(
        "/user-fs/chenzihao/zhengjunjie/datas/Music/openvocaldata/GTSinger"
    )

    # Otherwise, you can use this:
    # from datasets import load_dataset
    # ds_full = load_dataset("ASLP-lab/LyricEditBench", split="test")

    subset_1k = load_subset(ds_full, "1K")
    print(f"Loaded {len(subset_1k)} items")

    items = []
    for row in subset_1k:
        if input_type == "lyric_edit_bench_melody_control":
            items.append(
                {
                    "id": row.get("id", ""),
                    "melody_ref_path": row.get("melody_ref_path", ""),
                    "gen_text": row.get("gen_text", ""),
                    "timbre_ref_path": row.get("timbre_ref_path", ""),
                    "timbre_ref_text": row.get("timbre_ref_text", ""),
                }
            )
        elif input_type == "lyric_edit_bench_sing_edit":
            items.append(
                {
                    "id": row.get("id", ""),
                    "melody_ref_path": row.get("melody_ref_path", ""),
                    "gen_text": row.get("gen_text", ""),
                    "timbre_ref_path": row.get("melody_ref_path", ""),
                    "timbre_ref_text": row.get("melody_ref_text", ""),
                }
            )
        else:
            assert 0
    return items


def worker(
    rank: int,
    world_size: int,
    items: list[dict],
    output_dir: str,
    ckpt_path: str,
    args: argparse.Namespace,
):
    """Worker process running on each GPU."""
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    # ---- Load model ----
    from src.YingMusicSinger.infer.YingMusicSinger import YingMusicSinger

    model = YingMusicSinger.from_pretrained(ckpt_path)
    model.to(device)
    model.eval()

    # ---- Shard: each worker handles its own partition ----
    shard = items[rank::world_size]

    # ---- Only rank 0 shows the progress bar (unless --show_all_progress is set) ----
    pbar = tqdm(
        shard,
        desc=f"[GPU {rank}]",
        position=rank,
        leave=True,
        disable=(rank != 0 and not args.show_all_progress),
    )

    success, fail = 0, 0
    for item in pbar:
        item_id = item.get("id", f"unknown_{success + fail}")
        out_path = os.path.join(output_dir, f"{item_id}.wav")

        # Skip files that already exist
        if os.path.exists(out_path) and not args.overwrite:
            success += 1
            pbar.set_postfix(ok=success, err=fail)
            continue

        try:
            with torch.no_grad():
                audio, sr = model(
                    ref_audio_path=item["timbre_ref_path"],
                    melody_audio_path=item["melody_ref_path"],
                    ref_text=item.get("timbre_ref_text", ""),
                    target_text=item.get("gen_text", ""),
                    lrc_align_mode=args.lrc_align_mode,
                    sil_len_to_end=args.sil_len_to_end,
                    t_shift=args.t_shift,
                    nfe_step=args.nfe_step,
                    cfg_strength=args.cfg_strength,
                    seed=args.seed
                    if args.seed != -1
                    else torch.randint(0, 2**32, (1,)).item(),
                )

            torchaudio.save(out_path, audio, sample_rate=sr)
            success += 1

        except Exception as e:
            fail += 1
            print(f"\n[GPU {rank}] ERROR on {item_id}: {e}", file=sys.stderr)
            if args.verbose:
                traceback.print_exc()

        pbar.set_postfix(ok=success, err=fail)

    pbar.close()
    print(f"[GPU {rank}] Done. success={success}, fail={fail}")


def main():
    parser = argparse.ArgumentParser(description="YingMusicSinger Batch Inference")

    # ---- Input ----
    parser.add_argument(
        "--input_type",
        type=str,
        required=True,
        choices=[
            "jsonl",
            "lyric_edit_bench_melody_control",
            "lyric_edit_bench_sing_edit",
        ],
        help="Input type: jsonl / lyric_edit_bench_melody_control / lyric_edit_bench_sing_edit",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Path to the JSONL file (required when input_type=jsonl)",
    )

    # ---- Output ----
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory",
    )

    # ---- Model ----
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=False,
        help="Model checkpoint path (directory saved by save_pretrained)",
        default=None,
    )

    # ---- Inference parameters ----
    parser.add_argument(
        "--num_gpus", type=int, default=None, help="Number of GPUs to use (default: all available)"
    )
    parser.add_argument(
        "--lrc_align_mode",
        type=str,
        default="sentence_level",
        choices=["sentence_level"],
    )
    parser.add_argument("--sil_len_to_end", type=float, default=0.5)
    parser.add_argument("--t_shift", type=float, default=0.5)
    parser.add_argument("--nfe_step", type=int, default=32)
    parser.add_argument("--cfg_strength", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=-1)

    # ---- Miscellaneous ----
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument(
        "--show_all_progress", action="store_true", help="Show progress bars for all GPUs"
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed error messages")

    args = parser.parse_args()

    # ---- Validation ----
    if args.input_type == "jsonl":
        assert args.input_path is not None, "--input_path is required in jsonl mode"
        assert os.path.isfile(args.input_path), f"File not found: {args.input_path}"

    # ---- Load data ----
    print("Loading data...")
    if args.input_type == "jsonl":
        items = load_jsonl(args.input_path)
    else:
        items = load_lyric_edit_bench(args.input_type)
    print(f"Total {len(items)} items")

    # ---- Determine GPU count ----
    available_gpus = torch.cuda.device_count()
    num_gpus = args.num_gpus or available_gpus
    num_gpus = min(num_gpus, available_gpus, len(items))
    assert num_gpus > 0, "No GPUs available"
    print(f"Using {num_gpus} GPU(s)")

    # ---- Create output directory ----
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Launch processes ----
    if num_gpus == 1:
        # Single GPU: run directly without spawning
        worker(0, 1, items, args.output_dir, args.ckpt_path, args)
    else:
        mp.set_start_method("spawn", force=True)
        processes = []
        for rank in range(num_gpus):
            p = mp.Process(
                target=worker,
                args=(rank, num_gpus, items, args.output_dir, args.ckpt_path, args),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    print(f"\nInference complete! Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()