# Note: Set --ckpt_path to "ASLP-lab/YingMusic-Singer" to automatically
# download and load checkpoints from Hugging Face.
# Example:
#   --ckpt_path ASLP-lab/YingMusic-Singer

# json example
# {"id": "lyric_edit_en_moon_grinning", "melody_ref_path": "examples/hf_space/lyric_edit/SingEdit_EN_01.wav", "gen_text": "can you spot the moon is grinning|my lips will show you hints", "timbre_ref_path": "examples/hf_space/lyric_edit/SingEdit_EN_01.wav", "timbre_ref_text": "can you tell my heart is speaking|my eyes will give you clues"}

# JSONL input:
python inference_mp.py \
    --input_type jsonl \
    --input_path /path/to/your/input_jsonl.jsonl \
    --output_dir /path/to/your/output/Jsonl \
    --ckpt_path ASLP-lab/YingMusic-Singer \
    --num_gpus 8 \
    --show_all_progress


# LyricEditBench input (melody control):
python inference_mp.py \
    --input_type lyric_edit_bench_melody_control \
    --output_dir /path/to/your/output/LyricEditBench_melody_control \
    --ckpt_path ASLP-lab/YingMusic-Singer \
    --num_gpus 8 \
    --show_all_progress


# LyricEditBench input (sing edit):
python inference_mp.py \
    --input_type lyric_edit_bench_sing_edit \
    --output_dir /path/to/your/output/LyricEditBench_sing_edit \
    --ckpt_path ASLP-lab/YingMusic-Singer \
    --num_gpus 8 \
    --show_all_progress