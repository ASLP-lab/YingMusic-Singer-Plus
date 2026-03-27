import argparse
import json
import os
import random

import numpy as np
import torch
from f5_tts.infer.utils_infer import load_checkpoint
from f5_tts.model import CFM, DiT
from f5_tts.model.alsp_lance.data.npydata import FloatData
from f5_tts.model.alsp_lance.tools import LanceReader, LanceWriter
from tqdm import tqdm

filter_keyword_list = [
    "纯音乐",
    "编曲",
    "作词",
    "作曲",
    "调音",
    "制作人",
    "录音师",
]

filter_full_list = ["music", "end"]


def check_lyric(time: float, lyric: str):
    if time < 0.1:
        return False
    for filter_keyword in filter_keyword_list:
        if filter_keyword in lyric:
            return False
    for filter_full in filter_full_list:
        if filter_full == lyric.strip().lower():
            return False
    if len(lyric) == 0:
        return False
    return True


def parse_lyrics(lyrics: str):
    lyrics_with_time = []
    lyrics = lyrics.strip()
    for line in lyrics.split("\n"):
        try:
            time, lyric = line[1:9], line[10:]
            lyric = lyric.strip()
            mins, secs = time.split(":")
            secs = int(mins) * 60 + float(secs)
            # print(lyric, check_lyric(secs, lyric))
            if not check_lyric(secs, lyric):
                continue
            lyrics_with_time.append((secs, lyric))
        except:
            # traceback.print_exc()
            continue
            # print("error", line)
    return lyrics_with_time


class CNENTokenizer:
    def __init__(self):
        with open("./src/YingMusicSinger/utils/f5_tts/g2p/g2p/vocab.json", "r") as file:
            self.phone2id: dict = json.load(file)["vocab"]
        self.id2phone = {v: k for (k, v) in self.phone2id.items()}
        from f5_tts.g2p.g2p_generation import chn_eng_g2p

        self.tokenizer = chn_eng_g2p

    def encode(self, text):
        phone, token = self.tokenizer(text)
        token = [x + 1 for x in token]
        return token

    def decode(self, token):
        return "|".join([self.id2phone[x - 1] for x in token])


def inference(
    model,
    cond,
    text,
    duration,
    style_prompt,
    style,
    output_dir,
    song_name,
    ckpt_step,
    start_time,
    latent_pred_start_frame,
    latent_pred_end_frame,
    epoch,
    cfg_strength,
):
    # import pdb; pdb.set_trace()
    with torch.inference_mode():
        generated, _ = model.sample(
            cond=cond,
            text=text,
            duration=duration,
            style_prompt=style_prompt,
            steps=32,
            cfg_strength=cfg_strength,
            sway_sampling_coef=None,
            start_time=start_time,
            latent_pred_start_frame=latent_pred_start_frame,
            latent_pred_end_frame=latent_pred_end_frame,
        )

        generated = generated.to(torch.float32)  # [b t d]
        latent = generated.transpose(1, 2)  # [b d t]
        latent = latent.detach().cpu.numpy()

        return latent


def get_style_prompt(device, song_name, song_name2ref_npy):
    mulan_style_path = song_name2ref_npy[song_name]
    mulan_stlye = np.load(mulan_style_path)

    style_prompt = torch.from_numpy(mulan_stlye).to(device)  # [1, 512]
    style_prompt = style_prompt.half()

    return style_prompt


def get_lrc_prompt(text, tokenizer, dit_model, max_secs):
    max_frames = 2048
    lyrics_shift = 2
    sampling_rate = 44100
    downsample_rate = 2048

    pad_token_id = 0
    comma_token_id = 1
    period_token_id = 2

    fsmin = -10
    fsmax = 10

    lrc_with_time = parse_lyrics(text)

    modified_lrc_with_time = []
    for i in range(len(lrc_with_time)):
        time, line = lrc_with_time[i]
        # line_token = self.tokenizer.encode(line)
        line_token = tokenizer.encode(line)
        modified_lrc_with_time.append((time, line_token))

    lrc_with_time = modified_lrc_with_time

    lrc_with_time = [
        (time_start, line)
        for (time_start, line) in lrc_with_time
        if time_start < max_secs
    ]
    # latent_end_time = lrc_with_time[-1][0] if len(lrc_with_time) >= 1 else -1
    lrc_with_time = lrc_with_time[:-1] if len(lrc_with_time) >= 1 else lrc_with_time

    normalized_start_time = 0.0

    lrc = torch.zeros((max_frames,), dtype=torch.long)

    tokens_count = 0
    last_end_pos = 0
    for time_start, line in lrc_with_time:
        tokens = [
            token if token != period_token_id else comma_token_id for token in line
        ] + [period_token_id]
        tokens = torch.tensor(tokens, dtype=torch.long)
        num_tokens = tokens.shape[0]

        gt_frame_start = int(time_start * sampling_rate / downsample_rate)

        frame_shift = random.randint(int(fsmin), int(fsmax))

        frame_start = max(gt_frame_start - frame_shift, last_end_pos)
        frame_len = min(num_tokens, max_frames - frame_start)

        # print(gt_frame_start, frame_shift, frame_start, frame_len, tokens_count, last_end_pos, full_pos_emb.shape)

        lrc[frame_start : frame_start + frame_len] = tokens[:frame_len]

        tokens_count += num_tokens
        last_end_pos = frame_start + frame_len

    lrc_emb = lrc.unsqueeze(0).to(dit_model.device)

    normalized_start_time = (
        torch.tensor(normalized_start_time).unsqueeze(0).to(dit_model.device)
    )

    return lrc_emb, normalized_start_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)  # lance
    parser.add_argument("--lrc-path", type=str, default=None)
    parser.add_argument("--mulan-style-path", type=str, default=None)  # lance
    parser.add_argument("--cfg-strength", type=float, default=None)

    args = parser.parse_args()

    lrc_path = args.lrc_path
    cfg_strength = args.cfg_strength
    style_path = args.mulan_style_path

    with open(args.model_config) as f:
        model_config = json.load(f)

    model_cls = DiT
    ckpt_path = args.ckpt_path
    device = "cuda"
    use_style_prompt = True
    dit_model = CFM(
        transformer=model_cls(
            **model_config["model"], use_style_prompt=use_style_prompt
        ),
        num_channels=model_config["model"]["mel_dim"],
        use_style_prompt=use_style_prompt,
    )
    dit_model = dit_model.to(device)
    dit_model = load_checkpoint(dit_model, ckpt_path, device=device, use_ema=True)

    lrc_tokenizer = CNENTokenizer()

    sampling_rate = 44100
    downsample_rate = 2048
    max_frames = 2048
    max_secs = max_frames / (sampling_rate / downsample_rate)

    output_dir = args.output_dir
    writer = LanceWriter(output_dir, target_cls=FloatData)

    reader = LanceReader(style_path, target_cls=FloatData)

    WRITE_INTERVAL = 500

    latent_data = []
    for id in tqdm(reader.get_ids()):
        item = reader.get_datas_by_rowids(row_ids=[id._rowid])[0]
        data_id = item.data_id
        style_prompt = torch.from_numpy(item.data).to(device)
        stlye_prompt = style_prompt.half()

        lrc_path = os.path.join(lrc_path, f"{data_id}.lrc")
        with (open(lrc_path), "r") as f:
            lrc = [line.strip() for line in f.readlines()]
        lrc_prompt, start_time = get_lrc_prompt(lrc, lrc_tokenizer, dit_model, max_secs)

        latent_prompt = torch.zeros(1, max_frames, 64).to(device)
        sf = 0
        ef = max_frames

        generated_latent = inference(
            model=dit_model,
            cond=latent_prompt,
            text=lrc_prompt,
            duration=max_frames,
            style_prompt=style_prompt,
            output_dir=output_dir,
            start_time=start_time,
            latent_pred_start_frame=sf,
            latent_pred_end_frame=ef,
            cfg_strength=cfg_strength,
        )  # [b d t] numpy

        latent_data.append(generated_latent)

        if len(latent_data) > WRITE_INTERVAL:
            writer.write_parallel(latent_data)
            latent_data = []

    writer.write_parallel(latent_data)
