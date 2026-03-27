"""
YingMusic Singer - Gradio Web Interface
========================================
基于参考音色与旋律音频的歌声合成系统，支持自动分离人声与伴奏。
A singing voice synthesis system powered by YingMusicSinger,
with built-in vocal/accompaniment separation via MelBandRoformer.
"""

import os
import tempfile

import gradio as gr
import torch
import torchaudio

from initialization import download_files

IS_HF_SPACE = os.environ.get("SPACE_ID") is not None
HF_ENABLE = False
LOCAL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def local_move2gpu(x):
    """Move models to GPU on local environment. No-op on HuggingFace Spaces (ZeroGPU handles it)."""
    if IS_HF_SPACE:
        return x
    return x.to(LOCAL_DEVICE)


# ---------------------------------------------------------------------------
# Model loading (eager, at startup) / 启动时立即加载，常驻内存
# ---------------------------------------------------------------------------
print("🔄 Downloading required files...")
download_files(task="infer")

print("🔄 Loading YingMusicSinger model...")
from src.YingMusicSinger.infer.YingMusicSinger import YingMusicSinger
_model = YingMusicSinger.from_pretrained("ASLP-lab/YingMusic-Singer")
_model = local_move2gpu(_model)
_model.eval()
print("✅ YingMusicSinger model loaded.")

print("🔄 Loading MelBandRoformer separator...")
from src.third_party.MusicSourceSeparationTraining.inference_api import Separator
_separator = Separator(
    config_path="ckpts/config_vocals_mel_band_roformer_kj.yaml",
    checkpoint_path="ckpts/MelBandRoformer.ckpt",
    device=LOCAL_DEVICE
)
print("✅ MelBandRoformer separator loaded.")
print("🎤 All models ready. Starting UI...")


# ---------------------------------------------------------------------------
# Vocal separation utilities / 人声分离工具
# ---------------------------------------------------------------------------
def _separate_vocals_impl(audio_path: str) -> tuple:
    """
    Separate audio into vocals and accompaniment using MelBandRoformer.
    Must be called within an active GPU context.
    """
    separator = _separator

    wav, sr = torchaudio.load(audio_path)
    vocal_wav, inst_wav, out_sr = separator.separate(wav, sr)

    tmp_dir = tempfile.mkdtemp()
    vocals_path = os.path.join(tmp_dir, "vocals.wav")
    accomp_path = os.path.join(tmp_dir, "accompaniment.wav")
    torchaudio.save(vocals_path, torch.from_numpy(vocal_wav), out_sr)
    torchaudio.save(accomp_path, torch.from_numpy(inst_wav), out_sr)

    return vocals_path, accomp_path


def mix_vocal_and_accompaniment(
    vocal_path: str,
    accomp_path: str,
    vocal_gain: float = 1.0,
) -> str:
    """
    将合成人声与伴奏混合为最终音频。
    Mix synthesised vocals with accompaniment into a final audio file.
    """
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
    vocal_wav = vocal_wav[:, :min_len]
    accomp_wav = accomp_wav[:, :min_len]

    mixed = vocal_wav * vocal_gain + accomp_wav
    peak = mixed.abs().max()
    if peak > 1.0:
        mixed = mixed / peak

    out_path = os.path.join(tempfile.mkdtemp(), "mixed_output.wav")
    torchaudio.save(out_path, mixed, sample_rate=vocal_sr)
    return out_path


# ---------------------------------------------------------------------------
# Inference wrapper / 推理入口
# ---------------------------------------------------------------------------
def synthesize(
    ref_audio,
    melody_audio,
    ref_text,
    target_text,
    separate_vocals_flag,
    mix_accompaniment_flag,
    sil_len_to_end,
    t_shift,
    nfe_step,
    cfg_strength,
    seed,
):
    import random

    if ref_audio is None:
        raise gr.Error("请上传参考音频 / Please upload Reference Audio")
    if melody_audio is None:
        raise gr.Error("请上传旋律音频 / Please upload Melody Audio")
    if not ref_text.strip():
        raise gr.Error("请输入参考音频对应的歌词 / Please enter Reference Text")
    if not target_text.strip():
        raise gr.Error("请输入目标合成歌词 / Please enter Target Text")
    if mix_accompaniment_flag and not separate_vocals_flag:
        raise gr.Error("「输出时混入伴奏」需要先开启「分离人声后过模型」/ 'Mix accompaniment into output' requires 'Separate vocals before synthesis' to be enabled first")

    ref_audio_path = ref_audio if isinstance(ref_audio, str) else ref_audio[0]
    melody_audio_path = (
        melody_audio if isinstance(melody_audio, str) else melody_audio[0]
    )

    actual_seed = int(seed)
    if actual_seed < 0:
        actual_seed = random.randint(0, 2**31 - 1)

    melody_accomp_path = None
    actual_ref_path = ref_audio_path
    actual_melody_path = melody_audio_path

    if separate_vocals_flag:
        ref_vocals_path, _ = _separate_vocals_impl(ref_audio_path)
        actual_ref_path = ref_vocals_path

        melody_vocals_path, melody_accomp_path = _separate_vocals_impl(melody_audio_path)
        actual_melody_path = melody_vocals_path

    audio_tensor, sr = _model(
        ref_audio_path=actual_ref_path,
        melody_audio_path=actual_melody_path,
        ref_text=ref_text.strip(),
        target_text=target_text.strip(),
        lrc_align_mode="sentence_level",
        sil_len_to_end=float(sil_len_to_end),
        t_shift=float(t_shift),
        nfe_step=int(nfe_step),
        cfg_strength=float(cfg_strength),
        seed=actual_seed,
    )

    vocal_out_path = os.path.join(tempfile.mkdtemp(), "vocal_output.wav")
    torchaudio.save(vocal_out_path, audio_tensor.to("cpu"), sample_rate=sr)

    if (
        separate_vocals_flag
        and mix_accompaniment_flag
        and melody_accomp_path is not None
    ):
        final_path = mix_vocal_and_accompaniment(vocal_out_path, melody_accomp_path)
        return final_path
    else:
        return vocal_out_path


# ---------------------------------------------------------------------------
# Example presets / 预设示例
# ---------------------------------------------------------------------------
EXAMPLES_MELODY_CONTROL = [
    [
        "examples/hf_space/melody_control/melody_control_ZH_01_timbre.wav",
        "examples/hf_space/melody_control/melody_control_ZH_01_melody.wav",
        "人和人的沟通|有时候没有用",
        "此刻记忆中的点滴啊|能否再次被珍藏",
        True, True, 0.5, 0.5, 32, 3.0, -1,
    ],
    [
        "examples/hf_space/melody_control/melody_control_EN_01_timbre.wav",
        "examples/hf_space/melody_control/melody_control_EN_01_melody.wav",
        "i don't know feel|but i wanna try",
        "won't open the door|and say tomorrow",
        True, True, 0.5, 0.5, 32, 3.0, -1,
    ],
    [
        "examples/hf_space/melody_control/melody_control_EN_02_timbre.wav",
        "examples/hf_space/melody_control/melody_control_EN_02_melody.wav",
        "and she'll never know your story like|i do",
        "你将安然无恙|无人能再伤你",
        False, False, 0.5, 0.5, 32, 3.0, -1,
    ],
    [
        "examples/hf_space/melody_control/melody_control_ZH_02_timbre.wav",
        "examples/hf_space/melody_control/melody_control_ZH_02_melody.wav",
        "就让你|在别人怀里|快乐",
        "Missing you in my mind|missing you in my heart",
        False, False, 0.5, 0.5, 32, 3.0, -1,
    ]
]

EXAMPLES_LYRIC_EDIT = [
    [
        "examples/hf_space/lyric_edit/SingEdit_ZH_01.wav",
        "examples/hf_space/lyric_edit/SingEdit_ZH_01.wav",
        "天青色等烟雨|而我在等你|炊烟袅袅升起",
        "阳光中赏花香|花瓣在飘落|山间幽静致远",
        True, True, 0.5, 0.5, 32, 3.0, -1,
    ],
    [
        "examples/hf_space/lyric_edit/SingEdit_EN_01.wav",
        "examples/hf_space/lyric_edit/SingEdit_EN_01.wav",
        "can you tell my heart is speaking|my eyes will give you clues",
        "can you spot the moon is grinning|my lips will show you hints",
        True, True, 0.5, 0.5, 32, 3.0, -1,
    ],
    [
        "examples/hf_space/lyric_edit/SingEdit_ZH_02.wav",
        "examples/hf_space/lyric_edit/SingEdit_ZH_02.wav",
        "歌声是翅膀|唱出了希望|所有的付出只因爱的力量|和你一样",
        "火锅是梦想|煮出了欢畅|全部的辛劳全因肉的力量|与汤一样",
        False, False, 0.5, 0.5, 32, 3.0, -1,
    ],
    [
        "examples/hf_space/lyric_edit/SingEdit_EN_02.wav",
        "examples/hf_space/lyric_edit/SingEdit_EN_02.wav",
        "i can hear what you say|now i know|why know we can|make it",
        "i can see where you go|but i say|why not we will|break it",
        False, False, 0.5, 0.5, 32, 3.0, -1,
    ],
]


# ---------------------------------------------------------------------------
# Custom CSS / 自定义样式
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,500;0,9..40,700;1,9..40,400&family=Playfair+Display:wght@600;800&display=swap');

:root {
    --primary: #44ACFF;
    --primary-light: #89D4FF;
    --primary-warm: #FE9EC7;
    --palette-yellow: #F9F6C4;
    --bg-dark: #0d1117;
    --surface: #161b22;
    --surface-light: #21262d;
    --text: #f0f6fc;
    --text-muted: #8b949e;
    --accent-glow: rgba(68, 172, 255, 0.10);
    --border: #30363d;
}

.gradio-container {
    font-family: 'DM Sans', sans-serif !important;
    max-width: 1100px !important;
    margin: auto !important;
}

/* ---------- Badge links ---------- */
#app-header .badges a {
    text-decoration: none !important;
    display: inline-block;
    line-height: 0;
    margin: 3px 2px;
}
#app-header .badges a img,
#app-header .badges > img {
    display: inline-block;
    vertical-align: middle;
    margin: 0;
}
#app-header .badges {
    line-height: 1.8;
}

/* ---------- Header ---------- */
#app-header {
    text-align: center;
    padding: 1.8rem 1rem 0.5rem;
}
#app-header h1 {
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    line-height: 1.4;
    margin-bottom: 0.6rem !important;
}
#app-header .badges img {
    display: inline-block;
    margin: 3px 2px;
    vertical-align: middle;
}
#app-header .authors {
    color: var(--text-muted);
    font-size: 0.92rem;
    margin: 0.5rem 0 0.2rem;
    line-height: 1.7;
}
#app-header .affiliations {
    color: var(--text-muted);
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
}
#app-header .lang-links a {
    color: var(--primary-light);
    text-decoration: none;
    margin: 0 4px;
    font-size: 0.9rem;
}
#app-header .lang-links a:hover { text-decoration: underline; }

/* ---------- Disclaimer ---------- */
#disclaimer {
    border-top: 1px solid var(--border);
    margin: 24px 0 4px;
    padding: 14px 4px 4px;
    font-size: 0.80rem;
    color: #6e7681;
    line-height: 1.65;
    text-align: center;
}
#disclaimer strong {
    color: #8b949e;
    font-weight: 600;
}

/* ========== 2. Section labels: left accent bar ========== */
.resume-container .section-title {
    border: none;
    outline: none;
    box-shadow: none;
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--primary);
    display: block;
    padding: 3px 0 3px 10px;
    margin-bottom: 14px;
    border-left: 4px solid var(--primary-warm);
    background: linear-gradient(90deg, rgb(254 158 199 / 8%) 0%, transparent 70%);
    border-radius: 0 4px 4px 0;
}

.resume-container .section-title * {
    border: inherit;
    outline: none;
    box-shadow: none;
}

/* ---------- Example tabs ---------- */
.example-tab-label {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

/* ========== 3. Run button — palette blue ========== */
#run-btn {
    background: linear-gradient(135deg, #44ACFF, #89D4FF) !important;
    border: none !important;
    color: #0a1628 !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.04em;
    padding: 12px 0 !important;
    border-radius: 10px !important;
    transition: transform 0.15s, box-shadow 0.25s !important;
    box-shadow: 0 4px 20px rgba(68, 172, 255, 0.35) !important;
}
#run-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(68, 172, 255, 0.5) !important;
}

/* ---------- Output audio ---------- */
#output-audio {
    border: 2px solid #44ACFF !important;
    border-radius: 12px !important;
    background: rgba(68, 172, 255, 0.07) !important;
}

/* ========== 4. Vocal-sep info box: fix highlighted text colors ========== */
.vocal-sep-info b {
    color: #c9d1d9;
    font-weight: 700;
}
b[data-ref] {
    color: #FE9EC7 !important;
}
b[data-melody] {
    color: #89D4FF !important;
}
"""

# ---------------------------------------------------------------------------
# Header HTML / 头部 HTML
# ---------------------------------------------------------------------------
HEADER_HTML = """
<div id="app-header" align="center">
  <h1>
    🎤 YingMusic-Singer: Controllable Singing Voice Synthesis with Flexible Lyric Manipulation and Annotation-free Melody Guidance
  </h1>

  <div class="badges" style="margin: 10px 0;">
        <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white" alt="Python">
        <img src="https://img.shields.io/badge/License-CC--BY--4.0-lightgrey" alt="License">
        <a href="https://arxiv.org/abs/2603.24589"><img src="https://img.shields.io/badge/arXiv-2603.24589-b31b1b?logo=arxiv&logoColor=white" alt="arXiv Paper"></a>
        <a href="https://github.com/ASLP-lab/YingMusic-Singer"><img src="https://img.shields.io/badge/GitHub-YingMusic--Singer-181717?logo=github&logoColor=white" alt="GitHub"></a>
        <a href="https://aslp-lab.github.io/YingMusic-Singer-Demo/"><img src="https://img.shields.io/badge/GitHub-Demo--Page-8A2BE2?logo=github&logoColor=white&labelColor=181717" alt="Demo Page"></a>
        <a href="https://huggingface.co/spaces/ASLP-lab/YingMusic-Singer"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Space-FFD21E" alt="HuggingFace Space"></a>
        <a href="https://huggingface.co/ASLP-lab/YingMusic-Singer"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Model-FF9D00" alt="HuggingFace Model"></a>
        <a href="https://huggingface.co/datasets/ASLP-lab/LyricEditBench"><img src="https://img.shields.io/badge/🤗%20HuggingFace-LyricEditBench-FF6F00" alt="Dataset LyricEditBench"></a>
        <a href="https://discord.gg/RXghgWyvrn"><img src="https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
        <a href="https://github.com/ASLP-lab/YingMusic-Singer/blob/main/assets/wechat_qr.png"><img src="https://img.shields.io/badge/WeChat-Group-07C160?logo=wechat&logoColor=white" alt="WeChat"></a>
        <a href="http://www.npu-aslp.org/"><img src="https://img.shields.io/badge/🏫%20ASLP-Lab-4A90D9" alt="Lab"></a>
  </div>

  <p class="authors">
        <a href="https://orcid.org/0009-0005-5957-8936">Chunbo Hao</a><sup>1,2</sup> ·
        <a href="https://orcid.org/0009-0003-2602-2910">Junjie Zheng</a><sup>2</sup> ·
        <a href="https://orcid.org/0009-0001-6706-0572">Guobin Ma</a><sup>1</sup> ·
        Yuepeng Jiang<sup>1</sup> ·
        Huakang Chen<sup>1</sup> ·
        Wenjie Tian<sup>1</sup> ·
        <a href="https://orcid.org/0009-0003-9258-4006">Gongyu Chen</a><sup>2</sup> ·
        <a href="https://orcid.org/0009-0005-5413-6725">Zihao Chen</a><sup>2</sup> ·
        Lei Xie<sup>1</sup>
  </p>
  <p class="affiliations">
        <sup>1</sup> Audio, Speech and Language Processing Group (ASLP@NPU), School of Computer Science, Northwestern Polytechnical University, China<br>
        <sup>2</sup> AI Lab, GiantNetwork, China
  </p>
</div>
"""

DISCLAIMER_HTML = """
<div id="disclaimer" style="text-align:center;">
  <strong>免责声明 / Disclaimer</strong><br>
  YingMusic-Singer 可用于修改歌词后的歌声合成，支持艺术创作与娱乐应用场景。潜在风险包括未经授权的声音克隆与版权侵权问题。为确保负责任地使用，用户应在使用他人声音前取得授权、公开 AI 的参与情况，并确认音乐内容的原创性。<br>
  <span style="opacity:0.75;">YingMusic-Singer enables the creation of singing voices with modified lyrics, supporting artistic creation and entertainment. Potential risks include unauthorized voice cloning and copyright infringement. To ensure responsible deployment, users should obtain consent for voice usage, disclose AI involvement, and verify musical originality.</span>
</div>

<div style="text-align: center; margin: 3rem 0 2rem;">
      <img src="https://raw.githubusercontent.com/ASLP-lab/YingMusic-Singer/main/assets/institutional_logo.svg" alt="Institutional Logo" style="max-width: 600px; width: 80%; display: block; margin: 0 auto;">
</div>
"""


# ---------------------------------------------------------------------------
# Build the Gradio UI / 构建界面
# ---------------------------------------------------------------------------
def build_ui():
    with gr.Blocks(
        css=CUSTOM_CSS, title="YingMusic Singer", theme=gr.themes.Base()
    ) as demo:

        # ---- Header ----
        gr.HTML(HEADER_HTML)
        gr.HTML("<hr style='border-color:#30363d; margin: 8px 0 18px;'>")

        # ================================================================
        # ROW 1 – 音频输入 + 歌词
        # ================================================================
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gr.Markdown("#### 🎙️ 音频输入 / Audio Inputs", elem_classes="section-title")
                ref_audio = gr.Audio(
                    label="参考音频 / Reference Audio（提供音色 / Provides timbre）",
                    type="filepath",
                )
                melody_audio = gr.Audio(
                    label="旋律音频 / Melody Audio（提供旋律与时长 / Provides melody & duration）",
                    type="filepath",
                )
            with gr.Column(scale=1):
                gr.Markdown("#### ✏️ 歌词输入 / Lyrics")
                ref_text = gr.Textbox(
                    label="参考音频歌词 / Reference Lyrics",
                    placeholder="例如 / e.g.：该体谅的不执着|如果那天我",
                    lines=5,
                )
                target_text = gr.Textbox(
                    label="目标合成歌词 / Target Lyrics",
                    placeholder="例如 / e.g.：好多天|看不完你",
                    lines=5,
                )

        # ================================================================
        # ROW 2 – 伴奏分离
        # ================================================================
        gr.HTML("<hr style='border-color:#30363d; margin: 16px 0 12px;'>")
        gr.Markdown("#### 🎚️ 伴奏分离 / Vocal Separation")

        gr.HTML("""
<div style="font-size:0.85rem; color:#8b949e; line-height:1.75; margin: 0 0 12px; padding: 10px 16px;
            background: rgba(255,255,255,0.03); border-radius: 8px; border: 1px solid #21262d;">
  <ul style="margin:0; padding-left:1.2em; list-style: none;">
    <li style="margin-bottom:7px;">
      💡 若输入的<b style="color:#FE9EC7; font-weight:700;">参考音频</b>或<b style="color:#89D4FF; font-weight:700;">旋律音频</b>中含有伴奏或背景噪音，请开启「分离人声后过模型」—— 模型基于纯人声训练，混合音频会影响合成质量。<br>
      <span style="color:#6e7681; font-size:0.82rem;">If either input contains accompaniment or background noise, enable <i>Separate vocals before synthesis</i> — the model is trained on clean vocals only and mixed audio degrades quality.</span>
    </li>
    <li style="margin-bottom:7px;">
      💡 若两个输入均已为干净人声，则无需开启分离，强行开启反而可能因分离模型引入额外的不稳定性。<br>
      <span style="color:#6e7681; font-size:0.82rem;">If both inputs are already clean vocals, skip separation — enabling it unnecessarily may introduce artifacts from the separation model.</span>
    </li>
    <li>
      💡 若<b style="color:#89D4FF; font-weight:700;">旋律音频</b>含有伴奏，开启「分离人声后过模型」后，最终输出是否保留伴奏由「输出时混入伴奏」控制。<br>
      <span style="color:#6e7681; font-size:0.82rem;">If the melody audio contains accompaniment and separation is enabled, use <i>Mix accompaniment into output</i> to decide whether to include it in the final result.</span>
    </li>
  </ul>
</div>
""")
        with gr.Row():
            separate_vocals_flag = gr.Checkbox(
                value=True,
                label="分离人声后过模型 / Separate vocals before synthesis",
                info="从两个输入音频中分别提取纯人声再送入模型 / Extract clean vocals from both inputs before synthesis",
            )
            mix_accompaniment_flag = gr.Checkbox(
                value=False,
                interactive=True,
                label="输出时混入伴奏 / Mix accompaniment into output",
                info="将合成人声与分离出的伴奏混合作为最终输出（需先开启人声分离）/ Mix synthesised vocals with the separated accompaniment (requires separation enabled)",
            )

        # ================================================================
        # ROW 3 – 高级参数
        # ================================================================
        with gr.Accordion("⚙️ 高级参数 / Advanced Parameters", open=False):
            with gr.Row():
                nfe_step = gr.Slider(
                    minimum=4, maximum=128, value=32, step=1,
                    label="采样步数 / NFE Steps",
                    info="步数越多质量越高，但速度更慢 / More steps = higher quality, but slower",
                )
                cfg_strength = gr.Slider(
                    minimum=0.0, maximum=10.0, value=3.0, step=0.1,
                    label="引导强度 / CFG Strength",
                    info="无分类器引导强度 / Classifier-Free Guidance strength",
                )
                t_shift = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                    label="采样时间偏移 / t‑shift",
                )
            with gr.Row():
                sil_len_to_end = gr.Slider(
                    minimum=0.0, maximum=3.0, value=0.5, step=0.1,
                    label="末尾静音时长（秒）/ Silence Padding (s)",
                    info="在参考音频末尾追加的静音长度 / Silence appended after reference audio",
                )
                seed = gr.Number(
                    value=-1, precision=0,
                    label="随机种子 / Random Seed",
                    info="-1 表示随机生成 / -1 means random",
                )

        # ================================================================
        # ROW 4 – 预设示例（放在所有真实控件定义之后）
        # ================================================================
        gr.HTML("<hr style='border-color:#30363d; margin: 16px 0 12px;'>")
        gr.Markdown("#### 🎵 预设示例 / Example Presets")
        gr.Markdown(
            "<small style='color:#8b949e;'>点击任意行自动填入上方输入区域 / Click any row to auto-fill the inputs above</small>"
        )
        gr.HTML("""
<p style="text-align:center; font-size:0.78rem; color:#484f58; margin: 2px 0 10px; line-height:1.7;">
  示例中所用音频片段均仅供学术研究与演示目的，不用于任何商业用途。如有版权问题，请联系作者予以删除。<br>
  Audio clips used in the examples are for academic research and demonstration purposes only, with no commercial use intended. If you believe any content infringes your copyright, please contact the authors for removal.
</p>
""")

        # 所有真实控件均已定义，直接绑定
        _example_inputs = [
            ref_audio, melody_audio, ref_text, target_text,
            separate_vocals_flag, mix_accompaniment_flag,
            sil_len_to_end, t_shift, nfe_step, cfg_strength, seed,
        ]

        with gr.Tabs():
            with gr.Tab("🎼 Melody Control"):
                gr.Examples(
                    examples=EXAMPLES_MELODY_CONTROL,
                    inputs=_example_inputs,
                    label="Melody Control Examples",
                    examples_per_page=5,
                )
            with gr.Tab("✏️ Lyric Edit"):
                gr.Examples(
                    examples=EXAMPLES_LYRIC_EDIT,
                    inputs=_example_inputs,
                    label="Lyric Edit Examples",
                    examples_per_page=5,
                )

        # ================================================================
        # ROW 5 – 合成按钮与输出
        # ================================================================
        gr.HTML("<hr style='border-color:#30363d; margin: 12px 0;'>")
        run_btn = gr.Button("🎤  开始合成 / Start Synthesizing", elem_id="run-btn", size="lg")

        output_audio = gr.Audio(
            label="合成结果 / Generated Audio",
            type="filepath",
            elem_id="output-audio",
        )

        _all_inputs = [
            ref_audio, melody_audio, ref_text, target_text,
            separate_vocals_flag, mix_accompaniment_flag,
            sil_len_to_end, t_shift, nfe_step, cfg_strength, seed,
        ]

        run_btn.click(
            fn=synthesize,
            inputs=_all_inputs,
            outputs=output_audio,
        )

        gr.HTML(DISCLAIMER_HTML)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo = build_ui()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
