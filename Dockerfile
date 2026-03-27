# ============================================================
# YingMusic-Singer — Local Docker (NVIDIA GPU)
# ============================================================
# Build:  docker build -t yingmusic-singer .
# Run:    
#   docker run --gpus all -p 7860:7860 \
#   -v $(pwd)/ckpts:/app/ckpts \
#   -v $(pwd)/hf_cache:/root/.cache/huggingface \
#   yingmusic-singer
# ============================================================

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# ── System deps + espeak-ng ──────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip \
        espeak-ng libespeak-ng1 \
        ffmpeg libsndfile1 \
        git wget curl \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Upgrade pip ──────────────────────────────────────────────
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ── Python dependencies ──────────────────────────────────────
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Enable USTC mirror source for acceleration during domestic builds (remove the -i parameter for overseas environments)
# RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.ustc.edu.cn/pypi/simple --trusted-host mirrors.ustc.edu.cn

# ── Application code ─────────────────────────────────────────
COPY . .

#RUN python initialization.py --task infer

EXPOSE 7860

CMD ["python", "app_local.py"]
