FROM ubuntu:22.04

ARG GPU_TYPE=cpu

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:99
ENV SCREEN_WIDTH=1280
ENV SCREEN_HEIGHT=720

# System dependencies (remove APT Post-Invoke hooks that fail in Docker build)
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    x11vnc \
    novnc \
    websockify \
    wget \
    gnupg \
    python3 \
    python3-pip \
    python3-xlib \
    python3-tk \
    libgl1-mesa-glx \
    libglib2.0-0 \
    fonts-liberation \
    fonts-dejavu-core \
    fontconfig \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    libxshmfence1 \
    xauth \
    util-linux \
    && rm -rf /var/lib/apt/lists/*

# Chromium（.deb、arm64/amd64 両対応。Ubuntu 22.04 標準は snap のため PPA で .deb を導入）
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:xtradeb/apps \
    && apt-get update && apt-get install -y --no-install-recommends chromium \
    && apt-get remove -y --purge software-properties-common \
    && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
ENV CHROME_BIN=/usr/bin/chromium

WORKDIR /app

# Install Python dependencies (cached layer)
# "can't start new thread" を避けるため: スレッド数を抑え、pip を2段階に分ける
ENV PIP_PROGRESS_BAR=off
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch based on GPU type (nvidia / amd / cpu)
RUN if [ "$GPU_TYPE" = "nvidia" ]; then \
        echo "Installing PyTorch with CUDA (NVIDIA) support..." && \
        pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121; \
    elif [ "$GPU_TYPE" = "amd" ]; then \
        echo "Installing PyTorch with ROCm (AMD) support..." && \
        pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/rocm6.2; \
    else \
        echo "Installing PyTorch (CPU only)..." && \
        pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Copy source code
COPY . .

RUN chmod +x entrypoint.sh

EXPOSE 6080

ENTRYPOINT ["./entrypoint.sh"]
