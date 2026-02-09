FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

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

# Google Chrome (snap 不要、Docker 対応)
RUN wget -q -O /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
    && apt-get update && apt-get install -y /tmp/chrome.deb \
    && rm -f /tmp/chrome.deb && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (cached layer)
# "can't start new thread" を避けるため: スレッド数を抑え、pip を2段階に分ける
ENV PIP_PROGRESS_BAR=off
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Copy source code
COPY . .

RUN chmod +x entrypoint.sh

EXPOSE 6080

ENTRYPOINT ["./entrypoint.sh"]
