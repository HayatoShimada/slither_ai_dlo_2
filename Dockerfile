FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
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
    chromium-browser \
    chromium-chromedriver \
    python3 \
    python3-pip \
    python3-xlib \
    python3-tk \
    libgl1-mesa-glx \
    libglib2.0-0 \
    fonts-liberation \
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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (cached layer)
ENV PIP_PROGRESS_BAR=off
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Copy source code
COPY . .

RUN chmod +x entrypoint.sh

EXPOSE 6080

ENTRYPOINT ["./entrypoint.sh"]
