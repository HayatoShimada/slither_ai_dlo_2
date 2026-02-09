#!/bin/bash
set -e

cleanup() {
    echo "Shutting down..."
    kill $(jobs -p) 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT

# Qt/OpenCV のフォント警告を減らす（DejaVu を参照）
export QT_QPA_FONTDIR=/usr/share/fonts/truetype/dejavu

# X 認証を作成（pyautogui/mouseinfo が ~/.Xauthority を要求するため）
COOKIE=$(mcookie 2>/dev/null || printf '%016x' $(od -An -N8 -tx1 /dev/urandom | tr -d ' '))
xauth -f /root/.Xauthority add "${DISPLAY}" . "${COOKIE}" 2>/dev/null || touch /root/.Xauthority
chmod 600 /root/.Xauthority
export XAUTHORITY=/root/.Xauthority

# Start Xvfb（上で作った認証を使う）
echo "Starting Xvfb on display ${DISPLAY} (${SCREEN_WIDTH}x${SCREEN_HEIGHT})..."
Xvfb "${DISPLAY}" -screen 0 ${SCREEN_WIDTH}x${SCREEN_HEIGHT}x24 +extension RANDR -auth /root/.Xauthority &

# Wait for Xvfb to be ready
echo "Waiting for Xvfb..."
for i in $(seq 1 30); do
    if [ -e /tmp/.X11-unix/X99 ]; then
        echo "Xvfb is ready."
        break
    fi
    sleep 0.5
done

if [ ! -e /tmp/.X11-unix/X99 ]; then
    echo "ERROR: Xvfb failed to start."
    exit 1
fi

# Start x11vnc
echo "Starting x11vnc..."
x11vnc -display ${DISPLAY} -nopw -shared -forever -noxdamage -rfbport 5900 &
sleep 1

# Start noVNC websockify proxy
echo "Starting noVNC on port 6080..."
websockify --web=/usr/share/novnc/ 6080 localhost:5900 &
sleep 1

echo "=============================================="
echo "  noVNC (認識モニタ) のアクセス方法"
echo "  同じマシン:  http://localhost:6080"
echo "  リモート:    http://<このサーバのIP>:6080"
echo "  例:          http://192.168.1.10:6080"
echo "  (サーバのIPは hostname -I で確認)"
echo "=============================================="

# GPU 環境検出ログ
echo "--- GPU Environment ---"
if command -v nvidia-smi &>/dev/null; then
    echo "[GPU] NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi available but query failed"
elif [ -e /dev/kfd ]; then
    echo "[GPU] AMD GPU (ROCm) detected:"
    if command -v rocm-smi &>/dev/null; then
        rocm-smi --showproductname 2>/dev/null || echo "  rocm-smi available but query failed"
    else
        echo "  /dev/kfd exists (ROCm device available)"
    fi
else
    echo "[GPU] No GPU device found, using CPU"
fi
echo "-----------------------"
echo "Launching bot..."

# コンテナのスレッド数制限を引き上げ（matplotlib / PyTorch の "can't start new thread" 対策）
ulimit -u 8192 2>/dev/null || true

# Run main application as PID 1 replacement
exec python3 main.py bot
