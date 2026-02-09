#!/bin/bash
# ローカル (非Docker) 環境で Xvfb + VNC + bot を起動するスクリプト
# WSL2 / Linux デスクトップ用
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cleanup() {
    echo "Shutting down..."
    # disown されたプロセスも含めて終了
    kill "$XVFB_PID" "$X11VNC_PID" 2>/dev/null || true
    pkill -f "websockify.*6080" 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT

# --- 設定 ---
DISPLAY_NUM=":99"
# WSL2 では /tmp/.X11-unix が読み取り専用のため UNIX ソケットが作れない。
# TCP 接続を使うので DISPLAY を localhost:99 に設定する。
DISPLAY_TCP="localhost${DISPLAY_NUM}"
SCREEN_W="${SCREEN_WIDTH:-1280}"
SCREEN_H="${SCREEN_HEIGHT:-720}"

export DISPLAY="$DISPLAY_TCP"
export SCREEN_WIDTH="$SCREEN_W"
export SCREEN_HEIGHT="$SCREEN_H"
export PYTHONUNBUFFERED=1
export QT_QPA_FONTDIR=/usr/share/fonts/truetype/dejavu

# WSL2 の Wayland 変数を無効化（x11vnc が Wayland を誤検出するのを防ぐ）
unset WAYLAND_DISPLAY
export XDG_SESSION_TYPE=x11

# Chrome バイナリの自動検出（非snap版を優先）
if [ -z "$CHROME_BIN" ]; then
    for candidate in /usr/bin/google-chrome-stable /usr/bin/google-chrome /usr/bin/chromium-browser /usr/bin/chromium; do
        if [ -x "$candidate" ]; then
            export CHROME_BIN="$candidate"
            break
        fi
    done
fi
echo "CHROME_BIN=$CHROME_BIN"

# --- venv のアクティベート ---
VENV_DIR="$PROJECT_DIR/.venv"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: .venv not found. Run: python3 -m venv .venv && pip install -r requirements.txt"
    exit 1
fi
source "$VENV_DIR/bin/activate"

# --- X 認証 ---
XAUTH_FILE="$HOME/.Xauthority"
COOKIE=$(mcookie 2>/dev/null || printf '%016x' $(od -An -N8 -tx1 /dev/urandom | tr -d ' '))
# UNIX ソケット用と TCP 用の両方に認証を追加
xauth -f "$XAUTH_FILE" add "$DISPLAY_NUM" . "$COOKIE" 2>/dev/null || true
xauth -f "$XAUTH_FILE" add "$DISPLAY_TCP" . "$COOKIE" 2>/dev/null || true
touch "$XAUTH_FILE"
chmod 600 "$XAUTH_FILE"
export XAUTHORITY="$XAUTH_FILE"

# --- Xvfb 起動 ---
# WSL2 では /tmp/.X11-unix が読み取り専用のため、ソケットファイルではなく
# プロセス存在 + xdpyinfo で起動確認する
XVFB_PID=""
if xdpyinfo -display "$DISPLAY_TCP" >/dev/null 2>&1; then
    echo "Xvfb already running on $DISPLAY_TCP"
else
    echo "Starting Xvfb on $DISPLAY_NUM (${SCREEN_W}x${SCREEN_H}) with TCP..."
    Xvfb "$DISPLAY_NUM" -screen 0 "${SCREEN_W}x${SCREEN_H}x24" +extension RANDR -auth "$XAUTH_FILE" -listen tcp &
    XVFB_PID=$!
    XVFB_READY=0
    for i in $(seq 1 30); do
        if xdpyinfo -display "$DISPLAY_TCP" >/dev/null 2>&1; then
            echo "Xvfb is ready (PID=$XVFB_PID)."
            XVFB_READY=1
            break
        fi
        # プロセスが死んでいたら即失敗
        if ! kill -0 "$XVFB_PID" 2>/dev/null; then
            echo "ERROR: Xvfb process exited unexpectedly."
            exit 1
        fi
        sleep 0.5
    done
    if [ "$XVFB_READY" -eq 0 ]; then
        echo "ERROR: Xvfb failed to become ready within 15 seconds."
        exit 1
    fi
fi

# --- x11vnc 起動 ---
echo "Starting x11vnc..."
# env -u で WAYLAND_DISPLAY を完全除去してから x11vnc を起動
# -noshm: TCP接続では MIT-SHM が使えないため無効化
env -u WAYLAND_DISPLAY -u XDG_SESSION_TYPE \
    DISPLAY="$DISPLAY_TCP" XAUTHORITY="$XAUTH_FILE" \
    x11vnc -display "$DISPLAY_TCP" -auth "$XAUTH_FILE" \
    -nopw -shared -forever -noxdamage -noxrecord -noshm \
    -rfbport 5900 &
X11VNC_PID=$!
sleep 2

# x11vnc が生存しているか確認
if kill -0 "$X11VNC_PID" 2>/dev/null; then
    echo "x11vnc is running (PID=$X11VNC_PID)."
else
    echo "WARNING: x11vnc failed. VNC will not be available."
fi

# --- noVNC (websockify) 起動 ---
NOVNC_DIR=""
for d in /usr/share/novnc /usr/share/javascript/novnc; do
    if [ -d "$d" ]; then
        NOVNC_DIR="$d"
        break
    fi
done

if [ -n "$NOVNC_DIR" ]; then
    echo "Starting noVNC on port 6080..."
    websockify --web="$NOVNC_DIR" 6080 localhost:5900 &
    sleep 1
else
    echo "WARNING: noVNC directory not found. VNC available on port 5900 only."
fi

# バックグラウンドジョブを disown して exec 後も生存させる
disown -a

echo "=============================================="
echo "  noVNC: http://localhost:6080/vnc.html"
echo "  VNC:   localhost:5900"
echo "=============================================="

# --- スレッド数上限 ---
ulimit -u 8192 2>/dev/null || true

# --- Bot 起動 ---
MODE="${1:-bot}"
echo "Launching: python main.py $MODE"
cd "$PROJECT_DIR"
exec python main.py "$MODE"
