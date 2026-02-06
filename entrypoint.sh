#!/bin/bash
set -e

cleanup() {
    echo "Shutting down..."
    kill $(jobs -p) 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT

# Start Xvfb
echo "Starting Xvfb on display ${DISPLAY} (${SCREEN_WIDTH}x${SCREEN_HEIGHT})..."
Xvfb ${DISPLAY} -screen 0 ${SCREEN_WIDTH}x${SCREEN_HEIGHT}x24 +extension RANDR &

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

echo "noVNC available at http://localhost:6080"
echo "Launching bot..."

# Run main application as PID 1 replacement
exec python3 main.py bot
