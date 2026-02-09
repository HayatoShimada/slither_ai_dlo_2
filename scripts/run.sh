#!/usr/bin/env bash
# 仮想環境を利用して main.py を実行する。venv が無い／壊れていれば setup を実行してから起動。
# 使い方:
#   ./scripts/run.sh          # 骨格可視化
#   ./scripts/run.sh debug    # HSV デバッグ
#   ./scripts/run.sh bot      # 自動運転 + 強化学習

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

# venv が無い、またはインタープリタが動かない場合はセットアップを実行
if [ ! -x "$VENV_PYTHON" ] || ! "$VENV_PYTHON" -c "import sys" 2>/dev/null; then
  echo "Virtual environment missing or broken. Running setup..."
  bash "$SCRIPT_DIR/setup.sh"
fi

exec "$VENV_PYTHON" "$PROJECT_ROOT/main.py" "$@"
