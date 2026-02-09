#!/usr/bin/env bash
# どの環境でも同じ手順で仮想環境を用意するスクリプト。
# 使い方: ./scripts/setup.sh  または  bash scripts/setup.sh
# - Python は python3 → python の順で検出
# - .venv が無い／壊れている（別マシンで作った等）場合は作り直す
# - requirements.txt をインストール

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 利用する Python を決定（python3 を優先、無ければ python）
PYTHON=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
      PYTHON="$cand"
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  echo "Error: Python 3.9+ not found. Install python3 or python and retry." >&2
  exit 1
fi

echo "Using: $PYTHON ($($PYTHON --version 2>&1))"

# .venv が存在するがインタープリタが無い／動かない場合は削除して作り直す
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [ -d "$PROJECT_ROOT/.venv" ]; then
  if [ ! -x "$VENV_PYTHON" ] || ! "$VENV_PYTHON" -c "import sys" 2>/dev/null; then
    echo "Removing broken or foreign .venv and recreating..."
    rm -rf "$PROJECT_ROOT/.venv"
  fi
fi

# 仮想環境が無ければ作成
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
  echo "Creating .venv with $PYTHON..."
  "$PYTHON" -m venv "$PROJECT_ROOT/.venv"
fi

# 依存インストール
echo "Installing dependencies..."
"$PROJECT_ROOT/.venv/bin/pip" install -q --upgrade pip
"$PROJECT_ROOT/.venv/bin/pip" install -r "$PROJECT_ROOT/requirements.txt"

echo "Setup done. Run: ./scripts/run.sh [bot|debug]  or  source .venv/bin/activate && python main.py bot"
