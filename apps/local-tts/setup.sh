#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Prefer Python 3.12 or 3.11 on Intel Mac (onnxruntime 1.23.2 has wheels for 3.10-3.13)
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.12 python3.11 python3.13 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "Error: Python 3.10-3.13 is required. Install with: brew install python@3.12"
  exit 1
fi

echo "Using $PYTHON_BIN ($("$PYTHON_BIN" --version))"

if [[ ! -d ".venv" ]]; then
  echo "Creating virtual environment in .venv ..."
  "$PYTHON_BIN" -m venv .venv
fi

echo "Installing dependencies..."
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "Downloading Kokoro-82M model files (if not already present)..."
.venv/bin/python -m app.models

echo ""
echo "Setup complete! Start the local TTS server with:"
echo "  ./run.sh"
