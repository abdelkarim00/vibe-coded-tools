#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

is_compatible_python() {
  "$1" -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)" >/dev/null 2>&1
}

# Prefer Python 3.12 / 3.11 / 3.13 / 3.10 (onnxruntime 1.23.2 has Intel Mac wheels for 3.10-3.13)
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.12 python3.11 python3.13 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && is_compatible_python "$candidate"; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

# Fallback: check pyenv-installed versions directly (avoids inactive pyenv shim errors)
if [[ -z "$PYTHON_BIN" && -d "${PYENV_ROOT:-$HOME/.pyenv}/versions" ]]; then
  for py in "${PYENV_ROOT:-$HOME/.pyenv}"/versions/3.1[0-3]*/bin/python3; do
    if [[ -x "$py" ]] && is_compatible_python "$py"; then
      PYTHON_BIN="$py"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "Error: A runnable Python 3.10-3.13 is required."
  echo "Tip: If using pyenv, run: pyenv local 3.12.3"
  echo "     Or install with Homebrew: brew install python@3.12"
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
