#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -d ".venv" ]]; then
  echo "Virtual environment not found. Running ./setup.sh first..."
  ./setup.sh
fi

export TTS_HOST="${TTS_HOST:-127.0.0.1}"
export TTS_PORT="${TTS_PORT:-8000}"

echo "Starting Local TTS Studio on http://${TTS_HOST}:${TTS_PORT}"
exec .venv/bin/python -m app.server
