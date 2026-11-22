# Local TTS Studio (Kokoro-82M ONNX)

A free, 100% offline Text-to-Speech service designed to run smoothly on CPU — including **Intel Macs (x86_64)**, Apple Silicon Macs, and Linux.

## Features
- **High-quality neural voices:** 54 voices across American/British English, Spanish, French, Hindi, Italian, Japanese, Portuguese, and Mandarin (powered by [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)).
- **Intel Mac friendly:** Pins `onnxruntime==1.23.2` on macOS `x86_64` (the latest version with prebuilt Intel Mac wheels) and defaults to the fast ~109 MB `int8` quantized model.
- **Web Studio UI:** Clean local web app with voice selection, voice blending, speed control, instant playback, session history, and WAV download.
- **OpenAI-compatible API:** Exposes `POST /v1/audio/speech` so existing SDKs and local AI apps work out of the box.
- **CLI included:** Generate `.wav` files directly from your terminal.

## Quick Start (macOS / Linux)

Requires **Python 3.10 – 3.13** (on Intel Mac: `brew install python@3.12` if needed).

```bash
cd apps/local-tts
chmod +x setup.sh run.sh
./setup.sh
./run.sh
```

Then open **http://127.0.0.1:8000** in your browser.

> The first run automatically downloads `kokoro-v1.0.int8.onnx` (~88 MB) and `voices-v1.0.bin` (~27 MB) into `apps/local-tts/models/`.

## Switching Model Quality

You can pick between 3 model variants via `KOKORO_MODEL_VARIANT`:

| Variant | Size | Notes |
| :--- | :--- | :--- |
| `int8` *(default)* | ~109 MB | Fastest on Intel Mac CPU |
| `fp16` | ~156 MB | Higher fidelity, still compact |
| `fp32` | ~310 MB | Full precision, highest quality |

Example:
```bash
KOKORO_MODEL_VARIANT=fp16 ./run.sh
```

## CLI Usage

```bash
# List all available voices
.venv/bin/python -m app.cli --list-voices

# Generate speech to a WAV file
.venv/bin/python -m app.cli "Hello from my Intel Mac!" -v af_heart -o hello.wav

# Blend two voices
.venv/bin/python -m app.cli "Custom blended voice." -v af_heart -b am_michael --blend-ratio 0.6 -o blend.wav
```

## API Usage

### Native endpoint (`POST /api/tts`)
```bash
curl http://127.0.0.1:8000/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "voice": "af_heart", "speed": 1.0}' \
  --output speech.wav
```

### OpenAI-compatible endpoint (`POST /v1/audio/speech`)
```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro", "voice": "alloy", "input": "Hello world!"}' \
  --output speech.wav
```
