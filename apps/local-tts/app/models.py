"""Model management and automatic downloading for Kokoro-82M ONNX."""

from __future__ import annotations

import os
from pathlib import Path

import httpx

BASE_RELEASE_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
)

MODEL_VARIANTS: dict[str, dict[str, str]] = {
    "int8": {
        "filename": "kokoro-v1.0.int8.onnx",
        "description": "Quantized INT8 (~109 MB) — fastest on CPU / Intel Mac",
    },
    "fp16": {
        "filename": "kokoro-v1.0.fp16.onnx",
        "description": "Half-precision FP16 (~156 MB) — great quality & compact",
    },
    "fp32": {
        "filename": "kokoro-v1.0.onnx",
        "description": "Full-precision FP32 (~310 MB) — highest quality",
    },
}

VOICES_FILENAME = "voices-v1.0.bin"


def get_models_dir() -> Path:
    """Return the directory where ONNX models and voice packs are stored."""
    env_dir = os.environ.get("KOKORO_MODELS_DIR")
    if env_dir:
        models_dir = Path(env_dir).expanduser().resolve()
    else:
        models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_selected_variant() -> str:
    """Return the configured model variant (int8, fp16, or fp32)."""
    variant = os.environ.get("KOKORO_MODEL_VARIANT", "int8").strip().lower()
    if variant not in MODEL_VARIANTS:
        valid = ", ".join(MODEL_VARIANTS)
        raise ValueError(f"Unknown KOKORO_MODEL_VARIANT '{variant}'. Choose from: {valid}")
    return variant


def _download_file(url: str, dest: Path) -> None:
    """Stream-download a file to disk via a temporary .part file."""
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {dest.name} from {url} ...")

    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        last_pct = -1

        with tmp_path.open("wb") as fh:
            for chunk in response.iter_bytes(chunk_size=1024 * 256):
                fh.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = int(downloaded * 100 / total)
                    if pct >= last_pct + 10 or pct == 100:
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total / (1024 * 1024)
                        print(f"  {dest.name}: {pct:3d}% ({mb_done:.1f} / {mb_total:.1f} MB)")
                        last_pct = pct

    tmp_path.replace(dest)
    print(f"Saved {dest}")


def ensure_models(variant: str | None = None) -> tuple[Path, Path, str]:
    """Ensure both the Kokoro ONNX model and voices pack exist locally.

    Returns:
        Tuple of (model_path, voices_path, resolved_variant).
    """
    resolved_variant = variant or get_selected_variant()
    if resolved_variant not in MODEL_VARIANTS:
        valid = ", ".join(MODEL_VARIANTS)
        raise ValueError(f"Unknown model variant '{resolved_variant}'. Choose from: {valid}")

    models_dir = get_models_dir()
    model_filename = MODEL_VARIANTS[resolved_variant]["filename"]
    model_path = models_dir / model_filename
    voices_path = models_dir / VOICES_FILENAME

    if not model_path.exists():
        _download_file(f"{BASE_RELEASE_URL}/{model_filename}", model_path)

    if not voices_path.exists():
        _download_file(f"{BASE_RELEASE_URL}/{VOICES_FILENAME}", voices_path)

    return model_path, voices_path, resolved_variant


if __name__ == "__main__":
    ensure_models()
