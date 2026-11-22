"""Core TTS engine wrapper around kokoro-onnx with chunking and voice blending."""

from __future__ import annotations

import io
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

from .models import MODEL_VARIANTS, ensure_models

# Kokoro voice prefix -> (language_code, accent_label, gender)
VOICE_PREFIX_META: dict[str, tuple[str, str, str]] = {
    "af": ("en-us", "American English", "Female"),
    "am": ("en-us", "American English", "Male"),
    "bf": ("en-gb", "British English", "Female"),
    "bm": ("en-gb", "British English", "Male"),
    "ef": ("es", "Spanish", "Female"),
    "em": ("es", "Spanish", "Male"),
    "ff": ("fr-fr", "French", "Female"),
    "hf": ("hi", "Hindi", "Female"),
    "hm": ("hi", "Hindi", "Male"),
    "if": ("it", "Italian", "Female"),
    "im": ("it", "Italian", "Male"),
    "jf": ("ja", "Japanese", "Female"),
    "jm": ("ja", "Japanese", "Male"),
    "pf": ("pt-br", "Portuguese (BR)", "Female"),
    "pm": ("pt-br", "Portuguese (BR)", "Male"),
    "zf": ("cmn", "Mandarin Chinese", "Female"),
    "zm": ("cmn", "Mandarin Chinese", "Male"),
}

# Friendly mapping for OpenAI-compatible /v1/audio/speech voice names
OPENAI_VOICE_MAP: dict[str, str] = {
    "alloy": "af_heart",
    "ash": "am_adam",
    "ballad": "bm_george",
    "coral": "af_bella",
    "echo": "am_michael",
    "fable": "bf_emma",
    "nova": "af_nova",
    "onyx": "am_onyx",
    "sage": "af_sky",
    "shimmer": "af_sarah",
}

DEFAULT_VOICE = "af_heart"


@dataclass
class SynthesisResult:
    """Result of a TTS synthesis call."""

    wav_bytes: bytes
    sample_rate: int
    duration_seconds: float
    generation_seconds: float
    voice: str
    lang: str

    @property
    def real_time_factor(self) -> float:
        """Compute RTF (generation_time / audio_duration). < 1.0 means faster than real-time."""
        if self.duration_seconds <= 0:
            return 0.0
        return self.generation_seconds / self.duration_seconds


def _describe_voice(voice_id: str) -> dict[str, str]:
    """Return structured metadata for a Kokoro voice ID like 'af_heart'."""
    parts = voice_id.split("_", 1)
    prefix = parts[0] if len(parts) == 2 else ""
    raw_name = parts[1] if len(parts) == 2 else voice_id
    display_name = raw_name.replace("_", " ").title()
    lang, accent, gender = VOICE_PREFIX_META.get(prefix, ("en-us", "Other", "Neutral"))
    return {
        "id": voice_id,
        "name": display_name,
        "accent": accent,
        "gender": gender,
        "lang": lang,
    }


def _split_text_into_chunks(text: str, max_chars: int = 350) -> list[str]:
    """Split long text on sentence boundaries so Kokoro handles arbitrarily long input cleanly."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    sentences = re.split(r"(?<=[.!?。！？])\s+", cleaned)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            # Fallback: split overly long sentence by commas or words
            words = sentence.split(" ")
            buf = ""
            for word in words:
                if len(buf) + len(word) + 1 <= max_chars:
                    buf = f"{buf} {word}".strip()
                else:
                    if buf:
                        chunks.append(buf)
                    buf = word
            if buf:
                chunks.append(buf)
        elif len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current.strip())
            current = sentence

    if current:
        chunks.append(current.strip())

    return chunks


class LocalTTSEngine:
    """Thread-safe local TTS engine backed by Kokoro-82M ONNX."""

    def __init__(self, variant: str | None = None) -> None:
        self._lock = threading.Lock()
        self.model_path, self.voices_path, self.variant = ensure_models(variant)
        self.kokoro = Kokoro(str(self.model_path), str(self.voices_path))
        self._voices = sorted(self.kokoro.get_voices())

    def list_voices(self) -> list[dict[str, str]]:
        """Return all available voices with accent, gender, and language metadata."""
        return [_describe_voice(v) for v in self._voices]

    def resolve_voice_id(self, voice: str) -> str:
        """Map an OpenAI voice alias or validate a Kokoro voice ID."""
        candidate = OPENAI_VOICE_MAP.get(voice.lower(), voice)
        if candidate not in self._voices:
            raise ValueError(
                f"Unknown voice '{voice}'. Available voices include: "
                f"{', '.join(self._voices[:10])} ... (total {len(self._voices)})"
            )
        return candidate

    def get_info(self) -> dict[str, Any]:
        """Return runtime metadata about the loaded TTS model."""
        return {
            "engine": "kokoro-onnx",
            "variant": self.variant,
            "variant_description": MODEL_VARIANTS[self.variant]["description"],
            "model_file": self.model_path.name,
            "voices_count": len(self._voices),
            "default_voice": DEFAULT_VOICE if DEFAULT_VOICE in self._voices else self._voices[0],
        }

    def synthesize(
        self,
        text: str,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        lang: str | None = None,
        blend_voice: str | None = None,
        blend_ratio: float = 0.5,
    ) -> SynthesisResult:
        """Synthesize text into WAV audio bytes.

        Args:
            text: Input text to speak.
            voice: Kokoro voice ID (e.g. 'af_heart') or OpenAI alias (e.g. 'alloy').
            speed: Playback speed multiplier (0.5 to 2.0).
            lang: Optional espeak language code. Auto-detected from voice prefix if omitted.
            blend_voice: Optional second Kokoro voice ID to blend with the primary voice.
            blend_ratio: Weight of the primary voice (0.0 to 1.0) when blending.
        """
        chunks = _split_text_into_chunks(text)
        if not chunks:
            raise ValueError("Input text cannot be empty.")

        primary_id = self.resolve_voice_id(voice)
        resolved_lang = lang or _describe_voice(primary_id)["lang"]
        clamped_speed = max(0.5, min(2.0, float(speed)))

        voice_arg: str | np.ndarray = primary_id
        voice_label = primary_id

        if blend_voice:
            secondary_id = self.resolve_voice_id(blend_voice)
            ratio = max(0.0, min(1.0, float(blend_ratio)))
            primary_style = self.kokoro.get_voice_style(primary_id)
            secondary_style = self.kokoro.get_voice_style(secondary_id)
            voice_arg = primary_style * ratio + secondary_style * (1.0 - ratio)
            voice_label = f"{primary_id}:{int(ratio * 100)}+{secondary_id}:{int((1 - ratio) * 100)}"

        start_time = time.perf_counter()
        audio_segments: list[np.ndarray] = []
        sample_rate = 24000

        with self._lock:
            for idx, chunk in enumerate(chunks):
                samples, sr = self.kokoro.create(
                    chunk,
                    voice=voice_arg,
                    speed=clamped_speed,
                    lang=resolved_lang,
                )
                sample_rate = sr
                audio_segments.append(samples)
                # Add a brief natural pause (~180ms) between chunks
                if idx < len(chunks) - 1:
                    audio_segments.append(np.zeros(int(sample_rate * 0.18), dtype=np.float32))

        full_audio = np.concatenate(audio_segments) if len(audio_segments) > 1 else audio_segments[0]
        generation_seconds = time.perf_counter() - start_time
        duration_seconds = len(full_audio) / float(sample_rate)

        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, full_audio, sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = wav_buffer.getvalue()

        return SynthesisResult(
            wav_bytes=wav_bytes,
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            generation_seconds=generation_seconds,
            voice=voice_label,
            lang=resolved_lang,
        )
