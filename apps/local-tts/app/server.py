"""FastAPI server for the Local TTS service (Web UI + REST API + OpenAI-compatible endpoint)."""

from __future__ import annotations

import asyncio
import os
import platform
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .tts_engine import DEFAULT_VOICE, OPENAI_VOICE_MAP, LocalTTSEngine

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

_engine: LocalTTSEngine | None = None


def get_engine() -> LocalTTSEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="TTS engine is still loading.")
    return _engine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _engine
    _engine = await asyncio.to_thread(LocalTTSEngine)
    yield
    _engine = None


app = FastAPI(
    title="Local TTS Service",
    description="100% local Text-to-Speech API powered by Kokoro-82M ONNX.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Generation-Time", "X-Audio-Duration", "X-RTF", "X-Voice"],
)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000, description="Text to synthesize.")
    voice: str = Field(DEFAULT_VOICE, description="Kokoro voice ID or OpenAI alias.")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed multiplier.")
    lang: str | None = Field(None, description="Optional espeak language code override.")
    blend_voice: str | None = Field(None, description="Optional second voice ID to blend.")
    blend_ratio: float = Field(0.5, ge=0.0, le=1.0, description="Primary voice weight when blending.")


class OpenAISpeechRequest(BaseModel):
    model: str = Field("kokoro", description="Ignored; uses the loaded Kokoro model.")
    input: str = Field(..., min_length=1, max_length=20000, description="Text to synthesize.")
    voice: str = Field("alloy", description="OpenAI voice name or any Kokoro voice ID.")
    response_format: Literal["wav", "pcm"] = Field("wav", description="Audio output format.")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed multiplier.")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict:
    engine = get_engine()
    return {
        "status": "ok",
        "platform": f"{platform.system()} {platform.machine()}",
        **engine.get_info(),
    }


@app.get("/api/voices")
async def list_voices() -> dict:
    engine = get_engine()
    return {
        "voices": engine.list_voices(),
        "openai_aliases": OPENAI_VOICE_MAP,
    }


@app.post("/api/tts")
async def synthesize_tts(req: TTSRequest) -> Response:
    engine = get_engine()
    try:
        result = await asyncio.to_thread(
            engine.synthesize,
            text=req.text,
            voice=req.voice,
            speed=req.speed,
            lang=req.lang,
            blend_voice=req.blend_voice or None,
            blend_ratio=req.blend_ratio,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(
        content=result.wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'inline; filename="speech.wav"',
            "X-Generation-Time": f"{result.generation_seconds:.3f}",
            "X-Audio-Duration": f"{result.duration_seconds:.3f}",
            "X-RTF": f"{result.real_time_factor:.3f}",
            "X-Voice": result.voice,
        },
    )


@app.post("/v1/audio/speech")
async def openai_compatible_speech(req: OpenAISpeechRequest) -> Response:
    """OpenAI-compatible TTS endpoint so existing clients/SDKs work out of the box."""
    engine = get_engine()
    try:
        result = await asyncio.to_thread(
            engine.synthesize,
            text=req.input,
            voice=req.voice,
            speed=req.speed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if req.response_format == "pcm":
        # Strip standard 44-byte WAV header for raw 16-bit PCM
        content = result.wav_bytes[44:]
        media_type = "audio/pcm"
        filename = "speech.pcm"
    else:
        content = result.wav_bytes
        media_type = "audio/wav"
        filename = "speech.wav"

    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "X-Generation-Time": f"{result.generation_seconds:.3f}",
            "X-Audio-Duration": f"{result.duration_seconds:.3f}",
            "X-RTF": f"{result.real_time_factor:.3f}",
            "X-Voice": result.voice,
        },
    )


def main() -> None:
    import uvicorn

    host = os.environ.get("TTS_HOST", "127.0.0.1")
    port = int(os.environ.get("TTS_PORT", "8000"))
    uvicorn.run("app.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
