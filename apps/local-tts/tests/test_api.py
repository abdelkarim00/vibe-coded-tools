"""End-to-end smoke tests for the Local TTS FastAPI server."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.server import app


def test_health_and_voices() -> None:
    with TestClient(app) as client:
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["voices_count"] > 0

        voices_res = client.get("/api/voices")
        assert voices_res.status_code == 200
        voices_data = voices_res.json()
        assert len(voices_data["voices"]) == data["voices_count"]

        ui_res = client.get("/")
        assert ui_res.status_code == 200
        assert "Local TTS Studio" in ui_res.text


def test_tts_and_openai_endpoints() -> None:
    with TestClient(app) as client:
        res = client.post(
            "/api/tts",
            json={"text": "Hello from local TTS.", "voice": "af_heart", "speed": 1.0},
        )
        assert res.status_code == 200
        assert res.headers["content-type"] == "audio/wav"
        assert res.content[:4] == b"RIFF"
        assert float(res.headers["x-audio-duration"]) > 0.5

        openai_res = client.post(
            "/v1/audio/speech",
            json={"model": "kokoro", "voice": "alloy", "input": "Testing OpenAI compatibility."},
        )
        assert openai_res.status_code == 200
        assert openai_res.headers["content-type"] == "audio/wav"
        assert openai_res.content[:4] == b"RIFF"


if __name__ == "__main__":
    test_health_and_voices()
    test_tts_and_openai_endpoints()
    print("All API smoke tests passed!")
