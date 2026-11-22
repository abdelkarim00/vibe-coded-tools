"""Simple CLI for generating speech locally from the terminal."""

from __future__ import annotations

import argparse
from pathlib import Path

from .tts_engine import DEFAULT_VOICE, LocalTTSEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate local TTS audio with Kokoro-82M.")
    parser.add_argument("text", nargs="?", help="Text to convert to speech.")
    parser.add_argument("-v", "--voice", default=DEFAULT_VOICE, help=f"Voice ID (default: {DEFAULT_VOICE})")
    parser.add_argument("-b", "--blend-voice", default=None, help="Optional second voice to blend with.")
    parser.add_argument("--blend-ratio", type=float, default=0.5, help="Primary voice weight (0.0-1.0).")
    parser.add_argument("-s", "--speed", type=float, default=1.0, help="Speech speed (0.5-2.0).")
    parser.add_argument("-o", "--output", default="output.wav", help="Output WAV file path.")
    parser.add_argument("--list-voices", action="store_true", help="List all available voices and exit.")
    args = parser.parse_args()

    engine = LocalTTSEngine()

    if args.list_voices:
        for v in engine.list_voices():
            print(f"{v['id']:18s} {v['accent']:18s} {v['gender']:8s} ({v['lang']})")
        return

    if not args.text:
        parser.error("Provide text to synthesize, or pass --list-voices.")

    result = engine.synthesize(
        text=args.text,
        voice=args.voice,
        speed=args.speed,
        blend_voice=args.blend_voice,
        blend_ratio=args.blend_ratio,
    )

    out_path = Path(args.output)
    out_path.write_bytes(result.wav_bytes)
    print(
        f"Saved {out_path} ({result.duration_seconds:.2f}s audio generated in "
        f"{result.generation_seconds:.2f}s, RTF={result.real_time_factor:.2f}x)"
    )


if __name__ == "__main__":
    main()
