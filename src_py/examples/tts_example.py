# Copyright 2025 Prism Shadow. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example demonstrating Gemini single-speaker TTS generation.

This example sends a text-only prompt with transcript tags such as [excitedly],
collects the returned audio chunks, and saves the final output as a WAV file.
"""

import asyncio
import os
import wave
from pathlib import Path

from agenthub import AutoLLMClient


def save_wav_file(
    filename: Path, pcm_data: bytes, channels: int = 1, rate: int = 24000, sample_width: int = 2
) -> None:
    """Save raw PCM bytes as a WAV file using Gemini TTS defaults."""

    with wave.open(str(filename), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(rate)
        wav_file.writeframes(pcm_data)


async def main():
    """Example of single-speaker Gemini TTS generation."""
    print("=" * 60)
    print("Gemini TTS Example")
    print("=" * 60)

    # Get model from environment variable, default to gemini-3.1-flash-tts-preview
    model = os.getenv("MODEL", "gemini-3.1-flash-tts-preview")
    print(f"Using model: {model}")

    client = AutoLLMClient(model=model)
    prompt = """Synthesize speech for the transcript below using a single speaker.

### TRANSCRIPT
[excitedly] Welcome to AgentHub! We just added Gemini text-to-speech support.
[warmly] This demo saves the generated audio as a WAV file so you can play it back right away.
[whispers] And yes, prompt tags like this can shape the performance.
"""
    output_path = Path.cwd() / "generated_audio.wav"

    print("User:")
    print(prompt)
    print("Assistant:")
    print(f"Saving output to: {output_path}")

    audio_chunks: list[bytes] = []
    async for event in client.streaming_response(
        messages=[{"role": "user", "content_items": [{"type": "text", "text": prompt}]}],
        config={"tts_config": {"speaker_voices": [{"voice": "Kore"}]}},
    ):
        for item in event["content_items"]:
            if item["type"] == "inline_data":
                audio_chunks.append(item["data"])

    if not audio_chunks:
        raise RuntimeError("No audio data was returned by the model.")

    save_wav_file(output_path, b"".join(audio_chunks))
    print(f"Saved WAV file to: {output_path}")

    print("\n" + "=" * 60)
    print("Gemini TTS complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
