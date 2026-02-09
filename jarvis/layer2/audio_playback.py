import asyncio
import threading

import pyaudio

from jarvis.config import RECEIVE_SAMPLE_RATE, AUDIO_CHANNELS, AUDIO_FORMAT_WIDTH


class AudioPlayback:
    """Plays audio responses from Gemini Live API."""

    def __init__(self):
        self._queue = asyncio.Queue()
        self._pya = None
        self._stream = None
        self._running = False

    def _ensure_pyaudio(self):
        if self._pya is None:
            self._pya = pyaudio.PyAudio()

    async def start(self):
        """Start the audio playback loop."""
        self._ensure_pyaudio()
        self._running = True

        self._stream = self._pya.open(
            format=pyaudio.paInt16,
            channels=AUDIO_CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )

        print("[AudioPlayback] Started")

        while self._running:
            try:
                audio_data = await asyncio.wait_for(
                    self._queue.get(), timeout=0.5
                )
                await asyncio.to_thread(self._stream.write, audio_data)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"[AudioPlayback] Error: {e}")

    async def enqueue(self, audio_data):
        """Add audio data to the playback queue."""
        await self._queue.put(audio_data)

    def clear_queue(self):
        """Clear pending audio (e.g., on interruption)."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def stop(self):
        """Stop playback."""
        self._running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pya:
            self._pya.terminate()
            self._pya = None
        print("[AudioPlayback] Stopped")
