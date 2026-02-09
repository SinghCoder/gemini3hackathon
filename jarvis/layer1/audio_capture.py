import asyncio
import logging
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from jarvis.config import (
    SEND_SAMPLE_RATE,
    AUDIO_CHANNELS,
    WHISPER_MODEL,
    TRANSCRIPTION_CHUNK_SECONDS,
)
from jarvis.utils.buffer import TranscriptBuffer
from jarvis.utils.observe import trace_span, flush

log = logging.getLogger("jarvis.audio")


class AudioCapture:
    """Captures system audio and transcribes it locally using faster-whisper."""

    def __init__(self, transcript_buffer: TranscriptBuffer, device=None):
        self.transcript_buffer = transcript_buffer
        self.device = device
        self._running = False
        self._whisper = None
        self._audio_chunks = []
        self._lock = threading.Lock()
        self._chunk_count = 0

    def _load_whisper(self):
        if self._whisper is None:
            log.info("Loading Whisper model: %s", WHISPER_MODEL)
            self._whisper = WhisperModel(
                WHISPER_MODEL, device="cpu", compute_type="int8"
            )
            log.info("Whisper model loaded")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            log.warning("Audio status: %s", status)
        audio_data = indata[:, 0].copy()
        with self._lock:
            self._audio_chunks.append(audio_data)

    def _get_and_clear_chunks(self):
        with self._lock:
            chunks = self._audio_chunks.copy()
            self._audio_chunks.clear()
        if chunks:
            return np.concatenate(chunks)
        return None

    async def start(self):
        self._load_whisper()
        self._running = True

        dev_info = sd.query_devices(self.device, 'input') if self.device is not None else sd.query_devices(kind='input')
        log.info("Using audio device: %s (index=%s, sr=%.0f)",
                 dev_info['name'], dev_info.get('index', self.device), dev_info['default_samplerate'])

        stream = sd.InputStream(
            samplerate=SEND_SAMPLE_RATE,
            channels=AUDIO_CHANNELS,
            dtype="int16",
            device=self.device,
            callback=self._audio_callback,
            blocksize=int(SEND_SAMPLE_RATE * 0.1),
        )

        with stream:
            log.info("Audio stream started (rate=%d, channels=%d)", SEND_SAMPLE_RATE, AUDIO_CHANNELS)
            while self._running:
                await asyncio.sleep(TRANSCRIPTION_CHUNK_SECONDS)
                audio_data = self._get_and_clear_chunks()
                if audio_data is not None and len(audio_data) > 0:
                    self._chunk_count += 1
                    rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
                    peak = np.max(np.abs(audio_data))
                    log.debug(
                        "Audio chunk #%d: samples=%d, duration=%.1fs, rms=%.0f, peak=%d (of 32768)",
                        self._chunk_count, len(audio_data),
                        len(audio_data) / SEND_SAMPLE_RATE,
                        rms, peak,
                    )
                    if peak < 100:
                        log.debug("Audio level very low (peak=%d) — mic may not be capturing", peak)
                    await self._transcribe(audio_data)
                else:
                    log.debug("No audio data received in chunk window")

    async def _transcribe(self, audio_data):
        try:
            audio_float = audio_data.astype(np.float32) / 32768.0
            max_val = np.max(np.abs(audio_float))
            log.debug("Whisper input: max_amplitude=%.4f, duration=%.1fs",
                       max_val, len(audio_float) / SEND_SAMPLE_RATE)

            segments, info = await asyncio.to_thread(
                self._whisper.transcribe,
                audio_float,
                beam_size=1,
                language="en",
                vad_filter=False,
            )
            segments = list(segments)
            log.debug("Whisper returned %d segments (lang=%s, prob=%.2f)",
                       len(segments), info.language, info.language_probability)

            for segment in segments:
                text = segment.text.strip()
                if text:
                    self.transcript_buffer.add(text)
                    log.info("TRANSCRIPT: '%s'", text)

            if not segments:
                log.debug("No speech detected in chunk")

        except Exception as e:
            log.error("Transcription error: %s", e, exc_info=True)

    def get_raw_audio_for_streaming(self):
        return self._get_and_clear_chunks()

    def stop(self):
        self._running = False
        log.info("AudioCapture stopped")
