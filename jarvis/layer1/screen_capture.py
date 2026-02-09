import asyncio
import io
import logging

import mss
from PIL import Image

from jarvis.config import (
    SCREEN_CAPTURE_FPS,
    SCREEN_MAX_WIDTH,
    SCREEN_JPEG_QUALITY,
)
from jarvis.utils.buffer import FrameBuffer

log = logging.getLogger("jarvis.screen")


class ScreenCapture:
    """Captures screen at regular intervals and stores in rolling buffer."""

    def __init__(self, frame_buffer: FrameBuffer):
        self.frame_buffer = frame_buffer
        self._running = False
        self._sct = None
        self._capture_count = 0

    def _capture_frame(self):
        if self._sct is None:
            self._sct = mss.mss()

        monitor = self._sct.monitors[1]
        screenshot = self._sct.grab(monitor)

        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

        width, height = img.size
        if width > SCREEN_MAX_WIDTH:
            ratio = SCREEN_MAX_WIDTH / width
            new_height = int(height * ratio)
            img = img.resize(
                (SCREEN_MAX_WIDTH, new_height), Image.Resampling.LANCZOS
            )

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=SCREEN_JPEG_QUALITY)
        return buffer.getvalue()

    async def start(self):
        self._running = True
        interval = 1.0 / SCREEN_CAPTURE_FPS
        log.info("Starting screen capture at %d fps", SCREEN_CAPTURE_FPS)

        while self._running:
            try:
                frame_data = await asyncio.to_thread(self._capture_frame)
                self.frame_buffer.add(frame_data)
                self._capture_count += 1
                if self._capture_count % 30 == 0:
                    log.debug("Captured %d frames (latest: %d bytes)",
                               self._capture_count, len(frame_data))
            except Exception as e:
                log.error("Screen capture error: %s", e)
            await asyncio.sleep(interval)

    def capture_single_frame(self):
        return self._capture_frame()

    def stop(self):
        self._running = False
        if self._sct:
            self._sct.close()
            self._sct = None
        log.info("ScreenCapture stopped")
