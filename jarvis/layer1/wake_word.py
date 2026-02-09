import logging
import time

from jarvis.config import WAKE_WORD, WAKE_WORD_DEBOUNCE_SECONDS
from jarvis.utils.buffer import TranscriptBuffer

log = logging.getLogger("jarvis.wakeword")


class WakeWordDetector:
    """Detects wake word in transcript buffer with debounce."""

    def __init__(self, transcript_buffer: TranscriptBuffer):
        self.transcript_buffer = transcript_buffer
        self._last_trigger_time = 0
        self._wake_word = WAKE_WORD.lower()
        self._check_count = 0

    def check(self):
        self._check_count += 1
        now = time.time()
        if now - self._last_trigger_time < WAKE_WORD_DEBOUNCE_SECONDS:
            return False

        recent = self.transcript_buffer.get_recent_text(seconds=10)
        if self._check_count % 20 == 0:
            full = self.transcript_buffer.get_text()
            log.debug("Wake word check #%d — recent(10s): '%s' | full buffer: '%s'",
                       self._check_count, recent[:200], full[:200])

        if self._wake_word in recent.lower():
            self._last_trigger_time = now
            log.info("WAKE WORD DETECTED in: '%s'", recent)
            return True

        return False

    def suppress(self):
        """Suppress re-triggering for the debounce period after session ends."""
        self._last_trigger_time = time.time()
