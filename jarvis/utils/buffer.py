import time
import threading
from collections import deque


class TranscriptBuffer:
    """Rolling buffer for transcript text with time-based windowing."""

    def __init__(self, max_minutes=5):
        self._entries = deque()
        self._max_seconds = max_minutes * 60
        self._lock = threading.Lock()

    def add(self, text):
        with self._lock:
            self._entries.append((time.time(), text))
            self._trim()

    def _trim(self):
        cutoff = time.time() - self._max_seconds
        while self._entries and self._entries[0][0] < cutoff:
            self._entries.popleft()

    def get_text(self):
        with self._lock:
            self._trim()
            return " ".join(text for _, text in self._entries)

    def get_recent_text(self, seconds=10):
        with self._lock:
            cutoff = time.time() - seconds
            return " ".join(
                text for ts, text in self._entries if ts >= cutoff
            )

    def clear(self):
        with self._lock:
            self._entries.clear()


class FrameBuffer:
    """Rolling buffer for screen capture frames."""

    def __init__(self, max_frames=10):
        self._frames = deque(maxlen=max_frames)
        self._lock = threading.Lock()

    def add(self, frame_bytes, mime_type="image/jpeg"):
        with self._lock:
            self._frames.append((time.time(), frame_bytes, mime_type))

    def get_frames(self):
        with self._lock:
            return list(self._frames)

    def get_latest(self):
        with self._lock:
            if self._frames:
                return self._frames[-1]
            return None

    def clear(self):
        with self._lock:
            self._frames.clear()
