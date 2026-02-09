import os
import logging

LOG_LEVEL = logging.DEBUG if os.environ.get("JARVIS_DEBUG") else logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("opentelemetry").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)

# API Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Model names
LIVE_API_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
BACKGROUND_MODEL = "gemini-3-flash-preview"

# Audio settings
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
AUDIO_CHANNELS = 1
AUDIO_CHUNK_SIZE = 1024
AUDIO_FORMAT_WIDTH = 2  # 16-bit = 2 bytes

# Transcription settings
WHISPER_MODEL = "base"
TRANSCRIPTION_CHUNK_SECONDS = 3

# Buffer settings
TRANSCRIPT_BUFFER_MINUTES = 5
SCREEN_BUFFER_MAX_FRAMES = 10
SCREEN_CAPTURE_FPS = 1
SCREEN_MAX_WIDTH = 1024
SCREEN_JPEG_QUALITY = 50

# Wake word settings
WAKE_WORD = "jarvis"
WAKE_WORD_DEBOUNCE_SECONDS = 5

# Session settings
SILENCE_TIMEOUT_SECONDS = 30

# Langfuse
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "http://localhost:3001")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")

# System instruction for Live API
SYSTEM_INSTRUCTION = """You are Jarvis, an intelligent ambient AI assistant. You have been passively observing the user's screen and listening to their audio. When activated, you already have context about what they're working on.

Key behaviors:
- Be concise and helpful
- Reference what you've observed on screen or heard in audio when relevant
- If the user asks you to do something complex (research, analysis, coding, writing), delegate it using the start_background_task function
- When a background task completes, announce the results naturally
- Be conversational and natural in your voice responses"""
