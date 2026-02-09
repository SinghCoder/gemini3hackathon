# 🤖 Jarvis — Always-On Ambient AI Copilot

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Gemini API](https://img.shields.io/badge/Google-Gemini%20API-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**An always-on, screen-aware AI copilot that sees your screen, hears your audio, stays silent until called, and delegates complex tasks to a background agent — powered by Gemini 2.5 Flash (real-time perception) and Gemini 3 Flash (deep reasoning).**

*Built for the Gemini 3 Hackathon (Google DeepMind × Devpost)*

</div>

---

## How It Works

Jarvis runs three independent layers that work together:

| Mode | What Happens | API Cost |
|------|-------------|----------|
| **Passive** | Audio → Whisper → transcript buffer. Screen → JPEG → frame buffer. Zero API calls. | $0 |
| **Active** | Wake word → open Live API session → inject context → stream live audio + screen → voice responses | Free tier |
| **Task Delegation** | User asks complex task → Live API function call → Gemini 3 Flash runs autonomously → result announced via voice | Free tier |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  LAYER 1: LOCAL  (Always Running · Zero API Cost)            │
│                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐  │
│  │ System Audio   │  │ Screen Capture │  │  Wake Word    │  │
│  │ → faster-      │  │ → mss + Pillow │  │  Detection    │  │
│  │   whisper      │  │ → 1fps JPEG    │  │  ("Jarvis")   │  │
│  │   (local)      │  │   frames       │  │  string match │  │
│  └───────┬────────┘  └───────┬────────┘  └──────┬────────┘  │
│          │                   │                   │           │
│          ▼                   ▼                   │           │
│  ┌──────────────────────────────────┐            │           │
│  │     Rolling Context Buffer      │            │           │
│  │  (last 5 min transcript +       │            │           │
│  │   last 10 screen frames)        │            │           │
│  └──────────────────────────────────┘            │           │
│                                                  │           │
│  No API calls. No tokens burned. Local compute.  │           │
└──────────────────────────────────────────────────┼───────────┘
                                                   │
                               Wake word detected! │
                                                   ▼
┌──────────────────────────────────────────────────────────────┐
│  LAYER 2: GEMINI 2.5 FLASH LIVE API  (On-Demand Session)     │
│                                                              │
│  • Opened ONLY when wake word fires                          │
│  • Receives: context buffer + live audio + live screen       │
│  • Provides: real-time voice conversation                    │
│  • Streams screen at 1fps for visual understanding           │
│  • Closes after 30s silence → back to passive                │
│                                                              │
│  Function call: start_background_task(description)           │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          │ Function call triggered
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  LAYER 3: GEMINI 3 FLASH  (Background Task Execution)        │
│                                                              │
│  • Receives task description + relevant context              │
│  • Runs autonomously with code_execution tool                │
│  • Result injected back into Layer 2 session                 │
│  • Layer 2 announces result to user via voice                │
└──────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Package manager | [uv](https://github.com/astral-sh/uv) |
| Gemini SDK | `google-genai` |
| Local transcription | `faster-whisper` (runs on CPU, no API cost) |
| Screen capture | `mss` + `Pillow` |
| Audio capture | `sounddevice` |
| Audio playback | `pyaudio` |
| Audio processing | `numpy` |
| Observability | `langfuse` (optional) |

### Models Used

| Purpose | Model |
|---------|-------|
| Real-time perception (Live API) | `gemini-2.5-flash-native-audio-preview-12-2025` |
| Background task execution | `gemini-3-flash-preview` |

---

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- macOS: [BlackHole](https://existential.audio/blackhole/) for system audio capture

### Installation

```bash
# Clone
git clone https://github.com/user/jarvis.git
cd jarvis

# Install with uv
uv sync

# macOS: Install BlackHole for system audio capture
# Download from https://existential.audio/blackhole/

# Set API key
export GEMINI_API_KEY="your-key-here"

# Run
uv run jarvis

# Options
uv run jarvis --list-devices  # Show audio devices
uv run jarvis --device 1      # Use specific audio device
uv run jarvis --debug          # Enable verbose logging
```

---

## File Structure

```
jarvis/
├── main.py                 # Entry point, orchestrates all layers
├── config.py               # Settings, model names, buffer sizes
├── layer1/
│   ├── audio_capture.py    # System audio + local Whisper transcription
│   ├── screen_capture.py   # Screen capture + frame buffer
│   └── wake_word.py        # Wake word detection
├── layer2/
│   ├── live_session.py     # Gemini Live API session management
│   ├── context_inject.py   # Buffer → Live API context injection
│   └── audio_playback.py   # Play Gemini audio responses
├── layer3/
│   ├── task_executor.py    # Gemini 3 Flash background tasks
│   └── tools.py            # Function declarations
└── utils/
    ├── buffer.py           # Rolling buffer implementations
    └── observe.py          # Langfuse observability (optional)
```

---

## Data Flows

### Passive Mode (always running)

```
System Audio → faster-whisper (local) → transcript text → rolling buffer
Screen       → mss + Pillow (local)   → JPEG frames    → rolling buffer
Transcript   → string match "jarvis"  → no match       → continue buffering
```

Zero API calls. Runs indefinitely on local compute.

### Active Mode (wake word triggered)

```
1. "Jarvis" detected in transcript
2. Open Gemini Live API WebSocket session
3. Inject buffered context (transcript + screen frames)
4. Stream live audio + screen at 1fps
5. Receive and play voice responses
6. 30s silence → close session → return to passive
```

### Task Delegation

```
1. User: "Hey Jarvis, research competitor pricing and make a doc"
2. Live API: "On it, I'll let you know when it's ready."
3. Live API emits function_call → start_background_task(...)
4. Gemini 3 Flash runs with code_execution tool
5. Result injected back into Live API session
6. Jarvis speaks: "The competitor analysis is ready. I found..."
```

---

## Cost

| Layer | Cost |
|-------|------|
| Layer 1 — Local Whisper + screen capture | **$0** (local compute) |
| Layer 2 — Gemini 2.5 Flash Live API | Free tier (Google AI Studio) |
| Layer 3 — Gemini 3 Flash API | Free tier (Google AI Studio) |
| **Total for hackathon** | **$0** |

---

## Hackathon Tracks

| Track | How Jarvis Fits |
|-------|----------------|
| 🧠 **Marathon Agent** | Maintains continuity across long sessions via rolling context buffer. Background Gemini 3 agent runs autonomously on complex tasks. |
| 👨‍🏫 **Real-Time Teacher** | Uses Gemini Live API to synthesize live video + audio for adaptive, contextual assistance. |
| ☯️ **Vibe Engineering** | When asked to fix code, the Gemini 3 agent can write, test, and verify code autonomously via code execution. |

---

## License

MIT