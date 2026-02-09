import asyncio
import logging
import sys
import time

from jarvis.utils.buffer import TranscriptBuffer, FrameBuffer
from jarvis.layer1.audio_capture import AudioCapture
from jarvis.layer1.screen_capture import ScreenCapture
from jarvis.layer1.wake_word import WakeWordDetector
from jarvis.layer2.live_session import LiveSession
from jarvis.layer2.context_inject import ContextInjector
from jarvis.layer2.audio_playback import AudioPlayback
from jarvis.layer3.task_executor import TaskExecutor
from jarvis.config import SILENCE_TIMEOUT_SECONDS
from jarvis.utils.observe import trace_span, flush

log = logging.getLogger("jarvis.main")



class Jarvis:
    """Main orchestrator for the Jarvis ambient AI copilot."""

    def __init__(self, audio_device=None):
        self.transcript_buffer = TranscriptBuffer()
        self.frame_buffer = FrameBuffer()

        self.audio_capture = AudioCapture(
            self.transcript_buffer, device=audio_device
        )
        self.screen_capture = ScreenCapture(self.frame_buffer)
        self.wake_detector = WakeWordDetector(self.transcript_buffer)
        self.context_injector = ContextInjector(
            self.transcript_buffer, self.frame_buffer
        )
        self.audio_playback = AudioPlayback()
        self.task_executor = TaskExecutor()

        self._running = False
        self._in_session = False
        self._live_session = None

    async def run(self):
        """Main run loop."""
        self._running = True
        log.info("=" * 50)
        log.info("  JARVIS - Ambient AI Copilot")
        log.info("  Say 'Hey Jarvis' to activate")
        log.info("=" * 50)

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self.audio_capture.start())
                tg.create_task(self.screen_capture.start())
                tg.create_task(self._wake_word_loop())
        except* KeyboardInterrupt:
            log.info("Shutting down...")
        except* Exception as eg:
            for e in eg.exceptions:
                log.error("Fatal error: %s", e, exc_info=True)
        finally:
            self._shutdown()

    async def _wake_word_loop(self):
        """Poll for wake word and activate session."""
        while self._running:
            if not self._in_session and self.wake_detector.check():
                log.info("Wake word detected! Activating session...")
                await self._start_session()
                flush()
            await asyncio.sleep(0.5)

    async def _start_session(self):
        """Start a Live API session (Layer 2)."""
        self._in_session = True

        self._live_session = LiveSession(
            on_audio_response=self._handle_audio_response,
            on_function_call=self._handle_function_call,
        )

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._live_session.connect())
                tg.create_task(self._inject_and_stream())
                tg.create_task(self.audio_playback.start())
                tg.create_task(self._silence_monitor())
        except* Exception as eg:
            for e in eg.exceptions:
                log.error("Session error: %s", e, exc_info=True)
        finally:
            self._in_session = False
            self.audio_playback.stop()
            self.wake_detector.suppress()
            log.info("Returned to passive mode")

    async def _inject_and_stream(self):
        """Inject context then stream live audio + screen to the session."""
        await asyncio.sleep(0.5)

        log.info("Injecting buffered context...")
        await self.context_injector.inject(self._live_session)

        log.info("Streaming live audio + screen to session...")
        frame_count = 0
        audio_sends = 0
        last_frame_time = 0

        while self._in_session and self._live_session.is_active():
            try:
                audio_data = self.audio_capture.get_raw_audio_for_streaming()
                if audio_data is not None and len(audio_data) > 0:
                    pcm_bytes = audio_data.astype("int16").tobytes()
                    await self._live_session.send_audio(pcm_bytes)
                    audio_sends += 1
                    if audio_sends % 50 == 0:
                        log.debug("Streamed %d audio chunks to Live API", audio_sends)

                now = time.time()
                if now - last_frame_time >= 1.0:
                    latest = self.frame_buffer.get_latest()
                    if latest:
                        _, frame_data, _ = latest
                        await self._live_session.send_image(frame_data)
                        frame_count += 1
                        last_frame_time = now
                        if frame_count % 10 == 0:
                            log.debug("Streamed %d screen frames to Live API", frame_count)

            except Exception as e:
                log.error("Stream error: %s", e)

            await asyncio.sleep(0.1)

        log.info("Streaming ended (audio_chunks=%d, frames=%d)", audio_sends, frame_count)

    async def _handle_audio_response(self, audio_data):
        await self.audio_playback.enqueue(audio_data)

    async def _handle_function_call(self, name, args):
        if name == "start_background_task":
            task_desc = args.get("task_description", "")
            context = args.get("context", "")

            if not context:
                context = self.transcript_buffer.get_text()

            log.info("Delegating to Gemini 3 Pro: %s", task_desc[:150])

            result = await self.task_executor.execute(task_desc, context)
            return result
        else:
            log.warning("Unknown function call: %s", name)
            return f"Unknown function: {name}"

    async def _silence_monitor(self):
        await asyncio.sleep(10)
        while self._in_session and self._live_session and self._live_session.is_active():
            idle = self._live_session.time_since_last_activity()
            if idle > SILENCE_TIMEOUT_SECONDS and not self._live_session.has_pending_tasks():
                log.info("No activity for %.0fs, closing session...", idle)
                await self._live_session.close()
                break
            await asyncio.sleep(5)

    def _shutdown(self):
        self._running = False
        self._in_session = False
        self.audio_capture.stop()
        self.screen_capture.stop()
        self.audio_playback.stop()
        flush()
        log.info("Goodbye!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Jarvis - Ambient AI Copilot")
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device index (use --list-devices to see available)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger("jarvis").setLevel(logging.DEBUG)

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        sys.exit(0)

    jarvis = Jarvis(audio_device=args.device)

    try:
        asyncio.run(jarvis.run())
    except KeyboardInterrupt:
        log.info("Interrupted by user")


if __name__ == "__main__":
    main()
