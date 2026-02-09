import logging

from jarvis.utils.buffer import TranscriptBuffer, FrameBuffer

log = logging.getLogger("jarvis.context")


class ContextInjector:
    """Injects buffered context into a Live API session."""

    def __init__(
        self, transcript_buffer: TranscriptBuffer, frame_buffer: FrameBuffer
    ):
        self.transcript_buffer = transcript_buffer
        self.frame_buffer = frame_buffer

    async def inject(self, session):
        transcript = self.transcript_buffer.get_text()
        if transcript:
            context_msg = (
                f"[Context from recent audio - last few minutes of "
                f"conversation/audio you've been passively observing]:\n"
                f"{transcript}"
            )
            log.info("Injecting transcript (%d chars): '%s'",
                      len(transcript), transcript[:200])
            await session.inject_context(context_msg)
        else:
            log.info("No transcript to inject")

        frames = self.frame_buffer.get_frames()
        if frames:
            to_send = frames[-3:]
            log.info("Injecting %d screen frames (of %d buffered)", len(to_send), len(frames))
            for i, (ts, frame_data, _) in enumerate(to_send):
                log.debug("Frame %d: %d bytes", i, len(frame_data))
                await session.send_image(frame_data)
        else:
            log.info("No screen frames to inject")
