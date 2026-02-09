import asyncio
import logging
import time

from google import genai
from google.genai import types

from jarvis.config import (
    GEMINI_API_KEY,
    LIVE_API_MODEL,
    SEND_SAMPLE_RATE,
    SYSTEM_INSTRUCTION,
)
from jarvis.layer3.tools import get_function_declarations
from jarvis.utils.observe import trace_span, flush

log = logging.getLogger("jarvis.live")


class LiveSession:
    """Manages a Gemini Live API WebSocket session."""

    def __init__(self, on_audio_response=None, on_function_call=None):
        self.on_audio_response = on_audio_response
        self.on_function_call = on_function_call
        self._session = None
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._active = False
        self._last_activity = time.time()
        self._pending_tasks = 0

    def _build_config(self):
        tools = get_function_declarations()
        config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": SYSTEM_INSTRUCTION,
            "tools": tools,
            "input_audio_transcription": {},
            "output_audio_transcription": {},
        }
        log.debug("Live API config: model=%s, tools=%d declarations",
                   LIVE_API_MODEL, len(tools))
        return config

    async def connect(self):
        config = self._build_config()
        self._active = True
        log.info("Connecting to Gemini Live API (model=%s)...", LIVE_API_MODEL)

        try:
            async with self._client.aio.live.connect(
                model=LIVE_API_MODEL, config=config
            ) as session:
                self._session = session
                log.info("Connected to Gemini Live API")
                await self._receive_loop()
        except Exception as e:
            log.error("Live API connection error: %s", e, exc_info=True)
        finally:
            self._active = False
            flush()

    async def _receive_loop(self):
        self._last_activity = time.time()
        response_count = 0

        try:
            while self._active:
                turn = self._session.receive()
                async for response in turn:
                    self._last_activity = time.time()
                    response_count += 1

                    if response.server_content and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if part.inline_data and isinstance(
                                part.inline_data.data, bytes
                            ):
                                log.debug("Received audio chunk: %d bytes", len(part.inline_data.data))
                                if self.on_audio_response:
                                    await self.on_audio_response(part.inline_data.data)

                    if (
                        response.server_content
                        and response.server_content.output_transcription
                    ):
                        text = response.server_content.output_transcription.text
                        if text:
                            log.info("[Jarvis says] %s", text)

                    if (
                        response.server_content
                        and response.server_content.input_transcription
                    ):
                        text = response.server_content.input_transcription.text
                        if text:
                            log.info("[User says] %s", text)

                    if response.tool_call:
                        log.info("Tool call received")
                        await self._handle_tool_call(response.tool_call)

                    if (
                        response.server_content
                        and response.server_content.interrupted
                    ):
                        log.info("Generation interrupted by user")

        except Exception as e:
            log.error("Receive loop error: %s", e, exc_info=True)
        finally:
            log.info("Receive loop ended after %d responses", response_count)
            self._active = False

    async def _handle_tool_call(self, tool_call):
        for fc in tool_call.function_calls:
            log.info("Function call: %s(%s)", fc.name, fc.args)
            if self.on_function_call:
                asyncio.create_task(self._execute_and_respond(fc))

    async def _execute_and_respond(self, fc):
        self._pending_tasks += 1
        try:
            function_response = types.FunctionResponse(
                id=fc.id, name=fc.name,
                response={"result": "Task started. I will announce the result when it's ready."}
            )
            await self._session.send_tool_response(
                function_responses=[function_response]
            )
            log.info("Sent immediate ack for %s, running task in background...", fc.name)

            result = await self.on_function_call(fc.name, fc.args)
            if self._session and self._active:
                summary = result[:1000] if result else "Task completed."
                await self.inject_context(
                    f"[Background task completed. Announce these results to the user naturally]: {summary}"
                )
                self._last_activity = time.time()
                log.info("Injected result for %s", fc.name)
        except Exception as e:
            log.error("Tool execution error for %s: %s", fc.name, e)
        finally:
            self._pending_tasks -= 1

    async def send_audio(self, audio_data):
        if self._session and self._active:
            await self._session.send_realtime_input(
                audio=types.Blob(
                    data=audio_data,
                    mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}",
                )
            )

    async def send_image(self, image_data):
        if self._session and self._active:
            log.debug("Sending screen frame: %d bytes", len(image_data))
            await self._session.send_realtime_input(
                video=types.Blob(
                    data=image_data,
                    mime_type="image/jpeg",
                )
            )

    async def send_text(self, text):
        if self._session and self._active:
            log.debug("Sending text: %s", text[:100])
            await self._session.send_client_content(
                turns={"role": "user", "parts": [{"text": text}]},
                turn_complete=True,
            )

    async def inject_context(self, text):
        if self._session and self._active:
            log.debug("Injecting context: %d chars", len(text))
            await self._session.send_client_content(
                turns=[{"role": "user", "parts": [{"text": text}]}],
                turn_complete=False,
            )

    async def inject_model_message(self, text):
        if self._session and self._active:
            log.info("Injecting model message: %s", text[:200])
            await self._session.send_client_content(
                turns=[{"role": "model", "parts": [{"text": text}]}],
                turn_complete=True,
            )

    def is_active(self):
        return self._active

    def has_pending_tasks(self):
        return self._pending_tasks > 0

    def time_since_last_activity(self):
        return time.time() - self._last_activity

    async def close(self):
        self._active = False
        self._session = None
        log.info("Session closed")
        flush()
