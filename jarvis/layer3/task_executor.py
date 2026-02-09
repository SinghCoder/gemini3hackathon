import asyncio
import logging
import re

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

from jarvis.config import GEMINI_API_KEY, BACKGROUND_MODEL
from jarvis.utils.observe import trace_span, generation_span, flush

log = logging.getLogger("jarvis.task")

MAX_RETRIES = 3


class TaskExecutor:
    """Executes background tasks using Gemini 3 Pro."""

    def __init__(self):
        self._client = genai.Client(api_key=GEMINI_API_KEY)

    async def execute(self, task_description, context=None):
        log.info("Starting background task: %s", task_description[:150])

        prompt_parts = [f"Task: {task_description}"]
        if context:
            prompt_parts.append(f"\nRelevant context:\n{context}")
        prompt_parts.append(
            "\nPlease complete this task thoroughly. "
            "Use Google Search if you need current information. "
            "Use code execution if you need to compute or analyze something. "
            "Provide a clear, concise result summary."
        )
        prompt = "\n".join(prompt_parts)

        tools = [
            types.Tool(code_execution=types.ToolCodeExecution()),
        ]

        config = types.GenerateContentConfig(
            tools=tools,
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self._client.aio.models.generate_content(
                    model=BACKGROUND_MODEL,
                    contents=prompt,
                    config=config,
                )

                result = response.text if response.text else "Task completed but no text output was generated."
                log.info("Task complete: %s", result[:300])
                return result

            except (ClientError, ServerError) as e:
                if e.code in (429, 503) and attempt < MAX_RETRIES:
                    delay = self._parse_retry_delay(str(e), default=30)
                    log.warning("API error %d (attempt %d/%d), retrying in %.0fs...",
                                e.code, attempt, MAX_RETRIES, delay)
                    await asyncio.sleep(delay)
                    continue
                error_msg = f"Task failed: {str(e)}"
                log.error(error_msg, exc_info=True)
                return error_msg

            except Exception as e:
                error_msg = f"Task failed: {str(e)}"
                log.error(error_msg, exc_info=True)
                return error_msg
            finally:
                flush()

        return "Task failed after maximum retries due to rate limiting."

    @staticmethod
    def _parse_retry_delay(error_text, default=60):
        match = re.search(r"retry in ([\d.]+)s", error_text, re.IGNORECASE)
        if match:
            return min(float(match.group(1)) + 5, 120)
        return default
