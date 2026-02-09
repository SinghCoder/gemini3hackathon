import logging
import os
from contextlib import contextmanager

log = logging.getLogger("jarvis.observe")

os.environ.setdefault("LANGFUSE_SECRET_KEY", "")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "")
os.environ.setdefault("LANGFUSE_HOST", "http://localhost:3001")

_client = None


def get_langfuse():
    global _client
    if _client is None:
        try:
            from langfuse import Langfuse
            _client = Langfuse()
            log.info("Langfuse initialized (%s)", os.environ.get("LANGFUSE_HOST"))
        except Exception as e:
            log.warning("Langfuse init failed: %s", e)
    return _client


@contextmanager
def trace_span(name, **kwargs):
    """Create a Langfuse trace span using the v3 context manager API."""
    lf = get_langfuse()
    if lf:
        try:
            with lf.start_as_current_observation(
                as_type="span", name=name, **kwargs
            ) as span:
                yield span
        except Exception as e:
            log.debug("Langfuse span error: %s", e)
            yield None
    else:
        yield None


@contextmanager
def generation_span(name, **kwargs):
    """Create a Langfuse generation span."""
    lf = get_langfuse()
    if lf:
        try:
            with lf.start_as_current_observation(
                as_type="generation", name=name, **kwargs
            ) as gen:
                yield gen
        except Exception as e:
            log.debug("Langfuse generation error: %s", e)
            yield None
    else:
        yield None


def log_event(name, **kwargs):
    """Log a simple event to Langfuse."""
    lf = get_langfuse()
    if lf:
        try:
            with lf.start_as_current_observation(
                as_type="span", name=name, **kwargs
            ) as span:
                span.end()
        except Exception:
            pass


def flush():
    lf = get_langfuse()
    if lf:
        try:
            lf.flush()
        except Exception:
            pass
