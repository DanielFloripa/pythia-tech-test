"""
One key=value line per event, stdlib only.

Readable while tailing a terminal, still parseable by a log pipeline, and
JSON later would be a change to the formatter alone. Correlation fields go
in explicitly rather than through contextvars, which do not follow work into
ThreadPoolExecutor threads. Thread name is in the format so pool concurrency
shows up per line.
"""

import json
import logging
import os
from enum import Enum

LOGGER_NAME = "pythia"
_FORMAT = "%(asctime)s %(levelname)-5s [%(threadName)s] %(name)s %(message)s"


def get_logger(module: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{module}")


def kv(event: str, **fields: object) -> str:
    """`event=x k=v ...`, quoting any value that would break the parse."""
    parts = [f"event={event}"]
    for key, value in fields.items():
        if isinstance(value, Enum):
            value = value.value
        text = str(value)
        if not text or any(c in text for c in ' "='):
            text = json.dumps(text)
        parts.append(f"{key}={text}")
    return " ".join(parts)


def configure_logging() -> None:
    """Handler on the `pythia` tree only.

    basicConfig would touch the root logger and duplicate uvicorn's own
    output; propagate=False keeps our lines from being printed twice if
    something else configures the root. Safe to call more than once.
    """
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    logger.propagate = False
