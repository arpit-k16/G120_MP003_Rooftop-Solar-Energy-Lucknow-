"""Structured JSON logging for the prediction service.

Every log record becomes ONE line of JSON on stdout (never a file), so
`docker logs` / `kubectl logs` can be searched and parsed by log tools:

    {"timestamp": "2026-11-26T16:02:11.512+00:00", "level": "INFO",
     "logger": "mp003-api", "message": "prediction", "event": "prediction",
     "prediction_kwh": 31.42, "latency_ms": 3.1, ...}

Extra fields are passed with `logger.info("prediction", extra={"fields": {...}})`.
No personal details are logged: the request has only panel and weather readings.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Format a log record as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class StdoutHandler(logging.StreamHandler):
    """A StreamHandler that always writes to the *current* sys.stdout."""

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, _value) -> None:  # StreamHandler.__init__ assigns it
        pass


def get_logger(name: str = "mp003-api") -> logging.Logger:
    """Return a logger that writes one JSON line per record to stdout."""
    logger = logging.getLogger(name)
    if not any(isinstance(h, StdoutHandler) for h in logger.handlers):
        handler = StdoutHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    logger.propagate = False
    return logger
