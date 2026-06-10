"""
Structured JSON logging configuration.

Every log record emitted by the application contains:
  - timestamp   : ISO-8601 UTC
  - level       : DEBUG / INFO / WARNING / ERROR / CRITICAL
  - logger      : module logger name
  - request_id  : UUID injected by the request_id middleware (or "N/A")
  - message     : log message

The request_id is propagated via Python's contextvars so it is
automatically available to every logger called within the same
async task — no manual passing required.

Usage:
    from app.observability.logging import configure_logging, request_id_var

    configure_logging()            # call once at app startup

    # In middleware:
    token = request_id_var.set(some_id)
    ...
    request_id_var.reset(token)
"""

import logging
import contextvars
from pythonjsonlogger import jsonlogger

# ---------------------------------------------------------------------------
# Context variable — holds the current request's ID for the lifetime of
# each async task.  Defaults to "N/A" for background / startup log lines.
# ---------------------------------------------------------------------------
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="N/A"
)


class _RequestIdFilter(logging.Filter):
    """Inject request_id from the context variable into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging() -> None:
    """
    Configure application-wide structured JSON logging.

    Call this once at application startup (before the first log line).
    Safe to call multiple times — idempotent.
    """
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.addFilter(_RequestIdFilter())

    root = logging.getLogger()
    # Avoid adding duplicate handlers on repeated calls
    if not any(isinstance(h, logging.StreamHandler) and
               isinstance(getattr(h, "formatter", None), jsonlogger.JsonFormatter)
               for h in root.handlers):
        root.handlers.clear()
        root.addHandler(handler)

    root.setLevel(logging.INFO)

    # Suppress noisy third-party loggers
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("passlib").setLevel(logging.WARNING)
