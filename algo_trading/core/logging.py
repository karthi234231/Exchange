"""
logging_setup.py — structlog + stdlib logging configuration.

Call configure_logging() once at process startup, before any module logs anything.

Design:
  - structlog for all application logging (structured key=value context)
  - stdlib logging for third-party libraries (broker API, requests, etc.)
  - Two sinks: rotating file (JSON lines) + stdout (human-readable dev format)
  - Log level from settings.log_level
  - Secret redaction filter applied after configuration
"""
from __future__ import annotations

import logging
import logging.handlers
import shutil
import sys
from pathlib import Path

import structlog

from algo_trading.config import settings

_DISK_SPACE_MIN_BYTES: int = 2 * 1024 * 1024 * 1024  # 2 GB


def configure_logging() -> None:
    """
    Configure structlog + stdlib logging for the trading system.
    Idempotent — safe to call multiple times (subsequent calls are no-ops).
    """
    if getattr(configure_logging, "_done", False):
        return
    configure_logging._done = True  # type: ignore[attr-defined]

    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "algo_trading.log"

    level = getattr(logging, settings.log_level, logging.INFO)

    # ── Shared processors (applied to every log event) ─────────────────────
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.ExceptionRenderer(),
    ]

    # ── File handler — JSON lines (one event per line, machine-parseable) ──
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=50 * 1024 * 1024,  # 50 MB per file
        backupCount=10,
        encoding="utf-8",
    )
    file_handler.setLevel(level)

    # ── Stdout handler — human-readable dev format ──────────────────────────
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)

    # ── stdlib root logger ──────────────────────────────────────────────────
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    # Clear any default handlers added before configure_logging() was called
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stdout_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("websocket").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.INFO)

    # Apply per-module log level overrides from settings.log_levels
    for module, module_level in getattr(settings, "log_levels", {}).items():
        logging.getLogger(module).setLevel(module_level)

    # ── structlog configuration ─────────────────────────────────────────────
    structlog.configure(
        processors=shared_processors
        + [
            # Format for stdlib sink — allows structlog → stdlib → handlers
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )

    # Formatter for the stdlib handlers (renders structlog events)
    dev_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())],
        foreign_pre_chain=shared_processors,
    )
    json_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[structlog.processors.JSONRenderer()],
        foreign_pre_chain=shared_processors,
    )

    stdout_handler.setFormatter(dev_formatter)
    file_handler.setFormatter(json_formatter)

    # Install secret redaction filter on all handlers
    settings.install_log_secret_filter()

    # ── Disk space guard ────────────────────────────────────────────────────
    try:
        disk = shutil.disk_usage("/")
        if disk.free < _DISK_SPACE_MIN_BYTES:
            free_gb = disk.free / (1024 ** 3)
            _disk_log = structlog.get_logger(__name__)
            _disk_log.error(
                "disk_space_low",
                free_gb=round(free_gb, 2),
                min_required_gb=2,
                msg="Low disk space — log rotation and data writes may fail",
            )
            # Attempt to send alert if Telegram is configured
            try:
                from algo_trading.alerts.telegram_bot import TelegramAlerter  # noqa: PLC0415
                alerter = TelegramAlerter()
                alerter.send_sync(
                    f"LOW DISK SPACE: only {free_gb:.1f} GB free on /. "
                    "Log rotation may fail — free space immediately."
                )
            except Exception:  # noqa: BLE001
                pass  # Alert failure is non-fatal; log message is sufficient
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning("disk_space_check_failed: %s", exc)

    structlog.get_logger(__name__).info(
        "logging configured",
        level=settings.log_level,
        log_file=str(log_file),
        mode=settings.trading_mode.value,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog BoundLogger bound to the given name."""
    return structlog.get_logger(name)
