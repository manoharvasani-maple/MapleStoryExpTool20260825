import logging
from logging.handlers import RotatingFileHandler
import os
import platform
from pathlib import Path
import sys
import threading


LOGGER_NAME = "maplestory_exp_tool"
LOG_FILENAME = "error.log"
DEFAULT_MAX_BYTES = 1_000_000
DEFAULT_BACKUP_COUNT = 3


def get_app_data_dir() -> Path:
    base_dir = os.getenv("APPDATA") or os.path.expanduser("~")
    return Path(base_dir) / "ExpTracker"


def get_log_path() -> Path:
    return get_app_data_dir() / LOG_FILENAME


def get_logger(component: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{component}")


def configure_diagnostics(
        app_version: str,
        log_path: Path | str | None = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
        backup_count: int = DEFAULT_BACKUP_COUNT,
        install_hooks: bool = True,
) -> Path:
    path = Path(log_path) if log_path is not None else get_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    app_logger = logging.getLogger(LOGGER_NAME)
    app_logger.setLevel(logging.INFO)
    app_logger.propagate = False

    # Reconfiguration is useful in tests and protects against accidentally
    # registering duplicate handlers if startup code is executed twice.
    for handler in list(app_logger.handlers):
        if getattr(handler, "_exp_tracker_handler", False):
            app_logger.removeHandler(handler)
            handler.close()

    handler = RotatingFileHandler(
        path,
        maxBytes=max(1, int(max_bytes)),
        backupCount=max(0, int(backup_count)),
        encoding="utf-8",
    )
    handler._exp_tracker_handler = True
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(threadName)s | %(name)s | %(message)s"
    ))
    app_logger.addHandler(handler)

    if install_hooks:
        _install_exception_hooks()

    app_logger.info(
        "Application started version=%s frozen=%s executable=%s os=%s",
        app_version,
        bool(getattr(sys, "frozen", False)),
        sys.executable,
        platform.platform(),
    )
    app_logger.info("Diagnostic log path=%s", path)
    return path


def shutdown_diagnostics() -> None:
    app_logger = logging.getLogger(LOGGER_NAME)
    for handler in list(app_logger.handlers):
        if getattr(handler, "_exp_tracker_handler", False):
            handler.flush()
            handler.close()
            app_logger.removeHandler(handler)


def _install_exception_hooks() -> None:
    if getattr(sys.excepthook, "_exp_tracker_hook", False):
        return

    original_sys_hook = sys.excepthook

    def sys_hook(exc_type, exc_value, exc_traceback):
        get_logger("unhandled").critical(
            "Unhandled main-thread exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        original_sys_hook(exc_type, exc_value, exc_traceback)

    sys_hook._exp_tracker_hook = True
    sys.excepthook = sys_hook

    if hasattr(threading, "excepthook") and not getattr(
            threading.excepthook, "_exp_tracker_hook", False
    ):
        original_thread_hook = threading.excepthook

        def thread_hook(args):
            get_logger("unhandled").critical(
                "Unhandled worker-thread exception thread=%s",
                getattr(args.thread, "name", "unknown"),
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
            original_thread_hook(args)

        thread_hook._exp_tracker_hook = True
        threading.excepthook = thread_hook

