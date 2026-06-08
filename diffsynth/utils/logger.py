from __future__ import annotations

import logging
import os
import platform
import sys
from importlib.metadata import distributions
from pathlib import Path

import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)
logger.propagate = False

_LOGGING_SRCFILE = logging._srcfile
_THIS_SRCFILE = os.path.normcase(__file__)


def _caller_class_name() -> str:
    # Avoid materializing frame.f_locals unless the first argument is `self`/`cls`.
    # `co_varnames` is a static attribute on the code object, reading it is free.
    frame = sys._getframe()
    while frame is not None:
        co_file = frame.f_code.co_filename
        if co_file != _LOGGING_SRCFILE and co_file != _THIS_SRCFILE:
            varnames = frame.f_code.co_varnames
            if varnames:
                first = varnames[0]
                if first == "self":
                    return type(frame.f_locals["self"]).__name__
                if first == "cls":
                    return frame.f_locals["cls"].__name__
            return ""
        frame = frame.f_back
    return ""


def _current_node_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() // dist.get_world_size()
    # fallback to env vars set by torchrun/launch
    return int(os.environ.get("NODE_RANK", 0))


def _current_process_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    # fallback to env vars set by torchrun/launch
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))


class _ExtraFieldsFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self.cwd = os.getcwd()

    def filter(
        self,
        record: logging.LogRecord,
    ) -> bool:
        record.rank = _current_process_rank()
        try:
            record.relpath = os.path.relpath(record.pathname, self.cwd)
        except ValueError:
            # different drive on Windows, fall back to file name
            record.relpath = record.filename
        class_name = _caller_class_name()
        record.classname = class_name
        record.qualname = (
            f"{class_name}.{record.funcName}" if class_name else record.funcName
        )
        return True


def _install_excepthooks() -> None:
    """
    Route uncaught exceptions through the logging system so the traceback
    lands in the per-rank log file (and stderr handler), not just on stderr
    where torchrun may swallow it.
    """
    def _log_uncaught(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            # Keep Ctrl-C behavior: don't spam logs, fall back to default.
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_tb),
        )

    sys.excepthook = _log_uncaught

    # Python 3.8+: threading exceptions
    import threading
    def _log_thread_exc(args):
        if issubclass(args.exc_type, SystemExit):
            return
        logger.critical(
            f"Uncaught exception in thread {args.thread.name}",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
    threading.excepthook = _log_thread_exc

    # Unraisable exceptions (e.g. errors in __del__, weakref callbacks)
    def _log_unraisable(unraisable):
        logger.error(
            f"Unraisable exception: {unraisable.err_msg or ''}",
            exc_info=(unraisable.exc_type, unraisable.exc_value, unraisable.exc_traceback),
        )
    sys.unraisablehook = _log_unraisable


def _configure_logging(
    logfile: Path | str | None = None,
    logging_level: int = logging.INFO,
) -> None:
    """
    Configure the module-level logger without touching the root logger.
    """
    fmt = logging.Formatter(
        "[rank:{rank}] [{levelname}] [{asctime}] [{relpath}:{lineno}] {qualname} : {message}",
        style="{",
    )
    extra_fields_filter = _ExtraFieldsFilter()

    stream_handler = logging.StreamHandler()

    handlers = [
        stream_handler,
    ]
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile, mode="w"))

    for handler in handlers:
        handler.setLevel(logging_level)
        handler.setFormatter(fmt)
        handler.addFilter(extra_fields_filter)

    logger.setLevel(logging_level)
    logger.handlers[:] = handlers

    _install_excepthooks()


def _installed_package_versions() -> str:
    package_versions: dict[str, str] = {}
    for distribution in distributions():
        package_name = distribution.metadata.get("Name")
        if package_name is None:
            continue
        package_versions[package_name] = distribution.version

    return "\n".join(
        f"{package_name}=={package_versions[package_name]}"
        for package_name in sorted(package_versions, key=str.lower)
    )


def log_environment_versions() -> None:
    """
    Log runtime information and all installed Python package versions.
    """
    if not logger.isEnabledFor(logging.INFO):
        return

    cuda_device_summary = "unavailable"
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        current_device = torch.cuda.current_device()
        cuda_device_summary = (
            f"index={current_device}, "
            f"name={torch.cuda.get_device_name(current_device)}, "
            f"capability={torch.cuda.get_device_capability(current_device)}"
        )

    logger.info(
        f"""
[environment versions]
python={sys.version.split()[0]}, executable={sys.executable}
platform={platform.platform()}
torch={torch.__version__}, torch_cuda={torch.version.cuda}, cudnn={torch.backends.cudnn.version()}
cuda_available={cuda_available}, cuda_device={cuda_device_summary}
installed_packages:
{_installed_package_versions()}
"""
    )


def string_to_logging_level(
    level_str: str,
) -> int:
    """
    Convert a string logging level to the corresponding logging module constant.
    """
    level_str = level_str.upper()
    if hasattr(logging, level_str):
        return getattr(logging, level_str)
    raise ValueError(f"Invalid logging level: {level_str}")
