"""Configure loguru sinks for stdout and per-run file logging.

Every script in the pipeline calls :func:`configure_logging` once at
start-up. The function clears default loguru handlers and attaches two
sinks:

* stdout at ``INFO`` and above, formatted with colors.
* ``<run_dir>/run.log`` at ``DEBUG`` and above, plain-text.

Produces:
    A path to the per-run log file, which the caller can record in the
    run manifest.

Depends on:
    loguru.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_STDOUT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{name}:{function}:{line} | {message}"
)


def configure_logging(
    run_dir: Path | str,
    *,
    stdout_level: str = "INFO",
    file_level: str = "DEBUG",
) -> Path:
    """Attach loguru sinks for the current run.

    The function creates ``run_dir`` if it does not exist and writes the
    log file at ``<run_dir>/run.log``. Calling the function more than
    once replaces previously installed sinks, which is safe.

    Args:
        run_dir: The directory that holds run artifacts.
        stdout_level: Minimum log level emitted to stdout.
        file_level: Minimum log level written to the file sink.

    Returns:
        The path of the run log file.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    logger.remove()
    logger.add(
        sys.stdout,
        level=stdout_level,
        format=_STDOUT_FORMAT,
        enqueue=False,
        backtrace=False,
        diagnose=False,
    )
    logger.add(
        log_path,
        level=file_level,
        format=_FILE_FORMAT,
        enqueue=False,
        backtrace=True,
        diagnose=False,
    )
    return log_path
