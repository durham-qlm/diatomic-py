from contextvars import ContextVar
from functools import wraps
import logging
import time

logger = logging.getLogger(__name__)
_log_time_depth = ContextVar("diatomic_log_time_depth", default=0)


def _format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.3f} s"

    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} min {seconds:.1f} s"

    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)} h {int(minutes)} min {seconds:.0f} s"


def configure_logging(level=logging.INFO, filename=None):
    logger = logging.getLogger(__name__)
    logger.handlers = []  # remove existing handlers
    if filename:
        handler = logging.FileHandler(filename)
    else:
        handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    # Propagate this logger's level to all descendant loggers in the hierarchy.
    logger.propagate = False


def log_time(func):
    """
    Decorator that times the execution of a function while executing and outputs to the
    logger. Top-level timed calls log at INFO; nested timed calls log at DEBUG to keep
    long calculations readable while still making detailed timing available when
    debugging.
    """

    @wraps(func)
    def log_time_wrapper(*args, **kwargs):
        depth = _log_time_depth.get()
        log = logger.info if depth == 0 else logger.debug

        log(f"Starting {func.__name__}...")
        start_time = time.perf_counter()
        token = _log_time_depth.set(depth + 1)
        try:
            result = func(*args, **kwargs)
        except Exception:
            total_time = time.perf_counter() - start_time
            log(f"Failed {func.__name__} after {_format_duration(total_time)}")
            raise
        finally:
            _log_time_depth.reset(token)

        total_time = time.perf_counter() - start_time
        log(f"Finished {func.__name__}, took: {_format_duration(total_time)}")
        return result

    return log_time_wrapper
