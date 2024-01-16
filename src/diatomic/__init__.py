from functools import wraps
import logging
import time

logger = logging.getLogger(__name__)


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
    logger.
    """

    @wraps(func)
    def log_time_wrapper(*args, **kwargs):
        logger.info(f"Starting Function {func.__name__}...")
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.info(f"Function {func.__name__} took {total_time:.4f} seconds")
        return result

    return log_time_wrapper
