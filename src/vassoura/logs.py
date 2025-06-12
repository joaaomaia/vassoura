import logging
import sys


def get_logger(
    name: str = "vassoura", level: int = logging.INFO
) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "[%(asctime)s] %(levelname)s – %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
