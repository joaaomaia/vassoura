import logging
from typing import Optional


def configure_logging(
    level: int = logging.INFO, *, filename: Optional[str] = None
) -> logging.Logger:
    """Configure root ``vassoura`` logger with optional file output.

    Parameters
    ----------
    level : int, optional
        Logging level applied to the root logger, by default ``logging.INFO``.
    filename : str | None, optional
        If provided, logs are also written to this file. Otherwise
        :class:`logging.StreamHandler` is used.

    Returns
    -------
    logging.Logger
        The configured ``vassoura`` logger.
    """
    logger = logging.getLogger("vassoura")

    if filename:
        handler: logging.Handler = logging.FileHandler(filename)
    else:
        handler = logging.StreamHandler()

    handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))

    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
