from __future__ import annotations

import concurrent.futures
from typing import Any, Callable, Optional


class TimeoutExecutor:
    """Utility to run callables with a time limit."""

    def __init__(self, timeout_sec: Optional[float] = None) -> None:
        self.timeout_sec = timeout_sec

    def run(self, func: Callable[[], Any]) -> Any:
        """Execute ``func`` respecting ``timeout_sec``.

        Raises
        ------
        TimeoutError
            If the call does not finish within the allotted time.
        Any exception raised by ``func`` is propagated as-is.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(func)
            try:
                return future.result(timeout=self.timeout_sec)
            except concurrent.futures.TimeoutError as exc:  # pragma: no cover - rare
                future.cancel()
                raise TimeoutError(str(exc))

