from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Sequence, List

import pandas as pd


def chunk_iter(seq: Sequence[str], size: int) -> Iterator[List[str]]:
    """Yield successive chunks from *seq* of length *size*."""
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


class BaseHeuristic(ABC):
    """Abstract base class for cooperative heuristics."""

    @abstractmethod
    def run(
        self,
        df: pd.DataFrame,
        *,
        budget_sec: float | None = None,
        chunk_size: int = 50,
    ) -> List[str]:
        """Return columns to drop. Implementation must respect *budget_sec*."""
        raise NotImplementedError
