from __future__ import annotations

import math
import time
from typing import List

import pandas as pd

from .base_heuristic import BaseHeuristic
from .vif import compute_vif


class CooperativeVIF(BaseHeuristic):
    """Iterative VIF heuristic respecting a time budget."""

    def __init__(self, session) -> None:
        self.session = session

    def run(
        self,
        df: pd.DataFrame,
        *,
        budget_sec: float | None = None,
        chunk_size: int = 50,
    ) -> List[str]:
        start = time.perf_counter()
        thr = self.session.params.get("vif", 10.0)
        work = df.copy()
        removed: List[str] = []
        while True:
            if budget_sec is not None and time.perf_counter() - start >= budget_sec:
                break
            vif_df = compute_vif(
                work,
                target_col=self.session.target_col,
                include_target=False,
                engine=self.session.engine,
                verbose=self.session.verbose,
            )
            worst = vif_df[vif_df["vif"] > thr]
            if worst.empty:
                break
            step_limit = (
                1
                if self.session.vif_n_steps == 1
                else max(1, math.ceil(len(worst) / self.session.vif_n_steps))
            )
            for _, row in worst.sort_values("vif", ascending=False).iterrows():
                col = row["variable"]
                if col not in self.session.keep_cols and col in work.columns:
                    removed.append(col)
                    work = work.drop(columns=[col])
                    step_limit -= 1
                    if step_limit == 0:
                        break
                if budget_sec is not None and time.perf_counter() - start >= budget_sec:
                    break
        return removed
