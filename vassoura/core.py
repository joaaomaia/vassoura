"""core.py – High‑level object‑oriented API for Vassoura.

This module introduces the `VassouraSession` class which orchestrates
correlation, multicollinearity and feature‑selection heuristics in an
incremental, stateful way.

Key goals
---------
* Avoid recomputation when the user tweaks heuristics or thresholds;
* Keep a full audit trail of removed variables and intermediate
  artefacts (corr matrices, VIF tables, ACF, etc.);
* Offer a one‑line `generate_report()` that reflects the current state.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from .correlacao import compute_corr_matrix
from .vif import compute_vif
from .utils import suggest_corr_method
from .relatorio import generate_report

DEFAULT_HEURISTICS = ["corr", "vif"]  # order matters


class VassouraSession:
    """Stateful cleaning session.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataset – *never* mutated in‑place.
    target_col : str | None
        Target column (excluded from cleaning steps).
    keep_cols : list[str] | None
        Protected columns that shall never be removed.
    heuristics : list[str]
        Ordered list of heuristics to apply. Supported so far:
        ``"corr"``, ``"vif"``, ``"iv"``, ``"importance"``, ``"graph_cut"``.
    thresholds : dict[str, float]
        Per‑heuristic thresholds (e.g. ``{"corr": 0.9, "vif": 10}``).
    **kwargs
        Additional kwargs forwarded to underlying helper functions.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        target_col: str | None = None,
        keep_cols: Optional[List[str]] = None,
        heuristics: Optional[List[str]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> None:
        self.df_original = df.copy()
        self.df_current = df.copy()
        self.target_col = target_col
        self.keep_cols = keep_cols or []
        self.heuristics = heuristics or DEFAULT_HEURISTICS.copy()
        self.thresholds = thresholds or {"corr": 0.9, "vif": 10.0}
        self.params = kwargs

        # Internal caches
        self._corr_matrix: Optional[pd.DataFrame] = None
        self._vif_df: Optional[pd.DataFrame] = None
        self._history: List[Dict[str, Any]] = []  # each entry = snapshot

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def run(self, *, recompute: bool = False) -> pd.DataFrame:
        """Executes all heuristics in the specified order.

        Set ``recompute=True`` to ignore caches and start from scratch.
        """
        if recompute:
            self.reset()

        for h in self.heuristics:
            if h == "corr":
                self._apply_corr()
            elif h == "vif":
                self._apply_vif()
            # Future heuristics placeholders
            # elif h == "iv": ...
            # elif h == "importance": ...
            # elif h == "graph_cut": ...
            else:
                raise ValueError(f"Heuristic '{h}' not implemented")

        return self.df_current

    def remove_additional(self, columns: List[str]) -> None:
        """Force‑drops columns manually after initial cleaning."""
        self._drop(columns, reason="manual")

    def generate_report(self, path: str | Path = "vassoura_report.html") -> str:
        """Generates a report that reflects the *current* session state."""
        return generate_report(
            self.df_current,
            target_col=self.target_col,
            keep_cols=self.keep_cols,
            corr_threshold=self.thresholds.get("corr", 0.9),
            vif_threshold=self.thresholds.get("vif", 10.0),
            **self.params,
            output_path=path,
        )

    def reset(self) -> None:
        """Restores the session to its initial pristine state."""
        self.df_current = self.df_original.copy()
        self._corr_matrix = None
        self._vif_df = None
        self._history.clear()

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    def _apply_corr(self) -> None:
        """Removes highly‑correlated pairs according to the threshold."""
        if self._corr_matrix is None:
            self._corr_matrix = compute_corr_matrix(
                self.df_current,
                method=self.params.get("corr_method", "auto"),
                target_col=self.target_col,
                include_target=False,
                verbose=self.params.get("verbose", True),
            )

        thr = self.thresholds["corr"]
        # Only examine upper triangle to avoid duplicates
        mask = (self._corr_matrix.abs() > thr) & (~self._corr_matrix.eye(len(self._corr_matrix), dtype=bool))
        pairs = mask.stack().loc[lambda s: s]
        for var1, var2 in pairs.index:
            if var1 not in self.df_current.columns or var2 not in self.df_current.columns:
                continue
            drop_var = self._choose_var_to_drop(var1, var2)
            self._drop([drop_var], reason=f"corr>{thr}")

    def _apply_vif(self) -> None:
        thr = self.thresholds["vif"]
        while True:
            self._vif_df = compute_vif(
                self.df_current,
                target_col=self.target_col,
                include_target=False,
                verbose=self.params.get("verbose", True),
            )
            worst = self._vif_df[self._vif_df["vif"] > thr]
            if worst.empty:
                break
            worst_var = worst.sort_values("vif", ascending=False).iloc[0]["variable"]
            self._drop([worst_var], reason=f"vif>{thr}")

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _choose_var_to_drop(self, a: str, b: str) -> str:
        """Decides which variable to drop in a correlated pair.

        Simple heuristic: keep protected columns; otherwise drop the one
        with higher median absolute correlation against the rest.
        """
        if a in self.keep_cols and b in self.keep_cols:
            return b  # arbitrary fallback
        if a in self.keep_cols:
            return b
        if b in self.keep_cols:
            return a
        med_a = self._corr_matrix[a].abs().median()
        med_b = self._corr_matrix[b].abs().median()
        return a if med_a >= med_b else b

    def _drop(self, cols: List[str], reason: str) -> None:
        self.df_current.drop(columns=cols, errors="ignore", inplace=True)
        self._history.append({"cols": cols, "reason": reason})

    # ------------------------------------------------------------------ #
    # Introspection                                                      #
    # ------------------------------------------------------------------ #
    @property
    def history(self) -> List[Dict[str, Any]]:
        """Full audit trail of what was removed and why."""
        return self._history.copy()

    @property
    def dropped(self) -> List[str]:
        return [c for step in self._history for c in step["cols"]]
