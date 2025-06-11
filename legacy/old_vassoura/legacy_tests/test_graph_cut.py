"""test_graph_cut.py – Robust test‑suite for the `graph_cut` heuristic.

The aim is to exercise `vassoura.heuristics.graph_cut` under a variety
of real‑world‑like scenarios:

* Large row count (≈20 k) to stress performance and memory handling.
* Mixture of highly correlated and independent **numeric** variables.
* Highly correlated **categorical** variables (via shared generating
  mechanism) plus independent ones.
* Presence of missing values and unbalanced category distributions.
* Different correlation thresholds and correlation methods.
* Validation that `keep_cols` is honoured.
* Validation of metadata and artefacts returned.

These tests follow the same spirit (and level of rigour) as the
`test_vif.py` suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vassoura.heuristics import graph_cut


# ---------------------------------------------------------------------
# Helpers – dataset generators
# ---------------------------------------------------------------------
def _mixed_dataset(n: int = 20_000, seed: int = 0) -> pd.DataFrame:
    """Create a hefty DataFrame with mixed types & controlled correlations."""
    rng = np.random.default_rng(seed)

    # ---------------- Numeric block ----------------
    x1 = rng.normal(size=n)
    # Strong linear relation with x1 (ρ ≈ 0.97)
    x2 = x1 * 0.97 + rng.normal(scale=0.03, size=n)
    x3 = rng.normal(size=n)                  # weak relations
    x4 = rng.normal(size=n)
    num_df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4})

    # ---------------- Categorical block ----------------
    base = np.array(["A", "B", "C", "D", "E"])
    # Skewed distribution – acts like dominant categories
    cat1 = rng.choice(base, size=n, p=[0.4, 0.3, 0.15, 0.10, 0.05])

    # cat2 copies cat1 85 % of the time → very high association
    mask = rng.random(n) < 0.85
    cat2 = np.where(mask, cat1, rng.choice(base, size=n))

    # cat3 independent categorical
    cat3 = rng.choice(base, size=n)

    # Inject ≈5 % missing values in cat1 / cat3
    miss_idx = rng.choice(n, size=int(0.05 * n), replace=False)
    cat1 = cat1.astype(object)
    cat3 = cat3.astype(object)
    cat1[miss_idx] = None
    cat3[miss_idx] = None

    cat_df = pd.DataFrame({"cat1": cat1, "cat2": cat2, "cat3": cat3})

    # Binary, roughly balanced, target – triggers WOE encoding path
    target = rng.integers(0, 2, size=n)

    df = pd.concat([num_df, cat_df], axis=1)
    df["target"] = target
    return df


def _numeric_low_corr_dataset(n: int = 5_000, seed: int = 1) -> pd.DataFrame:
    """Purely numeric, deliberately low‑correlation dataset (control case)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        rng.normal(size=(n, 6)),
        columns=[f"n{i}" for i in range(6)],
    )
    df["target"] = rng.integers(0, 2, size=n)
    return df


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
def test_graph_cut_numeric_high_corr():
    """x1 ↔ x2 are strongly correlated → one of them must be removed."""
    df = _mixed_dataset()
    res = graph_cut(df, target_col="target", corr_threshold=0.9)
    removed = set(res["removed"])
    assert removed.intersection({"x1", "x2"}), "Expected at least one of (x1, x2) to be dropped"


def test_graph_cut_categorical_high_corr():
    """cat1 ↔ cat2 share 85 % values, so at least one should go."""
    df = _mixed_dataset()
    res = graph_cut(df, target_col="target", corr_threshold=0.8)
    removed = set(res["removed"])
    assert removed.intersection({"cat1", "cat2"}), "cat1/cat2 should trigger removal under high corr"


def test_graph_cut_threshold_effect():
    """Lower threshold must remove **at least** as many variables as higher threshold."""
    df = _mixed_dataset()
    res_low = graph_cut(df, target_col="target", corr_threshold=0.7)
    res_high = graph_cut(df, target_col="target", corr_threshold=0.9)
    assert len(res_high["removed"]) <= len(res_low["removed"])


def test_graph_cut_keep_cols_respected():
    """Columns in *keep_cols* must stay even if highly correlated."""
    df = _mixed_dataset()
    initial = graph_cut(df, target_col="target", corr_threshold=0.8)
    must_keep = initial["removed"]  # everything it *wanted* to remove

    # Re‑run, but explicitly protect those columns
    res = graph_cut(
        df,
        target_col="target",
        corr_threshold=0.8,
        keep_cols=must_keep,
    )
    assert not set(res["removed"]).intersection(must_keep), "keep_cols were not honoured"


def test_graph_cut_no_high_corr_returns_empty():
    """Dataset with no |corr| above threshold → nothing removed."""
    df = _numeric_low_corr_dataset()
    res = graph_cut(df, target_col="target", corr_threshold=0.99)  # effectively impossible
    assert res["removed"] == []


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_graph_cut_supports_multiple_methods(method):
    """graph_cut should operate with both Pearson & Spearman correlation."""
    df = _mixed_dataset()
    res = graph_cut(df, target_col="target", corr_threshold=0.85, method=method)
    # Basic sanity checks
    assert isinstance(res["removed"], list)
    assert "corr_threshold" in res["meta"] and res["meta"]["method"] == method


def test_graph_cut_returns_valid_artefacts():
    """`artefacts` should be either *None* or a NetworkX Graph subset."""
    import networkx as nx

    df = _mixed_dataset()
    res = graph_cut(df, target_col="target", corr_threshold=0.85)
    artefacts = res["artefacts"]
    assert artefacts is None or isinstance(artefacts, nx.Graph)
