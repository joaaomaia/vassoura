from __future__ import annotations

import numpy as np
import pandas as pd

from vassoura.autocorrelacao import compute_panel_acf, plot_panel_acf
from vassoura.analisador import analisar_autocorrelacao


def _make_panel_df(n_contracts: int = 10, months: int = 18) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_contracts):
        start = 202001
        for m in range(months):
            rows.append(
                {
                    "cid": i,
                    "ym": start + m,
                    "val": rng.normal() + i * 0.1,
                }
            )
    return pd.DataFrame(rows)


def test_compute_panel_acf_basic() -> None:
    df = _make_panel_df()
    panel = compute_panel_acf(
        df,
        value_col="val",
        time_col="ym",
        id_col="cid",
        nlags=6,
        min_periods=6,
    )
    assert not panel.empty
    assert panel["lag"].max() <= 6


def test_analisar_autocorrelacao_levels() -> None:
    df = _make_panel_df()
    panel = compute_panel_acf(
        df,
        value_col="val",
        time_col="ym",
        id_col="cid",
        nlags=6,
        min_periods=6,
    )
    result = analisar_autocorrelacao(panel, "val", verbose=False)
    assert {"feature", "acf_max", "acf_lag_max", "nivel", "recomendacao"} <= result.keys()


def test_plot_panel_acf() -> None:
    df = _make_panel_df()
    panel = compute_panel_acf(
        df,
        value_col="val",
        time_col="ym",
        id_col="cid",
        nlags=3,
        min_periods=3,
    )
    ax = plot_panel_acf(panel, title="test")
    assert ax.get_title() == "test"

