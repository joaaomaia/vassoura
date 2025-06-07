from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vassoura.autocorrelacao import compute_panel_acf, plot_panel_acf, _make_period_index
from vassoura.analisador import analisar_autocorrelacao


def _make_panel_df(n_contracts: int = 10, months: int = 18) -> pd.DataFrame:
    """Dataset sintético de painel com datas válidas no formato YYYYMM."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_contracts):
        start = pd.Period("2020-01", freq="M")
        for m in range(months):
            period = start + m
            rows.append(
                {
                    "cid": i,
                    "ym": int(period.strftime("%Y%m")),
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


def test_make_period_index_invalid():
    with pytest.raises(ValueError):
        _make_period_index(pd.Series([202013]))

