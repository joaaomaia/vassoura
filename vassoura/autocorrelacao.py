from __future__ import annotations

"""Vassoura – Autocorrelação em Painel
===================================

Ferramentas para calcular autocorrelação (ACF) em datasets de séries
 temporais em painel – típicos de modelagem PD Behavior, onde cada
 contrato (``id_col``) possui um histórico mensal (``time_col`` em formato
 YYYYMM).

Funções públicas
----------------
* ``compute_panel_acf`` – ACF por contrato agregada (média, mediana ou
  ponderada pelo inverso do comprimento)
* ``plot_panel_acf``   – gráfico de barras horizontais com rótulos de
  valor (duas casas decimais)
"""
import logging
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt

LOGGER = logging.getLogger("vassoura")

__all__ = ["compute_panel_acf", "plot_panel_acf"]


def _make_period_index(s: pd.Series) -> pd.PeriodIndex:
    """Converte coluna YYYYMM (int/str) em PeriodIndex mensal."""
    try:
        return pd.PeriodIndex(s.astype(str).str[:6], freq="M")
    except Exception as err:
        raise ValueError("time_col precisa estar no formato YYYYMM (e.g. 202403)") from err


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def compute_panel_acf(
    df: pd.DataFrame,
    value_col: str,
    time_col: str,
    id_col: str,
    nlags: int = 12,
    min_periods: int = 12,
    agg_method: str = "mean",  # ← esse precisa estar aqui
    verbose: bool = False
) -> pd.DataFrame:
    
    """Calcula ACF por contrato e agrega.

    Parâmetros
    ----------
    df, value_col, time_col, id_col
        Colunas do DataFrame.
    nlags : int
        Número máximo de lags (exclui lag 0). 12 = 1 ano para dados
        mensais.
    min_periods : int
        Comprimento mínimo de histórico para considerar o contrato.
    agg_method : {"mean", "median", "weighted"}
        Como agregar a ACF entre contratos. "weighted" usa o inverso do
        comprimento da série como peso.

    Retorna
    -------
    DataFrame com colunas ``lag``, ``acf`` e ``n_contracts``.
    """
    # Garantir ordenação temporal no DataFrame global
    df_sorted = df[[id_col, time_col, value_col]].dropna().copy()
    df_sorted[time_col] = _make_period_index(df_sorted[time_col])
    df_sorted = df_sorted.sort_values([id_col, time_col])

    # Containers
    lag_vals: Dict[int, List[float]] = {lag: [] for lag in range(1, nlags + 1)}
    lag_wts: Dict[int, List[float]] = {lag: [] for lag in range(1, nlags + 1)}

    # Loop por contrato
    for cid, grp in df_sorted.groupby(id_col, sort=False):
        if len(grp) < min_periods:
            continue

        # Reindexa a série do contrato para ter lags mensais contínuos
        idx_full = pd.period_range(grp[time_col].min(), grp[time_col].max(), freq="M")
        series = grp.set_index(time_col)[value_col].reindex(idx_full)

        # Se após reindex perder muitos pontos (<min_periods) skip
        if series.count() < min_periods:
            continue

        # Calcula ACF até nlags (statsmodels devolve lag0 … nlags)
        acf_vals = acf(series.fillna(series.mean()), nlags=nlags, fft=True)
        w = 1 / len(series) if len(series) else 0
        for lag in range(1, nlags + 1):
            lag_vals[lag].append(acf_vals[lag])
            lag_wts[lag].append(w)

    # Agrega
    rows = []
    for lag, values in lag_vals.items():
        if not values:
            continue
        if agg_method  == "median":
            agg_val = float(np.median(values))
        elif agg_method  == "weighted":
            weights = lag_wts[lag]
            agg_val = float(np.average(values, weights=weights))
        else:
            agg_val = float(np.mean(values))
        rows.append({"lag": lag, "acf": agg_val, "n_contracts": len(values)})

    result = pd.DataFrame(rows).sort_values("lag").reset_index(drop=True)
    if result.empty:
        LOGGER.warning("Nenhum contrato com histórico >= %d meses para calcular ACF", min_periods)
    return result


def plot_panel_acf(
    panel_acf: pd.DataFrame,
    *,
    title: str | None = None,
    conf_level: float = 0.95,
) -> plt.Axes:
    """Plota gráfico de barras horizontais com rótulos de valor.

    Adiciona linhas de confiança aproximadas ±z_{conf}/sqrt(n)."""
    if panel_acf.empty:
        raise ValueError("panel_acf DataFrame está vazio")

    n = panel_acf["n_contracts"].max()
    z = 1.96 if abs(conf_level - 0.95) < 1e-3 else 1.64  # approx 90%/95%
    conf = z / np.sqrt(n) if n else 0

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(panel_acf) + 1))
    sns.barplot(data=panel_acf, y="lag", x="acf", orient="h", ax=ax)

    # Linhas de confiança
    ax.axvline(conf, color="red", linestyle="--", linewidth=0.8)
    ax.axvline(-conf, color="red", linestyle="--", linewidth=0.8)

    # Labels
    for i, row in panel_acf.iterrows():
        ax.text(row["acf"] + 0.02 * np.sign(row["acf"]), i, f"{row['acf']:.2f}", va="center")

    ax.set_xlabel("Autocorrelação agregada")
    ax.set_ylabel("Lag (meses)")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return ax
