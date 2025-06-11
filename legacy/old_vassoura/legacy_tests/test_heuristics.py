"""
Test-suite robusta para heurísticas do módulo `heuristics.py`.

Cenários exercitados
--------------------
* Numéricas altamente correlacionadas  ➜ graph_cut, partial_corr_cluster
* Categóricas (com e sem target binário) ➜ iv, ks_separation
* Colunas constantes ou dominantes     ➜ variance
* Variáveis com drift temporal (PSI)   ➜ psi_stability
* Variável de baixa separação KS       ➜ ks_separation
* Ausência de dependências externas    ➜ skips elegantes

Requer: pytest, pandas, numpy (+ networkx opcional)
"""

from __future__ import annotations

import importlib.util
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

# ------------------------------------------------------------------ #
# Imports das heurísticas em teste
# ------------------------------------------------------------------ #
from heuristics import (
    graph_cut,
    iv,
    ks_separation,
    psi_stability,
    variance,
)

# networkx é opcional – só importa se existir
NETWORKX_AVAILABLE = importlib.util.find_spec("networkx") is not None
if NETWORKX_AVAILABLE:
    from heuristics import partial_corr_cluster


# ------------------------------------------------------------------ #
# Fixture principal de dados
# ------------------------------------------------------------------ #
@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    """Gera um DataFrame com variedade de situações para todas as heurísticas."""
    n = 500
    rng = np.random.default_rng(42)

    # Datas para PSI (duas janelas)
    date_24 = pd.date_range("2024-01-01", periods=n // 2, freq="D")
    date_25 = pd.date_range("2025-01-01", periods=n // 2, freq="D")
    dates = date_24.append(date_25)

    # Variáveis numéricas
    x1 = rng.normal(size=n)
    x2 = x1 * 0.95 + rng.normal(scale=0.05, size=n)  # alta correlação
    x3 = rng.normal(size=n)                           # pouco correlacionado

    # Variável de baixa IV (aleatória)
    low_iv = rng.random(size=n)

    # Variável com baixa separação KS (uniforme)
    low_ks = rng.random(size=n)

    # Constante / baixa variância
    const_num = np.ones(n)

    # Variável com drift temporal (PSI alto)
    drift_var = np.concatenate(
        [rng.normal(loc=0, scale=1, size=n // 2),
         rng.normal(loc=3, scale=1, size=n // 2)]
    )

    # Categóricas
    cat_base = np.array(["A", "B", "C"])
    cat1 = rng.choice(cat_base, size=n)
    # cat2 fortemente correlacionada com cat1
    cat2 = np.where(rng.random(size=n) < 0.9, cat1, rng.choice(cat_base, size=n))
    # Dominância (categoria quase única)
    cat_dom = np.where(rng.random(size=n) < 0.98, "A", "B")

    # Target binário (para IV, KS, etc.)
    target = (x1 + rng.normal(scale=0.3, size=n) > 0).astype(int)

    df = pd.DataFrame(
        {
            "date": dates.astype(str),
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "low_iv": low_iv,
            "low_ks": low_ks,
            "const_num": const_num,
            "drift_var": drift_var,
            "cat1": cat1,
            "cat2": cat2,
            "cat_dom": cat_dom,
            "target": target,
        }
    )

    # Injeta alguns valores ausentes
    na_idx = rng.choice(n, size=int(0.1 * n), replace=False)
    df.loc[na_idx, "x3"] = np.nan
    df.loc[na_idx, "cat1"] = np.nan
    return df


# ------------------------------------------------------------------ #
# IV – Information Value
# ------------------------------------------------------------------ #
def test_iv_low_iv_removed(synthetic_df: pd.DataFrame):
    res = iv(synthetic_df, target_col="target", threshold=0.10)  # limiar alto
    assert "low_iv" in res["removed"]
    assert "x1" not in res["removed"]  # variável realmente informativa


# ------------------------------------------------------------------ #
# Variance – baixa variância e dominância
# ------------------------------------------------------------------ #
def test_variance_filters_const_and_dominant(synthetic_df: pd.DataFrame):
    res = variance(
        synthetic_df,
        var_threshold=1e-6,
        dom_threshold=0.95,
        min_nonnull=10,
    )
    assert "const_num" in res["removed"]
    assert "cat_dom" in res["removed"]
    # Pelo menos uma variável legítima deve permanecer
    assert "x1" not in res["removed"]


# ------------------------------------------------------------------ #
# Graph-cut – correlação alta elimina ao menos uma feature
# ------------------------------------------------------------------ #
@pytest.mark.skipif(
    not NETWORKX_AVAILABLE, reason="networkx não instalado – graph_cut não testado"
)
def test_graph_cut_removes_one_of_highly_correlated_pair(synthetic_df: pd.DataFrame):
    res = graph_cut(
        synthetic_df,
        target_col="target",
        corr_threshold=0.8,
        method="pearson",
    )
    removed = set(res["removed"])
    # Ao menos uma das duas altamente correlacionadas sai
    assert bool(removed & {"x1", "x2"})


# ------------------------------------------------------------------ #
# KS Separation – filtra variáveis com baixa discriminação
# ------------------------------------------------------------------ #
def test_ks_separation_filters_low_separation(synthetic_df: pd.DataFrame):
    res = ks_separation(
        synthetic_df,
        target_col="target",
        ks_thr=0.05,
        n_bins=10,
    )
    assert "low_ks" in res["removed"]
    assert "x1" not in res["removed"]


# ------------------------------------------------------------------ #
# PSI Stability – identifica drift temporal
# ------------------------------------------------------------------ #
def test_psi_stability_detects_drift(synthetic_df: pd.DataFrame):
    res = psi_stability(
        synthetic_df,
        date_col="date",
        window=("2024", "2025"),
        psi_thr=0.25,
        bins=10,
    )
    assert "drift_var" in res["removed"]
    # Coluna sem drift deve ficar
    assert "x1" not in res["removed"]


# ------------------------------------------------------------------ #
# Partial correlation cluster – deve retirar algo em alta colinearidade
# ------------------------------------------------------------------ #
@pytest.mark.skipif(
    not NETWORKX_AVAILABLE, reason="networkx não instalado – partial_corr_cluster não testado"
)
def test_partial_corr_cluster_removes_in_correlated_block(synthetic_df: pd.DataFrame):
    res = partial_corr_cluster(
        synthetic_df[["x1", "x2", "x3"]],  # apenas numéricas
        corr_thr=0.6,
    )
    assert res["removed"], "Nenhuma coluna removida apesar de correlação forte"


# ------------------------------------------------------------------ #
# Robustez geral das assinaturas
# ------------------------------------------------------------------ #
def test_all_artefacts_present(synthetic_df: pd.DataFrame):
    """Garantir que todas as heurísticas retornem dict com chaves mínimas."""
    funcs: Tuple = (iv, variance, ks_separation)
    if NETWORKX_AVAILABLE:
        funcs += (graph_cut, partial_corr_cluster)
    for fn in funcs:
        res = fn(
            synthetic_df,
            **({"target_col": "target"} if "target_col" in fn.__code__.co_varnames else {}),
        )
        assert "removed" in res and "artefacts" in res and "meta" in res
