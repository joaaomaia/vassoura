"""test_session.py – Testes unitários básicos para Vassoura."""
import pandas as pd
import numpy as np
import pytest

from vassoura.core import Vassoura

@pytest.fixture
def df_toy():
    np.random.seed(42)
    df = pd.DataFrame({
        "a": np.random.normal(size=100),
        "b": np.random.normal(size=100),
        "c": np.random.normal(size=100),
        "d": np.random.choice([0, 1], size=100),  # target
    })
    df["b"] = df["a"] * 0.9 + np.random.normal(scale=0.1, size=100)  # alta correlação
    df["c"] = df["a"] * 0.5 + df["b"] * 0.5
    return df


def test_corr_removal(df_toy):
    vs = Vassoura(df_toy, target_col="d", heuristics=["corr"], thresholds={"corr": 0.85})
    df_clean = vs.run()
    assert df_clean.shape[1] < df_toy.shape[1], "Não removeu colunas correlacionadas"
    assert any("corr>" in h["reason"] for h in vs.history), "Histórico não registra remoção por correlação"


def test_vif_removal(df_toy):
    vs = Vassoura(df_toy, target_col="d", heuristics=["vif"], thresholds={"vif": 5})
    df_clean = vs.run()
    assert df_clean.shape[1] <= df_toy.shape[1]
    assert isinstance(vs.dropped, list)


def test_iv_removal(df_toy):
    vs = Vassoura(df_toy, target_col="d", heuristics=["iv"], thresholds={"iv": 0.0001})
    df_clean = vs.run()
    assert "iv<" in vs.history[-1]["reason"] or len(vs.dropped) == 0


def test_manual_removal(df_toy):
    vs = Vassoura(df_toy, target_col="d")
    vs.remove_additional(["a"])
    assert "a" not in vs.df_current.columns
    assert any("manual" in h["reason"] for h in vs.history)


def test_reset(df_toy):
    vs = Vassoura(df_toy, target_col="d")
    vs.remove_additional(["a"])
    vs.reset()
    assert "a" in vs.df_current.columns
    assert vs.history == []
