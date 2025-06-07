import pandas as pd
import numpy as np
import os
from vassoura.core import Vassoura
from vassoura.autocorrelacao import compute_panel_acf
from vassoura.analisador import analisar_autocorrelacao

def _make_complex_data(n_contracts=50, months=24):
    rng = np.random.default_rng(42)
    rows = []
    for cid in range(n_contracts):
        start_year = 2020 + rng.integers(0, 2)
        start_month = rng.integers(1, 13)
        for m in range(months):
            year = start_year + (start_month + m - 1) // 12
            month = (start_month + m - 1) % 12 + 1
            ym = year * 100 + month  # formato YYYYMM
            x1 = rng.normal()
            x2 = x1 * 0.9 + rng.normal(scale=0.1)
            x3 = rng.normal()
            target = int((x1 + x3 + rng.normal()) > 0)
            rows.append({
                "Contrato": cid,
                "AnoMes": ym,
                "x1": x1,
                "x2": x2,
                "x3": x3,
                "target": target
            })
    return pd.DataFrame(rows)

def test_vassoura_pipeline_completo(tmp_path):
    df = _make_complex_data()
    
    vs = Vassoura(
        df,
        target_col="target",
        keep_cols=["x1"],
        heuristics=["corr", "vif", "iv"],
        thresholds={"corr": 0.85, "vif": 5, "iv": 0.01, "missing": 0.2},
        verbose=False
    )

    df_clean = vs.run(recompute=True)

    # Verificações do core
    assert "x2" not in df_clean.columns
    assert "x1" in df_clean.columns
    assert vs.dropped  # ao menos uma variável removida
    assert any("corr>" in h["reason"] or "vif>" in h["reason"] for h in vs.history)

    # Geração do relatório
    report_path = tmp_path / "relatorio.html"
    path_gerado = vs.generate_report(path=report_path)
    assert os.path.exists(path_gerado)

    # Autocorrelação da variável x1
    acf_panel = compute_panel_acf(df, value_col="x1", time_col="AnoMes", id_col="Contrato", nlags=6, min_periods=6)
    acf_analysis = analisar_autocorrelacao(acf_panel, feature_name="x1", verbose=False)

    # Verificações da análise de autocorrelação
    assert "acf_max" in acf_analysis
    assert "recomendacao" in acf_analysis
    assert acf_analysis["nivel"] in {"ruido", "leve", "moderada", "alta"}
    assert isinstance(acf_analysis["acf_lag_max"], int)
