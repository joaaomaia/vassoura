from typing import Dict, Any
import pandas as pd
import numpy as np

__all__ = ["analisar_autocorrelacao"]


def analisar_autocorrelacao(
    panel_acf: pd.DataFrame,
    feature_name: str,
    *,
    threshold_baixo: float = 0.1,
    threshold_moderado: float = 0.3,
    threshold_alto: float = 0.6,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Analisa resultados de autocorrelação agregada (panel_acf) para uma feature,
    determina o lag de maior influência e sugere uma recomendação.

    Parâmetros
    ----------
    panel_acf : pd.DataFrame
        DataFrame com as colunas ['lag', 'acf', 'n_contracts'] (retorno de compute_panel_acf).
    feature_name : str
        Nome da feature que está sendo analisada (aparece nos logs/saída).
    threshold_baixo : float, opcional (default=0.1)
        Até esse valor de |ACF| considera-se que não há sinal relevante (nível “ruído”).
    threshold_moderado : float, opcional (default=0.3)
        Acima de threshold_baixo e abaixo de threshold_moderado considera-se nível “leve”.
    threshold_alto : float, opcional (default=0.6)
        Acima de threshold_moderado e abaixo de threshold_alto considera-se nível “moderado”.
        Acima de threshold_alto => nível “alto”.
    verbose : bool, opcional (default=True)
        Se True, imprime mensagem no console com nível e recomendação.

    Retorna
    -------
    dict
        {
            "feature": feature_name,
            "acf_max": float,         # valor absoluto máximo de acf
            "acf_lag_max": int,       # lag correspondente a acf_max
            "nivel": str,             # 'ruido' | 'leve' | 'moderada' | 'alta'
            "recomendacao": str       # texto recomendando ação
        }
    """

    if panel_acf.empty or "acf" not in panel_acf.columns or "lag" not in panel_acf.columns:
        raise ValueError(
            "panel_acf deve conter as colunas ['lag', 'acf'] e não pode estar vazio."
        )

    # Trabalhar com valores absolutos para determinar intensidade
    acf_abs = panel_acf["acf"].abs()
    idx_max = int(acf_abs.idxmax())
    acf_max = float(acf_abs.iloc[idx_max])
    lag_max = int(panel_acf.loc[idx_max, "lag"])

    # Determinar nível e recomendação
    if acf_max < threshold_baixo:
        nivel = "ruido"
        recomendacao = "Sem sinal relevante de autocorrelação. Não é necessário incluir lags."
    elif acf_max < threshold_moderado:
        nivel = "leve"
        recomendacao = (
            f"ACF leve em lag {lag_max} (≈ {acf_max:.2f}). "
            "Opcionalmente, pode-se incluir esse lag como nova feature."
        )
    elif acf_max < threshold_alto:
        nivel = "moderada"
        recomendacao = (
            f"ACF moderada em lag {lag_max} (≈ {acf_max:.2f}). "
            "Recomenda-se incluir esse lag como feature ou usar média móvel, "
            "controlando multicolinearidade depois."
        )
    else:
        nivel = "alta"
        recomendacao = (
            f"ACF alta em lag {lag_max} (≈ {acf_max:.2f}). "
            "Sugere-se criar múltiplos lags (e.g. lag_{lag_max}, lag_{lag_max*2}) ou "
            "usar modelo de séries temporais para capturar dependência."
        )

    if verbose:
        print(f"\n🔎 Análise de Autocorrelação – {feature_name}")
        print(f"   • maior |ACF| = {acf_max:.2f} no lag {lag_max}  →  Nível: {nivel.upper()}")
        print(f"   • Recomendação: {recomendacao}")

    return {
        "feature": feature_name,
        "acf_max": acf_max,
        "acf_lag_max": lag_max,
        "nivel": nivel,
        "recomendacao": recomendacao,
    }
