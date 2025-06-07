from __future__ import annotations

"""Vassoura — Utilities
======================

Este módulo concentra funções auxiliares utilizadas em todo o pacote
``vassoura``. Aqui ficam as ferramentas de inspeção de tipos de colunas,
decisão de métodos de correlação, definição de tamanhos dinâmicos de
figuras, além de helpers para identificação e remoção automática de
colunas de identificação (IDs).

Principais funções expostas
---------------------------
search_dtypes       – classifica colunas em numéricas/categóricas
suggest_corr_method – sugere método de correlação ideal
figsize_from_matrix – ajusta tamanho de figura para heat‑maps

Todas as funções são pensadas para fornecer *logs* detalhados via o
módulo ``logging``. Para visualizar, no *script* principal configure
algo como::

    import logging, vassoura
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s | %(message)s"
    )
"""

import logging
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "search_dtypes",
    "suggest_corr_method",
    "figsize_from_matrix",
    "criar_dataset_pd_behavior",
]

# ---------------------------------------------------------------------------
# Configuração de *logger* local
# ---------------------------------------------------------------------------
LOGGER = logging.getLogger("vassoura")
if not LOGGER.handlers:
    # Evita duplicar handlers quando importado diversas vezes
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    LOGGER.addHandler(_handler)
    LOGGER.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Funções internas auxiliares (não exportadas)
# ---------------------------------------------------------------------------


def _is_id_column(col_name: str, col_data: pd.Series, id_patterns: List[str]) -> bool:
    """Heurística simples para identificar colunas que provavelmente são IDs."""
    col_lower = col_name.lower()
    name_match = any(pat.lower() in col_lower for pat in id_patterns)
    unique_ratio = col_data.nunique(dropna=True) / max(len(col_data), 1)
    high_uniqueness = unique_ratio > 0.95
    return name_match or high_uniqueness


def _remove_id_columns(
    num_cols: List[str],
    cat_cols: List[str],
    id_patterns: List[str],
) -> Tuple[List[str], List[str]]:
    """Remove colunas identificadas como ID das listas *in‑place*."""

    def _filter(col_list: List[str]) -> List[str]:
        return [c for c in col_list if not any(p in c.lower() for p in id_patterns)]

    return _filter(num_cols), _filter(cat_cols)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def search_dtypes(
    df: pd.DataFrame,
    *,
    target_col: str | None = None,
    limite_categorico: int = 50,
    force_categorical: Optional[List[str]] = None,
    remove_ids: bool = False,
    id_patterns: Optional[List[str]] = None,
    date_col: Optional[List[str]] = None,
    verbose: bool = True,
    verbose_types: bool = False,
) -> Tuple[List[str], List[str]]:
    """Classifica as colunas de *df* em numéricas e categóricas.

    Parameters
    ----------
    df : pandas.DataFrame
        Conjunto de dados completo.
    target_col : str | None, default None
        Nome da coluna *target* que deve ser ignorada na classificação.
    limite_categorico : int, default 50
        Se a coluna do tipo *object* tiver número de valores únicos menor
        ou igual a este limite, ela é considerada categórica.
    force_categorical : list[str] | None
        Colunas explicitamente forçadas como categóricas, independente de
        tipo ou cardinalidade.
    remove_ids : bool, default False
        Se ``True``, tenta remover colunas identificadas como IDs.
    id_patterns : list[str] | None
        Lista de *substrings* indicativas de ID (``["_id", "id_", "codigo"]``).
    date_col : list[str] | None
        Colunas que devem ser tratadas como ``datetime`` de forma explícita.
    verbose : bool, default True
        Se ``True``, imprime apenas logs resumidos.
    verbose_types : bool, default False
        Se ``True``, exibe logs detalhados para cada coluna analisada.

    Returns
    -------
    (list[str], list[str])
        Tupla *(num_cols, cat_cols)* com as colunas classificadas.
    """
    # --------------------------- Validações básicas ------------------------
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df precisa ser um pandas.DataFrame")
    if df.empty:
        raise ValueError("DataFrame não pode estar vazio")

    if target_col is not None and target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' não encontrado em df")

    force_categorical = force_categorical or []
    id_patterns = id_patterns or ["_id", "id_", "codigo", "key"]
    date_col = date_col or []

    # Remove target para não ser classificada
    df_work = (
        df.drop(columns=[target_col], errors="ignore") if target_col else df.copy()
    )

    num_cols: List[str] = []
    cat_cols: List[str] = []

    for col in df_work.columns:
        s = df_work[col]
        try:
            if col in date_col:
                pd.to_datetime(s, errors="coerce")
                if verbose_types:
                    LOGGER.info("%s -> datetime (explicit)", col)
                continue

            if col in force_categorical:
                cat_cols.append(col)
                if verbose_types:
                    LOGGER.info("%s -> categórica (forçada)", col)
                continue

            if pd.api.types.is_numeric_dtype(s):
                num_cols.append(col)
                if verbose_types:
                    LOGGER.info("%s -> numérica", col)
                continue

            if pd.api.types.is_bool_dtype(s):  # trata bool como categórica
                cat_cols.append(col)
                if verbose_types:
                    LOGGER.info("%s -> categórica (bool)", col)
                continue

            if pd.api.types.is_datetime64_any_dtype(s):  # ignora datas
                if verbose_types:
                    LOGGER.info("%s ignorada (datetime)", col)
                continue

            # Colunas object / string
            try_dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
            if try_dt.notna().mean() > 0.8 and try_dt.notna().any():
                if verbose_types:
                    LOGGER.info("%s -> datetime (auto)", col)
                continue

            unique_cnt = s.nunique(dropna=True)
            if unique_cnt <= limite_categorico:
                cat_cols.append(col)
                if verbose_types:
                    LOGGER.info("%s -> categórica (%d categorias)", col, unique_cnt)
            else:
                if verbose_types:
                    LOGGER.info("%s ignorada (muitas categorias: %d)", col, unique_cnt)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"Falha ao processar coluna {col}: {exc}")

    if remove_ids:
        num_cols, cat_cols = _remove_id_columns(num_cols, cat_cols, id_patterns)

    if verbose:
        LOGGER.info(
            "Tipos detectados: %d numéricas, %d categóricas",
            len(num_cols),
            len(cat_cols),
        )

    return num_cols, cat_cols


# ---------------------------------------------------------------------------
# Utilidades complementares
# ---------------------------------------------------------------------------


def suggest_corr_method(num_cols: List[str], cat_cols: List[str]) -> str:
    """Sugere método de correlação com base na presença de tipos.

    * Se houver apenas numéricas → "pearson"
    * Se houver categóricas + numéricas → "spearman"
    * Se houver apenas categóricas → "cramer" (Cramér‑V)
    """
    if num_cols and not cat_cols:
        return "pearson"
    if num_cols and cat_cols:
        return "spearman"
    if cat_cols and not num_cols:
        return "cramer"
    return "pearson"  # fallback seguro


def figsize_from_matrix(
    n_features: int, base: float = 0.4, *, min_size: int = 6, max_size: int = 20
) -> Tuple[int, int]:
    """Calcula *figsize* adequado para heat‑maps baseado em ``n_features``.

    O tamanho cresce linearmente com o número de variáveis, ficando dentro
    dos limites ``min_size`` e ``max_size``.
    """
    size = np.clip(int(n_features * base), min_size, max_size)
    return size, size


def criar_dataset_pd_behavior(
    n_clientes: int = 1000,
    max_anos: int = 5,
    n_features: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Gera dataset sintético para modelagem de PD *behavior*.

    Os contratos começam em meses aleatórios entre 2018-01 e 2022-12 e
    possuem durações distintas. Cada linha representa um mês de histórico
    de um contrato.

    Parameters
    ----------
    n_clientes : int
        Quantidade de contratos (IDs) distintos.
    max_anos : int
        Duração máxima possível em anos para cada contrato.
    n_features : int
        Número de colunas preditoras geradas.
    seed : int
        Semente para reprodutibilidade.

    Returns
    -------
    pandas.DataFrame
        DataFrame contendo ``NroContrato``, ``AnoMesReferencia``, features e
        coluna ``ever90m12`` simulada.
    """

    rng = np.random.default_rng(seed)
    start_dates = pd.date_range("2018-01-01", "2022-12-01", freq="MS")

    rows = []
    for i in range(n_clientes):
        start = rng.choice(start_dates)
        dur = rng.integers(12, max_anos * 12 + 1)
        meses = pd.date_range(start, periods=dur, freq="MS")
        df_cli = pd.DataFrame(
            {
                "NroContrato": i + 1,
                "AnoMesReferencia": meses.strftime("%Y%m").astype(int),
            }
        )
        rows.append(df_cli)

    df = pd.concat(rows, ignore_index=True)

    time_num = (df["AnoMesReferencia"] // 100) * 12 + (df["AnoMesReferencia"] % 100) - 1

    for j in range(1, n_features + 1):
        base = rng.normal(0, 1, size=n_clientes)
        amplitude = rng.uniform(0.5, 1.5)
        phase = rng.uniform(0, 2 * np.pi)
        noise = rng.normal(0, 0.5, size=len(df))
        df[f"feature_{j:02d}"] = (
            base[df["NroContrato"] - 1]
            + amplitude * np.sin(time_num / 6 + phase)
            + noise
        )

    logits = (
        0.3 * df["feature_01"]
        + 0.2 * df["feature_02"]
        - 0.4 * df["feature_03"]
        + rng.normal(0, 1, size=len(df))
    )
    probs = 1 / (1 + np.exp(-logits))
    df["ever90m12"] = (probs > 0.8).astype(int)

    return df
