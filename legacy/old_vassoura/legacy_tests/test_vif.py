import numpy as np
import pandas as pd
import pytest

from vassoura.core import compute_vif

# ---------------------------------------------------------------------
# Helpers de geração de dados
# ---------------------------------------------------------------------
def _numeric_dataset(n=250, seed=0, high_corr=True):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    if high_corr:
        x2 = x1 * 0.95 + rng.normal(scale=0.05, size=n)
    else:
        x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    target = (x1 + rng.normal(size=n)) > 0
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": target.astype(int)})


def _categorical_dataset(all_categorical: bool = True, seed: int = 1) -> pd.DataFrame:
    """
    Gera um DataFrame com diversos cenários para testes de heurísticas:
      - Distribuições balanceadas e desbalanceadas
      - Alta correlação entre cat1 e cat2 (80% iguais, 20% aleatórios)
      - Valores missing em categorias e numéricos
      - Outliers na variável numérica
      - Cardinalidade moderada (5 categorias)
    Parâmetros
    ----------
    all_categorical : bool
        Se False, adiciona também uma coluna numérica (`num1`).
    seed : int
        Semente para reproducibilidade.
    """
    rng = np.random.default_rng(seed)
    n = 10_000

    # Definição das categorias e probabilidades
    base = np.array(["A", "B", "C", "D", "E"])
    probs = np.array([0.3, 0.25, 0.2, 0.15, 0.1])  # soma = 1.0

    # cat1: distribuição não uniforme
    cat1 = rng.choice(base, size=n, p=probs)

    # cat2: 80% igual a cat1, 20% aleatório
    mask = rng.random(n) < 0.8
    cat2 = np.where(mask, cat1, rng.choice(base, size=n, p=probs))

    # cat3: distribuição uniforme
    cat3 = rng.choice(base, size=n)

    # Injeta valores missing (~5%)
    missing_idx = rng.choice(n, size=int(0.05 * n), replace=False)
    cat1 = cat1.astype(object)
    cat3 = cat3.astype(object)
    cat1[missing_idx] = None
    cat3[missing_idx] = None

    df = pd.DataFrame({
        "cat1": cat1,
        "cat2": cat2,
        "cat3": cat3,
    })

    if not all_categorical:
        # Variável numérica com distribuição normal
        num1 = rng.normal(loc=0, scale=1, size=n)
        # Injeta missing nos mesmos índices
        num1[missing_idx] = np.nan
        # Cria alguns outliers
        outliers_idx = rng.choice(n, size=int(0.01 * n), replace=False)
        num1[outliers_idx] *= 10
        df["num1"] = num1

    # Target binário balanceado (~50/50)
    df["target"] = rng.integers(0, 2, size=n)

    return df


# ------------------------------------------------------------------
# Testes
# ------------------------------------------------------------------
def test_vif_numeric_high_corr():
    """Numéricas altamente correlacionadas → VIF alto em pelo menos uma."""
    df = _numeric_dataset(high_corr=True)
    vif = compute_vif(df, target_col="target", verbose="none")

    # Garantir colunas presentes
    assert {"x1", "x2", "x3"} == set(vif["variable"])

    # VIF máximo deve ser >5 pela correlação induzida
    assert vif["vif"].max() > 5
    # Nenhum valor NaN ou infinito
    assert np.isfinite(vif["vif"]).all()


@pytest.mark.parametrize("all_categorical", [True, False], ids=["only_cat", "mixed"])
def test_vif_handles_categorical(all_categorical):
    """DataFrames com categóricas (todas ou parcialmente) não explodem."""
    df = _categorical_dataset(all_categorical=all_categorical)

    vif = compute_vif(
        df,
        target_col="target",
        limite_categorico=10,  # força detecção categórica
        verbose="none",
    )

    exp_vars = {"cat1", "cat2", "cat3"} | ({"num1"} if not all_categorical else set())
    assert exp_vars == set(vif["variable"])
    assert np.isfinite(vif["vif"]).all()  # sem inf / nan

    # Como cat1 e cat2 têm alta dependência, pelo menos uma deve estar inflada
    assert vif.loc[vif["variable"].isin({"cat1", "cat2"}), "vif"].max() > 5


def test_vif_no_numeric_columns_raises():
    """Se não for possível derivar nenhuma coluna numérica, deve lançar ValueError."""
    df = pd.DataFrame({"cat_only": ["a", "b", "c"], "target": [0, 1, 0]})

    # Definimos limite_categorico=0 → impede fallback automático
    with pytest.raises(ValueError):
        compute_vif(df, target_col="target", limite_categorico=0, verbose="none")


def test_vif_all_inf_returns_nan():
    """DataFrames que ficam vazios após remoção de inf devem retornar NaN."""
    df = pd.DataFrame({"x": [np.inf, np.inf], "target": [0, 1]})
    vif = compute_vif(df, target_col="target", verbose="none")
    assert vif["vif"].isna().all()
