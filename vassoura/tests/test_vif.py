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


def _categorical_dataset(all_categorical=True, seed=1):
    rng = np.random.default_rng(seed)
    base = np.array(["A", "B", "C", "D"])

    cat1 = rng.choice(base, size=300, p=[0.4, 0.4, 0.1, 0.1])
    noise = rng.choice(["", "_x"], size=300, p=[0.9, 0.1])
    # Usa np.char.add para evitar UFuncNoLoopError
    cat2 = np.char.add(cat1, noise)

    cat3 = rng.choice(base, size=300)
    df = pd.DataFrame({"cat1": cat1, "cat2": cat2, "cat3": cat3})

    if not all_categorical:
        df["num1"] = rng.normal(size=300)  # coluna numérica extra

    df["target"] = rng.integers(0, 2, size=300)
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