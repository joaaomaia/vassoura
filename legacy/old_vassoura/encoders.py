# -*- coding: utf-8 -*-
from __future__ import annotations

"""Custom Weight-of-Evidence (WoE) encoder compatible with scikit-learn pipelines.

Features
--------
* Calculates WoE and Information Value (IV) for categorical features.
* Creates new columns with suffix "_woe"; optionally drops originals.
* Handles missing values as dedicated category.
* Laplace smoothing to avoid log(0).
* Stores full WoE mapping (`woe_log_`) and IV per feature (`iv_log_`).
* Provides `summary()` to export detailed report to Excel (``.xlsx``).
* Offers `plot_woe()` for quick visual inspection.
* Supports persistence (`save`, `load`, `export_log`, `load_from_json`) via `pickle` or JSON.
* Robust to unseen categories at transform time (configurable default), with warnings for missing columns.
"""

from pathlib import Path
import pickle
import json
import warnings
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["WOEGuard"]


class WOEGuard(BaseEstimator, TransformerMixin):
    """Encoder de Peso de Evidência (WoE) para variáveis categóricas.

    Parameters
    ----------
    categorical_cols : List[str]
        Lista de colunas categóricas a codificar.
    drop_original : bool, default=False
        Se `True`, remove as colunas originais após transformação.
    suffix : str, default="_woe"
        Sufixo para novas colunas.
    alpha : float, default=0.5
        Suavização de Laplace para evitar logs infinitos.
    default_woe : float, default=0.0
        Valor WoE default para categorias não vistas em `transform`.
    include_nan : bool, default=True
        Trata `NaN` como categoria separada (`"__nan__"`).
    """

    def __init__(
        self,
        categorical_cols: List[str],
        drop_original: bool = False,
        suffix: str = "_woe",
        alpha: float = 0.5,
        default_woe: float = 0.0,
        include_nan: bool = True,
    ) -> None:
        self.categorical_cols = categorical_cols
        self.drop_original = drop_original
        self.suffix = suffix
        self.alpha = alpha
        self.default_woe = default_woe
        self.include_nan = include_nan

        # Atributos pós-fit
        self.woe_log_: Dict[str, Dict[Union[str, float], float]] = {}
        self.iv_log_: Dict[str, float] = {}
        self.global_event_rate_: Optional[float] = None
        self.fitted_ = False

    def _validate_target(self, y: pd.Series) -> None:
        if not set(y.unique()).issubset({0, 1}):
            raise ValueError("Target deve ser binário contendo apenas 0 e 1.")

    def _prepare_series(self, s: pd.Series) -> pd.Series:
        """Converte para `category`, tratando `NaN` separadamente se configurado."""
        if self.include_nan:
            return s.astype(object).where(~s.isna(), other="__nan__").astype("category")
        return s.astype("category")

    def _calculate_woe_iv(self, s: pd.Series, y: pd.Series) -> Dict[str, object]:
        df = pd.DataFrame({"feature": s, "target": y})
        agg = (
            df.groupby("feature", observed=True)["target"]
            .agg(total="count", bad="sum")
        )
        agg["good"] = agg["total"] - agg["bad"]
        # laplace smoothing
        agg[["bad", "good"]] += self.alpha
        total_good = agg["good"].sum()
        total_bad = agg["bad"].sum()
        agg["dist_good"] = agg["good"] / total_good
        agg["dist_bad"] = agg["bad"] / total_bad
        agg["woe"] = np.log(agg["dist_good"] / agg["dist_bad"])
        agg["iv_component"] = (agg["dist_good"] - agg["dist_bad"]) * agg["woe"]
        return {"woe_mapping": agg["woe"].to_dict(), "iv": agg["iv_component"].sum()}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Calcula WoE e IV para `categorical_cols`. Retorna `self`."""
        X = X.copy()
        y = pd.Series(y).reset_index(drop=True)
        self._validate_target(y)
        self.global_event_rate_ = y.mean()

        for col in self.categorical_cols:
            if col not in X.columns:
                raise KeyError(f"Coluna '{{col}}' não encontrada em X.")
            s = self._prepare_series(X[col])
            res = self._calculate_woe_iv(s, y)
            self.woe_log_[col] = res["woe_mapping"]
            self.iv_log_[col] = res["iv"]

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica WoE criando novas colunas `_woe`, com warnings para colunas faltantes."""
        if not self.fitted_:
            raise RuntimeError("Encoder não foi ajustado. Execute `.fit()` primeiro.")

        X = X.copy()
        missing = [c for c in self.categorical_cols if c not in X.columns]
        if missing:
            warnings.warn(f"As colunas {missing} não foram encontradas no DataFrame de entrada e serão ignoradas.")
        for col in self.categorical_cols:
            if col not in X.columns:
                continue
            s = self._prepare_series(X[col])
            mapping = self.woe_log_.get(col, {})
            new_col = col + self.suffix
            vals = s.map(mapping).astype(float)
            X[new_col] = vals.fillna(self.default_woe)

        if self.drop_original:
            to_drop = [c for c in self.categorical_cols if c in X.columns]
            X = X.drop(columns=to_drop)
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:  # type: ignore[override]
        """Ajusta e transforma em uma só etapa e retorna `X` transformado com `y` como primeira coluna."""
        Xt = self.fit(X, y).transform(X)
        Xt[y.name] = y.values  # adiciona a coluna y
        # reorganiza para que y fique como primeira coluna
        cols = [y.name] + [c for c in Xt.columns if c != y.name]
        return Xt[cols]

    def view_log(self) -> Dict[str, Dict]:
        """Retorna o mapeamento interno `woe_log_`."""
        return self.woe_log_

    def export_log(self, path: Union[str, Path]) -> None:
        """Salva `woe_log_` e `iv_log_` em arquivo JSON."""
        data = {"woe_log": self.woe_log_, "iv_log": self.iv_log_}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    @classmethod
    def load_from_json(
        cls,
        path: Union[str, Path],
        drop_original: Optional[bool] = None,
        suffix: Optional[str] = None,
        alpha: Optional[float] = None,
        default_woe: Optional[float] = None,
        include_nan: Optional[bool] = None,
    ) -> "WOEGuard":
        """Carrega mapeamento de JSON e retorna um encoder pronto para `transform()`.

        Parâmetros opcionais servem para reconfigurar a instância."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        woe_log = data.get("woe_log", {})
        iv_log = data.get("iv_log", {})
        cols = list(woe_log.keys())
        encoder = cls(
            categorical_cols=cols,
            drop_original=drop_original if drop_original is not None else False,
            suffix=suffix if suffix is not None else "_woe",
            alpha=alpha if alpha is not None else 0.5,
            default_woe=default_woe if default_woe is not None else 0.0,
            include_nan=include_nan if include_nan is not None else True,
        )
        encoder.woe_log_ = woe_log
        encoder.iv_log_ = iv_log
        encoder.fitted_ = True
        return encoder

    def summary(self, path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Retorna DataFrame resumo e opcionalmente salva em XLSX."""
        if not self.fitted_:
            raise RuntimeError("Encoder não foi ajustado.")
        rows = []
        for col, mapping in self.woe_log_.items():
            for cat, woe in mapping.items():
                rows.append({
                    "feature": col,
                    "category": cat,
                    "woe": woe,
                    "iv": self.iv_log_.get(col, np.nan),
                })
        df = pd.DataFrame(rows)
        if path is not None:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="WoE_Summary", index=False)
        return df

    def plot_woe(self, feature: str, top_n: int = 30) -> None:
        """Plota WoE para `feature` (até `top_n` categorias)."""
        if not self.fitted_:
            raise RuntimeError("Encoder não foi ajustado.")
        if feature not in self.woe_log_:
            raise KeyError(f"Feature '{feature}' não encontrada no encoder.")
        ser = pd.Series(self.woe_log_[feature]).sort_values().iloc[:top_n]
        ser.plot(kind="barh")
        plt.axvline(0, linestyle="--", linewidth=1)
        plt.title(f"WoE por categoria – {feature}")
        plt.xlabel("WoE")
        plt.tight_layout()
        plt.show()

    def save(self, path: Union[str, Path]) -> None:
        """Serializa o encoder via pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Union[str, Path]) -> "WOEGuard":
        """Carrega encoder salvo via `save()`."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        status = "fitted" if getattr(self, "fitted_", False) else "unfitted"
        return (
            f"<WOEGuard n_features={len(self.categorical_cols)} status={status} "
            f"drop_original={self.drop_original}>"
        )
