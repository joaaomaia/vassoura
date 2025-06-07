"""core.py – High-level object-oriented API for Vassoura.

Refactored 2025-06-05
--------------------
* Introduces **dynamic dispatch** of heuristics via `_heuristic_funcs` map.
* Adds robust guardrails no VIF, pulando quando não houver colunas numéricas suficientes.
* Mantém heurística de IV funcional e placeholders para `importance` e `graph_cut`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

import warnings
import math

import numpy as np
import pandas as pd

from .correlacao import compute_corr_matrix
from .vif import compute_vif
from .relatorio import generate_report
from .heuristics import graph_cut


# --------------------------------------------------------------------- #
# Helper functions                                                      #
# --------------------------------------------------------------------- #


def _compute_iv(series: pd.Series, target: pd.Series, *, bins: int = 10) -> float:
    """Computes Information Value using quantile binning (numeric) or
    category grouping (categorical). Very light implementation – for
    production we may migrate to `scorecardpy` or `optbinning`.
    """
    # Ensure target has exactly two classes
    uniques = target.dropna().unique()
    if len(uniques) != 2:
        warnings.warn("Target must be binary (0/1) for IV calculation.")
        return 0.0
    # Map valores arbitrários → {0,1}
    mapping = {uniques[0]: 0, uniques[1]: 1}
    target_num = target.map(mapping)

    if series.dtype.kind in "bifc":
        # Numeric – quantile bins (duplicates handled gracefully)
        try:
            binned = pd.qcut(series, q=bins, duplicates="drop")
        except ValueError:
            # Constant series ou valores únicos insuficientes
            return 0.0
    else:
        binned = series.astype("category")

    tab = pd.crosstab(binned, target_num)
    if tab.shape[1] != 2:
        warnings.warn("Target must be binary (0/1) for IV calculation.")
        return 0.0
    tab = tab.rename(columns={0: "good", 1: "bad"}).replace(0, 0.5)  # suavização
    dist_good = tab["good"] / tab["good"].sum()
    dist_bad = tab["bad"] / tab["bad"].sum()
    woe = np.log(dist_good / dist_bad)
    iv = ((dist_good - dist_bad) * woe).sum()
    return iv


# --------------------------------------------------------------------- #
# Main class                                                            #
# --------------------------------------------------------------------- #

DEFAULT_HEURISTICS = ["corr", "vif"]


class Vassoura:
    """Stateful cleaning session – **one object, one dataset**.

    Usabilidade simples: basta instanciar com DataFrame e colunas-chave,
    chamar `run()` e então `generate_report()`. O fluxo cuida de:
    - Preservar o df_original intacto;
    - Cachear correlação, VIF e IV para evitar recomputação;
    - Pular VIF quando não houver colunas numéricas suficientes.

    Parâmetros
    ----------
    df : pd.DataFrame
        Dataset original – *nunca* alterado em-place.
    target_col : str | None
        Nome da coluna-alvo (excluída de cálculos). Se `None`, heurísticas
        que dependam do target (ex: IV) são ignoradas.
    keep_cols : list[str] | None
        Colunas protegidas que jamais serão removidas.
    heuristics : list[str] | None
        Sequência de heurísticas a executar. Valores suportados:
        `'corr'`, `'vif'`, `'iv'`, `'importance'`, `'graph_cut'`.
    thresholds : dict[str, float] | None
        Dicionário de limites por heurística, ex: `{'corr':0.9, 'vif':10, 'iv':0.02}`.
    missing_threshold : float | None
        Se definido, remove colunas com proporção de valores ausentes acima
        desse limite antes das outras heurísticas.
    engine : {"pandas", "dask", "polars"}
        Backend utilizado nos cálculos pesados. ``"pandas"`` é o padrão.
    verbose : bool
        Se `True`, imprime progresso no console.
    n_steps : int | None
        Quantidade de iterações fracionadas para remoção por correlação.
        ``None`` mantém o comportamento tradicional.
    vif_n_steps : int
        Número de etapas para remoção por VIF. Deve ser >= 1.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        target_col: str | None = None,
        keep_cols: Optional[List[str]] = None,
        heuristics: Optional[List[str]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        missing_threshold: Optional[float] = None,
        engine: str = "pandas",
        verbose: bool = True,
        n_steps: int | None = None,
        vif_n_steps: int = 1,
    ) -> None:
        self.df_original = df.copy()
        self.df_current = df.copy()
        self.target_col = target_col
        self.keep_cols = set(keep_cols or [])
        self.engine = engine
        self.verbose = verbose
        if n_steps is not None and n_steps < 1:
            raise ValueError("n_steps deve ser >= 1 ou None")
        if vif_n_steps < 1:
            raise ValueError("vif_n_steps deve ser >= 1")
        self.n_steps = n_steps
        self.vif_n_steps = vif_n_steps

        self.heuristics = heuristics or DEFAULT_HEURISTICS.copy()
        # Valores padrão, podem ser sobrescritos
        self.thresholds = {"corr": 0.9, "vif": 10.0, "iv": 0.02}
        if missing_threshold is not None:
            self.thresholds["missing"] = missing_threshold
        if thresholds:
            self.thresholds.update(thresholds)

        # Caches internos
        self._corr_matrix: Optional[pd.DataFrame] = None
        self._corr_matrix_final: Optional[pd.DataFrame] = None
        self._vif_df_before: Optional[pd.DataFrame] = None
        self._vif_df: Optional[pd.DataFrame] = None
        self._iv_series: Optional[pd.Series] = None
        self._history: List[Dict[str, Any]] = []  # cada entrada = {'cols', 'reason'}

        # Map heurísticas → métodos
        self._heuristic_funcs: Dict[str, Callable[[], None]] = {
            "missing": self._apply_missing,
            "corr": self._apply_corr,
            "vif": self._apply_vif,
            "iv": self._apply_iv,
            "importance": self._apply_importance,
            "graph_cut": self._apply_graph_cut,
        }

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def run(self, *, recompute: bool = False) -> pd.DataFrame:
        """Executa heurísticas na ordem definida. Retorna DataFrame limpo.

        Se `recompute=True`, reseta todos os caches antes de rodar.
        """
        if recompute:
            self.reset()
        if self.thresholds.get("missing") is not None:
            self._apply_missing()
        for h in self.heuristics:
            func = self._heuristic_funcs.get(h)
            if func is None:
                raise ValueError(f"Heuristic '{h}' not recognized.")
            func()
        # Armazena correlação e VIF finais para relatórios
        self._corr_matrix_final = compute_corr_matrix(
            self.df_current,
            method="auto",
            target_col=self.target_col,
            include_target=False,
            engine=self.engine,
            verbose=self.verbose,
        )
        if self._vif_df is None:
            try:
                self._vif_df = compute_vif(
                    self.df_current,
                    target_col=self.target_col,
                    include_target=False,
                    engine=self.engine,
                    verbose=self.verbose,
                )
            except Exception:
                self._vif_df = None
        return self.df_current

    def remove_additional(self, columns: List[str]) -> None:
        """Força remoção manual de colunas pós-limpeza."""
        self._drop(columns, reason="manual")

    def generate_report(self, path: str | Path = "vassoura_report.html") -> str:
        """Gera relatório utilizando caches já computados."""

        if self._corr_matrix is None:
            self._corr_matrix = compute_corr_matrix(
                self.df_original.drop(columns=[self.target_col], errors="ignore"),
                method="auto",
                target_col=None,
                include_target=False,
                engine=self.engine,
                verbose=self.verbose,
            )

        if self._corr_matrix_final is None:
            self._corr_matrix_final = compute_corr_matrix(
                self.df_current.drop(columns=[self.target_col], errors="ignore"),
                method="auto",
                target_col=None,
                include_target=False,
                engine=self.engine,
                verbose=self.verbose,
            )

        if self._vif_df_before is None:
            try:
                self._vif_df_before = compute_vif(
                    self.df_original,
                    target_col=self.target_col,
                    include_target=False,
                    engine=self.engine,
                    verbose=self.verbose,
                )
            except Exception:
                self._vif_df_before = None

        if self._vif_df is None:
            try:
                self._vif_df = compute_vif(
                    self.df_current,
                    target_col=self.target_col,
                    include_target=False,
                    engine=self.engine,
                    verbose=self.verbose,
                )
            except Exception:
                self._vif_df = None

        precomputed = {
            "df_clean": self.df_current,
            "corr_before": self._corr_matrix,
            "corr_after": self._corr_matrix_final,
            "vif_before": self._vif_df_before,
            "vif_after": self._vif_df,
            "dropped_cols": self.dropped,
        }

        return generate_report(
            self.df_original,
            target_col=self.target_col,
            keep_cols=list(self.keep_cols),
            corr_threshold=self.thresholds.get("corr"),
            vif_threshold=self.thresholds.get("vif"),
            verbose=self.verbose,
            output_path=path,
            precomputed=precomputed,
        )

    def help(self) -> None:
        """Imprime instruções básicas de uso da classe."""
        msg = (
            "Vassoura usage:\n"
            " 1. Instancie com Vassoura(df, target_col='alvo').\n"
            " 2. Chame run() para aplicar as heurísticas.\n"
            " 3. Acesse o resultado em df_current ou use generate_report()."
        )
        print(msg)

    def reset(self) -> None:
        """Restaura sessão ao estado inicial (apaga caches e histórico)."""
        self.df_current = self.df_original.copy()
        self._corr_matrix = None
        self._corr_matrix_final = None
        self._vif_df_before = None
        self._vif_df = None
        self._iv_series = None
        self._history.clear()

    # ------------------------------------------------------------------ #
    # Heurísticas                                                        #
    # ------------------------------------------------------------------ #
    def _apply_missing(self) -> None:
        thr = self.thresholds.get("missing")
        if thr is None:
            return
        if self.verbose:
            print(f"[Vassoura] Missing heuristic (thr>{thr})")
        miss_ratio = self.df_current.isna().mean()
        cols = [c for c, r in miss_ratio.items() if r > thr and c != self.target_col]
        self._drop(cols, reason=f"missing>{thr}")

    def _apply_corr(self) -> None:
        thr = self.thresholds.get("corr", 0.9)
        if self.verbose:
            msg = f"[Vassoura] Corr heuristic (thr={thr})"
            if self.n_steps is not None:
                msg += f" n_steps={self.n_steps}"
            print(msg)
        iteration = 0
        while True:
            if self._corr_matrix is None or iteration > 0:
                self._corr_matrix = compute_corr_matrix(
                    self.df_current,
                    method="auto",
                    target_col=self.target_col,
                    include_target=False,
                    engine=self.engine,
                    verbose=self.verbose,
                )
            upper_tri = self._corr_matrix.where(
                np.triu(np.ones_like(self._corr_matrix, dtype=bool), k=1)
            )
            pairs = upper_tri.stack().loc[lambda s: s.abs() > thr]
            if pairs.empty:
                break
            step_limit = len(pairs) if self.n_steps is None else math.ceil(
                len(pairs) / self.n_steps
            )
            removed_this_iter = 0
            for (var1, var2), corr_val in pairs.sort_values(
                key=lambda s: s.abs(), ascending=False
            ).items():
                if (
                    var1 not in self.df_current.columns
                    or var2 not in self.df_current.columns
                ):
                    continue
                drop_var = self._choose_var_to_drop(var1, var2)
                if drop_var and drop_var in self.df_current.columns:
                    self._drop([drop_var], reason=f"corr>{thr}")
                    removed_this_iter += 1
                    if removed_this_iter >= step_limit:
                        break
            iteration += 1

    def _apply_vif(self) -> None:
        thr = self.thresholds.get("vif", 10.0)
        # Identificar colunas numéricas previstas para VIF
        num_cols = self.df_current.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in num_cols:
            num_cols.remove(self.target_col)
        # Se tiver ≤1 coluna numérica, não faz sentido calcular VIF
        if len(num_cols) <= 1:
            if self.verbose:
                print(
                    f"[Vassoura] Pulando VIF (col numéricas insuficientes: {len(num_cols)})"
                )
            return
        if self.verbose:
            msg = f"[Vassoura] VIF heuristic (thr={thr})"
            if self.vif_n_steps != 1:
                msg += f" vif_n_steps={self.vif_n_steps}"
            print(msg)
        while True:
            try:
                self._vif_df = compute_vif(
                    self.df_current,
                    target_col=self.target_col,
                    include_target=False,
                    engine=self.engine,
                    verbose=self.verbose,
                )
                if self._vif_df_before is None:
                    self._vif_df_before = self._vif_df.copy()
            except Exception:
                if self.verbose:
                    print("[Vassoura] Erro no cálculo de VIF — pulando heurística.")
                break
            worst = self._vif_df[self._vif_df["vif"] > thr]
            if worst.empty:
                break
            step_limit = 1 if self.vif_n_steps == 1 else max(1, math.ceil(len(worst) / self.vif_n_steps))
            removed_this_iter = 0
            for _, row in worst.sort_values("vif", ascending=False).iterrows():
                var = row["variable"]
                self._drop([var], reason=f"vif>{thr}")
                removed_this_iter += 1
                if removed_this_iter >= step_limit:
                    break

    def _apply_iv(self) -> None:
        thr = self.thresholds.get("iv", 0.02)
        if self.target_col is None:
            warnings.warn("IV heuristic skipped: target_col not provided.")
            return
        target = self.df_current[self.target_col]
        if target.nunique(dropna=False) != 2:
            if self.verbose:
                print("[Vassoura] Skipping IV heuristic (target not binary)")
            return
        if self.verbose:
            print(f"[Vassoura] IV heuristic (thr<{thr}) – removendo low IV")
        if self._iv_series is None:
            iv_values = {}
            for col in self.df_current.columns:
                if col == self.target_col or col in self.keep_cols:
                    continue
                iv_values[col] = _compute_iv(self.df_current[col], target)
            self._iv_series = pd.Series(iv_values, name="iv")
        low_iv = self._iv_series[self._iv_series < thr].index.tolist()
        if low_iv:
            self._drop(low_iv, reason=f"iv<{thr}")
        else:
            # Log execução mesmo sem remoções para manter histórico consistente
            self._history.append({"cols": [], "reason": f"iv<{thr}"})

    def _apply_importance(self) -> None:
        if self.target_col is None:
            warnings.warn("Importance heuristic skipped: target_col not provided.")
            return

        thr = self.thresholds.get("importance", 0.2)
        if self.verbose:
            print(f"[Vassoura] Importance heuristic (drop_lowest={thr})")

        from .heuristics import importance

        result = importance(
            self.df_current,
            target_col=self.target_col,
            keep_cols=list(self.keep_cols),
            drop_lowest=thr,
        )

        removed = result.get("removed", [])
        if removed:
            self._drop(removed, reason=f"importance<{thr}")

    def _apply_graph_cut(self) -> None:
        """
        Aplica a heurística de mínimo conjunto de vértices (graph-cut):
        1) Remove target_col (se existir) antes de montar grafo.
        2) Filtra apenas colunas numéricas.
        3) Chama graph_cut(), obtém lista 'removed' e repassa a _drop().
        """
        thr = self.thresholds.get("graph_cut", 0.9)
        if self.verbose:
            print(f"[Vassoura] Graph-cut heuristic (thr={thr})")

        # 1) Preparar DataFrame para grafo: copiar e tirar coluna alvo
        df_for_graph = self.df_current.copy()
        if self.target_col in df_for_graph.columns:
            df_for_graph = df_for_graph.drop(columns=[self.target_col])

        # 2) Selecionar somente colunas numéricas
        num_cols = df_for_graph.select_dtypes(include=[np.number]).columns.tolist()
        df_for_graph = df_for_graph[num_cols]

        # Se tiver 0 ou 1 coluna numérica, não faz sentido rodar o grafo
        if len(df_for_graph.columns) <= 1:
            if self.verbose:
                print(
                    f"[Vassoura] Pulando Graph-cut (só {len(df_for_graph.columns)} coluna(s) numérica(s))."
                )
            return

        # 3) Chamar a função graph_cut do módulo heuristics
        result = graph_cut(
            df_for_graph,
            corr_threshold=thr,
            keep_cols=list(self.keep_cols),
        )

        removed = result.get("removed", [])
        if removed:
            self._drop(removed, reason=f"graph_cut>{thr}")

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _choose_var_to_drop(self, a: str, b: str) -> str:
        if a in self.keep_cols and b in self.keep_cols:
            return b
        if a in self.keep_cols:
            return b
        if b in self.keep_cols:
            return a
        med_a = self._corr_matrix[a].abs().median()
        med_b = self._corr_matrix[b].abs().median()
        return a if med_a >= med_b else b

    def _drop(self, cols: List[str], reason: str) -> None:
        cols = [c for c in cols if c not in self.keep_cols]
        if not cols:
            return
        self.df_current.drop(columns=cols, errors="ignore", inplace=True)
        self._history.append({"cols": cols, "reason": reason})
        if self.verbose:
            print(f"  → dropped {cols} ({reason})")

    # ------------------------------------------------------------------ #
    # Propriedades                                                       #
    # ------------------------------------------------------------------ #
    @property
    def history(self) -> List[Dict[str, Any]]:
        return self._history.copy()

    @property
    def dropped(self) -> List[str]:
        return [c for step in self._history for c in step["cols"]]
