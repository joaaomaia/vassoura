from __future__ import annotations

import logging
from typing import Optional, List

import pandas as pd
import numpy as np

from .helpers import calculate_ks, calculate_psi, search_dtypes

__all__ = ["AuditTrail"]


class AuditTrail:
    """Utility to take and compare DataFrame snapshots."""

    def __init__(
        self,
        track_histograms: bool = False,
        track_distributions: bool = False,
        enable_logging: bool = False,
        auto_detect_types: bool = False,
        target_col: str = "target",
        limite_categorico: int = 50,
        default_keys: Optional[List[str]] = None,
    ) -> None:
        self.track_histograms = track_histograms
        self.track_distributions = track_distributions
        self.enable_logging = enable_logging
        self.auto_detect_types = auto_detect_types
        self.target_col = target_col
        self.limite_categorico = limite_categorico
        self.default_keys = default_keys
        self.snapshots: dict[str, dict] = {}

        self.logger = logging.getLogger("audittrail")
        if self.enable_logging:
            logging.basicConfig(
                filename="audit_trail.log",
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(message)s",
            )

    # ------------------------------------------------------------------
    def _display(self, obj: object) -> None:
        try:  # pragma: no cover - environment check
            from IPython.display import display

            display(obj)
        except Exception:  # pragma: no cover - fallback
            print(obj)

    # ------------------------------------------------------------------
    def take_snapshot(
        self, df: pd.DataFrame, name: str, keys: Optional[List[str]] = None
    ) -> None:
        if name in self.snapshots:
            raise ValueError(f"Snapshot com nome '{name}' já existe.")

        keys = keys or self.default_keys
        try:
            cat_summary = df.describe(include=["object", "category"]).T
        except ValueError:
            cat_summary = pd.DataFrame()

        snap: dict[str, object] = {
            "shape": df.shape,
            "dtypes": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "num_summary": df.describe(include=[np.number]).T,
            "cat_summary": cat_summary,
            "key_dupes": df.duplicated(subset=keys).sum() if keys else None,
            "keys": keys,
        }

        if self.track_histograms:
            snap["histograms"] = {
                col: df[col].value_counts(dropna=False).to_dict()
                for col in df.columns
                if df[col].dtype.kind in {"O", "i", "u", "f", "b"}
            }

        if self.auto_detect_types:
            num_cols, cat_cols = search_dtypes(
                df, self.target_col, self.limite_categorico
            )
            snap["num_cols"] = num_cols
            snap["cat_cols"] = cat_cols

        self.snapshots[name] = snap
        self._log(f"Snapshot '{name}' salvo com sucesso. Shape: {df.shape}")

    # ------------------------------------------------------------------
    def describe_snapshot(self, name: str) -> None:
        if name not in self.snapshots:
            raise ValueError(f"Snapshot '{name}' não encontrado.")

        snap = self.snapshots[name]
        print(f"\n📄 Descrição do snapshot '{name}':\n")
        print(f"▶️ Shape: {snap['shape']}")
        print(f"▶️ Chaves de duplicação: {snap['keys']}")
        if snap["keys"]:
            print(f"   • Duplicatas nas chaves: {snap['key_dupes']}")

        print("\n🧱 Tipos de dados:")
        self._display(snap["dtypes"])

        if self.auto_detect_types:
            print("\n🔎 Colunas detectadas automaticamente:")
            print(
                f"   • Numéricas ({len(snap.get('num_cols', []))}): {snap.get('num_cols', [])}"
            )
            print(
                f"   • Categóricas ({len(snap.get('cat_cols', []))}): {snap.get('cat_cols', [])}"
            )

        print("\n🕳️ Valores ausentes:")
        missing = snap["missing"]
        missing = missing[missing > 0]
        if missing.empty:
            print("  ✅ Nenhuma coluna com valores ausentes.")
        else:
            self._display(missing.sort_values(ascending=False))

        print("\n📊 Estatísticas numéricas:")
        self._display(snap["num_summary"])

        print("\n🏷️ Estatísticas categóricas:")
        self._display(snap["cat_summary"])

        if self.track_histograms and "histograms" in snap:
            print("\n📈 Histogramas (categorias apenas):")
            for col in snap.get("cat_cols", []):
                hist = snap["histograms"].get(col)
                if hist:
                    top3 = dict(list(hist.items())[:3])
                    print(f"  {col}: {len(hist)} valores distintos (top 3: {top3})")

    # ------------------------------------------------------------------
    def compare_snapshots(self, name1: str, name2: str) -> None:
        if name1 not in self.snapshots or name2 not in self.snapshots:
            raise ValueError("Snapshot não encontrado.")

        snap1, snap2 = self.snapshots[name1], self.snapshots[name2]

        print(f"\n🔍 Comparando '{name1}' vs '{name2}':\n")
        print("▶️ Shape:")
        print(f"  {name1}: {snap1['shape']} vs {name2}: {snap2['shape']}\n")

        common_cols = snap1["dtypes"].index.intersection(snap2["dtypes"].index)
        dtypes_diff = snap1["dtypes"][common_cols] != snap2["dtypes"][common_cols]
        dtypes_diff = dtypes_diff.loc[lambda x: x]
        if not dtypes_diff.empty:
            print("▶️ Mudanças em tipos de dados:\n", dtypes_diff, "\n")

        missing_diff = (snap2["missing"] - snap1["missing"]).loc[lambda x: x != 0]
        if not missing_diff.empty:
            print("▶️ Diferença de valores ausentes:\n", missing_diff, "\n")

        if snap1["keys"] and snap2["keys"]:
            print("▶️ Duplicatas nas chaves:")
            print(f"  {name1}: {snap1['key_dupes']} vs {name2}: {snap2['key_dupes']}\n")

        print("▶️ Mudança na média de variáveis numéricas:")
        mean_diff = snap2["num_summary"]["mean"] - snap1["num_summary"]["mean"]
        mean_diff = mean_diff.dropna()
        self._display(mean_diff[mean_diff.abs() > 1e-6])

        if self.track_distributions and "histograms" in snap1 and "histograms" in snap2:
            print("\n▶️ KS-test e PSI para variáveis:")
            cols_common = set(snap1.get("histograms", {})).intersection(
                snap2.get("histograms", {})
            )
            for col in sorted(cols_common):
                dist1 = snap1["histograms"].get(col, {})
                dist2 = snap2["histograms"].get(col, {})
                ks_stat = calculate_ks(dist1, dist2)
                psi_val = calculate_psi(dist1, dist2)
                alerta = " ⚠️" if psi_val > 0.2 else ""
                print(f"  {col}: KS={ks_stat:.3f}, PSI={psi_val:.3f}{alerta}")
                if psi_val > 0.2:
                    self._log(
                        f"Alerta PSI>0.2 detectado para '{col}': PSI={psi_val:.3f}"
                    )

    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.enable_logging:
            self.logger.info(msg)

    # ------------------------------------------------------------------
    def list_snapshots(self) -> None:
        print("📚 Snapshots disponíveis:")
        for k in self.snapshots:
            print(f" - {k}")
