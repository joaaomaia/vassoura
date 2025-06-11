"""vassoura – public API & utilities (corrigido).
Corta o *import* antecipado de ``vassoura.heuristics`` que causava
importação parcial e quebra nos testes (relative‑import sem pacote).
Em vez disso, usamos apenas um *meta‑path finder* que resolve
``import heuristics`` quando (e somente quando) ele realmente ocorre.
"""
from __future__ import annotations

__version__ = "0.1.0"

# ---- reexportações principais (mantidas) -----------------------------------
from . import _compat  # noqa: F401  (mantém side‑effects)
from .analisador import analisar_autocorrelacao  # noqa: F401
from .autocorrelacao import compute_panel_acf, plot_panel_acf  # noqa: F401
from .core import Vassoura  # noqa: F401
from .correlacao import compute_corr_matrix, plot_corr_heatmap  # noqa: F401
from .limpeza import clean  # noqa: F401
from .logging_utils import configure_logging  # noqa: F401
from .relatorio import generate_report  # noqa: F401
from .utils import (  # noqa: F401
    criar_dataset_pd_behavior,
    figsize_from_matrix,
    search_dtypes,
    suggest_corr_method,
)
from .vif import compute_vif, remove_high_vif  # noqa: F401
from .leakage import target_leakage  # noqa: F401

__all__ = [
    "search_dtypes",
    "suggest_corr_method",
    "figsize_from_matrix",
    "criar_dataset_pd_behavior",
    "compute_corr_matrix",
    "plot_corr_heatmap",
    "compute_vif",
    "remove_high_vif",
    "target_leakage",
    "clean",
    "generate_report",
    "compute_panel_acf",
    "plot_panel_acf",
    "analisar_autocorrelacao",
    "Vassoura",
    "configure_logging",
]

# ----------------------------------------------------------------------------
# Alias *lazy*  →  permite ``import heuristics`` em qualquer ponto
# ----------------------------------------------------------------------------
import importlib as _imp, sys as _sys
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader

class _HeurFinder(MetaPathFinder, Loader):
    def find_spec(self, fullname, path=None, target=None):  # noqa: D401
        if fullname != "heuristics":
            return None
        return spec_from_loader(fullname, self)

    # create_module é chamado primeiro; importamos o submódulo real aqui
    def create_module(self, spec):  # noqa: D401
        real = _imp.import_module(__name__ + ".heuristics")
        _sys.modules["heuristics"] = real
        return real

    # exec_module não precisa fazer nada porque o módulo já está pronto
    def exec_module(self, module):
        pass

# Insere o finder no início da cadeia para garantir que seja avaliado primeiro
_sys.meta_path.insert(0, _HeurFinder())
