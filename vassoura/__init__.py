__version__ = "0.1.0"

from . import _compat  # importa ajustes de compatibilidade (não está no __all__)
from .analisador import analisar_autocorrelacao
from .autocorrelacao import compute_panel_acf, plot_panel_acf
from .core import Vassoura
from .correlacao import compute_corr_matrix, plot_corr_heatmap
from .limpeza import clean
from .logging_utils import configure_logging
from .relatorio import generate_report
from .utils import (
    criar_dataset_pd_behavior,
    figsize_from_matrix,
    search_dtypes,
    suggest_corr_method,
)
from .vif import compute_vif, remove_high_vif
from .leakage import target_leakage

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
