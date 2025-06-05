# vassoura/__init__.py

from .utils import search_dtypes, suggest_corr_method, figsize_from_matrix
from .correlacao import compute_corr_matrix, plot_corr_heatmap
from .vif import compute_vif, remove_high_vif
from .limpeza import clean
from .relatorio import generate_report
from .autocorrelacao import compute_panel_acf, plot_panel_acf

__all__ = [
    "search_dtypes",
    "suggest_corr_method",
    "figsize_from_matrix",
    "compute_corr_matrix",
    "plot_corr_heatmap",
    "compute_vif",
    "remove_high_vif",
    "clean",
    "generate_report",
    "compute_panel_acf",
    "plot_panel_acf"
]
