"""Importance heuristics."""

from .basic import basic_importance
from .medium import medium_importance
from typing import Callable, Optional  # noqa: F401

advanced_importance: Optional[Callable]
try:
    from .advanced import advanced_importance as _advanced_importance

    advanced_importance = _advanced_importance
except Exception:  # optional dependency "shap"
    advanced_importance = None

__all__ = [
    "basic_importance",
    "medium_importance",
    "advanced_importance",
]
