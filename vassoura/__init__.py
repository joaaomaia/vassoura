"""Vassoura – Unified feature-selection & reporting framework."""

from .models import get as get_model, list_models
from .core import Vassoura

__all__ = ["get_model", "list_models", "Vassoura"]
__version__ = "0.0.1a0"
