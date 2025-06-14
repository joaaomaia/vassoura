"""Vassoura – Unified feature-selection & reporting framework."""

import warnings
import sklearn

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="sklearn"
)

from .models import get as get_model, list_models
from .core import Vassoura
from .audit import AuditTrail


__all__ = ["get_model", "list_models", "Vassoura", "AuditTrail"]
__version__ = "0.1.0"
