"""Vassoura – Unified feature-selection & reporting framework."""

from .models import get as get_model, list_models
from .core import Vassoura
from .audit import AuditTrail


__all__ = ["get_model", "list_models", "Vassoura", "AuditTrail"]
__version__ = "0.0.1a0"
