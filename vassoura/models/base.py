from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class WrapperBase(ABC):
    """Abstract base class for model wrappers."""

    name: str
    model: Any

    @abstractmethod
    def __init__(self, **params) -> None:  # pragma: no cover - interface only
        ...

    @abstractmethod
    def fit(
        self, X, y, sample_weight=None
    ):  # pragma: no cover - interface only
        ...

    @abstractmethod
    def predict(self, X):  # pragma: no cover - interface only
        ...

    def get_params(self, deep: bool = True) -> Mapping[str, Any]:
        if hasattr(self.model, "get_params"):
            return self.model.get_params(deep=deep)
        return {}

    def set_params(self, **params):
        if hasattr(self.model, "set_params"):
            self.model.set_params(**params)
        return self
