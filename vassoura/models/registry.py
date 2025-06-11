from __future__ import annotations

from importlib import import_module
from pathlib import Path
import pkgutil

from .base import WrapperBase

_registry: dict[str, type] = {}


def _discover() -> None:
    pkg = Path(__file__).parent
    for _, name, _ in pkgutil.iter_modules([str(pkg)]):
        if name in {"__init__", "base", "registry", "utils"}:
            continue
        module = import_module(f"vassoura.models.{name}")
        for attr in dir(module):
            obj = getattr(module, attr)
            if (
                isinstance(obj, type)
                and hasattr(obj, "name")
                and issubclass(obj, WrapperBase)
            ):
                _registry[obj.name] = obj


def get(name: str):
    return _registry[name]


def list_models() -> list[str]:
    return list(_registry.keys())


_discover()
