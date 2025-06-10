"""
Stub que permite `import heuristics` antes de `import vassoura`.
Agora sem colisão com o alias interno – apenas garante que o
pacote‑pai seja carregado e delega tudo para `vassoura.heuristics`.
"""
import importlib as _imp
import sys as _sys

_imp.import_module("vassoura")                # garante pacote carregado
_real = _imp.import_module("vassoura.heuristics")
_sys.modules[__name__] = _real
globals().update(_real.__dict__)
