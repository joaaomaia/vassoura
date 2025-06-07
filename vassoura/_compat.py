# vassoura/_compat.py  (importado pelo __init__.py logo no começo)
import subprocess, functools, os

# Garante UTF-8 no I/O padrão – e ignora bytes ilegais
os.environ.setdefault("PYTHONIOENCODING", "utf-8:ignore")

_orig_popen = subprocess.Popen

def _safe_popen(*popen_args, **popen_kw):
    if popen_kw.get("text") or popen_kw.get("universal_newlines"):
        popen_kw.setdefault("encoding", "utf-8")
        popen_kw.setdefault("errors", "ignore")   # ou "replace"
    return _orig_popen(*popen_args, **popen_kw)

subprocess.Popen = _safe_popen