from __future__ import annotations

import numpy as np
from inspect import signature
from sklearn.utils.class_weight import compute_sample_weight


def make_sample_weights(y) -> np.ndarray:
    """Return array of weights using 'balanced' strategy."""
    return compute_sample_weight(class_weight="balanced", y=y)


def supports_sample_weight(estimator) -> bool:
    """Return True if the estimator.fit accepts sample_weight."""
    try:
        sig = signature(estimator.fit)
    except (AttributeError, ValueError):
        return False
    return "sample_weight" in sig.parameters
