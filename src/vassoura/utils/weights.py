from __future__ import annotations

import numpy as np
from sklearn.utils.class_weight import compute_sample_weight


def make_balanced_sample_weights(y) -> np.ndarray:
    """Return array of balanced sample weights.

    Parameters
    ----------
    y : array-like
        Target labels.

    Returns
    -------
    numpy.ndarray
        Computed sample weights.
    """
    return compute_sample_weight(class_weight="balanced", y=y)
