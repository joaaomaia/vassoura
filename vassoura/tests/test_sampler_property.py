from __future__ import annotations

import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from vassoura.preprocessing import SampleManager


@given(st.integers(min_value=100, max_value=500))
def test_frac_sampling(n):
    df = pd.DataFrame({"a": range(n)})
    sm = SampleManager(strategy="auto", limit_mb=0, frac=0.3, stratify=False)
    Xt = sm.fit_transform(df)
    assert abs(len(Xt) - int(n * 0.3)) <= 1
