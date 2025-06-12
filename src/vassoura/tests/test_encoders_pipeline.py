from __future__ import annotations

import pandas as pd

from vassoura.preprocessing import (
    WOEGuard,
    OneHotLite,
    SampleManager,
    make_default_pipeline,
)


def _make_df() -> tuple[pd.DataFrame, pd.Series]:
    cat = ["a"] * 50 + ["b"] * 50 + ["c"] * 50 + ["d"] * 50
    y = (
        [0] * 45
        + [1] * 5
        + [0] * 40
        + [1] * 10
        + [0] * 35
        + [1] * 15
        + [0] * 30
        + [1] * 20
    )
    df = pd.DataFrame({"cat": cat, "num": range(200)})
    return df, pd.Series(y)


def test_woe_values_monotonic():
    X, y = _make_df()
    enc = WOEGuard(min_samples=1)
    enc.fit(X[["cat"]], y)
    mapping = enc.woe_dict_["cat"]
    woe_vals = [mapping[v] for v in ["a", "b", "c", "d"]]
    assert all(x > y for x, y in zip(woe_vals[:-1], woe_vals[1:]))


def test_iv_positive():
    X, y = _make_df()
    enc = WOEGuard(min_samples=1)
    enc.fit(X[["cat"]], y)
    assert (enc.iv_ >= 0).all()


def test_make_default_pipeline_shapes():
    X, y = _make_df()
    pipe = make_default_pipeline(num_cols=["num"], cat_cols=["cat"])
    Xt = pipe.fit_transform(X, y)
    assert Xt.shape[0] == len(X)


def test_inference_auto_cols():
    X, y = _make_df()
    pipe = make_default_pipeline()
    Xt = pipe.fit_transform(X, y)
    assert Xt.shape[0] == len(X)


def test_encoder_switching():
    X, y = _make_df()
    pipe = make_default_pipeline(
        num_cols=["num"], cat_cols=["cat"], encoder="onehot"
    )
    pipe.fit(X, y)
    ct = pipe.named_steps["ct"]
    assert isinstance(ct.named_transformers_["cat"], OneHotLite)


def test_sampler_integration():
    X, y = _make_df()
    sm = SampleManager(strategy="stratified", frac=0.5, random_state=0)
    pipe = make_default_pipeline(
        num_cols=["num"], cat_cols=["cat"], sampler=sm
    )
    Xt = pipe.fit_transform(X, y)
    assert Xt.shape[0] == sm.sample_size_


def test_logging_encoder_counts(caplog):
    X, y = _make_df()
    enc = WOEGuard(min_samples=1, verbose=1)
    with caplog.at_level("INFO"):
        enc.fit(X[["cat"]], y)
    msgs = [rec.message for rec in caplog.records]
    assert any("fitted" in m for m in msgs)
