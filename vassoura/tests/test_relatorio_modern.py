import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch

import vassoura as vs


def _make_df(n=120):
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=n)
    x2 = x1 * 0.9 + rng.normal(scale=0.1, size=n)
    x3 = rng.normal(size=n)
    target = (x1 + x3 + rng.normal(size=n)) > 0
    data = {"x1": x1, "x2": x2, "x3": x3, "target": target.astype(int)}
    for i in range(60):
        data[f"f{i}"] = rng.normal(size=n)
    return pd.DataFrame(data)


def test_generate_report_modern(tmp_path):
    df = _make_df()
    sess = vs.Vassoura(
        df, target_col="target", heuristics=["corr", "vif"], verbose="none"
    )
    sess.run(recompute=True)
    with (
        patch("lightgbm.LGBMClassifier") as MockLGBM,
        patch("shap.TreeExplainer") as MockExpl,
    ):
        MockLGBM.return_value.fit.return_value = MockLGBM.return_value
        MockExpl.return_value.shap_values.return_value = [
            np.zeros((len(df), 63)),
            np.zeros((len(df), 63)),
        ]
        path = Path(sess.generate_report(path=tmp_path / "r.html"))
        assert MockLGBM.call_args[1]["class_weight"] == "balanced"
    html = path.read_text()
    import re

    section = re.search(
        r'<div class="vif-grid">(.*?)</div>', html, flags=re.S
    )
    assert section
    imgs = re.findall(r'<img src="data:image/[^;]+;base64,([^" ]+)', section.group(1))
    assert len(imgs) == 2
    assert all(len(i) > 10000 for i in imgs)
    assert html.count('<div class="vif-grid">') == 1
    assert "KS Separation" not in html
    assert "flare_" in html
    assert "__shadow__" in sess.df_current.columns
