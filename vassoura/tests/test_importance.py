import numpy as np
import pandas as pd
from vassoura.core import Vassoura


def test_importance_shadow_filter():
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(size=n)
    target = (x1 + rng.normal(scale=0.3, size=n) > 0).astype(int)
    x_noise = rng.normal(size=n)
    df = pd.DataFrame({"x1": x1, "x_noise": x_noise, "target": target})

    params = {
        "importance": {
            "models": [
                {
                    "name": "lr",
                    "estimator": __import__(
                        "sklearn.linear_model"
                    ).linear_model.LogisticRegression(max_iter=200),
                }
            ],
            "cv_folds": 3,
            "cv_type": "stratified",
        }
    }
    vs = Vassoura(df, target_col="target", heuristics=["importance"], params=params)
    out = vs.run()
    assert vs._importance_meta.get("cv_folds") == 3
