# heuristics module

Optional feature selection helpers that can be plugged into the `Vassoura`
workflow. They operate independently of the main class.

### `iv(df, target_col, threshold=0.02, bins=10, keep_cols=None)`
Remove features with Information Value below the threshold.

### `importance(df, target_col, n_estimators=100, learning_rate=0.1, subsample=0.8, keep_cols=None, drop_lowest=0.2, random_state=42)`
Train a light XGBoost model and drop the lowest scoring features using SHAP gain.
Requires `xgboost` and `shap` installed.

### `graph_cut(df, corr_threshold=0.9, keep_cols=None, method='pearson')`
Build a correlation graph and compute a minimal vertex cover to determine
which variables to drop.

### `variance(df, var_threshold=1e-4, dom_threshold=0.95, min_nonnull=30, keep_cols=None)`
Drop features with very low numerical variance or with a single dominant
category.
