# heuristics module

Optional feature selection helpers that can be plugged into the `Vassoura`
workflow. They operate independently of the main class.

### `iv(df, target_col, threshold=0.02, bins=10, keep_cols=None)`
Remove features with Information Value below the threshold.

### `importance(df, target_col, n_estimators=100, learning_rate=0.1, subsample=0.8, keep_cols=None, drop_lowest=0.2, random_state=42)`
Train a light XGBoost model and drop the lowest scoring features using SHAP gain.
Categorical columns are automatically encoded via a lightweight WOE scheme.
Requires `xgboost` and `shap` installed.

### `graph_cut(df, corr_threshold=0.9, keep_cols=None, method='pearson', target_col=None)`
Build a correlation graph and compute a minimal vertex cover to determine
which variables to drop. Categorical columns are temporarily WOE encoded
(missing values treated as their own category) when a binary `target_col`
is provided.

### `variance(df, var_threshold=1e-4, dom_threshold=0.95, min_nonnull=30, keep_cols=None)`
Drop features with very low numerical variance or with a single dominant
category.

### `psi_stability(df, date_col, window, target_col=None, bins=10, psi_thr=0.25, keep_cols=None)`
Calculate the Population Stability Index comparing two time windows.

### `ks_separation(df, target_col, ks_thr=0.05, n_bins=10, keep_cols=None)`
Remove columns with KS-statistic below the threshold.

### `perm_importance_lgbm(df, target_col, metric='auc', n_estimators=300, drop_lowest=0.2, random_state=42, keep_cols=None)`
Drop the least important features according to LightGBM permutation importance.

### `partial_corr_cluster(df, corr_thr=0.6, keep_cols=None, method='pearson')`
Build a graph using partial correlations and remove a minimal vertex cover.

### `drift_vs_target_leakage(df, date_col, target_col, drift_thr=0.3, leak_thr=0.5, keep_cols=None)`
Detect features highly correlated with both the date column and the target (potential leakage).

### `target_leakage(df, target_col, threshold=0.8, method='spearman', keep_cols=None, id_cols=None, date_cols=None)`
Highlight columns with very high absolute correlation with the target, indicating possible data leakage.
