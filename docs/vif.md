# vif module

Variance inflation factor utilities.

## Functions

### `compute_vif(df, target_col=None, include_target=False, engine='pandas', limite_categorico=50, force_categorical=None, remove_ids=False, id_patterns=None, verbose=True, use_woe=False)`
Return a dataframe with variables and their VIF values.
Categorical columns are automatically encoded. When `use_woe=True` and the target is binary, WOE encoding is applied before computing VIF.
Rows with NaN or infinite values are dropped automatically.

### `remove_high_vif(df, vif_threshold=10.0, target_col=None, include_target=False, keep_cols=None, max_iter=20, vif_n_steps=1, limite_categorico=50, force_categorical=None, remove_ids=False, id_patterns=None, engine='pandas', verbose=True)`
Iteratively drop features with VIF above the threshold and return the
clean dataframe, a list of removed columns and the final VIF table.
