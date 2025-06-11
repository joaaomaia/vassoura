# limpeza module

High level pipeline combining correlation filtering and VIF pruning.

### `clean(df, target_col=None, include_target=False, corr_threshold=0.9, corr_method='auto', vif_threshold=10.0, keep_cols=None, limite_categorico=50, force_categorical=None, remove_ids=False, id_patterns=None, max_vif_iter=20, n_steps=None, vif_n_steps=1, verbose=True)`
Return a tuple `(df_clean, dropped_cols, corr_matrix_final, vif_final)`.
