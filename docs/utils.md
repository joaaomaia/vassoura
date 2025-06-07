# utils module

Utility helpers used across the project.

## Functions

### `search_dtypes(df, target_col=None, limite_categorico=50, force_categorical=None, remove_ids=False, id_patterns=None, date_col=None, verbose=True, verbose_types=False)`
Classify dataframe columns into numeric and categorical. Explicit `date_col` allows forcing columns to `datetime` and `verbose_types` controls detailed logs.

### `suggest_corr_method(num_cols, cat_cols)`
Return `'pearson'`, `'spearman'` or `'cramer'` based on available columns.

### `figsize_from_matrix(n_features, base=0.4, min_size=6, max_size=20)`
Compute a square `figsize` adequate for correlation heatmaps.

### `criar_dataset_pd_behavior(n_clientes=1000, max_anos=5, n_features=20, seed=42)`
Generate a synthetic behavioural credit dataset useful for examples and
unit tests.
