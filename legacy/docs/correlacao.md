# correlacao module

Correlation matrix calculation and heat map drawing.

## Functions

### `compute_corr_matrix(df, method='auto', target_col=None, include_target=False, limite_categorico=50, force_categorical=None, remove_ids=False, id_patterns=None, verbose=True)`
Return a correlation matrix using Pearson or Spearman. Categorical features are
temporarily encoded via WoE when a binary `target_col` is provided, falling back
to integer codes otherwise.

### `plot_corr_heatmap(corr, title=None, annot=False, fmt='.2f', cmap='coolwarm', mask_upper=True, base_figsize=0.45, min_size=6, max_size=20, ax=None)`
Display a Seaborn heat map from the provided correlation matrix.
