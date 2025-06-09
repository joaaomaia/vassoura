# correlacao module

Correlation matrix calculation and heat map drawing.

## Functions

### `compute_corr_matrix(df, method='auto', target_col=None, include_target=False, engine='pandas', limite_categorico=50, force_categorical=None, remove_ids=False, id_patterns=None, verbose=True, cramer=False)`
Return a correlation matrix between features using Pearson or Spearman and,
optionally, Cramér‑V for categorical variables when `cramer=True`.

### `plot_corr_heatmap(corr, title=None, annot=False, fmt='.2f', cmap='coolwarm', mask_upper=True, base_figsize=0.45, min_size=6, max_size=20, ax=None)`
Display a Seaborn heat map from the provided correlation matrix.
