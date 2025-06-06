# autocorrelacao module

Functions to evaluate autocorrelation in panel data.

### `compute_panel_acf(df, value_col, time_col, id_col, nlags=12, min_periods=12, agg_method='mean', verbose=False)`
Return a dataframe with aggregated ACF per lag.

### `plot_panel_acf(panel_acf, title=None, conf_level=0.95)`
Plot a bar chart showing the aggregated ACF values and confidence lines.
