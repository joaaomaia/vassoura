# analisador module

Provide simple textual recommendations based on the aggregated ACF from
`compute_panel_acf`.

### `analisar_autocorrelacao(panel_acf, feature_name, threshold_baixo=0.1, threshold_moderado=0.3, threshold_alto=0.6, verbose=True)`
Return a dictionary containing the feature name, the maximum absolute ACF,
its lag and a suggested action.
