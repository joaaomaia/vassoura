# relatorio module

Tools to produce a standalone HTML or Markdown report showing correlation
heatmaps, VIF tables and the list of removed features.

### `generate_report(df, output_path='vassoura_report.html', target_col=None, corr_method='auto', corr_threshold=0.9, vif_threshold=10.0, keep_cols=None, limite_categorico=50, force_categorical=None, remove_ids=False, id_patterns=None, max_vif_iter=20, n_steps=None, vif_n_steps=1, heatmap_labels=True, heatmap_base_size=0.6, verbose=True, style='html')`
Return the path of the generated file.
