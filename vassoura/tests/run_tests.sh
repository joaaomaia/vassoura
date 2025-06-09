#!/bin/bash
mkdir -p test_results

# pytest -vv --tb=long vassoura/tests/test_advanced_heuristics.py                  > vassoura/tests/test_results/test_advanced_heuristics.txt
# pytest -vv --tb=long vassoura/tests/test_autocorr.py                             > vassoura/tests/test_results/test_autocorr.txt
# pytest -vv --tb=long vassoura/tests/test_core.py                                 > vassoura/tests/test_results/test_core.txt
# pytest -vv --tb=long vassoura/tests/test_correlacao.py                           > vassoura/tests/test_results/test_correlacao.txt
# pytest -vv --tb=long vassoura/tests/test_importance.py                           > vassoura/tests/test_results/test_importance.txt
# pytest -vv --tb=long vassoura/tests/test_leakage.py                                  > vassoura/tests/test_results/test_leakage.txt
# pytest -vv --tb=long vassoura/tests/test_limpeza.py                              > vassoura/tests/test_results/test_limpeza.txt
# pytest -vv --tb=long vassoura/tests/test_relatorio_modern.py                     > vassoura/tests/test_results/test_relatorio_modern.txt
# pytest -vv --tb=long vassoura/tests/test_scaler.py                               > vassoura/tests/test_results/test_scaler.txt
# pytest -vv --tb=long vassoura/tests/test_session.py                              > vassoura/tests/test_results/test_session.txt
# pytest -vv --tb=long vassoura/tests/test_special_cols.py                         > vassoura/tests/test_results/test_special_cols.txt
# pytest -vv --tb=long vassoura/tests/test_utils.py                                > vassoura/tests/test_results/test_utils.txt
# pytest -vv --tb=long vassoura/tests/test_variance.py                             > vassoura/tests/test_results/test_variance.txt
# pytest -vv --tb=long vassoura/tests/test_boruta_multi_shap.py                    > vassoura/tests/test_results/test_boruta_multi_shap.txt
# pytest -vv --tb=long vassoura/tests/test_vassoura_integration_super.py           > vassoura/tests/test_results/test_vassoura_integration_super.txt
pytest -vv --tb=long vassoura/tests/test_vif.py                                  > vassoura/tests/test_results/test_vif.txt
pytest -vv --tb=long vassoura/tests/test_graph_cut.py                                  > vassoura/tests/test_results/test_graph_cut.txt
pytest -vv --tb=long vassoura/tests/test_heuristics.py                                  > vassoura/tests/test_results/test_heuristics.txt
