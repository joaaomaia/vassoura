import time
import pandas as pd

import vassoura as vs


def _df():
    return pd.DataFrame({'x': [1, 2, 3], 'target': [0, 1, 0]})


def test_heuristic_completes_before_timeout(monkeypatch):
    sess = vs.Vassoura(
        _df(),
        target_col='target',
        heuristics=['slow'],
        process=['scaler'],
        timeout_map={'slow': 1},
        max_total_runtime=None,
        verbose='none',
    )
    sess._heuristic_funcs['slow'] = lambda: time.sleep(0.1)
    sess.run()
    assert sess.exec_log[-1]['status'] == 'success'


def test_heuristic_hits_timeout(monkeypatch):
    sess = vs.Vassoura(
        _df(),
        target_col='target',
        heuristics=['slow'],
        process=['scaler'],
        timeout_map={'slow': 0.05},
        verbose='none',
    )
    sess._heuristic_funcs['slow'] = lambda: time.sleep(0.2)
    sess.run()
    assert sess.exec_log[-1]['status'] == 'timeout'


def test_max_total_runtime(monkeypatch):
    sess = vs.Vassoura(
        _df(),
        target_col='target',
        heuristics=['a', 'b', 'c'],
        process=['scaler'],
        timeout_map={'a': 1, 'b': 1, 'c': 1},
        max_total_runtime=0.35,
        verbose='none',
    )
    slow = lambda: time.sleep(0.2)
    sess._heuristic_funcs['a'] = slow
    sess._heuristic_funcs['b'] = slow
    sess._heuristic_funcs['c'] = slow
    sess.run()
    assert len(sess.exec_log) == 2
