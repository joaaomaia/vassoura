# core module

The `Vassoura` class orchestrates the cleaning steps in a stateful
session.

```python
from vassoura import Vassoura
```

## Main methods

### `run(recompute=False)`
Execute the configured heuristics in sequence and return the cleaned dataframe.

### `generate_report(path='vassoura_report.html')`
Write an HTML report based on the current internal state.

### `remove_additional(columns)`
Manually drop extra columns after running the heuristics.

### `reset()`
Restore the object to its initial state clearing caches and history.

Properties `history` and `dropped` expose the cleaning trail.
