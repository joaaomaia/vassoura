# Vassoura
[![CI](https://github.com/example/vassoura/actions/workflows/python-ci.yml/badge.svg)](https://github.com/example/vassoura/actions/workflows/python-ci.yml) [![codecov](https://codecov.io/gh/example/vassoura/branch/main/graph/badge.svg)](https://codecov.io/gh/example/vassoura) [![PyPI](https://img.shields.io/pypi/v/vassoura.svg)](https://pypi.org/project/vassoura/)
Vassoura é um framework para seleção e auditoria de variáveis...

## Quick start

```python
from vassoura.preprocessing import SampleManager

# X, y são seus dados originais
sm = SampleManager(strategy="auto")
X_sampled, y_sampled = sm.fit_resample(X, y)
```

### Scaling

```python
from vassoura.preprocessing import DynamicScaler

scaler = DynamicScaler(strategy="auto")
X_scaled = scaler.fit_transform(X_sampled)
```

### Training with Vassoura

```python
from vassoura import Vassoura

v = Vassoura(target_col="target")
v.fit(df)
preds = v.predict(df)
```


### Example – Quick modelling

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from vassoura.preprocessing import make_default_pipeline

pipeline = make_default_pipeline()
model = Pipeline([
    ("prep", pipeline),
    ("clf", LogisticRegression())
])
model.fit(X, y)
```

### Cross-validation

```python
from vassoura.validation import get_stratified_cv
from vassoura.utils.metrics import SCORERS

cv = get_stratified_cv(5)
scores = cross_validate(pipe, X, y, cv=cv, scoring=SCORERS)
```

## Model Zoo

Ready-to-use wrappers are available via `vassoura.models` registry:

```python
from vassoura.models import get, list_models

print(list_models())  # ['logistic_balanced', ...]
Model = get('logistic_balanced')
model = Model()
model.fit(X_train, y_train)
```

All wrappers automatically handle class imbalance using `sample_weight` when
supported by the underlying library.

## Importance heuristics

```python
from vassoura.process import basic_importance

imp = basic_importance(X_train, y_train, model="logistic", method="coef")
print(imp.head())
```

## Audit & Reports

```python
from vassoura import Vassoura, AuditTrail
from vassoura.report import ReportManager, SECTION_REGISTRY

df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
audit = AuditTrail(auto_detect_types=True)
audit.take_snapshot(df, "raw")

v = Vassoura(target_col="target", report=True)
v.fit(df)

rm = ReportManager()
rm.add_section(
    SECTION_REGISTRY["overview"](
        audit=audit,
        snapshot_names=list(audit.snapshots.keys()),
        dataset_shape=df.shape,
    )
)
rm.render("report.html")
```

## Contributing
Run the following once to enable pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

