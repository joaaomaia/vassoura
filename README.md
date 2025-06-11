# Vassoura
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
