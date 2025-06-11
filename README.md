# Vassoura
Vassoura é um framework para seleção e auditoria de variáveis...

## Quick start

```python
from vassoura.preprocessing import SampleManager

# X, y são seus dados originais
sm = SampleManager(strategy="auto")
X_sampled, y_sampled = sm.fit_resample(X, y)
```
