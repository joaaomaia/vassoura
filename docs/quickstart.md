# Quickstart

```python
from vassoura.preprocessing import SampleManager, DynamicScaler
from vassoura import Vassoura

# assume df is your dataset with a 'target' column
sm = SampleManager(strategy="auto")
df_s, y_s = sm.fit_resample(df.drop('target', axis=1), df['target'])
sc = DynamicScaler(strategy="auto")
df_t = sc.fit_transform(df_s)

v = Vassoura(target_col='target')
v.fit(df)
```
