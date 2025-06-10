# Vassoura

*Limpeza de correlação, multicolinearidade e autocorrelação para **DataFrames** gigantes em **pandas**, **polars** e **dask***

![logo](vassoura/imgs/social_preview_github.png)

> **Vassoura** ajuda a **varrer** variáveis redundantes, multicolinearidade e dependências temporais, entregando *dataframes* enxutos e relatórios prontos para anexar em documentos técnicos.

---

## ⚡️ Instalação

```bash
# Versão estável (PyPI)
pip install vassoura

# Versão de desenvolvimento (recomendado para contribuir)
git clone https://github.com/SEU_USUARIO/vassoura.git
cd vassoura
pip install -e .[dev]
```

**Requisitos mínimos**: Python ≥ 3.9, `numpy`, `pandas`.

**Back‑ends opcionais** (instalação automática se presentes):

| Engine   | Benefício                                          | Instalação                   |
| -------- | -------------------------------------------------- | ---------------------------- |
| `polars` | Cálculo de correlação/VIF em *Rust* — muito rápido | `pip install polars`         |
| `dask`   | Processamento *out‑of‑core* em clusters            | `pip install dask[complete]` |

Caso esses pacotes não estejam disponíveis, o Vassoura faz *fallback* elegante para `pandas`.

Para configurar os logs de todo o pacote basta executar::

    import vassoura
    vassoura.configure_logging()

---

## ✨ Principais Funcionalidades

| Módulo                  | O que faz                                                 | Highlights                                                                                                                         |
| ----------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **`Vassoura` (classe)** | Sessão *stateful* que concentra todo o fluxo de limpeza   | cache inteligente, logs granulares, suporte a `id_cols`, `date_cols`, `ignore_cols`, remoção fracionada (`n_steps`, `vif_n_steps`) |
| **Correlação**          | `compute_corr_matrix`, `plot_corr_heatmap`                | Pearson ou Spearman com codificação WoE temporária; *heat‑maps* auto‑dimensionados                                                 |
| **VIF**                 | `compute_vif`, `remove_high_vif`                          | Algoritmo NumPy puro (fallback) ou Statsmodels; compatível com `polars` e `dask`                                                   |
| **Heurísticas Plug‑in** | `corr`, `vif`, `iv`, `importance`, `graph_cut`, `missing`, `variance` | escolha via parâmetro; **graph‑cut** resolve correlações complexas em grafo mínimo com WoE temporário |
| **Relatórios**          | `generate_report`                                         | HTML ou Markdown com seções ilustradas, imagens embutidas, lista de variáveis removidas                                            |
| **Autocorrelação**      | `compute_panel_acf`, `analisar_autocorrelacao`            | ACF por painel (contrato, cliente, etc.) com interpretação automática                                                              |

Além disso:

* **Adaptive Sampling**: amostragem estratificada (por `target_col`) até ~50 k linhas,
  preservando a ordem temporal caso `date_cols` seja informado. Pode ser aplicada
  como processo (`"adaptive_sampling"`) para reutilizar a mesma amostra em todo o
  pipeline. O resultado de `run()` sempre mantém todas as linhas originais do dataset.
* **Dynamic Engine**: basta passar `engine="polars"` ou `"dask"` que o pipeline inteiro muda de engrenagem.
* **Suporte total a IDs & Datas**: informe `id_cols` e `date_cols` na instância e elas ficarão protegidas do processo de limpeza e ordenação.

### Heurísticas avançadas

As seguintes heurísticas extras podem ser combinadas via parâmetro `heuristics`:

* `psi_stability` – avalia a estabilidade de distribuição (PSI) entre janelas temporais.
* `ks_separation` – descarta variáveis com baixo poder discriminatório medido pelo KS.
* `perm_importance` – ordena features usando LightGBM e permutação aleatória.
* `partial_corr_cluster` – remove grupos redundantes via correlação parcial e grafo mínimo.
* `drift_leak` – destaca atributos ligados à data e ao target, sinalizando vazamentos.
* `target_leakage` – identifica colunas fortemente correlacionadas ao target (possível vazamento).
* `boruta_multi_shap` – seleção robusta combinando Boruta, vários modelos e SHAP.

---

## 📚 Documentação

A documentação detalhada (API, tutoriais, FAQ) mora na pasta [`docs/`](docs) e será publicada em breve no **Read the Docs**.

*Enquanto isso → execute `help(Vassoura)` ou consulte as *docstrings* dos módulos para detalhes completos.*

---

## 🚀 Exemplos Rápidos

### 1. Pipeline completo em três linhas

```python
import pandas as pd

from vassoura import Vassoura

# dataset fictício
df = pd.read_csv("dados.csv")

vs = Vassoura(
    df,
    id_cols=["id", "member_id"],
    date_cols=["safra"],
    ignore_cols=["url"] + temporal_columns,
    drop_ignored=True,
    target_col=TARGET,
    verbose="basic",
    engine="pandas",
    adaptive_sampling=True,
    process=["missing", "variance", "scaler"],
    heuristics=[
        "target_leakage",
        "iv",
        "graph_cut",
        "corr",
        "vif",
        "importance",
        "boruta_multi_shap",
        "ks_separation",
    ],
    params={
        "missing": 0.60,
        "target_leakage": 0.70,
        "corr": 0.80,
        "vif": 10,
        "iv": 0.01,
        "graph_cut": 0.9,
    },
    n_steps=5,
    vif_n_steps=2,
    timeout_map={"importance": 90, "graph_cut": 60},
    chunk_size=25,
    max_total_runtime=600,
)

df_clean = vs.run_all()

vs.generate_report("relatorio_corr.html")
```

`timeout_map` define limites (em segundos) por heurística. Caso o passo
extrapole o tempo, ele é pulado sem interromper o fluxo. Já `max_total_runtime`
encerra o `run()` quando o orçamento global é excedido.

### 2. Limpeza funcional (atalho)

```python
from vassoura import clean

df_clean, dropped, corr_fin, vif_fin = clean(
    df,
    target_col="target",
    corr_threshold=0.85,
    vif_threshold=10,
    keep_cols=["idade", "renda"],
)
```

Para mais exemplos, veja [`examples/`](examples).

---

## 🗂️ Estrutura de Pacote (v0.5+)

```text
vassoura/
├── __init__.py          # API pública (clean, compute_corr_matrix, ...)
├── core.py              # classe Vassoura (orquestra tudo)
├── correlacao.py        # correlação & heat‑maps
├── vif.py               # VIF helpers
├── heuristics.py        # heurísticas plug‑in (corr, vif, iv, ...)
├── limpeza.py           # wrapper procedural clean()
├── relatorio.py         # geração de relatórios
└── utils.py             # misc helpers
```

---

## 🤝 Contribuindo

1. **Fork** este repositório e crie sua *feature branch*: `git checkout -b feat/minha‑feature`.
2. Garanta que todos os testes `pytest` passem: `pytest -q`.
3. Abra um *pull request* descrevendo sua motivação.

Sugerimos executar `pre‑commit install` para aderir ao nosso *style guide* automaticamente.

---

## 📜 Licença

Este projeto é distribuído sob a licença MIT – veja o arquivo [LICENSE](LICENSE) para detalhes.

> Este projeto também respeita as licenças de todas as bibliotecas utilizadas como dependência.
> Consulte o arquivo [NOTICE.md](NOTICE.md) para a lista completa de licenças de terceiros.

---

> Feito com ☕ + 🧹 por contribuidores da comunidade de *data science* brasileira.