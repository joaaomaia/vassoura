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

---

## ✨ Principais Funcionalidades

| Módulo                  | O que faz                                                 | Highlights                                                                                                                         |
| ----------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **`Vassoura` (classe)** | Sessão *stateful* que concentra todo o fluxo de limpeza   | cache inteligente, logs granulares, suporte a `id_cols`, `date_cols`, `ignore_cols`, remoção fracionada (`n_steps`, `vif_n_steps`) |
| **Correlação**          | `compute_corr_matrix`, `plot_corr_heatmap`                | Pearson, Spearman, Cramér‑V; amostragem adaptativa; *heat‑maps* auto‑dimensionados                                                 |
| **VIF**                 | `compute_vif`, `remove_high_vif`                          | Algoritmo NumPy puro (fallback) ou Statsmodels; compatível com `polars` e `dask`                                                   |
| **Heurísticas Plug‑in** | `corr`, `vif`, `iv`, `importance`, `graph_cut`, `missing`, `variance` | escolha via parâmetro; **graph‑cut** resolve correlações complexas em grafo mínimo                                                 |
| **Relatórios**          | `generate_report`                                         | HTML ou Markdown com seções ilustradas, imagens embutidas, lista de variáveis removidas                                            |
| **Autocorrelação**      | `compute_panel_acf`, `analisar_autocorrelacao`            | ACF por painel (contrato, cliente, etc.) com interpretação automática                                                              |

Além disso:

* **Adaptive Sampling**: para *datasets* gigantes, amostra até 50 k linhas sem comprometer tendências.
* **Dynamic Engine**: basta passar `engine="polars"` ou `"dask"` que o pipeline inteiro muda de engrenagem.
* **Suporte total a IDs & Datas**: informe `id_cols` e `date_cols` na instância e elas ficarão protegidas do processo de limpeza e ordenação.

### Heurísticas avançadas

As seguintes heurísticas extras podem ser combinadas livremente no parâmetro `heuristics`:

* `psi_stability` – calcula o Population Stability Index para duas janelas temporais.
* `ks_separation` – remove variáveis com baixo poder de separação pelo KS-statistic.
* `perm_importance` – ranking rápido via LightGBM e permutação.
* `partial_corr_cluster` – clusterização por correlação parcial com corte mínimo em grafo.
* `drift_leak` – identifica vazamentos de informação relacionados à data de referência.

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
    target_col="ever90m12",
    id_cols=["cpf"],             # preserva ordenação por CPF
    date_cols=["AnoMesReferencia"],
    heuristics=["corr", "vif", "iv", "graph_cut", "variance"],
    thresholds={"corr": 0.9, "vif": 8, "iv": 0.02, "variance": 1e-4},
    engine="polars",
    n_steps=3,                   # 3 rodadas para correlação
    vif_n_steps=1,
)

# 1️⃣ Limpeza
df_clean = vs.run()

# 2️⃣ Relatório interativo
vs.generate_report("relatorio_corr.html")
```

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