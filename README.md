# Vassoura

*Limpeza de correlação, multicolinearidade e autocorrelação para pandas DataFrames*

![logo](vassoura/imgs/social_preview_github.png)

> **Vassoura** ajuda a **varrer** variáveis redundantes e dependências temporais do seu dataset. Ele detecta correlações fortes, calcula VIF, analisa autocorrelação em painel, gera heat‑maps e plots de VIF, e produz relatórios HTML/Markdown prontos para anexar em documentos técnicos.

---

## 🧩 Visão Geral

`Vassoura` é uma biblioteca Python voltada para análises de correlação e multicolinearidade em dados tabulares e de séries temporais em painel, ideal para modelagem de Risco de Crédito e outras aplicações financeiras. Com ele você pode:

* **Classificar tipos de colunas** (*numéricas*, *categóricas*, *IDs*);
* **Calcular correlação** (Pearson, Spearman, Cramér‑V) e visualizar via heat‑map dinamicamente dimensionado;
* **Calcular VIF** (Variance Inflation Factor) usando `statsmodels` ou NumPy, e remover variáveis iterativamente conforme limiar;
* **Descartar colunas com muitos valores ausentes** ao definir `missing_threshold` na classe `Vassoura`;
* **Limpar multicolinearidade** combinando filtro por correlação e VIF em um único pipeline (`clean`);
* **Analisar autocorrelação em painel** para séries temporais por contrato, agregando ACF (ACF médio, mediana, ponderado) e exibindo correlogramas;
* **Gerar relatórios** HTML ou Markdown completos com seções de conceitos, heat‑maps, plots de VIF e autocorrelação, além de listas de variáveis removidas.


🧹 Como a remoção de correlação funciona (padrão)

Identificamos pares fortemente correlacionados (|corr| ≥ corr_threshold, padrão 0.9).

Para cada par (feat_1, feat_2) calculamos, para cada variável, a mediana das correlações absolutas com todas as demais colunas (excluindo a autocorrelação 1.00).

A variável com maior mediana tende a ser mais “redundante” no conjunto e é a candidata natural para descarte.

Prioridades absolutas: se a variável estiver em keep_cols (lista de features prioritárias), ela jamais é removida – mesmo que tenha maior mediana.

Resumo: “Remove quem cola mais com todo mundo, mas nunca toca nas prioridades.”

Você pode alterar a métrica – metric={"median","mean","max"} – ou ponderar correlação com colunas protegidas (weight_keep), mas a mediana respeitando keep_cols é o default.

---

## ⚙️ Instalação

```bash
# Instalar versão estável do PyPI (quando publicado)
pip install vassoura

# Para instalar do fonte (modo desenvolvimento):
git clone https://github.com/SEU_USUARIO/vassoura.git
cd vassoura
pip install -e .[dev]
```

> **Requisitos**: Python ≥ 3.9, pandas, numpy, seaborn, matplotlib, scipy, statsmodels.

---

## 📦 Estrutura de Pacote

```text
vassoura/                  # Código‑fonte
├── __init__.py            # API pública
├── utils.py               # Funções utilitárias (search_dtypes, suggest_corr_method, figsize)
├── correlacao.py          # compute_corr_matrix, plot_corr_heatmap
├── vif.py                 # compute_vif, remove_high_vif
├── limpeza.py             # clean
├── relatorio.py           # generate_report
├── autocorrelacao.py      # compute_panel_acf, plot_panel_acf
└── analisador.py          # analisar_autocorrelacao

examples/                  # Notebooks de exemplo
└── exemplo_uso.ipynb
└── exemplo2_autocorr.ipynb

tests/                     # Testes unitários pytest
│  └── test_core.py
│  └── test_autocorrelacao.py

README.md                  # Este arquivo
pyproject.toml             # Configuração do projeto
.gitignore                 # Arquivos ignorados no Git
```

---

## 🚀 Principais Funcionalidades

### 1. Detecção de Tipos (`search_dtypes`)

* Classifica *colunas numéricas*, *categóricas*, *booleans* e ignora *IDs* ou *datetime*.
* Parâmetros: `target_col`, `limite_categorico`, `force_categorical`, `remove_ids`, `id_patterns`, `date_col`, `verbose_types`.

### 2. Correlação (`compute_corr_matrix`, `plot_corr_heatmap`)

* **Métodos**: `pearson`, `spearman`, `cramer`, ou `auto` (decide com base nos tipos).
* Gera **DataFrame** de correlação e **heat‑map** Seaborn com dimensionamento automático (anotações opcionais).
* Pode utilizar `engine="dask"` ou `engine="polars"` para grandes DataFrames.
* Caso o método escolhido não seja suportado pelo engine, a função faz fallback
  para pandas e registra esse fato no log.

### 3. VIF (`compute_vif`, `remove_high_vif`)

* Calcula **Variance Inflation Factor** para variáveis numéricas. Usa `statsmodels` se disponível, ou *fallback* NumPy.
* Remove iterativamente variáveis que excedem o limiar `vif_threshold`, preservando colunas-chave (`keep_cols`).
* **Suporte opcional a Dask/Polars** passando `engine="dask"` ou `engine="polars"`.
* Linhas com valores NaN ou infinitos são descartadas antes do cálculo de VIF.
* Heurísticas extras: `importance` (XGBoost/SHAP) e `graph_cut` para correlações complexas.

### 4. Limpeza Combinada (`clean`)

* Pipeline de 2 passos:  **(a)** filtro por correlação (`corr_threshold`) → remove pares com |corr|>limiar.
  **(b)** filtro por VIF (`vif_threshold`) → remove iterativamente usando VIF.
* Parâmetros `keep_cols`, `target_col`, `include_target`, `limite_categorico`, `force_categorical`, etc.
* Retorna: `(df_limpo, colunas_removidas, corr_matrix_final, vif_final)`.

### 5. Autocorrelação em Painel (`compute_panel_acf`, `plot_panel_acf`)

* Calcula **ACF por contrato** (identificado por `id_col`) reindexando meses faltantes (`time_col` formatado como `YYYYMM`).
* Ignora contratos com menos de `min_periods` meses.
* Agrega ACF via: média (`mean`), mediana (`median`) ou ponderada (`weighted`,
  usando o comprimento da série como peso).
* Gera gráfico de barras horizontais com rótulos e linhas de confiança.

### 6. Analisador de Autocorrelação (`analisar_autocorrelacao`)

* Recebe resultado `panel_acf` e avalia maior ACF (`lag_max`, `acf_max`).
* Classifica em níveis: `ruido`, `leve`, `moderada`, `alta`.
* Recomenda ações: ignorar, incluir lag, usar rolling, ou modelos temporais.

### 7. Relatórios (`generate_report`)

* Gera **HTML** completo com:

  * Seção “Conceitos” sobre correlação e multicolinearidade.
  * Tipos de variáveis listadas.
  * **Heatmaps** antes/apos limpeza de correlação.
  * **Plots de VIF** antes/apos limpeza.
  * Seção **autocorrelação** (opcional) se time\_col/id\_col informados.
  * Lista de variáveis removidas.
* Também suporta **Markdown** resumido (sem imagens, apenas tabelas).

---

## 💡 Exemplo de Uso Básico

```python
import pandas as pd
import vassoura as vs

# 1. Carregar ou simular dataset
df = pd.read_csv("dados.csv")
# ou simular:
df = vs.criar_dataset_pd_behavior(n_clientes=1000, anos=3)

# 2. Pipeline de limpeza de multicolinearidade
df_limpo, removidas, corr_final, vif_final = vs.clean(
    df,
    target_col="ever90m12",
    keep_cols=["idade", "renda"],
    corr_threshold=0.9,
    vif_threshold=10,
)
print("Removidas:", removidas)

# 3. Análise de autocorrelação para 'feature_01'
panel_acf = vs.compute_panel_acf(
    df,
    value_col="feature_01",
    time_col="AnoMesReferencia",
    id_col="NroContrato",
    nlags=12,
    min_periods=12,
    agg_method="mean"
)
vs.plot_panel_acf(panel_acf, title="Autocorrelação média: feature_01")

# 4. Geração de relatório final
vs.generate_report(
    df,
    output_path="vassoura_report.html",
    target_col="ever90m12",
    corr_threshold=0.9,
    vif_threshold=10,
    keep_cols=["idade", "renda"],
)
```

---

## 📖 Documentação e Exemplos

* Veja notebooks em `examples/`:

  * **`exemplo_uso.ipynb`**: limpeza de correlação & VIF.
  * **`exemplo2_autocorr.ipynb`**: análise de autocorrelação em painel.
* Futuramente: documentação completa em ReadTheDocs.

---

## 🤝 Contribuindo

1. **Fork** do repositório
2. **Clone** local e crie branch: `git checkout -b minha-feature`
3. Instale as dependências de desenvolvimento: `pip install -e .[dev]`
4. **Desenvolva** código seguindo PEP 8 e padrões de commit (conventional commits).
5. **Teste** com `pytest -q` e `flake8`.
6. **Abra** Pull Request detalhando alterações.

Recomendações:

* Utilize `pre-commit` (Black + isort + flake8).
* Escreva testes unitários em `tests/`.

---

## 📄 Licença

Este projeto é licenciado sob a **MIT License**. Veja o arquivo [LICENSE](LICENSE) para detalhes.
