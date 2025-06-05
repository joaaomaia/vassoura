# Vassoura

*Limpeza de correlação & multicolinearidade para DataFrames em Python*

![logo](imgs/social_preview_github.png)

> **Vassoura** ajuda a *varrer* variáveis redundantes do seu dataset. Ele
> detecta correlações fortes, calcula VIF, gera heat‑maps automáticos e
> produz relatórios HTML/Markdown prontos para anexar em documentos
> técnicos.

---

## Principais recursos

| Categoria             | Descrição                                                                                 |
| --------------------- | ----------------------------------------------------------------------------------------- |
| **Detecção de tipos** | `search_dtypes` identifica numéricas, categóricas, ID e ignoradas com *logging* detalhado |
| **Correlação**        | Pearson, Spearman e Cramér‑V (auto‑seleção) + heat‑maps *seaborn* com tamanho dinâmico    |
| **VIF**               | Cálculo via *statsmodels* (ou fallback NumPy) e remoção iterativa com limiar configurável |
| **Limpeza combinada** | Pipeline `clean_multicollinearity` (correlação + VIF) preservando colunas chave           |
| **Relatórios**        | `generate_report` cria HTML/Markdown com imagens embutidas, tipos, heat‑maps e VIF        |

---

## Instalação

```bash
pip install vassoura  # após publicação no PyPI
```

Para instalar a partir do fonte:

```bash
git clone https://github.com/SEU_USUARIO/vassoura.git
cd vassoura
pip install -e .[dev]
```

> **Requisitos**: Python ≥ 3.9, pandas, numpy, seaborn, matplotlib,
> scipy, statsmodels.

---

## Exemplo rápido

```python
import pandas as pd
import vassoura as vs

# Carregar dataset de exemplo
df = pd.read_csv("dados.csv")

# Limpar multicolinearidade
limpo, removidas, corr, vif = vs.clean_multicollinearity(
    df,
    target_col="target",
    keep_cols=["idade", "renda"],
    corr_threshold=0.85,
)
print("Variáveis removidas:", removidas)

# Relatório detalhado
vs.generate_report(df, target_col="target")
```

---

## Documentação

A documentação completa (em desenvolvimento) estará disponível em
[`docs/`](docs/) e futuramente no ReadTheDocs.

---

## Contribuindo

1. Faça um *fork* do repositório
2. Crie sua *feature branch*: `git checkout -b minha-feature`
3. Commit suas alterações: `git commit -m 'feat: Minha nova feature'`
4. Envie o *pull request* ❤️

Siga a [PEP 8](https://peps.python.org/pep-0008/) e utilize `pre-commit`
(`pip install pre-commit && pre-commit install`).

---

## Licença

Este projeto é licenciado sob os termos da **MIT License** – veja o
arquivo [LICENSE](LICENSE) para detalhes.