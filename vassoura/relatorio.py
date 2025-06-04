from __future__ import annotations
"""Vassoura – Geração de relatórios HTML/Markdown
==============================================

Gera relatórios interativos contendo:

* Lista de variáveis numéricas/categóricas detectadas
* Heat‑maps de correlação (com e sem *target*)
* Tabela de VIF e resumo de multicolinearidade

A função principal ``generate_report`` grava um arquivo (HTML ou MD)
com imagens embutidas (base64) e devolve o caminho do arquivo criado.
"""
import base64
import io
import logging
import textwrap
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .correlacao import compute_corr_matrix, plot_corr_heatmap
from .utils import search_dtypes, figsize_from_matrix
from .vif import compute_vif

__all__ = ["generate_report"]

LOGGER = logging.getLogger("vassoura")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fig_to_base64(fig: plt.Figure, *, fmt: str = "png") -> str:
    """Converte uma figura Matplotlib em *string* base64 embutível."""
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    return f"data:image/{fmt};base64,{img_b64}"


def _df_to_html(df: pd.DataFrame, *, float_fmt: str = ".3f", index: bool = False) -> str:
    return df.to_html(classes="table table-striped", border=0, index=index, float_format=lambda x: format(x, float_fmt))

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def generate_report(
    df: pd.DataFrame,
    *,
    output_path: str | Path = "vassoura_report.html",
    target_col: str | None = None,
    corr_method: str = "auto",
    corr_threshold: float = 0.9,
    vif_threshold: float = 10.0,
    limite_categorico: int = 50,
    force_categorical: Optional[List[str]] = None,
    remove_ids: bool = False,
    id_patterns: Optional[List[str]] = None,
    verbose: bool = True,
    style: str = "html",  # "html" | "md"
) -> str:
    """Gera relatório de correlação/multicolinearidade e grava em *output_path*.

    Parameters
    ----------
    df : pandas.DataFrame
    output_path : str | Path
        Caminho onde gravar o arquivo gerado.
    target_col : str | None
    corr_method, corr_threshold, vif_threshold
        Parâmetros de análise.
    style : {"html", "md"}
        Formato do relatório.

    Returns
    -------
    str
        Caminho final gravado (string).
    """
    output_path = Path(output_path)

    # Detecta tipos de colunas
    num_cols, cat_cols = search_dtypes(
        df,
        target_col=target_col,
        limite_categorico=limite_categorico,
        force_categorical=force_categorical,
        remove_ids=remove_ids,
        id_patterns=id_patterns,
        verbose=verbose,
    )

    # Correlation sem target
    corr_no_target = compute_corr_matrix(
        df,
        method=corr_method,
        target_col=target_col,
        include_target=False,
        limite_categorico=limite_categorico,
        force_categorical=force_categorical,
        remove_ids=remove_ids,
        id_patterns=id_patterns,
        verbose=verbose,
    )
    fig_nt, ax_nt = plt.subplots(figsize=figsize_from_matrix(len(corr_no_target)))
    plot_corr_heatmap(corr_no_target, title="Matriz de Correlação (sem target)", ax=ax_nt)
    img_corr_nt = _fig_to_base64(fig_nt)

    # Correlation com target (se existir)
    img_corr_wt = ""
    if target_col and target_col in df.columns:
        corr_w_target = compute_corr_matrix(
            df,
            method=corr_method,
            target_col=target_col,
            include_target=True,
            limite_categorico=limite_categorico,
            force_categorical=force_categorical,
            remove_ids=remove_ids,
            id_patterns=id_patterns,
            verbose=verbose,
        )
        fig_wt, ax_wt = plt.subplots(figsize=figsize_from_matrix(len(corr_w_target)))
        plot_corr_heatmap(corr_w_target, title="Matriz de Correlação (com target)", ax=ax_wt)
        img_corr_wt = _fig_to_base64(fig_wt)

    # VIF
    vif_df = compute_vif(
        df,
        target_col=target_col,
        include_target=False,
        limite_categorico=limite_categorico,
        force_categorical=force_categorical,
        remove_ids=remove_ids,
        id_patterns=id_patterns,
        verbose=verbose,
    )

    # HTML ou Markdown
    num_ul = "\n".join(f"<li>{c}</li>" for c in num_cols) or "<i>nenhuma</i>"
    cat_ul = "\n".join(f"<li>{c}</li>" for c in cat_cols) or "<i>nenhuma</i>"

    vif_html_table = _df_to_html(vif_df, float_fmt=".2f")

    if style == "html":
        html = textwrap.dedent(
            f"""
            <!DOCTYPE html>
            <html lang=\"pt-BR\">
            <head>
              <meta charset=\"utf-8\">
              <title>Relatório Vassoura</title>
              <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #023059; }}
                .table {{ border-collapse: collapse; width: 100%; }}
                .table th, .table td {{ padding: 6px 8px; border: 1px solid #ddd; text-align: right; }}
                .table th {{ background-color: #f5f5f5; }}
                img {{ max-width: 100%; height: auto; }}
              </style>
            </head>
            <body>
              <h1>Relatório de Correlação & Multicolinearidade – Vassoura</h1>
              <h2>Resumo dos tipos de variáveis</h2>
              <h3>Numéricas ({len(num_cols)})</h3>
              <ul>{num_ul}</ul>
              <h3>Categóricas ({len(cat_cols)})</h3>
              <ul>{cat_ul}</ul>

              <h2>Matrizes de Correlação</h2>
              <h3>Sem Target</h3>
              <img src=\"{img_corr_nt}\" alt=\"correlacao_sem_target\" />
        """
        )
        if img_corr_wt:
            html += textwrap.dedent(
                f"""
                <h3>Com Target</h3>
                <img src=\"{img_corr_wt}\" alt=\"correlacao_com_target\" />
                """
            )

        html += textwrap.dedent(
            f"""
              <h2>Variance Inflation Factor (VIF)</h2>
              {vif_html_table}
            </body></html>
            """
        )
        output_path.write_text(html, encoding="utf-8")
    else:  # Markdown minimal (sem imagens embed)
        md = textwrap.dedent(
            f"""
            # Relatório de Correlação & Multicolinearidade – Vassoura

            ## Variáveis Numéricas ({len(num_cols)})
            {', '.join(num_cols) or '*nenhuma*'}

            ## Variáveis Categóricas ({len(cat_cols)})
            {', '.join(cat_cols) or '*nenhuma*'}

            ## VIF (top 10)
            {vif_df.head(10).to_markdown(index=False)}
            """
        )
        output_path.write_text(md, encoding="utf-8")

    if verbose:
        LOGGER.info("Relatório gerado em %s", output_path)
    return str(output_path)
