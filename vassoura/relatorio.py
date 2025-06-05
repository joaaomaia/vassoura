"""
Vassoura – Geração de relatórios HTML/Markdown
==============================================

Gera relatórios interativos contendo:
  * Notas explicativas sobre correlação e multicolinearidade
  * Heat-maps de correlação (antes e após limpeza) com e sem target
  * Gráficos de barras horizontais do VIF (antes e após limpeza)
  * Lista de variáveis removidas
  * Detecção de variáveis numéricas/categóricas
  * Indicação automática do método de correlação utilizado

A função principal `generate_report` grava um arquivo (HTML ou MD)
com imagens embutidas (base64) e devolve o caminho do arquivo criado.
"""

from __future__ import annotations

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
from .utils import search_dtypes, figsize_from_matrix, suggest_corr_method
from .vif import compute_vif, remove_high_vif
from .limpeza import clean

__all__ = ["generate_report"]

LOGGER = logging.getLogger("vassoura")


def _fig_to_base64(fig: plt.Figure, *, fmt: str = "png") -> str:
    """
    Converte uma figura Matplotlib em string base64 embutível.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    return f"data:image/{fmt};base64,{img_b64}"


def _plot_vif_barplot(vif_df: pd.DataFrame, title: str) -> plt.Figure:
    """
    Cria um gráfico de barras horizontais para o DataFrame de VIF.
    Rotula cada barra com valor formatado em duas casas decimais.
    """
    fig, ax = plt.subplots(figsize=(8, 0.5 * max(len(vif_df), 1) + 1))
    sns.barplot(data=vif_df, y="variable", x="vif", orient="h", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("VIF")
    ax.set_ylabel("Variável")

    # Adiciona rótulo de valor em cada barra
    for i, row in vif_df.sort_values("vif", ascending=True).reset_index(drop=True).iterrows():
        ax.text(
            row["vif"] + 0.02 * vif_df["vif"].max(),
            i,
            f"{row['vif']:.2f}",
            va="center",
            fontsize=9
        )

    plt.tight_layout()
    return fig


def generate_report(
    df: pd.DataFrame,
    *,
    output_path: str | Path = "vassoura_report.html",
    target_col: str | None = None,
    corr_method: str = "auto",
    corr_threshold: float = 0.9,
    vif_threshold: float = 10.0,
    keep_cols: Optional[List[str]] = None,
    limite_categorico: int = 50,
    force_categorical: Optional[List[str]] = None,
    remove_ids: bool = False,
    id_patterns: Optional[List[str]] = None,
    max_vif_iter: int = 20,
    verbose: bool = True,
    style: str = "html",  # "html" | "md"
) -> str:
    """
    Gera relatório de correlação/multicolinearidade e grava em output_path.

    Parâmetros
    ----------
    df : pandas.DataFrame
        DataFrame original.
    output_path : str | Path
        Caminho onde gravar o arquivo de saída.
    target_col : str | None
        Nome da coluna target. Será excluída das análises se include_target=False.
    corr_method : {"auto", "pearson", "spearman", "cramer"}
        Método de correlação a ser usado (ou "auto" para detecção automática).
    corr_threshold : float
        Limiar de correlação para limpeza (|corr| > corr_threshold).
    vif_threshold : float
        Limiar de VIF para limpeza (VIF > vif_threshold).
    keep_cols : list[str] | None
        Colunas que jamais devem ser removidas.
    limite_categorico, force_categorical, remove_ids, id_patterns :
        Parâmetros para `search_dtypes`.
    max_vif_iter : int
        Número máximo de iterações no filtro de VIF.
    verbose : bool
        Se True, imprime logs via logger "vassoura".
    style : {"html", "md"}
        Define o formato do relatório (HTML completo ou Markdown).

    Retorna
    -------
    str
        Caminho do arquivo gerado.
    """
    output_path = Path(output_path)

    # 1) Detecção de tipos de coluna
    num_cols, cat_cols = search_dtypes(
        df,
        target_col=target_col,
        limite_categorico=limite_categorico,
        force_categorical=force_categorical,
        remove_ids=remove_ids,
        id_patterns=id_patterns,
        verbose=verbose,
    )

    # 2) Cálculo de correlação ANTES da limpeza
    df_corr_source = df.copy()
    if target_col and target_col in df_corr_source:
        df_corr_source = df_corr_source.drop(columns=[target_col])

    # Determinar método efetivo
    if corr_method == "auto":
        corr_method_eff = suggest_corr_method(num_cols, cat_cols)
    else:
        corr_method_eff = corr_method

    corr_before = compute_corr_matrix(
        df_corr_source,
        method=corr_method_eff,
        target_col=None,
        include_target=False,
        limite_categorico=limite_categorico,
        force_categorical=force_categorical,
        remove_ids=remove_ids,
        id_patterns=id_patterns,
        verbose=verbose,
    )

    # 3) Cálculo de VIF ANTES da limpeza (nunca inclui target)
    vif_before = compute_vif(
        df,
        target_col=target_col,
        include_target=False,
        limite_categorico=limite_categorico,
        force_categorical=force_categorical,
        remove_ids=remove_ids,
        id_patterns=id_patterns,
        verbose=verbose,
    )

    # 4) Limpeza propriamente dita (corr + VIF)
    df_clean, dropped_cols, corr_after, vif_after = clean(
        df,
        target_col=target_col,
        include_target=False,
        corr_threshold=corr_threshold,
        corr_method=corr_method_eff,
        vif_threshold=vif_threshold,
        keep_cols=keep_cols,
        limite_categorico=limite_categorico,
        force_categorical=force_categorical,
        remove_ids=remove_ids,
        id_patterns=id_patterns,
        max_vif_iter=max_vif_iter,
        verbose=verbose,
    )

    # 5) Gera plots para heatmaps e VIF
    # Heatmap antes da limpeza
    fig_corr_before, ax_cb = plt.subplots(
        figsize=figsize_from_matrix(len(corr_before), base=0.4)
    )
    plot_corr_heatmap(
        corr_before,
        title=f"Correlação antes da limpeza ({corr_method_eff.capitalize()})",
        ax=ax_cb,
        annot=True,
        fmt=".2f",
    )
    img_corr_before = _fig_to_base64(fig_corr_before)

    # Heatmap final (apenas se houve remoção ou para exibir estado final)
    img_corr_after = ""
    show_final = bool(dropped_cols)
    if show_final:
        if not corr_after.empty:
            fig_corr_after, ax_ca = plt.subplots(
                figsize=figsize_from_matrix(len(corr_after), base=0.4)
            )
            plot_corr_heatmap(
                corr_after,
                title=f"Correlação após limpeza ({corr_method_eff.capitalize()})",
                ax=ax_ca,
                annot=True,
                fmt=".2f",
            )
            img_corr_after = _fig_to_base64(fig_corr_after)

    # 6) Plots de VIF antes e após limpeza
    fig_vif_before = _plot_vif_barplot(
        vif_before, title="VIF antes da limpeza"
    )
    img_vif_before = _fig_to_base64(fig_vif_before)

    img_vif_after = ""
    if show_final:
        fig_vif_after = _plot_vif_barplot(
            vif_after, title="VIF após limpeza"
        )
        img_vif_after = _fig_to_base64(fig_vif_after)

    # 7) Montagem do relatório HTML
    if style == "html":
        html = textwrap.dedent(
            f"""
            <!DOCTYPE html>
            <html lang="pt-BR">
            <head>
              <meta charset="utf-8">
              <title>Relatório Vassoura</title>
              <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #023059; }}
                p {{ max-width: 800px; line-height: 1.4; }}
                img {{ max-width: 100%; height: auto; margin-bottom: 20px; }}
                .section {{ margin-bottom: 40px; }}
              </style>
            </head>
            <body>
              <h1>Relatório de Correlação & Multicolinearidade – Vassoura</h1>

              <div class="section">
                <h2>Conceitos</h2>
                <p><strong>Correlação</strong> mede a relação linear (ou monotônica, no caso de Spearman) entre pares de variáveis. Valores próximos de +1 ou -1 indicam forte relação.</p>
                <p><strong>Multicolinearidade</strong> ocorre quando uma variável pode ser prevista (quase) perfeitamente a partir de uma combinação linear de outras variáveis. A presença de multicolinearidade alta prejudica a estabilidade e interpretação de coeficientes em modelos lineares.</p>
              </div>

              <div class="section">
                <h2>1. Tipos de Variáveis</h2>
                <h3>Numéricas ({len(num_cols)})</h3>
                <ul>
            """
        )
        for col in num_cols:
            html += f"<li>{col}</li>\n"
        if not num_cols:
            html += "<li><i>nenhuma</i></li>\n"
        html += "</ul>\n"

        html += "<h3>Categóricas ({})</h3>\n<ul>\n".format(len(cat_cols))
        for col in cat_cols:
            html += f"<li>{col}</li>\n"
        if not cat_cols:
            html += "<li><i>nenhuma</i></li>\n"
        html += "</ul>\n</div>\n"

        # Seção de correlação
        html += textwrap.dedent(
            f"""
            <div class="section">
              <h2>2. Heatmaps de Correlação</h2>
              <h3>Antes da Limpeza ({corr_method_eff.capitalize()})</h3>
              <img src="{img_corr_before}" alt="Correlação antes">
            """
        )
        if show_final and img_corr_after:
            html += textwrap.dedent(
                f"""
                <h3>Após a Limpeza ({corr_method_eff.capitalize()})</h3>
                <img src="{img_corr_after}" alt="Correlação após">
                """
            )
        elif not show_final:
            html += "<p><i>Nenhuma variável removida; não houve necessidade de limpeza por correlação ou multicolinearidade.</i></p>\n"
        html += "</div>\n"

        # Seção de VIF
        html += textwrap.dedent(
            f"""
            <div class="section">
              <h2>3. Variance Inflation Factor (VIF)</h2>
              <h3>Antes da Limpeza</h3>
              <img src="{img_vif_before}" alt="VIF antes">
            """
        )
        if show_final and img_vif_after:
            html += textwrap.dedent(
                f"""
                <h3>Após a Limpeza</h3>
                <img src="{img_vif_after}" alt="VIF após">
                """
            )
        elif not show_final:
            html += "<p><i>Nenhuma variável removida; VIF estava dentro do limiar definido.</i></p>\n"
        html += "</div>\n"

        # Seção de variáveis removidas
        html += textwrap.dedent(
            f"""
            <div class="section">
              <h2>4. Variáveis Removidas</h2>
              <ul>
            """
        )
        if dropped_cols:
            for var in dropped_cols:
                html += f"<li>{var}</li>\n"
        else:
            html += "<li><i>Nenhuma variável removida</i></li>\n"
        html += "</ul>\n</div>\n"

        html += "</body></html>\n"
        output_path.write_text(html, encoding="utf-8")

    else:
        # Versão Markdown básica
        md = textwrap.dedent(
            f"""
            # Relatório de Correlação & Multicolinearidade – Vassoura

            ## Conceitos
            **Correlação** mede a relação linear (ou monotônica, no caso de Spearman) entre pares de variáveis. Valores próximos de +1 ou -1 indicam forte relação.

            **Multicolinearidade** ocorre quando uma variável pode ser prevista (quase) perfeitamente a partir de uma combinação linear de outras variáveis, afetando modelos lineares.

            ## 1. Tipos de Variáveis
            **Numéricas ({len(num_cols)}):** {', '.join(num_cols) or '*nenhuma*'}

            **Categóricas ({len(cat_cols)}):** {', '.join(cat_cols) or '*nenhuma*'}

            ## 2. VIF Antes da Limpeza
            {vif_before.to_markdown(index=False)}

            ## 3. VIF Após a Limpeza
            {vif_after.to_markdown(index=False) if vif_after is not None else '*Nenhuma variável removida*'}

            ## 4. Variáveis Removidas
            {', '.join(dropped_cols) or '*nenhuma*'}
            """
        )
        output_path.write_text(md, encoding="utf-8")

    if verbose:
        LOGGER.info("Relatório gerado em %s", output_path)
    return str(output_path)
