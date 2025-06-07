"""
Vassoura – Geração de relatórios HTML/Markdown
==============================================

Gera relatórios interativos contendo:
    * Notas explicativas sobre correlação e multicolinearidade
    * Heat-maps de correlação (antes e após limpeza) com e sem target
    * Gráficos de barras horizontais do VIF (antes e após limpeza)
    * Seção de autocorrelação em painel (opcional)
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
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .correlacao import compute_corr_matrix, plot_corr_heatmap
from .limpeza import clean
from .utils import figsize_from_matrix, search_dtypes, suggest_corr_method
from .vif import compute_vif

__all__ = ["generate_report"]

LOGGER = logging.getLogger("vassoura")


def _fig_to_base64(fig: plt.Figure, *, fmt: str = "png") -> str:
    """Converte uma figura Matplotlib em string base64 embutível."""
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    return f"data:image/{fmt};base64,{img_b64}"


def _plot_vif_barplot(vif_df: pd.DataFrame, title: str) -> plt.Figure:
    """Cria um gráfico de barras horizontais para o DataFrame de VIF.
    Rotula cada barra com valor formatado em duas casas decimais."""
    fig, ax = plt.subplots(figsize=(8, 0.5 * max(len(vif_df), 1) + 1))
    sns.barplot(data=vif_df, y="variable", x="vif", orient="h", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("VIF")
    ax.set_ylabel("Variável")

    # Adiciona rótulo de valor em cada barra
    for i, row in (
        vif_df.sort_values("vif", ascending=True).reset_index(drop=True).iterrows()
    ):
        ax.text(
            row["vif"] + 0.02 * vif_df["vif"].max(),
            i,
            f"{row['vif']:.2f}",
            va="center",
            fontsize=9,
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
    n_steps: int | None = None,
    vif_n_steps: int = 1,
    heatmap_labels: bool = True,
    heatmap_base_size: float = 0.6,
    verbose: bool = True,
    style: str = "html",  # "html" | "md"
    precomputed: Optional[Dict[str, Any]] = None,
    id_cols: Optional[List[str]] = None,
    date_cols: Optional[List[str]] = None,
    ignore_cols: Optional[List[str]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
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
    n_steps : int | None
        Passos fracionados para remoção por correlação.
    vif_n_steps : int
        Passos fracionados para remoção por VIF.
    heatmap_labels : bool
        Se True, exibe anotações numéricas (valores) no heatmap; caso False, anotações são omitidas.
    heatmap_base_size : float
        Fator base para dimensionar o heatmap (multiplicado pelo número de features).
    verbose : bool
        Se True, imprime logs via logger "vassoura".
    style : {"html", "md"}
        Define o formato do relatório (HTML completo ou Markdown).
    id_cols, date_cols, ignore_cols : list[str] | None
        Listas de colunas identificadoras, de data e ignoradas para exibir
        no painel-resumo.
    history : list[dict] | None
        Registro de remoções, usado na tabela de audit trail.

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

    # 2) Definir e justificar método de correlação
    if corr_method == "auto":
        corr_method_eff = suggest_corr_method(num_cols, cat_cols)
        if corr_method_eff == "pearson":
            justificativa = (
                "<b>Somente variáveis numéricas foram detectadas</b>, portanto Pearson é apropriado "
                "para medir correlação linear."
            )
        elif corr_method_eff == "spearman":
            justificativa = (
                "<b>Há mistura de variáveis numéricas e categóricas</b>, ou espera-se relação monotônica, "
                "portanto Spearman é mais adequado para correlação."
            )
        else:  # cramer
            justificativa = "<b>A maioria das variáveis é categórica</b>, portanto Cramér-V é usado para medir associação."
        metodo_texto = f"Método de correlação selecionado: <b>{corr_method_eff.capitalize()}</b>. {justificativa}"
    else:
        corr_method_eff = corr_method
        metodo_texto = (
            f"Método de correlação especificado: <b>{corr_method_eff.capitalize()}</b>."
        )

    if precomputed is not None:
        df_clean = precomputed.get("df_clean", df.copy())
        dropped_cols = precomputed.get("dropped_cols", [])
        corr_before = precomputed.get("corr_before")
        vif_before = precomputed.get("vif_before")
        corr_after = precomputed.get("corr_after")
        vif_after = precomputed.get("vif_after")
        id_cols = precomputed.get("id_cols", id_cols or [])
        date_cols = precomputed.get("date_cols", date_cols or [])
        ignore_cols = precomputed.get("ignore_cols", ignore_cols or [])
        history = precomputed.get("history", history)
    else:
        df_clean = None
        dropped_cols = []
        corr_before = None
        vif_before = None
        corr_after = None
        vif_after = None
        id_cols = id_cols or []
        date_cols = date_cols or []
        ignore_cols = ignore_cols or []
        history = history or []

    # 3) Cálculo de correlação ANTES da limpeza
    if corr_before is None:
        df_corr_source = df.copy()
        if target_col and target_col in df_corr_source:
            df_corr_source = df_corr_source.drop(columns=[target_col])
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

    # 4) Cálculo de VIF ANTES da limpeza (nunca inclui target)
    if vif_before is None:
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

    # 5) Limpeza propriamente dita (corr + VIF)
    if df_clean is None:
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
            n_steps=n_steps,
            vif_n_steps=vif_n_steps,
            verbose=verbose,
        )

    if corr_after is None:
        corr_after = compute_corr_matrix(
            df_clean.drop(columns=[target_col], errors="ignore"),
            method=corr_method_eff,
            target_col=None,
            include_target=False,
            limite_categorico=limite_categorico,
            force_categorical=force_categorical,
            remove_ids=remove_ids,
            id_patterns=id_patterns,
            verbose=verbose,
        )

    if vif_after is None:
        vif_after = compute_vif(
            df_clean,
            target_col=target_col,
            include_target=False,
            limite_categorico=limite_categorico,
            force_categorical=force_categorical,
            remove_ids=remove_ids,
            id_patterns=id_patterns,
            verbose=verbose,
        )

    # 6) Gera plots para heatmaps e VIF
    # Heatmap antes da limpeza
    fig_corr_before, ax_cb = plt.subplots(
        figsize=figsize_from_matrix(len(corr_before), base=heatmap_base_size)
    )
    before_large = len(corr_before.columns) > 40
    plot_corr_heatmap(
        corr_before,
        title=f"Correlação antes da limpeza ({corr_method_eff.capitalize()})",
        ax=ax_cb,
        annot=False if before_large else heatmap_labels,
        fmt=".2f",
        highlight_labels=before_large,
        corr_threshold=corr_threshold,
    )
    img_corr_before = _fig_to_base64(fig_corr_before)

    # Heatmap final (apenas se houver variáveis removidas)
    img_corr_after = ""
    show_final = bool(dropped_cols)
    if show_final and not corr_after.empty:
        fig_corr_after, ax_ca = plt.subplots(
            figsize=figsize_from_matrix(len(corr_after), base=heatmap_base_size)
        )
        after_large = len(corr_after.columns) > 40
        plot_corr_heatmap(
            corr_after,
            title=f"Correlação após limpeza ({corr_method_eff.capitalize()})",
            ax=ax_ca,
            annot=False if after_large else heatmap_labels,
            fmt=".2f",
            highlight_labels=after_large,
            corr_threshold=corr_threshold,
        )
        img_corr_after = _fig_to_base64(fig_corr_after)

    # 7) Plots de VIF antes e após limpeza
    fig_vif_before = _plot_vif_barplot(vif_before, title="VIF antes da limpeza")
    img_vif_before = _fig_to_base64(fig_vif_before)

    img_vif_after = ""
    if show_final and vif_after is not None:
        fig_vif_after = _plot_vif_barplot(vif_after, title="VIF após limpeza")
        img_vif_after = _fig_to_base64(fig_vif_after)

    # 8) Montagem do relatório HTML
    if style == "html":
        html = textwrap.dedent(
            f"""
            <!DOCTYPE html>
            <html lang="pt-BR">
            <head>
                <meta charset="utf-8">
                <title>Relatório Vassoura</title>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" integrity="sha512-7MUo45l42UuDGuDX6zF98MXcEEoJacV8LjKzNoh+piQTWDpKSqwdt5Pzz36E6BjLsnGB2XjBt2PTP4MLs5q0Wg==" crossorigin="anonymous" referrerpolicy="no-referrer" />
                <style>
/* Paleta suave inspirado em design systems financeiros */
:root {{
  --bg: #f8f9fa;
  --card: #ffffff;
  --primary: #0d6efd;
  --success: #198754;
  --warning: #ffc107;
  --danger: #dc3545;
  --text: #212529;
  --muted: #6c757d;
  --radius: 8px;
  --shadow: 0 2px 6px rgba(0,0,0,.05);
  --mono: "Fira Code", monospace;
}}

body {{background:var(--bg);color:var(--text);font:15px/1.5 "Inter",sans-serif;}}
h1,h2,h3{{color:var(--primary);margin:0 0 .6em}}
.section{{margin-bottom:48px;padding:32px;background:var(--card);border-radius:var(--radius);box-shadow:var(--shadow);}}
.feature-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:6px;}}
.feature-grid div{{padding:6px 10px;background:var(--bg);border-radius:var(--radius);}}
.badge{{padding:2px 6px;border-radius:4px;font-size:.75rem;font-family:var(--mono)}}
.badge.num{{background:var(--success);color:#fff;}}
.badge.cat{{background:var(--warning);color:#000;}}
.pill{{display:inline-flex;align-items:center;gap:6px;margin-right:8px;margin-bottom:8px;background:var(--primary);color:#fff;padding:6px 12px;border-radius:99px;font-weight:500;box-shadow:var(--shadow);}}
.pill i{{opacity:.8}}
table.audit{{width:100%;border-collapse:collapse;font-size:.9rem;}}
table.audit th,table.audit td{{padding:6px 8px;border-bottom:1px solid #e1e6eb;}}
table.audit th{{background:var(--bg);text-align:left;}}
img{{border:1px solid #e1e6eb;border-radius:var(--radius);}}
@media(max-width:700px){{.section{{padding:20px}}.feature-grid{{grid-template-columns:1fr}}}}
                </style>
            </head>
            <body>
                <h1>Relatório de Correlação & Multicolinearidade – Vassoura</h1>
                <nav style="margin-bottom:16px"><a href="#tipos">Tipos</a> · <a href="#heatmaps">Heatmaps</a> · <a href="#vif">VIF</a> · <a href="#audit">Audit trail</a></nav>
                <div class="section">
            """
        )

        def _pill(icon: str, items: List[str], label: str, color: str) -> str:
            full = ", ".join(items)
            short = ", ".join(items[:10]) + ("..." if len(items) > 10 else "")
            if not items:
                short = "<i>nenhuma</i>"
            return (
                f"<span class='pill' style='background:{color}'><i class='{icon}'></i>"
                f"{len(items)} {label}: <span title='{full}'>{short}</span></span>"
            )

        html += _pill("fa-solid fa-id-card", id_cols, "ID Cols", "var(--primary)")
        html += _pill(
            "fa-solid fa-calendar-day", date_cols, "Date Cols", "var(--success)"
        )
        html += _pill("fa-solid fa-eye-slash", ignore_cols, "Ignored", "var(--warning)")
        html += _pill("fa-solid fa-trash-alt", dropped_cols, "Dropped", "var(--danger)")
        html += "</div>"

        html += textwrap.dedent(
            """
                <div class="section" id="conceitos">
                <h2>Conceitos</h2>
                <p><strong>Correlação</strong> mede a relação linear (ou monotônica, no caso de Spearman) entre pares de variáveis. Valores próximos de +1 ou -1 indicam forte relação.</p>
                <p><strong>Multicolinearidade</strong> ocorre quando uma variável pode ser prevista (quase) perfeitamente a partir de uma combinação linear de outras variáveis. A presença de multicolinearidade alta prejudica a estabilidade e interpretação de coeficientes em modelos lineares.</p>
                </div>
                <div class="section" id="tipos">
                <h2>1. Tipos de Variáveis</h2>
                <div class="feature-grid">
            """
        )
        for col in num_cols:
            html += f"<div>{col} <span class='badge num'>num</span></div>\n"
        for col in cat_cols:
            html += f"<div>{col} <span class='badge cat'>cat</span></div>\n"
        if not num_cols and not cat_cols:
            html += "<div><i>nenhuma coluna detectada</i></div>\n"
        html += "</div></div>\n"

        # Justificativa do método de correlação
        html += textwrap.dedent(
            f"""
            <div class="section">
                <h2>2. Método de Correlação</h2>
                <p>{metodo_texto}</p>
            </div>
            """
        )

        # Seção de correlação
        html += textwrap.dedent(
            f"""
            <div class="section" id="heatmaps">
                <h2>3. Heatmaps de Correlação</h2>
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
            <div class="section" id="vif">
                <h2>4. Variance Inflation Factor (VIF)</h2>
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
                <h2>5. Variáveis Removidas</h2>
                <ul>
            """
        )
        if dropped_cols:
            for var in dropped_cols:
                html += f"<li>{var}</li>\n"
        else:
            html += "<li><i>Nenhuma variável removida</i></li>\n"
        html += "</ul>\n</div>\n"

        if dropped_cols and history:

            def _split_reason(r: str) -> tuple[str, str]:
                for sep in (">", "<"):
                    if sep in r:
                        h, m = r.split(sep, 1)
                        return h, sep + m
                return r, ""

            html += '<div class="section" id="audit">'
            html += "<h2>Audit trail</h2>"
            html += '<table class="audit"><thead><tr><th>Colunas</th><th>Heurística</th><th>Motivo</th></tr></thead><tbody>'
            for step in history:
                if not step.get("cols"):
                    continue
                cols = ", ".join(step["cols"])
                heur, mot = _split_reason(step.get("reason", ""))
                html += f"<tr><td>{cols}</td><td>{heur}</td><td>{mot}</td></tr>"
            html += "</tbody></table></div>"

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

            ## 2. Método de Correlação
            {metodo_texto}

            ## 3. VIF Antes da Limpeza
            {vif_before.to_markdown(index=False)}

            ## 4. VIF Após a Limpeza
            {vif_after.to_markdown(index=False) if vif_after is not None else '*Nenhuma variável removida*'}

            ## 5. Variáveis Removidas
            {', '.join(dropped_cols) or '*nenhuma*'}
            """
        )
        output_path.write_text(md, encoding="utf-8")

    if verbose:
        LOGGER.info("Relatório gerado em %s", output_path)
    return str(output_path)
