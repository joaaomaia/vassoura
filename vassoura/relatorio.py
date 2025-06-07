"""
Vassoura – Geração de relatórios HTML/Markdown
==============================================

Gera relatórios interativos contendo:
    * Notas explicativas sobre correlação e multicolinearidade
    * Heat-maps de correlação (antes e após limpeza) com e sem target
    * Gráficos de barras horizontais do VIF (antes e após limpeza)
    * Seção de autocorrelação em painel (opcional)
    * Audit trail das variáveis removidas
    * Detecção de variáveis numéricas/categóricas
    * Indicação automática do método de correlação utilizado

A função principal `generate_report` grava um arquivo (HTML ou MD)
com imagens embutidas (base64) e devolve o caminho do arquivo criado.
"""

from __future__ import annotations

import base64
import io
import logging
import os

os.environ["LIGHTGBM_DISABLE_STDERR_REDIRECT"] = "1"
import textwrap
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
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


def _pick_text_color(rgb: tuple[float, float, float]) -> str:
    """Escolhe cor de texto preto/branco baseada na luminância."""
    r, g, b = rgb
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#000" if lum > 0.6 else "#fff"


def _plot_vif_barplot(
    vif_s: pd.Series, title: str, ax: plt.Axes, thr: float = 5.0
) -> int:
    """Barplot de VIF compacto com cores de alerta e rótulos externos.

    Retorna o número de barras exibidas para cálculo de ``figsize``.
    """
    vif_s = vif_s.sort_values(ascending=False)
    if len(vif_s) > 30:
        others_val = float(vif_s.iloc[25:].mean())
        vif_s = pd.concat([vif_s.iloc[:25], pd.Series({"others": others_val})])

    n = len(vif_s)
    df_plot = vif_s.reset_index()
    df_plot.columns = ["feature", "vif"]

    pal = sns.color_palette("flare", n)
    pal = ["#dc3545" if v > thr else c for v, c in zip(df_plot["vif"], pal)]

    sns.barplot(
        data=df_plot,
        y="feature",
        x="vif",
        palette=pal,
        legend=False,
        orient="h",
        ax=ax,
    )

    ax.axvline(thr, linestyle="--", color="grey", linewidth=1)
    ax.set_title(f"{title} (n = {n})")
    ax.set_xlabel("VIF")

    labels = list(df_plot["feature"])
    if n > 25:
        labels = [textwrap.shorten(str(l), width=12, placeholder="…") for l in labels]
        ax.set_yticklabels(labels, fontsize=8, rotation=45, ha="right")
    else:
        ax.set_yticklabels(labels, fontsize=8)

    max_v = df_plot["vif"].max()
    for bar, val in zip(ax.patches, df_plot["vif"]):
        fmt = "{:.1f}" if val < 100 else "{:.0f}"
        ax.text(
            val + 0.02 * max_v,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(val),
            va="center",
            ha="left",
            fontsize=8,
        )

    return n


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
    history : list[dict] | dict | None
        Registro de remoções usado na tabela de audit trail. Para
        retrocompatibilidade, aceita-se também um dicionário contendo
        chaves como ``"history"``, ``"ks"`` e afins, tal qual versões
        antigas da biblioteca.

    Retorna
    -------
    str
        Caminho do arquivo gerado.
    """
    output_path = Path(output_path)

    import warnings

    warnings.filterwarnings("ignore", message="No further splits with positive gain")
    warnings.filterwarnings(
        "ignore",
        message="LightGBM binary classifier with TreeExplainer shap values output has changed",
    )
    import shap
    from lightgbm import LGBMClassifier

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
        psi_series = precomputed.get("psi_series")
        ks_series = precomputed.get("ks_series")
        perm_series = precomputed.get("perm_series")
        partial_graph = precomputed.get("partial_graph")
        drift_leak_df = precomputed.get("drift_leak_df")
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
        psi_series = None
        ks_series = None
        perm_series = None
        partial_graph = None
        drift_leak_df = None
        id_cols = id_cols or []
        date_cols = date_cols or []
        ignore_cols = ignore_cols or []
        history = history or []

    if isinstance(history, dict):
        ks_series = history.get("ks", ks_series)
        perm_series = history.get("perm", perm_series)
        psi_series = history.get("psi", psi_series)
        drift_leak_df = history.get("drift_leak_df", drift_leak_df)
        history = history.get("history", history.get("steps", []))

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

    # 7) Plots de VIF antes e após limpeza em uma única figura
    vif_before_s = vif_before.set_index("variable")["vif"]
    vif_after_s = (
        vif_after.set_index("variable")["vif"]
        if vif_after is not None
        else pd.Series(dtype=float)
    )

    series_before = vif_before_s.sort_values(ascending=False)
    if len(series_before) > 30:
        others_b = float(series_before.iloc[25:].mean())
        series_before = pd.concat(
            [series_before.iloc[:25], pd.Series({"others": others_b})]
        )

    series_after = vif_after_s.sort_values(ascending=False)
    if len(series_after) > 30:
        others_a = float(series_after.iloc[25:].mean())
        series_after = pd.concat(
            [series_after.iloc[:25], pd.Series({"others": others_a})]
        )

    n_before = len(series_before)
    n_after = (
        len(series_after) if not (vif_after_s[vif_after_s > vif_threshold].empty) else 1
    )

    fig_vif, (ax_l, ax_r) = plt.subplots(
        ncols=2,
        sharey=True,
        figsize=(10, 0.25 * max(n_before, n_after) + 1),
    )

    _plot_vif_barplot(series_before, "VIF antes da limpeza", ax_l, thr=vif_threshold)

    if vif_after_s[vif_after_s > vif_threshold].empty:
        ax_r.axis("off")
        ax_r.text(0.5, 0.5, "Nenhum VIF acima do limite", ha="center", va="center")
    else:
        _plot_vif_barplot(series_after, "VIF após a limpeza", ax_r, thr=vif_threshold)

    fig_vif.tight_layout(w_pad=1)
    img_vif_pair = _fig_to_base64(fig_vif)

    # 8) Shadow-Feature Analysis
    if target_col is not None:
        if "__shadow__" not in df_clean.columns:
            np.random.seed(0)
            df_clean["__shadow__"] = np.random.rand(len(df_clean))

        def _compute_ks(s: pd.Series, target: pd.Series, n_bins: int = 10) -> float:
            if s.dtype.kind in "bifc" and s.nunique() > 1:
                try:
                    b = pd.qcut(s, q=n_bins, duplicates="drop")
                except ValueError:
                    return 0.0
            else:
                b = s.astype("category")
            tab = pd.crosstab(b, target)
            if tab.shape[1] != 2:
                return 0.0
            cdf_good = tab[0].cumsum() / tab[0].sum()
            cdf_bad = tab[1].cumsum() / tab[1].sum()
            return float((cdf_good - cdf_bad).abs().max())

        cols_eval = [
            c
            for c in df_clean.columns
            if c not in {target_col, *(id_cols or []), *(date_cols or [])}
        ]
        ks_s = pd.Series(
            {col: _compute_ks(df_clean[col], df_clean[target_col]) for col in cols_eval}
        )
        ks_s = ks_s.sort_values(ascending=False)

        X = df_clean[cols_eval]
        X = X.apply(lambda s: s.astype("category") if s.dtype == "object" else s)
        model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=0,
            class_weight="balanced",
            verbosity=-1,
        ).fit(X, df_clean[target_col])

        expl = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_vals = expl.shap_values(X)
        shap_arr = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

        gain_raw = getattr(
            getattr(model, "booster_", model),
            "feature_importance",
            lambda *a, **k: np.zeros(len(X.columns)),
        )("gain")
        gain_arr = np.array(gain_raw)
        if gain_arr.size != len(X.columns):
            gain_arr = np.zeros(len(X.columns))
        gain_s = pd.Series(gain_arr, index=X.columns).sort_values(ascending=False)

        n_features = len(ks_s)
        fig_shadow, (ax_shap, ax_ks, ax_imp) = plt.subplots(
            1,
            3,
            figsize=(18, 0.35 * n_features + 2),
            dpi=110,
        )

        # SHAP summary plot
        expl_values = shap.Explanation(
            shap_arr,
            base_values=np.zeros(len(X)),
            data=X,
            feature_names=X.columns,
        )
        shap.plots.beeswarm(expl_values, show=False, plot_size=None, ax=ax_shap)
        ax_shap.set_title("SHAP Summary")
        shadow_labels = [t.get_text() for t in ax_shap.get_yticklabels()]
        if "__shadow__" in shadow_labels:
            idx = shadow_labels.index("__shadow__")
            coll = ax_shap.collections[idx]
            coll.set_sizes([120] * len(coll.get_sizes()))
            coll.set_edgecolors(["red"] * len(coll.get_edgecolors()))

        # KS by feature
        pal_ks = sns.color_palette("flare", len(ks_s))
        ks_colors = {
            f: ("#FFB347" if f == "__shadow__" else pal_ks[i])
            for i, f in enumerate(ks_s.index)
        }
        ks_df = ks_s.reset_index().rename(columns={"index": "feature", 0: "KS"})
        sns.barplot(
            data=ks_df,
            y="feature",
            x="KS",
            hue="feature",
            legend=False,
            palette=ks_colors,
            orient="h",
            ax=ax_ks,
        )
        ax_ks.set_title("KS")

        # LightGBM Gain
        pal_imp = sns.color_palette("flare", len(gain_s))
        imp_colors = {
            f: ("#FFB347" if f == "__shadow__" else pal_imp[i])
            for i, f in enumerate(gain_s.index)
        }
        imp_df = gain_s.reset_index().rename(columns={"index": "feature", 0: "gain"})
        sns.barplot(
            data=imp_df,
            y="feature",
            x="gain",
            hue="feature",
            legend=False,
            palette=imp_colors,
            orient="h",
            ax=ax_imp,
        )
        ax_imp.set_title("LightGBM Gain")

        fig_shadow.tight_layout(w_pad=3)
        img_shadow_triplet = _fig_to_base64(fig_shadow)

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
.vif-grid{{display:flex;gap:12px;flex-wrap:wrap}}
.vif-grid img{{flex:1 1 48%}}
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
                <nav style="margin-bottom:16px"><a href="#tipos">Tipos</a> · <a href="#heatmaps">Heatmaps</a> · <a href="#vif">VIF</a> · <a href="#audit">Audit trail</a> · <a href="#shadow">Shadow</a></nav>
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
                <h2>Tipos de Variáveis</h2>
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
                <h2>Método de Correlação</h2>
                <p>{metodo_texto}</p>
            </div>
            """
        )

        # Seção de correlação
        html += textwrap.dedent(
            f"""
            <div class="section" id="heatmaps">
                <h2>Heatmaps de Correlação</h2>
                <h3>Antes da Limpeza ({corr_method_eff.capitalize()})</h3>
                <img src="{img_corr_before}" alt="flare_corr_antes">
            """
        )
        if show_final and img_corr_after:
            html += textwrap.dedent(
                f"""
                <h3>Após a Limpeza ({corr_method_eff.capitalize()})</h3>
                <img src="{img_corr_after}" alt="flare_corr_depois">
                """
            )
        elif not show_final:
            html += "<p><i>Nenhuma variável removida; não houve necessidade de limpeza por correlação ou multicolinearidade.</i></p>\n"
        html += "</div>\n"

        # Seção de VIF
        html += textwrap.dedent(
            f"""
            <div class="section" id="vif">
                <h2>Variance Inflation Factor (VIF)</h2>
                <div class="vif-grid"><img src="{img_vif_pair}" alt="VIF antes vs após" style="max-width:100%;height:auto"></div>
            </div>
            """
        )

        if psi_series is not None:
            html += '<div class="section" id="psi">'
            html += "<h2>Estabilidade Temporal (PSI)</h2>"
            html += psi_series.to_frame().to_html(
                classes="audit", float_format="{:.3f}".format
            )
            html += "</div>\n"

        if perm_series is not None:
            html += '<div class="section" id="perm">'
            html += "<h2>Permutation Importance</h2>"
            html += perm_series.to_frame().to_html(
                classes="audit", float_format="{:.3f}".format
            )
            html += "</div>\n"

        if partial_graph is not None:
            html += '<div class="section" id="partial">'
            html += "<h2>Partial Correlation Cluster</h2>"
            html += f"<p>{len(partial_graph.nodes())} variáveis no vertex cover.</p>"
            html += "</div>\n"

        if drift_leak_df is not None:
            html += '<div class="section" id="drift">'
            html += "<h2>Drift vs Target Leakage</h2>"
            html += drift_leak_df.to_html(classes="audit", float_format="{:.3f}".format)
            html += "</div>\n"

        if dropped_cols and history:

            def _split_reason(r: str) -> tuple[str, str]:
                for sep in (">", "<"):
                    if sep in r:
                        h, m = r.split(sep, 1)
                        return h, sep + m
                return r, ""

            html += '<div class="section" id="audit">'
            html += "<h2>Audit Trail: Colunas Removidas</h2>"
            html += '<table class="audit"><thead><tr><th>Heurística</th><th>Motivo</th><th>Colunas</th></tr></thead><tbody>'
            for step in history:
                if not step.get("cols"):
                    continue
                cols_sorted = sorted(step["cols"])
                heur, mot = _split_reason(step.get("reason", ""))
                html += (
                    f"<tr><td>{heur}</td><td>{mot}</td><td><div class='feature-grid'>"
                )
                for col in cols_sorted:
                    tipo = (
                        "num"
                        if col in num_cols
                        else "cat" if col in cat_cols else "unk"
                    )
                    badge = f" <span class='badge {tipo}'>{tipo if tipo!='unk' else 'unk'}</span>"
                    html += f"<div>{col}{badge}</div>"
                html += "</div></td></tr>"
            html += "</tbody></table></div>"

        if target_col is not None:
            html += (
                '<div class="section" id="shadow">'
                "<h2>Shadow-Feature Analysis</h2>"
                "<p>Inclui-se a variável aleatória <code>__shadow__</code> como referência.</p>"
                f"<img src='{img_shadow_triplet}' alt='SHAP · KS · LightGBM Gain'>"
                "</div>\n"
            )

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

            ## 5. Estabilidade Temporal (PSI)
            {psi_series.to_markdown() if psi_series is not None else '*não calculado*'}

            ## 6. Permutation Importance
            {perm_series.to_markdown() if perm_series is not None else '*não calculado*'}

            ## 7. Partial Correlation Cluster
            {len(partial_graph.nodes()) if partial_graph is not None else 0} variáveis no vertex cover

            ## 8. Drift vs Target Leakage
            {drift_leak_df.to_markdown() if drift_leak_df is not None else '*não calculado*'}
            """
        )
        output_path.write_text(md, encoding="utf-8")

    if verbose:
        LOGGER.info("Relatório gerado em %s", output_path)
    return str(output_path)
