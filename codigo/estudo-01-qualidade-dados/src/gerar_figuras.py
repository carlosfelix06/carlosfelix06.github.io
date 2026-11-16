"""Gera figuras para publicacao usando somente CSVs agregados do estudo.
Padronizado RIGOROSAMENTE com a paleta institucional do site:
- Concentrada (Faturas e Pagamentos): #D91568 (Magenta Serasa exato)
- MCAR (Aleatória): #0284C7 (Azul Ciano corporativo)
- Referência Íntegra: #64748B (Cinza ardósia suave pontilhado)
- Textos e eixos: #404040 (75% preto, sem preto puro)
- Fundo: #FFFFFF puro com grid suave (#F1F5F9)

Uso: python src/gerar_figuras.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "resultados" / "tabelas"
FIGURES = ROOT / "resultados" / "figuras"
SITE_ASSETS = ROOT / "app" / "site_assets" / "assets"

# Cores 100% padronizadas com a paleta do site
COLORS = {
    "mcar": "#0284C7",         # Azul Ciano oficial
    "concentrada": "#D91568",  # Magenta Serasa oficial (#D91568)
}
LABELS = {
    "mcar": "MCAR (Aleatória)",
    "concentrada": "Concentrada (Faturas e Pagamentos)"
}
MODELS = {
    "logistica": "Regressão Logística",
    "hist_gradient_boosting": "HistGradientBoosting"
}


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 1.0,
        "axes.labelcolor": "#404040",
        "text.color": "#404040",
        "xtick.color": "#64748B",
        "ytick.color": "#64748B",
        "grid.color": "#F1F5F9",
        "grid.linestyle": "-",
        "grid.linewidth": 1.0,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "svg.hashsalt": "estudo-qualidade-dados-202609-v3",
        "svg.fonttype": "none",
    })


def draw_metric(ax, data: pd.DataFrame, model: str, metric: str, title: str, percent: bool = False) -> None:
    frame = data[data.modelo == model]
    baseline = float(frame.loc[frame.cenario == "referencia", f"{metric}_mean"].iloc[0])
    
    # Linha de referência limpa e elegante em cinza suave
    ax.axhline(baseline, color="#64748B", linestyle="--", linewidth=1.8, label="Referência Íntegra", alpha=0.9)
    
    for scenario in ("mcar", "concentrada"):
        subset = frame[frame.cenario == scenario].sort_values("taxa")
        x = subset.taxa * 100
        y = subset[f"{metric}_mean"]
        sd = subset[f"{metric}_std"].fillna(0)
        
        # Linhas sólidas, nítidas, com marcadores redondos e borda branca destacada
        ax.plot(
            x, y,
            marker="o", markersize=7.5, markeredgewidth=1.8, markeredgecolor="#FFFFFF",
            linewidth=2.8, color=COLORS[scenario], label=LABELS[scenario]
        )
        # Faixa sombreada suave (alpha 0.10)
        ax.fill_between(x, y - sd, y + sd, color=COLORS[scenario], alpha=0.10)
        
    ax.set_title(title, pad=12, color="#404040", fontsize=13)
    ax.set_xticks([10, 30, 50], labels=["10%", "30%", "50%"])
    ax.set_xlabel("Ausência de células preditoras", labelpad=8)
    ax.grid(axis="y", visible=True)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def save(fig, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    SITE_ASSETS.mkdir(parents=True, exist_ok=True)
    
    png_path = FIGURES / f"{stem}.png"
    svg_path = FIGURES / f"{stem}.svg"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight", metadata={"Date": None})
    svg_cleaned = "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n"
    svg_path.write_text(svg_cleaned, encoding="utf-8")
    
    (SITE_ASSETS / f"{stem}.svg").write_text(svg_cleaned, encoding="utf-8")
    fig.savefig(SITE_ASSETS / f"{stem}.png", dpi=200, bbox_inches="tight")
    
    plt.close(fig)


def performance(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.8))
    fig.suptitle(
        "Falha concentrada derruba precisão e calibração; métrica global mascara o estrago",
        fontsize=15.0,
        fontweight="bold",
        color="#0B2038",
        y=0.975,
    )
    for row, model in enumerate(MODELS):
        draw_metric(axes[row, 0], data, model, "ap", f"{MODELS[model]} · Average Precision (maior é melhor)")
        draw_metric(axes[row, 1], data, model, "brier", f"{MODELS[model]} · Brier Score (menor é melhor)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#E2E8F0",
        bbox_to_anchor=(0.5, 0.935),
        fontsize=11,
    )
    fig.subplots_adjust(top=0.83, bottom=0.11, hspace=0.40, wspace=0.20)
    fig.text(
        0.5,
        0.02,
        "Médias de 5 máscaras determinísticas; faixa sombreada = ±1 desvio-padrão. Base UCI Taiwan (30.000 clientes).",
        ha="center",
        fontsize=10,
        color="#64748B",
    )
    save(fig, "01_desempenho_ap_brier")


def prioritization(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    fig.suptitle(
        "Fila operacional Top 10% perde até 4,9 p.p. (-66 inadimplentes) sob falha no faturamento",
        fontsize=14.8,
        fontweight="bold",
        color="#0B2038",
        y=0.97,
    )
    for ax, model in zip(axes, MODELS):
        draw_metric(ax, data, model, "captura_top_10", MODELS[model], percent=True)
        ax.set_ylabel("Inadimplentes capturados / total do teste")
        sub_c = data[(data.modelo == model) & (data.cenario == "concentrada")].sort_values("taxa")
        y50 = float(sub_c[sub_c.taxa == 0.5]["captura_top_10_mean"].iloc[0])
        base = float(data[(data.modelo == model) & (data.cenario == "referencia")]["captura_top_10_mean"].iloc[0])
        ax.annotate(
            f"{y50*100:.1f}% ({(y50-base)*100:+.1f} pp)",
            xy=(50, y50),
            xytext=(-14, -22),
            textcoords="offset points",
            ha="right",
            fontsize=9.5,
            fontweight="bold",
            color="#D91568",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#FBCFE8"),
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#E2E8F0",
        bbox_to_anchor=(0.5, 0.90),
        fontsize=11,
    )
    fig.subplots_adjust(top=0.74, bottom=0.18, wspace=0.20)
    fig.text(
        0.5,
        0.025,
        "Capacidade ilustrativa de 600 contas. Média de 5 máscaras; faixa = ±1 desvio-padrão.",
        ha="center",
        fontsize=10,
        color="#64748B",
    )
    save(fig, "02_captura_top10")


def calibration(curves: pd.DataFrame) -> None:
    subset = curves[curves.modelo == "hist_gradient_boosting"]
    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    fig.suptitle("Confiabilidade das probabilidades · HistGradientBoosting", fontsize=15, fontweight="bold", color="#404040")
    
    # Linha diagonal suave de calibração ideal
    ax.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--", linewidth=1.6, label="Calibração Ideal")
    
    for scenario, rate, color, label in (
        ("referencia", 0.0, "#64748B", "Referência Íntegra"),
        ("mcar", 0.5, COLORS["mcar"], "MCAR 50%"),
        ("concentrada", 0.5, COLORS["concentrada"], "Concentrada 50%"),
    ):
        values = subset[(subset.cenario == scenario) & (subset.taxa == rate)]
        grouped = values.groupby("grupo", as_index=False)[["prob_media", "taxa_observada"]].mean()
        ax.plot(
            grouped.prob_media, grouped.taxa_observada,
            marker="o", markersize=6.5, markeredgewidth=1.5, markeredgecolor="#FFFFFF",
            linewidth=2.4, color=color, label=label
        )
        
    ax.set(xlabel="Probabilidade média prevista por decil", ylabel="Taxa observada de inadimplência", xlim=(0, 0.85), ylim=(0, 0.85))
    ax.grid(linewidth=0.8, color="#F1F5F9", visible=True)
    ax.legend(frameon=False, loc="upper left", fontsize=11)
    fig.text(0.5, -0.045, "Decis de tamanho igual; pontos são médias das 5 máscaras. Acompanhe conjuntamente com o Brier Score.", ha="center", fontsize=10, color="#64748B")
    save(fig, "03_calibracao_hgb")


def main() -> None:
    style()
    summary = pd.read_csv(TABLES / "resumo_metricas.csv")
    curves = pd.read_csv(TABLES / "curvas_calibracao.csv")
    if len(summary) != 14 or len(curves) != 700:
        raise ValueError("Artefatos agregados incompletos; execute o experimento antes dos gráficos.")
    performance(summary)
    prioritization(summary)
    calibration(curves)
    print(f"Figuras geradas com sucesso com cores exatas do site em {FIGURES} e {SITE_ASSETS}")


if __name__ == "__main__":
    main()
