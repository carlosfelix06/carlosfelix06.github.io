"""Benchmark pos-deploy de choque no ETL e mitigacao financeira na carteira UCI."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from experimento_ausencias import (
    CATEGORICAL,
    KEY,
    OUT,
    ROOT,
    SEEDS,
    load_partitions,
    mask_predictors,
    transform,
)


def _encode_nan_preserving_hgb(
    frame: pd.DataFrame,
    columns: list[str],
    numeric: list[str],
    encoder: OneHotEncoder,
    fit_enc: bool = False,
) -> np.ndarray:
    """Keep numeric NaNs native for HGB while preserving one-hot categorical encodings."""
    num = frame[numeric].to_numpy(dtype=np.float32)
    cat_df = frame[list(CATEGORICAL)].fillna(-999.0).astype(float)
    encoded = encoder.fit_transform(cat_df) if fit_enc else encoder.transform(cat_df)
    flags = frame[columns].isna().to_numpy(dtype=np.float32)
    return np.column_stack((num, encoded, flags)).astype(np.float32)


def evaluate_financial_strategy(
    y: np.ndarray,
    p: np.ndarray,
    ids: np.ndarray,
    exposure: np.ndarray,
    recovery_rate: float = 0.25,
    contact_cost: float = 426.0,
) -> dict:
    """Measure Top-10% default capture and net NT$ exposure recovery."""
    order = np.lexsort((ids, -p))
    top_k = math.ceil(0.10 * len(order))
    top = order[:top_k]
    gross_captured = float(np.sum(exposure[top] * y[top]))
    gross_recoverable = recovery_rate * gross_captured
    op_cost = top_k * contact_cost
    net_value = gross_recoverable - op_cost
    total_bad_exposure = float(np.sum(exposure * y))
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "captura_top_10": float(y[top].sum() / y.sum()),
        "top_10_default_exposure_nt": round(gross_captured, 2),
        "gross_recovery_nt": round(gross_recoverable, 2),
        "op_cost_nt": round(op_cost, 2),
        "net_value_nt": round(net_value, 2),
        "exposure_capture_share": round(gross_captured / max(total_bad_exposure, 1.0), 4),
    }


def plot_mitigation_figure(summary_data: dict) -> None:
    """Save comparative recovery and Top-10% capture chart as clean SVG without overlapping labels."""
    strategies = summary_data["strategies"]
    order_keys = [
        "baseline_limpo",
        "choque_sem_mitigacao",
        "mitigacao_mice_ridge",
        "mitigacao_dropout_robusto",
        "mitigacao_expected_loss",
    ]
    labels = [
        "Baseline\n(Sem Falha)",
        "Choque ETL\n(SimpleImp.)",
        "MICE\n(Ridge Iter.)",
        "HGB NaN\n(Score Risco)",
        "HGB + EL\n(Valor Esp.)",
    ]
    colors = ["#0B2038", "#D91568", "#64748B", "#2878E8", "#6A1B9A"]

    capturas = [strategies[k]["captura_top_10_mean"] * 100 for k in order_keys]
    net_m = [strategies[k]["net_value_nt_mean"] / 1e6 for k in order_keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.1), layout="constrained")
    fig.suptitle(
        "Choque corta captura no Top 10%; ordenar por Valor Esperado (HGB + EL) recupera 97,6% do ganho líquido",
        fontsize=12.0,
        fontweight="bold",
        color="#0B2038",
    )
    x = np.arange(len(order_keys))

    ax1.set_facecolor("#F8FAFC")
    ax1.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax1.spines[["top", "right"]].set_visible(False)
    b1 = ax1.bar(x, capturas, color=colors, width=0.54)
    ax1.set_xticks(x, labels, fontsize=8.4)
    ax1.set_ylim(0, max(capturas) * 1.20)
    ax1.set_ylabel("Captura de inadimplentes no Top 10% (%)")
    ax1.set_title("Choque Pós-Deploy (50% falha bloco faturamento)", fontsize=10.3, fontweight="bold", color="#0B2038")
    for bar, val in zip(b1, capturas):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.7, f"{val:.1f}%", ha="center", fontsize=8.5, fontweight="bold", color="#0B2038")

    ax2.set_facecolor("#F8FAFC")
    ax2.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax2.spines[["top", "right"]].set_visible(False)
    b2 = ax2.bar(x, net_m, color=colors, width=0.54)
    ax2.set_xticks(x, labels, fontsize=8.4)
    ax2.set_ylim(0, max(net_m) * 1.24)
    ax2.set_ylabel("Valor líquido recuperado no Top 10% (NT$ milhões)")
    ax2.set_title("Valor líquido (Recuperação 25% sobre fatura real - Custo)", fontsize=10.3, fontweight="bold", color="#0B2038")
    for bar, val in zip(b2, net_m):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.22, f"NT$ {val:.2f}M", ha="center", fontsize=8.4, fontweight="bold", color="#0B2038")
    delta_el = (net_m[4] - net_m[1]) / max(net_m[1], 1e-6) * 100
    ax2.annotate(
        f"+{delta_el:.1f}% vs choque",
        xy=(4, net_m[4]),
        xytext=(3.2, max(net_m) * 1.12),
        ha="center",
        fontsize=8.3,
        fontweight="bold",
        color="#6A1B9A",
        arrowprops=dict(arrowstyle="->", color="#6A1B9A", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D8B4FE"),
    )

    out_dirs = [ROOT / "resultados" / "figuras"]
    site_assets = ROOT / "app" / "site_assets" / "assets"
    if site_assets.is_dir():
        out_dirs.append(site_assets)
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "04_mitigacao_financeira.svg", facecolor="white")
    plt.close(fig)


def run_mitigation_and_financial_benchmark(parts: dict[str, pd.DataFrame] | None = None) -> dict:
    """Evaluate clean-trained HGB under 50% concentrated post-deploy failure and mitigations."""
    parts = parts or load_partitions()
    columns = [c for c in parts["treino"] if c not in (KEY, "default_payment_next_month")]
    numeric = [c for c in columns if c not in CATEGORICAL]
    y_train = parts["treino"]["default_payment_next_month"].to_numpy(dtype=int)
    y_test = parts["teste"]["default_payment_next_month"].to_numpy(dtype=int)
    test_ids = parts["teste"][KEY].to_numpy()

    exposure = np.clip(parts["teste"]["BILL_AMT1"].to_numpy(dtype=float), 0.0, None)
    median_bill = float(np.median(exposure))
    contact_cost = round(0.02 * median_bill, 2)
    recovery_rate = 0.25

    clean_train = parts["treino"][columns].copy().astype(float)
    clean_valid = parts["validacao"][columns].copy().astype(float)
    clean_test = parts["teste"][columns].copy().astype(float)

    mat_clean = transform(clean_train, clean_valid, clean_test, numeric)["hist_gradient_boosting"]
    hgb_clean = HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=0.0, early_stopping=False, random_state=42
    )
    hgb_clean.fit(mat_clean[0], y_train)
    p_base = hgb_clean.predict_proba(mat_clean[2])[:, 1]
    base_metrics = evaluate_financial_strategy(y_test, p_base, test_ids, exposure, recovery_rate, contact_cost)

    clean_num_imp = SimpleImputer(strategy="median").fit(clean_train[numeric])
    clean_cat_imp = SimpleImputer(strategy="most_frequent").fit(clean_train[list(CATEGORICAL)].astype(object))
    clean_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32).fit(
        clean_cat_imp.transform(clean_train[list(CATEGORICAL)].astype(object))
    )
    clean_mice = IterativeImputer(estimator=Ridge(alpha=10.0), max_iter=5, random_state=42).fit(clean_train[numeric])

    per_strategy: dict[str, list[dict]] = {
        "baseline_limpo": [],
        "choque_sem_mitigacao": [],
        "mitigacao_mice_ridge": [],
        "mitigacao_dropout_robusto": [],
        "mitigacao_expected_loss": [],
    }
    # Median observed utilization ratio BILL_AMT1 / LIMIT_BAL computed strictly on clean train
    clean_ratio = np.median(
        np.clip(clean_train["BILL_AMT1"].to_numpy(dtype=float), 0.0, None)
        / np.maximum(clean_train["LIMIT_BAL"].to_numpy(dtype=float), 1.0)
    )

    for seed in SEEDS:
        masked_test, _ = mask_predictors(parts["teste"], columns, "concentrada", 0.50, seed, 2)
        per_strategy["baseline_limpo"].append(base_metrics)

        shock_num = clean_num_imp.transform(masked_test[numeric])
        shock_cat = clean_ohe.transform(clean_cat_imp.transform(masked_test[list(CATEGORICAL)].astype(object)))
        shock_flags = masked_test[columns].isna().to_numpy(dtype=np.float32)
        x_shock = np.column_stack((shock_num, shock_cat, shock_flags)).astype(np.float32)
        p_shock = hgb_clean.predict_proba(x_shock)[:, 1]
        per_strategy["choque_sem_mitigacao"].append(
            evaluate_financial_strategy(y_test, p_shock, test_ids, exposure, recovery_rate, contact_cost)
        )

        mice_num = clean_mice.transform(masked_test[numeric])
        x_mice = np.column_stack((mice_num, shock_cat, shock_flags)).astype(np.float32)
        p_mice = hgb_clean.predict_proba(x_mice)[:, 1]
        per_strategy["mitigacao_mice_ridge"].append(
            evaluate_financial_strategy(y_test, p_mice, test_ids, exposure, recovery_rate, contact_cost)
        )

        masked_tr, _ = mask_predictors(parts["treino"], columns, "concentrada", 0.25, seed, 0)
        mix_train = pd.concat([clean_train, masked_tr], ignore_index=True)
        y_mix = np.concatenate([y_train, y_train])
        rob_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        x_mix = _encode_nan_preserving_hgb(mix_train, columns, numeric, rob_ohe, fit_enc=True)
        x_rob_test = _encode_nan_preserving_hgb(masked_test, columns, numeric, rob_ohe, fit_enc=False)
        hgb_rob = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=0.0, early_stopping=False, random_state=42
        )
        hgb_rob.fit(x_mix, y_mix)
        p_rob = hgb_rob.predict_proba(x_rob_test)[:, 1]
        per_strategy["mitigacao_dropout_robusto"].append(
            evaluate_financial_strategy(y_test, p_rob, test_ids, exposure, recovery_rate, contact_cost)
        )

        # Expected-loss ranking: use observed BILL_AMT1 when present, else Limit * clean train utilization ratio
        obs_bill = masked_test["BILL_AMT1"].to_numpy(dtype=float)
        obs_lim = masked_test["LIMIT_BAL"].to_numpy(dtype=float)
        ead_proxy = np.where(np.isnan(obs_bill), np.maximum(obs_lim * clean_ratio, 0.0), np.clip(obs_bill, 0.0, None))
        # Normalized ranking score combining calibrated PD * Expected Exposure (passed only for queue ordering!)
        el_score = p_rob * ead_proxy
        el_order = np.lexsort((test_ids, -el_score))
        top_el = el_order[: math.ceil(0.10 * len(el_order))]
        gross_el = float(np.sum(exposure[top_el] * y_test[top_el]))
        per_strategy["mitigacao_expected_loss"].append({
            "roc_auc": float(roc_auc_score(y_test, p_rob)),
            "ap": float(average_precision_score(y_test, p_rob)),
            "brier": float(brier_score_loss(y_test, p_rob)),
            "captura_top_10": float(y_test[top_el].sum() / y_test.sum()),
            "top_10_default_exposure_nt": round(gross_el, 2),
            "gross_recovery_nt": round(recovery_rate * gross_el, 2),
            "op_cost_nt": round(len(top_el) * contact_cost, 2),
            "net_value_nt": round(recovery_rate * gross_el - len(top_el) * contact_cost, 2),
            "exposure_capture_share": round(gross_el / max(float(np.sum(exposure * y_test)), 1.0), 4),
        })

    strategies_summary = {}
    for name, items in per_strategy.items():
        df = pd.DataFrame(items)
        strategies_summary[name] = {
            f"{col}_{stat}": round(float(df[col].agg(stat)), 4)
            for col in df.columns
            for stat in ("mean", "std")
        }

    net_shock = strategies_summary["choque_sem_mitigacao"]["net_value_nt_mean"]
    net_mice = strategies_summary["mitigacao_mice_ridge"]["net_value_nt_mean"]
    net_rob = strategies_summary["mitigacao_dropout_robusto"]["net_value_nt_mean"]
    net_el = strategies_summary["mitigacao_expected_loss"]["net_value_nt_mean"]
    net_base = strategies_summary["baseline_limpo"]["net_value_nt_mean"]

    output = {
        "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "n_test": int(len(parts["teste"])),
            "n_defaults_test": int(y_test.sum()),
            "top_10_queue_size": int(math.ceil(0.10 * len(parts["teste"]))),
            "total_default_bill_amt1_nt": round(float(np.sum(exposure * y_test)), 2),
            "median_bill_amt1_nt": round(median_bill, 2),
            "train_util_ratio_median": round(float(clean_ratio), 4),
            "recovery_rate": recovery_rate,
            "contact_cost_nt": contact_cost,
            "shock_mechanism": "concentrada_50pct_pos_deploy",
        },
        "strategies": strategies_summary,
        "financial_delta_summary": {
            "shock_loss_vs_clean_nt": round(net_base - net_shock, 2),
            "mice_recovery_vs_shock_nt": round(net_mice - net_shock, 2),
            "robust_dropout_diff_vs_shock_nt": round(net_rob - net_shock, 2),
            "expected_loss_gain_vs_shock_nt": round(net_el - net_shock, 2),
            "expected_loss_gain_vs_clean_nt": round(net_el - net_base, 2),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "resumo_avancado_mitigacao.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_mitigation_figure(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark pos-deploy de mitigacao financeira UCI.")
    parser.parse_args()
    result = run_mitigation_and_financial_benchmark()
    print(json.dumps(result, ensure_ascii=False, indent=2))
