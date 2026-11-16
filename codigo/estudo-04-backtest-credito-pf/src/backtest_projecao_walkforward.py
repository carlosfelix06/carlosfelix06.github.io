from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
SERIES_SNAPSHOT = STUDY_ROOT / "resultados" / "series_bcb_sgs.json"
VITRINE_SVG_PATH = (
    REPO_ROOT
    / "estudos"
    / "202609 - Impacto da qualidade dos dados no risco de inadimplência"
    / "app"
    / "site_assets"
    / "assets"
    / "img"
    / "vitrine_estudo_04_bcb.svg"
)
PALETTE = {
    "navy": "#0B2038",
    "magenta": "#D91568",
    "purple": "#6A1B9A",
    "blue": "#2878E8",
    "emerald": "#059669",
    "muted": "#64748B",
    "grid": "#E2E8F0",
    "background": "#F8FAFC",
}
BCB_SERIES = [
    (20671, "consignado", "Crédito Pessoal Consignado"),
    (20666, "nao_consignado", "Crédito Pessoal Não-Consignado"),
    (20673, "veiculos", "Aquisição de Veículos PF"),
    (20682, "cartao_credito", "Cartão de Crédito Total PF"),
    (20665, "cheque_especial", "Cheque Especial PF"),
]
INITIAL_TRAIN_MONTHS = 12
BENCHMARK_START_MONTH = 24  # 2 ciclos sazonais completos -> 33 origens (01/2024-09/2026)
TOLERANCE = 0.05
DM_HAC_LAG = 4


def _easter_sunday(year: int) -> date:
    """Calcula a Páscoa pelo calendário gregoriano."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = (h + ell - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def br_banking_holidays(year: int) -> list[str]:
    """Retorna lista de feriados bancários ANBIMA em formato ISO (YYYY-MM-DD)."""
    fixed = ((1, 1), (4, 21), (5, 1), (9, 7), (10, 12), (11, 2), (11, 15), (12, 25))
    holidays = [date(year, month, day) for month, day in fixed]
    if year >= 2024:
        holidays.append(date(year, 11, 20))
    easter = _easter_sunday(year)
    holidays.extend(
        (
            easter - timedelta(days=48),
            easter - timedelta(days=47),
            easter - timedelta(days=2),
            easter + timedelta(days=60),
        )
    )
    return [h.isoformat() for h in sorted(holidays)]


@lru_cache(maxsize=None)
def _brazilian_bank_holidays(year: int) -> np.busdaycalendar:
    """Calendário nacional de feriados bancários da ANBIMA, sem dependência externa."""
    return np.busdaycalendar(holidays=np.asarray(br_banking_holidays(year), dtype="datetime64[D]"))


def refresh_bcb_snapshot(output_path: Path = SERIES_SNAPSHOT) -> dict:
    """Baixa as 5 séries mensais oficiais de concessões PF do BCB SGS."""
    raw_by_id: dict[int, dict[str, float]] = {}
    date_sets: list[set[str]] = []
    for sid, _, _ in BCB_SERIES:
        url = (
            f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{sid}"
            "/dados?formato=json&dataInicial=01/01/2022&dataFinal=31/12/2026"
        )
        req = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        with urlopen(req, timeout=30) as resp:
            rows = json.loads(resp.read().decode("utf-8"))
        mapping = {item["data"]: float(item["valor"]) for item in rows}
        raw_by_id[sid] = mapping
        date_sets.append(set(mapping.keys()))

    def _date_key(date_str: str) -> tuple[int, int, int]:
        dd, mm, yyyy = date_str.split("/")
        return int(yyyy), int(mm), int(dd)

    common_dates = sorted(set.intersection(*date_sets), key=_date_key)
    matrix = [[raw_by_id[sid][dt] for dt in common_dates] for sid, _, _ in BCB_SERIES]
    payload = {
        "source": "Banco Central do Brasil — SGS API (api.bcb.gov.br)",
        "unit": "R$ milhões (concessões mensais nominais PF)",
        "series": [{"id": sid, "key": key, "label": label} for sid, key, label in BCB_SERIES],
        "dates": common_dates,
        "matrix_brl_millions": matrix,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_bcb_snapshot(snapshot_path: Path = SERIES_SNAPSHOT) -> dict:
    if not snapshot_path.is_file():
        return refresh_bcb_snapshot(snapshot_path)
    return json.loads(snapshot_path.read_text(encoding="utf-8"))


def generate_series() -> tuple[np.ndarray, np.ndarray]:
    """Carrega a matriz oficial (5x57) do BCB SGS e estima tendência móvel."""
    snapshot = load_bcb_snapshot()
    realized = np.asarray(snapshot["matrix_brl_millions"], dtype=float)
    kernel = np.array([0.25, 0.50, 0.25], dtype=float)
    smoothed = np.empty_like(realized)
    for idx in range(realized.shape[0]):
        padded = np.pad(realized[idx], (1, 1), mode="edge")
        smoothed[idx] = np.convolve(padded, kernel, mode="valid")
    return smoothed, realized


def _month_features(dates: list[str], idx: int, variant: str = "anbima") -> np.ndarray:
    d = dates[idx]
    mm = int(d[3:5])
    yyyy = int(d[6:10])
    trend = idx / 60.0

    if variant == "raw":
        # Apenas tendência linear + lags no regressor
        return np.array([trend])

    angle = 2.0 * np.pi * (mm - 1) / 12.0
    season = [
        np.sin(angle),
        np.cos(angle),
        1.0 if mm == 12 else 0.0,
        1.0 if mm in (1, 2) else 0.0,
    ]
    if variant == "season":
        # Lags + tendência linear + termos sazonais harmônicos e pontuais
        return np.array([trend] + season)

    start = f"{yyyy:04d}-{mm:02d}-01"
    end = f"{yyyy + 1:04d}-01-01" if mm == 12 else f"{yyyy:04d}-{mm + 1:02d}-01"

    if variant == "weekdays":
        # Dias úteis brutos (segunda a sexta, sem subtrair feriados bancários)
        raw_bdays = int(np.busday_count(start, end))
        return np.array([trend] + season + [float(raw_bdays - 21)])

    # variant == "anbima": dias úteis bancários nacionais ANBIMA
    calendar = _brazilian_bank_holidays(yyyy)
    bdays = int(np.busday_count(start, end, busdaycal=calendar))
    return np.array([trend] + season + [float(bdays - 21)])


def _fit_holt_winters_add(y: np.ndarray, period: int = 12) -> float:
    n = len(y)
    if n < 2 * period:
        return float(y[-1])
    level0 = float(np.mean(y[:period]))
    trend0 = float((np.mean(y[period : 2 * period]) - np.mean(y[:period])) / period)
    season0 = y[:period] - level0

    def sse(params: np.ndarray) -> float:
        alpha, beta, gamma = params
        level, trend = level0, trend0
        season = season0.copy()
        err = 0.0
        for t in range(n):
            idx = t % period
            pred = level + trend + season[idx]
            diff = y[t] - pred
            err += diff * diff
            prev_level = level
            level = alpha * (y[t] - season[idx]) + (1.0 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1.0 - beta) * trend
            season[idx] = gamma * (y[t] - level) + (1.0 - gamma) * season[idx]
        return err

    res = optimize.minimize(
        sse,
        x0=np.array([0.3, 0.05, 0.3]),
        bounds=[(0.05, 0.9), (0.01, 0.4), (0.05, 0.9)],
        method="L-BFGS-B",
    )
    alpha, beta, gamma = res.x
    level, trend = level0, trend0
    season = season0.copy()
    for t in range(n):
        idx = t % period
        prev_level = level
        level = alpha * (y[t] - season[idx]) + (1.0 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1.0 - beta) * trend
        season[idx] = gamma * (y[t] - level) + (1.0 - gamma) * season[idx]
    return max(0.0, float(level + trend + season[n % period]))


def _fit_ridge_seasonal(hist: np.ndarray, dates: list[str], target_idx: int, variant: str = "anbima") -> float:
    rows_x, rows_y = [], []
    for t in range(12, target_idx):
        feat = np.concatenate([[np.log(hist[t - 1]), np.log(hist[t - 12])], _month_features(dates, t, variant=variant)])
        rows_x.append(feat)
        rows_y.append(np.log(hist[t]))
    x_tr = np.asarray(rows_x)
    y_tr = np.asarray(rows_y)
    scaler = StandardScaler().fit(x_tr)
    reg = Ridge(alpha=5.0).fit(scaler.transform(x_tr), y_tr)
    x_next = np.concatenate([[np.log(hist[-1]), np.log(hist[-12])], _month_features(dates, target_idx, variant=variant)]).reshape(1, -1)
    return float(np.exp(reg.predict(scaler.transform(x_next))[0]))


def _forecast_origins(realized: np.ndarray, dates: list[str]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    totals = realized.sum(axis=0)
    month_count = realized.shape[1]
    for month in range(INITIAL_TRAIN_MONTHS, month_count):
        product_forecast = realized[:, month - 1]
        aggregate_forecast = float(totals[month - 1])
        actual = float(totals[month])
        product_ape = float(np.mean(np.abs(product_forecast / realized[:, month] - 1)))
        product_wmape = float(np.abs(product_forecast - realized[:, month]).sum() / actual)
        aggregate_ape = abs(aggregate_forecast / actual - 1)
        rows.append(
            {
                "month": month + 1,
                "date": dates[month],
                "actual": actual,
                "product_forecast": float(product_forecast.sum()),
                "aggregate_forecast": aggregate_forecast,
                "product_ape": product_ape,
                "product_wmape": product_wmape,
                "aggregate_ape": float(aggregate_ape),
                "aggregate_error": aggregate_forecast - actual,
            }
        )
    return rows


def _mase(rows: list[dict[str, float | int | str]], realized: np.ndarray, strategy: str) -> float:
    scaled_errors = []
    totals = realized.sum(axis=0)
    for row in rows:
        month = int(row["month"]) - 1
        if strategy == "product":
            scale = np.abs(np.diff(realized[:, :month], axis=1)).mean(axis=1)
            error = np.abs(realized[:, month] - realized[:, month - 1])
            scaled_errors.append(float(np.mean(error / scale)))
        else:
            scale = float(np.abs(np.diff(totals[:month])).mean())
            error = abs(float(row["aggregate_error"]))
            scaled_errors.append(error / scale)
    return float(np.mean(scaled_errors))


def _ljung_box(residuals: np.ndarray, lag: int = 4) -> tuple[float, float]:
    centered = residuals - residuals.mean()
    denominator = float(centered @ centered)
    if denominator == 0:
        return 0.0, 1.0
    n = len(centered)
    autocorrelations = [float(centered[k:] @ centered[:-k]) / denominator for k in range(1, lag + 1)]
    statistic = n * (n + 2) * sum(correlation**2 / (n - k) for k, correlation in enumerate(autocorrelations, start=1))
    return float(statistic), float(stats.chi2.sf(statistic, df=lag))


def _diebold_mariano(loss_difference: np.ndarray) -> tuple[float, float]:
    centered = loss_difference - loss_difference.mean()
    n = len(centered)
    long_run_variance = float(centered @ centered) / n
    max_lag = min(DM_HAC_LAG, n - 1)
    for lag in range(1, max_lag + 1):
        autocovariance = float(centered[lag:] @ centered[:-lag]) / n
        weight = 1 - lag / (max_lag + 1)
        long_run_variance += 2 * weight * autocovariance
    if long_run_variance <= 0:
        return 0.0, 1.0
    statistic = float(loss_difference.mean() / np.sqrt(long_run_variance / n))
    p_value = float(2 * stats.t.sf(abs(statistic), df=n - 1))
    return statistic, p_value


def run_model_benchmark(realized: np.ndarray, dates: list[str], series_meta: list[dict]) -> dict:
    """Compara 4 modelos out-of-sample em 33 origens completas (01/2024–09/2026)."""
    totals = realized.sum(axis=0)
    n_orig = realized.shape[1] - BENCHMARK_START_MONTH
    model_names = ("naive_t1", "seasonal_t12", "ets_hw", "ridge_cal")
    labels_pt = {
        "naive_t1": "Baseline Ingênuo (t-1)",
        "seasonal_t12": "Sazonal Ingênuo (t-12)",
        "ets_hw": "Holt-Winters Aditivo (ETS)",
        "ridge_cal": "Ridge Sazonal + Calendário Bancário",
    }
    preds = {m: np.zeros((5, n_orig)) for m in model_names}
    direct = {m: np.zeros(n_orig) for m in model_names}

    for j, t_idx in enumerate(range(BENCHMARK_START_MONTH, realized.shape[1])):
        for i in range(5):
            hist = realized[i, :t_idx]
            preds["naive_t1"][i, j] = hist[-1]
            preds["seasonal_t12"][i, j] = hist[-12]
            preds["ets_hw"][i, j] = _fit_holt_winters_add(hist)
            preds["ridge_cal"][i, j] = _fit_ridge_seasonal(hist, dates, t_idx)

        thist = totals[:t_idx]
        direct["naive_t1"][j] = thist[-1]
        direct["seasonal_t12"][j] = thist[-12]
        direct["ets_hw"][j] = _fit_holt_winters_add(thist)
        direct["ridge_cal"][j] = _fit_ridge_seasonal(thist, dates, t_idx)

    actual_sub = realized[:, BENCHMARK_START_MONTH:]
    actual_tot = totals[BENCHMARK_START_MONTH:]
    bench_dates = dates[BENCHMARK_START_MONTH:]

    summary_rows = []
    naive_bu_apes = np.abs(preds["naive_t1"].sum(axis=0) / actual_tot - 1.0)
    ridge_bu_apes = np.abs(preds["ridge_cal"].sum(axis=0) / actual_tot - 1.0)
    dm_ridge_stat, dm_ridge_p = _diebold_mariano(naive_bu_apes**2 - ridge_bu_apes**2)

    for m in model_names:
        p_mat = preds[m]
        bu_tot = p_mat.sum(axis=0)
        dir_tot = direct[m]
        bu_ape = np.abs(bu_tot / actual_tot - 1.0)
        dir_ape = np.abs(dir_tot / actual_tot - 1.0)
        lb_bu_stat, lb_bu_p = _ljung_box(bu_tot - actual_tot, lag=4)
        lb_dir_stat, lb_dir_p = _ljung_box(dir_tot - actual_tot, lag=4)
        summary_rows.append(
            {
                "key": m,
                "label": labels_pt[m],
                "unweighted_product_mape_pct": round(float(np.mean(np.abs(p_mat - actual_sub) / actual_sub) * 100.0), 2),
                "weighted_product_wmape_pct": round(float(np.abs(p_mat - actual_sub).sum() / actual_sub.sum() * 100.0), 2),
                "bottomup_mape_pct": round(float(np.mean(bu_ape) * 100.0), 2),
                "bottomup_mae_brl_bi": round(float(np.mean(np.abs(bu_tot - actual_tot)) / 1000.0), 2),
                "bottomup_within_5pct": round(float(np.mean(bu_ape <= TOLERANCE) * 100.0), 1),
                "bottomup_within_5_count": int(np.sum(bu_ape <= TOLERANCE)),
                "bottomup_ljung_box_stat": round(lb_bu_stat, 2),
                "bottomup_ljung_box_p": round(lb_bu_p, 4),
                "direct_mape_pct": round(float(np.mean(dir_ape) * 100.0), 2),
                "direct_mae_brl_bi": round(float(np.mean(np.abs(dir_tot - actual_tot)) / 1000.0), 2),
                "direct_within_5pct": round(float(np.mean(dir_ape <= TOLERANCE) * 100.0), 1),
                "direct_ljung_box_p": round(lb_dir_p, 4),
            }
        )

    per_modality_ridge = []
    for i, meta in enumerate(series_meta):
        n_ape = np.abs(preds["naive_t1"][i] / actual_sub[i] - 1.0) * 100.0
        r_ape = np.abs(preds["ridge_cal"][i] / actual_sub[i] - 1.0) * 100.0
        per_modality_ridge.append(
            {
                "label": meta["label"],
                "share_pct": round(float(actual_sub[i].sum() / actual_tot.sum() * 100.0), 1),
                "naive_mape_pct": round(float(np.mean(n_ape)), 2),
                "ridge_mape_pct": round(float(np.mean(r_ape)), 2),
                "delta_pp": round(float(np.mean(r_ape) - np.mean(n_ape)), 2),
            }
        )

    origin_series = [
        {
            "date": bench_dates[j],
            "actual_bi": round(float(actual_tot[j] / 1000.0), 2),
            "naive_bu_bi": round(float(preds["naive_t1"].sum(axis=0)[j] / 1000.0), 2),
            "ridge_bu_bi": round(float(preds["ridge_cal"].sum(axis=0)[j] / 1000.0), 2),
            "naive_ape_pct": round(float(naive_bu_apes[j] * 100.0), 2),
            "ridge_ape_pct": round(float(ridge_bu_apes[j] * 100.0), 2),
        }
        for j in range(n_orig)
    ]

    ablation_study = run_ablation_benchmark(realized, dates)

    naive_row = next(r for r in summary_rows if r["key"] == "naive_t1")
    ridge_row = next(r for r in summary_rows if r["key"] == "ridge_cal")
    return {
        "window": f"{bench_dates[0]} a {bench_dates[-1]}",
        "origins": n_orig,
        "models": summary_rows,
        "per_modality_naive_vs_ridge": per_modality_ridge,
        "dm_ridge_vs_naive_stat": round(dm_ridge_stat, 2),
        "dm_ridge_vs_naive_p": float(dm_ridge_p),
        "mae_reduction_brl_bi_month": round(
            naive_row["bottomup_mae_brl_bi"] - ridge_row["bottomup_mae_brl_bi"], 2
        ),
        "forecast_error_reduction_brl_bi_month": round(
            naive_row["bottomup_mae_brl_bi"] - ridge_row["bottomup_mae_brl_bi"], 2
        ),
        "budget_saving_brl_bi_month": round(
            naive_row["bottomup_mae_brl_bi"] - ridge_row["bottomup_mae_brl_bi"], 2
        ),
        "origin_series": origin_series,
        "ablation_study": ablation_study,
    }


def run_ablation_benchmark(realized: np.ndarray, dates: list[str]) -> list[dict]:
    """Estudo de ablação empírico out-of-sample (33 origens walk-forward):
    isola sequencialmente a contribuição de lags/tendência, sazonalidade,
    dias úteis brutos e feriados nacionais ANBIMA."""
    totals = realized.sum(axis=0)
    n_orig = realized.shape[1] - BENCHMARK_START_MONTH
    actual_tot = totals[BENCHMARK_START_MONTH:]

    # 0. Baseline Ingênuo
    naive_bu = realized[:, BENCHMARK_START_MONTH - 1 : -1].sum(axis=0)
    naive_mae = float(np.mean(np.abs(naive_bu - actual_tot)) / 1000.0)
    naive_wmape = float(np.abs(naive_bu - actual_tot).sum() / actual_tot.sum() * 100.0)
    naive_w5 = float(np.mean(np.abs(naive_bu / actual_tot - 1.0) <= TOLERANCE) * 100.0)

    ablation_specs = [
        ("naive_t1", "Baseline Ingênuo (t-1)", "Projeta o valor observado no mês imediatamente anterior", "naive"),
        ("ridge_raw", "Ridge (Lags + Tendência Linear)", "Lags t-1, t-12 e tendência linear (t/60)", "raw"),
        ("ridge_season", "Ridge + Sazonalidade Harmônica", "Acrescenta sen/cos harmônicos e dummies de dez/jan/fev", "season"),
        ("ridge_weekdays", "Ridge + Dias Úteis Brutos (Seg-Sex)", "Acrescenta contagem de segundas a sextas do mês sem feriados", "weekdays"),
        ("ridge_anbima", "Ridge + Feriados Bancários ANBIMA", "Subtrai feriados bancários nacionais oficiais ANBIMA", "anbima"),
    ]

    results = []
    prev_mae = naive_mae

    for key, label, desc, variant_code in ablation_specs:
        if variant_code == "naive":
            results.append({
                "key": key,
                "label": label,
                "description": desc,
                "bottomup_mae_brl_bi": round(naive_mae, 2),
                "bottomup_wmape_pct": round(naive_wmape, 2),
                "bottomup_within_5pct": round(naive_w5, 1),
                "incremental_reduction_bi": 0.0,
                "cumulative_reduction_bi": 0.0,
                "reduction_vs_naive_pct": 0.0,
            })
            continue

        p_bu = np.zeros((5, n_orig))
        for j, t in enumerate(range(BENCHMARK_START_MONTH, realized.shape[1])):
            for i in range(5):
                p_bu[i, j] = _fit_ridge_seasonal(realized[i, :t], dates, t, variant=variant_code)

        bu_tot = p_bu.sum(axis=0)
        bu_ape = np.abs(bu_tot / actual_tot - 1.0)
        mae_bu = float(np.mean(np.abs(bu_tot - actual_tot)) / 1000.0)
        wmape_bu = float(np.abs(bu_tot - actual_tot).sum() / actual_tot.sum() * 100.0)
        w5_bu = float(np.mean(bu_ape <= TOLERANCE) * 100.0)
        inc_red = round(prev_mae - mae_bu, 2)
        cum_red = round(naive_mae - mae_bu, 2)
        pct_red = round((naive_mae - mae_bu) / naive_mae * 100.0, 1)
        prev_mae = mae_bu

        results.append({
            "key": key,
            "label": label,
            "description": desc,
            "bottomup_mae_brl_bi": round(mae_bu, 2),
            "bottomup_wmape_pct": round(wmape_bu, 2),
            "bottomup_within_5pct": round(w5_bu, 1),
            "incremental_reduction_bi": inc_red,
            "cumulative_reduction_bi": cum_red,
            "reduction_vs_naive_pct": pct_red,
        })

    return results


def calculate_metrics(
    latent: np.ndarray, realized: np.ndarray
) -> tuple[dict[str, float | int | list | dict], list[dict[str, float | int | str]]]:
    snapshot = load_bcb_snapshot()
    dates = snapshot["dates"]
    series_meta = snapshot["series"]
    rows = _forecast_origins(realized, dates)

    product_ape = np.array([float(row["product_ape"]) for row in rows])
    aggregate_ape = np.array([float(row["aggregate_ape"]) for row in rows])
    aggregate_errors = np.array([float(row["aggregate_error"]) for row in rows])
    t_test = stats.ttest_rel(product_ape, aggregate_ape)

    loss_difference = product_ape**2 - aggregate_ape**2
    dm_statistic, dm_p_value = _diebold_mariano(loss_difference)
    ljung_box_statistic, ljung_box_p_value = _ljung_box(aggregate_errors)

    mom_changes = realized[:, 1:] / realized[:, :-1] - 1.0
    totals = realized.sum(axis=0)
    agg_mom_changes = totals[1:] / totals[:-1] - 1.0
    noise_mom_pct = float(np.std(mom_changes, ddof=1) * 100.0)
    agg_mom_std_pct = float(np.std(agg_mom_changes, ddof=1) * 100.0)

    per_modality = []
    for idx, meta in enumerate(series_meta):
        mod_mom_std = float(np.std(mom_changes[idx], ddof=1) * 100.0)
        mod_apes = [
            abs(realized[idx, int(r["month"]) - 2] / realized[idx, int(r["month"]) - 1] - 1.0)
            for r in rows
        ]
        per_modality.append(
            {
                "id": meta["id"],
                "key": meta["key"],
                "label": meta["label"],
                "mom_std_percent": mod_mom_std,
                "mape_percent": float(np.mean(mod_apes) * 100.0),
            }
        )

    benchmark = run_model_benchmark(realized, dates, series_meta)

    metrics: dict[str, float | int | list | dict] = {
        "months": realized.shape[1],
        "forecast_origins": len(rows),
        "noise_mom_percent": noise_mom_pct,
        "aggregate_mom_std_percent": agg_mom_std_pct,
        "tolerance_percent": TOLERANCE * 100,
        "product_mape_percent": float(product_ape.mean() * 100),
        "aggregate_mape_percent": float(aggregate_ape.mean() * 100),
        "product_within_tolerance_count": int(np.sum(product_ape <= TOLERANCE)),
        "aggregate_within_tolerance_count": int(np.sum(aggregate_ape <= TOLERANCE)),
        "product_within_tolerance_percent": float(np.mean(product_ape <= TOLERANCE) * 100),
        "aggregate_within_tolerance_percent": float(np.mean(aggregate_ape <= TOLERANCE) * 100),
        "product_mase": _mase(rows, realized, "product"),
        "aggregate_mase": _mase(rows, realized, "aggregate"),
        "paired_t_statistic": float(t_test.statistic),
        "paired_t_p_value": float(t_test.pvalue),
        "ljung_box_lag": 4,
        "ljung_box_statistic": ljung_box_statistic,
        "ljung_box_p_value": ljung_box_p_value,
        "dm_statistic": dm_statistic,
        "dm_p_value": dm_p_value,
        "dm_hac_lag": DM_HAC_LAG,
        "per_modality": per_modality,
        "benchmark": benchmark,
    }
    return metrics, rows


def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("#FFFFFF")
    axis.grid(axis="y", color="#F1F5F9", linewidth=0.9, linestyle="--")
    # Limite visual nítido e profissional em todo o contorno da área de dados
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#CBD5E1")
        spine.set_linewidth(1.0)
    axis.tick_params(colors=PALETTE["muted"], length=4, width=0.9)


def _write_real_vitrine_svg(bench: dict) -> None:
    """Gera a capa executiva vitrine_estudo_04_bcb.svg a partir dos pontos REAIS do backtest."""
    if not VITRINE_SVG_PATH.parent.is_dir():
        return
    series = bench["origin_series"][-18:]  # últimos 18 meses reais (04/2025-09/2026)
    max_ape = max(max(p["naive_ape_pct"], p["ridge_ape_pct"]) for p in series)
    y_ceil = max(12.0, float(np.ceil(max_ape / 2.0) * 2.0))

    x0, x1 = 44.0, 336.0
    y_top, y_bot = 148.0, 266.0

    def _pts(key: str) -> str:
        out = []
        for idx, pt in enumerate(series):
            x = x0 + idx * (x1 - x0) / max(1, len(series) - 1)
            y = y_bot - min(pt[key], y_ceil) / y_ceil * (y_bot - y_top)
            out.append(f"{x:.1f},{y:.1f}")
        return " ".join(out)

    naive_pts = _pts("naive_ape_pct")
    ridge_pts = _pts("ridge_ape_pct")
    area_pts = f"{x0:.1f},{y_bot:.1f} {ridge_pts} {x1:.1f},{y_bot:.1f}"
    band_y = y_bot - (5.0 / y_ceil) * (y_bot - y_top)

    naive_row = next(r for r in bench["models"] if r["key"] == "naive_t1")
    ridge_row = next(r for r in bench["models"] if r["key"] == "ridge_cal")
    mae_reduction = bench["mae_reduction_brl_bi_month"]

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 380 460" width="380" height="460" role="img" aria-labelledby="t04 d04">
  <title id="t04">Backtest Walk-Forward BCB SGS — Ridge Sazonal vs Baseline t-1</title>
  <desc id="d04">Curvas reais dos últimos 18 meses out-of-sample (04/2025–09/2026): Bottom-Up Ridge reduz o erro percentual agregado para {ridge_row['bottomup_mape_pct']:.2f}%; a diferença de MAE médio é R$ {mae_reduction:.2f} bilhões/mês, sem representar economia de caixa realizada.</desc>
  <defs>
    <linearGradient id="bg4" x1="0" y1="0" x2="0.8" y2="1">
      <stop offset="0%" stop-color="#0B2038"/>
      <stop offset="100%" stop-color="#0F172A"/>
    </linearGradient>
    <linearGradient id="areaAgg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#38BDF8" stop-opacity="0.34"/>
      <stop offset="100%" stop-color="#38BDF8" stop-opacity="0.02"/>
    </linearGradient>
    <pattern id="grid4" width="28" height="28" patternUnits="userSpaceOnUse">
      <path d="M 28 0 L 0 0 0 28" fill="none" stroke="rgba(148,163,184,0.08)" stroke-width="1"/>
    </pattern>
  </defs>

  <rect width="380" height="460" rx="12" fill="url(#bg4)"/>
  <rect width="380" height="460" rx="12" fill="url(#grid4)"/>

  <rect x="24" y="22" width="256" height="28" rx="14" fill="rgba(56,189,248,0.14)" stroke="rgba(56,189,248,0.45)" stroke-width="1"/>
  <text x="152" y="40" text-anchor="middle" fill="#7DD3FC" font-family="system-ui,-apple-system,Segoe UI,sans-serif" font-size="11" font-weight="800" letter-spacing="0.5">BCB SGS · 4 MODELOS WALK-FORWARD</text>

  <text x="24" y="80" fill="#F8FAFC" font-family="system-ui,-apple-system,Segoe UI,sans-serif" font-size="21" font-weight="800">Previsão de Crédito PF</text>
  <text x="24" y="102" fill="#94A3B8" font-family="system-ui,-apple-system,Segoe UI,sans-serif" font-size="13" font-weight="500">wMAPE &amp; Bottom-Up Ridge (33 origens reais)</text>

  <rect x="24" y="122" width="332" height="160" rx="10" fill="rgba(15,23,42,0.75)" stroke="rgba(148,163,184,0.18)" stroke-width="1"/>
  <rect x="38" y="{band_y:.1f}" width="304" height="{y_bot - band_y:.1f}" rx="4" fill="rgba(56,189,248,0.08)"/>
  <line x1="38" y1="{band_y:.1f}" x2="342" y2="{band_y:.1f}" stroke="#38BDF8" stroke-width="1.3" stroke-dasharray="5 4"/>
  <text x="336" y="{band_y - 5:.1f}" text-anchor="end" fill="#7DD3FC" font-family="system-ui,-apple-system,Segoe UI,sans-serif" font-size="10.5" font-weight="700">Meta ±5,0% APE</text>

  <polygon fill="url(#areaAgg)" points="{area_pts}"/>
  <polyline fill="none" stroke="#F472B6" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" points="{naive_pts}"/>
  <polyline fill="none" stroke="#38BDF8" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round" points="{ridge_pts}"/>

  <circle cx="44" cy="140" r="4" fill="#F472B6"/>
  <text x="54" y="144" fill="#F8FAFC" font-family="system-ui,-apple-system,Segoe UI,sans-serif" font-size="11.5" font-weight="700">Naïve (t-1)</text>
  <circle cx="155" cy="140" r="4" fill="#38BDF8"/>
  <text x="165" y="144" fill="#E0F2FE" font-family="system-ui,-apple-system,Segoe UI,sans-serif" font-size="11.5" font-weight="700">Ridge + Calendário Bancário</text>

  <rect x="24" y="298" width="158" height="138" rx="10" fill="rgba(30,41,59,0.75)" stroke="rgba(244,114,182,0.38)" stroke-width="1"/>
  <text x="38" y="324" fill="#F472B6" font-family="system-ui,-apple-system,Segoe UI,sans-serif" font-size="10.5" font-weight="800" letter-spacing="0.5">INGÊNUO (t-1)</text>
  <text x="38" y="365" fill="#FFFFFF" font-family="system-ui,-apple-system,Segoe UI,sans-serif" font-size="31" font-weight="800">{str(naive_row['bottomup_mape_pct']).replace('.', ',')}%</text>
  <text x="38" y="390" fill="#CBD5E1" font-family="system-ui,-apple-system,Segoe UI,sans-serif" font-size="12" font-weight="600">Erro R$ {str(naive_row['bottomup_mae_brl_bi']).replace('.', ',')} bi/mês</text>
  <text x="38" y="416" fill="#94A3B8" font-family="system-ui,-apple-system,Segoe UI,sans-serif" font-size="11.5" font-weight="700">Ljung-Box p={str(naive_row['bottomup_ljung_box_p']).replace('.', ',')}</text>

  <rect x="198" y="298" width="158" height="138" rx="10" fill="rgba(14,116,144,0.28)" stroke="rgba(56,189,248,0.68)" stroke-width="1.4"/>
  <text x="212" y="324" fill="#7DD3FC" font-family="system-ui,-apple-system,Segoe UI,sans-serif" font-size="10.5" font-weight="800" letter-spacing="0.5">RIDGE BOTTOM-UP</text>
  <text x="212" y="365" fill="#FFFFFF" font-family="system-ui,-apple-system,Segoe UI,sans-serif" font-size="31" font-weight="800">{str(ridge_row['bottomup_mape_pct']).replace('.', ',')}%</text>
  <text x="212" y="390" fill="#BAE6FD" font-family="system-ui,-apple-system,Segoe UI,sans-serif" font-size="12" font-weight="600">MAE médio: R$ {str(ridge_row['bottomup_mae_brl_bi']).replace('.', ',')} bi/mês</text>
  <text x="212" y="416" fill="#34D399" font-family="system-ui,-apple-system,Segoe UI,sans-serif" font-size="11.5" font-weight="800">{str(ridge_row['bottomup_within_5pct']).replace('.', ',')}% em ±5% (p={str(ridge_row['bottomup_ljung_box_p']).replace('.', ',')})</text>
</svg>
"""
    VITRINE_SVG_PATH.write_text(svg, encoding="utf-8")


def generate_figures(
    latent: np.ndarray,
    realized: np.ndarray,
    rows: list[dict[str, float | int | str]],
    metrics: dict,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["text.parse_math"] = False
    bench = metrics["benchmark"]
    origin_series = bench["origin_series"]
    dates = [p["date"] for p in origin_series]
    short_d = [f"{d[3:5]}/{d[8:10]}" for d in dates]
    x_idx = np.arange(len(origin_series))

    # Fig 1: Concessões PF Agregadas — Realizado vs Ridge Bottom-Up vs Naive (t-1)
    fig, axis = plt.subplots(figsize=(11.4, 6.2), layout="constrained")
    _style_axis(axis)
    actual_vals = [p["actual_bi"] for p in origin_series]
    naive_vals = [p["naive_bu_bi"] for p in origin_series]
    ridge_vals = [p["ridge_bu_bi"] for p in origin_series]
    axis.plot(x_idx, actual_vals, color=PALETTE["navy"], linewidth=2.5, label="Realizado BCB SGS (R$ bi)")
    naive_row = next(model for model in bench["models"] if model["key"] == "naive_t1")
    ridge_row = next(model for model in bench["models"] if model["key"] == "ridge_cal")
    axis.plot(x_idx, naive_vals, color=PALETTE["magenta"], linewidth=1.7, linestyle="--", alpha=0.85, label=f"Ingênuo (t-1): MAE R$ {naive_row['bottomup_mae_brl_bi']:.2f} bi/mês")
    axis.plot(x_idx, ridge_vals, color=PALETTE["blue"], linewidth=2.2, marker="o", markersize=4.0, label=f"Ridge + calendário bancário: MAE R$ {ridge_row['bottomup_mae_brl_bi']:.2f} bi/mês")
    # Highlight 03/2026 divergence where calendar effect hits naive forecast
    mar26_idx = dates.index("01/03/2026") if "01/03/2026" in dates else len(dates) - 7
    axis.annotate(
        f"03/26: Real R$ {actual_vals[mar26_idx]:.0f} bi\nIngênuo subestima em R$ {actual_vals[mar26_idx] - naive_vals[mar26_idx]:.0f} bi",
        xy=(mar26_idx, naive_vals[mar26_idx]),
        xytext=(mar26_idx - 7.2, naive_vals[mar26_idx] - 32),
        fontsize=8.3,
        fontweight="bold",
        color="#9D174D",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#FBCFE8"),
        arrowprops=dict(arrowstyle="->", color="#D91568", lw=1.2),
    )
    ticks = list(range(0, len(origin_series), 4))
    axis.set_xticks(ticks)
    axis.set_xticklabels([short_d[i] for i in ticks])
    # Respiro amplo no eixo Y para impedir qualquer proximidade com a legenda ou teto
    axis.set_ylim(280, 448)
    axis.set_title(
        f"Ridge Sazonal + Calendário Bancário reduz MAE de R$ {naive_row['bottomup_mae_brl_bi']:.2f} bi para R$ {ridge_row['bottomup_mae_brl_bi']:.2f} bi/mês",
        pad=44,
        fontweight="bold",
        fontsize=11.0,
        color=PALETTE["navy"],
    )
    axis.set(
        xlabel="Origem out-of-sample (01/2024–09/2026 · 33 meses)",
        ylabel="Concessões totais 5 linhas PF (R$ bilhões)",
    )
    axis.legend(
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        framealpha=0.98,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=3,
        fontsize=9.0,
        borderpad=0.5,
        handletextpad=0.6,
    )
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"01_projetado_vs_realizado.{extension}", dpi=180, facecolor="white")
    plt.close(fig)

    # Fig 2: Comparativo 4 Modelos (wMAPE por Produto vs Bottom-Up Agregado + Erro % ao longo do tempo)
    fig, (ax_bar, ax_ts) = plt.subplots(1, 2, figsize=(12.4, 6.2), layout="constrained", gridspec_kw={"width_ratios": [1.0, 1.38]})
    _style_axis(ax_bar)
    models = bench["models"]
    labels = ["Ingênuo\n(t-1)", "Sazonal\n(t-12)", "Holt-Winters\n(ETS)", "Ridge +\nCalendário"]
    x_m = np.arange(len(models))
    w = 0.36
    bars_w = ax_bar.bar(x_m - w / 2, [m["weighted_product_wmape_pct"] for m in models], width=w, color=PALETTE["purple"], label="wMAPE por Produto (pond. R$)")
    bars_b = ax_bar.bar(x_m + w / 2, [m["bottomup_mape_pct"] for m in models], width=w, color=PALETTE["blue"], label="MAPE Bottom-Up Agregado")
    for bar in (*bars_w, *bars_b):
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 0.25, f"{h:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color=PALETTE["navy"])
    ax_bar.set_xticks(x_m)
    ax_bar.set_xticklabels(labels, fontsize=8.5)
    ax_bar.set_ylim(0, 15.0)
    ax_bar.set_title("Benchmark wMAPE & Bottom-Up (33 origens)", fontweight="bold", fontsize=10.5, color=PALETTE["navy"], pad=46)
    ax_bar.set_ylabel("Erro Percentual (%)")
    ax_bar.legend(
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        framealpha=0.98,
        fontsize=8.2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        borderpad=0.4,
    )

    _style_axis(ax_ts)
    naive_ape = [p["naive_ape_pct"] for p in origin_series]
    ridge_ape = [p["ridge_ape_pct"] for p in origin_series]
    naive_row = next(model for model in models if model["key"] == "naive_t1")
    ridge_row = next(model for model in models if model["key"] == "ridge_cal")
    ax_ts.plot(x_idx, naive_ape, color=PALETTE["magenta"], linewidth=1.8, marker="o", markersize=3, label=f"Ingênuo (t-1): {naive_row['bottomup_within_5pct']:.1f}% na meta ±5%")
    ax_ts.plot(x_idx, ridge_ape, color=PALETTE["emerald"], linewidth=2.2, marker="s", markersize=3.2, label=f"Ridge Bottom-Up: {ridge_row['bottomup_within_5pct']:.1f}% na meta ±5%")
    ax_ts.axhline(TOLERANCE * 100, color=PALETTE["navy"], linestyle="--", linewidth=1.2, label="Meta ±5,0%")
    peak_i = int(np.argmax(naive_ape))
    ax_ts.annotate(
        f"Choque {short_d[peak_i]}: {naive_ape[peak_i]:.1f}% vs {ridge_ape[peak_i]:.1f}%",
        xy=(peak_i, naive_ape[peak_i]),
        xytext=(peak_i - 12.5, naive_ape[peak_i] - 0.6),
        fontsize=8.1,
        fontweight="bold",
        color="#9D174D",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#FBCFE8"),
        arrowprops=dict(arrowstyle="->", color="#D91568", lw=1.1),
    )
    ax_ts.set_xticks(ticks)
    ax_ts.set_xticklabels([short_d[i] for i in ticks])
    ax_ts.set_ylim(0, 16.5)
    ax_ts.set_title(f"Erro Agregado por Mês Previsto (Ljung-Box p={ridge_row['bottomup_ljung_box_p']:.3f})", fontweight="bold", fontsize=10.5, color=PALETTE["navy"], pad=46)
    ax_ts.set_xlabel("Mês previsto (MM/AA)")
    ax_ts.set_ylabel("APE Mensal (%)")
    ax_ts.legend(
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        framealpha=0.98,
        fontsize=8.0,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        borderpad=0.4,
    )

    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"02_backtest_produto_vs_agregado.{extension}", dpi=180, facecolor="white")
    plt.close(fig)

    _write_real_vitrine_svg(bench)


def run_experiment(output_dir: Path, refresh_bcb: bool = False) -> dict[str, float | int | list | dict]:
    if refresh_bcb:
        refresh_bcb_snapshot()
    snapshot = load_bcb_snapshot()
    latent, realized = generate_series()
    metrics, rows = calculate_metrics(latent, realized)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["text.parse_math"] = False
    (output_dir / "resultados.json").write_text(
        json.dumps(
            {
                "source": snapshot["source"],
                "unit": snapshot["unit"],
                "window": f"{snapshot['dates'][0]} a {snapshot['dates'][-1]}",
                "design": {
                    "modalities": len(BCB_SERIES),
                    "months": realized.shape[1],
                    "initial_training_months": INITIAL_TRAIN_MONTHS,
                    "benchmark_start_month": BENCHMARK_START_MONTH,
                    "forecast_horizon": 1,
                },
                "metrics": metrics,
                "walk_forward": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    generate_figures(latent, realized, rows, metrics, output_dir / "figuras")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa o backtest walk-forward multi-modelo com dados oficiais BCB SGS.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STUDY_ROOT / "resultados",
        help="Diretório para resultados JSON e figuras.",
    )
    parser.add_argument(
        "--refresh-bcb",
        action="store_true",
        help="Baixa séries atualizadas diretamente da API do Banco Central (SGS).",
    )
    args = parser.parse_args()
    metrics = run_experiment(args.output_dir.resolve(), refresh_bcb=args.refresh_bcb)
    print(json.dumps(metrics["benchmark"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
