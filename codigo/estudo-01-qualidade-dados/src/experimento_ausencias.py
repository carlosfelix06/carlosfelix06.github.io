"""Compara ausencias artificiais em modelos de inadimplencia da base UCI.

Uso: python src/experimento_ausencias.py
O teste e avaliado somente apos as configuracoes abaixo estarem fixadas.
"""

from __future__ import annotations

import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dados" / "processed"
OUT = ROOT / "resultados" / "tabelas"
TARGET = "default_payment_next_month"
KEY = "ID"
SEEDS = (2026, 2027, 2028, 2029, 2030)
RATES = (0.10, 0.30, 0.50)
CATEGORICAL = (
    "SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3",
    "PAY_4", "PAY_5", "PAY_6",
)
# Fonte/processo mensal de faturas e pagamentos: 18 de 23 preditores.
CONCENTRATED = (
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
)
SCENARIOS = (("referencia", 0.0),) + tuple(
    (mechanism, rate) for rate in RATES for mechanism in ("mcar", "concentrada")
)


def load_partitions() -> dict[str, pd.DataFrame]:
    parts = {name: pd.read_parquet(DATA / f"{name}.parquet") for name in ("treino", "validacao", "teste")}
    columns = list(parts["treino"].columns)
    if KEY not in columns or TARGET not in columns or len(columns) != 25:
        raise ValueError("Particoes devem conter ID, alvo e 23 preditores; reexecute a preparacao.")
    ids = pd.concat([part[KEY] for part in parts.values()], ignore_index=True)
    if ids.isna().any() or ids.nunique() != len(ids):
        raise ValueError("ID ausente ou repetido entre as particoes.")
    for name, part in parts.items():
        if list(part.columns) != columns or part.isna().any().any():
            raise ValueError(f"Esquema divergente ou nulos originais na particao {name}.")
        if set(part[TARGET].unique()) != {0, 1}:
            raise ValueError(f"Alvo inesperado na particao {name}.")
    return parts


def overlap_mask(parts: dict[str, pd.DataFrame], columns: list[str]) -> np.ndarray:
    """Marca linhas do teste com preditores idênticos em treino/validação."""
    prior = pd.concat([parts["treino"][columns], parts["validacao"][columns]], ignore_index=True)
    prior = prior.drop_duplicates().assign(_present=1)
    matched = parts["teste"][columns].merge(prior, on=columns, how="left", sort=False)
    if len(matched) != len(parts["teste"]):
        raise ValueError("Auditoria de sobreposicao alterou o numero de linhas do teste.")
    return matched["_present"].notna().to_numpy()


def mask_predictors(frame: pd.DataFrame, columns: list[str], mechanism: str, rate: float, seed: int, partition: int) -> tuple[pd.DataFrame, int]:
    result = frame[columns].copy().astype(float)
    count = round(rate * len(result) * len(columns))
    if mechanism == "referencia":
        return result, 0
    eligible = columns if mechanism == "mcar" else list(CONCENTRATED)
    capacity = len(result) * len(eligible)
    if count > capacity:
        raise ValueError("Grupo concentrado nao comporta a taxa especificada.")
    # Sementes deterministicas distintas por particao e mecanismo; sem consultar alvo/valores.
    rng = np.random.default_rng(seed * 100 + partition * 10 + (0 if mechanism == "mcar" else 1))
    flat = rng.choice(capacity, size=count, replace=False)
    row_index, column_index = np.divmod(flat, len(eligible))
    array = result[eligible].to_numpy(copy=True)
    array[row_index, column_index] = np.nan
    result.loc[:, eligible] = array
    assert int(result.isna().sum().sum()) == count
    return result, count


def transform(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, numeric: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    frames = (train, validation, test)
    numeric_imputer = SimpleImputer(strategy="median")
    num = (numeric_imputer.fit_transform(train[numeric]),) + tuple(numeric_imputer.transform(f[numeric]) for f in frames[1:])
    # Categorias permanecem codigos nominais; nenhuma ordem numerica e imposta.
    cat_frames = [f[list(CATEGORICAL)].astype(object) for f in frames]
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    cat = (categorical_imputer.fit_transform(cat_frames[0]),) + tuple(categorical_imputer.transform(f) for f in cat_frames[1:])
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    encoded = (encoder.fit_transform(cat[0]),) + tuple(encoder.transform(c) for c in cat[1:])
    flags = tuple(f.isna().to_numpy(dtype=np.float32) for f in frames)
    scaler = StandardScaler()
    scaled = (scaler.fit_transform(num[0]),) + tuple(scaler.transform(n) for n in num[1:])
    return {
        "logistica": tuple(np.column_stack((n, c, m)).astype(np.float32) for n, c, m in zip(scaled, encoded, flags)),
        "hist_gradient_boosting": tuple(np.column_stack((n, c, m)).astype(np.float32) for n, c, m in zip(num, encoded, flags)),
    }


def calibration(y: np.ndarray, p: np.ndarray, ids: np.ndarray) -> tuple[float, float, list[dict]]:
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    logits = logit(clipped)
    alpha = brentq(lambda a: float(np.sum(expit(logits + a) - y)), -100, 100)
    fit = LogisticRegression(C=np.inf, max_iter=2000).fit(logits.reshape(-1, 1), y)
    slope = float(fit.coef_[0, 0])
    order = np.lexsort((ids, p))
    curve = []
    for group, idx in enumerate(np.array_split(order, 10), start=1):
        curve.append({"grupo": group, "n": len(idx), "prob_media": float(p[idx].mean()), "taxa_observada": float(y[idx].mean())})
    return float(alpha), slope, curve


def metrics(y: np.ndarray, p: np.ndarray, ids: np.ndarray) -> tuple[dict, list[dict]]:
    order = np.lexsort((ids, -p))
    top = order[: math.ceil(0.1 * len(order))]
    alpha, slope, curve = calibration(y, p, ids)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "captura_top_10": float(y[top].sum() / y.sum()),
        "calibracao_intercepto": alpha,
        "calibracao_inclinacao": slope,
    }, curve


def run() -> None:
    parts = load_partitions()
    columns = [c for c in parts["treino"] if c not in (KEY, TARGET)]
    if len(columns) != 23 or not set(CONCENTRATED).issubset(columns) or not set(CATEGORICAL).issubset(columns):
        raise ValueError("Preditores divergentes do protocolo congelado.")
    numeric = [c for c in columns if c not in CATEGORICAL]
    labels = {name: part[TARGET].to_numpy(dtype=int) for name, part in parts.items()}
    test_ids = parts["teste"][KEY].to_numpy()
    duplicated_predictors = overlap_mask(parts, columns)
    clean = ~duplicated_predictors
    rows, rows_sensitivity, curves, counts = [], [], [], []
    reference = {}
    reference_sensitivity = {}
    for seed in SEEDS:
        for mechanism, rate in SCENARIOS:
            masked = {}
            for partition, (name, part) in enumerate(parts.items()):
                masked[name], count = mask_predictors(part, columns, mechanism, rate, seed, partition)
                counts.append({"semente": seed, "cenario": mechanism, "taxa": rate, "particao": name, "celulas_mascaradas": count})
            matrices = transform(masked["treino"], masked["validacao"], masked["teste"], numeric)
            for model_name, (x_train, _x_valid, x_test) in matrices.items():
                if mechanism == "referencia" and model_name in reference:
                    result, curve = reference[model_name]
                    sensitivity = reference_sensitivity[model_name]
                else:
                    model = (
                        LogisticRegression(solver="lbfgs", max_iter=2000, class_weight=None)
                        if model_name == "logistica" else
                        HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=0.0, early_stopping=False, random_state=42)
                    )
                    model.fit(x_train, labels["treino"])
                    probabilities = model.predict_proba(x_test)[:, 1]
                    result, curve = metrics(labels["teste"], probabilities, test_ids)
                    sensitivity, _ = metrics(labels["teste"][clean], probabilities[clean], test_ids[clean])
                    if mechanism == "referencia":
                        reference[model_name] = (result, curve)
                        reference_sensitivity[model_name] = sensitivity
                context = {"modelo": model_name, "cenario": mechanism, "taxa": rate, "semente": seed}
                rows.append({**context, **result})
                rows_sensitivity.append({**context, "n_teste": int(clean.sum()), "n_positivos": int(labels["teste"][clean].sum()), **sensitivity})
                curves.extend({**context, **point} for point in curve)
            print(f"Concluido: semente={seed} cenario={mechanism} taxa={rate}", flush=True)

    results = pd.DataFrame(rows)
    sensitivity_results = pd.DataFrame(rows_sensitivity)
    measure_cols = ["roc_auc", "ap", "brier", "captura_top_10", "calibracao_intercepto", "calibracao_inclinacao"]
    for measure in measure_cols:
        baseline = results[results.cenario == "referencia"].set_index(["modelo", "semente"])[measure]
        results[f"delta_{measure}"] = [
            value - baseline.loc[(model, seed)]
            for model, seed, value in zip(results.modelo, results.semente, results[measure])
        ]
    group = results.groupby(["modelo", "cenario", "taxa"], sort=False)
    summary = group[measure_cols + [f"delta_{c}" for c in measure_cols]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    sensitivity_summary = sensitivity_results.groupby(["modelo", "cenario", "taxa"], sort=False)[measure_cols].agg(["mean", "std"])
    sensitivity_summary.columns = [f"{metric}_{stat}" for metric, stat in sensitivity_summary.columns]
    sensitivity_summary = sensitivity_summary.reset_index()
    evidence = []
    for (model, scenario, rate), subset in group:
        if scenario == "referencia":
            continue
        matches = {
            "ap": bool((subset.delta_ap < 0).all()),
            "brier": bool((subset.delta_brier > 0).all()),
            "captura_top_10": bool((subset.delta_captura_top_10 < 0).all()),
        }
        evidence.append({"modelo": model, "cenario": scenario, "taxa": rate, **matches, "criterio_exploratorio": any(matches.values())})
    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / "metricas_repeticoes.csv", index=False)
    summary.to_csv(OUT / "resumo_metricas.csv", index=False)
    sensitivity_results.to_csv(OUT / "sensibilidade_sem_sobreposicao.csv", index=False)
    sensitivity_summary.to_csv(OUT / "resumo_sensibilidade.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "curvas_calibracao.csv", index=False)
    pd.DataFrame(counts).to_csv(OUT / "contagens_mascaras.csv", index=False)
    pd.DataFrame(evidence).to_csv(OUT / "criterio_evidencia.csv", index=False)
    manifest = {
        "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
        "fonte_manifesto": "manifesto_processamento.json",
        "particoes": {name: len(part) for name, part in parts.items()},
        "teste_preditores_identicos_treino_ou_validacao": int(duplicated_predictors.sum()),
        "sensibilidade_teste_sem_sobreposicao": int(clean.sum()),
        "chave_rastreio_fora_dos_preditores": KEY,
        "alvo": TARGET,
        "preditores": columns,
        "categoricos_nominais": list(CATEGORICAL),
        "numericos": numeric,
        "grupo_concentrado": list(CONCENTRATED),
        "taxas": list(RATES),
        "sementes_mascaras": list(SEEDS),
        "mascaras": "contagem exata por particao; amostragem uniforme sem reposicao; independente do alvo e dos valores",
        "preparacao": "mediana numerica, moda categorica, one-hot, indicadores de ausencia de todos os preditores; ajuste somente no treino",
        "modelos": {
            "logistica": {"solver": "lbfgs", "max_iter": 2000, "class_weight": None, "escala_numerica": True},
            "hist_gradient_boosting": {"max_iter": 100, "learning_rate": 0.1, "max_leaf_nodes": 31, "l2_regularization": 0.0, "early_stopping": False, "escala_numerica": False},
        },
        "hiperparametros_ajustados": False,
        "validacao_usada_para_escolhas": False,
        "testes_por_repeticao": 1,
        "ambiente": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
    }
    (OUT / "manifesto_experimento.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
