from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special, stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

STUDY_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = STUDY_ROOT / "dados"
SNAPSHOT_PATH = DATA_DIR / "givemesomecredit_openml_46929.csv.gz"
METADATA_PATH = DATA_DIR / "dataset_metadata.json"
RESULTS_DIR = STUDY_ROOT / "resultados"
FIGURES_DIR = RESULTS_DIR / "figuras"
RESULTS_JSON_PATH = RESULTS_DIR / "resultados.json"

OPENML_DATA_ID = 46929
TARGET_COL = "FinancialDistressNextTwoYears"
RAW_COLUMNS = [
    TARGET_COL,
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

FEATURES_LP = [
    "age",
    "income_missing",
    "log_income",
    "dependents_missing",
    "dependents",
    "open_lines",
    "real_estate_loans",
    "real_estate_share",
    "has_real_estate",
    "zero_open_lines",
    "income_per_dep",
]

FEATURES_CP = [
    "rev_util_capped",
    "log_rev_util",
    "rev_over_08",
    "debt_ratio_known",
    "debt_amt_no_income",
    "d30",
    "d60",
    "d90",
    "severity_idx",
]

PALETTE = {
    "navy": "#0B2038",
    "magenta": "#D91568",
    "purple": "#6A1B9A",
    "blue": "#2878E8",
    "emerald": "#10B981",
    "amber": "#D97706",
    "muted": "#64748B",
    "grid": "#E2E8F0",
    "bg": "#F8FAFC",
}

LP_LABELS = ("Baixo Score LP", "Médio Score LP", "Alto Score LP")
CP_LABELS = ("Risco CP Baixo", "Risco CP Médio", "Risco CP Alto")

# Mapeamento oficial da Matriz 3x3: (lp_idx, cp_idx) -> Cluster ('A', 'B', 'C')
# lp_idx: 0=Baixo Score LP, 1=Médio Score LP, 2=Alto Score LP
# cp_idx: 0=Risco CP Baixo, 1=Risco CP Médio, 2=Risco CP Alto
MATRIX_3X3_CLUSTER_MAP: dict[tuple[int, int], str] = {
    (2, 0): "A",  # Alto Score LP + Risco CP Baixo -> Aprova 100%
    (1, 0): "A",  # Médio Score LP + Risco CP Baixo -> Aprova 100%
    (0, 0): "B",  # Baixo Score LP + Risco CP Baixo -> Contraproposta / Alçada parcial
    (1, 1): "B",  # Médio Score LP + Risco CP Médio -> Contraproposta / Alçada parcial
    (2, 1): "B",  # Alto Score LP + Risco CP Médio -> Contraproposta / Alçada parcial
    (0, 1): "C",  # Baixo Score LP + Risco CP Médio -> Reprova
    (0, 2): "C",  # Baixo Score LP + Risco CP Alto -> Reprova
    (1, 2): "C",  # Médio Score LP + Risco CP Alto -> Reprova
    (2, 2): "C",  # Alto Score LP + Risco CP Alto -> EARLY WARNING (Reprova preventiva!)
}

UNIT_ECONOMICS = {
    "ead_padrao_um": 10000.0,
    "cluster_b_fator_ead": 0.65,
    "margem_liquida_24m_padrao": 0.055,
    "margem_liquida_24m_cluster_b": 0.070,
    "opex_originacao_um": 100.0,
    "lgd_base": 0.65,
    "lgd_grid": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
}


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class ModelBundle:
    lp_model: Pipeline
    rf_base: Pipeline
    rf_cal_isotonic: CalibratedClassifierCV
    rf_cal_sigmoid: CalibratedClassifierCV
    cal_method: str
    blend_2d: LogisticRegression
    val_calibration_metrics: dict[str, dict[str, float]]
    val_r4_calibration_metrics: dict[str, dict[str, float]]
    lp_terciles: tuple[float, float]
    cp_terciles: tuple[float, float]


def load_or_download_dataset(
    snapshot_path: Path = SNAPSHOT_PATH,
    metadata_path: Path = METADATA_PATH,
    refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Carrega a base pública real GiveMeSomeCredit (OpenML 46929) do snapshot ou baixa via fetch_openml."""
    if snapshot_path.is_file() and metadata_path.is_file() and not refresh:
        df = pd.read_csv(snapshot_path, compression="gzip")
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        _validate_raw_schema(df)
        return df, meta

    dataset = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True, parser="auto")
    df = dataset.frame.copy()
    _validate_raw_schema(df)

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(snapshot_path, index=False, compression="gzip")

    meta = {
        "source": "OpenML public dataset (sklearn.datasets.fetch_openml)",
        "dataset_name": "GiveMeSomeCredit",
        "openml_data_id": OPENML_DATA_ID,
        "url": f"https://www.openml.org/d/{OPENML_DATA_ID}",
        "target_column": TARGET_COL,
        "positive_class": "Yes",
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "bad_rate_global": float((df[TARGET_COL] == "Yes").mean()),
        "pii_present": False,
        "synthetic_data": False,
        "snapshot_file": str(snapshot_path.relative_to(STUDY_ROOT)).replace("\\", "/"),
    }
    metadata_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return df, meta


def _validate_raw_schema(df: pd.DataFrame) -> None:
    missing = [col for col in RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes no dataset público: {missing}")
    if len(df) < 10000:
        raise ValueError(f"Volume inesperado de linhas: {len(df)}")


def build_feature_matrices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Constrói separadamente as matrizes de Longo Prazo (estrutural) e Curto Prazo (pressão)."""
    overlap = set(FEATURES_LP).intersection(FEATURES_CP)
    if overlap:
        raise AssertionError(f"Sobreposição proibida entre eixos LP e CP: {overlap}")

    y = (df[TARGET_COL].astype(str) == "Yes").astype(int).to_numpy()

    x_lp = pd.DataFrame(index=df.index)
    x_lp["age"] = df["age"].clip(18, 95).astype(float)
    x_lp["income_missing"] = df["MonthlyIncome"].isna().astype(float)
    x_lp["log_income"] = np.log1p(df["MonthlyIncome"].clip(lower=0))
    x_lp["dependents_missing"] = df["NumberOfDependents"].isna().astype(float)
    x_lp["dependents"] = df["NumberOfDependents"].clip(0, 10)
    x_lp["open_lines"] = df["NumberOfOpenCreditLinesAndLoans"].clip(0, 40).astype(float)
    x_lp["real_estate_loans"] = df["NumberRealEstateLoansOrLines"].clip(0, 20).astype(float)
    x_lp["real_estate_share"] = x_lp["real_estate_loans"] / (1.0 + x_lp["open_lines"])
    x_lp["has_real_estate"] = (x_lp["real_estate_loans"] > 0).astype(float)
    x_lp["zero_open_lines"] = (x_lp["open_lines"] == 0).astype(float)
    x_lp["income_per_dep"] = np.log1p(
        df["MonthlyIncome"].clip(lower=0) / (1.0 + x_lp["dependents"].fillna(0))
    )

    x_cp = pd.DataFrame(index=df.index)
    x_cp["rev_util_capped"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(0, 5.0)
    x_cp["log_rev_util"] = np.log1p(df["RevolvingUtilizationOfUnsecuredLines"].clip(0, 50.0))
    x_cp["rev_over_08"] = (df["RevolvingUtilizationOfUnsecuredLines"] > 0.80).astype(float)
    x_cp["debt_ratio_known"] = np.where(
        df["MonthlyIncome"].notna() & (df["MonthlyIncome"] > 0),
        df["DebtRatio"].clip(0, 10.0),
        np.nan,
    )
    x_cp["debt_amt_no_income"] = np.where(
        df["MonthlyIncome"].isna() | (df["MonthlyIncome"] == 0),
        np.log1p(df["DebtRatio"].clip(0, 50000)),
        0.0,
    )
    d30 = df["NumberOfTime30-59DaysPastDueNotWorse"].where(
        df["NumberOfTime30-59DaysPastDueNotWorse"] < 96, 10
    ).clip(0, 10)
    d60 = df["NumberOfTime60-89DaysPastDueNotWorse"].where(
        df["NumberOfTime60-89DaysPastDueNotWorse"] < 96, 10
    ).clip(0, 10)
    d90 = df["NumberOfTimes90DaysLate"].where(df["NumberOfTimes90DaysLate"] < 96, 10).clip(0, 10)
    x_cp["d30"] = d30.astype(float)
    x_cp["d60"] = d60.astype(float)
    x_cp["d90"] = d90.astype(float)
    x_cp["severity_idx"] = 1.0 * x_cp["d30"] + 2.5 * x_cp["d60"] + 4.0 * x_cp["d90"]

    return x_lp[FEATURES_LP], x_cp[FEATURES_CP], y


def split_train_val_test(y: np.ndarray, random_state: int = 42) -> SplitIndices:
    """Particiona em 60% treino, 20% validação e 20% teste de forma estratificada e disjunta."""
    idx = np.arange(len(y))
    idx_tr, idx_tmp, _, y_tmp = train_test_split(
        idx, y, test_size=0.40, stratify=y, random_state=random_state
    )
    idx_va, idx_te, _, _ = train_test_split(
        idx_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=random_state
    )
    set_tr, set_va, set_te = set(idx_tr.tolist()), set(idx_va.tolist()), set(idx_te.tolist())
    if not (set_tr.isdisjoint(set_va) and set_tr.isdisjoint(set_te) and set_va.isdisjoint(set_te)):
        raise AssertionError("Vazamento detectado: splits não são disjuntos.")
    return SplitIndices(train=np.sort(idx_tr), val=np.sort(idx_va), test=np.sort(idx_te))


def evaluate_funnel_layers(df: pd.DataFrame, policy_mode: str = "otimizada") -> dict[str, Any]:
    """Avalia as camadas R1 -> R2 -> R3 com substeps explícitos e retorno booleano conservativo."""
    if policy_mode not in {"otimizada", "rigida", "agressiva_1d"}:
        raise ValueError(f"Modo de política desconhecido: {policy_mode}")

    # R1: Elegibilidade cadastral & mínimos (idêntica em todas as políticas)
    r1_idade = ((df["age"] < 21) | (df["age"] > 85)).to_numpy()
    r1_renda_zero = (df["MonthlyIncome"].notna() & (df["MonthlyIncome"] <= 0)).to_numpy()
    ko_r1 = r1_idade | r1_renda_zero
    sub_r1_1 = r1_idade
    sub_r1_2 = (~r1_idade) & r1_renda_zero

    # R2: Restritivo / Bureau por gravidade
    sent_96_98 = (
        (df["NumberOfTime30-59DaysPastDueNotWorse"] >= 96)
        | (df["NumberOfTime60-89DaysPastDueNotWorse"] >= 96)
        | (df["NumberOfTimes90DaysLate"] >= 96)
    ).to_numpy()
    atraso_90_ou_60_mult = (
        (df["NumberOfTimes90DaysLate"] >= 1) | (df["NumberOfTime60-89DaysPastDueNotWorse"] >= 2)
    ).to_numpy()
    atraso_leve_raw = (
        (df["NumberOfTime30-59DaysPastDueNotWorse"] >= 1)
        | (df["NumberOfTime60-89DaysPastDueNotWorse"] == 1)
    ).to_numpy()

    alive_after_r1 = ~ko_r1
    sub_r2_sentinela = alive_after_r1 & sent_96_98
    sub_r2_severo = alive_after_r1 & (~sub_r2_sentinela) & atraso_90_ou_60_mult
    sub_r2_leve_candidato = alive_after_r1 & (~sub_r2_sentinela) & (~sub_r2_severo) & atraso_leve_raw

    if policy_mode == "rigida":
        ko_r2 = sub_r2_sentinela | sub_r2_severo | sub_r2_leve_candidato
        flag_atraso_leve = np.zeros(len(df), dtype=bool)
        r2_substeps = {
            "R2.1_SENTINELA_96_98": sub_r2_sentinela,
            "R2.2_ATRASO_SEVERO_90D": sub_r2_severo,
            "R2.3_KO_R2_RIGIDO_ATRASO_LEVE": sub_r2_leve_candidato,
        }
    else:
        ko_r2 = sub_r2_sentinela | sub_r2_severo
        flag_atraso_leve = sub_r2_leve_candidato
        r2_substeps = {
            "R2.1_SENTINELA_96_98": sub_r2_sentinela,
            "R2.2_ATRASO_SEVERO_90D": sub_r2_severo,
        }

    # R3: Capacidade & Alavancagem
    alive_after_r2 = alive_after_r1 & (~ko_r2)
    has_pos_income = (df["MonthlyIncome"].notna() & (df["MonthlyIncome"] > 0)).to_numpy()
    rev_util = df["RevolvingUtilizationOfUnsecuredLines"].to_numpy()
    debt_ratio = df["DebtRatio"].to_numpy()

    if policy_mode == "rigida":
        sub_r3_1 = alive_after_r2 & (rev_util > 0.50)
        sub_r3_2 = alive_after_r2 & (~sub_r3_1) & has_pos_income & (debt_ratio > 0.50)
        ko_r3 = sub_r3_1 | sub_r3_2
        flag_cap_pressao = np.zeros(len(df), dtype=bool)
        r3_substeps = {
            "R3.1_RIGIDO_REV_UTIL_50": sub_r3_1,
            "R3.2_RIGIDO_DEBTRATIO_50": sub_r3_2,
        }
    else:
        sub_r3_1 = alive_after_r2 & (rev_util > 1.05)
        sub_r3_2 = alive_after_r2 & (~sub_r3_1) & has_pos_income & (debt_ratio > 2.0)
        ko_r3 = sub_r3_1 | sub_r3_2
        flag_cap_pressao = (
            alive_after_r2
            & (~ko_r3)
            & ((rev_util > 0.75) | (has_pos_income & (debt_ratio > 0.50)))
        )
        r3_substeps = {
            "R3.1_ESTOURAMENTO_LIMITE_105": sub_r3_1,
            "R3.2_COMPROMETIMENTO_EXTREMO_200": sub_r3_2,
        }

    r4_eligible = ~(ko_r1 | ko_r2 | ko_r3)

    return {
        "policy_mode": policy_mode,
        "ko_r1": ko_r1,
        "ko_r2": ko_r2,
        "ko_r3": ko_r3,
        "r4_eligible": r4_eligible,
        "flag_atraso_leve": flag_atraso_leve,
        "flag_cap_pressao": flag_cap_pressao,
        "substeps": {
            "R1": {
                "R1.1_IDADE_FORA_POLITICA": sub_r1_1,
                "R1.2_RENDA_DECLARADA_ZERADA": sub_r1_2,
            },
            "R2": r2_substeps,
            "R3": r3_substeps,
        },
    }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calcula Expected Calibration Error (ECE) em bins uniformes de probabilidade."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for idx in range(n_bins):
        lo, hi = bins[idx], bins[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if np.any(mask):
            ece += (np.sum(mask) / n) * abs(float(np.mean(y_true[mask])) - float(np.mean(y_prob[mask])))
    return float(ece)


def fit_and_calibrate_models(
    df: pd.DataFrame,
    x_lp: pd.DataFrame,
    x_cp: pd.DataFrame,
    y: np.ndarray,
    splits: SplitIndices,
    random_state: int = 42,
) -> ModelBundle:
    """Treina os dois eixos em treino e calibra/define cortes exclusivamente na validação."""
    lp_model = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("sp", SplineTransformer(n_knots=4, degree=2, include_bias=False)),
            ("clf", LogisticRegression(max_iter=1000, C=0.5)),
        ]
    )
    lp_model.fit(x_lp.iloc[splits.train], y[splits.train])

    rf_base = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_leaf=15,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_base.fit(x_cp.iloc[splits.train], y[splits.train])

    # Avaliação honesta de calibração (bruto vs sigmoid vs isotônica) via 5-fold OOF na validação
    x_cp_val = x_cp.iloc[splits.val].reset_index(drop=True)
    y_val = y[splits.val]
    raw_val_prob = np.clip(rf_base.predict_proba(x_cp_val)[:, 1], 1e-5, 1 - 1e-5)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    oof_iso = np.zeros(len(y_val), dtype=float)
    oof_sig = np.zeros(len(y_val), dtype=float)
    for tr_i, te_i in skf.split(x_cp_val, y_val):
        cal_iso = CalibratedClassifierCV(FrozenEstimator(rf_base), method="isotonic")
        cal_iso.fit(x_cp_val.iloc[tr_i], y_val[tr_i])
        oof_iso[te_i] = np.clip(cal_iso.predict_proba(x_cp_val.iloc[te_i])[:, 1], 1e-5, 1 - 1e-5)

        cal_sig = CalibratedClassifierCV(FrozenEstimator(rf_base), method="sigmoid")
        cal_sig.fit(x_cp_val.iloc[tr_i], y_val[tr_i])
        oof_sig[te_i] = np.clip(cal_sig.predict_proba(x_cp_val.iloc[te_i])[:, 1], 1e-5, 1 - 1e-5)

    def _cal_summary(yt: np.ndarray, pr: np.ndarray) -> dict[str, float]:
        return {
            "brier": float(brier_score_loss(yt, pr)),
            "ece": expected_calibration_error(yt, pr),
            "log_loss": float(log_loss(yt, pr)),
            "auc": float(roc_auc_score(yt, pr)),
            "pr_auc": float(average_precision_score(yt, pr)),
        }

    funnel_val = evaluate_funnel_layers(df.iloc[splits.val].reset_index(drop=True), "otimizada")
    elig_val = funnel_val["r4_eligible"]

    val_cal_metrics = {
        "rf_bruto": _cal_summary(y_val, raw_val_prob),
        "calibracao_sigmoid": _cal_summary(y_val, oof_sig),
        "calibracao_isotonica": _cal_summary(y_val, oof_iso),
    }
    val_r4_cal_metrics = {
        "rf_bruto": _cal_summary(y_val[elig_val], raw_val_prob[elig_val]),
        "calibracao_sigmoid": _cal_summary(y_val[elig_val], oof_sig[elig_val]),
        "calibracao_isotonica": _cal_summary(y_val[elig_val], oof_iso[elig_val]),
    }

    candidates = [
        ("bruto_nativo", val_cal_metrics["rf_bruto"]["brier"], val_cal_metrics["rf_bruto"]["log_loss"]),
        ("isotonic", val_cal_metrics["calibracao_isotonica"]["brier"], val_cal_metrics["calibracao_isotonica"]["log_loss"]),
        ("sigmoid", val_cal_metrics["calibracao_sigmoid"]["brier"], val_cal_metrics["calibracao_sigmoid"]["log_loss"]),
    ]
    best_method = min(candidates, key=lambda item: (round(item[1], 7), item[2]))[0]

    rf_cal_isotonic = CalibratedClassifierCV(FrozenEstimator(rf_base), method="isotonic")
    rf_cal_isotonic.fit(x_cp_val, y_val)
    rf_cal_sigmoid = CalibratedClassifierCV(FrozenEstimator(rf_base), method="sigmoid")
    rf_cal_sigmoid.fit(x_cp_val, y_val)

    # Blend 2D logístico ajustado exclusivamente na validação usando o estimador vencedor
    pd_lp_val = np.clip(lp_model.predict_proba(x_lp.iloc[splits.val])[:, 1], 1e-5, 1 - 1e-5)
    if best_method == "bruto_nativo":
        pd_cp_val = raw_val_prob
    elif best_method == "isotonic":
        pd_cp_val = np.clip(rf_cal_isotonic.predict_proba(x_cp_val)[:, 1], 1e-5, 1 - 1e-5)
    else:
        pd_cp_val = np.clip(rf_cal_sigmoid.predict_proba(x_cp_val)[:, 1], 1e-5, 1 - 1e-5)

    z_val = np.column_stack([special.logit(pd_lp_val), special.logit(pd_cp_val)])
    blend_2d = LogisticRegression(max_iter=500, C=1.0)
    blend_2d.fit(z_val, y_val)

    # Tercis da Matriz 3x3 calibrados exclusivamente nos elegíveis R4 da validação
    score_lp_val = 1000.0 * (1.0 - pd_lp_val)
    q_lp = np.quantile(score_lp_val[elig_val], [1.0 / 3.0, 2.0 / 3.0])
    q_cp = np.quantile(pd_cp_val[elig_val], [1.0 / 3.0, 2.0 / 3.0])

    return ModelBundle(
        lp_model=lp_model,
        rf_base=rf_base,
        rf_cal_isotonic=rf_cal_isotonic,
        rf_cal_sigmoid=rf_cal_sigmoid,
        cal_method=best_method,
        blend_2d=blend_2d,
        val_calibration_metrics=val_cal_metrics,
        val_r4_calibration_metrics=val_r4_cal_metrics,
        lp_terciles=(float(q_lp[0]), float(q_lp[1])),
        cp_terciles=(float(q_cp[0]), float(q_cp[1])),
    )


def score_population(
    x_lp: pd.DataFrame,
    x_cp: pd.DataFrame,
    bundle: ModelBundle,
) -> dict[str, np.ndarray]:
    """Gera PDs, scores e bandas 3x3 a partir de modelos e tercis congelados."""
    pd_lp = np.clip(bundle.lp_model.predict_proba(x_lp)[:, 1], 1e-5, 1 - 1e-5)
    score_lp = 1000.0 * (1.0 - pd_lp)
    pd_cp_raw = np.clip(bundle.rf_base.predict_proba(x_cp)[:, 1], 1e-5, 1 - 1e-5)
    pd_cp_iso = np.clip(bundle.rf_cal_isotonic.predict_proba(x_cp)[:, 1], 1e-5, 1 - 1e-5)
    pd_cp_sig = np.clip(bundle.rf_cal_sigmoid.predict_proba(x_cp)[:, 1], 1e-5, 1 - 1e-5)
    if bundle.cal_method == "bruto_nativo":
        pd_cp = pd_cp_raw
    elif bundle.cal_method == "isotonic":
        pd_cp = pd_cp_iso
    else:
        pd_cp = pd_cp_sig

    z_2d = np.column_stack([special.logit(pd_lp), special.logit(pd_cp)])
    pd_2d = np.clip(bundle.blend_2d.predict_proba(z_2d)[:, 1], 1e-5, 1 - 1e-5)

    q_lp_0, q_lp_1 = bundle.lp_terciles
    q_cp_0, q_cp_1 = bundle.cp_terciles
    lp_band_idx = np.where(score_lp < q_lp_0, 0, np.where(score_lp < q_lp_1, 1, 2))
    cp_band_idx = np.where(pd_cp <= q_cp_0, 0, np.where(pd_cp <= q_cp_1, 1, 2))

    return {
        "pd_lp": pd_lp,
        "score_lp": score_lp,
        "pd_cp_raw": pd_cp_raw,
        "pd_cp_iso": pd_cp_iso,
        "pd_cp_sig": pd_cp_sig,
        "pd_cp": pd_cp,
        "pd_2d": pd_2d,
        "lp_band_idx": lp_band_idx,
        "cp_band_idx": cp_band_idx,
    }


def apply_matrix_3x3_clusters(
    r4_eligible: np.ndarray,
    lp_band_idx: np.ndarray,
    cp_band_idx: np.ndarray,
) -> dict[str, Any]:
    """Aplica a saída ordenada R4: Cluster C (incl. Early Warning) -> Cluster B -> Cluster A."""
    n = len(r4_eligible)
    cluster = np.full(n, "NONE", dtype=object)
    early_warning = np.zeros(n, dtype=bool)

    for i in range(n):
        if not r4_eligible[i]:
            continue
        lp_i = int(lp_band_idx[i])
        cp_i = int(cp_band_idx[i])
        assigned = MATRIX_3X3_CLUSTER_MAP[(lp_i, cp_i)]
        cluster[i] = assigned
        if lp_i == 2 and cp_i == 2:
            early_warning[i] = True

    cluster_c = r4_eligible & (cluster == "C")
    cluster_b = r4_eligible & (cluster == "B")
    cluster_a = r4_eligible & (cluster == "A")

    return {
        "cluster": cluster,
        "cluster_c": cluster_c,
        "cluster_b": cluster_b,
        "cluster_a": cluster_a,
        "early_warning": early_warning,
    }


def verify_mass_conservation(
    n_total: int,
    funnel: dict[str, Any],
    clusters: dict[str, Any] | None = None,
    r4_custom_masks: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Valida conservação exata de massa no funil e nos substeps."""
    ko_r1 = funnel["ko_r1"]
    ko_r2 = funnel["ko_r2"]
    ko_r3 = funnel["ko_r3"]
    r4_elig = funnel["r4_eligible"]

    # Substeps devem somar exatamente o KO da respectiva camada
    for layer_name, ko_mask in [("R1", ko_r1), ("R2", ko_r2), ("R3", ko_r3)]:
        sub_sum = sum(int(np.sum(m)) for m in funnel["substeps"][layer_name].values())
        if sub_sum != int(np.sum(ko_mask)):
            raise AssertionError(
                f"Conservação falhou nos substeps de {layer_name}: {sub_sum} != {int(np.sum(ko_mask))}"
            )

    if clusters is not None:
        c_cnt = int(np.sum(clusters["cluster_c"]))
        b_cnt = int(np.sum(clusters["cluster_b"]))
        a_cnt = int(np.sum(clusters["cluster_a"]))
        # Conjuntos devem ser mutuamente exclusivos
        overlap_ab = np.sum(clusters["cluster_a"] & clusters["cluster_b"])
        overlap_ac = np.sum(clusters["cluster_a"] & clusters["cluster_c"])
        overlap_bc = np.sum(clusters["cluster_b"] & clusters["cluster_c"])
        if (overlap_ab + overlap_ac + overlap_bc) != 0:
            raise AssertionError("Clusters R4 não são mutuamente exclusivos.")
    elif r4_custom_masks is not None:
        c_cnt = int(np.sum(r4_custom_masks["cluster_c"]))
        b_cnt = int(np.sum(r4_custom_masks["cluster_b"]))
        a_cnt = int(np.sum(r4_custom_masks["cluster_a"]))
    else:
        raise ValueError("Forneça clusters ou r4_custom_masks.")

    if (c_cnt + b_cnt + a_cnt) != int(np.sum(r4_elig)):
        raise AssertionError(
            f"Conservação falhou em R4: C+B+A={c_cnt + b_cnt + a_cnt} != R4_elig={int(np.sum(r4_elig))}"
        )

    reconstructed = int(np.sum(ko_r1) + np.sum(ko_r2) + np.sum(ko_r3) + c_cnt + b_cnt + a_cnt)
    if reconstructed != int(n_total):
        raise AssertionError(f"Conservação global falhou: {reconstructed} != {n_total}")

    return {
        "total_entrada": int(n_total),
        "ko_r1": int(np.sum(ko_r1)),
        "ko_r2": int(np.sum(ko_r2)),
        "ko_r3": int(np.sum(ko_r3)),
        "r4_elegiveis": int(np.sum(r4_elig)),
        "cluster_c": c_cnt,
        "cluster_b": b_cnt,
        "cluster_a": a_cnt,
        "total_reconstruido": reconstructed,
        "conservacao_exata": bool(reconstructed == int(n_total)),
    }


def _ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    return float(stats.ks_2samp(pos, neg).statistic)


def _partial_corr(x: np.ndarray, y_var: np.ndarray, covariates: np.ndarray) -> float:
    """Correlação parcial de Pearson entre x e y_var controlando por covariates via resíduos OLS."""
    design = np.column_stack([np.ones(len(x)), covariates])
    beta_x, _, _, _ = np.linalg.lstsq(design, x, rcond=None)
    beta_y, _, _, _ = np.linalg.lstsq(design, y_var, rcond=None)
    res_x = x - design @ beta_x
    res_y = y_var - design @ beta_y
    return float(np.corrcoef(res_x, res_y)[0, 1])


def _wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Intervalo de confiança binomial de Wilson (95%) sem valores negativos."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def _roc_curve_points(y_true: np.ndarray, y_score: np.ndarray, n_pts: int = 101) -> dict[str, list[float]]:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    grid = np.linspace(0.0, 1.0, n_pts)
    tpr_interp = np.interp(grid, fpr, tpr)
    tpr_interp[0] = 0.0
    tpr_interp[-1] = 1.0
    return {
        "fpr": [round(float(x), 4) for x in grid],
        "tpr": [round(float(y_v), 4) for y_v in tpr_interp],
    }


def _pr_curve_points(y_true: np.ndarray, y_score: np.ndarray, n_pts: int = 101) -> dict[str, list[float]]:
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    # precision_recall_curve retorna recall em ordem decrescente (1 -> 0); inverter para interpolação monotônica
    rec_asc = rec[::-1]
    prec_asc = prec[::-1]
    grid = np.linspace(0.0, 1.0, n_pts)
    prec_interp = np.interp(grid, rec_asc, prec_asc)
    return {
        "recall": [round(float(x), 4) for x in grid],
        "precision": [round(float(p), 4) for p in prec_interp],
    }


def _calibration_curve_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Curva de calibração real por decis de probabilidade prevista (com IC 95% de Wilson e N)."""
    order = np.argsort(y_prob, kind="mergesort")
    chunks = np.array_split(order, n_bins)
    rows: list[dict[str, Any]] = []
    for idx, ch in enumerate(chunks, start=1):
        n_b = int(len(ch))
        k_b = int(np.sum(y_true[ch]))
        ci_lo, ci_hi = _wilson_ci(k_b, n_b)
        rows.append(
            {
                "bin": idx,
                "n": n_b,
                "maus": k_b,
                "pd_media": float(np.mean(y_prob[ch])),
                "bad_rate_obs": float(k_b / n_b) if n_b > 0 else 0.0,
                "ic95_inf": ci_lo,
                "ic95_sup": ci_hi,
            }
        )
    return rows


def evaluate_complementarity_and_2d_gain(
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    scores_test: dict[str, np.ndarray],
    r4_elig_test: np.ndarray,
    bundle: ModelBundle | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Mede baixa colinearidade entre eixos e ganho estatístico incremental da Combinação 2D vs LP e vs RF isolado."""
    pd_lp = scores_test["pd_lp"]
    pd_cp = scores_test["pd_cp"]
    pd_2d = scores_test["pd_2d"]
    pd_cp_raw = scores_test["pd_cp_raw"]
    pd_cp_iso = scores_test["pd_cp_iso"]
    pd_cp_sig = scores_test["pd_cp_sig"]

    # Correlações (full test e R4-elegível test)
    pearson_full = float(np.corrcoef(pd_lp, pd_cp)[0, 1])
    spearman_full = float(stats.spearmanr(pd_lp, pd_cp).statistic)
    pearson_r4 = float(np.corrcoef(pd_lp[r4_elig_test], pd_cp[r4_elig_test])[0, 1])
    spearman_r4 = float(stats.spearmanr(pd_lp[r4_elig_test], pd_cp[r4_elig_test]).statistic)

    # Correlação parcial controlando pelo desfecho observado y_test e idade
    cov_full = np.column_stack([y_test, df_test["age"].clip(18, 95).to_numpy()])
    partial_corr_full = _partial_corr(pd_lp, pd_cp, cov_full)
    partial_corr_r4 = _partial_corr(pd_lp[r4_elig_test], pd_cp[r4_elig_test], cov_full[r4_elig_test])

    r2_shared_r4 = pearson_r4 ** 2
    vif_r4 = float(1.0 / max(1.0 - r2_shared_r4, 1e-9))

    def _model_eval(yt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
        return {
            "auc": float(roc_auc_score(yt, pred)),
            "ks": _ks_statistic(yt, pred),
            "pr_auc": float(average_precision_score(yt, pred)),
            "brier": float(brier_score_loss(yt, pred)),
            "log_loss": float(log_loss(yt, pred)),
            "ece": expected_calibration_error(yt, pred),
        }

    def _metrics_block(mask: np.ndarray) -> dict[str, Any]:
        yt = y_test[mask]
        return {
            "n": int(np.sum(mask)),
            "bad_rate": float(np.mean(yt)),
            "score_lp_isolado": _model_eval(yt, pd_lp[mask]),
            "rf_cp_bruto": _model_eval(yt, pd_cp_raw[mask]),
            "rf_cp_isotonico": _model_eval(yt, pd_cp_iso[mask]),
            "rf_cp_sigmoide": _model_eval(yt, pd_cp_sig[mask]),
            "combinacao_2d": _model_eval(yt, pd_2d[mask]),
        }

    # Bootstrap estratificado pareado (500 réplicas) no conjunto OOS R4:
    # avalia 2D vs Score LP E 2D vs RF CP Bruto / Isotônico (transparência do ganho marginal)
    rng = np.random.default_rng(random_state)
    yt_r4 = y_test[r4_elig_test]
    lp_r4 = pd_lp[r4_elig_test]
    raw_r4 = pd_cp_raw[r4_elig_test]
    iso_r4 = pd_cp_iso[r4_elig_test]
    sig_r4 = pd_cp_sig[r4_elig_test]
    m2d_r4 = pd_2d[r4_elig_test]

    pos_idx = np.where(yt_r4 == 1)[0]
    neg_idx = np.where(yt_r4 == 0)[0]
    n_boot = 500
    auc_lp_b = np.empty(n_boot, dtype=float)
    auc_raw_b = np.empty(n_boot, dtype=float)
    auc_iso_b = np.empty(n_boot, dtype=float)
    auc_2d_b = np.empty(n_boot, dtype=float)
    pr_raw_b = np.empty(n_boot, dtype=float)
    pr_2d_b = np.empty(n_boot, dtype=float)
    ll_raw_b = np.empty(n_boot, dtype=float)
    ll_2d_b = np.empty(n_boot, dtype=float)
    br_raw_b = np.empty(n_boot, dtype=float)
    br_2d_b = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        s_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        s_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([s_pos, s_neg])
        y_s = yt_r4[idx]
        auc_lp_b[b] = roc_auc_score(y_s, lp_r4[idx])
        auc_raw_b[b] = roc_auc_score(y_s, raw_r4[idx])
        auc_iso_b[b] = roc_auc_score(y_s, iso_r4[idx])
        auc_2d_b[b] = roc_auc_score(y_s, m2d_r4[idx])
        pr_raw_b[b] = average_precision_score(y_s, raw_r4[idx])
        pr_2d_b[b] = average_precision_score(y_s, m2d_r4[idx])
        ll_raw_b[b] = log_loss(y_s, raw_r4[idx])
        ll_2d_b[b] = log_loss(y_s, m2d_r4[idx])
        br_raw_b[b] = brier_score_loss(y_s, raw_r4[idx])
        br_2d_b[b] = brier_score_loss(y_s, m2d_r4[idx])

    d_2d_lp = auc_2d_b - auc_lp_b
    d_2d_raw = auc_2d_b - auc_raw_b
    d_2d_iso = auc_2d_b - auc_iso_b
    d_pr_2d_raw = pr_2d_b - pr_raw_b
    d_ll_raw_2d = ll_raw_b - ll_2d_b  # > 0 significa 2D reduziu LogLoss vs RF bruto
    d_br_raw_2d = br_raw_b - br_2d_b  # > 0 significa 2D reduziu Brier vs RF bruto

    blend_coefs = (
        {
            "coef_logit_lp": float(bundle.blend_2d.coef_[0][0]),
            "coef_logit_cp": float(bundle.blend_2d.coef_[0][1]),
            "intercept": float(bundle.blend_2d.intercept_[0]),
        }
        if bundle is not None
        else {}
    )

    return {
        "correlacao_entre_eixos": {
            "pearson_teste_completo": pearson_full,
            "spearman_teste_completo": spearman_full,
            "pearson_r4_elegiveis": pearson_r4,
            "spearman_r4_elegiveis": spearman_r4,
            "correlacao_parcial_ctrl_y_idade_teste": partial_corr_full,
            "correlacao_parcial_ctrl_y_idade_r4": partial_corr_r4,
            "r2_compartilhado_r4": float(r2_shared_r4),
            "vif_r4": vif_r4,
        },
        "coeficientes_blend_2d": blend_coefs,
        "desempenho_teste_completo": _metrics_block(np.ones(len(y_test), dtype=bool)),
        "desempenho_teste_r4_elegiveis": _metrics_block(r4_elig_test),
        "auc_ic95_r4": {
            "score_lp_isolado": [float(np.quantile(auc_lp_b, 0.025)), float(np.quantile(auc_lp_b, 0.975))],
            "rf_cp_bruto": [float(np.quantile(auc_raw_b, 0.025)), float(np.quantile(auc_raw_b, 0.975))],
            "rf_cp_isotonico": [float(np.quantile(auc_iso_b, 0.025)), float(np.quantile(auc_iso_b, 0.975))],
            "combinacao_2d": [float(np.quantile(auc_2d_b, 0.025)), float(np.quantile(auc_2d_b, 0.975))],
        },
        "ganho_2d_sobre_lp_r4_bootstrap": {
            "delta_auc_medio": float(np.mean(d_2d_lp)),
            "ic95_inferior": float(np.quantile(d_2d_lp, 0.025)),
            "ic95_superior": float(np.quantile(d_2d_lp, 0.975)),
            "p_val_unilateral": float(np.mean(d_2d_lp <= 0.0)),
        },
        "ganho_2d_sobre_rf_bruto_r4_bootstrap": {
            "delta_auc_medio": float(np.mean(d_2d_raw)),
            "ic95_inferior": float(np.quantile(d_2d_raw, 0.025)),
            "ic95_superior": float(np.quantile(d_2d_raw, 0.975)),
            "p_val_unilateral": float(np.mean(d_2d_raw <= 0.0)),
            "delta_pr_auc_medio": float(np.mean(d_pr_2d_raw)),
            "pr_auc_ic95": [float(np.quantile(d_pr_2d_raw, 0.025)), float(np.quantile(d_pr_2d_raw, 0.975))],
            "reducao_log_loss_media": float(np.mean(d_ll_raw_2d)),
            "log_loss_ic95": [float(np.quantile(d_ll_raw_2d, 0.025)), float(np.quantile(d_ll_raw_2d, 0.975))],
            "reducao_brier_media": float(np.mean(d_br_raw_2d)),
        },
        "ganho_2d_sobre_rf_isotonico_r4_bootstrap": {
            "delta_auc_medio": float(np.mean(d_2d_iso)),
            "ic95_inferior": float(np.quantile(d_2d_iso, 0.025)),
            "ic95_superior": float(np.quantile(d_2d_iso, 0.975)),
            "p_val_unilateral": float(np.mean(d_2d_iso <= 0.0)),
        },
        "curvas_roc_r4": {
            "score_lp_isolado": _roc_curve_points(yt_r4, lp_r4),
            "rf_cp_bruto": _roc_curve_points(yt_r4, raw_r4),
            "rf_cp_isotonico": _roc_curve_points(yt_r4, iso_r4),
            "combinacao_2d": _roc_curve_points(yt_r4, m2d_r4),
        },
        "curvas_pr_r4": {
            "score_lp_isolado": _pr_curve_points(yt_r4, lp_r4),
            "rf_cp_bruto": _pr_curve_points(yt_r4, raw_r4),
            "rf_cp_isotonico": _pr_curve_points(yt_r4, iso_r4),
            "combinacao_2d": _pr_curve_points(yt_r4, m2d_r4),
        },
        "curvas_calibracao_r4": {
            "score_lp_isolado": _calibration_curve_bins(yt_r4, lp_r4, 10),
            "rf_cp_bruto": _calibration_curve_bins(yt_r4, raw_r4, 10),
            "rf_cp_isotonico": _calibration_curve_bins(yt_r4, iso_r4, 10),
            "rf_cp_sigmoide": _calibration_curve_bins(yt_r4, sig_r4, 10),
            "combinacao_2d": _calibration_curve_bins(yt_r4, m2d_r4, 10),
        },
    }


prove_2d_orthogonality_and_gain = evaluate_complementarity_and_2d_gain


def build_funnel_stage_table(
    y: np.ndarray,
    funnel: dict[str, Any],
    clusters: dict[str, Any] | None = None,
    r4_custom_masks: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Constrói tabela auditável etapa a etapa do funil com substeps e conservação."""
    n_total = len(y)
    stages: list[dict[str, Any]] = [
        {
            "etapa": "Entrada",
            "tipo": "entrada",
            "volume": int(n_total),
            "pct_entrada": 100.0,
            "bad_rate": float(np.mean(y)),
            "maus": int(np.sum(y)),
        }
    ]

    for layer_key, ko_key, label in [
        ("R1", "ko_r1", "KO R1 — Elegibilidade"),
        ("R2", "ko_r2", "KO R2 — Restritivo/Bureau"),
        ("R3", "ko_r3", "KO R3 — Capacidade/Alavancagem"),
    ]:
        mask = funnel[ko_key]
        sub_list = []
        for sub_name, sub_mask in funnel["substeps"][layer_key].items():
            sub_cnt = int(np.sum(sub_mask))
            sub_list.append(
                {
                    "substep": sub_name,
                    "volume": sub_cnt,
                    "pct_entrada": float(sub_cnt / n_total * 100.0),
                    "bad_rate": float(np.mean(y[sub_mask])) if sub_cnt > 0 else 0.0,
                }
            )
        cnt = int(np.sum(mask))
        stages.append(
            {
                "etapa": label,
                "tipo": "saida_ko",
                "volume": cnt,
                "pct_entrada": float(cnt / n_total * 100.0),
                "bad_rate": float(np.mean(y[mask])) if cnt > 0 else 0.0,
                "maus": int(np.sum(y[mask])) if cnt > 0 else 0,
                "substeps": sub_list,
            }
        )

    r4_mask = funnel["r4_eligible"]
    stages.append(
        {
            "etapa": "Elegíveis R4 (Entrada Matriz)",
            "tipo": "checkpoint",
            "volume": int(np.sum(r4_mask)),
            "pct_entrada": float(np.sum(r4_mask) / n_total * 100.0),
            "bad_rate": float(np.mean(y[r4_mask])),
            "maus": int(np.sum(y[r4_mask])),
        }
    )

    src = clusters if clusters is not None else r4_custom_masks
    assert src is not None
    # Ordem oficial de saída da R4: Cluster C -> Cluster B -> Cluster A
    for key, label, tipo in [
        ("cluster_c", "R4 Cluster C — Reprova / Early Warning", "saida_r4_c"),
        ("cluster_b", "R4 Cluster B — Contraproposta (65% EAD)", "aprovado_parcial"),
        ("cluster_a", "R4 Cluster A — Aprova 100% (Fundo do Funil)", "aprovado_cheio"),
    ]:
        m = src[key]
        cnt = int(np.sum(m))
        stages.append(
            {
                "etapa": label,
                "tipo": tipo,
                "volume": cnt,
                "pct_entrada": float(cnt / n_total * 100.0),
                "bad_rate": float(np.mean(y[m])) if cnt > 0 else 0.0,
                "maus": int(np.sum(y[m])) if cnt > 0 else 0,
            }
        )
    return stages


def build_matrix_3x3_summary(
    y: np.ndarray,
    r4_eligible: np.ndarray,
    lp_band_idx: np.ndarray,
    cp_band_idx: np.ndarray,
) -> list[dict[str, Any]]:
    """Resumo completo das 9 células da Matriz 3x3 fora da amostra com Unit Economics auditável."""
    n_r4 = int(np.sum(r4_eligible))
    ead_unit = UNIT_ECONOMICS["ead_padrao_um"]
    ead_b_frac = UNIT_ECONOMICS["cluster_b_fator_ead"]
    m_a = UNIT_ECONOMICS["margem_liquida_24m_padrao"]
    m_b = UNIT_ECONOMICS["margem_liquida_24m_cluster_b"]
    opex = UNIT_ECONOMICS["opex_originacao_um"]
    lgd_base = UNIT_ECONOMICS["lgd_base"]

    cells: list[dict[str, Any]] = []
    for lp_i in range(3):
        for cp_i in range(3):
            mask = r4_eligible & (lp_band_idx == lp_i) & (cp_band_idx == cp_i)
            cnt = int(np.sum(mask))
            bad = int(np.sum(y[mask])) if cnt > 0 else 0
            br = float(bad / cnt) if cnt > 0 else 0.0
            cluster = MATRIX_3X3_CLUSTER_MAP[(lp_i, cp_i)]
            is_ew = lp_i == 2 and cp_i == 2

            # Unit economics contrafactual da célula (LGD base = 65%)
            if cluster == "A":
                pnl_unit_efetivo = (1.0 - br) * (ead_unit * m_a) - br * (ead_unit * lgd_base) - opex
            elif cluster == "B":
                ead_b = ead_unit * ead_b_frac
                pnl_unit_efetivo = (1.0 - br) * (ead_b * m_b) - br * (ead_b * lgd_base) - opex
            else:
                pnl_unit_efetivo = 0.0

            # P&L unitário contrafactual caso fosse roteado para Cluster B (65% EAD @ 7,0%)
            ead_b_ref = ead_unit * ead_b_frac
            pnl_unit_se_b = (1.0 - br) * (ead_b_ref * m_b) - br * (ead_b_ref * lgd_base) - opex
            # Margem mínima de equilíbrio em 100% EAD e em 65% EAD
            be_margin_100 = ((br * ead_unit * lgd_base + opex) / ((1.0 - br) * ead_unit)) if br < 1.0 else 1.0
            be_margin_65 = ((br * ead_b_ref * lgd_base + opex) / ((1.0 - br) * ead_b_ref)) if br < 1.0 else 1.0

            cells.append(
                {
                    "lp_idx": lp_i,
                    "cp_idx": cp_i,
                    "lp_label": LP_LABELS[lp_i],
                    "cp_label": CP_LABELS[cp_i],
                    "cluster": cluster,
                    "early_warning": is_ew,
                    "volume": cnt,
                    "pct_r4": float(cnt / n_r4 * 100.0) if n_r4 > 0 else 0.0,
                    "maus": bad,
                    "bad_rate": br,
                    "pnl_por_contrato_efetivo_um": float(pnl_unit_efetivo),
                    "pnl_celula_efetivo_um": float(pnl_unit_efetivo * cnt),
                    "pnl_por_contrato_se_cluster_b_um": float(pnl_unit_se_b),
                    "breakeven_margem_100_ead": float(be_margin_100),
                    "breakeven_margem_65_ead": float(be_margin_65),
                }
            )
    return cells


def _compute_policy_pnl(
    y: np.ndarray,
    exposure_frac: np.ndarray,
    margin_rate: np.ndarray,
    lgd: float,
) -> dict[str, Any]:
    ead_unit = UNIT_ECONOMICS["ead_padrao_um"]
    opex = UNIT_ECONOMICS["opex_originacao_um"]
    approved = exposure_frac > 0.0
    ead_i = exposure_frac * ead_unit
    pnl_i = np.where(
        approved,
        np.where(y == 0, ead_i * margin_rate, -ead_i * lgd) - opex,
        0.0,
    )
    total_ead = float(np.sum(ead_i))
    bad_ead = float(np.sum(ead_i * y))
    total_pnl = float(np.sum(pnl_i))
    n_app = int(np.sum(approved))
    return {
        "aprovados": n_app,
        "taxa_aprovacao": float(n_app / len(y)),
        "bad_rate_volume": float(np.mean(y[approved])) if n_app > 0 else 0.0,
        "bad_rate_ead": float(bad_ead / total_ead) if total_ead > 0 else 0.0,
        "ead_total_um": total_ead,
        "pnl_total_um": total_pnl,
        "pnl_por_aprovado_um": float(total_pnl / n_app) if n_app > 0 else 0.0,
        "pnl_sobre_ead": float(total_pnl / total_ead) if total_ead > 0 else 0.0,
    }


def _policy_pnl_vector(
    y: np.ndarray,
    exposure_frac: np.ndarray,
    margin_rate: np.ndarray,
    lgd: float,
) -> np.ndarray:
    ead_unit = UNIT_ECONOMICS["ead_padrao_um"]
    opex = UNIT_ECONOMICS["opex_originacao_um"]
    approved = exposure_frac > 0.0
    ead_i = exposure_frac * ead_unit
    return np.where(
        approved,
        np.where(y == 0, ead_i * margin_rate, -ead_i * lgd) - opex,
        0.0,
    )


def simulate_policies_and_lgd_sensitivity(
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    scores_test: dict[str, np.ndarray],
    random_state: int = 42,
) -> dict[str, Any]:
    """Compara políticas (incluindo RF isolado pareado e igualdade de capital), atribuição fatorial e LGD com IC 95%."""
    lp_b = scores_test["lp_band_idx"]
    cp_b = scores_test["cp_band_idx"]
    pd_cp = scores_test["pd_cp"]
    pd_2d = scores_test["pd_2d"]
    n_test = len(y_test)

    m_a = UNIT_ECONOMICS["margem_liquida_24m_padrao"]
    m_b = UNIT_ECONOMICS["margem_liquida_24m_cluster_b"]
    ead_b = UNIT_ECONOMICS["cluster_b_fator_ead"]
    lgd_base = UNIT_ECONOMICS["lgd_base"]

    # Política 1: Conservadora Rígida
    funnel_rig = evaluate_funnel_layers(df_test, "rigida")
    rig_masks = {
        "cluster_c": funnel_rig["r4_eligible"] & (lp_b == 0),
        "cluster_b": np.zeros(n_test, dtype=bool),
        "cluster_a": funnel_rig["r4_eligible"] & (lp_b >= 1),
    }
    cons_rig = verify_mass_conservation(n_test, funnel_rig, r4_custom_masks=rig_masks)

    # Política 2: Agressiva 1D (Score LP isolado, sem filtro curto prazo RF)
    funnel_agr = evaluate_funnel_layers(df_test, "agressiva_1d")
    agr_masks = {
        "cluster_c": funnel_agr["r4_eligible"] & (lp_b == 0),
        "cluster_b": np.zeros(n_test, dtype=bool),
        "cluster_a": funnel_agr["r4_eligible"] & (lp_b >= 1),
    }
    cons_agr = verify_mass_conservation(n_test, funnel_agr, r4_custom_masks=agr_masks)

    # Política 3: Otimizada 2D Matriz 3x3 (C -> B -> A) sobre R1-R3 otimizadas
    funnel_opt = evaluate_funnel_layers(df_test, "otimizada")
    r4_opt = funnel_opt["r4_eligible"]
    clusters_opt = apply_matrix_3x3_clusters(r4_opt, lp_b, cp_b)
    cons_opt = verify_mass_conservation(n_test, funnel_opt, clusters=clusters_opt)

    n_a_opt = int(np.sum(clusters_opt["cluster_a"]))
    n_b_opt = int(np.sum(clusters_opt["cluster_b"]))
    ead_units_opt = int(round(n_a_opt + ead_b * n_b_opt))

    # Ranking determinístico do RF CP Isolado dentro dos elegíveis R4
    order_rf = np.argsort(np.where(r4_opt, pd_cp, 999.0), kind="mergesort")
    order_2d = np.argsort(np.where(r4_opt, pd_2d, 999.0), kind="mergesort")

    # Política 4: RF CP Isolado 1D (corte único a 100% EAD, mesma exposição EAD total da Otimizada 2D)
    rf1d_cut_masks = {
        "cluster_a": np.zeros(n_test, dtype=bool),
        "cluster_b": np.zeros(n_test, dtype=bool),
        "cluster_c": r4_opt.copy(),
    }
    rf1d_cut_masks["cluster_a"][order_rf[:ead_units_opt]] = True
    rf1d_cut_masks["cluster_c"][order_rf[:ead_units_opt]] = False
    cons_rf1d_cut = verify_mass_conservation(n_test, funnel_opt, r4_custom_masks=rf1d_cut_masks)

    # Política 5: RF CP Isolado Pareado A/B (mesma taxa de aprovação, mesmo EAD e mesma tabela 5,5%/7,0%)
    rf1d_pair_masks = {
        "cluster_a": np.zeros(n_test, dtype=bool),
        "cluster_b": np.zeros(n_test, dtype=bool),
        "cluster_c": r4_opt.copy(),
    }
    idx_a_rf = order_rf[:n_a_opt]
    idx_b_rf = order_rf[n_a_opt : n_a_opt + n_b_opt]
    rf1d_pair_masks["cluster_a"][idx_a_rf] = True
    rf1d_pair_masks["cluster_b"][idx_b_rf] = True
    rf1d_pair_masks["cluster_c"][idx_a_rf] = False
    rf1d_pair_masks["cluster_c"][idx_b_rf] = False
    cons_rf1d_pair = verify_mass_conservation(n_test, funnel_opt, r4_custom_masks=rf1d_pair_masks)

    exp_opt = np.where(clusters_opt["cluster_a"], 1.0, np.where(clusters_opt["cluster_b"], ead_b, 0.0))
    m_opt = np.where(clusters_opt["cluster_b"], m_b, m_a)

    exp_rf1d_pair = np.where(rf1d_pair_masks["cluster_a"], 1.0, np.where(rf1d_pair_masks["cluster_b"], ead_b, 0.0))
    m_rf1d_pair = np.where(rf1d_pair_masks["cluster_b"], m_b, m_a)

    policy_specs = [
        (
            "conservadora_rigida",
            "Conservadora Rígida (LP 1D)",
            np.where(rig_masks["cluster_a"], 1.0, 0.0),
            np.full(n_test, m_a),
            funnel_rig,
            rig_masks,
            cons_rig,
        ),
        (
            "agressiva_1d",
            "Agressiva 1D (Score LP + R2/R3 flex)",
            np.where(agr_masks["cluster_a"], 1.0, 0.0),
            np.full(n_test, m_a),
            funnel_agr,
            agr_masks,
            cons_agr,
        ),
        (
            "rf_cp_isolado_1d_mesmo_ead",
            "RF CP Isolado 1D (corte único @ mesmo EAD)",
            np.where(rf1d_cut_masks["cluster_a"], 1.0, 0.0),
            np.full(n_test, m_a),
            funnel_opt,
            rf1d_cut_masks,
            cons_rf1d_cut,
        ),
        (
            "rf_cp_isolado_pareado_ab",
            "RF CP Isolado 1D (pareado A/B + pricing 7,0%)",
            exp_rf1d_pair,
            m_rf1d_pair,
            funnel_opt,
            rf1d_pair_masks,
            cons_rf1d_pair,
        ),
        (
            "otimizada_2d",
            "Otimizada 2D (Matriz 3×3 C→B→A)",
            exp_opt,
            m_opt,
            funnel_opt,
            clusters_opt,
            cons_opt,
        ),
    ]

    base_summary: dict[str, Any] = {}
    for key, label, exp_arr, margin_arr, fun, cl_or_m, cons in policy_specs:
        res_base = _compute_policy_pnl(y_test, exp_arr, margin_arr, lgd_base)
        res_base["nome"] = label
        res_base["conservacao"] = cons
        res_base["pnl_margem_uniforme_55_um"] = _compute_policy_pnl(
            y_test, exp_arr, np.full(n_test, m_a), lgd_base
        )["pnl_total_um"]
        res_base["funil_etapas"] = build_funnel_stage_table(
            y_test,
            fun,
            clusters=cl_or_m if key == "otimizada_2d" else None,
            r4_custom_masks=cl_or_m if key != "otimizada_2d" else None,
        )
        if key == "otimizada_2d":
            ew_mask = clusters_opt["early_warning"]
            res_base["early_warning_retido"] = {
                "volume": int(np.sum(ew_mask)),
                "registros_maus_observados": int(np.sum(y_test[ew_mask])),
                "bad_rate": float(np.mean(y_test[ew_mask])),
            }
            res_base["flags_preservadas_r4"] = {
                "flag_atraso_leve_total_r4": int(np.sum(fun["flag_atraso_leve"] & fun["r4_eligible"])),
                "flag_atraso_leve_aprovados_ab": int(
                    np.sum(
                        fun["flag_atraso_leve"]
                        & (clusters_opt["cluster_a"] | clusters_opt["cluster_b"])
                    )
                ),
                "flag_cap_pressao_total_r4": int(np.sum(fun["flag_cap_pressao"] & fun["r4_eligible"])),
                "flag_cap_pressao_aprovados_ab": int(
                    np.sum(
                        fun["flag_cap_pressao"]
                        & (clusters_opt["cluster_a"] | clusters_opt["cluster_b"])
                    )
                ),
            }
        base_summary[key] = res_base

    # Comparativo estrito sob IGUALDADE DE CAPITAL = 117,32M u.m. (EAD exato da Rígida, margem uniforme 5,5%)
    cap_117 = int(np.sum(rig_masks["cluster_a"]))
    exp_rf_117 = np.zeros(n_test, dtype=float)
    exp_rf_117[order_rf[:cap_117]] = 1.0
    exp_2d_117 = np.zeros(n_test, dtype=float)
    exp_2d_117[order_2d[:cap_117]] = 1.0
    m_unif = np.full(n_test, m_a)
    comp_117 = {
        "ead_teto_um": float(cap_117 * UNIT_ECONOMICS["ead_padrao_um"]),
        "n_contratos_100pct": cap_117,
        "conservadora_rigida": _compute_policy_pnl(y_test, np.where(rig_masks["cluster_a"], 1.0, 0.0), m_unif, lgd_base),
        "rf_cp_isolado_1d": _compute_policy_pnl(y_test, exp_rf_117, m_unif, lgd_base),
        "score_2d_continuo": _compute_policy_pnl(y_test, exp_2d_117, m_unif, lgd_base),
    }

    # Decomposição fatorial aditiva do P&L (Conservadora Rígida -> Otimizada 2D)
    pnl_rig = base_summary["conservadora_rigida"]["pnl_total_um"]
    pnl_agr = base_summary["agressiva_1d"]["pnl_total_um"]
    pnl_opt_m55 = base_summary["otimizada_2d"]["pnl_margem_uniforme_55_um"]
    pnl_opt_full = base_summary["otimizada_2d"]["pnl_total_um"]
    pnl_rf_pair = base_summary["rf_cp_isolado_pareado_ab"]["pnl_total_um"]
    atribuicao_pnl = {
        "baseline_conservadora_rigida_um": pnl_rig,
        "efeito_regras_r2_r3_sem_filtro_cp_um": float(pnl_agr - pnl_rig),
        "efeito_selecao_2d_e_contraproposta_margem_uniforme_55_um": float(pnl_opt_m55 - pnl_agr),
        "efeito_precificacao_cluster_b_55_para_70_um": float(pnl_opt_full - pnl_opt_m55),
        "delta_liquido_total_otimizada_vs_rigida_um": float(pnl_opt_full - pnl_rig),
        "total_otimizada_2d_um": pnl_opt_full,
        "custo_discretizacao_grade_3x3_vs_rf1d_pareado_um": float(pnl_opt_full - pnl_rf_pair),
    }

    # Sensibilidade a premissas comerciais simuladas no Cluster B (aceitação 50/75/100% x margem 5,5/6,25/7,0%)
    premissas_grid: list[dict[str, Any]] = []
    pnl_a_only = _compute_policy_pnl(
        y_test, np.where(clusters_opt["cluster_a"], 1.0, 0.0), np.full(n_test, m_a), lgd_base
    )
    exp_b_only = np.where(clusters_opt["cluster_b"], ead_b, 0.0)
    ead_a = pnl_a_only["ead_total_um"]
    for accept_rate in (0.50, 0.75, 1.00):
        for marg_b in (0.055, 0.0625, 0.070):
            res_b = _compute_policy_pnl(y_test, exp_b_only, np.full(n_test, marg_b), lgd_base)
            pnl_comb = pnl_a_only["pnl_total_um"] + accept_rate * res_b["pnl_total_um"]
            ead_comb = ead_a + accept_rate * res_b["ead_total_um"]
            premissas_grid.append(
                {
                    "aceitacao_contraproposta": accept_rate,
                    "margem_cluster_b": marg_b,
                    "ead_total_um": float(ead_comb),
                    "pnl_total_um": float(pnl_comb),
                    "pnl_sobre_ead": float(pnl_comb / ead_comb) if ead_comb > 0 else 0.0,
                }
            )

    # Sensibilidade de LGD (50% a 85%) com bandas IC 95% via bootstrap estratificado (250 réplicas)
    rng = np.random.default_rng(random_state)
    pos_idx = np.where(y_test == 1)[0]
    neg_idx = np.where(y_test == 0)[0]
    boot_indices = [
        np.concatenate(
            [
                rng.choice(pos_idx, size=len(pos_idx), replace=True),
                rng.choice(neg_idx, size=len(neg_idx), replace=True),
            ]
        )
        for _ in range(250)
    ]

    lgd_sensitivity: list[dict[str, Any]] = []
    for lgd_val in UNIT_ECONOMICS["lgd_grid"]:
        lgd_f = float(lgd_val)
        row: dict[str, Any] = {"lgd": lgd_f}
        for key, _, exp_arr, margin_arr, _, _, _ in policy_specs:
            p_vec = _policy_pnl_vector(y_test, exp_arr, margin_arr, lgd_f)
            row[key] = float(np.sum(p_vec))
            boots = np.array([float(np.sum(p_vec[idx])) for idx in boot_indices], dtype=float)
            row[f"{key}_ic95_inf"] = float(np.quantile(boots, 0.025))
            row[f"{key}_ic95_sup"] = float(np.quantile(boots, 0.975))
        lgd_sensitivity.append(row)

    return {
        "lgd_base": lgd_base,
        "comparativo_base": base_summary,
        "comparativo_igualdade_capital_117m": comp_117,
        "atribuicao_fatorial_pnl": atribuicao_pnl,
        "sensibilidade_premissas_comerciais": premissas_grid,
        "sensibilidade_lgd": lgd_sensitivity,
        "clusters_otimizada_test": clusters_opt,
        "funnel_otimizada_test": funnel_opt,
    }


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.parse_math": False,
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.8,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "svg.fonttype": "path",
        }
    )


def _save_figure(fig: plt.Figure, stem: str, out_dir: Path = FIGURES_DIR) -> tuple[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / f"{stem}.svg"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(svg_path), str(png_path)


def plot_funnel_waterfall(stages: list[dict[str, Any]], out_dir: Path = FIGURES_DIR) -> tuple[str, str]:
    """Figura 1: Funil waterfall R1 -> R2 -> R3 -> R4 [C -> B -> A] com Bad Rate por corte."""
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(12.2, 6.2))

    # Construir barras waterfall em ordem do funil:
    # Entrada -> -KO R1 -> -KO R2 -> -KO R3 -> -Cluster C -> Cluster B (65%) -> Cluster A (100%)
    stage_map = {s["etapa"]: s for s in stages}
    entrada = stage_map["Entrada"]
    ko_r1 = stage_map["KO R1 — Elegibilidade"]
    ko_r2 = stage_map["KO R2 — Restritivo/Bureau"]
    ko_r3 = stage_map["KO R3 — Capacidade/Alavancagem"]
    cl_c = stage_map["R4 Cluster C — Reprova / Early Warning"]
    cl_b = stage_map["R4 Cluster B — Contraproposta (65% EAD)"]
    cl_a = stage_map["R4 Cluster A — Aprova 100% (Fundo do Funil)"]

    n0 = entrada["volume"]
    after_r1 = n0 - ko_r1["volume"]
    after_r2 = after_r1 - ko_r2["volume"]
    after_r3 = after_r2 - ko_r3["volume"]
    after_c = after_r3 - cl_c["volume"]  # == cl_b + cl_a

    labels = [
        "Entrada\n(Teste OOS)",
        "KO R1\nElegibilidade",
        "KO R2\nAtraso Severo",
        "KO R3\nAlavancagem",
        "R4 Cluster C\nReprova / EW",
        "R4 Cluster B\nContraproposta",
        "R4 Cluster A\nAprova 100%",
    ]
    bottoms = [0, after_r1, after_r2, after_r3, after_c, cl_a["volume"], 0]
    heights = [
        n0,
        ko_r1["volume"],
        ko_r2["volume"],
        ko_r3["volume"],
        cl_c["volume"],
        cl_b["volume"],
        cl_a["volume"],
    ]
    bad_rates = [
        entrada["bad_rate"],
        ko_r1["bad_rate"],
        ko_r2["bad_rate"],
        ko_r3["bad_rate"],
        cl_c["bad_rate"],
        cl_b["bad_rate"],
        cl_a["bad_rate"],
    ]
    colors = [
        PALETTE["navy"],
        PALETTE["muted"],
        PALETTE["magenta"],
        PALETTE["purple"],
        "#B91C1C",
        PALETTE["amber"],
        PALETTE["emerald"],
    ]

    x = np.arange(len(labels))
    bars = ax.bar(x, heights, bottom=bottoms, width=0.66, color=colors, edgecolor="#FFFFFF", linewidth=1.0)

    # Conectores pontilhados entre etapas para leitura clara de cascata
    top_levels = [n0, after_r1, after_r2, after_r3, after_c, after_c, cl_a["volume"]]
    for idx in range(len(labels) - 1):
        y_conn = top_levels[idx] if idx == 0 else bottoms[idx]
        if idx == 4:
            y_conn = after_c
        elif idx == 5:
            y_conn = cl_a["volume"]
        ax.plot([idx + 0.33, idx + 0.67], [y_conn, y_conn], color="#94A3B8", linestyle="--", linewidth=0.9)

    for idx, bar in enumerate(bars):
        h = heights[idx]
        b = bottoms[idx]
        pct = h / n0 * 100.0
        br_pct = bad_rates[idx] * 100.0
        if idx in (0, 5, 6) and h > 4000:
            txt_in = f"{h:,}\n({pct:.1f}%)\nBR: {br_pct:.2f}%".replace(",", ".")
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                b + h / 2.0,
                txt_in,
                ha="center",
                va="center",
                fontsize=8.2,
                fontweight="bold",
                color="#FFFFFF",
                linespacing=1.25,
            )
        else:
            txt = f"{h:,} ({pct:.1f}%)\nBR: {br_pct:.2f}%".replace(",", ".")
            y_pos = b + h + 650
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_pos,
                txt,
                ha="center",
                va="bottom",
                fontsize=8.2,
                fontweight="bold",
                color=PALETTE["navy"],
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFFFFF", edgecolor="#E2E8F0", alpha=0.95),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.2, color=PALETTE["navy"], fontweight="bold")
    ax.set_ylim(0, n0 * 1.15)
    ax.set_ylabel("Volume de propostas (coorte de teste)", fontsize=10, color=PALETTE["navy"])
    ax.yaxis.grid(True, linestyle=":", color=PALETTE["grid"], alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    br_in = entrada["bad_rate"] * 100.0
    br_a = cl_a["bad_rate"] * 100.0
    ax.set_title(
        f"Funil R1 → R2 → R3 → R4 (C → B → A) reduz inadimplência de {br_in:.2f}% na entrada para {br_a:.2f}% no Cluster A",
        fontsize=12.4,
        fontweight="bold",
        color=PALETTE["navy"],
        loc="left",
        pad=18,
    )
    fig.text(
        0.01,
        0.935,
        f"Base: GiveMeSomeCredit (OpenML #46929) · Teste fora da amostra (N = {n0:,} registros · simulação contrafactual) · Conservação exata".replace(",", "."),
        fontsize=9.2,
        color=PALETTE["muted"],
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    return _save_figure(fig, "01_funil_politica_waterfall", out_dir)


def plot_matrix_3x3(
    cells: list[dict[str, Any]],
    out_dir: Path = FIGURES_DIR,
    cal_method: str = "bruto_nativo",
) -> tuple[str, str]:
    """Figura 2: Heatmap 3x3 Score LP x RF Curto Prazo destacando Clusters C/B/A e Early Warning."""
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(10.6, 6.8))

    # Matriz ordenada: linha 0 = Alto Score LP (topo visual), linha 2 = Baixo Score LP (base)
    br_grid = np.zeros((3, 3), dtype=float)
    cell_lookup = {(c["lp_idx"], c["cp_idx"]): c for c in cells}
    row_order = [2, 1, 0]  # Alto -> Médio -> Baixo Score LP (de cima para baixo)
    for r_pos, lp_i in enumerate(row_order):
        for cp_i in range(3):
            br_grid[r_pos, cp_i] = cell_lookup[(lp_i, cp_i)]["bad_rate"] * 100.0

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "risk_grad", ["#ECFDF5", "#FEF3C7", "#FECDD3", "#9F1239"]
    )
    im = ax.imshow(br_grid, cmap=cmap, vmin=0.0, vmax=max(14.0, float(np.max(br_grid)) * 1.05), aspect="auto")

    badge_colors = {"A": PALETTE["emerald"], "B": PALETTE["amber"], "C": PALETTE["magenta"]}

    for r_pos, lp_i in enumerate(row_order):
        for cp_i in range(3):
            c = cell_lookup[(lp_i, cp_i)]
            br_pct = c["bad_rate"] * 100.0
            cluster = c["cluster"]
            is_ew = c["early_warning"]
            text_color = "#FFFFFF" if br_pct >= 10.5 else PALETTE["navy"]

            # Borda destacada no Early Warning (Alto Score LP + Risco CP Alto)
            if is_ew:
                rect = patches.Rectangle(
                    (cp_i - 0.47, r_pos - 0.47),
                    0.94,
                    0.94,
                    linewidth=3.2,
                    edgecolor=PALETTE["navy"],
                    facecolor="none",
                    linestyle="--",
                    zorder=4,
                )
                ax.add_patch(rect)

            ax.text(
                cp_i,
                r_pos - 0.14,
                f"Cluster {cluster}" + (" · EARLY WARNING" if is_ew else ""),
                ha="center",
                va="center",
                fontsize=9.2 if is_ew else 9.6,
                fontweight="bold",
                color="#FFFFFF" if is_ew else badge_colors[cluster],
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor=PALETTE["navy"] if is_ew else "#FFFFFF",
                    edgecolor=badge_colors[cluster],
                    linewidth=1.3,
                ),
                zorder=5,
            )
            ax.text(
                cp_i,
                r_pos + 0.11,
                f"Bad Rate: {br_pct:.2f}%",
                ha="center",
                va="center",
                fontsize=12.0,
                fontweight="bold",
                color=text_color,
                zorder=5,
            )
            ax.text(
                cp_i,
                r_pos + 0.31,
                f"N = {c['volume']:,} ({c['pct_r4']:.1f}% da R4)".replace(",", "."),
                ha="center",
                va="center",
                fontsize=8.8,
                color=text_color,
                zorder=5,
            )

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(CP_LABELS, fontsize=10.2, fontweight="bold", color=PALETTE["navy"])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([LP_LABELS[i] for i in row_order], fontsize=10.2, fontweight="bold", color=PALETTE["navy"])
    rf_axis_tag = "Random Forest Nativo" if cal_method == "bruto_nativo" else "Random Forest Calibrado"
    ax.set_xlabel(f"Eixo Curto Prazo — {rf_axis_tag} (Pressão Imediata / Atraso Leve)", fontsize=10.2, color=PALETTE["navy"], labelpad=10)
    ax.set_ylabel("Eixo Longo Prazo — Score Estrutural", fontsize=10.2, color=PALETTE["navy"], labelpad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.03)
    cbar.set_label("Bad Rate real fora da amostra (%)", fontsize=9.5, color=PALETTE["navy"])

    ew_br = cell_lookup[(2, 2)]["bad_rate"] * 100.0
    aa_br = cell_lookup[(2, 0)]["bad_rate"] * 100.0
    ratio = ew_br / max(aa_br, 1e-6)
    ax.set_title(
        f"Matriz 3×3 isola Early Warning no Alto Score LP ({ew_br:.2f}% vs {aa_br:.2f}% no Cluster A; {ratio:.1f}× maior)",
        fontsize=12.0,
        fontweight="bold",
        color=PALETTE["navy"],
        loc="left",
        pad=18,
    )
    fig.text(
        0.01,
        0.935,
        "Elegíveis R4 fora da amostra · Ordem de saída: Cluster C (Reprova / EW) → Cluster B (Contraproposta 65% EAD) → Cluster A (Aprova 100%)",
        fontsize=9.1,
        color=PALETTE["muted"],
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    return _save_figure(fig, "02_matriz_3x3_score_rf", out_dir)


def plot_efficient_frontier_and_lgd(
    comparativo_base: dict[str, Any],
    lgd_sensitivity: list[dict[str, Any]],
    out_dir: Path = FIGURES_DIR,
) -> tuple[str, str]:
    """Figura 3: Fronteira Risco x Retorno (incluindo RF isolado mesmo EAD) e LGD com IC 95%."""
    _configure_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.8))

    specs = [
        ("conservadora_rigida", "Conservadora Rígida", PALETTE["muted"], "s", (-40, -56)),
        ("rf_cp_isolado_1d_mesmo_ead", "RF CP Isolado 1D (corte único)", PALETTE["blue"], "D", (-32, 46)),
        ("agressiva_1d", "Agressiva 1D (Score LP)", PALETTE["magenta"], "^", (-185, -14)),
        ("otimizada_2d", "Otimizada 2D (Matriz 3×3)", PALETTE["emerald"], "o", (18, 12)),
    ]

    for key, label, color, marker, offset in specs:
        d = comparativo_base[key]
        x_val = d["taxa_aprovacao"] * 100.0
        y_val = d["bad_rate_ead"] * 100.0
        pnl_m = d["pnl_total_um"] / 1e6
        roi = d["pnl_sobre_ead"] * 100.0
        ax1.scatter(x_val, y_val, s=195, color=color, marker=marker, edgecolor=PALETTE["navy"], linewidth=1.2, zorder=5)
        ax1.annotate(
            f"{label}\nAprov: {x_val:.1f}% | BR EAD: {y_val:.2f}%\nP&L: {pnl_m:.2f}M ({roi:.2f}% s/ EAD)",
            xy=(x_val, y_val),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.3,
            fontweight="bold",
            color=PALETTE["navy"],
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#FFFFFF", edgecolor=color, linewidth=1.2),
            arrowprops=dict(arrowstyle="-", color=color, lw=1.0),
            zorder=6,
        )

    ax1.set_xlim(32, 67)
    ax1.set_ylim(0.15, 3.6)
    ax1.set_xlabel("Taxa de aprovação na entrada (%)", fontsize=9.8, color=PALETTE["navy"])
    ax1.set_ylabel("Bad Rate ponderado por EAD (%)", fontsize=9.8, color=PALETTE["navy"])
    ax1.set_title("Fronteira Risco–Retorno (LGD base = 65%)", fontsize=10.8, fontweight="bold", color=PALETTE["navy"], loc="left")
    ax1.grid(True, linestyle=":", color=PALETTE["grid"])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    lgds = [row["lgd"] * 100.0 for row in lgd_sensitivity]
    lgd_specs = [
        ("conservadora_rigida", "Conservadora Rígida", PALETTE["muted"], "s", "-", 1.9, True),
        ("rf_cp_isolado_1d_mesmo_ead", "RF CP Isolado 1D (corte único)", PALETTE["blue"], "D", "-", 2.0, True),
        ("rf_cp_isolado_pareado_ab", "RF CP Pareado A/B (+pricing 7%)", PALETTE["purple"], "P", "--", 1.8, False),
        ("agressiva_1d", "Agressiva 1D (Score LP)", PALETTE["magenta"], "^", "-", 1.9, True),
        ("otimizada_2d", "Otimizada 2D (Matriz 3×3)", PALETTE["emerald"], "o", "-", 2.8, True),
    ]
    for key, label, color, marker, ls, lw, show_band in lgd_specs:
        pnls = [row[key] / 1e6 for row in lgd_sensitivity]
        if show_band:
            lo = [row[f"{key}_ic95_inf"] / 1e6 for row in lgd_sensitivity]
            hi = [row[f"{key}_ic95_sup"] / 1e6 for row in lgd_sensitivity]
            ax2.fill_between(lgds, lo, hi, color=color, alpha=0.11)
        ax2.plot(lgds, pnls, label=label, color=color, marker=marker, linestyle=ls, linewidth=lw, markersize=5.8)

    ax2.axvline(UNIT_ECONOMICS["lgd_base"] * 100.0, color=PALETTE["navy"], linestyle="--", linewidth=1.1, alpha=0.7)
    ax2.text(65.7, 5.42, "LGD Base (65%)", fontsize=8.3, color=PALETTE["navy"], fontweight="bold")
    ax2.set_xlabel("Loss Given Default — LGD (%)", fontsize=9.8, color=PALETTE["navy"])
    ax2.set_ylabel("Resultado Líquido Simulado OOS (Milhões u.m.)", fontsize=9.8, color=PALETTE["navy"])
    ax2.set_title("Sensibilidade do P&L ao estresse de LGD (IC 95% bootstrap)", fontsize=10.8, fontweight="bold", color=PALETTE["navy"], loc="left")
    ax2.set_ylim(2.5, 5.75)
    ax2.grid(True, linestyle=":", color=PALETTE["grid"])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="lower left", frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=8.2)

    p_opt = comparativo_base["otimizada_2d"]["pnl_total_um"] / 1e6
    p_rig = comparativo_base["conservadora_rigida"]["pnl_total_um"] / 1e6
    p_rf1d = comparativo_base["rf_cp_isolado_1d_mesmo_ead"]["pnl_total_um"] / 1e6
    fig.suptitle(
        f"Simulação OOS: Matriz 3×3 gera {p_opt:.2f}M u.m. vs {p_rf1d:.2f}M (RF 1D @ mesmo EAD) e {p_rig:.2f}M (Rígida)",
        fontsize=11.8,
        fontweight="bold",
        color=PALETTE["navy"],
        x=0.01,
        ha="left",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return _save_figure(fig, "03_fronteira_eficiente_pl", out_dir)


def plot_roc_and_calibration_curves(
    eval_2d: dict[str, Any],
    out_dir: Path = FIGURES_DIR,
) -> tuple[str, str]:
    """Figura 4: Curvas ROC comparativas (LP vs RF bruto/isotônico vs 2D) e Calibração real por decis."""
    _configure_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.8))

    r4_m = eval_2d["desempenho_teste_r4_elegiveis"]
    ic_m = eval_2d["auc_ic95_r4"]
    roc_d = eval_2d["curvas_roc_r4"]
    cal_d = eval_2d["curvas_calibracao_r4"]

    roc_specs = [
        ("score_lp_isolado", "Score LP isolado", PALETTE["muted"], "--", 1.9),
        ("rf_cp_isotonico", "RF CP isotônico", PALETTE["amber"], "-.", 2.0),
        ("rf_cp_bruto", "RF CP bruto nativo", PALETTE["blue"], "-", 2.3),
        ("combinacao_2d", "Combinação 2D (LP + RF)", PALETTE["emerald"], "-", 2.8),
    ]
    ax1.plot([0, 1], [0, 1], color="#CBD5E1", linestyle=":", linewidth=1.2, label="Aleatório (AUC 0,500)")
    for key, label, color, ls, lw in roc_specs:
        auc_v = r4_m[key]["auc"]
        lo, hi = ic_m[key]
        ax1.plot(
            roc_d[key]["fpr"],
            roc_d[key]["tpr"],
            color=color,
            linestyle=ls,
            linewidth=lw,
            label=f"{label}: AUC {auc_v:.4f} [{lo:.4f}–{hi:.4f}]",
        )
    ax1.set_xlabel("Taxa de Falsos Positivos (1 − Especificidade)", fontsize=9.6, color=PALETTE["navy"])
    ax1.set_ylabel("Taxa de Verdadeiros Positivos (Sensibilidade)", fontsize=9.6, color=PALETTE["navy"])
    ax1.set_title("Curvas ROC na população elegível R4 (N = 26.868 OOS)", fontsize=10.6, fontweight="bold", color=PALETTE["navy"], loc="left")
    ax1.grid(True, linestyle=":", color=PALETTE["grid"])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=8.1)

    cal_specs = [
        ("score_lp_isolado", "Score LP isolado", PALETTE["muted"], "s", "--"),
        ("rf_cp_isotonico", "RF CP isotônico", PALETTE["amber"], "^", "-."),
        ("rf_cp_sigmoide", "RF CP sigmoide", PALETTE["magenta"], "v", ":"),
        ("rf_cp_bruto", "RF CP bruto nativo", PALETTE["blue"], "D", "-"),
        ("combinacao_2d", "Combinação 2D", PALETTE["emerald"], "o", "-"),
    ]
    max_coord = 0.22
    ax2.plot([0, max_coord], [0, max_coord], color=PALETTE["navy"], linestyle="--", linewidth=1.3, label="Calibração ideal (y = x)")
    for key, label, color, marker, ls in cal_specs:
        bins = cal_d[key]
        xs = np.array([b["pd_media"] for b in bins], dtype=float)
        ys = np.array([b["bad_rate_obs"] for b in bins], dtype=float)
        ece_v = r4_m[key]["ece"]
        if key == "combinacao_2d":
            yerr_lo = np.maximum(0.0, ys - np.array([b["ic95_inf"] for b in bins], dtype=float))
            yerr_hi = np.maximum(0.0, np.array([b["ic95_sup"] for b in bins], dtype=float) - ys)
            ax2.errorbar(
                xs,
                ys,
                yerr=[yerr_lo, yerr_hi],
                color=color,
                marker=marker,
                linestyle=ls,
                linewidth=2.3,
                markersize=5.5,
                capsize=2.5,
                elinewidth=1.0,
                alpha=0.92,
                label=f"{label} (ECE {ece_v:.5f} · IC 95%)",
            )
        else:
            ax2.plot(
                xs,
                ys,
                color=color,
                marker=marker,
                linestyle=ls,
                linewidth=2.2 if key == "rf_cp_bruto" else 1.7,
                markersize=5.5,
                label=f"{label} (ECE {ece_v:.5f})",
            )
    ax2.set_xlim(0.0, max_coord)
    ax2.set_ylim(0.0, max_coord)
    ax2.set_xlabel("Probabilidade prevista média por decil (PD)", fontsize=9.6, color=PALETTE["navy"])
    ax2.set_ylabel("Inadimplência grave observada por decil (Bad Rate)", fontsize=9.6, color=PALETTE["navy"])
    ax2.set_title("Diagrama de confiabilidade (10 decis · ~2.687/decil R4)", fontsize=10.6, fontweight="bold", color=PALETTE["navy"], loc="left")
    ax2.grid(True, linestyle=":", color=PALETTE["grid"])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper left", frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=8.0)

    d_auc = eval_2d["ganho_2d_sobre_rf_bruto_r4_bootstrap"]["delta_auc_medio"]
    fig.suptitle(
        f"RF CP concentra o ganho preditivo (AUC {r4_m['rf_cp_bruto']['auc']:.4f}, ECE {r4_m['rf_cp_bruto']['ece']:.5f}); 2D soma +{d_auc:.4f} AUC (p < 0,001)",
        fontsize=11.6,
        fontweight="bold",
        color=PALETTE["navy"],
        x=0.01,
        ha="left",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return _save_figure(fig, "04_curvas_roc_e_calibracao", out_dir)


def run_study(refresh_data: bool = False) -> dict[str, Any]:
    """Executa o pipeline completo do Estudo 05, salva artefatos e retorna resultados."""
    df, meta = load_or_download_dataset(refresh=refresh_data)
    x_lp, x_cp, y = build_feature_matrices(df)
    splits = split_train_val_test(y, random_state=42)

    bundle = fit_and_calibrate_models(df, x_lp, x_cp, y, splits, random_state=42)

    df_test = df.iloc[splits.test].reset_index(drop=True)
    y_test = y[splits.test]
    scores_test = score_population(
        x_lp.iloc[splits.test].reset_index(drop=True),
        x_cp.iloc[splits.test].reset_index(drop=True),
        bundle,
    )

    sim = simulate_policies_and_lgd_sensitivity(df_test, y_test, scores_test, random_state=42)
    funnel_opt_test = sim["funnel_otimizada_test"]
    eval_2d = evaluate_complementarity_and_2d_gain(
        df_test, y_test, scores_test, funnel_opt_test["r4_eligible"], bundle=bundle, random_state=42
    )
    matrix_3x3_test = build_matrix_3x3_summary(
        y_test,
        funnel_opt_test["r4_eligible"],
        scores_test["lp_band_idx"],
        scores_test["cp_band_idx"],
    )

    # Conservação exata também na base completa (150.000) com os cortes congelados
    scores_full = score_population(x_lp, x_cp, bundle)
    funnel_full = evaluate_funnel_layers(df, "otimizada")
    clusters_full = apply_matrix_3x3_clusters(
        funnel_full["r4_eligible"], scores_full["lp_band_idx"], scores_full["cp_band_idx"]
    )
    cons_full = verify_mass_conservation(len(df), funnel_full, clusters=clusters_full)

    fig1_svg, fig1_png = plot_funnel_waterfall(
        sim["comparativo_base"]["otimizada_2d"]["funil_etapas"]
    )
    fig2_svg, fig2_png = plot_matrix_3x3(matrix_3x3_test, cal_method=bundle.cal_method)
    fig3_svg, fig3_png = plot_efficient_frontier_and_lgd(
        sim["comparativo_base"], sim["sensibilidade_lgd"]
    )
    fig4_svg, fig4_png = plot_roc_and_calibration_curves(eval_2d)

    comparativo_serializable = dict(sim["comparativo_base"])

    results: dict[str, Any] = {
        "estudo": "Estudo 05 — Funil de Política de Crédito (R1→R2→R3→R4) & Matriz 3×3 Score × Random Forest",
        "dataset": meta,
        "splits": {
            "treino_n": int(len(splits.train)),
            "validacao_n": int(len(splits.val)),
            "teste_n": int(len(splits.test)),
            "bad_rate_treino": float(np.mean(y[splits.train])),
            "bad_rate_validacao": float(np.mean(y[splits.val])),
            "bad_rate_teste": float(np.mean(y_test)),
            "splits_disjuntos_validados": True,
        },
        "eixos_modelagem": {
            "features_longo_prazo": FEATURES_LP,
            "features_curto_prazo": FEATURES_CP,
            "sobreposicao_features": 0,
            "metodo_calibracao_rf_escolhido": bundle.cal_method,
            "calibracao_validacao_5fold_oof": bundle.val_calibration_metrics,
            "calibracao_validacao_r4_5fold_oof": bundle.val_r4_calibration_metrics,
            "tercis_calibrados_na_validacao_r4": {
                "score_lp_tercis": list(bundle.lp_terciles),
                "pd_cp_tercis": list(bundle.cp_terciles),
            },
        },
        "eixos_complementares_e_ganho_2d": eval_2d,
        "conservacao_massa_base_completa": cons_full,
        "matriz_3x3_teste": matrix_3x3_test,
        "parametros_unit_economics": UNIT_ECONOMICS,
        "politicas_teste_oos": comparativo_serializable,
        "comparativo_igualdade_capital_117m": sim["comparativo_igualdade_capital_117m"],
        "atribuicao_fatorial_pnl": sim["atribuicao_fatorial_pnl"],
        "sensibilidade_premissas_comerciais": sim["sensibilidade_premissas_comerciais"],
        "sensibilidade_lgd_teste_oos": sim["sensibilidade_lgd"],
        "figuras": {
            "waterfall": [fig1_svg, fig1_png],
            "matriz_3x3": [fig2_svg, fig2_png],
            "fronteira_pl_lgd": [fig3_svg, fig3_png],
            "curvas_roc_e_calibracao": [fig4_svg, fig4_png],
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results


def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Estudo 05 — Funil de Política de Crédito e Matriz 3x3")
    parser.add_argument("--refresh-data", action="store_true", help="Rebaixa o snapshot público do OpenML.")
    args = parser.parse_args()

    res = run_study(refresh_data=args.refresh_data)
    print(f"Dataset: {res['dataset']['dataset_name']} (OpenML {res['dataset']['openml_data_id']}) | Linhas={res['dataset']['n_rows']} | Colunas={res['dataset']['n_columns']}")
    print(f"Calibracao RF selecionada na validacao: {res['eixos_modelagem']['metodo_calibracao_rf_escolhido']}")
    m_r4 = res["eixos_complementares_e_ganho_2d"]["desempenho_teste_r4_elegiveis"]
    print(f"AUC R4 Teste -> LP={m_r4['score_lp_isolado']['auc']:.4f} | RF_bruto={m_r4['rf_cp_bruto']['auc']:.4f} | RF_iso={m_r4['rf_cp_isotonico']['auc']:.4f} | 2D={m_r4['combinacao_2d']['auc']:.4f}")
    for pol_key, pol in res["politicas_teste_oos"].items():
        print(
            f"Politica [{pol['nome']}] -> Aprov={pol['taxa_aprovacao']*100:.2f}% ({pol['aprovados']}) | BR_EAD={pol['bad_rate_ead']*100:.2f}% | P&L={pol['pnl_total_um']/1e6:.2f}M | P&L/EAD={pol['pnl_sobre_ead']*100:.2f}%"
        )


if __name__ == "__main__":
    main()
