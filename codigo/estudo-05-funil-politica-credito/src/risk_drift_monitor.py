"""Risk Drift Monitor - Observabilidade de Dados, Drift Populacional e Alertas de MLOps.

Este módulo implementa a camada de governança estatística e observabilidade contínua:
1. Population Stability Index (PSI) canônico com segmentação em 10 decis e suavização epsilon.
2. Teste biamostral de Kolmogorov-Smirnov (KS) para detecção de separabilidade e drift.
3. Checagem de integridade de dados (Data Quality Gates: taxa de nulos e range anomalies).
4. Relatório executivo automatizado de risco para acionamento de comitê de crédito ou re-treino.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

import numpy as np
from scipy import stats


@dataclass
class PSICell:
    bin_index: int
    lower_bound: float
    upper_bound: float
    baseline_count: int
    baseline_pct: float
    actual_count: int
    actual_pct: float
    psi_contribution: float


@dataclass
class PSIReport:
    psi_total: float
    status: Literal["STABLE", "MODERATE_SHIFT", "SIGNIFICANT_DRIFT"]
    action_recommended: str
    num_bins: int
    total_baseline: int
    total_actual: int
    breakdown: list[PSICell]


@dataclass
class KSTestReport:
    ks_statistic: float
    p_value: float
    is_drift_detected: bool
    significance_level: float


@dataclass
class DataQualityReport:
    feature_name: str
    baseline_missing_rate: float
    actual_missing_rate: float
    missing_rate_drift: float
    out_of_bounds_count: int
    out_of_bounds_pct: float
    status: Literal["OK", "WARNING", "CRITICAL"]


class RiskDriftMonitor:
    """Monitor de observabilidade estatística e estabilidade de modelos de crédito."""

    def __init__(self, epsilon: float = 1e-5):
        self.epsilon = epsilon

    def calculate_psi(
        self,
        baseline: np.ndarray | list[float],
        actual: np.ndarray | list[float],
        num_bins: int = 10,
        custom_bins: Optional[np.ndarray] = None,
    ) -> PSIReport:
        """Calcula o Population Stability Index (PSI) canônico.

        Classificação Regulatória Padrão:
        - PSI < 0.10: Estável. Nenhuma intervenção necessária.
        - 0.10 <= PSI < 0.25: Alerta Moderado. Monitorar safra subsequente.
        - PSI >= 0.25: Drift Significativo. Necessário re-treino ou ajuste de régua.
        """
        base_arr = np.asarray(baseline, dtype=float)
        act_arr = np.asarray(actual, dtype=float)

        base_clean = base_arr[~np.isnan(base_arr)]
        act_clean = act_arr[~np.isnan(act_arr)]

        if len(base_clean) == 0 or len(act_clean) == 0:
            raise ValueError("Baseline e actual devem conter observações válidas e não nulas.")

        if custom_bins is not None:
            bin_edges = np.asarray(custom_bins, dtype=float)
        else:
            # Gerar decis a partir do baseline
            quantiles = np.linspace(0.0, 1.0, num_bins + 1)
            bin_edges = np.percentile(base_clean, quantiles * 100.0)
            # Garantir estrita monotonicidade para evitar bins de largura zero
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            for i in range(1, len(bin_edges) - 1):
                if bin_edges[i] <= bin_edges[i - 1]:
                    bin_edges[i] = bin_edges[i - 1] + 1e-6

        n_base = len(base_clean)
        n_act = len(act_clean)

        base_counts, _ = np.histogram(base_clean, bins=bin_edges)
        act_counts, _ = np.histogram(act_clean, bins=bin_edges)

        psi_cells: list[PSICell] = []
        psi_total = 0.0

        for i in range(len(base_counts)):
            b_cnt = int(base_counts[i])
            a_cnt = int(act_counts[i])

            b_pct = max(b_cnt / n_base, self.epsilon)
            a_pct = max(a_cnt / n_act, self.epsilon)

            cell_psi = (a_pct - b_pct) * np.log(a_pct / b_pct)
            psi_total += cell_psi

            psi_cells.append(
                PSICell(
                    bin_index=i + 1,
                    lower_bound=float(bin_edges[i]),
                    upper_bound=float(bin_edges[i + 1]),
                    baseline_count=b_cnt,
                    baseline_pct=round(b_cnt / n_base, 4),
                    actual_count=a_cnt,
                    actual_pct=round(a_cnt / n_act, 4),
                    psi_contribution=round(float(cell_psi), 6),
                )
            )

        psi_total_val = round(float(psi_total), 5)

        if psi_total_val < 0.10:
            status = "STABLE"
            rec = "População estável. Modelo operando dentro das premissas de validação."
        elif psi_total_val < 0.25:
            status = "MODERATE_SHIFT"
            rec = "Alerta moderado de drift. Aumentar cadência de monitoramento para ciclo semanal."
        else:
            status = "SIGNIFICANT_DRIFT"
            rec = "Drift populacional severo detectado. Acionar comitê de crédito e pipeline de re-treino."

        return PSIReport(
            psi_total=psi_total_val,
            status=status,
            action_recommended=rec,
            num_bins=len(base_counts),
            total_baseline=n_base,
            total_actual=n_act,
            breakdown=psi_cells,
        )

    def test_ks_drift(
        self,
        baseline: np.ndarray | list[float],
        actual: np.ndarray | list[float],
        alpha: float = 0.01,
    ) -> KSTestReport:
        """Executa teste biamostral Kolmogorov-Smirnov (KS) para avaliar se as distribuições diferem."""
        base_clean = np.asarray(baseline, dtype=float)
        act_clean = np.asarray(actual, dtype=float)

        base_clean = base_clean[~np.isnan(base_clean)]
        act_clean = act_clean[~np.isnan(act_clean)]

        res = stats.ks_2samp(base_clean, act_clean)
        ks_stat = float(res.statistic)
        p_val = float(res.pvalue)

        is_drift = bool(p_val < alpha)

        return KSTestReport(
            ks_statistic=round(ks_stat, 5),
            p_value=float(p_val),
            is_drift_detected=is_drift,
            significance_level=alpha,
        )

    def audit_data_quality(
        self,
        feature_name: str,
        baseline_series: np.ndarray | list[float],
        actual_series: np.ndarray | list[float],
        lower_limit: Optional[float] = None,
        upper_limit: Optional[float] = None,
    ) -> DataQualityReport:
        """Audita integridade de variáveis: taxa de preenchimento e valores fora da distribuição esperada."""
        base_arr = np.asarray(baseline_series, dtype=float)
        act_arr = np.asarray(actual_series, dtype=float)

        b_nans = int(np.sum(np.isnan(base_arr)))
        a_nans = int(np.sum(np.isnan(act_arr)))

        b_rate = b_nans / len(base_arr) if len(base_arr) > 0 else 0.0
        a_rate = a_nans / len(act_arr) if len(act_arr) > 0 else 0.0
        drift_rate = a_rate - b_rate

        # Range bounds padrão baseados no percentil 0.1% e 99.9% do baseline se não fornecidos
        base_valid = base_arr[~np.isnan(base_arr)]
        if lower_limit is None and len(base_valid) > 0:
            lower_limit = float(np.percentile(base_valid, 0.05))
        if upper_limit is None and len(base_valid) > 0:
            upper_limit = float(np.percentile(base_valid, 99.95))

        act_valid = act_arr[~np.isnan(act_arr)]
        oob_count = 0
        if lower_limit is not None and upper_limit is not None and len(act_valid) > 0:
            oob_mask = (act_valid < lower_limit) | (act_valid > upper_limit)
            oob_count = int(np.sum(oob_mask))

        oob_pct = oob_count / len(act_valid) if len(act_valid) > 0 else 0.0

        if abs(drift_rate) > 0.15 or oob_pct > 0.05:
            status = "CRITICAL"
        elif abs(drift_rate) > 0.05 or oob_pct > 0.01:
            status = "WARNING"
        else:
            status = "OK"

        return DataQualityReport(
            feature_name=feature_name,
            baseline_missing_rate=round(b_rate, 4),
            actual_missing_rate=round(a_rate, 4),
            missing_rate_drift=round(drift_rate, 4),
            out_of_bounds_count=oob_count,
            out_of_bounds_pct=round(oob_pct, 4),
            status=status,
        )

    def generate_governance_report(
        self,
        score_baseline: np.ndarray,
        score_actual: np.ndarray,
        features_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, Any]:
        """Gera sumário consolidado de governança e integridade de modelo."""
        psi_rep = self.calculate_psi(score_baseline, score_actual)
        ks_rep = self.test_ks_drift(score_baseline, score_actual)

        dq_reports = []
        for feat_name, (b_feat, a_feat) in features_dict.items():
            dq = self.audit_data_quality(feat_name, b_feat, a_feat)
            dq_reports.append(asdict(dq))

        overall_status = "HEALTHY"
        if psi_rep.status == "SIGNIFICANT_DRIFT" or ks_rep.is_drift_detected:
            overall_status = "DRIFT_ALERT"
        elif psi_rep.status == "MODERATE_SHIFT":
            overall_status = "ATTENTION"

        return {
            "overall_status": overall_status,
            "psi_score": psi_rep.psi_total,
            "psi_status": psi_rep.status,
            "ks_statistic": ks_rep.ks_statistic,
            "ks_p_value": ks_rep.p_value,
            "data_quality_checks": dq_reports,
            "action_plan": psi_rep.action_recommended,
        }
