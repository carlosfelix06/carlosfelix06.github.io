"""Testes Automatizados de Qualidade, Estabilidade e MLOps - Credit Decision Lab.

Valida:
1. Contratos de dados estritos com Pydantic v2.
2. Latência (< 25ms) e determinismo da esteira de crédito (R1 a R4).
3. Cálculo canônico e limiares de PSI (Population Stability Index) e KS drift.
4. Otimizador de alocação de capital sob restrições de risco via SciPy.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np
from pydantic import ValidationError

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import capital_optimizer as co  # noqa: E402
import credit_decision_engine as cde  # noqa: E402
import risk_drift_monitor as rdm  # noqa: E402


class TestCreditDecisionLab(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = cde.CreditDecisionEngine()
        self.monitor = rdm.RiskDriftMonitor()

    # ==========================================================================
    # 1. Contratos Pydantic & Validações de Borda
    # ==========================================================================

    def test_01_pydantic_valid_payload_parsing(self) -> None:
        payload = {
            "proposal_id": "PROP-2026-001",
            "age": 38,
            "monthly_income": 8500.0,
            "number_of_dependents": 2,
            "number_of_open_credit_lines_and_loans": 5,
            "number_real_estate_loans_or_lines": 1,
            "revolving_utilization_of_unsecured_lines": 0.25,
            "debt_ratio": 0.32,
            "number_of_time30_59_days_past_due_not_worse": 0,
            "number_of_time60_89_days_past_due_not_worse": 0,
            "number_of_times90_days_late": 0,
            "requested_limit": 15000.0,
        }
        prop = cde.ProposalInput(**payload)
        self.assertEqual(prop.proposal_id, "PROP-2026-001")
        self.assertEqual(prop.age, 38)
        self.assertEqual(prop.requested_limit, 15000.0)

    def test_02_pydantic_rejects_invalid_inputs(self) -> None:
        base_payload = {
            "proposal_id": "PROP-BAD",
            "age": 30,
            "monthly_income": 5000.0,
            "number_of_open_credit_lines_and_loans": 3,
            "number_real_estate_loans_or_lines": 0,
            "revolving_utilization_of_unsecured_lines": 0.2,
            "debt_ratio": 0.3,
            "number_of_time30_59_days_past_due_not_worse": 0,
            "number_of_time60_89_days_past_due_not_worse": 0,
            "number_of_times90_days_late": 0,
            "requested_limit": 5000.0,
        }

        # Idade menor que 18 anos
        bad_age = {**base_payload, "age": 16}
        with self.assertRaises(ValidationError):
            cde.ProposalInput(**bad_age)

        # Renda negativa
        bad_inc = {**base_payload, "monthly_income": -100.0}
        with self.assertRaises(ValidationError):
            cde.ProposalInput(**bad_inc)

        # Limite solicitado abaixo do mínimo de R$ 100
        bad_lim = {**base_payload, "requested_limit": 50.0}
        with self.assertRaises(ValidationError):
            cde.ProposalInput(**bad_lim)

    # ==========================================================================
    # 2. Motor de Decisão, Latência e Regras de Risco (R1-R4)
    # ==========================================================================

    def test_03_filtro_r1_inadimplencia_grave(self) -> None:
        prop = cde.ProposalInput(
            proposal_id="PROP-R1",
            age=40,
            monthly_income=12000.0,
            number_of_dependents=1,
            number_of_open_credit_lines_and_loans=4,
            number_real_estate_loans_or_lines=1,
            revolving_utilization_of_unsecured_lines=0.15,
            debt_ratio=0.20,
            number_of_time30_59_days_past_due_not_worse=0,
            number_of_time60_89_days_past_due_not_worse=0,
            number_of_times90_days_late=1,  # Inadimplência 90+ dias
            requested_limit=20000.0,
        )
        res = self.engine.evaluate(prop)
        self.assertEqual(res.decision, "DECLINED_POLICY")
        self.assertEqual(res.approved_limit, 0.0)
        self.assertEqual(res.policy_trigger, "REJECTED_FILTER_R1")
        self.assertTrue(any("R1:" in reason for reason in res.rejection_reasons))
        self.assertTrue(len(res.audit_hash) == 64)

    def test_04_filtro_r2_comprometimento_renda(self) -> None:
        prop = cde.ProposalInput(
            proposal_id="PROP-R2",
            age=35,
            monthly_income=5000.0,
            number_of_dependents=0,
            number_of_open_credit_lines_and_loans=6,
            number_real_estate_loans_or_lines=0,
            revolving_utilization_of_unsecured_lines=0.40,
            debt_ratio=0.75,  # Debt Ratio 75% > teto de 50%
            number_of_time30_59_days_past_due_not_worse=0,
            number_of_time60_89_days_past_due_not_worse=0,
            number_of_times90_days_late=0,
            requested_limit=5000.0,
        )
        res = self.engine.evaluate(prop)
        self.assertEqual(res.decision, "DECLINED_POLICY")
        self.assertEqual(res.policy_trigger, "REJECTED_FILTER_R2")
        self.assertTrue(any("R2:" in reason for reason in res.rejection_reasons))

    def test_05_aprovacao_full_cluster_a(self) -> None:
        prop = cde.ProposalInput(
            proposal_id="PROP-PRIME",
            age=48,
            monthly_income=18000.0,
            number_of_dependents=1,
            number_of_open_credit_lines_and_loans=8,
            number_real_estate_loans_or_lines=2,
            revolving_utilization_of_unsecured_lines=0.12,
            debt_ratio=0.22,
            number_of_time30_59_days_past_due_not_worse=0,
            number_of_time60_89_days_past_due_not_worse=0,
            number_of_times90_days_late=0,
            requested_limit=30000.0,
        )
        res = self.engine.evaluate(prop)
        self.assertEqual(res.decision, "APPROVED_FULL")
        self.assertEqual(res.approved_limit, 30000.0)
        self.assertEqual(res.assigned_interest_rate_monthly, 0.045)
        self.assertIn("Cluster A", res.cluster_assignment or "")
        self.assertLess(res.execution_latency_ms, 25.0)

    def test_06_contraproposta_cluster_b(self) -> None:
        prop = cde.ProposalInput(
            proposal_id="PROP-MODERATE",
            age=29,
            monthly_income=4500.0,
            number_of_dependents=0,
            number_of_open_credit_lines_and_loans=4,
            number_real_estate_loans_or_lines=0,
            revolving_utilization_of_unsecured_lines=0.45,
            debt_ratio=0.35,
            number_of_time30_59_days_past_due_not_worse=1,  # Atraso leve 30d
            number_of_time60_89_days_past_due_not_worse=0,
            number_of_times90_days_late=0,
            requested_limit=10000.0,
        )
        res = self.engine.evaluate(prop)
        self.assertEqual(res.decision, "COUNTEROFFER_ADJUSTED")
        self.assertEqual(res.approved_limit, 6500.0)  # 65% do limite solicitado
        self.assertEqual(res.assigned_interest_rate_monthly, 0.070)  # Taxa ajustada ao risco
        self.assertIn("Cluster B", res.cluster_assignment or "")

    # ==========================================================================
    # 3. Observabilidade, PSI e Teste de Kolmogorov-Smirnov
    # ==========================================================================

    def test_07_psi_identico_retorna_estavel(self) -> None:
        np.random.seed(42)
        base = np.random.normal(loc=600, scale=80, size=5000)
        same = base.copy()
        report = self.monitor.calculate_psi(base, same, num_bins=10)
        self.assertAlmostEqual(report.psi_total, 0.0, places=3)
        self.assertEqual(report.status, "STABLE")

    def test_08_psi_detecta_drift_populacional_severo(self) -> None:
        np.random.seed(42)
        base = np.random.normal(loc=650, scale=50, size=5000)
        shifted = np.random.normal(loc=480, scale=70, size=5000)  # Mudança brusca para pior
        report = self.monitor.calculate_psi(base, shifted, num_bins=10)
        self.assertGreaterEqual(report.psi_total, 0.25)
        self.assertEqual(report.status, "SIGNIFICANT_DRIFT")
        self.assertIn("Drift populacional severo", report.action_recommended)

    def test_09_ks_drift_test_biamostral(self) -> None:
        np.random.seed(42)
        base = np.random.exponential(scale=1.0, size=1000)
        shifted = np.random.exponential(scale=1.5, size=1000)
        ks_rep = self.monitor.test_ks_drift(base, shifted, alpha=0.01)
        self.assertTrue(ks_rep.is_drift_detected)
        self.assertLess(ks_rep.p_value, 0.01)

    # ==========================================================================
    # 4. Otimização Convexa de Capital e Fronteira de Risco
    # ==========================================================================

    def test_10_otimizador_capital_respeita_restricoes(self) -> None:
        np.random.seed(42)
        n = 100
        limits = np.random.uniform(2000, 25000, size=n)
        pds = np.random.beta(a=1, b=15, size=n)  # PDs concentradas entre 1% e 15%

        budget_capital = 400_000.0
        max_el = 20_000.0
        constraints = co.OptimizationConstraints(
            total_capital_budget=budget_capital,
            max_expected_loss_budget=max_el,
            default_lgd=0.65,
        )
        optimizer = co.CreditCapitalOptimizer(constraints)
        summary, decisions = optimizer.optimize_portfolio(limits, pds, allow_counteroffer=True)

        self.assertTrue(summary.is_optimal)
        self.assertLessEqual(summary.total_allocated_capital, budget_capital * 1.001)
        self.assertLessEqual(summary.total_expected_loss, max_el * 1.001)
        self.assertGreater(summary.total_expected_pnl, 0.0)
        self.assertEqual(len(decisions), n)
        self.assertEqual(
            summary.total_approved_count + summary.total_counteroffer_count + summary.total_rejected_count,
            n,
        )

    def test_11_geracao_fronteira_eficiente(self) -> None:
        np.random.seed(42)
        n = 50
        limits = np.random.uniform(5000, 20000, size=n)
        pds = np.random.beta(a=1, b=12, size=n)

        constraints = co.OptimizationConstraints(
            total_capital_budget=200_000.0,
            max_expected_loss_budget=15_000.0,
        )
        optimizer = co.CreditCapitalOptimizer(constraints)
        frontier = optimizer.generate_efficient_frontier(limits, pds, num_points=5)

        self.assertEqual(len(frontier), 5)
        # O P&L esperado deve ser monotônico crescente ou estável conforme o orçamento de capital aumenta
        pnls = [pt["expected_pnl"] for pt in frontier]
        self.assertGreaterEqual(pnls[-1], pnls[0])


if __name__ == "__main__":
    unittest.main()
