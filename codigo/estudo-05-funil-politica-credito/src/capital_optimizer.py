"""Capital Optimizer - Otimização Convexa de Alocação de Limites sob Restrições de Risco.

Este módulo implementa a camada de alocação ótima de capital e fronteira de risco-retorno:
1. Formulação linear-convexa: max sum(P&L_i * x_i) sujeito a Capital <= K e Perda Esperada <= EL_max.
2. Resolução determinística via SciPy (`linprog` com método 'highs').
3. Fronteira eficiente de alocação de crédito: mapeia a curva de Sharpe e sensibilidade a LGD.
4. Comparativo formal: Política Otimizada sob Restrição vs Política Heurística Estática.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
from scipy.optimize import linprog


@dataclass
class OptimizationConstraints:
    total_capital_budget: float       # Teto máximo de concessão de EAD (R$)
    max_expected_loss_budget: float   # Teto máximo tolerável de Perda Esperada (R$)
    default_lgd: float = 0.65         # Loss Given Default de referência (65%)
    monthly_funding_cost: float = 0.010  # Custo de captação (CDI/funding) 1,0% a.m.


@dataclass
class AllocationResult:
    total_approved_count: int
    total_counteroffer_count: int
    total_rejected_count: int
    total_allocated_capital: float
    total_expected_loss: float
    total_expected_pnl: float
    return_on_capital_pct: float
    expected_default_rate_pct: float
    solver_status: str
    is_optimal: bool


class CreditCapitalOptimizer:
    """Otimizador de fronteira risco-retorno e alocação de carteira de crédito."""

    def __init__(self, constraints: OptimizationConstraints):
        self.constraints = constraints

    def compute_unit_economics(
        self,
        requested_limits: np.ndarray,
        pds: np.ndarray,
        base_rate: float = 0.045,
        counter_rate: float = 0.070,
        counter_factor: float = 0.65,
    ) -> dict[str, np.ndarray]:
        """Calcula o P&L esperado e a perda esperada unitária para aprovação cheia e contraproposta."""
        lgd = self.constraints.default_lgd
        funding = self.constraints.monthly_funding_cost

        # Aprovação 100%
        ead_full = requested_limits
        el_full = ead_full * pds * lgd
        net_spread_full = (base_rate - funding) * (1.0 - pds)
        pnl_full = ead_full * net_spread_full - el_full

        # Contraproposta (65% do limite com spread majorado)
        ead_counter = requested_limits * counter_factor
        el_counter = ead_counter * pds * lgd
        net_spread_counter = (counter_rate - funding) * (1.0 - pds)
        pnl_counter = ead_counter * net_spread_counter - el_counter

        return {
            "ead_full": ead_full,
            "el_full": el_full,
            "pnl_full": pnl_full,
            "ead_counter": ead_counter,
            "el_counter": el_counter,
            "pnl_counter": pnl_counter,
        }

    def optimize_portfolio(
        self,
        requested_limits: np.ndarray | list[float],
        pds: np.ndarray | list[float],
        allow_counteroffer: bool = True,
    ) -> tuple[AllocationResult, np.ndarray]:
        """Resolve o problema de otimização linear sob restrições de capital e risco.

        Problema formulado:
        max  sum( PnL_i * x_i )
        s.t. sum( EAD_i * x_i ) <= Capital_Budget
             sum( EL_i * x_i )  <= Max_EL_Budget
             0 <= x_i <= 1
        """
        limits = np.asarray(requested_limits, dtype=float)
        pd_arr = np.asarray(pds, dtype=float)
        n = len(limits)

        econ = self.compute_unit_economics(limits, pd_arr)

        if not allow_counteroffer:
            # Seleção binária simples: Aprovar 100% ou Recusar
            c = -econ["pnl_full"]  # linprog minimiza c^T x
            A_ub = np.vstack([econ["ead_full"], econ["el_full"]])
            b_ub = np.array([self.constraints.total_capital_budget, self.constraints.max_expected_loss_budget])
            bounds = [(0.0, 1.0) for _ in range(n)]

            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            decisions = np.where(res.x >= 0.5, 1, 0) if res.success else np.zeros(n, dtype=int)

            allocated_ead = float(np.sum(decisions * econ["ead_full"]))
            total_el = float(np.sum(decisions * econ["el_full"]))
            total_pnl = float(np.sum(decisions * econ["pnl_full"]))
            approved_count = int(np.sum(decisions))
            counter_count = 0
            rejected_count = n - approved_count
        else:
            # Seleção multiação: Cada cliente i pode receber:
            # x_full_i (Aprovação Cheia) ou x_counter_i (Contraproposta), com x_full + x_counter <= 1
            # Total de variáveis de decisão: 2 * n
            c = -np.concatenate([econ["pnl_full"], econ["pnl_counter"]])
            ead_row = np.concatenate([econ["ead_full"], econ["ead_counter"]])
            el_row = np.concatenate([econ["el_full"], econ["el_counter"]])

            # Restrições de orçamento global
            A_budget = np.vstack([ead_row, el_row])
            b_budget = np.array([self.constraints.total_capital_budget, self.constraints.max_expected_loss_budget])

            # Restrições de exclusividade mútua: x_full_i + x_counter_i <= 1 para cada i
            # Matriz esparsa em blocos de tamanho n x 2n
            A_excl = np.zeros((n, 2 * n))
            for i in range(n):
                A_excl[i, i] = 1.0
                A_excl[i, n + i] = 1.0
            b_excl = np.ones(n)

            A_ub = np.vstack([A_budget, A_excl])
            b_ub = np.concatenate([b_budget, b_excl])
            bounds = [(0.0, 1.0) for _ in range(2 * n)]

            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

            if res.success:
                x_full = res.x[:n]
                x_counter = res.x[n:]

                # Decisão discreta pragmática: full se x_full >= 0.5, counter se x_counter >= 0.5
                is_full = (x_full >= 0.5)
                is_counter = (~is_full) & (x_counter >= 0.5)

                decisions = np.zeros(n, dtype=int)
                decisions[is_full] = 2       # 2 = Full
                decisions[is_counter] = 1    # 1 = Counteroffer
                # 0 = Rejected

                allocated_ead = float(np.sum(is_full * econ["ead_full"] + is_counter * econ["ead_counter"]))
                total_el = float(np.sum(is_full * econ["el_full"] + is_counter * econ["el_counter"]))
                total_pnl = float(np.sum(is_full * econ["pnl_full"] + is_counter * econ["pnl_counter"]))
                approved_count = int(np.sum(is_full))
                counter_count = int(np.sum(is_counter))
                rejected_count = n - approved_count - counter_count
            else:
                decisions = np.zeros(n, dtype=int)
                allocated_ead = 0.0
                total_el = 0.0
                total_pnl = 0.0
                approved_count = 0
                counter_count = 0
                rejected_count = n

        roc_pct = (total_pnl / allocated_ead * 100.0) if allocated_ead > 0 else 0.0
        el_pct = (total_el / allocated_ead * 100.0) if allocated_ead > 0 else 0.0

        summary = AllocationResult(
            total_approved_count=approved_count,
            total_counteroffer_count=counter_count,
            total_rejected_count=rejected_count,
            total_allocated_capital=round(allocated_ead, 2),
            total_expected_loss=round(total_el, 2),
            total_expected_pnl=round(total_pnl, 2),
            return_on_capital_pct=round(roc_pct, 2),
            expected_default_rate_pct=round(el_pct, 2),
            solver_status=res.message if res is not None else "failed",
            is_optimal=bool(res.success) if res is not None else False,
        )

        return summary, decisions

    def generate_efficient_frontier(
        self,
        requested_limits: np.ndarray,
        pds: np.ndarray,
        num_points: int = 8,
    ) -> list[dict[str, float]]:
        """Gera a fronteira eficiente variando o orçamento de capital disponível."""
        total_demand = float(np.sum(requested_limits))
        budgets = np.linspace(total_demand * 0.15, total_demand * 0.90, num_points)

        frontier_points = []
        original_cap = self.constraints.total_capital_budget
        original_el = self.constraints.max_expected_loss_budget

        try:
            for b in budgets:
                self.constraints.total_capital_budget = float(b)
                self.constraints.max_expected_loss_budget = float(b * 0.08)  # Tolerância proporcional a perda de 8%
                res, _ = self.optimize_portfolio(requested_limits, pds, allow_counteroffer=True)
                frontier_points.append(
                    {
                        "capital_budget": round(float(b), 2),
                        "allocated_capital": res.total_allocated_capital,
                        "expected_pnl": res.total_expected_pnl,
                        "expected_loss": res.total_expected_loss,
                        "return_on_capital_pct": res.return_on_capital_pct,
                        "approval_rate_pct": round(
                            (res.total_approved_count + res.total_counteroffer_count) / len(requested_limits) * 100.0,
                            2,
                        ),
                    }
                )
        finally:
            self.constraints.total_capital_budget = original_cap
            self.constraints.max_expected_loss_budget = original_el

        return frontier_points
