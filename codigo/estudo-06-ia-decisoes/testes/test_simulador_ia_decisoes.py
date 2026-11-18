"""Testes unitários e de integração para o Simulador do Estudo 06 (KAN-61)."""

import json
from pathlib import Path
import unittest

ESTUDO_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ESTUDO_DIR / "src"
RESULTADOS_DIR = ESTUDO_DIR / "resultados"


class TestSimuladorIaDecisoes(unittest.TestCase):
    """Testa a geração de propostas, os paradigmas de decisão e a exportação de métricas."""

    def test_01_script_execution_and_schema(self):
        """Verifica se o JSON gerado contém a estrutura e tipos esperados."""
        json_path = RESULTADOS_DIR / "metricas_comparativo_ia_vs_ml.json"
        self.assertTrue(json_path.is_file(), "Arquivo de resultados deve existir")

        with open(json_path, encoding="utf-8") as f:
            dados = json.load(f)

        self.assertEqual(dados.get("estudo"), "KAN-61")
        self.assertEqual(dados.get("seed"), 61)
        self.assertEqual(dados["amostras"]["total"], 5000)
        self.assertEqual(dados["amostras"]["treino"], 3000)
        self.assertEqual(dados["amostras"]["calibracao"], 1000)
        self.assertEqual(dados["amostras"]["teste"], 1000)

        resultados = dados.get("resultados", [])
        self.assertEqual(len(resultados), 9, "Devem existir 9 combinações (3 paradigmas x 3 tickets)")

        paradigmas = {r["paradigma"] for r in resultados}
        self.assertEqual(paradigmas, {"heuristica", "ml_tradicional", "ia_hibrida"})

    def test_02_metricas_ordenacao_economica(self):
        """Valida se o poder discriminatório (AUC e KS) respeita a superioridade estatística."""
        with open(RESULTADOS_DIR / "metricas_comparativo_ia_vs_ml.json", encoding="utf-8") as f:
            dados = json.load(f)

        resultados = dados["resultados"]
        # Agrupar por paradigma para validar médias
        by_paradigm = {}
        for r in resultados:
            by_paradigm.setdefault(r["paradigma"], []).append(r)

        mean_auc_heuristica = sum(r["auc_roc"] for r in by_paradigm["heuristica"]) / 3
        mean_auc_ml = sum(r["auc_roc"] for r in by_paradigm["ml_tradicional"]) / 3
        mean_auc_ia = sum(r["auc_roc"] for r in by_paradigm["ia_hibrida"]) / 3

        self.assertLess(mean_auc_heuristica, mean_auc_ml)
        self.assertLess(mean_auc_ml, mean_auc_ia)

        # FinOps: custo da heurística < ML < IA Híbrida
        mean_cost_heuristica = sum(r["custo_por_decisao_brl"] for r in by_paradigm["heuristica"]) / 3
        mean_cost_ml = sum(r["custo_por_decisao_brl"] for r in by_paradigm["ml_tradicional"]) / 3
        mean_cost_ia = sum(r["custo_por_decisao_brl"] for r in by_paradigm["ia_hibrida"]) / 3

        self.assertLess(mean_cost_heuristica, mean_cost_ml)
        self.assertLess(mean_cost_ml, mean_cost_ia)


if __name__ == "__main__":
    unittest.main()
