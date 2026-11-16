import importlib.util
import json
import unittest
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = STUDY_ROOT / "src" / "estudo_alta_renda_publico.py"
SNAPSHOT_PATH = STUDY_ROOT / "resultados" / "snapshot_publico.json"
SUMMARY_PATH = STUDY_ROOT / "resultados" / "resultados.json"


def load_alta_renda_module():
    import sys
    if "estudo_alta_renda_publico" in sys.modules:
        return sys.modules["estudo_alta_renda_publico"]
    spec = importlib.util.spec_from_file_location("estudo_alta_renda_publico", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["estudo_alta_renda_publico"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestEstudoAltaRendaPublico(unittest.TestCase):
    def test_01_public_snapshot_and_summary_exist(self):
        self.assertTrue(SNAPSHOT_PATH.is_file())
        self.assertTrue(SUMMARY_PATH.is_file())

    def test_02_bank_pricing_persistence_and_spread(self):
        data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        b = data["bank_dispersion"]
        self.assertEqual(b["unique_institutions"], 138)
        self.assertEqual(b["persistence_11m"]["matched_bank_modality_pairs"], 370)
        self.assertAlmostEqual(b["persistence_11m"]["spearman_rho"], 0.8656, delta=0.005)
        nao_consig = next(
            m for m in b["modalities"]
            if "não consignado" in m["modalidade"].lower() or m["ratio_p90_p10"] > 14
        )
        self.assertAlmostEqual(nao_consig["ratio_p90_p10"], 15.57, delta=0.1)

    def test_03_state_panel_and_oos_rejection(self):
        data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        p = data["uf_analysis"]
        self.assertEqual(p["n_panel_obs"], 108)
        self.assertAlmostEqual(p["panel_pooled_r2"], 0.0529, delta=0.01)
        self.assertAlmostEqual(p["panel_fe_r2"], 0.5504, delta=0.01)
        self.assertGreater(p["oos_2025_ridge_mae_pp"], p["oos_2025_persist_mae_pp"])

    def test_04_string_normalization_and_modality_classification(self):
        """Auditoria de Engenharia: Executa diretamente a classificacao de garantias e normalizacao."""
        mod = load_alta_renda_module()
        self.assertEqual(mod._norm("São Paulo - Crédito"), "sao paulo - credito")
        self.assertEqual(mod._norm("  ÁGUA CLARA  "), "agua clara")

        # Classificacao de modalidades garantidas vs rotativas
        self.assertTrue(mod._is_secured_modality("Crédito Pessoal Consignado Público"))
        self.assertTrue(mod._is_secured_modality("Financiamento Imobiliário Taxa Fixa"))
        self.assertTrue(mod._is_secured_modality("Aquisição de Veículos PF"))
        self.assertFalse(mod._is_secured_modality("Cartão de Crédito Rotativo Total"))
        self.assertFalse(mod._is_secured_modality("Cheque Especial PF"))

    def test_05_bank_competitive_structure_execution(self):
        """Auditoria Estatistica: Reexecuta a esteira de dispersao, clustering e persistencia."""
        mod = load_alta_renda_module()
        snap = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
        res = mod._analyze_bank_competitive_structure(snap["olinda_pf"], snap["olinda_pf_prev"])

        self.assertEqual(res["unique_institutions"], 138)
        self.assertEqual(res["persistence_11m"]["matched_bank_modality_pairs"], 370)
        self.assertAlmostEqual(res["persistence_11m"]["spearman_rho"], 0.8656, delta=0.005)
        self.assertGreater(len(res["modalities"]), 0)


if __name__ == "__main__":
    unittest.main()
