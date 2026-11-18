import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

STUDY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = STUDY_ROOT / "src" / "experimento_mdm_cvm.py"
SNAPSHOT_PATH = STUDY_ROOT / "resultados" / "cvm_snapshot.json"
SUMMARY_PATH = STUDY_ROOT / "resultados" / "resultados.json"


def load_mdm_module():
    import sys
    if "experimento_mdm_cvm" in sys.modules:
        return sys.modules["experimento_mdm_cvm"]
    spec = importlib.util.spec_from_file_location("experimento_mdm_cvm", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["experimento_mdm_cvm"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestExperimentoMdmCvm(unittest.TestCase):
    def test_01_cvm_snapshot_and_summary_exist(self):
        self.assertTrue(SNAPSHOT_PATH.is_file())
        self.assertTrue(SUMMARY_PATH.is_file())

    def test_02_golden_record_deduplication_and_survivorship(self):
        data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(data["raw_records"], 7905)
        self.assertEqual(data["golden_records"], 6736)
        self.assertAlmostEqual(abs(data["dedup_reduction_percent"]), 14.8, delta=0.1)
        self.assertEqual(data["conflict_sit"], 42)
        self.assertEqual(data["flipped_sit_trust_vs_recency"], 33)
        self.assertAlmostEqual(
            data["flipped_sit_trust_vs_recency"] / data["conflict_sit"] * 100.0,
            78.57,
            delta=0.1,
        )

    def test_03_blind_record_linkage_benchmarks(self):
        data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        regimes = data["record_linkage"]["regimes"]
        formal = regimes["legal_name"]["models"]
        drift = regimes["commercial_drift"]["models"]
        self.assertGreater(formal["exact_normalized"]["f1"], 0.98)
        self.assertAlmostEqual(drift["exact_normalized"]["pr_auc"], 0.7626, delta=0.01)
        self.assertAlmostEqual(drift["hybrid_logistic"]["pr_auc"], 0.8835, delta=0.01)
        if "hybrid_hgb" in drift:
            self.assertAlmostEqual(drift["hybrid_hgb"]["pr_auc"], 0.8835, delta=0.01)
        self.assertGreater(
            drift["hybrid_logistic"]["pr_auc"] - drift["exact_normalized"]["pr_auc"],
            0.11,
        )

    def test_04_cnpj_cleaning_and_normalization_pipeline(self):
        """Auditoria de Engenharia: Executa diretamente os algoritmos de sanitizacao cadastral."""
        mod = load_mdm_module()
        self.assertEqual(mod._clean_cnpj("33.000.167/0001-01"), "33000167000101")
        self.assertEqual(mod._clean_cnpj("  33000167000101  "), "33000167000101")
        self.assertEqual(mod._clean_cnpj("12345678901"), "00012345678901")
        self.assertEqual(mod._clean_cnpj(None), "")
        self.assertEqual(mod._clean_cnpj(""), "")
        self.assertEqual(mod._clean_cnpj("invalid-alpha"), "")

        # Normalizacao textual de razao social
        norm = mod._normalize_text("PETRÓLEO BRASILEIRO S.A. - PETROBRAS")
        self.assertIn("PETROLEO", norm)
        self.assertIn("BRASILEIRO", norm)

        # Extracao de core de negocio (remocao de sulfixos societarios)
        core = mod._normalize_core("BANCO DO BRASIL S/A")
        self.assertNotIn("S/A", core)
        self.assertIn("BANCO DO BRASIL", core)

    def test_05_mdm_deduplication_engine_execution(self):
        """Auditoria de Modelagem: Reexecuta a esteira de deduplicacao e survivorship sobre registros sinteticos."""
        mod = load_mdm_module()
        CvmRecord = mod.CvmRecord

        rec1 = CvmRecord(
            source="cad_cia_aberta.csv",
            trust_rank=1,
            cnpj="11111111000111",
            denom_social="EMPRESA ALFA S.A.",
            denom_comerc="ALFA TECH",
            uf="SP",
            sit="FASE OPERACIONAL",
            dt_ini_sit="2023-01-01",
        )
        rec2 = CvmRecord(
            source="cad_fi.csv",
            trust_rank=2,
            cnpj="11111111000111",
            denom_social="EMPRESA ALFA SERVICOS LTDA",
            denom_comerc="ALFA TECH",
            uf="SP",
            sit="CANCELADA",
            dt_ini_sit="2024-06-01",  # Mais recente, mas menor confianca
        )
        rec3 = CvmRecord(
            source="cad_cia_aberta.csv",
            trust_rank=1,
            cnpj="22222222000122",
            denom_social="EMPRESA BETA S.A.",
            denom_comerc="BETA INVEST",
            uf="RJ",
            sit="FASE OPERACIONAL",
            dt_ini_sit="2022-05-10",
        )

        res = mod.analyze_cvm_mdm([rec1, rec2, rec3])
        self.assertEqual(res["raw_records"], 3)
        self.assertEqual(res["golden_records"], 2)
        self.assertAlmostEqual(res["dedup_reduction_percent"], -33.33, delta=0.1)
        self.assertEqual(res["conflict_sit"], 1)

    def test_06_f1_threshold_optimization_execution(self):
        """Auditoria Estatistica: Testa a calibracao de limiar F1 de classificacao."""
        mod = load_mdm_module()
        y_true = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.1])
        best_th = mod._best_f1_threshold(y_true, scores)
        self.assertGreaterEqual(best_th, 0.4)
        self.assertLessEqual(best_th, 0.8)

    def test_07_pipeline_medalhao_duckdb_execution(self):
        """Auditoria de Engenharia: Executa e valida o pipeline Medalhao (Bronze -> Silver -> Gold) no DuckDB."""
        import sys
        import pytest
        pytest.importorskip("duckdb", reason="DuckDB não instalado neste ambiente")
        sys.path.insert(0, str(STUDY_ROOT / "src"))
        import pipeline_medalhao_duckdb as pmd
        import duckdb

        summary = pmd.run_pipeline(refresh_db=False)
        self.assertIn("layers", summary)
        self.assertEqual(summary["layers"]["bronze"]["total_rows"], 7905)
        self.assertEqual(summary["layers"]["gold"]["golden_records"], 6736)
        self.assertEqual(summary["layers"]["gold"]["multi_papel_records"], 813)
        self.assertGreater(summary["layers"]["gold"]["financial_impact"]["concentrated_credit_exposure_brl"], 100_000_000.0)

        # Valida conexao e consultas SQL diretas nas camadas
        con = duckdb.connect(str(pmd.LAKEHOUSE_DB), read_only=True)
        bronze_cnt = con.execute("SELECT count(*) FROM bronze_cvm_cadastros").fetchone()[0]
        silver_cnt = con.execute("SELECT count(*) FROM silver_cvm_entidades_conformed").fetchone()[0]
        gold_cnt = con.execute("SELECT count(*) FROM gold_dim_empresa_golden_record").fetchone()[0]
        con.close()

        self.assertEqual(bronze_cnt, 7905)
        self.assertEqual(silver_cnt, 7905)
        self.assertEqual(gold_cnt, 6736)
        self.assertTrue((pmd.RESULTS_DIR / "bronze_cvm_cadastros.parquet").is_file())
        self.assertTrue((pmd.RESULTS_DIR / "silver_cvm_entidades.parquet").is_file())
        self.assertTrue((pmd.RESULTS_DIR / "gold_dim_empresa_golden_record.parquet").is_file())


if __name__ == "__main__":
    unittest.main()
