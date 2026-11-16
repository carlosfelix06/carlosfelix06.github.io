import importlib.util
import unittest
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = STUDY_ROOT / "src" / "backtest_projecao_walkforward.py"


def load_module():
    spec = importlib.util.spec_from_file_location("backtest_projecao_walkforward", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestBacktestProjecaoWalkforward(unittest.TestCase):
    def test_01_br_banking_holidays_includes_fixed_and_movable(self):
        mod = load_module()
        hols_2025 = mod.br_banking_holidays(2025)
        self.assertIn("2025-01-01", hols_2025)
        self.assertIn("2025-03-03", hols_2025)  # Segunda de Carnaval 2025
        self.assertIn("2025-03-04", hols_2025)  # Terca de Carnaval 2025
        self.assertIn("2025-04-18", hols_2025)  # Paixao de Cristo 2025
        self.assertIn("2025-11-20", hols_2025)  # Consciencia Negra

    def test_02_walk_forward_benchmark_with_holidays(self):
        mod = load_module()
        latent, realized = mod.generate_series()
        metrics, rows = mod.calculate_metrics(latent, realized)
        bench = metrics["benchmark"]
        self.assertEqual(bench["origins"], 33)
        ridge = next(m for m in bench["models"] if m["key"] == "ridge_cal")
        naive = next(m for m in bench["models"] if m["key"] == "naive_t1")
        self.assertAlmostEqual(ridge["weighted_product_wmape_pct"], 2.79, delta=0.05)
        self.assertAlmostEqual(ridge["bottomup_mae_brl_bi"], 6.24, delta=0.05)
        self.assertAlmostEqual(naive["bottomup_mae_brl_bi"], 13.63, delta=0.05)
        self.assertAlmostEqual(bench["budget_saving_brl_bi_month"], 7.39, delta=0.05)
        self.assertGreaterEqual(ridge["bottomup_within_5_count"], 32)
        self.assertGreater(ridge["bottomup_ljung_box_p"], 0.05)


if __name__ == "__main__":
    unittest.main()
