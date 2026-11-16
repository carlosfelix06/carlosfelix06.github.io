from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

STUDY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = STUDY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import funil_politica_credito as fpc  # noqa: E402


class TestFunilPoliticaCredito(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df, cls.meta = fpc.load_or_download_dataset()
        cls.x_lp, cls.x_cp, cls.y = fpc.build_feature_matrices(cls.df)
        cls.splits = fpc.split_train_val_test(cls.y, random_state=42)
        cls.bundle = fpc.fit_and_calibrate_models(
            cls.df, cls.x_lp, cls.x_cp, cls.y, cls.splits, random_state=42
        )
        cls.df_test = cls.df.iloc[cls.splits.test].reset_index(drop=True)
        cls.y_test = cls.y[cls.splits.test]
        cls.scores_test = fpc.score_population(
            cls.x_lp.iloc[cls.splits.test].reset_index(drop=True),
            cls.x_cp.iloc[cls.splits.test].reset_index(drop=True),
            cls.bundle,
        )
        cls.sim = fpc.simulate_policies_and_lgd_sensitivity(
            cls.df_test, cls.y_test, cls.scores_test
        )

    def test_01_sem_vazamento_entre_splits_e_eixos(self) -> None:
        tr = set(self.splits.train.tolist())
        va = set(self.splits.val.tolist())
        te = set(self.splits.test.tolist())
        self.assertTrue(tr.isdisjoint(va))
        self.assertTrue(tr.isdisjoint(te))
        self.assertTrue(va.isdisjoint(te))
        self.assertEqual(len(tr) + len(va) + len(te), len(self.df))

        # Estratificação preservada
        self.assertAlmostEqual(float(np.mean(self.y[self.splits.train])), float(np.mean(self.y_test)), places=3)
        self.assertAlmostEqual(float(np.mean(self.y[self.splits.val])), float(np.mean(self.y_test)), places=3)

        # Features estruturais (LP) e pressão imediata (CP) estritamente disjuntas
        self.assertTrue(set(fpc.FEATURES_LP).isdisjoint(fpc.FEATURES_CP))

    def test_02_conservacao_exata_de_massa_funil_e_substeps(self) -> None:
        # Teste fora da amostra nas 3 políticas
        for pol_key, pol_data in self.sim["comparativo_base"].items():
            cons = pol_data["conservacao"]
            self.assertTrue(cons["conservacao_exata"], msg=f"Falha conservação em {pol_key}")
            self.assertEqual(cons["total_entrada"], len(self.df_test))
            self.assertEqual(
                cons["ko_r1"] + cons["ko_r2"] + cons["ko_r3"] + cons["cluster_c"] + cons["cluster_b"] + cons["cluster_a"],
                cons["total_entrada"],
            )
            self.assertEqual(
                cons["cluster_c"] + cons["cluster_b"] + cons["cluster_a"],
                cons["r4_elegiveis"],
            )

        # Conservação também na base completa (150.000 linhas)
        funnel_full = fpc.evaluate_funnel_layers(self.df, "otimizada")
        scores_full = fpc.score_population(self.x_lp, self.x_cp, self.bundle)
        clusters_full = fpc.apply_matrix_3x3_clusters(
            funnel_full["r4_eligible"], scores_full["lp_band_idx"], scores_full["cp_band_idx"]
        )
        cons_full = fpc.verify_mass_conservation(len(self.df), funnel_full, clusters=clusters_full)
        self.assertEqual(cons_full["total_reconstruido"], 150000)

    def test_03_calibracao_e_ganho_2d_sobre_lp_e_rf(self) -> None:
        pd_cp = self.scores_test["pd_cp"]
        self.assertTrue(np.all((pd_cp >= 0.0) & (pd_cp <= 1.0)))
        self.assertIn(self.bundle.cal_method, {"bruto_nativo", "isotonic", "sigmoid"})

        # Brier da calibração escolhida na validação <= Brier das demais candidatas
        val_m = self.bundle.val_calibration_metrics
        key_map = {
            "bruto_nativo": "rf_bruto",
            "isotonic": "calibracao_isotonica",
            "sigmoid": "calibracao_sigmoid",
        }
        chosen_brier = val_m[key_map[self.bundle.cal_method]]["brier"]
        for cand in val_m.values():
            self.assertLessEqual(round(chosen_brier, 7), round(cand["brier"], 7))

        # Baixa colinearidade e ganho 2D estatisticamente significativo vs LP E vs RF bruto isolado
        eval_2d = fpc.evaluate_complementarity_and_2d_gain(
            self.df_test,
            self.y_test,
            self.scores_test,
            self.sim["funnel_otimizada_test"]["r4_eligible"],
            bundle=self.bundle,
            random_state=42,
        )
        corr = eval_2d["correlacao_entre_eixos"]
        self.assertLess(abs(corr["spearman_r4_elegiveis"]), 0.50)
        self.assertLess(corr["vif_r4"], 2.0)
        self.assertGreater(eval_2d["ganho_2d_sobre_lp_r4_bootstrap"]["ic95_inferior"], 0.05)
        self.assertGreater(eval_2d["ganho_2d_sobre_rf_bruto_r4_bootstrap"]["ic95_inferior"], 0.0)
        self.assertLess(eval_2d["ganho_2d_sobre_rf_bruto_r4_bootstrap"]["p_val_unilateral"], 0.05)

        # Curvas ROC/PR (101 pts) e curvas de calibração reais (10 decis somando exatamente R4)
        for mkey in ("score_lp_isolado", "rf_cp_bruto", "rf_cp_isotonico", "combinacao_2d"):
            self.assertEqual(len(eval_2d["curvas_roc_r4"][mkey]["fpr"]), 101)
            self.assertEqual(len(eval_2d["curvas_pr_r4"][mkey]["recall"]), 101)
            cal_bins = eval_2d["curvas_calibracao_r4"][mkey]
            self.assertEqual(len(cal_bins), 10)
            self.assertEqual(sum(b["n"] for b in cal_bins), int(np.sum(self.sim["funnel_otimizada_test"]["r4_eligible"])))

    def test_04_matriz_3x3_early_warning_e_monotonicidade(self) -> None:
        # Regra determinística da matriz: Alto LP (2) + Risco CP Alto (2) -> Cluster C (nunca A!)
        self.assertEqual(fpc.MATRIX_3X3_CLUSTER_MAP[(2, 2)], "C")
        self.assertEqual(fpc.MATRIX_3X3_CLUSTER_MAP[(2, 0)], "A")

        # Teste sintético pontual apenas das combinações 3x3 (9 células unitárias)
        r4_elig = np.ones(9, dtype=bool)
        lp_idx = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        cp_idx = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        res = fpc.apply_matrix_3x3_clusters(r4_elig, lp_idx, cp_idx)
        # Índice 8 = (lp=2, cp=2) = Alto Score LP + Risco CP Alto
        self.assertEqual(res["cluster"][8], "C")
        self.assertTrue(res["early_warning"][8])
        self.assertFalse(res["cluster_a"][8])
        # Índice 6 = (lp=2, cp=0) = Alto Score LP + Risco CP Baixo
        self.assertEqual(res["cluster"][6], "A")

        # Verificação empírica real nas 9 células fora da amostra
        cells = fpc.build_matrix_3x3_summary(
            self.y_test,
            self.sim["funnel_otimizada_test"]["r4_eligible"],
            self.scores_test["lp_band_idx"],
            self.scores_test["cp_band_idx"],
        )
        self.assertEqual(len(cells), 9)
        lookup = {(c["lp_idx"], c["cp_idx"]): c for c in cells}
        for c in cells:
            self.assertGreater(c["volume"], 500)
            self.assertIn("breakeven_margem_65_ead", c)
            self.assertGreater(c["breakeven_margem_65_ead"], 0.0)

        # Monotonicidade de risco real dentro de cada linha LP e comparação Early Warning vs Cluster A
        self.assertGreater(lookup[(2, 2)]["bad_rate"], lookup[(2, 0)]["bad_rate"] * 5.0)
        for lp_i in range(3):
            self.assertLess(lookup[(lp_i, 0)]["bad_rate"], lookup[(lp_i, 1)]["bad_rate"])
            self.assertLess(lookup[(lp_i, 1)]["bad_rate"], lookup[(lp_i, 2)]["bad_rate"])

    def test_05_monotonicidade_rigida_vs_flexivel_e_lgd(self) -> None:
        fun_rig = fpc.evaluate_funnel_layers(self.df_test, "rigida")
        fun_opt = fpc.evaluate_funnel_layers(self.df_test, "otimizada")

        # R2 rígido bloqueia estritamente mais ou igual que R2 por gravidade
        self.assertGreater(int(np.sum(fun_rig["ko_r2"])), int(np.sum(fun_opt["ko_r2"])))
        self.assertGreater(int(np.sum(fun_rig["ko_r2"] | fun_rig["ko_r3"])), int(np.sum(fun_opt["ko_r2"] | fun_opt["ko_r3"])))

        # Aprovação: Conservadora Rígida <= Otimizada 2D <= Agressiva 1D
        comp = self.sim["comparativo_base"]
        self.assertLess(comp["conservadora_rigida"]["taxa_aprovacao"], comp["otimizada_2d"]["taxa_aprovacao"])
        self.assertLess(comp["otimizada_2d"]["taxa_aprovacao"], comp["agressiva_1d"]["taxa_aprovacao"])

        # P&L cai monotonicamente com aumento de LGD, IC 95% envolve o ponto e Otimizada 2D supera Rígida e Agressiva 1D
        sens = self.sim["sensibilidade_lgd"]
        for i in range(len(sens) - 1):
            self.assertLess(sens[i]["lgd"], sens[i + 1]["lgd"])
            for k in ("conservadora_rigida", "agressiva_1d", "rf_cp_isolado_1d_mesmo_ead", "rf_cp_isolado_pareado_ab", "otimizada_2d"):
                self.assertGreater(sens[i][k], sens[i + 1][k])
        for row in sens:
            self.assertGreater(row["otimizada_2d"], row["conservadora_rigida"])
            self.assertGreater(row["otimizada_2d"], row["agressiva_1d"])
            for k in ("conservadora_rigida", "agressiva_1d", "rf_cp_isolado_1d_mesmo_ead", "otimizada_2d"):
                self.assertLessEqual(row[f"{k}_ic95_inf"], row[k])
                self.assertLessEqual(row[k], row[f"{k}_ic95_sup"])

        # Igualdade estrita de capital (117,32M EAD) e atribuição fatorial aditiva exata
        c117 = self.sim["comparativo_igualdade_capital_117m"]
        self.assertEqual(c117["conservadora_rigida"]["ead_total_um"], c117["rf_cp_isolado_1d"]["ead_total_um"])
        self.assertEqual(c117["rf_cp_isolado_1d"]["ead_total_um"], c117["score_2d_continuo"]["ead_total_um"])
        self.assertGreater(c117["score_2d_continuo"]["pnl_total_um"], c117["rf_cp_isolado_1d"]["pnl_total_um"])
        self.assertGreater(c117["rf_cp_isolado_1d"]["pnl_total_um"], c117["conservadora_rigida"]["pnl_total_um"])

        atr = self.sim["atribuicao_fatorial_pnl"]
        soma_passos = (
            atr["baseline_conservadora_rigida_um"]
            + atr["efeito_regras_r2_r3_sem_filtro_cp_um"]
            + atr["efeito_selecao_2d_e_contraproposta_margem_uniforme_55_um"]
            + atr["efeito_precificacao_cluster_b_55_para_70_um"]
        )
        self.assertAlmostEqual(soma_passos, atr["total_otimizada_2d_um"], places=4)
        self.assertEqual(len(self.sim["sensibilidade_premissas_comerciais"]), 9)


if __name__ == "__main__":
    unittest.main()
