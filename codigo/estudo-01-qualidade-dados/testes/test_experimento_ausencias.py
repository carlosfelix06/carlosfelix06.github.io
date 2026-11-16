"""Validacoes pequenas das regras que sustentam a comparacao pareada."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experimento_ausencias import CONCENTRATED, mask_predictors, metrics, overlap_mask


class MascarasTest(unittest.TestCase):
    def test_contagem_grupo_e_determinismo(self):
        columns = list(CONCENTRATED) + [f"extra_{i}" for i in range(5)]
        frame = pd.DataFrame(1.0, index=range(20), columns=columns)
        original = frame.copy()
        mcar, n_mcar = mask_predictors(frame, columns, "mcar", 0.5, 2026, 0)
        concentrated, n_concentrated = mask_predictors(frame, columns, "concentrada", 0.5, 2026, 0)
        self.assertEqual(n_mcar, n_concentrated)
        self.assertEqual(n_mcar, 230)
        self.assertEqual(int(mcar.isna().sum().sum()), 230)
        self.assertEqual(int(concentrated.isna().sum().sum()), 230)
        self.assertFalse(concentrated[[f"extra_{i}" for i in range(5)]].isna().any().any())
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_frame_equal(concentrated, mask_predictors(frame, columns, "concentrada", 0.5, 2026, 0)[0])

    def test_captura_desempata_por_id(self):
        y = np.array([1] + [0] * 9)
        p = np.full(10, 0.5)
        ids = np.array([1] + list(range(2, 11)))
        result, curve = metrics(y, p, ids)
        self.assertEqual(result["captura_top_10"], 1.0)
        self.assertEqual(sum(point["n"] for point in curve), 10)

    def test_sobreposicao_considera_apenas_preditores(self):
        parts = {
            "treino": pd.DataFrame({"ID": [1], "x": [7], "y": [0]}),
            "validacao": pd.DataFrame({"ID": [2], "x": [8], "y": [1]}),
            "teste": pd.DataFrame({"ID": [3, 4], "x": [7, 9], "y": [1, 0]}),
        }
        self.assertEqual(overlap_mask(parts, ["x"]).tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
