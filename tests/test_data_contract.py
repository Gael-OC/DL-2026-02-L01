"""Pruebas pequeñas para el contrato de datos del laboratorio."""

import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import FEATURE_COLUMNS  # noqa: E402
from data_loader import build_input_matrix  # noqa: E402
from preprocessing import prepare_experiment_data  # noqa: E402


def make_valid_dataframe() -> pd.DataFrame:
    data = {column: np.zeros(3) for column in FEATURE_COLUMNS}
    data["GDS_R2"] = [1, 2, 3]
    return pd.DataFrame(data)


class DataContractTests(unittest.TestCase):
    def test_valid_data_is_prepared(self) -> None:
        dataframe = make_valid_dataframe()

        X, y, classes, class_to_idx = prepare_experiment_data(dataframe, "GDS_R2")

        self.assertEqual(X.shape, (3, 15))
        self.assertEqual(X.dtype, np.float32)
        np.testing.assert_array_equal(y, [0, 1, 2])
        self.assertEqual(classes, [1, 2, 3])
        self.assertEqual(class_to_idx, {1: 0, 2: 1, 3: 2})

    def test_non_finite_feature_is_rejected(self) -> None:
        dataframe = make_valid_dataframe()
        dataframe.loc[0, FEATURE_COLUMNS[0]] = np.nan

        with self.assertRaisesRegex(ValueError, "NaN o valores infinitos"):
            build_input_matrix(dataframe)

    def test_decimal_target_is_rejected(self) -> None:
        dataframe = make_valid_dataframe()
        dataframe["GDS_R2"] = dataframe["GDS_R2"].astype(float)
        dataframe.loc[0, "GDS_R2"] = 1.5

        with self.assertRaisesRegex(ValueError, "valores enteros"):
            prepare_experiment_data(dataframe, "GDS_R2")


if __name__ == "__main__":
    unittest.main()
