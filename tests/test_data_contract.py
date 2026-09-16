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
from data_loader import build_input_matrix, load_dataframe  # noqa: E402
from preprocessing import encode_target_as_indices, prepare_experiment_data  # noqa: E402


DATASET_PATH = Path(__file__).resolve().parents[1] / "dataset" / "15 atributos R0-R5.sav"


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

    @unittest.skipUnless(DATASET_PATH.exists(), "El dataset real no esta disponible.")
    def test_real_dataset_class_counts(self) -> None:
        dataframe = load_dataframe(DATASET_PATH)
        expected_counts = {
            "GDS": {1: 149, 2: 500, 3: 298, 4: 108, 5: 42, 6: 20, 7: 2},
            "GDS_R1": {1: 947, 2: 150, 3: 22},
            "GDS_R2": {1: 649, 2: 298, 3: 172},
            "GDS_R3": {1: 947, 3: 172},
            "GDS_R4": {1: 149, 2: 906, 3: 64},
            "GDS_R5": {1: 149, 2: 798, 3: 172},
        }

        for target_name, expected in expected_counts.items():
            with self.subTest(target=target_name):
                y, classes, _ = encode_target_as_indices(dataframe, target_name)
                actual = {
                    class_value: int((y == class_index).sum())
                    for class_index, class_value in enumerate(classes)
                }
                self.assertEqual(actual, expected)
                self.assertEqual(sum(actual.values()), len(dataframe))


if __name__ == "__main__":
    unittest.main()
