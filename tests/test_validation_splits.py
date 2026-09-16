"""Pruebas pequeñas para los folds compartidos del laboratorio."""

import unittest
from pathlib import Path
import sys

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import split_for_validation  # noqa: E402


class ValidationSplitTests(unittest.TestCase):
    def test_same_seed_reconstructs_the_same_folds(self) -> None:
        y = np.repeat([0, 1, 2], 10)

        first_splits = split_for_validation(y, n_splits=5, random_state=42)
        repeated_splits = split_for_validation(y, n_splits=5, random_state=42)

        for first, repeated in zip(first_splits, repeated_splits):
            np.testing.assert_array_equal(first[0], repeated[0])
            np.testing.assert_array_equal(first[1], repeated[1])


if __name__ == "__main__":
    unittest.main()
