"""Pruebas pequeñas para los folds compartidos del laboratorio."""

import unittest
from pathlib import Path
import sys

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import (  # noqa: E402
    labels_for_inner_stratification,
    split_for_validation,
)


class ValidationSplitTests(unittest.TestCase):
    def test_same_seed_reconstructs_the_same_folds(self) -> None:
        y = np.repeat([0, 1, 2], 10)

        first_splits = split_for_validation(y, n_splits=5, random_state=42)
        repeated_splits = split_for_validation(y, n_splits=5, random_state=42)

        for first, repeated in zip(first_splits, repeated_splits):
            np.testing.assert_array_equal(first[0], repeated[0])
            np.testing.assert_array_equal(first[1], repeated[1])

    def test_outer_train_and_test_are_disjoint(self) -> None:
        y = np.repeat([0, 1, 2], 10)

        outer_splits = split_for_validation(y, n_splits=5, random_state=42)

        for outer_train_idx, outer_test_idx in outer_splits:
            overlap = np.intersect1d(outer_train_idx, outer_test_idx)
            self.assertEqual(len(overlap), 0)

    def test_outer_test_is_absent_from_inner_folds(self) -> None:
        y = np.repeat([0, 1, 2], 10)

        outer_splits = split_for_validation(y, n_splits=5, random_state=42)

        for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(
            outer_splits, start=1
        ):
            inner_splits = split_for_validation(
                y[outer_train_idx], n_splits=3, random_state=42 + outer_fold
            )
            for inner_train_idx, inner_val_idx in inner_splits:
                inner_global_idx = outer_train_idx[
                    np.concatenate([inner_train_idx, inner_val_idx])
                ]
                overlap = np.intersect1d(inner_global_idx, outer_test_idx)
                self.assertEqual(len(overlap), 0)

    def test_gds_groups_rare_classes_only_for_inner_stratification(self) -> None:
        y = np.repeat(np.arange(7), [4, 4, 4, 4, 4, 4, 2])
        class_to_idx = {original: index for index, original in enumerate(range(1, 8))}
        outer_splits = split_for_validation(y, n_splits=2, random_state=42)

        for outer_fold, (outer_train_idx, _) in enumerate(outer_splits, start=1):
            y_outer_train = y[outer_train_idx]
            original_labels = y_outer_train.copy()
            stratification_labels = labels_for_inner_stratification(
                y_outer_train,
                target_name="GDS",
                class_to_idx=class_to_idx,
            )

            split_for_validation(
                stratification_labels,
                n_splits=2,
                random_state=42 + outer_fold,
            )
            np.testing.assert_array_equal(y_outer_train, original_labels)
            self.assertIn(class_to_idx[7], y_outer_train)
            self.assertNotIn(class_to_idx[7], stratification_labels)


if __name__ == "__main__":
    unittest.main()
