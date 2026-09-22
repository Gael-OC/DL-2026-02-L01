"""Prueba minima del flujo compartido de H3."""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import main
from src.losses import effective_number_weights
from src.methods import METHODS
from src.reporting import save_fold_details


class H3IntegrationTests(unittest.TestCase):
    def test_four_methods_share_outer_folds_and_export_complete_oof(self) -> None:
        torch.set_num_threads(1)
        X = np.arange(12 * 15, dtype=np.float32).reshape(12, 15) / 100
        y = np.tile(np.arange(3), 4)
        artifacts = {
            "X": X, "y": y, "classes": [1, 2, 3],
            "class_to_idx": {1: 0, 2: 1, 3: 2},
            "X_shape": X.shape, "y_shape": y.shape,
        }
        with patch("main.build_project_objects", return_value=artifacts), patch(
            "main.effective_number_weights", wraps=effective_number_weights
        ) as weights_mock:
            results = [
                main.train_one_experiment(
                    "unused.sav", method=method, outer_folds=2,
                    inner_folds=2, epochs=1, batch_size=6,
                )
                for method in METHODS
            ]

        outer_splits = main.split_for_validation(y, n_splits=2, random_state=42)
        self.assertEqual(len(weights_mock.call_args_list), 2)
        for call, (train_indices, _) in zip(weights_mock.call_args_list, outer_splits):
            np.testing.assert_array_equal(call.args[0], y[train_indices])

        with tempfile.TemporaryDirectory() as directory:
            paths = save_fold_details(results, directory)
            with paths["configuraciones"].open(newline="") as handle:
                configs = list(csv.DictReader(handle))
            with paths["predicciones_oof"].open(newline="") as handle:
                oof = list(csv.DictReader(handle))

        self.assertEqual(len(configs), 4 * 2)
        self.assertTrue(all(row["class_labels"] == "[1, 2, 3]" for row in configs))
        self.assertEqual(len(oof), 4 * len(y))
        for result in results:
            rows = [row for row in oof if row["algorithm"] == result["algorithm"]]
            self.assertEqual(sorted(int(row["original_index"]) for row in rows), list(range(len(y))))
            self.assertEqual(
                [fold["test_indices"].tolist() for fold in result["outer_folds"]],
                [test_indices.tolist() for _, test_indices in outer_splits],
            )
            for row in rows:
                probabilities = np.array([float(row[f"prob_{index}"]) for index in range(3)])
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertAlmostEqual(probabilities.sum(), 1.0, places=5)
                self.assertEqual(int(row["y_true"]), y[int(row["original_index"])])


if __name__ == "__main__":
    unittest.main()
