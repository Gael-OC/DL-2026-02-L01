"""Comprobaciones minimas de los artefactos y fallos de H4."""

import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import main
from src.reporting import save_outer_fold, save_target_confusion


class H4TraceabilityTests(unittest.TestCase):
    def test_fold_and_target_confusion_are_saved(self) -> None:
        fold = {
            "outer_fold": 1,
            "test_indices": np.array([0, 2]),
            "y_true": np.array([0, 1]),
            "y_pred": np.array([0, 2]),
            "y_proba": np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]]),
            "best_config": {"hidden_dim": 32},
            "inner_grid_results": [{
                "config": {"hidden_dim": 32}, "mae_scores": [0.2, 0.4],
                "qwk_scores": [0.7, 0.5], "mae_mean": 0.3, "qwk_mean": 0.6,
            }],
            "inner_mae_mean": float("nan"),
            "inner_qwk_mean": float("nan"),
            "outer_metrics": {"accuracy": 0.5},
            "final_train_loss": 0.4,
        }
        with tempfile.TemporaryDirectory() as directory:
            target_path = Path(directory)
            save_outer_fold(fold, [1, 2, 3], target_path / "fold_01")
            save_target_confusion({"outer_folds": [fold], "classes": [1, 2, 3]}, target_path)
            with (target_path / "fold_01" / "predicciones_oof.csv").open(newline="") as handle:
                predictions = list(csv.DictReader(handle))
            with (target_path / "confusion_oof.csv").open(newline="") as handle:
                matrix = list(csv.DictReader(handle))
            metrics = json.loads((target_path / "fold_01" / "metricas.json").read_text())
            self.assertEqual([int(row["original_index"]) for row in predictions], [0, 2])
            self.assertEqual(sum(int(row[str(label)]) for row in matrix for label in [1, 2, 3]), 2)
            self.assertIsNone(metrics["inner_mae_mean"])
            self.assertEqual(metrics["inner_grid_results"], fold["inner_grid_results"])

    def test_execution_error_is_visible_and_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "run"
            argv = [
                "main.py", "--data-path", "unused.sav", "--methods", "softmax_fixed",
                "--output-dir", str(output_path),
            ]
            with patch.object(sys, "argv", argv), patch(
                "main.train_one_experiment", side_effect=RuntimeError("fallo controlado")
            ), patch.object(sys, "stderr", io.StringIO()):
                with self.assertRaises(SystemExit) as exit_result:
                    main.main()
            self.assertEqual(exit_result.exception.code, 1)
            status = json.loads((output_path / "softmax_fixed" / "GDS_R2" / "estado.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertIn("fallo controlado", status["traceback"])
            self.assertEqual(json.loads((output_path / "estado.json").read_text())["status"], "failed")
            self.assertFalse((output_path / "resultados.csv").exists())


if __name__ == "__main__":
    unittest.main()
