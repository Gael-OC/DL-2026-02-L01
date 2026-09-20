"""Pruebas pequenas para la busqueda interna de hiperparametros."""

import unittest
from unittest.mock import patch

import numpy as np

import main
from src.config import HYPERPARAMETER_GRID
from src.evaluation import compute_all_metrics


def fake_training_cycle(**kwargs) -> dict:
    y_true = kwargs["y_eval"]
    y_pred = np.zeros_like(y_true)
    metrics = compute_all_metrics(y_true, y_pred, num_classes=3)
    metrics["mae_ordinal"] = float(kwargs["seed"] % 10)
    metrics["qwk"] = float(kwargs["seed"] % 10) / 10
    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "final_train_loss": 0.0,
    }


class HyperparameterSearchTests(unittest.TestCase):
    def test_selection_uses_lower_mae_then_higher_qwk(self) -> None:
        candidates = [
            {"name": "higher-mae", "mae_mean": 0.6, "qwk_mean": 0.9},
            {"name": "lower-qwk", "mae_mean": 0.5, "qwk_mean": 0.2},
            {"name": "winner", "mae_mean": 0.5, "qwk_mean": 0.4},
        ]

        selected = main.select_best_inner_result(candidates)

        self.assertEqual(selected["name"], "winner")

    @patch("main.run_training_cycle", side_effect=fake_training_cycle)
    @patch("main.build_project_objects")
    def test_every_grid_configuration_runs_on_every_inner_fold(
        self,
        build_project_objects_mock,
        _run_training_cycle_mock,
    ) -> None:
        X = np.zeros((12, 15), dtype=np.float32)
        y = np.tile(np.arange(3), 4)
        build_project_objects_mock.return_value = {
            "X": X,
            "y": y,
            "classes": [1, 2, 3],
            "class_to_idx": {1: 0, 2: 1, 3: 2},
            "X_shape": X.shape,
            "y_shape": y.shape,
        }

        results = main.train_one_experiment(
            data_path="unused.csv",
            target_name="GDS_R2",
            epochs=1,
            outer_folds=2,
            inner_folds=2,
        )

        for outer_result in results["outer_folds"]:
            grid_results = outer_result["inner_grid_results"]
            self.assertEqual(len(grid_results), len(HYPERPARAMETER_GRID))
            self.assertEqual(
                [result["config"] for result in grid_results],
                HYPERPARAMETER_GRID,
            )
            for candidate_result in grid_results:
                self.assertEqual(len(candidate_result["mae_scores"]), 2)
                self.assertEqual(len(candidate_result["qwk_scores"]), 2)
                self.assertAlmostEqual(candidate_result["mae_mean"], 3.5)
                self.assertAlmostEqual(candidate_result["qwk_mean"], 0.35)

    @patch("main.run_training_cycle")
    @patch("main.build_project_objects")
    def test_outer_training_uses_winner_and_only_outer_test_for_evaluation(
        self,
        build_project_objects_mock,
        run_training_cycle_mock,
    ) -> None:
        X = np.arange(12 * 15, dtype=np.float32).reshape(12, 15)
        y = np.tile(np.arange(3), 4)
        build_project_objects_mock.return_value = {
            "X": X,
            "y": y,
            "classes": [1, 2, 3],
            "class_to_idx": {1: 0, 2: 1, 3: 2},
            "X_shape": X.shape,
            "y_shape": y.shape,
        }
        winning_config = HYPERPARAMETER_GRID[1]

        def score_configuration(**kwargs) -> dict:
            y_true = kwargs["y_eval"]
            metrics = compute_all_metrics(y_true, y_true, num_classes=3)
            metrics["mae_ordinal"] = (
                0.0 if kwargs["hidden_dim"] == winning_config["hidden_dim"] else 1.0
            )
            return {
                "metrics": metrics,
                "y_true": y_true,
                "y_pred": y_true,
                "final_train_loss": 0.0,
            }

        run_training_cycle_mock.side_effect = score_configuration
        outer_splits = main.split_for_validation(y, n_splits=2, random_state=42)

        main.train_one_experiment(
            data_path="unused.csv",
            target_name="GDS_R2",
            epochs=1,
            outer_folds=2,
            inner_folds=2,
        )

        calls_per_outer_fold = len(HYPERPARAMETER_GRID) * 2 + 1
        final_calls = run_training_cycle_mock.call_args_list[
            calls_per_outer_fold - 1 :: calls_per_outer_fold
        ]
        self.assertEqual(len(final_calls), 2)
        for final_call, (outer_train_idx, outer_test_idx) in zip(
            final_calls, outer_splits
        ):
            kwargs = final_call.kwargs
            for parameter_name, expected_value in winning_config.items():
                self.assertEqual(kwargs[parameter_name], expected_value)
            np.testing.assert_array_equal(kwargs["X_train"], X[outer_train_idx])
            np.testing.assert_array_equal(kwargs["y_train"], y[outer_train_idx])
            np.testing.assert_array_equal(kwargs["X_eval"], X[outer_test_idx])
            np.testing.assert_array_equal(kwargs["y_eval"], y[outer_test_idx])


if __name__ == "__main__":
    unittest.main()
