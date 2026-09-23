"""Valida los artefactos de E1 sin entrenar modelos."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main  # noqa: E402
from src.config import HYPERPARAMETER_GRID, TARGET_COLUMNS  # noqa: E402
from src.data_loader import load_dataframe  # noqa: E402
from src.evaluation import METRIC_KEYS, compute_all_metrics, compute_confusion_matrix  # noqa: E402
from src.preprocessing import (  # noqa: E402
    labels_for_inner_stratification, prepare_experiment_data, split_for_validation,
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual: float, expected: float, message: str) -> None:
    check(np.isclose(actual, expected, rtol=0, atol=1e-6), message)


def validate(run: Path, data_path: Path, smoke: bool = False) -> None:
    targets = ["GDS_R3"] if smoke else TARGET_COLUMNS
    expected_rows, expected_folds, expected_oof, expected_fits = (
        (1, 2, 1119, 18) if smoke else (6, 27, 6714, 343)
    )
    status = json.loads((run / "estado.json").read_text())
    manifest = json.loads((run / "manifiesto.json").read_text())
    check(status["status"] == "complete" and not status["failures"], "Estado raíz incompleto")
    check(status["completed_experiments"] == expected_rows, "Cantidad de objetivos")
    check(manifest["method_versions"] == {"corn": "CORN v2 (pares elegibles)"}, "Versión incorrecta")
    check(manifest["data"]["sha256"] == hashlib.sha256(data_path.read_bytes()).hexdigest(), "SAV distinto")
    check(len(manifest["code"]["files_sha256"]) >= 2 and manifest["code"]["combined_sha256"], "Código sin hash")
    check(manifest["command"] and manifest["environment"]["packages"], "Comando o entorno ausente")

    summary = pd.read_csv(run / "resultados.csv")
    configs = pd.read_csv(run / "configuraciones_folds.csv")
    oof = pd.read_csv(run / "predicciones_oof.csv")
    coverage = pd.read_csv(run / "verificacion_oof.csv")
    check(len(summary) == expected_rows and len(configs) == expected_folds and len(oof) == expected_oof, "Cobertura global incorrecta")
    check(len(coverage) == expected_rows and set(summary.target) == set(targets), "Faltan objetivos")
    check(set(summary.algorithm) == {"CORN v2 (pares elegibles)"}, "Fila de otro método")
    check(all(column in summary for column in ("base_hidden_dim", "base_dropout", "base_learning_rate", "base_weight_decay")), "HP base ambiguos")

    dataframe = load_dataframe(data_path)
    fit_count = 0
    for target in targets:
        _, y, classes, class_to_idx = prepare_experiment_data(dataframe, target)
        k = len(classes)
        n_outer, n_inner = (2, 2) if smoke or target == "GDS" else (5, 3)
        outer_splits = split_for_validation(y, n_outer, random_state=42)
        target_oof = oof[oof.target == target]
        check(len(target_oof) == len(y), f"{target}: cantidad OOF")
        check(sorted(target_oof.original_index.tolist()) == list(range(len(y))), f"{target}: índices OOF")
        check(np.array_equal(target_oof.sort_values("original_index").y_true.to_numpy(), y), f"{target}: etiquetas SAV")
        check((run / "corn" / target / "estado.json").read_text().find('"complete"') >= 0, f"{target}: estado")
        check(len(configs[configs.target == target]) == n_outer, f"{target}: configuraciones")

        fold_metrics = {key: [] for key in METRIC_KEYS}
        for fold_number, (train_idx, test_idx) in enumerate(outer_splits, start=1):
            name = f"{target}/fold_{fold_number:02d}"
            folder = run / "corn" / target / f"fold_{fold_number:02d}"
            details = json.loads((folder / "metricas.json").read_text())
            fold_oof = target_oof[target_oof.outer_fold == fold_number]
            check(set(fold_oof.original_index) == set(test_idx), f"{name}: fold externo")
            check(len(fold_oof) == details["n_test"], f"{name}: n_test")
            check(details["class_labels"] == classes, f"{name}: clases")
            config_row = configs[(configs.target == target) & (configs.outer_fold == fold_number)].iloc[0]
            check(json.loads(config_row.best_config) == details["best_config"], f"{name}: config CSV")
            check(json.loads(config_row.class_labels) == classes, f"{name}: clases CSV")

            strat_y = labels_for_inner_stratification(y[train_idx], target, class_to_idx)
            for inner_train, inner_val in split_for_validation(strat_y, n_inner, 42 + fold_number):
                check(not set(train_idx[inner_train]) & set(test_idx), f"{name}: fuga inner train")
                check(not set(train_idx[inner_val]) & set(test_idx), f"{name}: fuga inner val")
                check(not set(inner_train) & set(inner_val), f"{name}: solape interno")

            grid = details["inner_grid_results"]
            check(len(grid) == 4, f"{name}: grid incompleto")
            check([item["config"] for item in grid] == HYPERPARAMETER_GRID, f"{name}: candidatos")
            for item in grid:
                check(len(item["mae_scores"]) == len(item["qwk_scores"]) == n_inner, f"{name}: scores")
                close(item["mae_mean"], np.mean(item["mae_scores"]), f"{name}: MAE interno")
                close(item["qwk_mean"], np.mean(item["qwk_scores"]), f"{name}: QWK interno")
            best = min(grid, key=lambda item: (item["mae_mean"], -item["qwk_mean"]))
            check(best["config"] == details["best_config"], f"{name}: ganador")
            close(best["mae_mean"], details["inner_mae_mean"], f"{name}: MAE ganador")
            close(best["qwk_mean"], details["inner_qwk_mean"], f"{name}: QWK ganador")
            fit_count += 4 * n_inner + 1

            probs = fold_oof[[f"prob_{i}" for i in range(k)]].to_numpy(dtype=float)
            check(np.isfinite(probs).all() and (probs >= -1e-7).all() and (probs <= 1 + 1e-7).all(), f"{name}: probabilidades")
            check(np.allclose(probs.sum(axis=1), 1, rtol=0, atol=1e-6), f"{name}: suma probabilidades")
            true = fold_oof.y_true.to_numpy(dtype=int)
            pred = fold_oof.y_pred.to_numpy(dtype=int)
            check(((0 <= pred) & (pred < k)).all(), f"{name}: clase predicha")
            check(np.array_equal(pred, (np.cumsum(probs[:, ::-1], axis=1)[:, ::-1][:, 1:] > 0.5).sum(axis=1)), f"{name}: decisión CORN")
            metrics = compute_all_metrics(true, pred, k)
            for key in METRIC_KEYS:
                close(metrics[key], details["outer_metrics"][key], f"{name}: {key}")
                fold_metrics[key].append(metrics[key])
            matrix = compute_confusion_matrix(true, pred, k)
            check(np.array_equal(pd.read_csv(folder / "confusion.csv", index_col=0).to_numpy(), matrix), f"{name}: matriz")

        row = summary[summary.target == target].iloc[0]
        for key in METRIC_KEYS:
            close(row[key], np.mean(fold_metrics[key]), f"{target}: media {key}")
            close(row[f"{key}_std"], np.std(fold_metrics[key]), f"{target}: std {key}")
        all_matrix = compute_confusion_matrix(target_oof.y_true.to_numpy(), target_oof.y_pred.to_numpy(), k)
        check(np.array_equal(pd.read_csv(run / "corn" / target / "confusion_oof.csv", index_col=0).to_numpy(), all_matrix), f"{target}: matriz OOF")
        check(coverage[coverage.target == target].iloc[0]["n_unique"] == len(y), f"{target}: cobertura CSV")

    check(fit_count == expected_fits, "Presupuesto de ajustes incorrecto")
    print(f"Corrida válida: {expected_rows} objetivos, {expected_folds} folds, {expected_oof} OOF y {expected_fits} ajustes previstos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--data-path", type=Path, default=ROOT / "dataset/15 atributos R0-R5.sav")
    parser.add_argument("--smoke", action="store_true", help="Comprueba el smoke GDS_R3 2×2 de una época")
    args = parser.parse_args()
    validate(args.run, args.data_path, smoke=args.smoke)
