"""Fit and evaluate the Baseline V1 DummyClassifier at prefix k=10.

This is Baseline V1 Part 5 only. It consumes the frozen Part 3 case order and
the frozen Part 4 target arrays, fits on training targets only, and evaluates
the resulting prior baseline on the frozen test set. No preprocessing is fit
or modified by this script.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "baseline_v1"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "baseline_k10"

PREPROCESSING_MANIFEST_PATH = RESULTS_DIR / "preprocessing_manifest.json"
SPLIT_SUMMARY_PATH = RESULTS_DIR / "split_summary.json"
TRAIN_IDS_PATH = RESULTS_DIR / "train_case_ids.csv"
TEST_IDS_PATH = RESULTS_DIR / "test_case_ids.csv"
Y_TRAIN_PATH = PROCESSED_DIR / "y_train.npy"
Y_TEST_PATH = PROCESSED_DIR / "y_test.npy"
METRICS_PATH = RESULTS_DIR / "dummy_metrics.json"
PREDICTIONS_PATH = RESULTS_DIR / "dummy_predictions.csv"

EXPECTED_TRAIN_CASES = 25_100
EXPECTED_TEST_CASES = 6_276
EXPECTED_TARGET_COUNTS = {
    "train": {0: 11_322, 1: 13_778},
    "test": {0: 2_826, 1: 3_450},
}
POSITIVE_CLASS = 1
CONFUSION_MATRIX_LABELS = [0, 1]
FLOAT_TOLERANCE = 1e-12


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load a required JSON object."""
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a required artifact."""
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _project_path(relative_path: str) -> Path:
    """Resolve a manifest path and ensure it stays inside the project."""
    candidate = (PROJECT_ROOT / Path(relative_path)).resolve()
    candidate.relative_to(PROJECT_ROOT.resolve())
    return candidate


def _load_case_ids(path: Path, expected_rows: int) -> list[str]:
    """Load one canonical case-ID file without changing row order."""
    if not path.is_file():
        raise FileNotFoundError(f"Canonical split file not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8", dtype={"case_id": "string"})
    if frame.columns.tolist() != ["case_id"]:
        raise ValueError(f"Expected only a case_id column in {path}")
    if len(frame) != expected_rows:
        raise AssertionError(f"Unexpected row count in {path}: {len(frame)}")
    if frame["case_id"].isna().any() or not frame["case_id"].is_unique:
        raise AssertionError(f"Case IDs must be non-null and unique in {path}")
    return frame["case_id"].astype(str).tolist()


def _target_counts(target: np.ndarray) -> dict[int, int]:
    """Return integer label counts for a one-dimensional target array."""
    labels, counts = np.unique(target, return_counts=True)
    return {
        int(label): int(count)
        for label, count in zip(labels, counts, strict=True)
    }


def _validate_frozen_inputs(
    preprocessing_manifest: dict[str, Any],
    split_summary: dict[str, Any],
    train_ids: list[str],
    test_ids: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Validate the frozen split and Part 4 target-array contract."""
    if y_train.ndim != 1 or len(y_train) != EXPECTED_TRAIN_CASES:
        raise AssertionError(f"Unexpected y_train shape: {y_train.shape}")
    if y_test.ndim != 1 or len(y_test) != EXPECTED_TEST_CASES:
        raise AssertionError(f"Unexpected y_test shape: {y_test.shape}")
    if _target_counts(y_train) != EXPECTED_TARGET_COUNTS["train"]:
        raise AssertionError("y_train distribution differs from frozen Part 3")
    if _target_counts(y_test) != EXPECTED_TARGET_COUNTS["test"]:
        raise AssertionError("y_test distribution differs from frozen Part 3")

    if set(np.unique(y_train).tolist()) != {0, 1}:
        raise AssertionError("y_train is not binary with labels {0, 1}")
    if set(np.unique(y_test).tolist()) != {0, 1}:
        raise AssertionError("y_test is not binary with labels {0, 1}")
    if not set(train_ids).isdisjoint(test_ids):
        raise AssertionError("Canonical train and test case IDs overlap")

    expected_manifest_distributions = {
        split: {str(label): count for label, count in counts.items()}
        for split, counts in EXPECTED_TARGET_COUNTS.items()
    }
    if preprocessing_manifest.get("split_sizes") != {
        "train": EXPECTED_TRAIN_CASES,
        "test": EXPECTED_TEST_CASES,
    }:
        raise AssertionError("Part 4 split sizes differ from the frozen split")
    if (
        preprocessing_manifest.get("target_distributions")
        != expected_manifest_distributions
    ):
        raise AssertionError("Part 4 target distributions differ from Part 3")
    if preprocessing_manifest.get("preprocessing_fit_scope") != "train_only":
        raise AssertionError("Part 4 preprocessing was not recorded as train-only")
    if preprocessing_manifest.get("classifier_trained") is not False:
        raise AssertionError("Part 4 manifest unexpectedly records a classifier")

    if split_summary.get("train_cases") != EXPECTED_TRAIN_CASES:
        raise AssertionError("Split summary has an unexpected train size")
    if split_summary.get("test_cases") != EXPECTED_TEST_CASES:
        raise AssertionError("Split summary has an unexpected test size")
    if split_summary.get("train_success") != EXPECTED_TARGET_COUNTS["train"][1]:
        raise AssertionError("Split summary has an unexpected train success count")
    if (
        split_summary.get("train_unsuccessful")
        != EXPECTED_TARGET_COUNTS["train"][0]
    ):
        raise AssertionError("Split summary has an unexpected train failure count")
    if split_summary.get("test_success") != EXPECTED_TARGET_COUNTS["test"][1]:
        raise AssertionError("Split summary has an unexpected test success count")
    if (
        split_summary.get("test_unsuccessful")
        != EXPECTED_TARGET_COUNTS["test"][0]
    ):
        raise AssertionError("Split summary has an unexpected test failure count")
    if split_summary.get("class_mapping") != {"Success": 1, "Unsuccessful": 0}:
        raise AssertionError("Split summary class mapping changed")

    manifest_inputs = preprocessing_manifest.get("inputs")
    if not isinstance(manifest_inputs, dict):
        raise ValueError("Part 4 manifest is missing its inputs object")
    manifest_hashes = manifest_inputs.get("sha256")
    if not isinstance(manifest_hashes, dict):
        raise ValueError("Part 4 manifest is missing its input hashes")
    if manifest_hashes.get("canonical_train_ids") != _sha256(TRAIN_IDS_PATH):
        raise AssertionError("Canonical train case-ID file changed after Part 4")
    if manifest_hashes.get("canonical_test_ids") != _sha256(TEST_IDS_PATH):
        raise AssertionError("Canonical test case-ID file changed after Part 4")

    saved_artifacts = preprocessing_manifest.get("saved_artifacts")
    if not isinstance(saved_artifacts, dict):
        raise ValueError("Part 4 manifest is missing saved_artifacts")
    if _project_path(str(saved_artifacts.get("y_train"))) != Y_TRAIN_PATH.resolve():
        raise AssertionError("Part 4 manifest points to a different y_train array")
    if _project_path(str(saved_artifacts.get("y_test"))) != Y_TEST_PATH.resolve():
        raise AssertionError("Part 4 manifest points to a different y_test array")


def _part4_artifact_paths(
    preprocessing_manifest: dict[str, Any],
) -> list[Path]:
    """Collect the frozen Part 4 artifacts to protect during this run."""
    saved_artifacts = preprocessing_manifest.get("saved_artifacts")
    feature_names = preprocessing_manifest.get("feature_names")
    if not isinstance(saved_artifacts, dict) or not isinstance(feature_names, dict):
        raise ValueError("Part 4 manifest has incomplete artifact metadata")

    relative_paths = [
        *(str(path) for path in saved_artifacts.values()),
        str(feature_names.get("linear")),
        str(feature_names.get("tree")),
    ]
    paths = [PREPROCESSING_MANIFEST_PATH]
    paths.extend(_project_path(path) for path in relative_paths)
    return list(dict.fromkeys(paths))


def _artifact_hashes(paths: list[Path]) -> dict[Path, str]:
    """Hash artifacts so this task can prove it did not modify Part 4."""
    return {path: _sha256(path) for path in paths}


def _validate_saved_outputs(
    expected_predictions: pd.DataFrame,
    expected_metrics: dict[str, Any],
) -> None:
    """Reload outputs and verify exact row alignment and metric content."""
    saved_predictions = pd.read_csv(
        PREDICTIONS_PATH,
        encoding="utf-8",
        dtype={
            "case_id": "string",
            "y_true": np.int8,
            "y_pred": np.int8,
            "y_prob_success": np.float64,
        },
    )
    expected_columns = ["case_id", "y_true", "y_pred", "y_prob_success"]
    if saved_predictions.columns.tolist() != expected_columns:
        raise AssertionError("Saved prediction columns differ from the contract")
    pd.testing.assert_frame_equal(
        saved_predictions,
        expected_predictions.astype({"case_id": "string"}),
        check_dtype=True,
        check_exact=False,
        rtol=0.0,
        atol=FLOAT_TOLERANCE,
    )

    saved_metrics = _load_json_object(METRICS_PATH)
    if saved_metrics != expected_metrics:
        raise AssertionError("Reloaded metrics differ from computed metrics")


def run_dummy_baseline() -> dict[str, Any]:
    """Fit the train-only prior baseline and evaluate the frozen test set."""
    preprocessing_manifest = _load_json_object(PREPROCESSING_MANIFEST_PATH)
    split_summary = _load_json_object(SPLIT_SUMMARY_PATH)
    train_ids = _load_case_ids(TRAIN_IDS_PATH, EXPECTED_TRAIN_CASES)
    test_ids = _load_case_ids(TEST_IDS_PATH, EXPECTED_TEST_CASES)
    y_train = np.load(Y_TRAIN_PATH, allow_pickle=False)
    y_test = np.load(Y_TEST_PATH, allow_pickle=False)
    _validate_frozen_inputs(
        preprocessing_manifest,
        split_summary,
        train_ids,
        test_ids,
        y_train,
        y_test,
    )

    protected_paths = _part4_artifact_paths(preprocessing_manifest)
    hashes_before = _artifact_hashes(protected_paths)

    # DummyClassifier ignores feature values. One deterministic placeholder
    # column avoids loading either large transformed Part 4 feature matrix.
    X_train_placeholder = np.zeros((len(y_train), 1), dtype=np.uint8)
    X_test_placeholder = np.zeros((len(y_test), 1), dtype=np.uint8)
    classifier = DummyClassifier(strategy="prior", random_state=42)
    classifier.fit(X_train_placeholder, y_train)

    if classifier.n_features_in_ != X_train_placeholder.shape[1]:
        raise AssertionError("Classifier fit did not use the placeholder contract")
    if classifier.classes_.tolist() != [0, 1]:
        raise AssertionError(f"Unexpected classifier classes: {classifier.classes_}")
    positive_columns = np.flatnonzero(classifier.classes_ == POSITIVE_CLASS)
    if positive_columns.size != 1:
        raise AssertionError("Class label 1 has no unique probability column")
    positive_column = int(positive_columns[0])

    y_pred = classifier.predict(X_test_placeholder).astype(np.int8, copy=False)
    probability_matrix = classifier.predict_proba(X_test_placeholder)
    y_prob_success = probability_matrix[:, positive_column]
    training_success_prior = float(np.mean(y_train == POSITIVE_CLASS))

    if len(y_pred) != EXPECTED_TEST_CASES:
        raise AssertionError("Test prediction count is not 6,276")
    if len(y_prob_success) != EXPECTED_TEST_CASES:
        raise AssertionError("Test probability count is not 6,276")
    if not np.isfinite(y_prob_success).all():
        raise AssertionError("Predicted probabilities contain non-finite values")
    if not np.logical_and(y_prob_success >= 0, y_prob_success <= 1).all():
        raise AssertionError("Predicted probabilities fall outside [0, 1]")
    if not set(np.unique(y_pred).tolist()).issubset({0, 1}):
        raise AssertionError("Predicted labels contain a value outside {0, 1}")
    if not np.allclose(
        y_prob_success,
        training_success_prior,
        rtol=0.0,
        atol=FLOAT_TOLERANCE,
    ):
        raise AssertionError("Success probability differs from the training prior")
    if np.unique(y_prob_success).size != 1:
        raise AssertionError("Prior-strategy test probabilities are not constant")

    roc_auc = float(roc_auc_score(y_test, y_prob_success))
    expected_constant_auc = float(
        roc_auc_score(
            y_test,
            np.full(y_test.shape, training_success_prior, dtype=np.float64),
        )
    )
    if not np.isclose(
        roc_auc,
        expected_constant_auc,
        rtol=0.0,
        atol=FLOAT_TOLERANCE,
    ):
        raise AssertionError("ROC-AUC differs from the derived constant-score AUC")
    if not np.isclose(roc_auc, 0.5, rtol=0.0, atol=FLOAT_TOLERANCE):
        raise AssertionError("Constant prior probabilities did not yield ROC-AUC 0.5")

    average_precision = float(average_precision_score(y_test, y_prob_success))
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(
        precision_score(y_test, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)
    )
    recall = float(
        recall_score(y_test, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)
    )
    f1 = float(f1_score(y_test, y_pred, pos_label=POSITIVE_CLASS, zero_division=0))
    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=CONFUSION_MATRIX_LABELS,
    )

    predictions = pd.DataFrame(
        {
            "case_id": test_ids,
            "y_true": y_test.astype(np.int8, copy=False),
            "y_pred": y_pred,
            "y_prob_success": y_prob_success,
        }
    )
    if predictions["case_id"].tolist() != test_ids:
        raise AssertionError("Prediction rows do not match canonical test-ID order")

    assertions = {
        "A_y_train_length_25100": "PASS",
        "B_y_test_length_6276": "PASS",
        "C_target_distributions_match_part3": "PASS",
        "D_test_prediction_count_6276": "PASS",
        "E_test_probability_count_6276": "PASS",
        "F_probabilities_finite": "PASS",
        "G_probabilities_in_unit_interval": "PASS",
        "H_predicted_labels_binary": "PASS",
        "I_classifier_fit_on_train_only": "PASS",
        "J_success_probability_uses_class_label_1": "PASS",
        "K_prediction_order_matches_canonical_test_ids": "PASS",
        "L_no_test_label_threshold_selection": "PASS",
        "M_part4_preprocessing_not_fit_or_modified": "PASS",
        "N_success_probability_equals_training_prior": "PASS",
        "O_success_probability_constant_across_test": "PASS",
        "P_roc_auc_is_constant_score_half": "PASS",
    }
    metrics: dict[str, Any] = {
        "baseline_version": "v1_part5",
        "model_name": "DummyClassifier",
        "model_parameters": {"strategy": "prior", "random_state": 42},
        "fit_scope": "train_only",
        "reported_evaluation_split": "test",
        "positive_class": POSITIVE_CLASS,
        "positive_class_name": "Success",
        "classifier_classes": [int(label) for label in classifier.classes_],
        "positive_probability_column_index": positive_column,
        "train_cases": len(y_train),
        "test_cases": len(y_test),
        "train_target_counts": {
            str(label): count for label, count in _target_counts(y_train).items()
        },
        "test_target_counts": {
            str(label): count for label, count in _target_counts(y_test).items()
        },
        "training_success_prior": training_success_prior,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "average_precision_label": "PR-AUC (Average Precision)",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "labels": CONFUSION_MATRIX_LABELS,
            "rows": "true labels",
            "columns": "predicted labels",
            "values": matrix.astype(int).tolist(),
        },
        "classification_decision_rule": (
            "DummyClassifier.predict default decision (highest learned class "
            "prior); no test-set threshold selection or tuning"
        ),
        "threshold_optimization": False,
        "test_labels_used_for_threshold_selection": False,
        "input_artifacts": {
            "y_train": Y_TRAIN_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "y_test": Y_TEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "canonical_test_ids": TEST_IDS_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": {
                "y_train": _sha256(Y_TRAIN_PATH),
                "y_test": _sha256(Y_TEST_PATH),
                "canonical_test_ids": _sha256(TEST_IDS_PATH),
            },
        },
        "placeholder_features": {
            "reason": "DummyClassifier ignores feature values",
            "train_shape": list(X_train_placeholder.shape),
            "test_shape": list(X_test_placeholder.shape),
        },
        "assertions": assertions,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8")
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
        file.write("\n")

    if _artifact_hashes(protected_paths) != hashes_before:
        raise AssertionError("A frozen Part 4 artifact changed during Part 5")
    _validate_saved_outputs(predictions, metrics)

    print_report(metrics, y_prob_success, y_pred)
    return metrics


def print_report(
    metrics: dict[str, Any],
    y_prob_success: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Print a compact, reproducible Part 5 audit report."""
    print("\n=== BASELINE V1 PART 5: DUMMYCLASSIFIER ===")
    print(f"Train cases: {metrics['train_cases']:,}")
    print(f"Test cases: {metrics['test_cases']:,}")
    print(f"Classifier classes_: {metrics['classifier_classes']}")
    print(f"Training success prior: {metrics['training_success_prior']:.16f}")
    print(f"Unique predicted probabilities: {np.unique(y_prob_success).tolist()}")
    print(f"Unique predicted classes: {np.unique(y_pred).tolist()}")
    print(f"ROC-AUC: {metrics['roc_auc']:.16f}")
    print(f"PR-AUC (Average Precision): {metrics['average_precision']:.16f}")
    print(f"Accuracy: {metrics['accuracy']:.16f}")
    print(f"Precision: {metrics['precision']:.16f}")
    print(f"Recall: {metrics['recall']:.16f}")
    print(f"F1: {metrics['f1']:.16f}")
    print("Confusion matrix labels: [0, 1]")
    print("Rows=true labels; columns=predicted labels")
    print(np.asarray(metrics["confusion_matrix"]["values"]))
    print("Threshold optimization: false")
    print("\n=== SANITY ASSERTIONS ===")
    for name, result in metrics["assertions"].items():
        print(f"{name}: {result}")
    print("\nAll Baseline V1 Part 5 assertions passed.")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved predictions: {PREDICTIONS_PATH}")


def main() -> None:
    """Run Baseline V1 Part 5 and stop before learned models."""
    run_dummy_baseline()


if __name__ == "__main__":
    main()
