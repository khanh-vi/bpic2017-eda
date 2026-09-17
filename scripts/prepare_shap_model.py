"""Freeze and verify the Baseline V1 Random Forest for SHAP Pilot V1.

This Part 0 script only consumes frozen Baseline V1 artifacts, reconstructs
the exact Part 7 estimator from training data, verifies its scientific
identity against the saved test outputs, and persists the verified model.
It does not fit preprocessing, tune the model, optimize a threshold, run
cross-validation, or perform any SHAP analysis.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
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
BASELINE_RESULTS_DIR = PROJECT_ROOT / "results" / "baseline_v1"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "baseline_k10"
BASELINE_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "baseline_v1"
PILOT_RESULTS_DIR = PROJECT_ROOT / "results" / "shap_pilot_v1"
PILOT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "shap_pilot_v1"

PREPROCESSING_MANIFEST_PATH = BASELINE_RESULTS_DIR / "preprocessing_manifest.json"
SPLIT_SUMMARY_PATH = BASELINE_RESULTS_DIR / "split_summary.json"
TRAIN_IDS_PATH = BASELINE_RESULTS_DIR / "train_case_ids.csv"
TEST_IDS_PATH = BASELINE_RESULTS_DIR / "test_case_ids.csv"
FEATURE_NAMES_PATH = BASELINE_RESULTS_DIR / "feature_names_tree.csv"
PART7_METRICS_PATH = BASELINE_RESULTS_DIR / "random_forest_metrics.json"
PART7_PREDICTIONS_PATH = BASELINE_RESULTS_DIR / "random_forest_predictions.csv"
PART7_IMPORTANCES_PATH = (
    BASELINE_RESULTS_DIR / "random_forest_feature_importances.csv"
)

X_TRAIN_PATH = PROCESSED_DIR / "X_train_tree.npz"
X_TEST_PATH = PROCESSED_DIR / "X_test_tree.npz"
Y_TRAIN_PATH = PROCESSED_DIR / "y_train.npy"
Y_TEST_PATH = PROCESSED_DIR / "y_test.npy"

MODEL_ARTIFACT_PATH = PILOT_ARTIFACTS_DIR / "random_forest_baseline.joblib"
MANIFEST_PATH = PILOT_RESULTS_DIR / "model_freeze_manifest.json"

EXPECTED_TRAIN_CASES = 25_100
EXPECTED_TEST_CASES = 6_276
EXPECTED_FEATURES = 165
EXPECTED_TARGET_COUNTS = {
    "train": {0: 11_322, 1: 13_778},
    "test": {0: 2_826, 1: 3_450},
}
POSITIVE_CLASS = 1
CONFUSION_MATRIX_LABELS = [0, 1]
FLOAT_TOLERANCE = 1e-12

# This is the complete, unchanged Part 7 construction contract.
FIXED_PARAMETERS: dict[str, Any] = {
    "n_estimators": 300,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
    "class_weight": None,
    "random_state": 42,
    "n_jobs": -1,
}

EXPLICIT_PROTECTED_PATHS = [
    PART7_METRICS_PATH,
    PART7_PREDICTIONS_PATH,
    PART7_IMPORTANCES_PATH,
    TRAIN_IDS_PATH,
    TEST_IDS_PATH,
    X_TRAIN_PATH,
    X_TEST_PATH,
    Y_TRAIN_PATH,
    Y_TEST_PATH,
]


def _relative(path: Path) -> str:
    """Return a repository-relative path with stable separators."""
    return path.relative_to(PROJECT_ROOT).as_posix()


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a required file."""
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    """Load a required JSON object."""
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _protected_paths() -> list[Path]:
    """Collect every existing Part 1-8 baseline artifact for hashing."""
    roots = [BASELINE_RESULTS_DIR, PROCESSED_DIR, BASELINE_ARTIFACTS_DIR]
    paths = sorted(
        {
            path
            for root in roots
            if root.is_dir()
            for path in root.rglob("*")
            if path.is_file()
        },
        key=lambda path: _relative(path),
    )
    missing = [path for path in EXPLICIT_PROTECTED_PATHS if path not in paths]
    if missing:
        raise FileNotFoundError(f"Protected artifacts are missing: {missing}")
    return paths


def _artifact_hashes(paths: list[Path]) -> dict[Path, str]:
    """Hash protected artifacts without modifying them."""
    return {path: _sha256(path) for path in paths}


def _target_counts(target: np.ndarray) -> dict[int, int]:
    """Return integer target counts."""
    labels, counts = np.unique(target, return_counts=True)
    return {
        int(label): int(count)
        for label, count in zip(labels, counts, strict=True)
    }


def _load_case_ids(path: Path, expected_rows: int) -> list[str]:
    """Load a canonical case-ID file without changing its order."""
    frame = pd.read_csv(path, encoding="utf-8", dtype={"case_id": "string"})
    if frame.columns.tolist() != ["case_id"]:
        raise AssertionError(f"Unexpected columns in {path}: {frame.columns.tolist()}")
    if len(frame) != expected_rows:
        raise AssertionError(f"Unexpected row count in {path}: {len(frame)}")
    if frame["case_id"].isna().any() or not frame["case_id"].is_unique:
        raise AssertionError(f"Case IDs must be non-null and unique in {path}")
    return frame["case_id"].astype(str).tolist()


def _load_feature_names() -> pd.DataFrame:
    """Load and validate the ordered TREE feature metadata."""
    frame = pd.read_csv(FEATURE_NAMES_PATH, encoding="utf-8")
    expected_columns = ["feature_index", "feature_name", "source_group"]
    if frame.columns.tolist() != expected_columns:
        raise AssertionError(
            f"Unexpected feature-name columns: {frame.columns.tolist()}"
        )
    if len(frame) != EXPECTED_FEATURES:
        raise AssertionError(f"Unexpected feature-name count: {len(frame)}")
    if frame["feature_index"].tolist() != list(range(EXPECTED_FEATURES)):
        raise AssertionError("Feature indices are not contiguous and ordered")
    if frame["feature_name"].isna().any():
        raise AssertionError("Feature names contain missing values")
    if not frame["feature_name"].is_unique:
        raise AssertionError("TREE feature names are not unique")
    return frame


def _validate_frozen_inputs(
    preprocessing_manifest: dict[str, Any],
    split_summary: dict[str, Any],
    part7_metrics: dict[str, Any],
    train_ids: list[str],
    test_ids: list[str],
    X_train: sparse.spmatrix,
    X_test: sparse.spmatrix,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: pd.DataFrame,
) -> None:
    """Assert the frozen Part 3/4/7 data and configuration contract."""
    if X_train.shape != (EXPECTED_TRAIN_CASES, EXPECTED_FEATURES):
        raise AssertionError(f"Unexpected X_train_tree shape: {X_train.shape}")
    if X_test.shape != (EXPECTED_TEST_CASES, EXPECTED_FEATURES):
        raise AssertionError(f"Unexpected X_test_tree shape: {X_test.shape}")
    if not sparse.isspmatrix_csr(X_train) or not sparse.isspmatrix_csr(X_test):
        raise AssertionError("Frozen TREE matrices must be CSR sparse matrices")
    if y_train.shape != (EXPECTED_TRAIN_CASES,) or y_train.ndim != 1:
        raise AssertionError(f"Unexpected y_train shape: {y_train.shape}")
    if y_test.shape != (EXPECTED_TEST_CASES,) or y_test.ndim != 1:
        raise AssertionError(f"Unexpected y_test shape: {y_test.shape}")
    if _target_counts(y_train) != EXPECTED_TARGET_COUNTS["train"]:
        raise AssertionError("y_train distribution differs from frozen Part 3")
    if _target_counts(y_test) != EXPECTED_TARGET_COUNTS["test"]:
        raise AssertionError("y_test distribution differs from frozen Part 3")
    if not np.isfinite(X_train.data).all() or not np.isfinite(X_test.data).all():
        raise AssertionError("Frozen TREE matrices contain NaN or infinity")
    if not np.isfinite(y_train).all() or not np.isfinite(y_test).all():
        raise AssertionError("Frozen targets contain NaN or infinity")
    if set(np.unique(y_train).tolist()) != {0, 1}:
        raise AssertionError("y_train labels differ from {0, 1}")
    if set(np.unique(y_test).tolist()) != {0, 1}:
        raise AssertionError("y_test labels differ from {0, 1}")
    if len(feature_names) != X_train.shape[1] or len(feature_names) != X_test.shape[1]:
        raise AssertionError("Feature-name count differs from matrix columns")
    if not set(train_ids).isdisjoint(test_ids):
        raise AssertionError("Canonical train and test case IDs overlap")

    expected_string_counts = {
        split: {str(label): count for label, count in counts.items()}
        for split, counts in EXPECTED_TARGET_COUNTS.items()
    }
    if preprocessing_manifest.get("preprocessing_fit_scope") != "train_only":
        raise AssertionError("Part 4 preprocessing was not train-only")
    if preprocessing_manifest.get("classifier_trained") is not False:
        raise AssertionError("Part 4 unexpectedly records classifier training")
    if preprocessing_manifest.get("split_sizes") != {
        "train": EXPECTED_TRAIN_CASES,
        "test": EXPECTED_TEST_CASES,
    }:
        raise AssertionError("Part 4 split sizes changed")
    if preprocessing_manifest.get("target_distributions") != expected_string_counts:
        raise AssertionError("Part 4 target distributions changed")

    tree = preprocessing_manifest.get("representations", {}).get("tree", {})
    if tree.get("numeric_processing") != "unscaled":
        raise AssertionError("TREE representation is not recorded as unscaled")
    if tree.get("feature_dimension") != EXPECTED_FEATURES:
        raise AssertionError("Part 4 TREE feature count changed")
    expected_shapes = {
        "train_matrix": (EXPECTED_TRAIN_CASES, EXPECTED_FEATURES),
        "test_matrix": (EXPECTED_TEST_CASES, EXPECTED_FEATURES),
    }
    for key, expected_shape in expected_shapes.items():
        metadata = tree.get(key, {})
        if (metadata.get("rows"), metadata.get("columns")) != expected_shape:
            raise AssertionError(f"Part 4 {key} shape changed")

    if split_summary.get("train_cases") != EXPECTED_TRAIN_CASES:
        raise AssertionError("Part 3 train size changed")
    if split_summary.get("test_cases") != EXPECTED_TEST_CASES:
        raise AssertionError("Part 3 test size changed")
    split_counts = {
        "train": {
            0: split_summary.get("train_unsuccessful"),
            1: split_summary.get("train_success"),
        },
        "test": {
            0: split_summary.get("test_unsuccessful"),
            1: split_summary.get("test_success"),
        },
    }
    if split_counts != EXPECTED_TARGET_COUNTS:
        raise AssertionError("Part 3 target distributions changed")
    if split_summary.get("class_mapping") != {"Success": 1, "Unsuccessful": 0}:
        raise AssertionError("Part 3 class mapping changed")

    preprocessing_hashes = preprocessing_manifest.get("inputs", {}).get("sha256", {})
    if preprocessing_hashes.get("canonical_train_ids") != _sha256(TRAIN_IDS_PATH):
        raise AssertionError("Canonical Part 3 train case IDs changed")
    if preprocessing_hashes.get("canonical_test_ids") != _sha256(TEST_IDS_PATH):
        raise AssertionError("Canonical Part 3 test case IDs changed")

    if part7_metrics.get("parameters") != FIXED_PARAMETERS:
        raise AssertionError("Stored Part 7 model parameters differ from the contract")
    if part7_metrics.get("fit_scope") != "train_only":
        raise AssertionError("Stored Part 7 fit scope is not train-only")
    if part7_metrics.get("input_representation") != "unscaled TREE":
        raise AssertionError("Stored Part 7 input representation changed")
    if part7_metrics.get("positive_class") != POSITIVE_CLASS:
        raise AssertionError("Stored Part 7 positive class changed")
    if part7_metrics.get("hyperparameter_tuning") is not False:
        raise AssertionError("Stored Part 7 unexpectedly records tuning")
    if part7_metrics.get("threshold_optimization") is not False:
        raise AssertionError(
            "Stored Part 7 unexpectedly records threshold optimization"
        )

    recorded_hashes = part7_metrics.get("input_artifacts", {})
    input_paths = [
        X_TRAIN_PATH,
        X_TEST_PATH,
        Y_TRAIN_PATH,
        Y_TEST_PATH,
        TEST_IDS_PATH,
        FEATURE_NAMES_PATH,
        PREPROCESSING_MANIFEST_PATH,
        SPLIT_SUMMARY_PATH,
    ]
    for path in input_paths:
        recorded = recorded_hashes.get(_relative(path))
        if recorded != _sha256(path):
            raise AssertionError(f"Part 7 input identity changed: {_relative(path)}")


def _new_classifier() -> RandomForestClassifier:
    """Construct the exact fixed Baseline V1 Part 7 estimator."""
    return RandomForestClassifier(**FIXED_PARAMETERS)


def _validate_fitted_classifier(model: RandomForestClassifier) -> int:
    """Validate fitted model semantics and return the Success column index."""
    if type(model) is not RandomForestClassifier:
        raise AssertionError(f"Unexpected fitted model type: {type(model)!r}")
    effective = model.get_params(deep=False)
    changed = {
        name: (expected, effective.get(name))
        for name, expected in FIXED_PARAMETERS.items()
        if effective.get(name) != expected
    }
    if changed:
        raise AssertionError(f"Random Forest parameters changed: {changed}")
    if not np.array_equal(model.classes_, np.asarray([0, 1])):
        raise AssertionError(f"Unexpected model.classes_: {model.classes_}")
    if model.n_features_in_ != EXPECTED_FEATURES:
        raise AssertionError(f"Unexpected n_features_in_: {model.n_features_in_}")
    if len(model.estimators_) != FIXED_PARAMETERS["n_estimators"]:
        raise AssertionError("Fitted estimator count differs from 300")
    positive_columns = np.flatnonzero(model.classes_ == POSITIVE_CLASS)
    if positive_columns.size != 1:
        raise AssertionError("Class 1 has no unique predict_proba column")
    return int(positive_columns[0])


def _predict(
    model: RandomForestClassifier,
    X_test: sparse.spmatrix,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Predict with the fitted model using the explicitly resolved class-1 column."""
    positive_column = _validate_fitted_classifier(model)
    y_pred = model.predict(X_test).astype(np.int8, copy=False)
    probabilities = model.predict_proba(X_test)
    y_prob_success = probabilities[:, positive_column]
    if y_pred.shape != (EXPECTED_TEST_CASES,):
        raise AssertionError(f"Unexpected prediction shape: {y_pred.shape}")
    if y_prob_success.shape != (EXPECTED_TEST_CASES,):
        raise AssertionError(f"Unexpected probability shape: {y_prob_success.shape}")
    if not np.isfinite(y_prob_success).all():
        raise AssertionError("Predicted probabilities contain non-finite values")
    if not np.logical_and(y_prob_success >= 0, y_prob_success <= 1).all():
        raise AssertionError("Predicted probabilities fall outside [0, 1]")
    return y_pred, y_prob_success, positive_column


def _metric_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_success: np.ndarray,
) -> dict[str, Any]:
    """Recompute the Part 7 test metrics for verification only."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob_success)),
        "average_precision": float(
            average_precision_score(y_true, y_prob_success)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=CONFUSION_MATRIX_LABELS,
        ).astype(int).tolist(),
    }


def _structure(model: RandomForestClassifier) -> dict[str, Any]:
    """Return complete tree-depth and leaf-count audits."""
    depths = np.asarray(
        [estimator.tree_.max_depth for estimator in model.estimators_],
        dtype=np.int64,
    )
    leaves = np.asarray(
        [estimator.tree_.n_leaves for estimator in model.estimators_],
        dtype=np.int64,
    )
    return {
        "classes": [int(value) for value in model.classes_],
        "n_estimators": len(model.estimators_),
        "n_features_in": int(model.n_features_in_),
        "tree_depth_summary": {
            "minimum": int(depths.min()),
            "mean": float(depths.mean()),
            "maximum": int(depths.max()),
        },
        "leaf_count_summary": {
            "minimum": int(leaves.min()),
            "mean": float(leaves.mean()),
            "maximum": int(leaves.max()),
        },
        "tree_depths": depths,
        "leaf_counts": leaves,
    }


def _validate_metric_reproduction(
    current: dict[str, Any],
    part7_metrics: dict[str, Any],
) -> None:
    """Compare recomputed metrics with the immutable Part 7 report."""
    for name in [
        "roc_auc",
        "average_precision",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]:
        if not np.isclose(
            current[name],
            float(part7_metrics[name]),
            rtol=0.0,
            atol=FLOAT_TOLERANCE,
        ):
            raise AssertionError(
                f"Recomputed {name} differs from Part 7: "
                f"{current[name]} != {part7_metrics[name]}"
            )
    if current["confusion_matrix"] != part7_metrics.get("confusion_matrix", {}).get(
        "values"
    ):
        raise AssertionError("Recomputed confusion matrix differs from Part 7")


def _validate_structure_reproduction(
    current: dict[str, Any],
    part7_metrics: dict[str, Any],
) -> None:
    """Compare available fitted structure summaries with Part 7."""
    if current["classes"] != part7_metrics.get("classifier_classes"):
        raise AssertionError("Classifier classes differ from Part 7")
    if current["n_estimators"] != part7_metrics.get("n_estimators"):
        raise AssertionError("Estimator count differs from Part 7")
    if current["n_features_in"] != part7_metrics.get("feature_count"):
        raise AssertionError("Fitted feature count differs from Part 7")
    for current_key, saved_key in [
        ("tree_depth_summary", "tree_depth_summary"),
        ("leaf_count_summary", "leaf_count_summary"),
    ]:
        current_summary = current[current_key]
        saved_summary = part7_metrics.get(saved_key, {})
        for endpoint in ["minimum", "maximum"]:
            if current_summary[endpoint] != saved_summary.get(endpoint):
                raise AssertionError(f"{current_key} {endpoint} differs from Part 7")
        if not np.isclose(
            current_summary["mean"],
            float(saved_summary.get("mean")),
            rtol=0.0,
            atol=FLOAT_TOLERANCE,
        ):
            raise AssertionError(f"{current_key} mean differs from Part 7")


def _validate_importances(
    model: RandomForestClassifier,
    feature_names: pd.DataFrame,
) -> tuple[np.ndarray, bool, float, float]:
    """Validate and reproduce the stored impurity importances."""
    current = model.feature_importances_
    if current.shape != (EXPECTED_FEATURES,):
        raise AssertionError(f"Unexpected feature-importance shape: {current.shape}")
    if not np.isfinite(current).all() or (current < 0).any():
        raise AssertionError("Feature importances must be finite and non-negative")
    if not np.isclose(current.sum(), 1.0, rtol=0.0, atol=FLOAT_TOLERANCE):
        raise AssertionError("Feature importances do not sum approximately to 1")

    saved = pd.read_csv(PART7_IMPORTANCES_PATH, encoding="utf-8")
    if saved.columns.tolist() != ["feature", "importance"]:
        raise AssertionError("Stored Part 7 importance columns changed")
    if len(saved) != EXPECTED_FEATURES:
        raise AssertionError("Stored Part 7 importance count differs from 165")
    canonical_names = feature_names["feature_name"].astype(str).tolist()
    if saved["feature"].astype(str).tolist() != canonical_names:
        raise AssertionError("Stored importances do not map to canonical feature order")
    saved_values = saved["importance"].to_numpy(dtype=np.float64)
    if not np.isfinite(saved_values).all() or (saved_values < 0).any():
        raise AssertionError("Stored Part 7 importances are invalid")
    differences = np.abs(current - saved_values)
    maximum_difference = float(differences.max(initial=0.0))
    mean_difference = float(differences.mean())
    exact = np.array_equal(current, saved_values)
    reproduced = np.allclose(
        current,
        saved_values,
        rtol=0.0,
        atol=FLOAT_TOLERANCE,
    )
    if not reproduced:
        raise AssertionError(
            "Reconstructed feature importances differ from Part 7; "
            f"max_abs_diff={maximum_difference:.17g}, "
            f"mean_abs_diff={mean_difference:.17g}"
        )
    return current, exact, maximum_difference, mean_difference


def _validate_deterministic_rerun(
    first_model: RandomForestClassifier,
    second_model: RandomForestClassifier,
    X_test: sparse.spmatrix,
    y_test: np.ndarray,
    first_pred: np.ndarray,
    first_prob: np.ndarray,
    first_metrics: dict[str, Any],
    first_structure: dict[str, Any],
) -> dict[str, Any]:
    """Require a second independent train-only fit to be identical."""
    second_pred, second_prob, second_positive_column = _predict(second_model, X_test)
    second_metrics = _metric_values(y_test, second_pred, second_prob)
    second_structure = _structure(second_model)
    checks = {
        "classes_exact": np.array_equal(first_model.classes_, second_model.classes_),
        "positive_probability_column_exact": second_positive_column
        == int(np.flatnonzero(first_model.classes_ == POSITIVE_CLASS)[0]),
        "estimator_count_exact": len(first_model.estimators_)
        == len(second_model.estimators_),
        "tree_depths_exact": np.array_equal(
            first_structure["tree_depths"], second_structure["tree_depths"]
        ),
        "leaf_counts_exact": np.array_equal(
            first_structure["leaf_counts"], second_structure["leaf_counts"]
        ),
        "predictions_exact": np.array_equal(first_pred, second_pred),
        "probabilities_exact": np.array_equal(first_prob, second_prob),
        "feature_importances_exact": np.array_equal(
            first_model.feature_importances_, second_model.feature_importances_
        ),
        "metrics_exact": first_metrics == second_metrics,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Deterministic rerun checks failed: {failed}")
    return {
        "method": "two independent fits with the fixed train-only configuration",
        **{name: bool(passed) for name, passed in checks.items()},
        "structural_audit_exact": True,
        "result": "PASS",
    }


def _verify_reloaded_model(
    model_path: Path,
    X_test: sparse.spmatrix,
    expected_pred: np.ndarray,
    expected_prob: np.ndarray,
) -> RandomForestClassifier:
    """Reload a serialized model and require exact test outputs."""
    reloaded = joblib.load(model_path)
    if type(reloaded) is not RandomForestClassifier:
        raise AssertionError(f"Unexpected reloaded model type: {type(reloaded)!r}")
    reloaded_pred, reloaded_prob, _ = _predict(reloaded, X_test)
    if not np.array_equal(reloaded_pred, expected_pred):
        raise AssertionError("Reloaded model predictions differ from pre-save outputs")
    if not np.array_equal(reloaded_prob, expected_prob):
        differences = np.abs(reloaded_prob - expected_prob)
        raise AssertionError(
            "Reloaded model probabilities differ from pre-save outputs; "
            f"max_abs_diff={differences.max():.17g}"
        )
    return reloaded


def _json_ready_structure(structure: dict[str, Any]) -> dict[str, Any]:
    """Remove per-tree arrays from a structure audit for JSON output."""
    return {
        key: value
        for key, value in structure.items()
        if key not in {"tree_depths", "leaf_counts"}
    }


def prepare_shap_model() -> dict[str, Any]:
    """Reconstruct, verify, serialize, and document the Part 7 model."""
    protected_paths = _protected_paths()
    protected_hashes_before = _artifact_hashes(protected_paths)

    preprocessing_manifest = _load_json(PREPROCESSING_MANIFEST_PATH)
    split_summary = _load_json(SPLIT_SUMMARY_PATH)
    part7_metrics = _load_json(PART7_METRICS_PATH)
    train_ids = _load_case_ids(TRAIN_IDS_PATH, EXPECTED_TRAIN_CASES)
    test_ids = _load_case_ids(TEST_IDS_PATH, EXPECTED_TEST_CASES)
    feature_names = _load_feature_names()
    X_train = sparse.load_npz(X_TRAIN_PATH)
    X_test = sparse.load_npz(X_TEST_PATH)
    y_train = np.load(Y_TRAIN_PATH, allow_pickle=False)
    y_test = np.load(Y_TEST_PATH, allow_pickle=False)

    _validate_frozen_inputs(
        preprocessing_manifest,
        split_summary,
        part7_metrics,
        train_ids,
        test_ids,
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
    )

    saved_predictions = pd.read_csv(
        PART7_PREDICTIONS_PATH,
        encoding="utf-8",
        dtype={
            "case_id": "string",
            "y_true": np.int8,
            "y_pred": np.int8,
            "y_prob_success": np.float64,
        },
    )
    expected_prediction_columns = [
        "case_id",
        "y_true",
        "y_pred",
        "y_prob_success",
    ]
    if saved_predictions.columns.tolist() != expected_prediction_columns:
        raise AssertionError("Stored Part 7 prediction columns changed")
    if len(saved_predictions) != EXPECTED_TEST_CASES:
        raise AssertionError("Stored Part 7 prediction row count differs from 6,276")
    if saved_predictions["case_id"].astype(str).tolist() != test_ids:
        raise AssertionError(
            "Stored Part 7 case-ID order differs from the canonical test IDs"
        )
    if not np.array_equal(saved_predictions["y_true"].to_numpy(), y_test):
        raise AssertionError("Stored Part 7 y_true differs from frozen y_test")

    # These are the only estimator fit calls. Both use frozen training inputs only.
    model = _new_classifier()
    model.fit(X_train, y_train)
    y_pred, y_prob_success, positive_column = _predict(model, X_test)
    structure = _structure(model)
    _validate_structure_reproduction(structure, part7_metrics)

    saved_y_pred = saved_predictions["y_pred"].to_numpy(dtype=np.int8)
    saved_y_prob = saved_predictions["y_prob_success"].to_numpy(dtype=np.float64)
    predictions_exact = np.array_equal(y_pred, saved_y_pred)
    if not predictions_exact:
        mismatch_count = int(np.count_nonzero(y_pred != saved_y_pred))
        raise AssertionError(
            f"Reconstructed predictions differ from Part 7 in {mismatch_count} rows"
        )
    probability_differences = np.abs(y_prob_success - saved_y_prob)
    maximum_probability_difference = float(probability_differences.max(initial=0.0))
    mean_probability_difference = float(probability_differences.mean())
    probabilities_exact = np.array_equal(y_prob_success, saved_y_prob)
    probabilities_reproduced = np.allclose(
        y_prob_success,
        saved_y_prob,
        rtol=0.0,
        atol=FLOAT_TOLERANCE,
    )
    if not probabilities_reproduced:
        raise AssertionError(
            "Reconstructed probabilities differ from Part 7; "
            f"max_abs_diff={maximum_probability_difference:.17g}, "
            f"mean_abs_diff={mean_probability_difference:.17g}"
        )

    metrics = _metric_values(y_test, y_pred, y_prob_success)
    _validate_metric_reproduction(metrics, part7_metrics)
    (
        importances,
        importances_exact,
        max_importance_difference,
        mean_importance_difference,
    ) = _validate_importances(model, feature_names)

    second_model = _new_classifier()
    second_model.fit(X_train, y_train)
    deterministic_rerun = _validate_deterministic_rerun(
        model,
        second_model,
        X_test,
        y_test,
        y_pred,
        y_prob_success,
        metrics,
        structure,
    )
    del second_model

    if _artifact_hashes(protected_paths) != protected_hashes_before:
        raise AssertionError(
            "A protected Baseline V1 artifact changed before serialization"
        )
    if any(name == "shap" or name.startswith("shap.") for name in sys.modules):
        raise AssertionError("SHAP was unexpectedly imported during Part 0")

    PILOT_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    temporary_model_path = PILOT_ARTIFACTS_DIR / ".random_forest_baseline.joblib.tmp"
    if temporary_model_path.exists():
        temporary_model_path.unlink()
    try:
        joblib.dump(model, temporary_model_path, compress=3)
        _verify_reloaded_model(
            temporary_model_path,
            X_test,
            y_pred,
            y_prob_success,
        )
        artifact_sha256 = _sha256(temporary_model_path)
        temporary_model_path.replace(MODEL_ARTIFACT_PATH)
    finally:
        if temporary_model_path.exists():
            temporary_model_path.unlink()

    reloaded_model = _verify_reloaded_model(
        MODEL_ARTIFACT_PATH,
        X_test,
        y_pred,
        y_prob_success,
    )
    reloaded_structure = _structure(reloaded_model)
    if not np.array_equal(structure["tree_depths"], reloaded_structure["tree_depths"]):
        raise AssertionError("Reloaded tree depths differ from the fitted model")
    if not np.array_equal(structure["leaf_counts"], reloaded_structure["leaf_counts"]):
        raise AssertionError("Reloaded leaf counts differ from the fitted model")
    if _sha256(MODEL_ARTIFACT_PATH) != artifact_sha256:
        raise AssertionError("Model artifact bytes changed after finalization")

    protected_hashes_after = _artifact_hashes(protected_paths)
    if protected_hashes_after != protected_hashes_before:
        raise AssertionError("A protected Baseline V1 artifact changed during Part 0")

    assertions = {
        "A_training_shape_25100_by_165": "PASS",
        "B_test_shape_6276_by_165": "PASS",
        "C_training_target_distribution_matches_part3": "PASS",
        "D_test_target_distribution_matches_part3": "PASS",
        "E_feature_count_165": "PASS",
        "F_feature_names_unique": "PASS",
        "G_model_fit_uses_training_data_only": "PASS",
        "H_model_parameters_match_part7": "PASS",
        "I_model_classes_are_0_1": "PASS",
        "J_positive_probability_maps_to_class_1": "PASS",
        "K_fitted_estimator_count_300": "PASS",
        "L_reconstructed_prediction_count_6276": "PASS",
        "M_predictions_exactly_match_part7": "PASS",
        "N_probabilities_reproduce_part7_at_strict_tolerance": "PASS",
        "O_metrics_reproduce_part7": "PASS",
        "P_confusion_matrix_reproduces_part7": "PASS",
        "Q_feature_importance_count_165": "PASS",
        "R_feature_importances_reproduce_part7": "PASS",
        "S_part1_to_part8_artifacts_unmodified": "PASS",
        "T_reloaded_predictions_exact": "PASS",
        "U_reloaded_probabilities_exact": "PASS",
        "V_no_preprocessing_component_fitted_or_modified": "PASS",
        "W_no_hyperparameter_tuning": "PASS",
        "X_no_threshold_optimization": "PASS",
        "Y_no_shap_import_or_execution": "PASS",
    }
    manifest: dict[str, Any] = {
        "experiment": "shap_pilot_v1",
        "stage": "part_0_model_freeze",
        "model_source": "baseline_v1_random_forest",
        "prediction_point": 10,
        "positive_class": POSITIVE_CLASS,
        "train_cases": EXPECTED_TRAIN_CASES,
        "test_cases": EXPECTED_TEST_CASES,
        "feature_count": EXPECTED_FEATURES,
        "model": "RandomForestClassifier",
        "model_parameters": FIXED_PARAMETERS,
        "scikit_learn_version": sklearn.__version__,
        "training_scope": "train_only",
        "model_tuned": False,
        "hyperparameter_search": False,
        "cross_validation": False,
        "threshold_optimization": False,
        "preprocessing_fitted": False,
        "feature_space_changed": False,
        "part7_predictions_reproduced": predictions_exact,
        "part7_probabilities_reproduced": probabilities_reproduced,
        "probability_comparison": {
            "exact": probabilities_exact,
            "rtol": 0.0,
            "atol": FLOAT_TOLERANCE,
            "maximum_absolute_difference": maximum_probability_difference,
            "mean_absolute_difference": mean_probability_difference,
            "explanation": (
                "Last-bit floating-point refit variation; predictions, metrics, "
                "tree structure, deterministic same-run outputs, and reload "
                "outputs satisfy their required identity checks."
            ),
        },
        "part7_metrics_reproduced": True,
        "verified_metrics": metrics,
        "feature_importances_reproduced": True,
        "feature_importance_comparison": {
            "exact": importances_exact,
            "rtol": 0.0,
            "atol": FLOAT_TOLERANCE,
            "count": int(importances.size),
            "all_finite": bool(np.isfinite(importances).all()),
            "all_non_negative": bool((importances >= 0).all()),
            "sum": float(importances.sum()),
            "maximum_absolute_difference": max_importance_difference,
            "mean_absolute_difference": mean_importance_difference,
        },
        "model_structure": _json_ready_structure(structure),
        "positive_probability_column_index": positive_column,
        "deterministic_rerun": deterministic_rerun,
        "protected_artifacts": {
            "count": len(protected_paths),
            "unchanged": True,
            "sha256": {
                _relative(path): digest
                for path, digest in protected_hashes_before.items()
            },
        },
        "artifact_path": _relative(MODEL_ARTIFACT_PATH),
        "artifact_sha256": artifact_sha256,
        "reload_verification": {
            "type_is_random_forest_classifier": True,
            "n_estimators_300": True,
            "classes_0_1": True,
            "n_features_in_165": True,
            "predictions_exact": True,
            "probabilities_exact": True,
            "result": "PASS",
        },
        "shap_executed": False,
        "assertions": assertions,
        "assertion_status": "PASS",
    }

    PILOT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    temporary_manifest_path = PILOT_RESULTS_DIR / ".model_freeze_manifest.json.tmp"
    if temporary_manifest_path.exists():
        temporary_manifest_path.unlink()
    try:
        with temporary_manifest_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, ensure_ascii=False)
            file.write("\n")
        if _artifact_hashes(protected_paths) != protected_hashes_before:
            raise AssertionError(
                "A protected artifact changed before manifest finalization"
            )
        temporary_manifest_path.replace(MANIFEST_PATH)
    finally:
        if temporary_manifest_path.exists():
            temporary_manifest_path.unlink()

    if _artifact_hashes(protected_paths) != protected_hashes_before:
        raise AssertionError("A protected artifact changed after manifest finalization")
    print_report(manifest)
    return manifest


def print_report(manifest: dict[str, Any]) -> None:
    """Print the measured Part 0 verification results."""
    print("\n=== SHAP PILOT V1 PART 0: VERIFIED MODEL FREEZE ===")
    print(
        f"X_train_tree shape: ({manifest['train_cases']}, "
        f"{manifest['feature_count']})"
    )
    print(f"X_test_tree shape: ({manifest['test_cases']}, {manifest['feature_count']})")
    print(f"Model parameters: {manifest['model_parameters']}")
    structure = manifest["model_structure"]
    print(f"model.classes_: {structure['classes']}")
    print(
        "Positive-class probability index: "
        f"{manifest['positive_probability_column_index']}"
    )
    print(f"Fitted estimators: {structure['n_estimators']}")
    print(f"Tree-depth summary: {structure['tree_depth_summary']}")
    print(f"Leaf-count summary: {structure['leaf_count_summary']}")
    print("Predictions reproduce Part 7 exactly: PASS")
    probability = manifest["probability_comparison"]
    print(
        "Probabilities reproduce Part 7 at strict tolerance: PASS "
        f"(exact={probability['exact']})"
    )
    print(
        "Probability absolute differences: "
        f"max={probability['maximum_absolute_difference']:.17g}, "
        f"mean={probability['mean_absolute_difference']:.17g}"
    )
    print(f"Verified metrics: {manifest['verified_metrics']}")
    importance = manifest["feature_importance_comparison"]
    print(
        "Feature importances reproduce Part 7 at strict tolerance: PASS "
        f"(exact={importance['exact']})"
    )
    print(
        "Protected baseline artifacts: "
        f"{manifest['protected_artifacts']['count']} unchanged"
    )
    print(f"Deterministic rerun: {manifest['deterministic_rerun']['result']}")
    print(f"Model artifact: {manifest['artifact_path']}")
    print(f"Model SHA-256: {manifest['artifact_sha256']}")
    print(f"Reload verification: {manifest['reload_verification']['result']}")
    print(f"Required assertions: {manifest['assertion_status']}")
    print(f"Manifest: {_relative(MANIFEST_PATH)}")
    print("SHAP executed: no")


def main() -> None:
    """Run SHAP Pilot V1 Part 0 and stop."""
    prepare_shap_model()


if __name__ == "__main__":
    main()
