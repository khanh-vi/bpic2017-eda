"""Fit and evaluate the fixed Baseline V1 Random Forest at prefix k=10.

This is Baseline V1 Part 7 only. It consumes the frozen Part 4 unscaled TREE
matrices and target arrays, fits RandomForestClassifier on training data only,
and evaluates once on the frozen temporal test set. It does not fit or modify
preprocessing, select features, tune hyperparameters, optimize a threshold,
run cross-validation, generate comparison plots, or perform SHAP analysis.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

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
RESULTS_DIR = PROJECT_ROOT / "results" / "baseline_v1"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "baseline_k10"

PREPROCESSING_MANIFEST_PATH = RESULTS_DIR / "preprocessing_manifest.json"
SPLIT_SUMMARY_PATH = RESULTS_DIR / "split_summary.json"
TRAIN_IDS_PATH = RESULTS_DIR / "train_case_ids.csv"
TEST_IDS_PATH = RESULTS_DIR / "test_case_ids.csv"
FEATURE_NAMES_PATH = RESULTS_DIR / "feature_names_tree.csv"
DUMMY_METRICS_PATH = RESULTS_DIR / "dummy_metrics.json"
LOGISTIC_METRICS_PATH = RESULTS_DIR / "logistic_metrics.json"

X_TRAIN_PATH = PROCESSED_DIR / "X_train_tree.npz"
X_TEST_PATH = PROCESSED_DIR / "X_test_tree.npz"
Y_TRAIN_PATH = PROCESSED_DIR / "y_train.npy"
Y_TEST_PATH = PROCESSED_DIR / "y_test.npy"

METRICS_PATH = RESULTS_DIR / "random_forest_metrics.json"
PREDICTIONS_PATH = RESULTS_DIR / "random_forest_predictions.csv"
FEATURE_IMPORTANCES_PATH = (
    RESULTS_DIR / "random_forest_feature_importances.csv"
)

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


def _relative(path: Path) -> str:
    """Return a platform-independent repository-relative path."""
    return path.relative_to(PROJECT_ROOT).as_posix()


def _project_path(relative_path: str) -> Path:
    """Resolve a manifest path and ensure it remains inside the project."""
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


def _load_feature_names() -> pd.DataFrame:
    """Load and validate the canonical ordered TREE feature metadata."""
    metadata = pd.read_csv(FEATURE_NAMES_PATH, encoding="utf-8")
    expected_columns = ["feature_index", "feature_name", "source_group"]
    if metadata.columns.tolist() != expected_columns:
        raise AssertionError(
            f"Unexpected feature-name columns: {metadata.columns.tolist()}"
        )
    if len(metadata) != EXPECTED_FEATURES:
        raise AssertionError(f"Unexpected feature-name count: {len(metadata)}")
    if metadata["feature_index"].tolist() != list(range(EXPECTED_FEATURES)):
        raise AssertionError("Feature indices are not contiguous and ordered")
    if metadata["feature_name"].isna().any():
        raise AssertionError("Feature names contain missing values")
    if not metadata["feature_name"].is_unique:
        raise AssertionError("TREE feature names are not unique")
    return metadata


def _validate_frozen_contract(
    preprocessing_manifest: dict[str, Any],
    split_summary: dict[str, Any],
    train_ids: list[str],
    test_ids: list[str],
    X_train: sparse.csr_matrix,
    X_test: sparse.csr_matrix,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: pd.DataFrame,
) -> None:
    """Validate the complete frozen Part 3/4 TREE-data contract."""
    if X_train.shape != (EXPECTED_TRAIN_CASES, EXPECTED_FEATURES):
        raise AssertionError(f"Unexpected X_train_tree shape: {X_train.shape}")
    if X_test.shape != (EXPECTED_TEST_CASES, EXPECTED_FEATURES):
        raise AssertionError(f"Unexpected X_test_tree shape: {X_test.shape}")
    if y_train.ndim != 1 or y_train.shape != (EXPECTED_TRAIN_CASES,):
        raise AssertionError(f"Unexpected y_train shape: {y_train.shape}")
    if y_test.ndim != 1 or y_test.shape != (EXPECTED_TEST_CASES,):
        raise AssertionError(f"Unexpected y_test shape: {y_test.shape}")
    if _target_counts(y_train) != EXPECTED_TARGET_COUNTS["train"]:
        raise AssertionError("y_train distribution differs from frozen Part 3")
    if _target_counts(y_test) != EXPECTED_TARGET_COUNTS["test"]:
        raise AssertionError("y_test distribution differs from frozen Part 3")
    if set(np.unique(y_train).tolist()) != {0, 1}:
        raise AssertionError("y_train labels differ from {0, 1}")
    if set(np.unique(y_test).tolist()) != {0, 1}:
        raise AssertionError("y_test labels differ from {0, 1}")
    if len(feature_names) != X_train.shape[1] or len(feature_names) != X_test.shape[1]:
        raise AssertionError("Feature-name count differs from matrix column count")
    if not sparse.isspmatrix_csr(X_train) or not sparse.isspmatrix_csr(X_test):
        raise AssertionError("Frozen TREE matrices must be CSR sparse matrices")
    if not np.isfinite(X_train.data).all() or not np.isfinite(X_test.data).all():
        raise AssertionError("Frozen TREE matrices contain non-finite values")
    if not set(train_ids).isdisjoint(test_ids):
        raise AssertionError("Canonical train and test IDs overlap")

    expected_manifest_distributions = {
        split: {str(label): count for label, count in counts.items()}
        for split, counts in EXPECTED_TARGET_COUNTS.items()
    }
    if preprocessing_manifest.get("preprocessing_fit_scope") != "train_only":
        raise AssertionError("Part 4 preprocessing was not recorded as train-only")
    if preprocessing_manifest.get("classifier_trained") is not False:
        raise AssertionError("Part 4 unexpectedly records a fitted classifier")
    if preprocessing_manifest.get("split_sizes") != {
        "train": EXPECTED_TRAIN_CASES,
        "test": EXPECTED_TEST_CASES,
    }:
        raise AssertionError("Part 4 split sizes differ from the frozen contract")
    if preprocessing_manifest.get("target_distributions") != expected_manifest_distributions:
        raise AssertionError("Part 4 target distributions differ from Part 3")

    representations = preprocessing_manifest.get("representations")
    if not isinstance(representations, dict):
        raise ValueError("Part 4 manifest is missing representations")
    tree = representations.get("tree")
    if not isinstance(tree, dict):
        raise ValueError("Part 4 manifest is missing the TREE representation")
    if tree.get("numeric_processing") != "unscaled":
        raise AssertionError("TREE representation is not recorded as unscaled")
    if tree.get("feature_dimension") != EXPECTED_FEATURES:
        raise AssertionError("Part 4 manifest has a different TREE feature count")
    expected_shapes = {
        "train_matrix": (EXPECTED_TRAIN_CASES, EXPECTED_FEATURES),
        "test_matrix": (EXPECTED_TEST_CASES, EXPECTED_FEATURES),
    }
    for key, expected_shape in expected_shapes.items():
        matrix_metadata = tree.get(key)
        if not isinstance(matrix_metadata, dict):
            raise ValueError(f"Part 4 manifest is missing TREE {key}")
        recorded_shape = (
            matrix_metadata.get("rows"),
            matrix_metadata.get("columns"),
        )
        if recorded_shape != expected_shape:
            raise AssertionError(f"Part 4 manifest has an unexpected TREE {key} shape")

    if split_summary.get("train_cases") != EXPECTED_TRAIN_CASES:
        raise AssertionError("Split summary has an unexpected train size")
    if split_summary.get("test_cases") != EXPECTED_TEST_CASES:
        raise AssertionError("Split summary has an unexpected test size")
    if split_summary.get("train_success") != EXPECTED_TARGET_COUNTS["train"][1]:
        raise AssertionError("Split summary train-success count changed")
    if split_summary.get("train_unsuccessful") != EXPECTED_TARGET_COUNTS["train"][0]:
        raise AssertionError("Split summary train-unsuccessful count changed")
    if split_summary.get("test_success") != EXPECTED_TARGET_COUNTS["test"][1]:
        raise AssertionError("Split summary test-success count changed")
    if split_summary.get("test_unsuccessful") != EXPECTED_TARGET_COUNTS["test"][0]:
        raise AssertionError("Split summary test-unsuccessful count changed")
    if split_summary.get("class_mapping") != {"Success": 1, "Unsuccessful": 0}:
        raise AssertionError("Split summary class mapping changed")

    manifest_inputs = preprocessing_manifest.get("inputs")
    if not isinstance(manifest_inputs, dict):
        raise ValueError("Part 4 manifest is missing inputs")
    manifest_hashes = manifest_inputs.get("sha256")
    if not isinstance(manifest_hashes, dict):
        raise ValueError("Part 4 manifest is missing input hashes")
    if manifest_hashes.get("canonical_train_ids") != _sha256(TRAIN_IDS_PATH):
        raise AssertionError("Canonical train IDs changed after Part 4")
    if manifest_hashes.get("canonical_test_ids") != _sha256(TEST_IDS_PATH):
        raise AssertionError("Canonical test IDs changed after Part 4")

    saved_artifacts = preprocessing_manifest.get("saved_artifacts")
    feature_name_paths = preprocessing_manifest.get("feature_names")
    if not isinstance(saved_artifacts, dict) or not isinstance(feature_name_paths, dict):
        raise ValueError("Part 4 manifest has incomplete saved-artifact metadata")
    expected_paths = {
        "X_train_tree": X_TRAIN_PATH,
        "X_test_tree": X_TEST_PATH,
        "y_train": Y_TRAIN_PATH,
        "y_test": Y_TEST_PATH,
    }
    for name, expected_path in expected_paths.items():
        if _project_path(str(saved_artifacts.get(name))) != expected_path.resolve():
            raise AssertionError(f"Part 4 manifest points to a different {name}")
    if _project_path(str(feature_name_paths.get("tree"))) != FEATURE_NAMES_PATH.resolve():
        raise AssertionError("Part 4 manifest points to different TREE feature names")


def _protected_artifact_paths(
    preprocessing_manifest: dict[str, Any],
) -> list[Path]:
    """Collect frozen Part 1-6 artifacts that this run must not modify."""
    paths = [
        RESULTS_DIR / "eligibility_summary.json",
        RESULTS_DIR / "eligible_cases_k10.csv",
        RESULTS_DIR / "feature_manifest.json",
        PROCESSED_DIR / "case_features_k10.parquet",
        SPLIT_SUMMARY_PATH,
        TRAIN_IDS_PATH,
        TEST_IDS_PATH,
        PREPROCESSING_MANIFEST_PATH,
        DUMMY_METRICS_PATH,
        RESULTS_DIR / "dummy_predictions.csv",
        LOGISTIC_METRICS_PATH,
        RESULTS_DIR / "logistic_predictions.csv",
        RESULTS_DIR / "logistic_coefficients.csv",
    ]
    saved_artifacts = preprocessing_manifest.get("saved_artifacts")
    feature_names = preprocessing_manifest.get("feature_names")
    if not isinstance(saved_artifacts, dict) or not isinstance(feature_names, dict):
        raise ValueError("Part 4 manifest has incomplete artifact metadata")
    paths.extend(_project_path(str(path)) for path in saved_artifacts.values())
    for representation in ("linear", "tree"):
        relative_path = feature_names.get(representation)
        if not isinstance(relative_path, str):
            raise ValueError(f"Part 4 manifest has no {representation} feature-name path")
        paths.append(_project_path(relative_path))
    return list(dict.fromkeys(paths))


def _artifact_hashes(paths: list[Path]) -> dict[Path, str]:
    """Hash frozen artifacts to prove this run does not modify them."""
    return {path: _sha256(path) for path in paths}


def _new_classifier() -> RandomForestClassifier:
    """Construct the exact fixed Baseline V1 Random Forest."""
    return RandomForestClassifier(**FIXED_PARAMETERS)


def _validate_fitted_classifier(classifier: RandomForestClassifier) -> None:
    """Confirm the fitted estimator retains the complete fixed contract."""
    effective = classifier.get_params(deep=False)
    changed = {
        name: (value, effective.get(name))
        for name, value in FIXED_PARAMETERS.items()
        if effective.get(name) != value
    }
    if changed:
        raise AssertionError(f"Random Forest parameters changed: {changed}")
    if classifier.class_weight is not None:
        raise AssertionError("class_weight changed from None")
    if classifier.n_features_in_ != EXPECTED_FEATURES:
        raise AssertionError("Classifier fit used an unexpected feature count")
    if classifier.classes_.tolist() != [0, 1]:
        raise AssertionError(f"Unexpected classifier classes_: {classifier.classes_}")
    if len(classifier.estimators_) != FIXED_PARAMETERS["n_estimators"]:
        raise AssertionError("Fitted estimator count differs from 300")


def _positive_probability_column(classifier: RandomForestClassifier) -> int:
    """Resolve the probability column for the explicit Success=1 class."""
    positive_columns = np.flatnonzero(classifier.classes_ == POSITIVE_CLASS)
    if positive_columns.size != 1:
        raise AssertionError("Class label 1 has no unique probability column")
    return int(positive_columns[0])


def _metric_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_success: np.ndarray,
) -> dict[str, Any]:
    """Calculate the fixed test-only Baseline V1 metrics."""
    matrix = confusion_matrix(y_true, y_pred, labels=CONFUSION_MATRIX_LABELS)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob_success)),
        "average_precision": float(average_precision_score(y_true, y_prob_success)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=POSITIVE_CLASS,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=POSITIVE_CLASS,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=POSITIVE_CLASS,
                zero_division=0,
            )
        ),
        "confusion_matrix_values": matrix.astype(int).tolist(),
    }


def _tree_summaries(classifier: RandomForestClassifier) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return fitted tree-depth and leaf-count summaries."""
    depths = np.asarray(
        [estimator.tree_.max_depth for estimator in classifier.estimators_],
        dtype=np.int64,
    )
    leaves = np.asarray(
        [estimator.tree_.n_leaves for estimator in classifier.estimators_],
        dtype=np.int64,
    )
    return (
        {
            "minimum": int(depths.min()),
            "mean": float(depths.mean()),
            "maximum": int(depths.max()),
        },
        {
            "minimum": int(leaves.min()),
            "mean": float(leaves.mean()),
            "maximum": int(leaves.max()),
        },
    )


def _feature_importance_frame(
    classifier: RandomForestClassifier,
    feature_names: pd.DataFrame,
) -> pd.DataFrame:
    """Associate all impurity importances with canonical TREE feature names."""
    importances = classifier.feature_importances_
    if importances.shape != (EXPECTED_FEATURES,):
        raise AssertionError(f"Unexpected feature-importance shape: {importances.shape}")
    if len(importances) != len(feature_names):
        raise AssertionError("Importance count differs from feature-name count")
    if not np.isfinite(importances).all():
        raise AssertionError("Feature importances contain non-finite values")
    if (importances < 0).any():
        raise AssertionError("Feature importances contain negative values")
    if not np.isclose(
        importances.sum(),
        1.0,
        rtol=0.0,
        atol=FLOAT_TOLERANCE,
    ):
        raise AssertionError("Feature importances do not sum approximately to 1")
    return pd.DataFrame(
        {
            "feature": feature_names["feature_name"].astype(str),
            "importance": importances,
        }
    )


def _assert_reproducible(
    first: RandomForestClassifier,
    second: RandomForestClassifier,
    X_test: sparse.csr_matrix,
    y_test: np.ndarray,
    first_pred: np.ndarray,
    first_prob: np.ndarray,
    first_metrics: dict[str, Any],
    positive_column: int,
) -> dict[str, Any]:
    """Verify a second independent fixed train-only fit is deterministic."""
    _validate_fitted_classifier(second)
    second_positive_column = _positive_probability_column(second)
    second_pred = second.predict(X_test).astype(np.int8, copy=False)
    second_prob = second.predict_proba(X_test)[:, second_positive_column]
    second_metrics = _metric_values(y_test, second_pred, second_prob)
    checks = {
        "classes_exact": np.array_equal(first.classes_, second.classes_),
        "positive_probability_column_exact": positive_column == second_positive_column,
        "estimator_count_exact": len(first.estimators_) == len(second.estimators_),
        "tree_depths_exact": np.array_equal(
            [tree.tree_.max_depth for tree in first.estimators_],
            [tree.tree_.max_depth for tree in second.estimators_],
        ),
        "leaf_counts_exact": np.array_equal(
            [tree.tree_.n_leaves for tree in first.estimators_],
            [tree.tree_.n_leaves for tree in second.estimators_],
        ),
        "predictions_exact": np.array_equal(first_pred, second_pred),
        "probabilities_exact": np.array_equal(first_prob, second_prob),
        "feature_importances_exact": np.array_equal(
            first.feature_importances_,
            second.feature_importances_,
        ),
        "metrics_exact": first_metrics == second_metrics,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Determinism checks failed: {failed}")
    return {
        "method": "two independent fits with the fixed train-only configuration",
        **{name: bool(passed) for name, passed in checks.items()},
        "numerical_results_identical": True,
        "model_artifact_byte_comparison": "not_applicable_model_not_saved",
        "result": "PASS",
    }


def _metric_deltas(
    current: dict[str, Any],
    baseline: dict[str, Any],
    baseline_name: str,
) -> dict[str, Any]:
    """Calculate factual RF-minus-baseline deltas for the requested metrics."""
    names = ["roc_auc", "average_precision", "f1"]
    missing = [name for name in names if name not in baseline]
    if missing:
        raise ValueError(f"{baseline_name} metrics are missing: {missing}")
    return {
        "definition": f"Random Forest minus {baseline_name}",
        **{name: float(current[name]) - float(baseline[name]) for name in names},
    }


def _validate_saved_outputs(
    expected_predictions: pd.DataFrame,
    expected_importances: pd.DataFrame,
    expected_metrics: dict[str, Any],
    canonical_test_ids: list[str],
) -> None:
    """Reload all Part 7 outputs and validate their content and alignment."""
    predictions = pd.read_csv(
        PREDICTIONS_PATH,
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
    if predictions.columns.tolist() != expected_prediction_columns:
        raise AssertionError("Saved prediction columns differ from the contract")
    if predictions["case_id"].astype(str).tolist() != canonical_test_ids:
        raise AssertionError("Saved predictions do not match canonical test-ID order")
    pd.testing.assert_frame_equal(
        predictions,
        expected_predictions.astype({"case_id": "string"}),
        check_dtype=True,
        check_exact=False,
        rtol=0.0,
        atol=FLOAT_TOLERANCE,
    )

    importances = pd.read_csv(FEATURE_IMPORTANCES_PATH, encoding="utf-8")
    if importances.columns.tolist() != ["feature", "importance"]:
        raise AssertionError("Saved feature-importance columns differ from contract")
    pd.testing.assert_frame_equal(
        importances,
        expected_importances,
        check_dtype=True,
        check_exact=False,
        rtol=0.0,
        atol=FLOAT_TOLERANCE,
    )
    if len(importances) != EXPECTED_FEATURES:
        raise AssertionError("Reloaded importance count differs from 165")
    if not np.isfinite(importances["importance"]).all():
        raise AssertionError("Reloaded importances contain non-finite values")
    if (importances["importance"] < 0).any():
        raise AssertionError("Reloaded importances contain negative values")
    if not np.isclose(
        importances["importance"].sum(),
        1.0,
        rtol=0.0,
        atol=FLOAT_TOLERANCE,
    ):
        raise AssertionError("Reloaded importances do not sum approximately to 1")

    metrics = _load_json_object(METRICS_PATH)
    if metrics != expected_metrics:
        raise AssertionError("Reloaded metrics differ from computed metrics")
    reloaded_metric_values = _metric_values(
        predictions["y_true"].to_numpy(),
        predictions["y_pred"].to_numpy(),
        predictions["y_prob_success"].to_numpy(),
    )
    for name in [
        "roc_auc",
        "average_precision",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]:
        if not np.isclose(
            reloaded_metric_values[name],
            metrics[name],
            rtol=0.0,
            atol=FLOAT_TOLERANCE,
        ):
            raise AssertionError(f"Reloaded {name} differs from saved metrics")
    if reloaded_metric_values["confusion_matrix_values"] != metrics["confusion_matrix"]["values"]:
        raise AssertionError("Reloaded confusion matrix differs from saved metrics")


def run_random_forest_baseline() -> dict[str, Any]:
    """Fit, evaluate, audit, and persist Baseline V1 Part 7 only."""
    preprocessing_manifest = _load_json_object(PREPROCESSING_MANIFEST_PATH)
    split_summary = _load_json_object(SPLIT_SUMMARY_PATH)
    dummy_metrics = _load_json_object(DUMMY_METRICS_PATH)
    logistic_metrics = _load_json_object(LOGISTIC_METRICS_PATH)
    train_ids = _load_case_ids(TRAIN_IDS_PATH, EXPECTED_TRAIN_CASES)
    test_ids = _load_case_ids(TEST_IDS_PATH, EXPECTED_TEST_CASES)
    feature_names = _load_feature_names()
    X_train = sparse.load_npz(X_TRAIN_PATH).tocsr()
    X_test = sparse.load_npz(X_TEST_PATH).tocsr()
    y_train = np.load(Y_TRAIN_PATH, allow_pickle=False)
    y_test = np.load(Y_TEST_PATH, allow_pickle=False)

    _validate_frozen_contract(
        preprocessing_manifest,
        split_summary,
        train_ids,
        test_ids,
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
    )
    protected_paths = _protected_artifact_paths(preprocessing_manifest)
    protected_hashes_before = _artifact_hashes(protected_paths)

    # These are the only fit calls. Both consume frozen X_train_tree/y_train
    # only. X_test is used after fitting; y_test is used only for evaluation.
    classifier = _new_classifier()
    classifier.fit(X_train, y_train)
    _validate_fitted_classifier(classifier)
    positive_column = _positive_probability_column(classifier)

    y_pred = classifier.predict(X_test).astype(np.int8, copy=False)
    probability_matrix = classifier.predict_proba(X_test)
    y_prob_success = probability_matrix[:, positive_column]
    if len(y_pred) != EXPECTED_TEST_CASES:
        raise AssertionError("Prediction length is not 6,276")
    if len(y_prob_success) != EXPECTED_TEST_CASES:
        raise AssertionError("Probability length is not 6,276")
    if not np.isfinite(y_prob_success).all():
        raise AssertionError("Predicted probabilities contain non-finite values")
    if not np.logical_and(y_prob_success >= 0, y_prob_success <= 1).all():
        raise AssertionError("Predicted probabilities fall outside [0, 1]")
    if not set(np.unique(y_pred).tolist()).issubset({0, 1}):
        raise AssertionError("Predicted labels contain a value outside {0, 1}")

    metric_values = _metric_values(y_test, y_pred, y_prob_success)
    depth_summary, leaf_summary = _tree_summaries(classifier)
    importance_frame = _feature_importance_frame(classifier, feature_names)
    importances = importance_frame["importance"].to_numpy()
    top_ten = (
        importance_frame.assign(
            feature_index=np.arange(EXPECTED_FEATURES, dtype=np.int64)
        )
        .sort_values(
            ["importance", "feature_index"],
            ascending=[False, True],
            kind="mergesort",
        )
        .head(10)
    )

    second_classifier = _new_classifier()
    second_classifier.fit(X_train, y_train)
    reproducibility = _assert_reproducible(
        classifier,
        second_classifier,
        X_test,
        y_test,
        y_pred,
        y_prob_success,
        metric_values,
        positive_column,
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
        raise AssertionError("Prediction order differs from canonical test IDs")

    if _artifact_hashes(protected_paths) != protected_hashes_before:
        raise AssertionError("A frozen Part 1-6 artifact changed during Part 7")

    assertions = {
        "A_X_train_tree_shape_25100_by_165": "PASS",
        "B_X_test_tree_shape_6276_by_165": "PASS",
        "C_y_train_length_25100": "PASS",
        "D_y_test_length_6276": "PASS",
        "E_train_target_distribution": "PASS",
        "F_test_target_distribution": "PASS",
        "G_prediction_length_6276": "PASS",
        "H_probability_length_6276": "PASS",
        "I_probabilities_finite": "PASS",
        "J_probabilities_in_unit_interval": "PASS",
        "K_predicted_labels_binary": "PASS",
        "L_positive_probability_column_is_class_1": "PASS",
        "M_prediction_order_matches_test_case_ids": "PASS",
        "N_preprocessing_artifacts_unmodified": "PASS",
        "O_training_used_only_frozen_training_rows_and_labels": "PASS",
        "P_no_test_label_participated_in_fitting": "PASS",
        "Q_class_weight_is_none": "PASS",
        "R_no_hyperparameter_search": "PASS",
        "S_no_cross_validation": "PASS",
        "T_no_classification_threshold_optimization": "PASS",
        "U_n_estimators_is_300": "PASS",
        "V_feature_name_count_is_165": "PASS",
        "W_feature_names_are_unique": "PASS",
    }
    input_paths = [
        X_TRAIN_PATH,
        X_TEST_PATH,
        Y_TRAIN_PATH,
        Y_TEST_PATH,
        TEST_IDS_PATH,
        FEATURE_NAMES_PATH,
        PREPROCESSING_MANIFEST_PATH,
        SPLIT_SUMMARY_PATH,
        DUMMY_METRICS_PATH,
        LOGISTIC_METRICS_PATH,
    ]
    feature_importance_audit = {
        "importance_count": len(importances),
        "all_finite": True,
        "all_non_negative": True,
        "sum": float(importances.sum()),
        "sum_approximately_one": True,
        "minimum": float(importances.min()),
        "maximum": float(importances.max()),
        "zero_importance_feature_count": int(np.count_nonzero(importances == 0.0)),
        "top_10_impurity_importances": [
            {
                "feature": str(row.feature),
                "importance": float(row.importance),
            }
            for row in top_ten.itertuples(index=False)
        ],
        "importance_type": "mean decrease in impurity",
        "causal_interpretation_performed": False,
        "shap_analysis_performed": False,
        "feature_selection_performed": False,
    }
    metrics: dict[str, Any] = {
        "baseline_version": "v1_part7",
        "model": "RandomForestClassifier",
        "parameters": FIXED_PARAMETERS,
        "sklearn_api_adaptation": {
            "applied": False,
            "installed_sklearn_version": sklearn.__version__,
            "effective_constructor_parameters": FIXED_PARAMETERS,
        },
        "fit_scope": "train_only",
        "reported_evaluation_split": "test",
        "input_representation": "unscaled TREE",
        "positive_class": POSITIVE_CLASS,
        "positive_class_name": "Success",
        "classifier_classes": [int(label) for label in classifier.classes_],
        "positive_probability_column_index": positive_column,
        "train_cases": EXPECTED_TRAIN_CASES,
        "test_cases": EXPECTED_TEST_CASES,
        "feature_count": EXPECTED_FEATURES,
        "train_target_counts": {
            str(label): count for label, count in _target_counts(y_train).items()
        },
        "test_target_counts": {
            str(label): count for label, count in _target_counts(y_test).items()
        },
        "n_estimators": len(classifier.estimators_),
        "tree_depth_summary": depth_summary,
        "leaf_count_summary": leaf_summary,
        "roc_auc": metric_values["roc_auc"],
        "average_precision": metric_values["average_precision"],
        "average_precision_label": "PR-AUC (Average Precision)",
        "accuracy": metric_values["accuracy"],
        "precision": metric_values["precision"],
        "recall": metric_values["recall"],
        "f1": metric_values["f1"],
        "confusion_matrix": {
            "labels": CONFUSION_MATRIX_LABELS,
            "rows": "true labels",
            "columns": "predicted labels",
            "values": metric_values["confusion_matrix_values"],
        },
        "classification_decision_rule": (
            "RandomForestClassifier.predict standard decision; no test-derived "
            "threshold adjustment"
        ),
        "threshold_optimization": False,
        "hyperparameter_tuning": False,
        "cross_validation": False,
        "class_weight": None,
        "deltas_vs_dummy": _metric_deltas(
            metric_values,
            dummy_metrics,
            "DummyClassifier",
        ),
        "deltas_vs_logistic": _metric_deltas(
            metric_values,
            logistic_metrics,
            "Logistic Regression",
        ),
        "feature_importance_audit": feature_importance_audit,
        "reproducibility": reproducibility,
        "input_artifacts": {_relative(path): _sha256(path) for path in input_paths},
        "assertions": assertions,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8")
    importance_frame.to_csv(
        FEATURE_IMPORTANCES_PATH,
        index=False,
        encoding="utf-8",
    )
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
        file.write("\n")

    if _artifact_hashes(protected_paths) != protected_hashes_before:
        raise AssertionError("A frozen Part 1-6 artifact changed while saving Part 7")
    _validate_saved_outputs(
        predictions,
        importance_frame,
        metrics,
        test_ids,
    )
    print_report(metrics)
    return metrics


def print_report(metrics: dict[str, Any]) -> None:
    """Print a compact Baseline V1 Part 7 audit report."""
    print("\n=== BASELINE V1 PART 7: RANDOM FOREST ===")
    print(f"X_train_tree shape: ({metrics['train_cases']}, {metrics['feature_count']})")
    print(f"X_test_tree shape: ({metrics['test_cases']}, {metrics['feature_count']})")
    print(f"Train target counts: {metrics['train_target_counts']}")
    print(f"Test target counts: {metrics['test_target_counts']}")
    print(f"Fixed parameters: {metrics['parameters']}")
    print(f"Classifier classes_: {metrics['classifier_classes']}")
    print(f"Fitted estimators: {metrics['n_estimators']}")
    print(f"Tree-depth summary: {metrics['tree_depth_summary']}")
    print(f"Leaf-count summary: {metrics['leaf_count_summary']}")
    print(f"ROC-AUC: {metrics['roc_auc']:.16f}")
    print(f"PR-AUC (Average Precision): {metrics['average_precision']:.16f}")
    print(f"Accuracy: {metrics['accuracy']:.16f}")
    print(f"Precision: {metrics['precision']:.16f}")
    print(f"Recall: {metrics['recall']:.16f}")
    print(f"F1: {metrics['f1']:.16f}")
    print("Confusion matrix labels: [0, 1]")
    print("Rows=true labels; columns=predicted labels")
    print(np.asarray(metrics["confusion_matrix"]["values"]))
    print(f"Deltas versus Dummy: {metrics['deltas_vs_dummy']}")
    print(f"Deltas versus Logistic: {metrics['deltas_vs_logistic']}")
    print(f"Feature-importance audit: {metrics['feature_importance_audit']}")
    print(f"Reproducibility: {metrics['reproducibility']['result']}")
    print("\n=== REQUIRED ASSERTIONS ===")
    for name, result in metrics["assertions"].items():
        print(f"{name}: {result}")
    print("\nAll Baseline V1 Part 7 assertions passed.")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved predictions: {PREDICTIONS_PATH}")
    print(f"Saved feature importances: {FEATURE_IMPORTANCES_PATH}")


def main() -> None:
    """Run Baseline V1 Part 7 and stop before all later analysis."""
    run_random_forest_baseline()


if __name__ == "__main__":
    main()
