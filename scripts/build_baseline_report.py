"""Build the consolidated Baseline V1 Part 8 report from frozen artifacts.

This script audits and reads the artifacts produced by Parts 1--7. It does not
load a fitted model, fit a model or preprocessor, select features, tune a
threshold, or alter any input artifact. All evaluation metrics and plot data
are recomputed directly from the three saved test-prediction files.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "baseline_v1"
EDA_TARGET_DISTRIBUTION_PATH = PROJECT_ROOT / "outputs" / "eda" / "target_distribution.csv"

ELIGIBILITY_PATH = RESULTS_DIR / "eligibility_summary.json"
FEATURE_MANIFEST_PATH = RESULTS_DIR / "feature_manifest.json"
SPLIT_SUMMARY_PATH = RESULTS_DIR / "split_summary.json"
PREPROCESSING_MANIFEST_PATH = RESULTS_DIR / "preprocessing_manifest.json"
TEST_CASE_IDS_PATH = RESULTS_DIR / "test_case_ids.csv"

METRICS_CSV_PATH = RESULTS_DIR / "baseline_metrics.csv"
METRICS_JSON_PATH = RESULTS_DIR / "baseline_metrics.json"
ROC_CURVE_PATH = RESULTS_DIR / "roc_curve.png"
PR_CURVE_PATH = RESULTS_DIR / "pr_curve.png"
REPORT_PATH = RESULTS_DIR / "baseline_v1.md"

EXPECTED_ROWS = 6_276
EXPECTED_COLUMNS = ["case_id", "y_true", "y_pred", "y_prob_success"]
EXPECTED_TARGET_COUNTS = {0: 2_826, 1: 3_450}
METRIC_NAMES = [
    "roc_auc",
    "average_precision",
    "accuracy",
    "precision",
    "recall",
    "f1",
]
NUMERICAL_TOLERANCE = 1e-12

MODEL_SPECS = (
    {
        "key": "dummy",
        "display_name": "Dummy",
        "metrics_path": RESULTS_DIR / "dummy_metrics.json",
        "predictions_path": RESULTS_DIR / "dummy_predictions.csv",
        "confusion_path": RESULTS_DIR / "confusion_matrix_dummy.png",
        "color": "#6b7280",
    },
    {
        "key": "logistic_regression",
        "display_name": "Logistic Regression",
        "metrics_path": RESULTS_DIR / "logistic_metrics.json",
        "predictions_path": RESULTS_DIR / "logistic_predictions.csv",
        "confusion_path": RESULTS_DIR / "confusion_matrix_logistic.png",
        "color": "#2563eb",
    },
    {
        "key": "random_forest",
        "display_name": "Random Forest",
        "metrics_path": RESULTS_DIR / "random_forest_metrics.json",
        "predictions_path": RESULTS_DIR / "random_forest_predictions.csv",
        "confusion_path": RESULTS_DIR / "confusion_matrix_random_forest.png",
        "color": "#059669",
    },
)

EXPECTED_METRICS = {
    "dummy": {
        "roc_auc": 0.5,
        "average_precision": 0.5497131931166348,
        "accuracy": 0.5497131931166348,
        "precision": 0.5497131931166348,
        "recall": 1.0,
        "f1": 0.7094386181369525,
    },
    "logistic_regression": {
        "roc_auc": 0.6002674954101151,
        "average_precision": 0.6425684064989117,
        "accuracy": 0.5811026131293817,
        "precision": 0.5966564633859195,
        "recall": 0.7344927536231884,
        "f1": 0.6584383526049110,
    },
    "random_forest": {
        "roc_auc": 0.5875780793255176,
        "average_precision": 0.6301435500864094,
        "accuracy": 0.5678776290630975,
        "precision": 0.5975674246430460,
        "recall": 0.6550724637681159,
        "f1": 0.625,
    },
}


def _relative(path: Path) -> str:
    """Return a repository-relative path with portable separators."""
    return path.relative_to(PROJECT_ROOT).as_posix()


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a required file."""
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load a required JSON object."""
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _assert_equal(actual: Any, expected: Any, description: str) -> None:
    """Raise a descriptive error when an exact contract differs."""
    if actual != expected:
        raise AssertionError(f"{description}: expected {expected!r}, got {actual!r}")


def _assert_close(actual: float, expected: float, description: str) -> None:
    """Compare finite numbers at the report's strict numerical tolerance."""
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise AssertionError(f"{description}: compared values must be finite")
    if not math.isclose(
        actual,
        expected,
        rel_tol=NUMERICAL_TOLERANCE,
        abs_tol=NUMERICAL_TOLERANCE,
    ):
        raise AssertionError(
            f"{description}: expected {expected:.17g}, got {actual:.17g}"
        )


def _load_case_ids() -> list[str]:
    """Load the canonical test IDs without changing their order."""
    if not TEST_CASE_IDS_PATH.is_file():
        raise FileNotFoundError(f"Required artifact not found: {TEST_CASE_IDS_PATH}")
    frame = pd.read_csv(
        TEST_CASE_IDS_PATH,
        encoding="utf-8",
        dtype={"case_id": "string"},
    )
    _assert_equal(frame.columns.tolist(), ["case_id"], "Test-ID columns")
    _assert_equal(len(frame), EXPECTED_ROWS, "Test-ID row count")
    if frame["case_id"].isna().any():
        raise AssertionError("Canonical test case IDs contain missing values")
    if not frame["case_id"].is_unique:
        raise AssertionError("Canonical test case IDs contain duplicates")
    return frame["case_id"].astype(str).tolist()


def _load_predictions(path: Path, canonical_ids: list[str]) -> pd.DataFrame:
    """Load and fully audit one frozen prediction file."""
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8", dtype={"case_id": "string"})
    _assert_equal(frame.columns.tolist(), EXPECTED_COLUMNS, f"Columns in {path.name}")
    _assert_equal(len(frame), EXPECTED_ROWS, f"Row count in {path.name}")

    if frame.isna().any().any():
        raise AssertionError(f"Missing values found in {path.name}")
    ids = frame["case_id"].astype(str).tolist()
    _assert_equal(ids, canonical_ids, f"Case-ID ordering in {path.name}")
    if not frame["case_id"].is_unique:
        raise AssertionError(f"Duplicate case IDs found in {path.name}")

    for column in ("y_true", "y_pred", "y_prob_success"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in ("y_true", "y_pred"):
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise AssertionError(f"{column} must contain finite integer labels in {path.name}")
        frame[column] = frame[column].astype(np.int8)

    if set(frame["y_true"].unique().tolist()) != {0, 1}:
        raise AssertionError(f"y_true labels differ from {{0, 1}} in {path.name}")
    if not set(frame["y_pred"].unique().tolist()).issubset({0, 1}):
        raise AssertionError(f"y_pred labels differ from {{0, 1}} in {path.name}")

    probabilities = frame["y_prob_success"].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise AssertionError(f"Non-finite probabilities found in {path.name}")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise AssertionError(f"Probabilities outside [0, 1] found in {path.name}")

    counts = frame["y_true"].value_counts().sort_index().to_dict()
    _assert_equal(counts, EXPECTED_TARGET_COUNTS, f"Target distribution in {path.name}")
    return frame


def _compute_metrics(frame: pd.DataFrame) -> tuple[dict[str, float], list[list[int]]]:
    """Recompute all Baseline V1 metrics from saved predictions."""
    y_true = frame["y_true"].to_numpy(dtype=np.int8)
    y_pred = frame["y_pred"].to_numpy(dtype=np.int8)
    y_prob = frame["y_prob_success"].to_numpy(dtype=float)
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int).tolist()
    return metrics, matrix


def _validate_source_contracts(
    eligibility: dict[str, Any],
    feature_manifest: dict[str, Any],
    split: dict[str, Any],
    preprocessing: dict[str, Any],
) -> None:
    """Validate population, leakage, split, and preprocessing metadata."""
    eligibility_expected = {
        "prediction_point": 10,
        "total_cases": 31_509,
        "valid_target_cases": 31_411,
        "unresolved_cases": 98,
        "cases_lt_k_events": 0,
        "outcome_at_or_before_k": 35,
        "eligible_cases": 31_376,
        "eligible_success": 17_228,
        "eligible_unsuccessful": 14_148,
        "outcome_leakage_pass": True,
        "future_event_leakage_pass": True,
    }
    for key, expected in eligibility_expected.items():
        _assert_equal(eligibility.get(key), expected, f"Eligibility field {key}")

    _assert_equal(feature_manifest.get("prediction_point"), 10, "Feature prediction point")
    _assert_equal(feature_manifest.get("number_of_cases"), 31_376, "Feature-table cases")
    feature_validation = feature_manifest.get("validation", {})
    _assert_equal(
        feature_validation.get("part1_outcome_leakage_audit"),
        "PASS",
        "Outcome leakage audit",
    )
    _assert_equal(
        feature_validation.get("part1_future_event_leakage_audit"),
        "PASS",
        "Future-event leakage audit",
    )

    split_expected = {
        "split_type": "temporal_holdout",
        "prediction_point": 10,
        "train_cases": 25_100,
        "test_cases": 6_276,
        "train_success": 13_778,
        "train_unsuccessful": 11_322,
        "test_success": 3_450,
        "test_unsuccessful": 2_826,
        "train_start": "2016-01-01T09:51:15.304000+00:00",
        "train_end": "2016-10-18T11:25:01.193000+00:00",
        "test_start": "2016-10-18T11:25:16.948000+00:00",
        "test_end": "2016-12-31T21:37:53.216000+00:00",
        "strict_temporal_boundary": True,
    }
    for key, expected in split_expected.items():
        _assert_equal(split.get(key), expected, f"Split field {key}")
    _assert_equal(
        split.get("class_mapping"),
        {"Success": 1, "Unsuccessful": 0},
        "Class mapping",
    )

    _assert_equal(
        preprocessing.get("preprocessing_fit_scope"),
        "train_only",
        "Preprocessing fit scope",
    )
    _assert_equal(
        preprocessing.get("split_sizes"),
        {"train": 25_100, "test": 6_276},
        "Preprocessing split sizes",
    )
    for representation in ("linear", "tree"):
        dimension = preprocessing.get("representations", {}).get(representation, {}).get(
            "feature_dimension"
        )
        _assert_equal(dimension, 165, f"{representation} transformed dimension")
    _assert_equal(
        preprocessing.get("train_fitted_sizes", {}).get("numeric_features"),
        5,
        "Numeric feature count",
    )
    _assert_equal(
        preprocessing.get("test_only_unknowns", {}).get("resource", {}).get(
            "test_only_count"
        ),
        9,
        "Test-only resource-key count",
    )
    _assert_equal(
        preprocessing.get("zero_variance_train_features", {}).get("features"),
        [
            {
                "feature_name": "activity::A_Create Application",
                "source_group": "structured_map::activity",
            }
        ],
        "Retained zero-variance train feature",
    )


def _validate_early_outcome_class() -> None:
    """Confirm the stored evidence that all 35 early outcomes were unsuccessful."""
    if not EDA_TARGET_DISTRIBUTION_PATH.is_file():
        raise FileNotFoundError(
            "Stored target distribution needed to confirm the early-outcome class "
            f"was not found: {EDA_TARGET_DISTRIBUTION_PATH}"
        )
    frame = pd.read_csv(EDA_TARGET_DISTRIBUTION_PATH, encoding="utf-8")
    _assert_equal(
        frame.columns.tolist(),
        ["target", "case_count", "percent"],
        "EDA target-distribution columns",
    )
    counts = dict(zip(frame["target"], frame["case_count"], strict=True))
    _assert_equal(counts.get("A_Pending"), 17_228, "Valid Success count")
    unsuccessful = int(counts.get("A_Cancelled", -1)) + int(counts.get("A_Denied", -1))
    _assert_equal(unsuccessful, 14_183, "Valid Unsuccessful count")
    _assert_equal(unsuccessful - 14_148, 35, "Excluded Unsuccessful count")


def _audit_predictions_and_metrics(
    canonical_ids: list[str],
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, dict[str, float]],
    dict[str, list[list[int]]],
    dict[str, dict[str, Any]],
]:
    """Audit prediction identity and exact agreement with stored metrics."""
    predictions: dict[str, pd.DataFrame] = {}
    recomputed: dict[str, dict[str, float]] = {}
    matrices: dict[str, list[list[int]]] = {}
    stored_metrics: dict[str, dict[str, Any]] = {}

    reference_y_true: np.ndarray | None = None
    for spec in MODEL_SPECS:
        key = str(spec["key"])
        frame = _load_predictions(Path(spec["predictions_path"]), canonical_ids)
        stored = _load_json_object(Path(spec["metrics_path"]))
        metrics, matrix = _compute_metrics(frame)

        y_true = frame["y_true"].to_numpy(dtype=np.int8)
        if reference_y_true is None:
            reference_y_true = y_true.copy()
        elif not np.array_equal(y_true, reference_y_true):
            raise AssertionError(f"y_true differs across models for {key}")

        for metric_name, value in metrics.items():
            stored_value = stored.get(metric_name)
            if not isinstance(stored_value, (int, float)):
                raise ValueError(f"Missing numeric {metric_name} in {spec['metrics_path']}")
            _assert_close(
                value,
                float(stored_value),
                f"Recomputed vs stored {key} {metric_name}",
            )
            _assert_close(
                value,
                EXPECTED_METRICS[key][metric_name],
                f"Recomputed vs validated {key} {metric_name}",
            )

        stored_matrix = stored.get("confusion_matrix", {}).get("values")
        _assert_equal(matrix, stored_matrix, f"Recomputed vs stored {key} confusion matrix")
        _assert_equal(
            stored.get("confusion_matrix", {}).get("labels"),
            [0, 1],
            f"Stored {key} confusion labels",
        )
        predictions[key] = frame
        recomputed[key] = metrics
        matrices[key] = matrix
        stored_metrics[key] = stored

    _assert_equal(
        set(predictions["dummy"]["y_pred"].unique().tolist()),
        {1},
        "Dummy predicted-label set",
    )
    _assert_equal(
        set(predictions["logistic_regression"]["y_pred"].unique().tolist()),
        {0, 1},
        "Logistic Regression predicted-label set",
    )
    _assert_equal(
        set(predictions["random_forest"]["y_pred"].unique().tolist()),
        {0, 1},
        "Random Forest predicted-label set",
    )
    return predictions, recomputed, matrices, stored_metrics


def _validate_model_contracts(stored: dict[str, dict[str, Any]]) -> None:
    """Validate the fixed, untuned configurations used by Parts 5--7."""
    dummy = stored["dummy"]
    _assert_equal(dummy.get("model_name"), "DummyClassifier", "Dummy model name")
    _assert_equal(
        dummy.get("model_parameters", {}).get("strategy"), "prior", "Dummy strategy"
    )

    logistic = stored["logistic_regression"]
    logistic_parameters = logistic.get("parameters", {})
    for key, expected in {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "class_weight": None,
    }.items():
        _assert_equal(logistic_parameters.get(key), expected, f"Logistic parameter {key}")

    forest = stored["random_forest"]
    forest_parameters = forest.get("parameters", {})
    for key, expected in {
        "n_estimators": 300,
        "criterion": "gini",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "class_weight": None,
        "random_state": 42,
    }.items():
        _assert_equal(forest_parameters.get(key), expected, f"Random Forest parameter {key}")
    _assert_equal(
        forest.get("tree_depth_summary"),
        {"minimum": 79, "mean": 99.52, "maximum": 136},
        "Random Forest depth summary",
    )
    _assert_close(
        float(forest.get("leaf_count_summary", {}).get("mean")),
        9_373.73,
        "Random Forest mean leaf count",
    )

    for key, model_metrics in stored.items():
        _assert_equal(model_metrics.get("fit_scope"), "train_only", f"{key} fit scope")
        _assert_equal(
            model_metrics.get("reported_evaluation_split"),
            "test",
            f"{key} evaluation split",
        )
        _assert_equal(model_metrics.get("positive_class"), 1, f"{key} positive class")
        _assert_equal(model_metrics.get("test_cases"), EXPECTED_ROWS, f"{key} test cases")


def _metrics_frame(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Create the fixed-order consolidated metrics table."""
    return pd.DataFrame(
        [
            {"model": spec["key"], **metrics[str(spec["key"])]}
            for spec in MODEL_SPECS
        ],
        columns=["model", *METRIC_NAMES],
    )


def _plot_roc(
    predictions: dict[str, pd.DataFrame],
    metrics: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """Plot ROC curves using only frozen saved predictions."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for spec in MODEL_SPECS:
        key = str(spec["key"])
        frame = predictions[key]
        fpr, tpr, _ = roc_curve(frame["y_true"], frame["y_prob_success"])
        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            color=str(spec["color"]),
            label=f"{spec['display_name']} (ROC-AUC = {metrics[key]['roc_auc']:.4f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="black", label="Random classification")
    ax.set(title="Baseline V1 ROC Curves", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", format="png")
    plt.close(fig)


def _plot_pr(
    predictions: dict[str, pd.DataFrame],
    metrics: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """Plot PR curves and label the non-trapezoidal Average Precision metric."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for spec in MODEL_SPECS:
        key = str(spec["key"])
        frame = predictions[key]
        precision, recall, _ = precision_recall_curve(
            frame["y_true"], frame["y_prob_success"]
        )
        ax.plot(
            recall,
            precision,
            linewidth=2,
            color=str(spec["color"]),
            label=f"{spec['display_name']} (AP = {metrics[key]['average_precision']:.4f})",
        )
    prevalence = EXPECTED_TARGET_COUNTS[1] / EXPECTED_ROWS
    ax.axhline(
        prevalence,
        linestyle="--",
        linewidth=1.5,
        color="black",
        label=f"Positive prevalence = 3450 / 6276 = {prevalence:.4f}",
    )
    ax.set(
        title="Baseline V1 Precision-Recall Curves",
        xlabel="Recall",
        ylabel="Precision",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    ax.text(
        0.99,
        0.02,
        "AP = Average Precision (not trapezoidal PR area)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", format="png")
    plt.close(fig)


def _plot_confusion_matrix(
    matrix: list[list[int]], display_name: str, output_path: Path
) -> None:
    """Plot a raw-count confusion matrix with consistent conventions."""
    values = np.asarray(matrix, dtype=int)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    image = ax.imshow(values, interpolation="nearest", cmap="Blues", vmin=0, vmax=3450)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Case count")
    ax.set(
        title=f"{display_name} Confusion Matrix (Raw Counts)",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["0", "1"],
        yticklabels=["0", "1"],
    )
    threshold = 3450 / 2
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                f"{values[row, column]:,}",
                ha="center",
                va="center",
                color="white" if values[row, column] > threshold else "black",
                fontsize=12,
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", format="png")
    plt.close(fig)


def _render_report(metrics: dict[str, dict[str, float]]) -> str:
    """Render the human-readable Baseline V1 report."""
    metric_rows = []
    for spec in MODEL_SPECS:
        key = str(spec["key"])
        values = metrics[key]
        metric_rows.append(
            "| "
            + str(spec["display_name"])
            + " | "
            + " | ".join(f"{values[name]:.4f}" for name in METRIC_NAMES)
            + " |"
        )
    table = "\n".join(
        [
            "| Model | ROC-AUC | Average Precision | Accuracy | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *metric_rows,
        ]
    )

    return f"""# Baseline Evaluation V1

## 1. Objective

The purpose of Baseline V1 is to establish reference predictive performance at prediction point $k=10$ before model tuning and explainable-AI (XAI) analysis. This reporting layer consolidates the frozen outputs from Parts 1–7; it does not retrain or refit anything.

## 2. Evaluation Population

- Total BPIC17 cases: 31,509
- Valid target cases: 31,411
- Unresolved cases: 98
- Cases with fewer than 10 events: 0
- Cases excluded because the outcome occurred at or before $k=10$: 35
- Eligible cases: 31,376
- Eligible Success: 17,228
- Eligible Unsuccessful: 14,148

All 35 outcome-at-or-before-$k$ exclusions came from the Unsuccessful class. This is confirmed by the stored valid-target counts: 17,228 Success and 14,183 Unsuccessful cases before the timing exclusion, compared with 17,228 Success and 14,148 Unsuccessful eligible cases afterward.

## 3. Leakage Controls

The recorded outcome-leakage audit and future-event-leakage audit both passed. Features were restricted to the observed prefix, events 1 through 10. Vocabulary and preprocessing were fitted on training data only, and test-only resources did not expand the fitted feature space. These statements summarize the existing audits; this report does not perform or claim additional leakage analyses.

## 4. Features

The model representations contain five numeric features, one-hot categories for `LoanGoal` and `ApplicationType`, and count features for activities, `Action`, `EventOrigin`, resources, and lifecycle transitions. The final transformed dimension is 165 features.

Vocabulary was learned from training data only. Nine resource keys occurred only in the test set and were ignored by the fitted vocabulary. One zero-variance training feature was retained: `activity::A_Create Application`.

## 5. Temporal Split

The split used no shuffle and no stratification, with a strict chronological boundary.

| Split | Cases | Start | End | Success (1) | Unsuccessful (0) |
|---|---:|---|---|---:|---:|
| Train | 25,100 | 2016-01-01T09:51:15.304000+00:00 | 2016-10-18T11:25:01.193000+00:00 | 13,778 | 11,322 |
| Test | 6,276 | 2016-10-18T11:25:16.948000+00:00 | 2016-12-31T21:37:53.216000+00:00 | 3,450 | 2,826 |

The small prevalence difference between these periods is descriptive and is not evidence of no drift.

## 6. Models

- **DummyClassifier:** `strategy="prior"`; it learns the training class prior and uses its standard prediction rule.
- **Logistic Regression:** fixed L2 semantics, $C=1.0$, `solver="lbfgs"`, and `class_weight=None`.
- **Random Forest:** 300 estimators, Gini criterion, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, `max_features="sqrt"`, bootstrap sampling, `class_weight=None`, and `random_state=42`.

There was no hyperparameter tuning, cross-validation, class rebalancing, or threshold optimization.

## 7. Metrics

The primary metric is ROC-AUC. Secondary metrics are Average Precision, accuracy, precision, recall, F1, and the confusion matrix. The positive class is Success = 1.

Average Precision is reported as AP; it is not a trapezoidal area computed from the plotted precision-recall curve.

## 8. Results

{table}

The full-precision values are preserved in [`baseline_metrics.csv`](baseline_metrics.csv) and [`baseline_metrics.json`](baseline_metrics.json).

- [`roc_curve.png`](roc_curve.png)
- [`pr_curve.png`](pr_curve.png)
- Confusion matrices: [`Dummy`](confusion_matrix_dummy.png), [`Logistic Regression`](confusion_matrix_logistic.png), and [`Random Forest`](confusion_matrix_random_forest.png)

All figures use the same 6,276-case frozen test population and were produced only from the saved prediction files.

## 9. Baseline Interpretation

The Dummy model predicts every test case as Success because Success is the majority training class. Its Success recall is therefore 1.0, and its F1 of {metrics['dummy']['f1']:.4f} can appear high. However, its constant scores provide no ranking discrimination, so ROC-AUC remains 0.5. Dummy F1 alone is not evidence of predictive quality.

Logistic Regression produced non-trivial predictions for both classes. Its ROC-AUC ({metrics['logistic_regression']['roc_auc']:.4f}) and Average Precision ({metrics['logistic_regression']['average_precision']:.4f}) both exceed the Dummy values.

Random Forest also exceeds Dummy in ROC-AUC ({metrics['random_forest']['roc_auc']:.4f}) and Average Precision ({metrics['random_forest']['average_precision']:.4f}). In this fixed baseline, its metrics differ descriptively from Logistic Regression; no tuning or model selection follows from that observation.

The untuned forest is highly complex: the previously validated tree depths range from 79 to 136, with a mean of 99.52, and each tree has approximately 9.3 thousand leaves on average (9,373.73). This is evidence of structural complexity, not proof of overfitting by itself.

These results are untuned baseline estimates on one frozen temporal holdout. They are not final optimized model performance, cross-validated estimates, SHAP results, or concept-drift conclusions. No final research model is designated here.

## 10. Limitations and Next Steps

Baseline V1 uses one temporal holdout and fixed untuned configurations. It has not performed cross-validation or tuning, has not evaluated temporal explanation stability, and has not run SHAP. Those activities are outside Part 8 and are not implemented by this reporting task.
"""


def _json_payload(
    metrics: dict[str, dict[str, float]], source_hashes: dict[str, str]
) -> dict[str, Any]:
    """Build deterministic machine-readable output with experiment metadata."""
    return {
        "baseline_version": "v1_part8",
        "prediction_point": 10,
        "positive_class": 1,
        "positive_class_name": "Success",
        "evaluation_split": "frozen_temporal_test_holdout",
        "model_order": [str(spec["key"]) for spec in MODEL_SPECS],
        "test_population": {
            "cases": EXPECTED_ROWS,
            "success": EXPECTED_TARGET_COUNTS[1],
            "unsuccessful": EXPECTED_TARGET_COUNTS[0],
        },
        "metric_definitions": {
            "primary": "roc_auc",
            "secondary": [
                "average_precision",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "confusion_matrix",
            ],
            "average_precision": "scikit-learn average_precision_score; not trapezoidal PR area",
        },
        "metrics": [
            {"model": str(spec["key"]), **metrics[str(spec["key"])]}
            for spec in MODEL_SPECS
        ],
        "cross_artifact_audit": {
            "status": "PASS",
            "numerical_tolerance": NUMERICAL_TOLERANCE,
            "prediction_rows_per_model": EXPECTED_ROWS,
            "case_id_order_identical": True,
            "y_true_identical": True,
            "target_distribution_verified": True,
            "probabilities_finite_and_in_unit_interval": True,
            "predicted_labels_binary": True,
            "duplicate_case_ids": 0,
            "recomputed_metrics_match_stored_metrics": True,
            "plot_inputs": "saved prediction CSV files only",
        },
        "experiment_scope": {
            "hyperparameter_tuning": False,
            "cross_validation": False,
            "class_rebalancing": False,
            "threshold_optimization": False,
            "shap": False,
            "final_model_selection": False,
        },
        "source_artifact_sha256": source_hashes,
    }


def _validate_staged_outputs(
    csv_path: Path,
    json_path: Path,
    report_path: Path,
    plot_paths: list[Path],
    expected_frame: pd.DataFrame,
) -> None:
    """Check CSV/JSON/report agreement and that every plot is non-empty."""
    csv_frame = pd.read_csv(csv_path, encoding="utf-8", float_precision="round_trip")
    _assert_equal(csv_frame.columns.tolist(), ["model", *METRIC_NAMES], "Output CSV columns")
    _assert_equal(csv_frame["model"].tolist(), expected_frame["model"].tolist(), "Output model order")
    for metric_name in METRIC_NAMES:
        for row_index in range(len(expected_frame)):
            actual = float(csv_frame.loc[row_index, metric_name])
            expected = float(expected_frame.loc[row_index, metric_name])
            _assert_equal(
                actual.hex(),
                expected.hex(),
                f"Exact CSV verification {metric_name} row {row_index}",
            )

    payload = _load_json_object(json_path)
    json_frame = pd.DataFrame(payload.get("metrics"), columns=["model", *METRIC_NAMES])
    _assert_equal(json_frame["model"].tolist(), csv_frame["model"].tolist(), "JSON model order")
    for metric_name in METRIC_NAMES:
        for row_index in range(len(csv_frame)):
            json_value = float(json_frame.loc[row_index, metric_name])
            csv_value = float(csv_frame.loc[row_index, metric_name])
            _assert_equal(
                json_value.hex(),
                csv_value.hex(),
                f"Exact JSON vs CSV {metric_name} row {row_index}",
            )

    report = report_path.read_text(encoding="utf-8")
    for _, row in expected_frame.iterrows():
        for metric_name in METRIC_NAMES:
            formatted = f"{float(row[metric_name]):.4f}"
            if formatted not in report:
                raise AssertionError(
                    f"Report does not contain displayed {metric_name} value {formatted}"
                )
    for path in plot_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Plot was not created or is empty: {path}")


def main() -> None:
    """Audit frozen artifacts and publish all Part 8 reporting outputs."""
    eligibility = _load_json_object(ELIGIBILITY_PATH)
    feature_manifest = _load_json_object(FEATURE_MANIFEST_PATH)
    split = _load_json_object(SPLIT_SUMMARY_PATH)
    preprocessing = _load_json_object(PREPROCESSING_MANIFEST_PATH)
    _validate_source_contracts(eligibility, feature_manifest, split, preprocessing)
    _validate_early_outcome_class()

    canonical_ids = _load_case_ids()
    predictions, metrics, matrices, stored_metrics = _audit_predictions_and_metrics(
        canonical_ids
    )
    _validate_model_contracts(stored_metrics)

    source_paths = [
        ELIGIBILITY_PATH,
        FEATURE_MANIFEST_PATH,
        SPLIT_SUMMARY_PATH,
        PREPROCESSING_MANIFEST_PATH,
        TEST_CASE_IDS_PATH,
        EDA_TARGET_DISTRIBUTION_PATH,
        *[Path(spec["metrics_path"]) for spec in MODEL_SPECS],
        *[Path(spec["predictions_path"]) for spec in MODEL_SPECS],
    ]
    source_hashes = {_relative(path): _sha256(path) for path in source_paths}
    frame = _metrics_frame(metrics)
    payload = _json_payload(metrics, source_hashes)
    report = _render_report(metrics)

    output_paths = [
        METRICS_CSV_PATH,
        METRICS_JSON_PATH,
        ROC_CURVE_PATH,
        PR_CURVE_PATH,
        *[Path(spec["confusion_path"]) for spec in MODEL_SPECS],
        REPORT_PATH,
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    staged = {
        path: path.with_name(f".{path.name}.part8.tmp") for path in output_paths
    }
    try:
        frame.to_csv(
            staged[METRICS_CSV_PATH],
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )
        staged[METRICS_JSON_PATH].write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        staged[REPORT_PATH].write_text(report, encoding="utf-8", newline="\n")
        _plot_roc(predictions, metrics, staged[ROC_CURVE_PATH])
        _plot_pr(predictions, metrics, staged[PR_CURVE_PATH])
        for spec in MODEL_SPECS:
            key = str(spec["key"])
            destination = Path(spec["confusion_path"])
            _plot_confusion_matrix(
                matrices[key], str(spec["display_name"]), staged[destination]
            )

        staged_plots = [staged[path] for path in output_paths if path.suffix == ".png"]
        _validate_staged_outputs(
            staged[METRICS_CSV_PATH],
            staged[METRICS_JSON_PATH],
            staged[REPORT_PATH],
            staged_plots,
            frame,
        )
        for destination in output_paths:
            os.replace(staged[destination], destination)
    finally:
        for temporary_path in staged.values():
            temporary_path.unlink(missing_ok=True)

    print("Cross-artifact audit: PASS")
    print(f"Prediction rows per model: {EXPECTED_ROWS}")
    print("Case-ID order and y_true identity across models: PASS")
    print(f"Metric recomputation agreement (tolerance {NUMERICAL_TOLERANCE:g}): PASS")
    print(frame.to_string(index=False))
    print("Created reporting outputs:")
    for path in output_paths:
        print(f"- {_relative(path)}")


if __name__ == "__main__":
    main()
