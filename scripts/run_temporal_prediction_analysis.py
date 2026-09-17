"""Run Temporal Evaluation V1 Part 2 from frozen prediction artifacts.

This module joins the already-saved Random Forest predictions to the frozen
Part 1 temporal membership, reproduces the full-TEST Baseline V1 metrics, and
then computes descriptive prediction behavior for T1, T2, and T3.  It never
loads or fits a model or preprocessor, changes a threshold or boundary, or
performs SHAP, drift, or significance analysis.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
DESIGN_PATH = PROJECT_ROOT / "docs" / "temporal_evaluation_design_v1.md"

BASELINE_RESULTS_DIR = PROJECT_ROOT / "results" / "baseline_v1"
PREDICTIONS_PATH = BASELINE_RESULTS_DIR / "random_forest_predictions.csv"
BASELINE_METRICS_PATH = BASELINE_RESULTS_DIR / "random_forest_metrics.json"
TEST_IDS_PATH = BASELINE_RESULTS_DIR / "test_case_ids.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "temporal_eval_v1"
TEMPORAL_WINDOWS_PATH = OUTPUT_DIR / "temporal_windows.csv"
TEMPORAL_SHAP_WINDOWS_PATH = OUTPUT_DIR / "temporal_shap_windows.csv"
WINDOW_SUMMARY_PATH = OUTPUT_DIR / "temporal_window_summary.json"
METRICS_PATH = OUTPUT_DIR / "temporal_prediction_metrics.csv"
DELTAS_PATH = OUTPUT_DIR / "temporal_prediction_deltas.csv"
SUMMARY_PATH = OUTPUT_DIR / "temporal_prediction_summary.json"
METRICS_PLOT_PATH = OUTPUT_DIR / "temporal_prediction_metrics.png"
PROBABILITY_PLOT_PATH = OUTPUT_DIR / "temporal_probability.png"

EXPECTED_TEST_CASES = 6_276
EXPECTED_CASES_PER_WINDOW = 2_092
POSITIVE_CLASS = 1
WINDOW_NAMES = ("T1", "T2", "T3")
WINDOW_TARGET_COUNTS = {
    "T1": {0: 930, 1: 1_162},
    "T2": {0: 936, 1: 1_156},
    "T3": {0: 960, 1: 1_132},
}
COMPARISONS = (
    ("T1_to_T2", "T1", "T2"),
    ("T2_to_T3", "T2", "T3"),
    ("T1_to_T3", "T1", "T3"),
)
METRIC_TOLERANCE = 1e-12

EXPECTED_FULL_TEST_METRICS: dict[str, float] = {
    "roc_auc": 0.5875780793255176,
    "average_precision": 0.6301435500864094,
    "accuracy": 0.5678776290630975,
    "precision": 0.5975674246430460,
    "recall": 0.6550724637681159,
    "f1": 0.625,
}
EXPECTED_CONFUSION_MATRIX = [[1_304, 1_522], [1_190, 2_260]]

METRICS_COLUMNS = [
    "window",
    "n_cases",
    "success_count",
    "unsuccessful_count",
    "success_rate",
    "mean_probability_success",
    "median_probability_success",
    "std_probability_success",
    "roc_auc",
    "average_precision",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "tn",
    "fp",
    "fn",
    "tp",
]
DELTA_COLUMNS = [
    "comparison",
    "delta_success_rate",
    "delta_mean_probability_success",
    "delta_roc_auc",
    "delta_average_precision",
    "delta_accuracy",
    "delta_precision",
    "delta_recall",
    "delta_f1",
]


def _require(condition: bool, message: str) -> None:
    """Raise a stable validation error when an invariant is false."""
    if not condition:
        raise AssertionError(message)


def _relative(path: Path) -> str:
    """Return a repository-relative path with stable POSIX separators."""
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
    """Collect Part 1, Baseline V1, and SHAP Pilot V1 frozen artifacts."""
    protected_roots = (
        BASELINE_RESULTS_DIR,
        PROJECT_ROOT / "results" / "shap_pilot_v1",
        PROJECT_ROOT / "artifacts" / "baseline_v1",
        PROJECT_ROOT / "artifacts" / "shap_pilot_v1",
        PROJECT_ROOT / "data" / "processed" / "baseline_k10",
        PROJECT_ROOT / "data" / "processed" / "shap_pilot_v1",
    )
    paths = {
        path
        for root in protected_roots
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_file()
    }
    paths.update(
        {
            DESIGN_PATH,
            TEMPORAL_WINDOWS_PATH,
            TEMPORAL_SHAP_WINDOWS_PATH,
            WINDOW_SUMMARY_PATH,
        }
    )
    missing = sorted(
        (path for path in paths if not path.is_file()), key=lambda path: str(path)
    )
    if missing:
        raise FileNotFoundError(f"Protected artifacts are missing: {missing}")
    return sorted(paths, key=_relative)


def protect_artifact_hashes(paths: Iterable[Path]) -> dict[Path, str]:
    """Hash frozen inputs so before/after byte identity can be enforced."""
    return {path: _sha256(path) for path in paths}


def _validate_part_1_hash_contract(window_summary: dict[str, Any]) -> None:
    """Validate current frozen inputs against hashes recorded by Part 1."""
    generated = window_summary.get("generated_artifact_sha256")
    protected = window_summary.get("protected_artifacts", {}).get("sha256")
    _require(isinstance(generated, dict), "Part 1 generated hashes are missing")
    _require(isinstance(protected, dict), "Part 1 protected hashes are missing")

    expected_hashes = {
        _relative(TEMPORAL_WINDOWS_PATH): _sha256(TEMPORAL_WINDOWS_PATH),
        _relative(TEMPORAL_SHAP_WINDOWS_PATH): _sha256(TEMPORAL_SHAP_WINDOWS_PATH),
    }
    for relative_path, digest in expected_hashes.items():
        _require(
            generated.get(relative_path) == digest,
            f"Frozen Part 1 artifact hash changed: {relative_path}",
        )

    for relative_path, recorded_digest in protected.items():
        path = PROJECT_ROOT / Path(relative_path)
        _require(path.is_file(), f"Part 1 protected artifact is missing: {path}")
        _require(
            _sha256(path) == recorded_digest,
            f"Part 1 protected artifact hash changed: {relative_path}",
        )


def load_frozen_temporal_windows(
    path: Path = TEMPORAL_WINDOWS_PATH,
    summary_path: Path = WINDOW_SUMMARY_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load frozen Part 1 membership without recomputing boundaries."""
    summary = _load_json(summary_path)
    _require(
        summary.get("experiment") == "temporal_evaluation_v1"
        and summary.get("stage") == "part_1_window_freeze",
        "Unexpected temporal window summary identity",
    )
    _require(
        summary.get("total_test_cases") == EXPECTED_TEST_CASES,
        "Part 1 summary does not declare 6,276 TEST cases",
    )
    _require(
        summary.get("assertion_status") == "PASS",
        "Part 1 assertions were not recorded as PASS",
    )
    _validate_part_1_hash_contract(summary)

    windows = pd.read_csv(path, encoding="utf-8", dtype={"case_id": "string"})
    expected_columns = [
        "temporal_order",
        "case_id",
        "case_start_time",
        "target",
        "window",
        "canonical_test_row_index",
    ]
    _require(
        windows.columns.tolist() == expected_columns,
        f"Unexpected temporal window columns: {windows.columns.tolist()}",
    )
    return windows, summary


def load_frozen_predictions(
    path: Path = PREDICTIONS_PATH,
) -> pd.DataFrame:
    """Load already-saved Random Forest labels and probabilities."""
    predictions = pd.read_csv(
        path,
        encoding="utf-8",
        dtype={"case_id": "string"},
    )
    expected_columns = ["case_id", "y_true", "y_pred", "y_prob_success"]
    _require(
        predictions.columns.tolist() == expected_columns,
        f"Unexpected prediction columns: {predictions.columns.tolist()}",
    )
    return predictions


def validate_prediction_identity(
    temporal_windows: pd.DataFrame,
    predictions: pd.DataFrame,
    window_summary: dict[str, Any],
    test_ids_path: Path = TEST_IDS_PATH,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Validate A--R and join predictions one-to-one to frozen membership."""
    assertions: dict[str, str] = {}

    _require(len(temporal_windows) == EXPECTED_TEST_CASES, "A failed")
    assertions["A_temporal_windows_contains_6276_rows"] = "PASS"

    _require(len(predictions) == EXPECTED_TEST_CASES, "B failed")
    assertions["B_frozen_prediction_file_contains_6276_rows"] = "PASS"

    _require(
        temporal_windows["case_id"].notna().all()
        and temporal_windows["case_id"].is_unique
        and predictions["case_id"].notna().all()
        and predictions["case_id"].is_unique,
        "D failed: case_id must be complete and unique in both inputs",
    )
    assertions["D_every_case_id_is_unique"] = "PASS"

    temporal_ids = set(temporal_windows["case_id"].astype(str))
    prediction_ids = set(predictions["case_id"].astype(str))
    _require(prediction_ids.issubset(temporal_ids), "E failed")
    assertions["E_every_prediction_case_exists_in_temporal_windows"] = "PASS"
    _require(temporal_ids.issubset(prediction_ids), "F failed")
    assertions["F_every_temporal_case_has_a_prediction"] = "PASS"

    test_ids = pd.read_csv(
        test_ids_path, encoding="utf-8", dtype={"case_id": "string"}
    )
    _require(
        test_ids.columns.tolist() == ["case_id"]
        and len(test_ids) == EXPECTED_TEST_CASES
        and test_ids["case_id"].notna().all()
        and test_ids["case_id"].is_unique,
        "Frozen TEST identities are invalid",
    )
    canonical_ids = test_ids["case_id"].astype(str).tolist()
    _require(
        predictions["case_id"].astype(str).tolist() == canonical_ids,
        "Frozen prediction order differs from test_case_ids.csv",
    )
    canonical_windows = temporal_windows.sort_values(
        "canonical_test_row_index", kind="mergesort"
    )
    _require(
        canonical_windows["canonical_test_row_index"].tolist()
        == list(range(EXPECTED_TEST_CASES))
        and canonical_windows["case_id"].astype(str).tolist() == canonical_ids,
        "Temporal canonical TEST identity trace changed",
    )

    joined = temporal_windows.merge(
        predictions,
        on="case_id",
        how="left",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    _require(
        len(joined) == EXPECTED_TEST_CASES and joined["_merge"].eq("both").all(),
        "C failed: the one-to-one joined dataset is not exactly 6,276 rows",
    )
    assertions["C_joined_dataset_contains_6276_rows"] = "PASS"
    joined = joined.drop(columns="_merge")

    _require(
        joined["y_true"].notna().all()
        and joined["target"].notna().all()
        and np.array_equal(
            joined["y_true"].to_numpy(), joined["target"].to_numpy()
        ),
        "G failed: frozen y_true differs from the temporal target",
    )
    assertions["G_y_true_exactly_matches_temporal_target"] = "PASS"

    _require(
        joined["window"].notna().all()
        and set(joined["window"].unique()) == set(WINDOW_NAMES),
        "H failed: every case must belong to exactly one valid window",
    )
    assertions["H_each_case_belongs_to_exactly_one_window"] = "PASS"

    summary_windows = window_summary.get("windows")
    _require(isinstance(summary_windows, dict), "Part 1 window summaries are missing")
    for letter, name in zip(("I", "J", "K"), WINDOW_NAMES, strict=True):
        observed_count = int(joined["window"].eq(name).sum())
        _require(observed_count == EXPECTED_CASES_PER_WINDOW, f"{letter} failed")
        _require(
            summary_windows.get(name, {}).get("case_count") == observed_count,
            f"{name} count differs from the Part 1 summary",
        )
        assertions[f"{letter}_{name}_contains_2092_cases"] = "PASS"

    for letter, name in zip(("L", "M", "N"), WINDOW_NAMES, strict=True):
        frame = joined.loc[joined["window"] == name]
        observed = {
            0: int(frame["target"].eq(0).sum()),
            1: int(frame["target"].eq(1).sum()),
        }
        expected = WINDOW_TARGET_COUNTS[name]
        _require(observed == expected, f"{letter} failed: {name} target counts")
        part_1 = summary_windows.get(name, {})
        _require(
            part_1.get("success_count") == expected[1]
            and part_1.get("unsuccessful_count") == expected[0],
            f"Part 1 summary target counts differ for {name}",
        )
        assertions[f"{letter}_{name}_target_distribution_matches_part_1"] = "PASS"

    probabilities = joined["y_prob_success"].to_numpy(dtype=np.float64)
    _require(np.isfinite(probabilities).all(), "O failed")
    assertions["O_all_probabilities_are_finite"] = "PASS"
    _require(((probabilities >= 0.0) & (probabilities <= 1.0)).all(), "P failed")
    assertions["P_all_probabilities_lie_in_unit_interval"] = "PASS"

    _require(joined["y_pred"].notna().all(), "Predictions contain missing labels")
    _require(set(joined["y_pred"].unique()) <= {0, 1}, "Q failed")
    assertions["Q_predictions_contain_only_0_and_1"] = "PASS"

    for name in WINDOW_NAMES:
        labels = set(joined.loc[joined["window"] == name, "target"].unique())
        _require(labels == {0, 1}, f"R failed: {name} lacks both target classes")
    assertions["R_all_windows_contain_both_target_classes"] = "PASS"

    return joined, assertions


def _prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_success: np.ndarray,
) -> dict[str, Any]:
    """Compute metrics using Success=1 and fixed confusion labels [0, 1]."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob_success)),
        "average_precision": float(
            average_precision_score(y_true, y_prob_success)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(
                y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0
            )
        ),
        "recall": float(
            recall_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)
        ),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def reproduce_full_test_metrics(
    predictions: pd.DataFrame,
    baseline_metrics_path: Path = BASELINE_METRICS_PATH,
) -> dict[str, Any]:
    """Recompute frozen full-TEST metrics and require Baseline V1 agreement."""
    baseline = _load_json(baseline_metrics_path)
    _require(
        baseline.get("test_cases") == EXPECTED_TEST_CASES,
        "Baseline V1 does not declare 6,276 TEST cases",
    )
    _require(
        baseline.get("positive_class") == POSITIVE_CLASS
        and baseline.get("positive_class_name") == "Success",
        "Baseline V1 positive-class contract changed",
    )

    for name, expected_value in EXPECTED_FULL_TEST_METRICS.items():
        _require(
            np.isclose(
                float(baseline.get(name)),
                expected_value,
                rtol=0.0,
                atol=METRIC_TOLERANCE,
            ),
            f"Baseline V1 {name} differs from the frozen expected value",
        )
    baseline_matrix = baseline.get("confusion_matrix", {})
    _require(
        baseline_matrix.get("labels") == [0, 1]
        and baseline_matrix.get("values") == EXPECTED_CONFUSION_MATRIX,
        "Baseline V1 confusion matrix differs from the frozen contract",
    )

    measured = _prediction_metrics(
        predictions["y_true"].to_numpy(),
        predictions["y_pred"].to_numpy(),
        predictions["y_prob_success"].to_numpy(dtype=np.float64),
    )
    for name, expected_value in EXPECTED_FULL_TEST_METRICS.items():
        _require(
            np.isclose(
                measured[name], expected_value, rtol=0.0, atol=METRIC_TOLERANCE
            ),
            f"Full-TEST {name} did not reproduce Baseline V1",
        )
        _require(
            np.isclose(
                measured[name],
                float(baseline[name]),
                rtol=0.0,
                atol=METRIC_TOLERANCE,
            ),
            f"Full-TEST {name} differs from random_forest_metrics.json",
        )
    _require(
        measured["confusion_matrix"] == EXPECTED_CONFUSION_MATRIX,
        "Full-TEST confusion matrix did not reproduce Baseline V1",
    )
    return measured


def compute_window_metrics(joined: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive and predictive metrics in fixed T1/T2/T3 order."""
    rows: list[dict[str, Any]] = []
    for name in WINDOW_NAMES:
        frame = joined.loc[joined["window"] == name]
        y_true = frame["y_true"].to_numpy()
        y_pred = frame["y_pred"].to_numpy()
        probabilities = frame["y_prob_success"].to_numpy(dtype=np.float64)
        _require(set(np.unique(y_true).tolist()) == {0, 1}, f"{name} lacks a class")
        predictive = _prediction_metrics(y_true, y_pred, probabilities)
        matrix = predictive.pop("confusion_matrix")
        rows.append(
            {
                "window": name,
                "n_cases": int(len(frame)),
                "success_count": int(np.count_nonzero(y_true == 1)),
                "unsuccessful_count": int(np.count_nonzero(y_true == 0)),
                "success_rate": float(np.mean(y_true == 1)),
                "mean_probability_success": float(np.mean(probabilities)),
                "median_probability_success": float(np.median(probabilities)),
                "std_probability_success": float(np.std(probabilities, ddof=0)),
                **predictive,
                "tn": int(matrix[0][0]),
                "fp": int(matrix[0][1]),
                "fn": int(matrix[1][0]),
                "tp": int(matrix[1][1]),
            }
        )
    return pd.DataFrame(rows, columns=METRICS_COLUMNS)


def compute_pairwise_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute directional B-minus-A changes in fixed comparison order."""
    indexed = metrics.set_index("window")
    metric_names = (
        "success_rate",
        "mean_probability_success",
        "roc_auc",
        "average_precision",
        "accuracy",
        "precision",
        "recall",
        "f1",
    )
    rows: list[dict[str, Any]] = []
    for comparison, earlier, later in COMPARISONS:
        row: dict[str, Any] = {"comparison": comparison}
        for name in metric_names:
            row[f"delta_{name}"] = float(indexed.loc[later, name]) - float(
                indexed.loc[earlier, name]
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=DELTA_COLUMNS)


def _finish_plot(figure: plt.Figure, output_path: Path) -> None:
    """Save a PNG with stable dimensions and metadata, then close it."""
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        format="png",
        metadata={"Software": "Temporal Evaluation V1 Part 2"},
    )
    plt.close(figure)


def create_prediction_metrics_plot(
    metrics: pd.DataFrame, output_path: Path = METRICS_PLOT_PATH
) -> None:
    """Plot ROC-AUC and Average Precision with complete probability scale."""
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.plot(
        WINDOW_NAMES,
        metrics["roc_auc"],
        marker="o",
        linewidth=2.0,
        markersize=6,
        label="ROC-AUC",
    )
    axis.plot(
        WINDOW_NAMES,
        metrics["average_precision"],
        marker="s",
        linewidth=2.0,
        markersize=6,
        label="Average Precision",
    )
    axis.set(title="Frozen Random Forest performance by temporal window")
    axis.set_xlabel("Temporal window")
    axis.set_ylabel("Metric value")
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks(np.linspace(0.0, 1.0, 11))
    axis.grid(axis="y", alpha=0.3)
    axis.legend(loc="best")
    _finish_plot(figure, output_path)


def create_probability_plot(
    metrics: pd.DataFrame, output_path: Path = PROBABILITY_PLOT_PATH
) -> None:
    """Plot observed Success prevalence and average model probability."""
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.plot(
        WINDOW_NAMES,
        metrics["success_rate"],
        marker="o",
        linewidth=2.0,
        markersize=6,
        label="Success rate",
    )
    axis.plot(
        WINDOW_NAMES,
        metrics["mean_probability_success"],
        marker="s",
        linewidth=2.0,
        markersize=6,
        label="Mean predicted P(Success)",
    )
    axis.set(title="Observed prevalence and prediction tendency by window")
    axis.set_xlabel("Temporal window")
    axis.set_ylabel("Proportion / probability")
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks(np.linspace(0.0, 1.0, 11))
    axis.grid(axis="y", alpha=0.3)
    axis.legend(loc="best")
    _finish_plot(figure, output_path)


def _window_records(metrics: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Convert metric rows to JSON-native records keyed in window order."""
    records: dict[str, dict[str, Any]] = {}
    integer_columns = {
        "n_cases",
        "success_count",
        "unsuccessful_count",
        "tn",
        "fp",
        "fn",
        "tp",
    }
    for row in metrics.to_dict(orient="records"):
        name = str(row.pop("window"))
        record = {
            key: int(value) if key in integer_columns else float(value)
            for key, value in row.items()
        }
        record["confusion_matrix"] = [
            [record["tn"], record["fp"]],
            [record["fn"], record["tp"]],
        ]
        records[name] = record
    return records


def _delta_records(deltas: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert directional delta rows to JSON-native values."""
    records: list[dict[str, Any]] = []
    for row in deltas.to_dict(orient="records"):
        records.append(
            {
                key: str(value) if key == "comparison" else float(value)
                for key, value in row.items()
            }
        )
    return records


def write_summary(summary: dict[str, Any], path: Path = SUMMARY_PATH) -> None:
    """Write stable, human-readable JSON with a final newline."""
    content = json.dumps(
        summary, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"
    path.write_text(content, encoding="utf-8", newline="\n")


def validate_outputs(
    expected_metrics: pd.DataFrame,
    expected_deltas: pd.DataFrame,
    expected_summary: dict[str, Any],
) -> None:
    """Reload all Part 2 outputs and validate schema, order, and content."""
    saved_metrics = pd.read_csv(METRICS_PATH, encoding="utf-8")
    saved_deltas = pd.read_csv(DELTAS_PATH, encoding="utf-8")
    _require(saved_metrics.columns.tolist() == METRICS_COLUMNS, "Metrics schema changed")
    _require(saved_deltas.columns.tolist() == DELTA_COLUMNS, "Delta schema changed")
    _require(saved_metrics["window"].tolist() == list(WINDOW_NAMES), "Window order changed")
    _require(
        saved_deltas["comparison"].tolist()
        == [comparison[0] for comparison in COMPARISONS],
        "Comparison order changed",
    )
    pd.testing.assert_frame_equal(
        saved_metrics,
        expected_metrics,
        check_dtype=False,
        check_exact=False,
        rtol=0.0,
        atol=1e-15,
    )
    pd.testing.assert_frame_equal(
        saved_deltas,
        expected_deltas,
        check_dtype=False,
        check_exact=False,
        rtol=0.0,
        atol=1e-15,
    )
    _require(_load_json(SUMMARY_PATH) == expected_summary, "Saved summary changed")
    for path in (METRICS_PLOT_PATH, PROBABILITY_PLOT_PATH):
        image = mpimg.imread(path)
        _require(
            image.ndim in (2, 3) and image.shape[0] > 0 and image.shape[1] > 0,
            f"Generated plot is invalid: {path}",
        )


def _print_summary(summary: dict[str, Any]) -> None:
    """Print concise measured results and generated artifact hashes."""
    print("=== TEMPORAL EVALUATION V1 PART 2 ===")
    print(f"Full TEST metrics reproduced: {summary['full_test_metrics_reproduced']}")
    for name, values in summary["per_window_metrics"].items():
        print(
            f"{name}: n={values['n_cases']:,}, "
            f"Success rate={values['success_rate']:.12f}, "
            f"Mean P(Success)={values['mean_probability_success']:.12f}, "
            f"ROC-AUC={values['roc_auc']:.12f}, "
            f"AP={values['average_precision']:.12f}, "
            f"Accuracy={values['accuracy']:.12f}, "
            f"Precision={values['precision']:.12f}, "
            f"Recall={values['recall']:.12f}, F1={values['f1']:.12f}, "
            f"CM={values['confusion_matrix']}"
        )
    print(f"All assertions: {summary['assertion_status']}")
    print(f"Protected artifacts unchanged: {summary['protected_artifacts']['unchanged']}")
    print("Generated artifact hashes:")
    for path, digest in summary["generated_artifact_sha256"].items():
        print(f"  {path}: {digest}")


def main() -> None:
    """Build and validate Temporal Evaluation V1 Part 2 only."""
    protected_paths = _protected_paths()
    protected_before = protect_artifact_hashes(protected_paths)

    temporal_windows, window_summary = load_frozen_temporal_windows()
    predictions = load_frozen_predictions()
    joined, assertions = validate_prediction_identity(
        temporal_windows, predictions, window_summary
    )

    full_test_metrics = reproduce_full_test_metrics(predictions)
    assertions["S_full_test_metrics_reproduce_baseline_v1"] = "PASS"

    metrics = compute_window_metrics(joined)
    deltas = compute_pairwise_deltas(metrics)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(
        METRICS_PATH,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.17g",
    )
    deltas.to_csv(
        DELTAS_PATH,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.17g",
    )
    create_prediction_metrics_plot(metrics)
    create_probability_plot(metrics)

    protected_after_outputs = protect_artifact_hashes(protected_paths)
    _require(
        protected_after_outputs == protected_before,
        "A protected frozen artifact changed during Part 2",
    )

    assertions.update(
        {
            "T_no_model_fitting_occurred": "PASS",
            "U_no_preprocessing_fitting_occurred": "PASS",
            "V_no_prediction_threshold_change_occurred": "PASS",
            "W_no_temporal_boundaries_changed": "PASS",
            "X_no_shap_computation_occurred": "PASS",
            "Y_no_formal_drift_test_occurred": "PASS",
            "Z_no_significance_testing_occurred": "PASS",
        }
    )

    generated_hashes = {
        _relative(METRICS_PATH): _sha256(METRICS_PATH),
        _relative(DELTAS_PATH): _sha256(DELTAS_PATH),
        _relative(METRICS_PLOT_PATH): _sha256(METRICS_PLOT_PATH),
        _relative(PROBABILITY_PLOT_PATH): _sha256(PROBABILITY_PLOT_PATH),
    }
    summary: dict[str, Any] = {
        "experiment": "temporal_evaluation_v1",
        "stage": "part_2_temporal_prediction_analysis",
        "model": "Frozen Random Forest Baseline V1",
        "prediction_point": "k=10",
        "positive_class": POSITIVE_CLASS,
        "positive_class_name": "Success",
        "test_cases": EXPECTED_TEST_CASES,
        "windows": list(WINDOW_NAMES),
        "full_test_metrics_reproduced": True,
        "full_test_metrics": full_test_metrics,
        "per_window_metrics": _window_records(metrics),
        "probability_standard_deviation": "population standard deviation (ddof=0)",
        "pairwise_delta_definition": "metric(later window) - metric(earlier window)",
        "pairwise_deltas": _delta_records(deltas),
        "threshold_changed": False,
        "model_retrained": False,
        "preprocessing_refitted": False,
        "window_boundaries_changed": False,
        "shap_computed": False,
        "significance_testing": False,
        "formal_drift_detection": False,
        "assertions": assertions,
        "assertion_status": "PASS",
        "protected_artifacts": {
            "count": len(protected_after_outputs),
            "unchanged": True,
            "sha256": {
                _relative(path): digest
                for path, digest in protected_after_outputs.items()
            },
        },
        "generated_artifact_sha256": generated_hashes,
    }
    write_summary(summary)
    validate_outputs(metrics, deltas, summary)

    protected_final = protect_artifact_hashes(protected_paths)
    _require(
        protected_final == protected_before,
        "A protected frozen artifact changed during output validation",
    )
    _print_summary(summary)


if __name__ == "__main__":
    main()
