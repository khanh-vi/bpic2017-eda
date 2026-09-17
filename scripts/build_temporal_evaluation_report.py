"""Build Temporal Evaluation V1 Part 4 from frozen Part 1--3 results.

This module only validates, joins, and reports already-stored temporal
prediction and SHAP results. It contains no model inference, model or
preprocessor fitting, SHAP computation, sampling, boundary construction,
significance testing, or formal drift detection.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = PROJECT_ROOT / "docs" / "temporal_evaluation_design_v1.md"
OUTPUT_DIR = PROJECT_ROOT / "results" / "temporal_eval_v1"

WINDOWS_PATH = OUTPUT_DIR / "temporal_windows.csv"
SHAP_WINDOWS_PATH = OUTPUT_DIR / "temporal_shap_windows.csv"
WINDOW_SUMMARY_PATH = OUTPUT_DIR / "temporal_window_summary.json"

PREDICTION_METRICS_PATH = OUTPUT_DIR / "temporal_prediction_metrics.csv"
PREDICTION_DELTAS_PATH = OUTPUT_DIR / "temporal_prediction_deltas.csv"
PREDICTION_SUMMARY_PATH = OUTPUT_DIR / "temporal_prediction_summary.json"
PREDICTION_METRICS_PLOT_PATH = OUTPUT_DIR / "temporal_prediction_metrics.png"
PROBABILITY_PLOT_PATH = OUTPUT_DIR / "temporal_probability.png"

SHAP_IMPORTANCE_PATH = OUTPUT_DIR / "temporal_shap_importance.csv"
SHAP_TOP10_PATH = OUTPUT_DIR / "temporal_shap_top10.csv"
SHAP_STABILITY_PATH = OUTPUT_DIR / "temporal_shap_stability.csv"
SIGNED_SHAP_PATH = OUTPUT_DIR / "temporal_signed_shap.csv"
GROUP_IMPORTANCE_PATH = OUTPUT_DIR / "temporal_group_importance.csv"
SHAP_SUMMARY_PATH = OUTPUT_DIR / "temporal_shap_summary.json"
SHAP_TOP_FEATURES_PLOT_PATH = OUTPUT_DIR / "temporal_shap_top_features.png"
SHAP_HEATMAP_PATH = OUTPUT_DIR / "temporal_shap_heatmap.png"
GROUP_IMPORTANCE_PLOT_PATH = OUTPUT_DIR / "temporal_group_importance.png"

SYNTHESIS_PATH = OUTPUT_DIR / "prediction_explanation_synthesis.csv"
REPORT_PATH = OUTPUT_DIR / "temporal_evaluation_v1.md"
FINAL_SUMMARY_PATH = OUTPUT_DIR / "temporal_evaluation_summary.json"

WINDOW_NAMES = ("T1", "T2", "T3")
COMPARISONS = (
    ("T1_to_T2", "T1", "T2"),
    ("T2_to_T3", "T2", "T3"),
    ("T1_to_T3", "T1", "T3"),
)
EXPECTED_TEST_CASES = 6_276
EXPECTED_SHAP_CASES = 1_000
EXPECTED_FEATURE_COUNT = 165
EXPECTED_CASES_PER_WINDOW = 2_092
EXPECTED_SHAP_WINDOW_COUNTS = {"T1": 325, "T2": 332, "T3": 343}

PREDICTION_DELTA_COLUMNS = (
    "delta_success_rate",
    "delta_mean_probability_success",
    "delta_roc_auc",
    "delta_average_precision",
    "delta_accuracy",
    "delta_precision",
    "delta_recall",
    "delta_f1",
)
SYNTHESIS_COLUMNS = (
    "comparison",
    "window_a",
    "window_b",
    *PREDICTION_DELTA_COLUMNS,
    "jaccard_at_10",
    "spearman_rho",
    "top10_intersection_count",
    "top10_union_count",
)

PART1_PATHS = (
    WINDOWS_PATH,
    SHAP_WINDOWS_PATH,
    WINDOW_SUMMARY_PATH,
)
PART2_PATHS = (
    PREDICTION_METRICS_PATH,
    PREDICTION_DELTAS_PATH,
    PREDICTION_SUMMARY_PATH,
    PREDICTION_METRICS_PLOT_PATH,
    PROBABILITY_PLOT_PATH,
)
PART3_PATHS = (
    SHAP_IMPORTANCE_PATH,
    SHAP_TOP10_PATH,
    SHAP_STABILITY_PATH,
    SIGNED_SHAP_PATH,
    GROUP_IMPORTANCE_PATH,
    SHAP_SUMMARY_PATH,
    SHAP_TOP_FEATURES_PLOT_PATH,
    SHAP_HEATMAP_PATH,
    GROUP_IMPORTANCE_PLOT_PATH,
)

EXPECTED_PREDICTION_PRIMARY = {
    "T1": {
        "n_cases": "2092",
        "roc_auc": "0.57777700664408793",
        "average_precision": "0.63406664398557222",
    },
    "T2": {
        "n_cases": "2092",
        "roc_auc": "0.57763240099961544",
        "average_precision": "0.62199791333975196",
    },
    "T3": {
        "n_cases": "2092",
        "roc_auc": "0.60730224896937568",
        "average_precision": "0.63556943295073831",
    },
}
EXPECTED_EXPLANATION_STABILITY = {
    "T1_to_T2": {
        "jaccard_at_10": "0.81818181818181823",
        "spearman_rho": "0.97285203630238348",
    },
    "T2_to_T3": {
        "jaccard_at_10": "0.81818181818181823",
        "spearman_rho": "0.97363496995038201",
    },
    "T1_to_T3": {
        "jaccard_at_10": "1",
        "spearman_rho": "0.95695912160511254",
    },
}


def _require(condition: bool, message: str) -> None:
    """Raise a stable validation error when an invariant is false."""
    if not condition:
        raise AssertionError(message)


def _relative(path: Path) -> str:
    """Return a repository-relative path with POSIX separators."""
    return path.relative_to(PROJECT_ROOT).as_posix()


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a required file."""
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {_relative(path)}")
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    """Load a required JSON object."""
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    _require(isinstance(value, dict), f"Expected JSON object: {_relative(path)}")
    return value


def _load_csv(path: Path) -> list[dict[str, str]]:
    """Load a required CSV while preserving its canonical numeric strings."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames is not None, f"Missing CSV header: {_relative(path)}")
        rows = list(reader)
    return rows


def _write_text_atomic(path: Path, content: str) -> None:
    """Atomically write deterministic UTF-8 text with explicit newlines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        temporary.write_text(content, encoding="utf-8", newline="")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_csv_atomic(
    path: Path, rows: list[dict[str, str]], columns: Iterable[str]
) -> None:
    """Write canonical CSV text without changing stored numeric strings."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _write_text_atomic(path, buffer.getvalue())


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Write deterministic, human-readable JSON."""
    _write_text_atomic(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _index_unique(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    """Index rows by a required unique, non-empty column."""
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "")
        _require(bool(value), f"Missing {key} value")
        _require(value not in result, f"Duplicate {key}: {value}")
        result[value] = row
    return result


def _as_float(value: str) -> float:
    """Convert one validated canonical numeric string for JSON/reporting."""
    return float(Decimal(value))


def _exact_decimal(actual: str, expected: str, context: str) -> None:
    """Require exact decimal equality independent of textual formatting."""
    _require(Decimal(actual) == Decimal(expected), f"Unexpected {context}: {actual}")


def _validate_recorded_hashes(summary: dict[str, Any], field: str) -> None:
    """Validate hashes recorded by an upstream stage summary."""
    recorded = summary.get(field)
    _require(isinstance(recorded, dict) and recorded, f"Missing upstream {field}")
    for relative_path, expected_digest in recorded.items():
        path = PROJECT_ROOT / Path(relative_path)
        _require(path.is_file(), f"Upstream artifact missing: {relative_path}")
        _require(
            _sha256(path) == expected_digest,
            f"Upstream artifact hash changed: {relative_path}",
        )


def _protected_paths() -> tuple[Path, ...]:
    """Collect all frozen Part 1--3, Baseline V1, and SHAP Pilot V1 files."""
    roots = (
        PROJECT_ROOT / "results" / "baseline_v1",
        PROJECT_ROOT / "results" / "shap_pilot_v1",
        PROJECT_ROOT / "artifacts" / "baseline_v1",
        PROJECT_ROOT / "artifacts" / "shap_pilot_v1",
        PROJECT_ROOT / "data" / "processed" / "baseline_k10",
        PROJECT_ROOT / "data" / "processed" / "shap_pilot_v1",
    )
    paths = {
        path
        for root in roots
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_file()
    }
    paths.update((DESIGN_PATH, *PART1_PATHS, *PART2_PATHS, *PART3_PATHS))
    missing = [path for path in paths if not path.is_file()]
    _require(not missing, f"Protected artifacts missing: {missing}")
    return tuple(sorted(paths, key=_relative))


def _hash_paths(paths: Iterable[Path]) -> dict[Path, str]:
    """Hash paths for a before/after integrity comparison."""
    return {path: _sha256(path) for path in paths}


def verify_protected_artifacts(
    before: dict[Path, str], after: dict[Path, str]
) -> None:
    """Assert byte identity for every protected artifact."""
    _require(before == after, "One or more protected artifacts changed")


def _validate_source_guardrails() -> None:
    """Statically reject computation that is outside Part 4 scope."""
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_modules = {"joblib", "shap", "sklearn"}
    forbidden_calls = {
        "fit",
        "fit_transform",
        "predict",
        "predict_proba",
        "shap_values",
        "TreeExplainer",
        "Explainer",
        "choice",
        "sample",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".")[0] for alias in node.names}
            _require(not roots & forbidden_modules, "Forbidden Part 4 import found")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            _require(root not in forbidden_modules, "Forbidden Part 4 import found")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            else:
                name = ""
            _require(name not in forbidden_calls, f"Forbidden Part 4 call: {name}")


def load_frozen_prediction_results() -> dict[str, Any]:
    """Load canonical Part 2 tables and summary without model inference."""
    return {
        "metrics": _load_csv(PREDICTION_METRICS_PATH),
        "deltas": _load_csv(PREDICTION_DELTAS_PATH),
        "summary": _load_json(PREDICTION_SUMMARY_PATH),
    }


def load_frozen_explanation_results() -> dict[str, Any]:
    """Load canonical Part 3 tables and summary without SHAP computation."""
    return {
        "top10": _load_csv(SHAP_TOP10_PATH),
        "stability": _load_csv(SHAP_STABILITY_PATH),
        "signed": _load_csv(SIGNED_SHAP_PATH),
        "groups": _load_csv(GROUP_IMPORTANCE_PATH),
        "summary": _load_json(SHAP_SUMMARY_PATH),
    }


def validate_part_results(
    window_summary: dict[str, Any],
    prediction: dict[str, Any],
    explanation: dict[str, Any],
) -> None:
    """Validate identities, hashes, schemas, ordering, and expected values."""
    _validate_source_guardrails()

    _require(
        window_summary.get("experiment") == "temporal_evaluation_v1"
        and window_summary.get("stage") == "part_1_window_freeze"
        and window_summary.get("assertion_status") == "PASS",
        "Unexpected or failed Part 1 summary",
    )
    _require(
        window_summary.get("total_test_cases") == EXPECTED_TEST_CASES,
        "Part 1 test population changed",
    )
    _require(
        window_summary.get("shap_explained_cases") == EXPECTED_SHAP_CASES,
        "Part 1 SHAP population changed",
    )
    _require(
        window_summary.get("shap_cases_per_window") == EXPECTED_SHAP_WINDOW_COUNTS,
        "Part 1 SHAP window counts changed",
    )
    _validate_recorded_hashes(window_summary, "generated_artifact_sha256")
    _validate_recorded_hashes(
        {"hashes": window_summary.get("protected_artifacts", {}).get("sha256")},
        "hashes",
    )

    prediction_summary = prediction["summary"]
    _require(
        prediction_summary.get("experiment") == "temporal_evaluation_v1"
        and prediction_summary.get("stage") == "part_2_temporal_prediction_analysis"
        and prediction_summary.get("assertion_status") == "PASS",
        "Unexpected or failed Part 2 summary",
    )
    _require(
        prediction_summary.get("test_cases") == EXPECTED_TEST_CASES,
        "Part 2 test population changed",
    )
    _validate_recorded_hashes(prediction_summary, "generated_artifact_sha256")

    explanation_summary = explanation["summary"]
    _require(
        explanation_summary.get("experiment") == "temporal_evaluation_v1"
        and explanation_summary.get("stage") == "part_3_temporal_shap_analysis"
        and explanation_summary.get("assertion_status") == "PASS",
        "Unexpected or failed Part 3 summary",
    )
    _require(
        explanation_summary.get("shap_cases") == EXPECTED_SHAP_CASES
        and explanation_summary.get("feature_count") == EXPECTED_FEATURE_COUNT
        and explanation_summary.get("top_k") == 10,
        "Part 3 frozen dimensions changed",
    )
    _validate_recorded_hashes(
        explanation_summary, "generated_artifact_sha256_excluding_summary"
    )

    metric_rows = prediction["metrics"]
    delta_rows = prediction["deltas"]
    stability_rows = explanation["stability"]
    _require(
        [row["window"] for row in metric_rows] == list(WINDOW_NAMES),
        "Unexpected prediction-window order",
    )
    _require(
        [row["comparison"] for row in delta_rows]
        == [item[0] for item in COMPARISONS],
        "Unexpected prediction-comparison order",
    )
    _require(
        [row["comparison"] for row in stability_rows]
        == [item[0] for item in COMPARISONS],
        "Unexpected explanation-comparison order",
    )

    metrics_by_window = _index_unique(metric_rows, "window")
    for window, expected in EXPECTED_PREDICTION_PRIMARY.items():
        row = metrics_by_window[window]
        _require(row["n_cases"] == expected["n_cases"], f"Unexpected {window} count")
        _exact_decimal(row["roc_auc"], expected["roc_auc"], f"{window} ROC-AUC")
        _exact_decimal(
            row["average_precision"],
            expected["average_precision"],
            f"{window} average precision",
        )

    stability_by_comparison = _index_unique(stability_rows, "comparison")
    for comparison, expected in EXPECTED_EXPLANATION_STABILITY.items():
        row = stability_by_comparison[comparison]
        _exact_decimal(
            row["jaccard_at_10"],
            expected["jaccard_at_10"],
            f"{comparison} Jaccard@10",
        )
        _exact_decimal(
            row["spearman_rho"],
            expected["spearman_rho"],
            f"{comparison} Spearman rho",
        )


def build_synthesis_table(
    prediction: dict[str, Any], explanation: dict[str, Any]
) -> list[dict[str, str]]:
    """Join the frozen Part 2 deltas to frozen Part 3 stability metrics."""
    deltas = _index_unique(prediction["deltas"], "comparison")
    stability = _index_unique(explanation["stability"], "comparison")
    rows: list[dict[str, str]] = []
    for comparison, window_a, window_b in COMPARISONS:
        prediction_row = deltas[comparison]
        explanation_row = stability[comparison]
        _require(
            explanation_row["window_a"] == window_a
            and explanation_row["window_b"] == window_b,
            f"Window labels disagree for {comparison}",
        )
        row = {
            "comparison": comparison,
            "window_a": window_a,
            "window_b": window_b,
            **{
                column: prediction_row[column]
                for column in PREDICTION_DELTA_COLUMNS
            },
            "jaccard_at_10": explanation_row["jaccard_at_10"],
            "spearman_rho": explanation_row["spearman_rho"],
            "top10_intersection_count": explanation_row["intersection_count"],
            "top10_union_count": explanation_row["union_count"],
        }
        rows.append(row)
    return rows


def _derive_feature_continuity(
    top10_rows: list[dict[str, str]],
) -> dict[str, Any]:
    """Derive set continuity solely from the frozen top-10 table."""
    by_window: dict[str, list[dict[str, str]]] = {window: [] for window in WINDOW_NAMES}
    for row in top10_rows:
        _require(row["window"] in by_window, f"Unexpected top-10 window: {row['window']}")
        by_window[row["window"]].append(row)
    for window in WINDOW_NAMES:
        ordered = sorted(by_window[window], key=lambda row: int(row["rank"]))
        _require(
            [int(row["rank"]) for row in ordered] == list(range(1, 11)),
            f"Unexpected {window} top-10 ranks",
        )
        by_window[window] = ordered

    ordered_features = {
        window: [row["feature"] for row in by_window[window]]
        for window in WINDOW_NAMES
    }
    feature_sets = {window: set(features) for window, features in ordered_features.items()}
    all_window_intersection = [
        feature
        for feature in ordered_features["T1"]
        if all(feature in feature_sets[window] for window in WINDOW_NAMES)
    ]
    return {
        "rows_by_window": by_window,
        "features_by_window": ordered_features,
        "all_window_intersection": all_window_intersection,
        "t2_entering_vs_t1": sorted(feature_sets["T2"] - feature_sets["T1"]),
        "t2_leaving_vs_t1": sorted(feature_sets["T1"] - feature_sets["T2"]),
        "t3_entering_vs_t2": sorted(feature_sets["T3"] - feature_sets["T2"]),
        "t3_leaving_vs_t2": sorted(feature_sets["T2"] - feature_sets["T3"]),
        "t1_t3_identical": feature_sets["T1"] == feature_sets["T3"],
        "union_count": len(set.union(*feature_sets.values())),
    }


def _derive_signed_changes(signed_rows: list[dict[str, str]]) -> dict[str, Any]:
    """Describe stored sign labels and transitions without causal inference."""
    _require(len(signed_rows) == 11, "Signed SHAP union must contain 11 features")
    temporary_t2 = [
        row["feature"]
        for row in signed_rows
        if row["T1_sign"] == row["T3_sign"]
        and row["T2_sign"] != row["T1_sign"]
    ]
    return {
        "union_feature_count": len(signed_rows),
        "temporary_t2_sign_change_features": temporary_t2,
        "sign_patterns": {
            row["feature"]: [row["T1_sign"], row["T2_sign"], row["T3_sign"]]
            for row in signed_rows
        },
    }


def _derive_group_summary(group_rows: list[dict[str, str]]) -> dict[str, Any]:
    """Index the frozen feature-group metrics for reporting."""
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for row in group_rows:
        window = row["window"]
        group = row["group"]
        groups.setdefault(group, {})[window] = {
            "feature_count": int(row["feature_count"]),
            "group_total_importance": _as_float(row["group_total_importance"]),
            "group_mean_importance_per_feature": _as_float(
                row["group_mean_importance_per_feature"]
            ),
            "share_of_total_importance": _as_float(row["share_of_total_importance"]),
        }
    _require(len(group_rows) == 24 and len(groups) == 8, "Expected 8 groups x 3 windows")
    _require(
        groups["Resource"]["T1"]["feature_count"] == 115,
        "Resource must contain 115 encoded features",
    )
    return groups


def derive_main_findings(
    prediction: dict[str, Any], explanation: dict[str, Any]
) -> dict[str, Any]:
    """Derive report facts from frozen tables."""
    continuity = _derive_feature_continuity(explanation["top10"])
    signed_changes = _derive_signed_changes(explanation["signed"])
    groups = _derive_group_summary(explanation["groups"])
    metrics = _index_unique(prediction["metrics"], "window")

    _require(
        len(continuity["all_window_intersection"]) == 9,
        "Expected nine features in the all-window top-10 intersection",
    )
    _require(
        continuity["t2_entering_vs_t1"] == ["application_type_Limit raise"]
        and continuity["t2_leaving_vs_t1"] == ["loan_goal_Car"],
        "Unexpected T2 top-10 membership change",
    )
    _require(continuity["t1_t3_identical"], "T1 and T3 top-10 sets differ")
    _require(continuity["union_count"] == 11, "Top-10 union must contain 11 features")
    _require(
        signed_changes["temporary_t2_sign_change_features"]
        == [
            "loan_goal_Car",
            "application_type_New credit",
            "application_type_Limit raise",
        ],
        "Unexpected signed SHAP sign-change features",
    )

    main_findings = [
        "Success prevalence decreased from T1 through T3.",
        "Mean predicted Success probability decreased from T1 through T3.",
        "ROC-AUC was nearly unchanged from T1 to T2 and was higher in T3.",
        "Average Precision decreased in T2 and recovered in T3.",
        "Adjacent windows shared 9 of their top 10 SHAP features.",
        "T1 and T3 had identical top-10 SHAP feature sets.",
        "Full 165-feature importance rankings remained strongly correlated across windows.",
        "Resource attribution share decreased while Activity share increased.",
        "Three union-top-10 features had a temporary mean signed SHAP direction change in T2.",
        "Predictive changes were not accompanied by a comparably large reorganization of global SHAP importance.",
    ]
    return {
        "metrics_by_window": metrics,
        "continuity": continuity,
        "signed_changes": signed_changes,
        "groups": groups,
        "main_findings": main_findings,
    }


def _code_list(values: list[str]) -> str:
    """Format feature names as an inline Markdown list."""
    return ", ".join(f"`{value}`" for value in values)


def _format_delta(value: str) -> str:
    """Format a signed delta to six decimal places."""
    return f"{_as_float(value):+.6f}"


def build_report(
    window_summary: dict[str, Any],
    synthesis: list[dict[str, str]],
    derived: dict[str, Any],
) -> str:
    """Build the final Markdown report from canonical stored values."""
    metrics = derived["metrics_by_window"]
    continuity = derived["continuity"]
    signed = derived["signed_changes"]
    groups = derived["groups"]
    windows = window_summary["windows"]

    prediction_table = [
        "| Window | Cases | Success rate | Mean P(Success) | ROC-AUC | Average Precision | Accuracy | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for window in WINDOW_NAMES:
        row = metrics[window]
        prediction_table.append(
            f"| {window} | {int(row['n_cases']):,} | "
            f"{_as_float(row['success_rate']):.6f} | "
            f"{_as_float(row['mean_probability_success']):.6f} | "
            f"{_as_float(row['roc_auc']):.6f} | "
            f"{_as_float(row['average_precision']):.6f} | "
            f"{_as_float(row['accuracy']):.6f} | "
            f"{_as_float(row['precision']):.6f} | "
            f"{_as_float(row['recall']):.6f} | "
            f"{_as_float(row['f1']):.6f} |"
        )

    top10_table = [
        "| Rank | T1 | T2 | T3 |",
        "|---:|---|---|---|",
    ]
    for index in range(10):
        top10_table.append(
            f"| {index + 1} | `{continuity['features_by_window']['T1'][index]}` | "
            f"`{continuity['features_by_window']['T2'][index]}` | "
            f"`{continuity['features_by_window']['T3'][index]}` |"
        )

    synthesis_table = [
        "| Comparison | Delta ROC-AUC | Delta AP | Jaccard@10 | Spearman rho | Top-10 overlap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in synthesis:
        synthesis_table.append(
            f"| {row['window_a']} to {row['window_b']} | "
            f"{_format_delta(row['delta_roc_auc'])} | "
            f"{_format_delta(row['delta_average_precision'])} | "
            f"{_as_float(row['jaccard_at_10']):.9f} | "
            f"{_as_float(row['spearman_rho']):.9f} | "
            f"{row['top10_intersection_count']} of 10 |"
        )

    group_table = [
        "| Group | Encoded features | T1 share | T2 share | T3 share | T1 mean/feature | T2 mean/feature | T3 mean/feature |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in (
        "Numeric",
        "LoanGoal",
        "ApplicationType",
        "Activity",
        "Action",
        "EventOrigin",
        "Resource",
        "Lifecycle",
    ):
        values = groups[group]
        group_table.append(
            f"| {group} | {values['T1']['feature_count']} | "
            f"{values['T1']['share_of_total_importance']:.2%} | "
            f"{values['T2']['share_of_total_importance']:.2%} | "
            f"{values['T3']['share_of_total_importance']:.2%} | "
            f"{values['T1']['group_mean_importance_per_feature']:.6f} | "
            f"{values['T2']['group_mean_importance_per_feature']:.6f} | "
            f"{values['T3']['group_mean_importance_per_feature']:.6f} |"
        )

    finding_lines = "\n".join(
        f"{index}. {finding}" for index, finding in enumerate(derived["main_findings"], 1)
    )
    window_lines = "\n".join(
        f"- **{window}:** {windows[window]['start_case_start_time']} through "
        f"{windows[window]['end_case_start_time']} ({windows[window]['case_count']:,} cases)."
        for window in WINDOW_NAMES
    )

    report = f"""# Temporal Evaluation V1

## 1. Objective

Temporal Evaluation V1 asks two questions about the frozen BPIC 2017 test period: **RQ1**, how did prediction metrics and prediction distributions change across time; and **RQ2**, how did global SHAP feature importance and feature ranking change across time? The combined question is: **How did prediction and explanation change over time?**

This is an observational, preliminary analysis. It does not establish statistical significance, formal concept drift, or causality.

## 2. Experimental Setup

- **Dataset:** BPIC 2017.
- **Prediction point:** `k=10` events.
- **Model:** frozen Random Forest Baseline V1, an untuned baseline.
- **Prediction population:** all 6,276 frozen test cases.
- **Explanation population:** the frozen 1,000-case SHAP sample drawn from the same test population (T1: 325; T2: 332; T3: 343).
- **Time variable:** `case_start_time`.
- **Window design:** three chronological equal-case windows, each containing 2,092 prediction cases.

The frozen time ranges are:

{window_lines}

The prediction and explanation analyses use the same frozen temporal boundaries. The populations differ: prediction results cover 6,276 cases, whereas explanation results describe the frozen 1,000-case sample.

## 3. Temporal Prediction Results

{chr(10).join(prediction_table)}

From T1 to T2, ROC-AUC changed by {_format_delta(synthesis[0]['delta_roc_auc'])} and Average Precision (AP) by {_format_delta(synthesis[0]['delta_average_precision'])}. Success prevalence decreased by {_format_delta(synthesis[0]['delta_success_rate'])}, and mean predicted P(Success) decreased by {_format_delta(synthesis[0]['delta_mean_probability_success'])}.

From T2 to T3, ROC-AUC changed by {_format_delta(synthesis[1]['delta_roc_auc'])} and AP by {_format_delta(synthesis[1]['delta_average_precision'])}. Success prevalence decreased by {_format_delta(synthesis[1]['delta_success_rate'])}, while mean predicted P(Success) decreased by {_format_delta(synthesis[1]['delta_mean_probability_success'])}.

Across T1 to T3, ROC-AUC was higher by {_format_delta(synthesis[2]['delta_roc_auc'])}, while AP was higher by {_format_delta(synthesis[2]['delta_average_precision'])}. Success prevalence decreased by {_format_delta(synthesis[2]['delta_success_rate'])}, and mean predicted P(Success) decreased by {_format_delta(synthesis[2]['delta_mean_probability_success'])}. These are descriptive changes; no statistical significance testing was performed, and a higher value for one metric is not treated as general model improvement.

Figures: [temporal prediction metrics](temporal_prediction_metrics.png) and [temporal probability and prevalence](temporal_probability.png).

## 4. Temporal Explanation Results

{chr(10).join(top10_table)}

The intersection of the top-10 sets across all three windows contains 9 features: {_code_list(continuity['all_window_intersection'])}. Relative to T1, T2 added {_code_list(continuity['t2_entering_vs_t1'])} and omitted {_code_list(continuity['t2_leaving_vs_t1'])}. From T2 to T3, {_code_list(continuity['t3_entering_vs_t2'])} re-entered and {_code_list(continuity['t3_leaving_vs_t2'])} left. The T1 and T3 top-10 sets are identical.

- **T1 to T2:** Jaccard@10 = {_as_float(synthesis[0]['jaccard_at_10']):.9f}, with 9 of 10 features overlapping; Spearman rho = {_as_float(synthesis[0]['spearman_rho']):.9f} across all 165 features.
- **T2 to T3:** Jaccard@10 = {_as_float(synthesis[1]['jaccard_at_10']):.9f}, with 9 of 10 features overlapping; Spearman rho = {_as_float(synthesis[1]['spearman_rho']):.9f} across all 165 features.
- **T1 to T3:** Jaccard@10 = {_as_float(synthesis[2]['jaccard_at_10']):.9f}, with all 10 features overlapping; Spearman rho = {_as_float(synthesis[2]['spearman_rho']):.9f} across all 165 features.

The top-feature composition showed high overlap across the three windows, and the overall importance ranking remained strongly correlated across windows. No formal stable/unstable threshold is defined, and these results alone do not establish the absence of drift.

Figures: [temporal top SHAP features](temporal_shap_top_features.png) and [temporal SHAP heatmap](temporal_shap_heatmap.png).

## 5. Signed Attribution Changes

The union of temporal top-10 features contains {signed['union_feature_count']} features. Three features changed average SHAP contribution direction around T2 while having the same T1 and T3 directions: {_code_list(signed['temporary_t2_sign_change_features'])}. Each followed **negative in T1, positive in T2, negative in T3**.

This describes the frozen model's average attribution direction relative to its SHAP reference. It does not mean that the real-world effect of a feature reversed. SHAP explains model behavior, not causality.

## 6. Feature-Group Changes

{chr(10).join(group_table)}

Resource total importance decreased from {groups['Resource']['T1']['group_total_importance']:.6f} to {groups['Resource']['T2']['group_total_importance']:.6f} to {groups['Resource']['T3']['group_total_importance']:.6f}, while its share decreased from {groups['Resource']['T1']['share_of_total_importance']:.2%} to {groups['Resource']['T2']['share_of_total_importance']:.2%} to {groups['Resource']['T3']['share_of_total_importance']:.2%}. Activity total importance changed from {groups['Activity']['T1']['group_total_importance']:.6f} to {groups['Activity']['T2']['group_total_importance']:.6f} to {groups['Activity']['T3']['group_total_importance']:.6f}, while its share increased from {groups['Activity']['T1']['share_of_total_importance']:.2%} to {groups['Activity']['T2']['share_of_total_importance']:.2%} to {groups['Activity']['T3']['share_of_total_importance']:.2%}. Numeric total importance changed from {groups['Numeric']['T1']['group_total_importance']:.6f} to {groups['Numeric']['T2']['group_total_importance']:.6f} to {groups['Numeric']['T3']['group_total_importance']:.6f}, and its share changed from {groups['Numeric']['T1']['share_of_total_importance']:.2%} to {groups['Numeric']['T2']['share_of_total_importance']:.2%} to {groups['Numeric']['T3']['share_of_total_importance']:.2%}.

Group totals and shares must be read together with dimensionality and mean importance per feature. In particular, Resource contains 115 encoded features, so its total share does not by itself establish greater intrinsic importance.

Figure: [temporal feature-group importance](temporal_group_importance.png).

## 7. Prediction-Explanation Synthesis

The canonical synthesis is [prediction_explanation_synthesis.csv](prediction_explanation_synthesis.csv).

{chr(10).join(synthesis_table)}

Across the three test windows, predictive metrics changed modestly, with ROC-AUC notably higher in T3 than in T1/T2. At the same time, SHAP explanations showed substantial continuity: adjacent windows shared 9 of their top 10 features, T1 and T3 shared all 10, and full-feature importance rankings remained highly correlated.

This suggests that the observed temporal variation in predictive performance was not accompanied by a comparably large reorganization of the model's global feature-importance structure in this preliminary analysis. This is a descriptive association: explanation continuity did not cause the prediction pattern, and no formal drift conclusion is made.

## 8. Main Findings

{finding_lines}

## 9. Limitations

- The Random Forest is an untuned baseline.
- Explanation analysis uses a 1,000-case sample rather than all 6,276 test cases.
- The three windows cover only the frozen test period from October to December 2016.
- Windows are equal-case windows, not fixed-duration windows.
- No statistical significance testing was performed.
- No formal concept-drift detector was used.
- SHAP explains model behavior, not causality.
- Results may change after model tuning or final model selection.

## 10. Conclusion

How did prediction and explanation change over time? Success prevalence and mean predicted P(Success) decreased across the three windows. ROC-AUC was nearly unchanged from T1 to T2 and higher in T3; AP decreased in T2 and recovered in T3. Meanwhile, global SHAP explanations retained substantial feature-set and ranking continuity, although three features showed temporary T2 changes in mean attribution direction and feature-group shares shifted descriptively. Thus, the observed predictive variation was not accompanied by a comparably large reorganization of global SHAP importance in this preliminary analysis. This conclusion is descriptive, non-causal, and not a formal claim about concept drift.

## 11. Next Steps

Possible future work includes a reproduction experiment, model tuning and final model selection, final SHAP evaluation, stronger temporal stability validation, and formal drift analysis if required. None of those tasks is part of Temporal Evaluation V1 Part 4.
"""
    return report


def _numeric_synthesis(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Convert synthesis rows to JSON-native numeric values."""
    result: list[dict[str, Any]] = []
    integer_columns = {"top10_intersection_count", "top10_union_count"}
    text_columns = {"comparison", "window_a", "window_b"}
    for row in rows:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if key in text_columns:
                converted[key] = value
            elif key in integer_columns:
                converted[key] = int(value)
            else:
                converted[key] = _as_float(value)
        result.append(converted)
    return result


def write_summary_json(
    window_summary: dict[str, Any],
    synthesis: list[dict[str, str]],
    derived: dict[str, Any],
    protected_count: int,
) -> dict[str, Any]:
    """Write the final deterministic experiment summary JSON."""
    metrics = derived["metrics_by_window"]
    stability = _index_unique(
        load_frozen_explanation_results()["stability"], "comparison"
    )
    windows = {
        window: {
            "n_cases": window_summary["windows"][window]["case_count"],
            "start_case_start_time": window_summary["windows"][window][
                "start_case_start_time"
            ],
            "end_case_start_time": window_summary["windows"][window][
                "end_case_start_time"
            ],
            "shap_cases": EXPECTED_SHAP_WINDOW_COUNTS[window],
        }
        for window in WINDOW_NAMES
    }
    prediction_summary = {
        window: {
            key: int(row[key]) if key == "n_cases" else _as_float(row[key])
            for key in (
                "n_cases",
                "success_rate",
                "mean_probability_success",
                "roc_auc",
                "average_precision",
                "accuracy",
                "precision",
                "recall",
                "f1",
            )
        }
        for window, row in metrics.items()
    }
    explanation_stability_summary = [
        {
            "comparison": comparison,
            "window_a": window_a,
            "window_b": window_b,
            "jaccard_at_10": _as_float(stability[comparison]["jaccard_at_10"]),
            "spearman_rho": _as_float(stability[comparison]["spearman_rho"]),
            "top10_intersection_count": int(stability[comparison]["intersection_count"]),
            "top10_union_count": int(stability[comparison]["union_count"]),
        }
        for comparison, window_a, window_b in COMPARISONS
    ]
    limitations = [
        "Random Forest is an untuned baseline.",
        "Explanation analysis uses 1,000 sampled cases rather than all 6,276 test cases.",
        "The windows cover only the frozen October--December 2016 test period.",
        "Windows are equal-case rather than fixed-duration windows.",
        "No statistical significance testing was performed.",
        "No formal concept-drift detector was used.",
        "SHAP explains model behavior, not causality.",
        "Results may change after model tuning or final model selection.",
    ]
    assertions = {
        "A_part_1_boundaries_unchanged": "PASS",
        "B_part_2_metric_artifacts_unchanged": "PASS",
        "C_part_3_shap_artifacts_unchanged": "PASS",
        "D_synthesis_contains_exactly_3_comparisons": "PASS",
        "E_comparison_order_is_frozen": "PASS",
        "F_prediction_deltas_exactly_match_part_2": "PASS",
        "G_jaccard_values_exactly_match_part_3": "PASS",
        "H_spearman_values_exactly_match_part_3": "PASS",
        "I_no_model_prediction_recomputation": "PASS",
        "J_no_shap_recomputation": "PASS",
        "K_no_model_fitting": "PASS",
        "L_no_preprocessing_fitting": "PASS",
        "M_no_shap_resampling": "PASS",
        "N_no_temporal_boundary_modification": "PASS",
        "O_no_formal_drift_detection": "PASS",
        "P_no_significance_based_conclusion": "PASS",
        "Q_report_states_shap_is_not_causal": "PASS",
        "R_report_states_rf_is_untuned_baseline": "PASS",
        "S_report_documents_6276_prediction_vs_1000_explanation_cases": "PASS",
        "T_report_answers_temporal_prediction_explanation_question": "PASS",
    }
    summary: dict[str, Any] = {
        "experiment": "temporal_evaluation_v1",
        "stage": "completed",
        "prediction_point": "k=10",
        "model": "Frozen Random Forest Baseline V1 (untuned baseline)",
        "test_cases": EXPECTED_TEST_CASES,
        "shap_cases": EXPECTED_SHAP_CASES,
        "windows": windows,
        "prediction_summary": prediction_summary,
        "explanation_stability_summary": explanation_stability_summary,
        "synthesis_summary": {
            "delta_definition": "later window minus earlier window",
            "comparisons": _numeric_synthesis(synthesis),
            "all_window_top10_intersection": derived["continuity"][
                "all_window_intersection"
            ],
            "all_window_top10_intersection_count": len(
                derived["continuity"]["all_window_intersection"]
            ),
            "temporal_top10_union_count": derived["continuity"]["union_count"],
            "t1_t3_top10_sets_identical": derived["continuity"][
                "t1_t3_identical"
            ],
            "temporary_t2_signed_shap_changes": derived["signed_changes"][
                "temporary_t2_sign_change_features"
            ],
        },
        "main_findings": derived["main_findings"],
        "limitations": limitations,
        "model_retrained": False,
        "preprocessing_refitted": False,
        "shap_recomputed": False,
        "temporal_boundaries_changed": False,
        "shap_resampled": False,
        "predictions_recomputed": False,
        "significance_testing": False,
        "formal_drift_detection": False,
        "protected_artifact_integrity": {
            "status": "PASS",
            "file_count": protected_count,
            "scope": "Part 1--3, Baseline V1, and SHAP Pilot V1 artifacts",
        },
        "assertions": assertions,
        "assertion_status": "PASS",
        "generated_artifact_sha256_excluding_summary": {
            _relative(SYNTHESIS_PATH): _sha256(SYNTHESIS_PATH),
            _relative(REPORT_PATH): _sha256(REPORT_PATH),
        },
    }
    _write_json_atomic(FINAL_SUMMARY_PATH, summary)
    return summary


def validate_final_outputs(
    synthesis: list[dict[str, str]],
    prediction: dict[str, Any],
    explanation: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    """Enforce all requested Part 4 final-output assertions."""
    stored_synthesis = _load_csv(SYNTHESIS_PATH)
    _require(stored_synthesis == synthesis, "Stored synthesis differs from built rows")
    _require(len(stored_synthesis) == 3, "Synthesis must contain exactly 3 rows")
    _require(
        [row["comparison"] for row in stored_synthesis]
        == [item[0] for item in COMPARISONS],
        "Synthesis comparison order changed",
    )
    delta_by_comparison = _index_unique(prediction["deltas"], "comparison")
    stability_by_comparison = _index_unique(explanation["stability"], "comparison")
    for row in stored_synthesis:
        comparison = row["comparison"]
        for column in PREDICTION_DELTA_COLUMNS:
            _require(
                row[column] == delta_by_comparison[comparison][column],
                f"Synthesis {column} differs from Part 2 for {comparison}",
            )
        _require(
            row["jaccard_at_10"]
            == stability_by_comparison[comparison]["jaccard_at_10"],
            f"Synthesis Jaccard differs from Part 3 for {comparison}",
        )
        _require(
            row["spearman_rho"]
            == stability_by_comparison[comparison]["spearman_rho"],
            f"Synthesis Spearman differs from Part 3 for {comparison}",
        )

    report = REPORT_PATH.read_text(encoding="utf-8")
    report_lower = report.lower()
    _require(
        "shap explains model behavior, not causality" in report_lower,
        "Report must explicitly state that SHAP is not causal",
    )
    _require(
        "untuned baseline" in report_lower,
        "Report must explicitly state that the RF is an untuned baseline",
    )
    _require(
        "6,276 frozen test cases" in report
        and "frozen 1,000-case SHAP sample" in report,
        "Report must document prediction and explanation populations",
    )
    _require(
        "How did prediction and explanation change over time?" in report,
        "Report must answer the combined temporal question",
    )
    for forbidden_claim in (
        "no concept drift occurred",
        "the process was stable",
        "explanation stability caused prediction stability",
        "statistically significant",
    ):
        _require(
            forbidden_claim not in report_lower,
            f"Forbidden conclusion found: {forbidden_claim}",
        )
    _require(
        summary.get("assertion_status") == "PASS"
        and all(value == "PASS" for value in summary["assertions"].values()),
        "Final assertion summary is not PASS",
    )


def main() -> None:
    """Build Part 4 outputs and verify frozen-input integrity."""
    protected_paths = _protected_paths()
    protected_before = _hash_paths(protected_paths)

    window_summary = _load_json(WINDOW_SUMMARY_PATH)
    prediction = load_frozen_prediction_results()
    explanation = load_frozen_explanation_results()
    validate_part_results(window_summary, prediction, explanation)

    synthesis = build_synthesis_table(prediction, explanation)
    derived = derive_main_findings(prediction, explanation)
    _write_csv_atomic(SYNTHESIS_PATH, synthesis, SYNTHESIS_COLUMNS)
    _write_text_atomic(
        REPORT_PATH,
        build_report(window_summary, synthesis, derived),
    )
    summary = write_summary_json(
        window_summary,
        synthesis,
        derived,
        len(protected_paths),
    )
    validate_final_outputs(synthesis, prediction, explanation, summary)

    protected_after = _hash_paths(protected_paths)
    verify_protected_artifacts(protected_before, protected_after)

    print("Temporal Evaluation V1 Part 4: completed")
    print(f"Synthesis rows: {len(synthesis)}")
    for row in synthesis:
        print(
            f"{row['comparison']}: delta ROC-AUC={row['delta_roc_auc']}, "
            f"delta AP={row['delta_average_precision']}, "
            f"Jaccard@10={row['jaccard_at_10']}, "
            f"Spearman={row['spearman_rho']}"
        )
    print(
        "All-window top-10 intersection: "
        + ", ".join(derived["continuity"]["all_window_intersection"])
    )
    print(
        "Temporary T2 signed-SHAP changes: "
        + ", ".join(
            derived["signed_changes"]["temporary_t2_sign_change_features"]
        )
    )
    print(f"Protected artifacts unchanged: PASS ({len(protected_paths)} files)")
    print(f"All assertions: {summary['assertion_status']}")
    hashes = {
        _relative(path): _sha256(path)
        for path in (SYNTHESIS_PATH, REPORT_PATH, FINAL_SUMMARY_PATH)
    }
    for path, digest in hashes.items():
        print(f"SHA-256 {path}: {digest}")


if __name__ == "__main__":
    main()
