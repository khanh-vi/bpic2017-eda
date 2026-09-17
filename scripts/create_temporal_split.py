"""Create the frozen Baseline V1 Part 3 temporal train/test split.

This script operates only on case metadata and the existing target. It does
not transform predictive features, construct vocabularies, fit preprocessing,
or train/evaluate a model.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_TABLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_k10"
    / "case_features_k10.parquet"
)
ELIGIBILITY_SUMMARY_PATH = (
    PROJECT_ROOT / "results" / "baseline_v1" / "eligibility_summary.json"
)
FEATURE_MANIFEST_PATH = (
    PROJECT_ROOT / "results" / "baseline_v1" / "feature_manifest.json"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "baseline_v1"
SPLIT_SUMMARY_PATH = OUTPUT_DIR / "split_summary.json"
TRAIN_IDS_PATH = OUTPUT_DIR / "train_case_ids.csv"
TEST_IDS_PATH = OUTPUT_DIR / "test_case_ids.csv"

PREDICTION_POINT = 10
NOMINAL_TRAIN_RATIO = 0.8
EXPECTED_CASES = 31_376
EXPECTED_TARGET_MAPPING = {
    "A_Pending": 1,
    "A_Cancelled": 0,
    "A_Denied": 0,
}
EXPECTED_CLASS_MAPPING = {"Success": 1, "Unsuccessful": 0}
EXPECTED_TARGET_COUNTS = {0: 14_148, 1: 17_228}
EXPECTED_METADATA_COLUMNS = ["case_id", "case_start_time"]
EXPECTED_TARGET_COLUMN = "target"


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from an existing validated artifact."""
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    """Return a content hash used to prove the source Parquet was untouched."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def load_case_features(
    feature_path: Path = FEATURE_TABLE_PATH,
    eligibility_path: Path = ELIGIBILITY_SUMMARY_PATH,
    manifest_path: Path = FEATURE_MANIFEST_PATH,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Load Part 2 case features and verify its declared metadata contract."""
    eligibility = _load_json(eligibility_path)
    manifest = _load_json(manifest_path)

    if not feature_path.is_file():
        raise FileNotFoundError(f"Case-feature dataset not found: {feature_path}")

    metadata_columns = manifest.get("metadata_columns")
    target_column = manifest.get("target_column")
    if metadata_columns != EXPECTED_METADATA_COLUMNS:
        raise AssertionError(
            "Unexpected metadata columns in feature manifest: "
            f"{metadata_columns!r}"
        )
    if target_column != EXPECTED_TARGET_COLUMN:
        raise AssertionError(
            f"Unexpected target column in feature manifest: {target_column!r}"
        )
    if manifest.get("target_mapping") != EXPECTED_TARGET_MAPPING:
        raise AssertionError(
            "Target mapping changed from the validated Part 2 contract: "
            f"{manifest.get('target_mapping')!r}"
        )

    required_columns = set(metadata_columns + [target_column])
    case_features = pd.read_parquet(
        feature_path,
        columns=metadata_columns + [target_column],
        engine="pyarrow",
    )
    missing_columns = sorted(required_columns.difference(case_features.columns))
    if missing_columns:
        raise ValueError(
            f"Case-feature dataset is missing required columns: {missing_columns}"
        )

    predictive_columns = manifest.get("predictive_feature_columns")
    if not isinstance(predictive_columns, list) or not predictive_columns:
        raise ValueError("Feature manifest has no predictive feature column list.")
    assert case_features.columns.tolist() == metadata_columns + [target_column]

    case_features = case_features.copy()
    case_features["case_start_time"] = pd.to_datetime(
        case_features["case_start_time"], utc=True, errors="raise"
    )
    return case_features, eligibility, manifest


def determine_temporal_cutoff(
    case_features: pd.DataFrame,
    train_ratio: float = NOMINAL_TRAIN_RATIO,
) -> tuple[pd.DataFrame, int, int, pd.Timestamp]:
    """Order cases and choose the closest split that never divides a timestamp.

    The returned actual split index is the number of training rows. If the
    nominal boundary falls within a tied timestamp group, the whole group is
    placed on whichever side produces the count closest to the nominal index.
    A tie in distance is resolved toward the earlier split (the tied timestamp
    remains in test), without consulting the target.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    ordered = case_features.sort_values(
        ["case_start_time", "case_id"], kind="mergesort"
    ).reset_index(drop=True)
    nominal_split_index = math.floor(len(ordered) * train_ratio)
    if not 0 < nominal_split_index < len(ordered):
        raise ValueError("Nominal split would produce an empty train or test set.")

    previous_time = ordered.loc[nominal_split_index - 1, "case_start_time"]
    next_time = ordered.loc[nominal_split_index, "case_start_time"]
    actual_split_index = nominal_split_index

    if previous_time == next_time:
        tied_time = previous_time
        group_start = int(
            ordered["case_start_time"].searchsorted(tied_time, side="left")
        )
        group_end = int(
            ordered["case_start_time"].searchsorted(tied_time, side="right")
        )
        candidates = [
            index
            for index in (group_start, group_end)
            if 0 < index < len(ordered)
        ]
        if not candidates:
            raise ValueError(
                "The boundary timestamp covers the full population, so a strict "
                "temporal holdout is impossible."
            )
        actual_split_index = min(
            candidates,
            key=lambda index: (abs(index - nominal_split_index), index),
        )

    boundary_timestamp = ordered.loc[
        actual_split_index - 1, "case_start_time"
    ]
    return ordered, nominal_split_index, actual_split_index, boundary_timestamp


def create_temporal_split(
    ordered: pd.DataFrame,
    actual_split_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice the chronologically ordered case table at the frozen boundary."""
    train = ordered.iloc[:actual_split_index].copy()
    test = ordered.iloc[actual_split_index:].copy()
    return train, test


def validate_split(
    original: pd.DataFrame,
    ordered: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    eligibility: dict[str, Any],
    manifest: dict[str, Any],
    source_hash_before: str,
    source_hash_after: str,
) -> dict[str, str]:
    """Enforce the Part 3 population, chronology, and non-transformation rules."""
    results: dict[str, str] = {}

    # A. The validated Part 2 population is unchanged.
    assert len(original) == EXPECTED_CASES
    assert eligibility.get("eligible_cases") == EXPECTED_CASES
    assert manifest.get("number_of_cases") == EXPECTED_CASES
    results["A_input_row_count"] = "PASS"

    # B. There is one row per case.
    assert original["case_id"].notna().all()
    assert original["case_id"].is_unique
    results["B_case_id_unique"] = "PASS"

    train_ids = set(train["case_id"])
    test_ids = set(test["case_id"])
    all_ids = set(original["case_id"])

    # C. No case belongs to both splits.
    assert train_ids.isdisjoint(test_ids)
    results["C_no_train_test_overlap"] = "PASS"

    # D. The split covers the complete eligible population.
    assert train_ids.union(test_ids) == all_ids
    results["D_complete_population_union"] = "PASS"

    # E. No row was added or removed.
    assert len(train) + len(test) == EXPECTED_CASES
    results["E_split_row_count"] = "PASS"

    # F. Targets remain binary in both splits.
    assert set(train["target"].unique()).issubset({0, 1})
    assert set(test["target"].unique()).issubset({0, 1})
    results["F_binary_targets"] = "PASS"

    # G. Both target classes occur in both splits.
    assert set(train["target"].unique()) == {0, 1}
    assert set(test["target"].unique()) == {0, 1}
    results["G_both_classes_in_each_split"] = "PASS"

    # H. Every training case starts strictly before every test case.
    assert train["case_start_time"].max() < test["case_start_time"].min()
    results["H_strict_temporal_order"] = "PASS"

    # I. Only metadata was loaded, and the source feature file is byte-identical.
    reconstructed = pd.concat([train, test], ignore_index=True)
    pd.testing.assert_frame_equal(
        reconstructed,
        ordered,
        check_exact=True,
    )
    assert source_hash_before == source_hash_after
    results["I_predictive_features_unmodified"] = "PASS"

    # J. This checkpoint creates ID metadata only and fits no preprocessing.
    assert original.columns.tolist() == [
        "case_id",
        "case_start_time",
        "target",
    ]
    results["J_no_preprocessing_fitted"] = "PASS"

    # K. The audited Part 1 assumptions recorded by Parts 1 and 2 still hold.
    assert eligibility.get("prediction_point") == PREDICTION_POINT
    assert eligibility.get("outcome_leakage_pass") is True
    assert eligibility.get("future_event_leakage_pass") is True
    assert eligibility.get("eligible_success") == EXPECTED_TARGET_COUNTS[1]
    assert eligibility.get("eligible_unsuccessful") == EXPECTED_TARGET_COUNTS[0]
    validation = manifest.get("validation", {})
    assert validation.get("part1_outcome_leakage_audit") == "PASS"
    assert validation.get("part1_future_event_leakage_audit") == "PASS"
    assert validation.get("part2_assertions") == "PASS"
    assert original["target"].value_counts().to_dict() == EXPECTED_TARGET_COUNTS
    results["K_part1_leakage_assumptions_valid"] = "PASS"

    return results


def summarize_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    nominal_split_index: int,
    boundary_timestamp: pd.Timestamp,
    assertions: dict[str, str],
    source_sha256: str,
) -> dict[str, Any]:
    """Build JSON-serializable split metadata and descriptive target counts."""
    train_success = int(train["target"].eq(1).sum())
    train_unsuccessful = int(train["target"].eq(0).sum())
    test_success = int(test["target"].eq(1).sum())
    test_unsuccessful = int(test["target"].eq(0).sum())
    train_success_rate = train_success / len(train)
    test_success_rate = test_success / len(test)

    return {
        "split_type": "temporal_holdout",
        "prediction_point": PREDICTION_POINT,
        "case_id_column": "case_id",
        "case_start_time_column": "case_start_time",
        "target_column": "target",
        "target_mapping": EXPECTED_TARGET_MAPPING,
        "class_mapping": EXPECTED_CLASS_MAPPING,
        "nominal_train_ratio": NOMINAL_TRAIN_RATIO,
        "nominal_split_index": nominal_split_index,
        "actual_split_index": len(train),
        "boundary_adjustment_cases": len(train) - nominal_split_index,
        "timestamp_boundary_adjusted": len(train) != nominal_split_index,
        "total_cases": len(train) + len(test),
        "train_cases": len(train),
        "test_cases": len(test),
        "actual_train_percentage": len(train) / (len(train) + len(test)),
        "actual_test_percentage": len(test) / (len(train) + len(test)),
        "train_start": train["case_start_time"].min().isoformat(),
        "train_end": train["case_start_time"].max().isoformat(),
        "test_start": test["case_start_time"].min().isoformat(),
        "test_end": test["case_start_time"].max().isoformat(),
        "train_success": train_success,
        "train_unsuccessful": train_unsuccessful,
        "test_success": test_success,
        "test_unsuccessful": test_unsuccessful,
        "train_success_rate": train_success_rate,
        "test_success_rate": test_success_rate,
        "test_minus_train_success_rate": test_success_rate - train_success_rate,
        "boundary_timestamp": boundary_timestamp.isoformat(),
        "boundary_policy": (
            "train case_start_time <= boundary_timestamp; "
            "test case_start_time > boundary_timestamp"
        ),
        "strict_temporal_boundary": True,
        "canonical_train_ids": str(TRAIN_IDS_PATH.relative_to(PROJECT_ROOT)),
        "canonical_test_ids": str(TEST_IDS_PATH.relative_to(PROJECT_ROOT)),
        "source_feature_table_sha256": source_sha256,
        "predictive_features_modified": False,
        "preprocessing_fitted": False,
        "assertions": assertions,
    }


def print_summary(summary: dict[str, Any]) -> None:
    """Print the required Part 3 split and target summaries."""
    print("\n=== TEMPORAL SPLIT SUMMARY ===")
    print(f"Total eligible cases: {summary['total_cases']:,}")
    print(f"Nominal train ratio: {summary['nominal_train_ratio']:.1%}")
    print(f"Nominal split index: {summary['nominal_split_index']:,}")
    print()
    print(f"Actual train cases: {summary['train_cases']:,}")
    print(f"Actual test cases: {summary['test_cases']:,}")
    print(f"Actual train percentage: {summary['actual_train_percentage']:.4%}")
    print(f"Actual test percentage: {summary['actual_test_percentage']:.4%}")
    print()
    print(f"Train start: {summary['train_start']}")
    print(f"Train end: {summary['train_end']}")
    print()
    print(f"Test start: {summary['test_start']}")
    print(f"Test end: {summary['test_end']}")
    print()
    print(
        "Temporal boundary: "
        f"train <= {summary['boundary_timestamp']}; "
        f"test > {summary['boundary_timestamp']}"
    )
    print(
        "Timestamp-boundary adjustment: "
        f"{summary['boundary_adjustment_cases']:+,} cases"
    )

    print("\n=== TRAIN TARGET DISTRIBUTION ===")
    print(f"Success: {summary['train_success']:,}")
    print(f"Unsuccessful: {summary['train_unsuccessful']:,}")
    print(f"Success percentage: {summary['train_success_rate']:.4%}")
    print(
        "Unsuccessful percentage: "
        f"{1.0 - summary['train_success_rate']:.4%}"
    )

    print("\n=== TEST TARGET DISTRIBUTION ===")
    print(f"Success: {summary['test_success']:,}")
    print(f"Unsuccessful: {summary['test_unsuccessful']:,}")
    print(f"Success percentage: {summary['test_success_rate']:.4%}")
    print(
        "Unsuccessful percentage: "
        f"{1.0 - summary['test_success_rate']:.4%}"
    )
    print(
        "\ntest_success_rate - train_success_rate: "
        f"{summary['test_minus_train_success_rate']:+.6f}"
    )

    print("\n=== PART 3 ASSERTIONS ===")
    for name, result in summary["assertions"].items():
        print(f"{name}: {result}")


def save_split_artifacts(
    train: pd.DataFrame,
    test: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path = OUTPUT_DIR,
) -> tuple[Path, Path, Path]:
    """Save the canonical case-ID lists and their audit summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train[["case_id"]].to_csv(TRAIN_IDS_PATH, index=False, encoding="utf-8")
    test[["case_id"]].to_csv(TEST_IDS_PATH, index=False, encoding="utf-8")
    with SPLIT_SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
        file.write("\n")

    saved_train = pd.read_csv(TRAIN_IDS_PATH, encoding="utf-8")
    saved_test = pd.read_csv(TEST_IDS_PATH, encoding="utf-8")
    assert saved_train.columns.tolist() == ["case_id"]
    assert saved_test.columns.tolist() == ["case_id"]
    assert saved_train["case_id"].tolist() == train["case_id"].tolist()
    assert saved_test["case_id"].tolist() == test["case_id"].tolist()
    return SPLIT_SUMMARY_PATH, TRAIN_IDS_PATH, TEST_IDS_PATH


def main() -> None:
    """Create, validate, report, and freeze the temporal holdout."""
    source_hash_before = _sha256(FEATURE_TABLE_PATH)
    case_features, eligibility, manifest = load_case_features()
    ordered, nominal_index, actual_index, boundary = determine_temporal_cutoff(
        case_features
    )
    train, test = create_temporal_split(ordered, actual_index)
    assertions = validate_split(
        case_features,
        ordered,
        train,
        test,
        eligibility,
        manifest,
        source_hash_before,
        _sha256(FEATURE_TABLE_PATH),
    )
    summary = summarize_split(
        train,
        test,
        nominal_index,
        boundary,
        assertions,
        source_hash_before,
    )
    summary_path, train_path, test_path = save_split_artifacts(
        train, test, summary
    )
    print_summary(summary)
    print("\nAll Baseline V1 Part 3 assertions passed.")
    print(f"Saved: {summary_path}")
    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")


if __name__ == "__main__":
    main()
