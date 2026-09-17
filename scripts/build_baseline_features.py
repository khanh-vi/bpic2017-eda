"""Build leakage-safe Baseline V1 case features at prefix k=10.

This is Part 2 only. It reuses the Part 1 population and prefix builders, then
creates an intermediate case-level representation without fitting an encoder,
preprocessor, feature selector, or model.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from build_baseline_dataset import (
    ACTIVITY,
    CASE_ID,
    OUTCOME_ACTIVITIES,
    PREDICTION_POINT,
    TIMESTAMP,
    build_case_summary,
    build_safe_prefix,
    calculate_summary,
    determine_eligibility,
    load_event_log,
    order_events,
    print_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_MATRIX_PATH = PROJECT_ROOT / "results" / "feature_availability_matrix.csv"
PART1_ELIGIBLE_PATH = (
    PROJECT_ROOT / "results" / "baseline_v1" / "eligible_cases_k10.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "baseline_k10"
FEATURE_TABLE_PATH = OUTPUT_DIR / "case_features_k10.parquet"
MANIFEST_PATH = PROJECT_ROOT / "results" / "baseline_v1" / "feature_manifest.json"

EXPECTED_ELIGIBLE_CASES = 31_376
EXPECTED_PREFIX_POSITIONS = tuple(range(1, PREDICTION_POINT + 1))

STATIC_OUTPUT_NAMES = {
    "case:LoanGoal": "loan_goal",
    "case:ApplicationType": "application_type",
    "case:RequestedAmount": "requested_amount",
}
COUNT_MAP_OUTPUT_NAMES = {
    "concept:name": "activity_counts",
    "Action": "action_counts",
    "EventOrigin": "event_origin_counts",
    "org:resource": "resource_counts",
    "lifecycle:transition": "lifecycle_transition_counts",
}
SUMMARY_FEATURE_COLUMNS = [
    "n_unique_activities",
    "n_unique_resources",
    "prefix_duration_seconds",
    "mean_event_gap_seconds",
]
METADATA_COLUMNS = ["case_id", "case_start_time"]
TARGET_COLUMN = "target"

REQUIRED_MATRIX_COLUMNS = {
    "feature",
    "feature_group",
    "availability_at_k",
    "availability_decision",
    "model_handling",
    "reason",
    "notes",
}
ALLOWED_DECISIONS = {
    "Available",
    "Conditionally Available",
    "Leakage Risk",
}

# Every Available matrix row must match one unambiguous handling rule. An
# unfamiliar combination stops the build instead of silently inventing policy.
AVAILABLE_HANDLING_POLICY = {
    ("case-level", "Candidate model feature"): "raw_static",
    ("event-process", "Prefix-safe encoding or aggregation"): "count_map",
    (
        "event-process",
        "Evaluate raw-resource generalization separately",
    ): "count_map",
    ("event-process", "Use prefix-derived temporal features only"): "timing",
    ("identifier", "Exclude as a raw identifier"): "identifier_only",
}


def load_feature_availability(
    matrix_path: Path = FEATURE_MATRIX_PATH,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Load the matrix and convert its explicit decisions into feature policy."""
    if not matrix_path.is_file():
        raise FileNotFoundError(f"Feature Availability Matrix not found: {matrix_path}")

    matrix = pd.read_csv(matrix_path, encoding="utf-8")
    missing_columns = sorted(REQUIRED_MATRIX_COLUMNS.difference(matrix.columns))
    if missing_columns:
        raise ValueError(
            "Feature Availability Matrix is missing required columns: "
            f"{missing_columns}"
        )
    if matrix["feature"].isna().any() or not matrix["feature"].is_unique:
        raise ValueError("Matrix feature names must be non-null and unique.")

    unknown_decisions = sorted(
        set(matrix["availability_decision"].dropna()).difference(
            ALLOWED_DECISIONS
        )
    )
    if matrix["availability_decision"].isna().any() or unknown_decisions:
        raise ValueError(
            "Matrix contains missing or unknown availability decisions: "
            f"{unknown_decisions}"
        )

    available = matrix.loc[matrix["availability_decision"].eq("Available")]
    policy: dict[str, list[str]] = {
        "available_considered": available["feature"].tolist(),
        "raw_static": [],
        "count_map": [],
        "timing": [],
        "identifier_only": [],
        "conditional_omitted": matrix.loc[
            matrix["availability_decision"].eq("Conditionally Available"),
            "feature",
        ].tolist(),
        "leakage_risk_omitted": matrix.loc[
            matrix["availability_decision"].eq("Leakage Risk"), "feature"
        ].tolist(),
    }

    for row in available.itertuples(index=False):
        key = (row.feature_group, row.model_handling)
        handling = AVAILABLE_HANDLING_POLICY.get(key)
        if handling is None:
            raise ValueError(
                "Ambiguous handling for Available matrix feature "
                f"{row.feature!r}: group={row.feature_group!r}, "
                f"model_handling={row.model_handling!r}."
            )
        policy[handling].append(row.feature)

    if set(policy["raw_static"]) != set(STATIC_OUTPUT_NAMES):
        raise ValueError(
            "Available static features do not match the implemented raw-static "
            f"mapping: {policy['raw_static']}"
        )
    if set(policy["count_map"]) != set(COUNT_MAP_OUTPUT_NAMES):
        raise ValueError(
            "Available event aggregation features do not match the implemented "
            f"count-map mapping: {policy['count_map']}"
        )
    if policy["timing"] != [TIMESTAMP]:
        raise ValueError(
            "Available timing policy is ambiguous; expected only "
            f"{TIMESTAMP!r}, found {policy['timing']}."
        )

    return matrix, policy


def _json_count_map(values: pd.Series) -> str:
    """Serialize observed per-case values without creating a global vocabulary."""
    if values.isna().any():
        raise ValueError(f"Count-map source {values.name!r} contains missing values.")
    counts = Counter(str(value) for value in values)
    return json.dumps(
        dict(sorted(counts.items())),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_timing_features(prefix: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic timing features from prefix timestamps only."""
    if prefix["event_pos"].gt(PREDICTION_POINT).any():
        raise AssertionError("Future events were supplied to timing feature logic.")

    grouped = prefix.groupby(CASE_ID, sort=False)
    timing = grouped[TIMESTAMP].agg(
        case_start_time="min",
        prefix_end_time="max",
    )
    timing["prefix_duration_seconds"] = (
        timing["prefix_end_time"] - timing["case_start_time"]
    ).dt.total_seconds()

    gaps = grouped[TIMESTAMP].diff().dt.total_seconds()
    if gaps.dropna().lt(0).any():
        raise AssertionError("A prefix contains a negative event-time gap.")
    timing["mean_event_gap_seconds"] = (
        gaps.groupby(prefix[CASE_ID], sort=False).mean().reindex(timing.index)
    ).fillna(0.0)
    timing = timing.drop(columns="prefix_end_time")

    if timing[["prefix_duration_seconds", "mean_event_gap_seconds"]].isna().any().any():
        raise AssertionError("Timing features contain undefined values.")
    return timing


def build_prefix_case_features(
    prefix: pd.DataFrame,
    eligible_summary: pd.DataFrame,
    policy: dict[str, list[str]],
    k: int = PREDICTION_POINT,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Create one intermediate feature record per Part 1 eligible case."""
    if k != PREDICTION_POINT:
        raise ValueError(
            f"Baseline V1 Part 2 is fixed at k={PREDICTION_POINT}, received {k}."
        )

    required_sources = set(
        policy["raw_static"] + policy["count_map"] + policy["timing"]
    )
    missing_sources = sorted(required_sources.difference(prefix.columns))
    if missing_sources:
        raise ValueError(f"Included matrix source features are missing: {missing_sources}")

    positions = prefix.groupby(CASE_ID, sort=False)["event_pos"].agg(tuple)
    if not positions.map(lambda value: value == EXPECTED_PREFIX_POSITIONS).all():
        raise AssertionError("Every eligible prefix must contain positions 1..10.")
    if not positions.index.equals(eligible_summary.index):
        raise AssertionError("Prefix cases do not match Part 1 eligible cases in order.")

    feature_table = pd.DataFrame(index=eligible_summary.index.copy())
    feature_table.index.name = CASE_ID
    feature_table[TARGET_COLUMN] = eligible_summary[TARGET_COLUMN].astype("int8")

    static_sources = policy["raw_static"]
    static_distinct = prefix.groupby(CASE_ID, sort=False)[static_sources].nunique(
        dropna=False
    )
    if not static_distinct.eq(1).all().all():
        inconsistent = static_distinct.ne(1).sum().loc[lambda values: values.gt(0)]
        raise ValueError(
            "Included static features are missing or inconsistent within prefixes: "
            f"{inconsistent.to_dict()}"
        )
    static = prefix.groupby(CASE_ID, sort=False)[static_sources].first()
    static = static.rename(columns=STATIC_OUTPUT_NAMES)
    feature_table = feature_table.join(static, validate="one_to_one")

    count_map_columns: list[str] = []
    for source in policy["count_map"]:
        output_column = COUNT_MAP_OUTPUT_NAMES[source]
        counts = prefix.groupby(CASE_ID, sort=False)[source].agg(_json_count_map)
        feature_table[output_column] = counts
        count_map_columns.append(output_column)

    feature_table["n_unique_activities"] = (
        prefix.groupby(CASE_ID, sort=False)[ACTIVITY].nunique().astype("int16")
    )
    feature_table["n_unique_resources"] = (
        prefix.groupby(CASE_ID, sort=False)["org:resource"]
        .nunique()
        .astype("int16")
    )
    timing = build_timing_features(prefix)
    feature_table = feature_table.join(timing, validate="one_to_one")

    feature_table = feature_table.reset_index().rename(columns={CASE_ID: "case_id"})
    predictive_columns = (
        list(STATIC_OUTPUT_NAMES.values())
        + count_map_columns
        + SUMMARY_FEATURE_COLUMNS
    )
    ordered_columns = METADATA_COLUMNS + [TARGET_COLUMN] + predictive_columns
    feature_table = feature_table[ordered_columns]

    numeric_columns = [
        column
        for column in predictive_columns
        if is_numeric_dtype(feature_table[column])
    ]
    categorical_columns = [
        column for column in predictive_columns if column not in numeric_columns
    ]
    return feature_table, numeric_columns, categorical_columns


def validate_part1_checkpoint(
    summary: dict[str, object], eligible_summary: pd.DataFrame
) -> None:
    """Confirm Part 2 uses the validated Part 1 population unchanged."""
    expected = {
        "prediction_point": 10,
        "total_cases": 31_509,
        "valid_target_cases": 31_411,
        "unresolved_cases": 98,
        "cases_lt_k_events": 0,
        "outcome_at_or_before_k": 35,
        "eligible_cases": EXPECTED_ELIGIBLE_CASES,
        "eligible_success": 17_228,
        "eligible_unsuccessful": 14_148,
        "outcome_leakage_pass": True,
        "future_event_leakage_pass": True,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": summary.get(key)}
        for key, expected_value in expected.items()
        if summary.get(key) != expected_value
    }
    if mismatches:
        raise AssertionError(f"Part 1 checkpoint changed: {mismatches}")

    if not PART1_ELIGIBLE_PATH.is_file():
        raise FileNotFoundError(
            f"Validated Part 1 eligible-case output is missing: {PART1_ELIGIBLE_PATH}"
        )
    saved = pd.read_csv(
        PART1_ELIGIBLE_PATH,
        usecols=["case_id", "target"],
        encoding="utf-8",
    )
    recomputed = eligible_summary[["target"]].reset_index().rename(
        columns={CASE_ID: "case_id"}
    )
    recomputed["target"] = recomputed["target"].astype("int64")
    pd.testing.assert_frame_equal(saved, recomputed, check_dtype=False)


def validate_feature_table(
    feature_table: pd.DataFrame,
    prefix: pd.DataFrame,
    eligible_summary: pd.DataFrame,
    predictive_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    part1_summary: dict[str, object],
) -> dict[str, Any]:
    """Run Part 2 population, leakage, timing, and table-quality checks."""
    assert len(feature_table) == EXPECTED_ELIGIBLE_CASES
    assert feature_table["case_id"].is_unique
    assert eligible_summary["valid_target"].all()
    assert set(feature_table[TARGET_COLUMN].unique()).issubset({0, 1})
    assert not prefix[ACTIVITY].isin(OUTCOME_ACTIVITIES).any()
    assert prefix["event_pos"].le(PREDICTION_POINT).all()
    assert prefix.groupby(CASE_ID, sort=False).size().eq(PREDICTION_POINT).all()
    assert not set(OUTCOME_ACTIVITIES).intersection(predictive_columns)
    assert "outcome_activity" not in predictive_columns
    assert "outcome_position" not in predictive_columns
    assert "n_events" not in predictive_columns
    assert "case_id" not in predictive_columns
    assert "case_start_time" not in predictive_columns
    assert TARGET_COLUMN not in predictive_columns
    assert all("outcome" not in column.lower() for column in predictive_columns)

    for count_column in COUNT_MAP_OUTPUT_NAMES.values():
        count_keys = feature_table[count_column].map(
            lambda value: set(json.loads(value))
        )
        assert not count_keys.map(
            lambda keys: bool(keys.intersection(OUTCOME_ACTIVITIES))
        ).any(), f"{count_column} contains a target outcome value."
        count_totals = feature_table[count_column].map(
            lambda value: sum(json.loads(value).values())
        )
        assert count_totals.eq(PREDICTION_POINT).all(), (
            f"{count_column} does not account for exactly 10 prefix events."
        )

    recomputed_timing = build_timing_features(prefix).reset_index().rename(
        columns={CASE_ID: "case_id"}
    )
    timing_check = feature_table[
        [
            "case_id",
            "case_start_time",
            "prefix_duration_seconds",
            "mean_event_gap_seconds",
        ]
    ].merge(recomputed_timing, on="case_id", validate="one_to_one", suffixes=("", "_check"))
    pd.testing.assert_series_equal(
        timing_check["case_start_time"],
        timing_check["case_start_time_check"],
        check_names=False,
    )
    for column in ["prefix_duration_seconds", "mean_event_gap_seconds"]:
        pd.testing.assert_series_equal(
            timing_check[column],
            timing_check[f"{column}_check"],
            check_names=False,
        )

    assert part1_summary["outcome_leakage_pass"] is True
    assert part1_summary["future_event_leakage_pass"] is True
    assert set(numeric_columns).isdisjoint(categorical_columns)
    assert set(numeric_columns + categorical_columns) == set(predictive_columns)

    missing_value_counts = {
        column: int(count) for column, count in feature_table.isna().sum().items()
    }
    missing_values_nonzero = {
        column: count
        for column, count in missing_value_counts.items()
        if count > 0
    }
    constant_columns = [
        column
        for column in predictive_columns
        if feature_table[column].nunique(dropna=False) == 1
    ]
    return {
        "output_rows": int(len(feature_table)),
        "output_columns": int(feature_table.shape[1]),
        "predictive_feature_count": len(predictive_columns),
        "numeric_feature_count": len(numeric_columns),
        "categorical_or_structured_feature_count": len(categorical_columns),
        "missing_value_counts": missing_value_counts,
        "missing_value_counts_nonzero": missing_values_nonzero,
        "total_missing_values": int(feature_table.isna().sum().sum()),
        "constant_predictive_columns": constant_columns,
        "duplicate_case_id_rows": int(feature_table["case_id"].duplicated().sum()),
        "duplicate_full_rows": int(feature_table.duplicated().sum()),
        "duplicate_predictor_rows": int(
            feature_table.duplicated(subset=predictive_columns).sum()
        ),
        "part1_outcome_leakage_audit": "PASS",
        "part1_future_event_leakage_audit": "PASS",
        "part2_assertions": "PASS",
    }


def _matrix_records(matrix: pd.DataFrame, features: list[str]) -> list[dict[str, Any]]:
    """Return concise matrix-derived records in matrix order."""
    selected = matrix.loc[matrix["feature"].isin(features)]
    return selected[
        [
            "feature",
            "availability_decision",
            "model_handling",
            "reason",
            "notes",
        ]
    ].to_dict(orient="records")


def write_feature_manifest(
    matrix: pd.DataFrame,
    policy: dict[str, list[str]],
    feature_table: pd.DataFrame,
    predictive_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    validation: dict[str, Any],
    manifest_path: Path = MANIFEST_PATH,
) -> Path:
    """Document the matrix policy, feature provenance, and validation results."""
    included_sources = (
        policy["raw_static"] + policy["count_map"] + policy["timing"]
    )
    manifest = {
        "baseline_version": "v1_part2",
        "prediction_point": PREDICTION_POINT,
        "number_of_cases": int(len(feature_table)),
        "input_eligible_population": str(PART1_ELIGIBLE_PATH.relative_to(PROJECT_ROOT)),
        "feature_availability_matrix": str(FEATURE_MATRIX_PATH.relative_to(PROJECT_ROOT)),
        "output_dataset": str(FEATURE_TABLE_PATH.relative_to(PROJECT_ROOT)),
        "output_format": "parquet",
        "metadata_columns": METADATA_COLUMNS,
        "target_column": TARGET_COLUMN,
        "target_mapping": {
            "A_Pending": 1,
            "A_Cancelled": 0,
            "A_Denied": 0,
        },
        "available_source_features_considered": policy["available_considered"],
        "included_source_features": included_sources,
        "included_source_feature_details": _matrix_records(
            matrix, included_sources
        ),
        "excluded_available_features": _matrix_records(
            matrix, policy["identifier_only"]
        ),
        "omitted_conditional_features": _matrix_records(
            matrix, policy["conditional_omitted"]
        ),
        "omitted_leakage_risk_features": _matrix_records(
            matrix, policy["leakage_risk_omitted"]
        ),
        "static_source_features": {
            source: output for source, output in STATIC_OUTPUT_NAMES.items()
        },
        "derived_prefix_features": {
            "per_case_count_maps": {
                source: output for source, output in COUNT_MAP_OUTPUT_NAMES.items()
            },
            "prefix_summary": SUMMARY_FEATURE_COLUMNS,
        },
        "predictive_feature_columns": predictive_columns,
        "numeric_feature_columns": numeric_columns,
        "categorical_or_structured_feature_columns": categorical_columns,
        "activity_vocabulary_deferred_to_train_only_processing": True,
        "activity_vocabulary_strategy": (
            "activity_counts is a deterministic JSON object per case. No global "
            "activity vocabulary is created in Part 2. After the temporal split, "
            "fit/freeze keys on training cases only and handle unseen test keys "
            "explicitly. The same strategy applies to the other event count maps."
        ),
        "excluded_from_predictive_features": [
            "case_id",
            "case_start_time",
            "target",
            "outcome_activity",
            "outcome_position",
            "n_events",
        ],
        "validation": validation,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
        file.write("\n")
    return manifest_path


def print_feature_policy(
    matrix: pd.DataFrame, policy: dict[str, list[str]]
) -> None:
    """Print the matrix-driven inclusion and omission decisions."""
    included = policy["raw_static"] + policy["count_map"] + policy["timing"]
    print("\n=== FEATURE AVAILABILITY POLICY ===")
    print(f"Available source features considered: {policy['available_considered']}")
    print(f"Included: {included}")
    print("Excluded and reason:")
    for record in _matrix_records(matrix, policy["identifier_only"]):
        print(
            f"- {record['feature']}: {record['model_handling']} "
            f"({record['reason']})"
        )
    print(f"Conditionally Available omitted: {policy['conditional_omitted']}")
    print(f"Leakage Risk omitted: {policy['leakage_risk_omitted']}")


def save_feature_table(
    feature_table: pd.DataFrame,
    output_path: Path = FEATURE_TABLE_PATH,
) -> Path:
    """Write Parquet and verify the case-level round trip."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_parquet(output_path, index=False, engine="pyarrow")
    exported = pd.read_parquet(output_path, engine="pyarrow")
    assert exported.columns.tolist() == feature_table.columns.tolist()
    assert len(exported) == len(feature_table)
    assert exported["case_id"].is_unique
    return output_path


def main() -> None:
    """Rebuild Part 1 in memory, then produce and validate Part 2 features."""
    matrix, policy = load_feature_availability()
    print_feature_policy(matrix, policy)

    print("\nRebuilding the validated Part 1 population...")
    event_log = load_event_log()
    ordered_events = order_events(event_log)
    case_summary = determine_eligibility(build_case_summary(ordered_events))
    prefix_k10, eligible_summary = build_safe_prefix(
        ordered_events, case_summary
    )
    part1_summary = calculate_summary(case_summary, prefix_k10)
    print_summary(part1_summary)
    validate_part1_checkpoint(part1_summary, eligible_summary)

    feature_table, numeric_columns, categorical_columns = (
        build_prefix_case_features(prefix_k10, eligible_summary, policy)
    )
    predictive_columns = [
        column
        for column in feature_table.columns
        if column not in METADATA_COLUMNS + [TARGET_COLUMN]
    ]
    validation = validate_feature_table(
        feature_table,
        prefix_k10,
        eligible_summary,
        predictive_columns,
        numeric_columns,
        categorical_columns,
        part1_summary,
    )

    feature_path = save_feature_table(feature_table)
    manifest_path = write_feature_manifest(
        matrix,
        policy,
        feature_table,
        predictive_columns,
        numeric_columns,
        categorical_columns,
        validation,
    )

    print("\n=== PART 2 FEATURE TABLE ===")
    print(f"Rows: {validation['output_rows']:,}")
    print(f"Output columns: {validation['output_columns']:,}")
    print(f"Predictive feature columns: {validation['predictive_feature_count']:,}")
    print(f"Numeric features: {validation['numeric_feature_count']:,}")
    print(
        "Categorical/structured features: "
        f"{validation['categorical_or_structured_feature_count']:,}"
    )
    print(
        "Missing values (nonzero only): "
        f"{validation['missing_value_counts_nonzero']}"
    )
    print(
        "Constant predictive columns: "
        f"{validation['constant_predictive_columns']}"
    )
    print(
        f"Duplicate case ID rows: {validation['duplicate_case_id_rows']:,}"
    )
    print(
        "Duplicate predictor rows: "
        f"{validation['duplicate_predictor_rows']:,}"
    )
    print("All Part 2 assertions passed.")
    print(f"Saved: {feature_path}")
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
