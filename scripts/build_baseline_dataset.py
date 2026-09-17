"""Build and audit the leakage-safe BPIC 2017 population at prefix k=10.

This checkpoint intentionally stops at case eligibility and in-memory prefix
construction. It does not create model features or split/model the data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pm4py


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "BPI Challenge 2017.xes"
OUTPUT_DIR = PROJECT_ROOT / "results" / "baseline_v1"

PREDICTION_POINT = 10
CASE_ID = "case:concept:name"
ACTIVITY = "concept:name"
TIMESTAMP = "time:timestamp"

OUTCOME_ACTIVITIES = frozenset(
    {
        "A_Pending",
        "A_Cancelled",
        "A_Denied",
    }
)
TARGET_MAPPING = {
    "A_Pending": 1,
    "A_Cancelled": 0,
    "A_Denied": 0,
}


def load_event_log(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the unmodified XES event log used by the existing EDA."""
    if not data_path.is_file():
        raise FileNotFoundError(f"BPIC 2017 event log not found: {data_path}")
    return pm4py.read_xes(str(data_path))


def order_events(event_log: pd.DataFrame) -> pd.DataFrame:
    """Return events in deterministic case order with 1-based positions.

    The ordering matches the repository's existing k=10 implementation:
    timestamp is primary and source-row position is the stable tie-breaker.
    No event is removed or changed.
    """
    required_columns = {CASE_ID, ACTIVITY, TIMESTAMP}
    missing_columns = sorted(required_columns.difference(event_log.columns))
    if missing_columns:
        raise ValueError(f"Required columns are missing: {missing_columns}")

    null_counts = event_log[[CASE_ID, ACTIVITY, TIMESTAMP]].isna().sum()
    if null_counts.any():
        raise ValueError(
            "Core event columns contain missing values: "
            f"{null_counts[null_counts.gt(0)].to_dict()}"
        )

    ordered = event_log.copy()
    ordered[TIMESTAMP] = pd.to_datetime(
        ordered[TIMESTAMP], utc=True, errors="raise"
    )
    ordered["_source_order"] = range(len(ordered))
    ordered = ordered.sort_values(
        [CASE_ID, TIMESTAMP, "_source_order"],
        kind="mergesort",
    ).reset_index(drop=True)
    ordered["event_pos"] = (
        ordered.groupby(CASE_ID, sort=False).cumcount() + 1
    )

    first_positions = ordered.groupby(CASE_ID, sort=False)["event_pos"].min()
    assert first_positions.eq(1).all(), "Not every case starts at event_pos=1."
    return ordered


def build_case_summary(ordered_events: pd.DataFrame) -> pd.DataFrame:
    """Recompute case-level outcomes, labels, and outcome positions."""
    case_index = pd.Index(
        ordered_events[CASE_ID].drop_duplicates(), name=CASE_ID
    )
    outcome_events = ordered_events.loc[
        ordered_events[ACTIVITY].isin(OUTCOME_ACTIVITIES),
        [CASE_ID, ACTIVITY, "event_pos"],
    ]

    outcome_event_count = (
        outcome_events.groupby(CASE_ID, sort=False)
        .size()
        .reindex(case_index, fill_value=0)
    )
    distinct_outcome_count = (
        outcome_events.groupby(CASE_ID, sort=False)[ACTIVITY]
        .nunique()
        .reindex(case_index, fill_value=0)
    )
    contradictory_case_ids = distinct_outcome_count.index[
        distinct_outcome_count.gt(1)
    ]
    repeated_same_outcome = int(
        (outcome_event_count.gt(1) & distinct_outcome_count.eq(1)).sum()
    )

    print("\n=== GROUND-TRUTH VALIDATION ===")
    print(f"Cases with any outcome event: {outcome_event_count.gt(0).sum():,}")
    print(
        "Cases with repeated occurrences of one outcome: "
        f"{repeated_same_outcome:,}"
    )
    print(f"Cases with contradictory outcomes: {len(contradictory_case_ids):,}")

    if len(contradictory_case_ids):
        contradictory = (
            outcome_events.loc[
                outcome_events[CASE_ID].isin(contradictory_case_ids)
            ]
            .groupby(CASE_ID, sort=False)[ACTIVITY]
            .agg(lambda values: sorted(set(values)))
            .head(10)
            .to_dict()
        )
        raise ValueError(
            "Contradictory target activities found. No label was selected. "
            f"Affected cases: {len(contradictory_case_ids):,}; "
            f"sample: {contradictory}"
        )

    valid_case_ids = distinct_outcome_count.index[
        distinct_outcome_count.eq(1)
    ]
    valid_outcomes = (
        outcome_events.loc[outcome_events[CASE_ID].isin(valid_case_ids)]
        .groupby(CASE_ID, sort=False)
        .agg(
            outcome_activity=(ACTIVITY, "first"),
            outcome_position=("event_pos", "min"),
        )
    )

    case_summary = (
        ordered_events.groupby(CASE_ID, sort=False)
        .agg(
            case_start_time=(TIMESTAMP, "min"),
            n_events=("event_pos", "size"),
        )
        .join(outcome_event_count.rename("outcome_event_count"))
        .join(distinct_outcome_count.rename("distinct_outcome_count"))
        .join(valid_outcomes)
    )
    case_summary["outcome_position"] = case_summary[
        "outcome_position"
    ].astype("Int64")
    case_summary["valid_target"] = case_summary[
        "distinct_outcome_count"
    ].eq(1)
    case_summary["target"] = case_summary["outcome_activity"].map(
        TARGET_MAPPING
    ).astype("Int64")

    unmapped_valid_targets = case_summary.loc[
        case_summary["valid_target"] & case_summary["target"].isna()
    ]
    if not unmapped_valid_targets.empty:
        raise ValueError(
            "A valid outcome has no binary target mapping: "
            f"{unmapped_valid_targets['outcome_activity'].unique().tolist()}"
        )

    assert case_summary.index.is_unique, (
        "Case IDs are not unique at the case-summary level."
    )
    return case_summary


def determine_eligibility(
    case_summary: pd.DataFrame, k: int = PREDICTION_POINT
) -> pd.DataFrame:
    """Apply the sequential valid-target, length, and leakage-safe filters."""
    result = case_summary.copy()
    result["has_k_events"] = result["n_events"].ge(k)
    result["outcome_after_k"] = result["outcome_position"].gt(k).fillna(False)
    result[f"eligible_k{k}"] = (
        result["valid_target"]
        & result["has_k_events"]
        & result["outcome_after_k"]
    )
    return result


def build_safe_prefix(
    ordered_events: pd.DataFrame,
    case_summary: pd.DataFrame,
    k: int = PREDICTION_POINT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct prefix positions 1..k and enforce all Part 1 assertions."""
    eligible_column = f"eligible_k{k}"
    eligible_summary = case_summary.loc[case_summary[eligible_column]].copy()
    eligible_case_ids = eligible_summary.index
    prefix = ordered_events.loc[
        ordered_events[CASE_ID].isin(eligible_case_ids)
        & ordered_events["event_pos"].le(k)
    ].copy()

    prefix_sizes = prefix.groupby(CASE_ID, sort=False).size()

    # A. Exactly k prefix events per eligible case.
    assert len(prefix_sizes) == len(eligible_summary), (
        "Prefix does not contain every eligible case."
    )
    assert prefix_sizes.eq(k).all(), (
        "At least one eligible case does not have exactly k prefix events."
    )
    # B. No future position enters the prefix.
    assert prefix["event_pos"].le(k).all(), (
        f"Prefix contains event_pos greater than {k}."
    )
    # C. No outcome activity enters the prefix.
    leaking_outcomes = prefix[ACTIVITY].isin(OUTCOME_ACTIVITIES)
    assert not leaking_outcomes.any(), (
        "Prefix contains an outcome activity: "
        f"{prefix.loc[leaking_outcomes, ACTIVITY].value_counts().to_dict()}"
    )
    # D. No unresolved case is eligible.
    assert eligible_summary["valid_target"].all(), (
        "An unresolved case is included in the eligible population."
    )
    # E. Each eligible case has one distinct outcome and one binary target.
    assert eligible_summary["distinct_outcome_count"].eq(1).all(), (
        "An eligible case does not have exactly one distinct outcome."
    )
    assert eligible_summary["target"].notna().all(), (
        "An eligible case has no binary target."
    )
    assert set(eligible_summary["target"].astype(int).unique()).issubset(
        {0, 1}
    ), "An eligible target is not binary."
    # F. Case-level rows are unique.
    assert eligible_summary.index.is_unique, (
        "Eligible case IDs are not unique at the case-summary level."
    )

    return prefix, eligible_summary


def calculate_summary(
    case_summary: pd.DataFrame,
    prefix: pd.DataFrame,
    k: int = PREDICTION_POINT,
) -> dict[str, object]:
    """Calculate descriptive and sequential eligibility counts."""
    valid_target = case_summary["valid_target"]
    has_k_events = case_summary["has_k_events"]
    outcome_at_or_before_k = (
        valid_target & case_summary["outcome_position"].le(k).fillna(False)
    )
    eligible = case_summary[f"eligible_k{k}"]

    valid_stage = case_summary.loc[valid_target]
    enough_events_stage = valid_stage.loc[valid_stage["has_k_events"]]
    sequential_lt_k = int((~valid_stage["has_k_events"]).sum())
    sequential_early_outcome = int(
        enough_events_stage["outcome_position"].le(k).fillna(False).sum()
    )
    eligible_targets = case_summary.loc[eligible, "target"].astype(int)

    return {
        "prediction_point": k,
        "total_cases": int(len(case_summary)),
        "valid_target_cases": int(valid_target.sum()),
        "unresolved_cases": int((~valid_target).sum()),
        "cases_lt_k_events": int((~has_k_events).sum()),
        "outcome_at_or_before_k": int(outcome_at_or_before_k.sum()),
        "eligible_cases": int(eligible.sum()),
        "eligible_success": int(eligible_targets.eq(1).sum()),
        "eligible_unsuccessful": int(eligible_targets.eq(0).sum()),
        "outcome_leakage_pass": bool(
            ~prefix[ACTIVITY].isin(OUTCOME_ACTIVITIES).any()
        ),
        "future_event_leakage_pass": bool(prefix["event_pos"].le(k).all()),
        "sequential_exclusion_counts": {
            "no_valid_target": int((~valid_target).sum()),
            "fewer_than_k_after_valid_target": sequential_lt_k,
            "outcome_at_or_before_k_after_length_filter": (
                sequential_early_outcome
            ),
        },
    }


def print_summary(summary: dict[str, object]) -> None:
    """Print the eligibility flow, distribution, and conditional audits."""
    total = int(summary["total_cases"])
    valid = int(summary["valid_target_cases"])
    eligible = int(summary["eligible_cases"])
    success = int(summary["eligible_success"])
    unsuccessful = int(summary["eligible_unsuccessful"])
    sequential = summary["sequential_exclusion_counts"]
    assert isinstance(sequential, dict)

    print("\n=== BASELINE K=10 ELIGIBILITY SUMMARY ===")
    print(f"Total cases: {total:,}")
    print(
        f"Cases with valid target: {valid:,} "
        f"({valid / total:.2%} of all cases)"
    )
    print(
        f"Unresolved cases: {int(summary['unresolved_cases']):,} "
        f"({int(summary['unresolved_cases']) / total:.2%} of all cases)"
    )
    print(
        "Cases with fewer than 10 total events (descriptive): "
        f"{int(summary['cases_lt_k_events']):,}"
    )
    print(
        "Cases with outcome at/before k=10 (valid targets, descriptive): "
        f"{int(summary['outcome_at_or_before_k']):,}"
    )

    print("\nSequential eligibility flow (mutually exclusive exclusions):")
    print(f"All cases: {total:,}")
    print(
        "-> valid-target cases: "
        f"{valid:,} (excluded {int(sequential['no_valid_target']):,})"
    )
    after_length = valid - int(sequential["fewer_than_k_after_valid_target"])
    print(
        "-> at least 10 events: "
        f"{after_length:,} "
        f"(excluded {int(sequential['fewer_than_k_after_valid_target']):,})"
    )
    print(
        "-> outcome strictly after k=10 / eligible: "
        f"{eligible:,} "
        "(excluded "
        f"{int(sequential['outcome_at_or_before_k_after_length_filter']):,})"
    )

    print("\n=== ELIGIBLE TARGET DISTRIBUTION ===")
    print(f"Success: {success:,} ({success / eligible:.2%})")
    print(f"Unsuccessful: {unsuccessful:,} ({unsuccessful / eligible:.2%})")

    if summary["outcome_leakage_pass"]:
        print("\nOutcome Leakage Audit: PASS")
    else:
        raise AssertionError("Outcome Leakage Audit: FAIL")

    if summary["future_event_leakage_pass"]:
        print("Future Event Leakage Audit: PASS")
    else:
        raise AssertionError("Future Event Leakage Audit: FAIL")


def save_outputs(
    summary: dict[str, object],
    eligible_summary: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
    k: int = PREDICTION_POINT,
) -> tuple[Path, Path]:
    """Save the lightweight summary and eligible case-level audit table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "eligibility_summary.json"
    eligible_path = output_dir / f"eligible_cases_k{k}.csv"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    export_columns = [
        "case_start_time",
        "n_events",
        "outcome_activity",
        "outcome_position",
        "target",
        f"eligible_k{k}",
    ]
    export = eligible_summary[export_columns].reset_index().rename(
        columns={CASE_ID: "case_id"}
    )
    export["target"] = export["target"].astype("int8")
    export.to_csv(eligible_path, index=False, encoding="utf-8")

    assert export["case_id"].is_unique, (
        "Exported eligible case IDs are not unique."
    )
    assert len(export) == int(summary["eligible_cases"]), (
        "Exported eligible-case count does not match the summary."
    )
    return summary_path, eligible_path


def main() -> None:
    """Run the Part 1 population checkpoint and leakage audits."""
    print(f"Dataset: {DATA_PATH}")
    print(f"Case ID column: {CASE_ID}")
    print(f"Activity column: {ACTIVITY}")
    print(f"Timestamp column: {TIMESTAMP}")
    print("Loading event log...")

    event_log = load_event_log()
    ordered_events = order_events(event_log)
    case_summary = build_case_summary(ordered_events)
    case_summary = determine_eligibility(case_summary)
    prefix_k10, eligible_summary = build_safe_prefix(
        ordered_events, case_summary
    )
    summary = calculate_summary(case_summary, prefix_k10)
    print_summary(summary)
    summary_path, eligible_path = save_outputs(summary, eligible_summary)

    print("\nAll Part 1 assertions passed.")
    print(f"Saved: {summary_path}")
    print(f"Saved: {eligible_path}")


if __name__ == "__main__":
    main()
