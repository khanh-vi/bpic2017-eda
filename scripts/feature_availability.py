"""Feature Availability analysis for the BPIC 2017 event log."""

from pathlib import Path

import pandas as pd
import pm4py


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "BPI Challenge 2017.xes"

K = 10
CASE_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"
TIME_COL = "time:timestamp"

OUTCOME_ACTIVITIES = {
    "A_Pending",
    "A_Cancelled",
    "A_Denied",
}

TARGET_MAPPING = {
    "A_Pending": 1,
    "A_Cancelled": 0,
    "A_Denied": 0,
}

CASE_LEVEL_FEATURES = [
    "case:LoanGoal",
    "case:ApplicationType",
    "case:RequestedAmount",
]

EXPECTED_K10_SUMMARY = {
    "prediction_point": 10,
    "total_cases": 31_509,
    "cases_with_outcome": 31_411,
    "cases_with_one_distinct_outcome": 31_411,
    "cases_with_ambiguous_outcome": 0,
    "cases_with_repeated_outcome_events": 1,
    "cases_without_outcome": 98,
    "cases_with_outcome_by_k": 35,
    "eligible_cases": 31_376,
    "prefix_rows": 313_760,
    "future_rows": 884_184,
    "outcome_inside_prefix": 0,
}


def load_event_log():
    """Load the unmodified BPIC 2017 XES source used by the EDA script."""
    return pm4py.read_xes(str(DATA_PATH))


def build_prediction_prefix(dataframe, k=K):
    """Construct and validate the observation boundary at prefix length ``k``."""
    df_work = dataframe.copy()
    df_work[TIME_COL] = pd.to_datetime(
        df_work[TIME_COL],
        utc=True,
        errors="raise",
    )

    # Source position is the deterministic tie-breaker for equal timestamps.
    # Source event rows are deliberately retained; no deduplication occurs.
    df_work["_original_order"] = range(len(df_work))
    df_work = df_work.sort_values(
        [CASE_COL, TIME_COL, "_original_order"],
        kind="mergesort",
    ).reset_index(drop=True)
    df_work["event_nr"] = (
        df_work.groupby(CASE_COL, sort=False).cumcount() + 1
    )

    first_event_numbers = df_work.groupby(CASE_COL)["event_nr"].min()
    assert first_event_numbers.eq(1).all(), "Not every case starts at event 1."

    outcome_events = df_work.loc[
        df_work[ACTIVITY_COL].isin(OUTCOME_ACTIVITIES),
        [CASE_COL, ACTIVITY_COL, "event_nr"],
    ]
    all_case_ids = pd.Index(
        df_work[CASE_COL].drop_duplicates(),
        name=CASE_COL,
    )
    outcome_event_count = (
        outcome_events.groupby(CASE_COL)
        .size()
        .reindex(all_case_ids, fill_value=0)
    )
    outcome_label_count = (
        outcome_events.groupby(CASE_COL)[ACTIVITY_COL]
        .nunique()
        .reindex(all_case_ids, fill_value=0)
    )

    # Repeated occurrences of one label are valid. Only cases containing more
    # than one distinct outcome label are ambiguous.
    valid_outcome_case_ids = outcome_label_count.index[
        outcome_label_count.eq(1)
    ]
    valid_outcomes = (
        outcome_events.loc[
            outcome_events[CASE_COL].isin(valid_outcome_case_ids)
        ]
        .groupby(CASE_COL, sort=False)
        .agg(
            outcome_activity=(ACTIVITY_COL, "first"),
            outcome_event_nr=("event_nr", "min"),
        )
    )

    case_info = (
        df_work.groupby(CASE_COL, sort=False)
        .size()
        .rename("trace_length")
        .to_frame()
        .join(outcome_event_count.rename("outcome_event_count"))
        .join(outcome_label_count.rename("outcome_label_count"))
        .join(valid_outcomes[["outcome_activity", "outcome_event_nr"]])
    )
    case_info["outcome_event_nr"] = case_info["outcome_event_nr"].astype(
        "Int64"
    )
    case_info["has_k_events"] = case_info["trace_length"].ge(k)
    case_info["outcome_after_k"] = (
        case_info["outcome_event_nr"].gt(k).fillna(False).astype(bool)
    )
    case_info["eligible_at_k"] = (
        case_info["outcome_label_count"].eq(1)
        & case_info["has_k_events"]
        & case_info["outcome_after_k"]
    )

    eligible_cases = case_info.index[case_info["eligible_at_k"]].copy()
    eligible_event_mask = df_work[CASE_COL].isin(eligible_cases)
    prefix_k10 = df_work.loc[
        eligible_event_mask & df_work["event_nr"].le(k)
    ].copy()
    future_k10 = df_work.loc[
        eligible_event_mask & df_work["event_nr"].gt(k)
    ].copy()

    case_target = case_info.loc[
        eligible_cases,
        ["outcome_activity", "outcome_event_nr"],
    ].reset_index()
    case_target["target"] = (
        case_target["outcome_activity"].map(TARGET_MAPPING).astype("int8")
    )

    prefix_sizes = prefix_k10.groupby(CASE_COL).size()
    outcome_inside_prefix = int(
        prefix_k10[ACTIVITY_COL].isin(OUTCOME_ACTIVITIES).sum()
    )

    assert len(prefix_k10) == len(eligible_cases) * k
    assert prefix_k10["event_nr"].max() == k
    assert prefix_sizes.index.equals(eligible_cases)
    assert prefix_sizes.eq(k).all(), "An eligible case does not have k events."
    assert outcome_inside_prefix == 0, "An outcome occurs inside the prefix."
    assert case_info.loc[eligible_cases, "outcome_label_count"].eq(1).all()
    assert len(case_target) == len(eligible_cases)
    assert case_target[CASE_COL].is_unique
    assert case_target[CASE_COL].isin(eligible_cases).all()
    assert set(case_target["target"].unique()).issubset({0, 1})

    return (
        df_work,
        case_info,
        eligible_cases,
        prefix_k10,
        future_k10,
        case_target,
    )


def summarize_prediction_prefix(
    case_info,
    eligible_cases,
    prefix_k10,
    future_k10,
    k=K,
):
    """Calculate the validation summary without changing the prefix data."""
    outcome_event_count = case_info["outcome_event_count"]
    outcome_label_count = case_info["outcome_label_count"]
    outcome_inside_prefix = int(
        prefix_k10[ACTIVITY_COL].isin(OUTCOME_ACTIVITIES).sum()
    )
    return {
        "prediction_point": k,
        "total_cases": len(case_info),
        "cases_with_outcome": int(outcome_event_count.gt(0).sum()),
        "cases_with_one_distinct_outcome": int(
            outcome_label_count.eq(1).sum()
        ),
        "cases_with_ambiguous_outcome": int(outcome_label_count.gt(1).sum()),
        "cases_with_repeated_outcome_events": int(
            outcome_event_count.gt(outcome_label_count).sum()
        ),
        "cases_without_outcome": int(outcome_event_count.eq(0).sum()),
        "cases_with_outcome_by_k": int(
            case_info["outcome_event_nr"].le(k).fillna(False).sum()
        ),
        "eligible_cases": len(eligible_cases),
        "prefix_rows": len(prefix_k10),
        "future_rows": len(future_k10),
        "outcome_inside_prefix": outcome_inside_prefix,
    }


def audit_case_level_feature_availability(
    df_work,
    eligible_cases,
    prefix_k10,
    future_k10,
    features=CASE_LEVEL_FEATURES,
    k=K,
):
    """Section 3.1 — audit case-level feature availability."""
    total_cases = len(eligible_cases)
    eligible_events = df_work.loc[
        df_work[CASE_COL].isin(eligible_cases)
    ]

    assert eligible_cases.is_unique
    assert eligible_events[CASE_COL].nunique() == total_cases
    assert len(prefix_k10) + len(future_k10) == len(eligible_events)
    assert prefix_k10["event_nr"].le(k).all()
    assert future_k10["event_nr"].gt(k).all()

    audit_rows = []
    first_seen_distributions = {}
    event_1_missing_counts = {}
    requested_amount_zero_cases_at_event_1 = 0

    event_1 = prefix_k10.loc[prefix_k10["event_nr"].eq(1)]
    assert event_1[CASE_COL].is_unique
    assert event_1[CASE_COL].nunique() == total_cases

    for feature in features:
        observed_events = eligible_events.loc[
            eligible_events[feature].notna(),
            [CASE_COL, "event_nr"],
        ]
        first_seen_event = observed_events.groupby(
            CASE_COL,
            sort=False,
        )["event_nr"].min()
        first_seen_distribution = (
            first_seen_event.value_counts()
            .sort_index()
            .rename_axis("first_seen_event")
            .rename("case_count")
        )

        ever_available = len(first_seen_event)
        available_by_k = int(first_seen_event.le(k).sum())
        prefix_available = prefix_k10.loc[
            prefix_k10[feature].notna(),
            CASE_COL,
        ].nunique()
        assert available_by_k == prefix_available

        distinct_non_null_values = eligible_events.groupby(
            CASE_COL,
            sort=False,
        )[feature].nunique(dropna=True)

        event_1_values = (
            event_1[[CASE_COL, feature]]
            .set_index(CASE_COL)[feature]
            .reindex(eligible_cases)
        )
        event_1_missing = int(event_1_values.isna().sum())

        if feature == "case:RequestedAmount":
            requested_amount_zero_cases_at_event_1 = int(
                event_1_values.eq(0).sum()
            )
            assert requested_amount_zero_cases_at_event_1 <= (
                total_cases - event_1_missing
            )

        first_seen_distributions[feature] = first_seen_distribution
        event_1_missing_counts[feature] = event_1_missing
        audit_rows.append(
            {
                "feature": feature,
                "total_eligible_cases": total_cases,
                "ever_available": ever_available,
                "available_by_k10": available_by_k,
                "availability_percentage": (
                    available_by_k / total_cases * 100
                    if total_cases
                    else float("nan")
                ),
                "first_seen_event_distribution": (
                    first_seen_distribution.to_dict()
                ),
                "median_first_seen_event": first_seen_event.median(),
                "maximum_first_seen_event": first_seen_event.max(),
                "first_observed_after_k10": int(first_seen_event.gt(k).sum()),
                "never_observed": total_cases - ever_available,
                "cases_with_multiple_distinct_non_null_values": int(
                    distinct_non_null_values.gt(1).sum()
                ),
                "missing_at_event_1": event_1_missing,
            }
        )

    case_level_feature_availability = pd.DataFrame(audit_rows)
    return (
        case_level_feature_availability,
        first_seen_distributions,
        event_1_missing_counts,
        requested_amount_zero_cases_at_event_1,
    )


def print_results(
    summary,
    prefix_sizes,
    case_level_feature_availability,
    first_seen_distributions,
    requested_amount_zero_cases_at_event_1,
):
    """Print the boundary validation and Section 3.1 evidence."""
    print("\n=== K=10 PREDICTION-PREFIX DATASET ===")
    for name, value in summary.items():
        print(f"{name}: {value:,}")

    print("\n=== PREFIX EVENTS PER ELIGIBLE CASE ===")
    print(prefix_sizes.value_counts().sort_index().to_string())
    print("\nAll k=10 prediction-prefix assertions passed.")

    scalar_columns = [
        column
        for column in case_level_feature_availability.columns
        if column != "first_seen_event_distribution"
    ]
    print("\n=== SECTION 3.1 — CASE-LEVEL ATTRIBUTE AVAILABILITY ===")
    print(
        case_level_feature_availability[scalar_columns].to_string(
            index=False,
            formatters={"availability_percentage": "{:.2f}%".format},
        )
    )

    print("\n=== FIRST-SEEN EVENT DISTRIBUTIONS ===")
    for feature, distribution in first_seen_distributions.items():
        print(f"\n--- {feature} ---")
        if distribution.empty:
            print("No eligible case ever contains a non-null value.")
        else:
            print(distribution.to_string())

    print("\n=== EVENT-1 MISSING COUNTS ===")
    event_1_missing = case_level_feature_availability.set_index("feature")[
        "missing_at_event_1"
    ]
    print(event_1_missing.to_string())
    print(
        "\nRequestedAmount == 0 cases at event 1: "
        f"{requested_amount_zero_cases_at_event_1:,}"
    )


def main():
    """Run the validated k=10 boundary and Section 3.1 audit."""
    global df_work
    global case_info
    global eligible_cases
    global prefix_k10
    global future_k10
    global case_target
    global k10_validation_summary
    global case_level_feature_availability
    global case_level_first_seen_distributions
    global case_level_event_1_missing_counts
    global requested_amount_zero_cases_at_event_1

    print("Reading dataset...")
    dataframe = load_event_log()
    (
        df_work,
        case_info,
        eligible_cases,
        prefix_k10,
        future_k10,
        case_target,
    ) = build_prediction_prefix(dataframe)

    prefix_sizes = prefix_k10.groupby(CASE_COL).size()
    k10_validation_summary = summarize_prediction_prefix(
        case_info=case_info,
        eligible_cases=eligible_cases,
        prefix_k10=prefix_k10,
        future_k10=future_k10,
    )

    assert k10_validation_summary == EXPECTED_K10_SUMMARY, (
        "The k=10 validation summary differs from the validated baseline."
    )

    (
        case_level_feature_availability,
        case_level_first_seen_distributions,
        case_level_event_1_missing_counts,
        requested_amount_zero_cases_at_event_1,
    ) = audit_case_level_feature_availability(
        df_work=df_work,
        eligible_cases=eligible_cases,
        prefix_k10=prefix_k10,
        future_k10=future_k10,
    )
    print_results(
        summary=k10_validation_summary,
        prefix_sizes=prefix_sizes,
        case_level_feature_availability=case_level_feature_availability,
        first_seen_distributions=case_level_first_seen_distributions,
        requested_amount_zero_cases_at_event_1=(
            requested_amount_zero_cases_at_event_1
        ),
    )


if __name__ == "__main__":
    main()
