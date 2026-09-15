"""Feature Availability analysis for the BPIC 2017 event log."""

from pathlib import Path

import pandas as pd
import pm4py


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "BPI Challenge 2017.xes"
FEATURE_AVAILABILITY_MATRIX_PATH = (
    PROJECT_ROOT / "results" / "feature_availability_matrix.csv"
)

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

EVENT_PROCESS_FEATURES = [
    "concept:name",
    "Action",
    "EventOrigin",
    "org:resource",
    "lifecycle:transition",
    "time:timestamp",
]

SPARSE_OFFER_FEATURES = [
    "FirstWithdrawalAmount",
    "NumberOfTerms",
    "Accepted",
    "MonthlyCost",
    "Selected",
    "CreditScore",
    "OfferedAmount",
    "OfferID",
]

SEMANTIC_AUDIT_FEATURES = ["Accepted", "Selected"]

IDENTIFIER_ATTRIBUTES = ["EventID", CASE_COL]

CONTEXT_ONLY_OFFER_FEATURES = [
    "FirstWithdrawalAmount",
    "NumberOfTerms",
    "MonthlyCost",
    "CreditScore",
    "OfferedAmount",
]

OFFER_LIFECYCLE_STATES = [
    "O_Accepted",
    "O_Refused",
    "O_Cancelled",
    "O_Returned",
    "O_Sent",
]

ORIGINAL_RAW_FEATURES = [
    "case:LoanGoal",
    "case:ApplicationType",
    "case:RequestedAmount",
    "concept:name",
    "Action",
    "EventOrigin",
    "org:resource",
    "lifecycle:transition",
    "time:timestamp",
    "FirstWithdrawalAmount",
    "NumberOfTerms",
    "Accepted",
    "MonthlyCost",
    "Selected",
    "CreditScore",
    "OfferedAmount",
    "OfferID",
    "EventID",
    "case:concept:name",
]

FEATURE_AVAILABILITY_DECISIONS = {
    "case:LoanGoal": {
        "practically_available": "Yes",
        "future_dependency": "No",
        "availability_decision": "Available",
        "model_handling": "Candidate model feature",
        "reason": "Observed as a case-level attribute by the prediction point.",
        "notes": "Use the value observed in the prefix.",
    },
    "case:ApplicationType": {
        "practically_available": "Yes",
        "future_dependency": "No",
        "availability_decision": "Available",
        "model_handling": "Candidate model feature",
        "reason": "Observed as a case-level attribute by the prediction point.",
        "notes": "Use the value observed in the prefix.",
    },
    "case:RequestedAmount": {
        "practically_available": "Yes",
        "future_dependency": "No",
        "availability_decision": "Available",
        "model_handling": "Candidate model feature",
        "reason": "Observed as a case-level attribute by the prediction point.",
        "notes": "A value of zero is observed, not missing.",
    },
    "concept:name": {
        "practically_available": "Yes",
        "future_dependency": "No",
        "availability_decision": "Available",
        "model_handling": "Prefix-safe encoding or aggregation",
        "reason": "Observed event-process context is available in the prefix.",
        "notes": "Do not encode events after the prediction point.",
    },
    "Action": {
        "practically_available": "Yes",
        "future_dependency": "No",
        "availability_decision": "Available",
        "model_handling": "Prefix-safe encoding or aggregation",
        "reason": "Observed event-process context is available in the prefix.",
        "notes": "Do not encode events after the prediction point.",
    },
    "EventOrigin": {
        "practically_available": "Yes",
        "future_dependency": "No",
        "availability_decision": "Available",
        "model_handling": "Prefix-safe encoding or aggregation",
        "reason": "Observed event-process context is available in the prefix.",
        "notes": "Do not encode events after the prediction point.",
    },
    "org:resource": {
        "practically_available": "Yes",
        "future_dependency": "No",
        "availability_decision": "Available",
        "model_handling": "Evaluate raw-resource generalization separately",
        "reason": "The resource is observed in the prefix.",
        "notes": "Availability does not establish out-of-sample generalization.",
    },
    "lifecycle:transition": {
        "practically_available": "Yes",
        "future_dependency": "No",
        "availability_decision": "Available",
        "model_handling": "Prefix-safe encoding or aggregation",
        "reason": "Observed event-process context is available in the prefix.",
        "notes": "Do not encode events after the prediction point.",
    },
    "time:timestamp": {
        "practically_available": "Yes",
        "future_dependency": "No",
        "availability_decision": "Available",
        "model_handling": "Use prefix-derived temporal features only",
        "reason": "Timestamps for observed prefix events are available.",
        "notes": "Exclude final-event timestamp and total case duration.",
    },
    "FirstWithdrawalAmount": {
        "practically_available": "Conditional",
        "future_dependency": "No, if observed by k",
        "availability_decision": "Conditionally Available",
        "model_handling": "Use only when observed within events 1-10",
        "reason": "Offer creation may occur before or after the prediction point.",
        "notes": "Never backfill from the suffix; multiple offers are valid.",
    },
    "NumberOfTerms": {
        "practically_available": "Conditional",
        "future_dependency": "No, if observed by k",
        "availability_decision": "Conditionally Available",
        "model_handling": "Use only when observed within events 1-10",
        "reason": "Offer creation may occur before or after the prediction point.",
        "notes": "Never backfill from the suffix; multiple offers are valid.",
    },
    "Accepted": {
        "practically_available": "Conditional",
        "future_dependency": "No, if observed by k",
        "availability_decision": "Conditionally Available",
        "model_handling": "Use only when observed within events 1-10",
        "reason": "It does not deterministically encode later O_Accepted state.",
        "notes": "Never backfill from the suffix; multiple offers are valid.",
    },
    "MonthlyCost": {
        "practically_available": "Conditional",
        "future_dependency": "No, if observed by k",
        "availability_decision": "Conditionally Available",
        "model_handling": "Use only when observed within events 1-10",
        "reason": "Offer creation may occur before or after the prediction point.",
        "notes": "Never backfill from the suffix; multiple offers are valid.",
    },
    "Selected": {
        "practically_available": "No",
        "future_dependency": "Yes, semantic",
        "availability_decision": "Leakage Risk",
        "model_handling": "Exclude",
        "reason": "It reflects later offer lifecycle or customer selection.",
        "notes": "Recorded on O_Create Offer but treated as future-derived.",
    },
    "CreditScore": {
        "practically_available": "Conditional",
        "future_dependency": "No, if observed by k",
        "availability_decision": "Conditionally Available",
        "model_handling": "Use only when observed within events 1-10",
        "reason": "Offer creation may occur before or after the prediction point.",
        "notes": "Never backfill from the suffix; multiple offers are valid.",
    },
    "OfferedAmount": {
        "practically_available": "Conditional",
        "future_dependency": "No, if observed by k",
        "availability_decision": "Conditionally Available",
        "model_handling": "Use only when observed within events 1-10",
        "reason": "Offer creation may occur before or after the prediction point.",
        "notes": "Never backfill from the suffix; multiple offers are valid.",
    },
    "OfferID": {
        "practically_available": "Grouping only",
        "future_dependency": "No, if observed by k",
        "availability_decision": "Conditionally Available",
        "model_handling": "Exclude raw ID; retain for offer grouping",
        "reason": "Prefix availability is conditional and the value is an ID.",
        "notes": "Multiple offer IDs within a case are valid.",
    },
    "EventID": {
        "practically_available": "Grouping only",
        "future_dependency": "No",
        "availability_decision": "Available",
        "model_handling": "Exclude as a raw identifier",
        "reason": "Observed in the prefix but unsuitable as a raw predictor.",
        "notes": "Retain only where event identity is operationally required.",
    },
    "case:concept:name": {
        "practically_available": "Grouping only",
        "future_dependency": "No",
        "availability_decision": "Available",
        "model_handling": "Exclude as a raw identifier",
        "reason": "Observed in the prefix but unsuitable as a raw predictor.",
        "notes": "Retain for case grouping and joins.",
    },
}

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


def audit_event_process_completeness(
    prefix_k10,
    eligible_cases,
    features=EVENT_PROCESS_FEATURES,
    k=K,
):
    """Section 3.2.1: measure event/process completeness in the prefix."""
    total_prefix_rows = len(prefix_k10)
    total_eligible_cases = len(eligible_cases)

    assert prefix_k10[CASE_COL].nunique() == total_eligible_cases
    assert len(prefix_k10) == total_eligible_cases * k
    assert prefix_k10["event_nr"].between(1, k).all()
    assert set(prefix_k10["event_nr"].unique()) == set(range(1, k + 1))

    completeness_rows = []
    missingness_by_event_position = {}

    for feature in features:
        assert feature in prefix_k10.columns, f"Missing feature: {feature}"

        non_null_mask = prefix_k10[feature].notna()
        non_null_prefix_rows = int(non_null_mask.sum())
        missing_prefix_rows = total_prefix_rows - non_null_prefix_rows

        cases_with_value = int(
            prefix_k10.loc[non_null_mask, CASE_COL].nunique()
        )
        cases_with_no_value = total_eligible_cases - cases_with_value

        completeness_rows.append(
            {
                "feature": feature,
                "total_prefix_rows": total_prefix_rows,
                "non_null_prefix_rows": non_null_prefix_rows,
                "missing_prefix_rows": missing_prefix_rows,
                "row_completeness_percentage": (
                    non_null_prefix_rows / total_prefix_rows * 100
                ),
                "missing_rate": missing_prefix_rows / total_prefix_rows * 100,
                "cases_with_at_least_one_non_null_value": cases_with_value,
                "cases_with_no_value": cases_with_no_value,
                "case_coverage_percentage": (
                    cases_with_value / total_eligible_cases * 100
                ),
            }
        )

        by_position = (
            prefix_k10.assign(_non_missing=non_null_mask)
            .groupby("event_nr", sort=True)
            .agg(
                total=(CASE_COL, "size"),
                non_missing=("_non_missing", "sum"),
            )
            .reindex(range(1, k + 1))
        )
        by_position["non_missing"] = by_position["non_missing"].astype(int)
        by_position["missing"] = (
            by_position["total"] - by_position["non_missing"]
        )
        by_position["missing_pct"] = (
            by_position["missing"] / by_position["total"] * 100
        )
        by_position.index.name = "event_nr"

        assert by_position["total"].eq(total_eligible_cases).all()
        assert int(by_position["total"].sum()) == total_prefix_rows
        assert int(by_position["non_missing"].sum()) == non_null_prefix_rows
        assert int(by_position["missing"].sum()) == missing_prefix_rows
        assert cases_with_value + cases_with_no_value == total_eligible_cases

        missingness_by_event_position[feature] = by_position

    event_process_completeness = pd.DataFrame(completeness_rows)
    return event_process_completeness, missingness_by_event_position


def audit_sparse_feature_availability(
    df_work,
    eligible_cases,
    eligible_log,
    prefix_k10,
    features=SPARSE_OFFER_FEATURES,
    k=K,
):
    """Section 3.3.1: empirically audit sparse/offer availability."""
    total_cases = len(eligible_cases)

    assert eligible_cases.is_unique
    assert eligible_log[CASE_COL].nunique() == total_cases
    assert eligible_log[CASE_COL].isin(eligible_cases).all()
    assert len(eligible_log) == int(df_work[CASE_COL].isin(eligible_cases).sum())
    assert prefix_k10[CASE_COL].isin(eligible_cases).all()
    assert prefix_k10["event_nr"].between(1, k).all()

    audit_rows = []
    first_seen_distributions = {}
    activity_context = {}
    event_origin_context = {}

    for feature in features:
        assert feature in eligible_log.columns, f"Missing feature: {feature}"

        non_null_mask = eligible_log[feature].notna()
        observed_rows = eligible_log.loc[
            non_null_mask,
            [CASE_COL, ACTIVITY_COL, "EventOrigin", "event_nr"],
        ]
        first_seen_event = observed_rows.groupby(
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
        future_only = int(first_seen_event.gt(k).sum())
        never_observed = total_cases - ever_available

        # This independent prefix calculation must agree with first-seen timing.
        prefix_available_by_k = int(
            prefix_k10.loc[prefix_k10[feature].notna(), CASE_COL].nunique()
        )
        assert prefix_available_by_k == available_by_k, (
            f"{feature}: prefix availability differs from first_seen_event <= {k}."
        )
        assert available_by_k + future_only + never_observed == total_cases, (
            f"{feature}: availability groups do not reconcile."
        )
        assert ever_available == available_by_k + future_only
        assert int(first_seen_distribution.sum()) == ever_available

        distinct_non_null_values = eligible_log.groupby(
            CASE_COL,
            sort=False,
        )[feature].nunique(dropna=True)
        multi_value_cases = int(distinct_non_null_values.gt(1).sum())

        first_seen_distributions[feature] = first_seen_distribution
        activity_context[feature] = (
            observed_rows[ACTIVITY_COL]
            .value_counts(dropna=False)
            .rename_axis(ACTIVITY_COL)
            .rename("row_count")
        )
        event_origin_context[feature] = (
            observed_rows["EventOrigin"]
            .value_counts(dropna=False)
            .rename_axis("EventOrigin")
            .rename("row_count")
        )

        audit_rows.append(
            {
                "feature": feature,
                "total_cases": total_cases,
                "ever_available": ever_available,
                "available_by_k": available_by_k,
                "available_by_k_pct": available_by_k / total_cases * 100,
                "future_only": future_only,
                "future_only_pct": future_only / total_cases * 100,
                "never_observed": never_observed,
                "never_observed_pct": never_observed / total_cases * 100,
                "first_seen_min": first_seen_event.min(),
                "first_seen_median": first_seen_event.median(),
                "first_seen_max": first_seen_event.max(),
                "multi_value_cases": multi_value_cases,
            }
        )

    sparse_feature_availability = pd.DataFrame(audit_rows)
    return (
        sparse_feature_availability,
        first_seen_distributions,
        activity_context,
        event_origin_context,
    )


def _normalize_offer_lifecycle_activity(activity):
    """Combine the two recorded O_Sent variants for semantic auditing."""
    if isinstance(activity, str) and activity.startswith("O_Sent"):
        return "O_Sent"
    if activity in OFFER_LIFECYCLE_STATES:
        return activity
    return None


def audit_sparse_feature_semantics(
    eligible_log,
    prefix_k10,
    case_target,
    sparse_feature_availability,
    k=K,
):
    """Section 3.3.2: audit semantics and possible lifecycle leakage."""
    availability_snapshot = sparse_feature_availability.copy(deep=True)
    create_mask = eligible_log[ACTIVITY_COL].eq("O_Create Offer")
    create_rows = eligible_log.loc[create_mask].copy()
    target_by_case = case_target.set_index(CASE_COL)["outcome_activity"]

    assert target_by_case.index.is_unique
    assert create_rows[SEMANTIC_AUDIT_FEATURES].notna().all().all()

    value_distributions = {}
    outcome_value_presence_crosstabs = {}
    outcome_value_pattern_crosstabs = {}
    multiplicity_summary_rows = []

    for feature in SEMANTIC_AUDIT_FEATURES:
        counts = create_rows[feature].value_counts(dropna=False)
        value_distributions[feature] = pd.DataFrame(
            {
                "count": counts,
                "percentage": counts / len(create_rows) * 100,
            }
        ).rename_axis(feature)

        case_value_presence = create_rows[
            [CASE_COL, feature]
        ].drop_duplicates()
        case_value_presence["outcome_activity"] = case_value_presence[
            CASE_COL
        ].map(target_by_case)
        presence_crosstab = pd.crosstab(
            case_value_presence[feature],
            case_value_presence["outcome_activity"],
            dropna=False,
        ).reindex(columns=sorted(OUTCOME_ACTIVITIES), fill_value=0)
        presence_crosstab["case_value_memberships"] = presence_crosstab.sum(
            axis=1
        )
        outcome_value_presence_crosstabs[feature] = presence_crosstab

        case_patterns = (
            create_rows.groupby(CASE_COL, sort=False)[feature]
            .agg(lambda values: " | ".join(sorted({str(v) for v in values})))
            .rename("observed_value_pattern")
            .to_frame()
        )
        case_patterns["outcome_activity"] = case_patterns.index.map(
            target_by_case
        )
        pattern_crosstab = pd.crosstab(
            case_patterns["observed_value_pattern"],
            case_patterns["outcome_activity"],
        ).reindex(columns=sorted(OUTCOME_ACTIVITIES), fill_value=0)
        pattern_crosstab["unique_cases"] = pattern_crosstab.sum(axis=1)
        outcome_value_pattern_crosstabs[feature] = pattern_crosstab

        offers_per_case = create_rows.groupby(CASE_COL).size()
        values_per_case = create_rows.groupby(CASE_COL)[feature].nunique()
        multiplicity_summary_rows.append(
            {
                "feature": feature,
                "cases_with_create_offer": int(offers_per_case.size),
                "cases_with_multiple_create_offers": int(
                    offers_per_case.gt(1).sum()
                ),
                "cases_with_multiple_observed_values": int(
                    values_per_case.gt(1).sum()
                ),
                "case_value_memberships": len(case_value_presence),
            }
        )

    multiplicity_summary = pd.DataFrame(multiplicity_summary_rows)

    # O_Create Offer has no OfferID. Link it only to an immediately following
    # O_Created row in the same case, then audit later states of that OfferID.
    ordered = eligible_log[[CASE_COL, ACTIVITY_COL, "OfferID", "event_nr"]]
    create_rows["_next_activity"] = ordered.groupby(
        CASE_COL, sort=False
    )[ACTIVITY_COL].shift(-1).loc[create_rows.index]
    create_rows["_associated_offer_id"] = ordered.groupby(
        CASE_COL, sort=False
    )["OfferID"].shift(-1).loc[create_rows.index]
    create_rows["_offer_linked"] = (
        create_rows["_next_activity"].eq("O_Created")
        & create_rows["_associated_offer_id"].notna()
    )

    lifecycle_rows = eligible_log.loc[
        eligible_log[ACTIVITY_COL].map(
            _normalize_offer_lifecycle_activity
        ).notna()
        & eligible_log["OfferID"].notna(),
        [CASE_COL, "OfferID", ACTIVITY_COL, "event_nr"],
    ].copy()
    lifecycle_rows["_lifecycle_state"] = lifecycle_rows[ACTIVITY_COL].map(
        _normalize_offer_lifecycle_activity
    )
    first_state_events = lifecycle_rows.pivot_table(
        index=[CASE_COL, "OfferID"],
        columns="_lifecycle_state",
        values="event_nr",
        aggfunc="min",
    ).reset_index()

    create_rows = create_rows.merge(
        first_state_events,
        how="left",
        left_on=[CASE_COL, "_associated_offer_id"],
        right_on=[CASE_COL, "OfferID"],
        suffixes=("", "_lifecycle"),
        validate="many_to_one",
    )
    for state in OFFER_LIFECYCLE_STATES:
        if state not in create_rows:
            create_rows[state] = pd.NA
        create_rows[f"later_{state}"] = (
            create_rows["_offer_linked"]
            & create_rows[state].gt(create_rows["event_nr"]).fillna(False)
        )

    offer_linkage_summary = pd.Series(
        {
            "create_offer_rows": len(create_rows),
            "linked_to_immediate_o_created": int(
                create_rows["_offer_linked"].sum()
            ),
            "unlinked_create_offer_rows": int(
                (~create_rows["_offer_linked"]).sum()
            ),
        },
        name="row_count",
    )

    lifecycle_relationships = {}
    for feature in SEMANTIC_AUDIT_FEATURES:
        rows = []
        for value, value_rows in create_rows.groupby(feature, dropna=False):
            row = {
                "value": value,
                "create_offer_rows": len(value_rows),
                "linked_offer_rows": int(value_rows["_offer_linked"].sum()),
            }
            for state in OFFER_LIFECYCLE_STATES:
                state_count = int(value_rows[f"later_{state}"].sum())
                row[f"later_{state}"] = state_count
                row[f"later_{state}_pct"] = state_count / len(value_rows) * 100
            rows.append(row)
        lifecycle_relationships[feature] = pd.DataFrame(rows)

    offer_activity_mask = eligible_log[ACTIVITY_COL].str.startswith(
        "O_", na=False
    )
    create_counts = create_rows.groupby(CASE_COL).size()
    selected_nunique = create_rows.groupby(CASE_COL)["Selected"].nunique()
    accepted_nunique = create_rows.groupby(CASE_COL)["Accepted"].nunique()
    offer_trace_sizes = eligible_log.loc[offer_activity_mask].groupby(
        CASE_COL
    ).size()

    candidate_groups = [
        create_counts.index[create_counts.gt(1)],
        selected_nunique.index[selected_nunique.gt(1)],
        accepted_nunique.index[accepted_nunique.gt(1)],
    ]
    representative_case_ids = []
    for candidates in candidate_groups:
        remaining = candidates.difference(representative_case_ids)
        if len(remaining):
            selected_case = (
                offer_trace_sizes.reindex(remaining)
                .sort_values(kind="mergesort")
                .index[0]
            )
            representative_case_ids.append(selected_case)

    trace_columns = [
        "event_nr",
        ACTIVITY_COL,
        "OfferID",
        "Accepted",
        "Selected",
        "OfferedAmount",
        TIME_COL,
    ]
    representative_offer_traces = {
        case_id: eligible_log.loc[
            eligible_log[CASE_COL].eq(case_id) & offer_activity_mask,
            trace_columns,
        ].sort_values("event_nr")
        for case_id in representative_case_ids
    }

    context_only_confirmation = []
    prefix_source = eligible_log.loc[eligible_log["event_nr"].le(k)]
    for feature in CONTEXT_ONLY_OFFER_FEATURES:
        non_null_rows = eligible_log.loc[eligible_log[feature].notna()]
        assert non_null_rows[ACTIVITY_COL].eq("O_Create Offer").all()
        assert non_null_rows[TIME_COL].notna().all()
        pd.testing.assert_series_equal(
            prefix_k10[feature],
            prefix_source[feature],
            check_names=True,
        )
        context_only_confirmation.append(
            {
                "feature": feature,
                "non_null_rows": len(non_null_rows),
                "activity": "O_Create Offer",
                "timestamps_present": True,
                "prefix_matches_source_events_1_to_k": True,
                "suffix_values_propagated": False,
            }
        )

    # Section 3.3.2 must not alter Section 3.3.1 results.
    pd.testing.assert_frame_equal(
        sparse_feature_availability,
        availability_snapshot,
    )

    return (
        value_distributions,
        outcome_value_presence_crosstabs,
        outcome_value_pattern_crosstabs,
        multiplicity_summary,
        offer_linkage_summary,
        lifecycle_relationships,
        representative_offer_traces,
        pd.DataFrame(context_only_confirmation),
    )


def audit_identifier_attributes(
    prefix_k10,
    eligible_cases,
    identifiers=IDENTIFIER_ATTRIBUTES,
    k=K,
):
    """Section 3.4: audit identifier availability in the prefix only."""
    total_prefix_rows = len(prefix_k10)
    total_eligible_cases = len(eligible_cases)

    assert identifiers == ["EventID", CASE_COL]
    assert eligible_cases.is_unique
    assert len(prefix_k10) == total_eligible_cases * k
    assert prefix_k10[CASE_COL].isin(eligible_cases).all()
    assert prefix_k10["event_nr"].between(1, k).all()

    availability_rows = []
    for identifier in identifiers:
        assert identifier in prefix_k10.columns, (
            f"Missing identifier: {identifier}"
        )

        non_null_mask = prefix_k10[identifier].notna()
        non_null_rows = int(non_null_mask.sum())
        missing_rows = total_prefix_rows - non_null_rows
        covered_cases = int(
            prefix_k10.loc[non_null_mask, CASE_COL].nunique()
        )

        availability_rows.append(
            {
                "identifier": identifier,
                "total_prefix_rows": total_prefix_rows,
                "non_null_rows": non_null_rows,
                "missing_rows": missing_rows,
                "row_completeness_percentage": (
                    non_null_rows / total_prefix_rows * 100
                ),
                "eligible_cases_with_at_least_one_observed_value": (
                    covered_cases
                ),
                "case_coverage_percentage": (
                    covered_cases / total_eligible_cases * 100
                ),
            }
        )

    identifier_availability = pd.DataFrame(availability_rows)

    non_null_event_ids = prefix_k10.loc[
        prefix_k10["EventID"].notna(), "EventID"
    ]
    event_id_uniqueness_results = pd.Series(
        {
            "non_null_values": len(non_null_event_ids),
            "distinct_values": int(non_null_event_ids.nunique()),
            "duplicated_event_id_rows": int(
                non_null_event_ids.duplicated().sum()
            ),
        },
        name="EventID",
    )
    assert (
        event_id_uniqueness_results["distinct_values"]
        + event_id_uniqueness_results["duplicated_event_id_rows"]
        == event_id_uniqueness_results["non_null_values"]
    )

    case_id_frequencies = prefix_k10[CASE_COL].value_counts(sort=False)
    distinct_case_ids = int(prefix_k10[CASE_COL].nunique())
    distinct_matches_eligible_cases = distinct_case_ids == total_eligible_cases
    every_eligible_case_occurs_k_times = (
        case_id_frequencies.index.isin(eligible_cases).all()
        and len(case_id_frequencies) == total_eligible_cases
        and case_id_frequencies.eq(k).all()
    )
    case_id_frequency_validation = pd.Series(
        {
            "distinct_case_ids": distinct_case_ids,
            "eligible_cases": total_eligible_cases,
            "distinct_ids_equal_eligible_cases": (
                distinct_matches_eligible_cases
            ),
            "expected_rows_per_eligible_case": k,
            "minimum_rows_per_case": int(case_id_frequencies.min()),
            "maximum_rows_per_case": int(case_id_frequencies.max()),
            "every_eligible_case_occurs_exactly_k_times": (
                every_eligible_case_occurs_k_times
            ),
        },
        name=CASE_COL,
    )

    assert distinct_matches_eligible_cases
    assert every_eligible_case_occurs_k_times

    return (
        identifier_availability,
        event_id_uniqueness_results,
        case_id_frequency_validation,
    )


def build_final_feature_availability_matrix(
    case_level_feature_availability,
    event_process_completeness,
    sparse_feature_availability,
    identifier_availability,
    eligible_cases,
):
    """Section 3.5.1: consolidate empirical and semantic availability."""
    case_level = case_level_feature_availability[
        ["feature", "available_by_k10", "availability_percentage"]
    ].rename(
        columns={
            "available_by_k10": "availability_at_k",
            "availability_percentage": "availability_pct",
        }
    )
    case_level["feature_group"] = "case-level"

    event_process = event_process_completeness[
        [
            "feature",
            "cases_with_at_least_one_non_null_value",
            "case_coverage_percentage",
        ]
    ].rename(
        columns={
            "cases_with_at_least_one_non_null_value": "availability_at_k",
            "case_coverage_percentage": "availability_pct",
        }
    )
    event_process["feature_group"] = "event-process"

    offer_sparse = sparse_feature_availability[
        ["feature", "available_by_k", "available_by_k_pct"]
    ].rename(
        columns={
            "available_by_k": "availability_at_k",
            "available_by_k_pct": "availability_pct",
        }
    )
    offer_sparse["feature_group"] = "offer-sparse"

    identifiers = identifier_availability[
        [
            "identifier",
            "eligible_cases_with_at_least_one_observed_value",
            "case_coverage_percentage",
        ]
    ].rename(
        columns={
            "identifier": "feature",
            "eligible_cases_with_at_least_one_observed_value": (
                "availability_at_k"
            ),
            "case_coverage_percentage": "availability_pct",
        }
    )
    identifiers["feature_group"] = "identifier"

    empirical_results = pd.concat(
        [case_level, event_process, offer_sparse, identifiers],
        ignore_index=True,
    )
    empirical_results["availability_pct"] /= 100

    semantic_decisions = (
        pd.DataFrame.from_dict(
            FEATURE_AVAILABILITY_DECISIONS,
            orient="index",
        )
        .rename_axis("feature")
        .reset_index()
    )

    assert len(ORIGINAL_RAW_FEATURES) == 19
    assert len(set(ORIGINAL_RAW_FEATURES)) == 19
    assert empirical_results["feature"].is_unique
    assert semantic_decisions["feature"].is_unique
    assert set(empirical_results["feature"]) == set(ORIGINAL_RAW_FEATURES)
    assert set(semantic_decisions["feature"]) == set(ORIGINAL_RAW_FEATURES), (
        "Every empirical result must have exactly one semantic decision."
    )

    final_matrix = empirical_results.merge(
        semantic_decisions,
        on="feature",
        how="inner",
        validate="one_to_one",
    ).set_index("feature").loc[ORIGINAL_RAW_FEATURES].reset_index()
    final_matrix = final_matrix[
        [
            "feature",
            "feature_group",
            "availability_at_k",
            "availability_pct",
            "practically_available",
            "future_dependency",
            "availability_decision",
            "model_handling",
            "reason",
            "notes",
        ]
    ]

    assert len(final_matrix) == 19
    assert final_matrix["feature"].nunique() == 19
    assert final_matrix["feature"].tolist() == ORIGINAL_RAW_FEATURES
    assert final_matrix["availability_pct"].between(0, 1).all()
    assert final_matrix["availability_at_k"].le(len(eligible_cases)).all()
    assert final_matrix["availability_at_k"].ge(0).all()
    assert final_matrix["feature_group"].isin(
        ["case-level", "event-process", "offer-sparse", "identifier"]
    ).all()

    decision_counts = final_matrix["availability_decision"].value_counts()
    assert decision_counts.to_dict() == {
        "Available": 11,
        "Conditionally Available": 7,
        "Leakage Risk": 1,
    }

    return final_matrix, decision_counts


def export_final_feature_availability_matrix(
    final_feature_availability_matrix,
    output_path=FEATURE_AVAILABILITY_MATRIX_PATH,
):
    """Export the final matrix and validate its machine-readable round trip."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_feature_availability_matrix.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    exported_matrix = pd.read_csv(output_path, encoding="utf-8")
    expected_columns = [
        "feature",
        "feature_group",
        "availability_at_k",
        "availability_pct",
        "practically_available",
        "future_dependency",
        "availability_decision",
        "model_handling",
        "reason",
        "notes",
    ]

    assert exported_matrix.columns.tolist() == expected_columns
    assert len(exported_matrix) == 19
    assert exported_matrix["feature"].notna().all()
    assert exported_matrix["feature"].nunique() == 19
    assert exported_matrix["feature"].tolist() == (
        final_feature_availability_matrix["feature"].tolist()
    )
    pd.testing.assert_series_equal(
        exported_matrix["availability_decision"],
        final_feature_availability_matrix["availability_decision"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        exported_matrix["availability_at_k"],
        final_feature_availability_matrix["availability_at_k"],
        check_names=False,
        check_dtype=False,
    )
    pd.testing.assert_series_equal(
        exported_matrix["availability_pct"],
        final_feature_availability_matrix["availability_pct"],
        check_names=False,
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=0,
    )

    return output_path


def print_results(
    summary,
    prefix_sizes,
    case_level_feature_availability,
    first_seen_distributions,
    requested_amount_zero_cases_at_event_1,
    event_process_completeness,
    event_process_missingness_by_event_position,
    sparse_feature_availability,
    sparse_first_seen_distributions,
    sparse_activity_context,
    sparse_event_origin_context,
    semantic_value_distributions,
    semantic_outcome_value_presence_crosstabs,
    semantic_outcome_value_pattern_crosstabs,
    semantic_multiplicity_summary,
    semantic_offer_linkage_summary,
    semantic_lifecycle_relationships,
    representative_offer_traces,
    context_only_offer_confirmation,
    identifier_availability,
    event_id_uniqueness_results,
    case_id_frequency_validation,
    final_feature_availability_matrix,
    availability_decision_summary,
):
    """Print the boundary validation and evidence through Section 3.5.1."""
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

    print("\nAll Section 3.1 assertions passed.")

    print("\n=== SECTION 3.2.1 -- EVENT / PROCESS ATTRIBUTE COMPLETENESS ===")
    percentage_columns = {
        "row_completeness_percentage": "{:.2f}%".format,
        "missing_rate": "{:.2f}%".format,
        "case_coverage_percentage": "{:.2f}%".format,
    }
    print(
        event_process_completeness.to_string(
            index=False,
            formatters=percentage_columns,
        )
    )

    print("\n=== SECTION 3.2.1 -- MISSINGNESS BY EVENT POSITION ===")
    for feature, by_position in (
        event_process_missingness_by_event_position.items()
    ):
        print(f"\n--- {feature} ---")
        print(
            by_position.to_string(
                formatters={"missing_pct": "{:.2f}%".format}
            )
        )

    print("\n=== CASES COMPLETELY MISSING EACH FEATURE IN EVENTS 1-10 ===")
    completely_missing = event_process_completeness.set_index("feature")[
        "cases_with_no_value"
    ]
    print(completely_missing.to_string())
    print("\nAll Section 3.2.1 assertions passed.")

    print("\n=== SECTION 3.3.1 -- SPARSE / OFFER AVAILABILITY AUDIT ===")
    print(
        sparse_feature_availability.to_string(
            index=False,
            formatters={
                "available_by_k_pct": "{:.2f}%".format,
                "future_only_pct": "{:.2f}%".format,
                "never_observed_pct": "{:.2f}%".format,
            },
        )
    )

    print("\n=== SECTION 3.3.1 -- FIRST-SEEN EVENT DISTRIBUTIONS ===")
    for feature, distribution in sparse_first_seen_distributions.items():
        print(f"\n--- {feature} ---")
        if distribution.empty:
            print("No eligible case ever contains a non-null value.")
        else:
            print(distribution.to_string())

    print("\n=== SECTION 3.3.1 -- NON-NULL ROW ACTIVITY CONTEXT ===")
    for feature, distribution in sparse_activity_context.items():
        print(f"\n--- {feature} ---")
        if distribution.empty:
            print("No non-null rows in eligible traces.")
        else:
            print(distribution.to_string())

    print("\n=== SECTION 3.3.1 -- NON-NULL ROW EVENTORIGIN CONTEXT ===")
    for feature, distribution in sparse_event_origin_context.items():
        print(f"\n--- {feature} ---")
        if distribution.empty:
            print("No non-null rows in eligible traces.")
        else:
            print(distribution.to_string())

    print("\n=== SECTION 3.3.1 -- MULTI-VALUE CASE COUNTS ===")
    print(
        sparse_feature_availability.set_index("feature")[
            "multi_value_cases"
        ].to_string()
    )
    print("\nAll Section 3.3.1 assertions passed.")

    print("\n=== SECTION 3.3.2 -- ACCEPTED / SELECTED VALUE DISTRIBUTIONS ===")
    for feature, distribution in semantic_value_distributions.items():
        print(f"\n--- {feature} on O_Create Offer rows ---")
        print(
            distribution.to_string(
                formatters={"percentage": "{:.2f}%".format}
            )
        )

    print("\n=== SECTION 3.3.2 -- CASE / VALUE MULTIPLICITY ===")
    print(semantic_multiplicity_summary.to_string(index=False))
    print(
        "\nValue-presence cross-tabs count a case once under every distinct "
        "value it contains; totals can therefore exceed unique cases."
    )
    for feature, crosstab in (
        semantic_outcome_value_presence_crosstabs.items()
    ):
        print(f"\n--- {feature} value presence by application outcome ---")
        print(crosstab.to_string())

    print(
        "\nValue-pattern cross-tabs retain exactly one row per case and "
        "show multi-value cases explicitly."
    )
    for feature, crosstab in (
        semantic_outcome_value_pattern_crosstabs.items()
    ):
        print(f"\n--- {feature} observed value pattern by outcome ---")
        print(crosstab.to_string())

    print("\n=== SECTION 3.3.2 -- O_CREATE OFFER LINKAGE ===")
    print(semantic_offer_linkage_summary.to_string())
    print(
        "\nLifecycle rows are linked only through the OfferID on the "
        "immediately following O_Created event. O_Sent variants are combined."
    )
    for feature, relationship in semantic_lifecycle_relationships.items():
        print(f"\n--- {feature} versus later states of the same offer ---")
        percentage_columns = {
            column: "{:.2f}%".format
            for column in relationship.columns
            if column.endswith("_pct")
        }
        print(
            relationship.to_string(
                index=False,
                formatters=percentage_columns,
            )
        )

    print("\n=== SECTION 3.3.2 -- REPRESENTATIVE MULTI-OFFER TRACES ===")
    for case_id, trace in representative_offer_traces.items():
        print(f"\n--- {case_id} ---")
        print(trace.to_string(index=False))

    print("\n=== SECTION 3.3.2 -- CONTEXT-ONLY OFFER ATTRIBUTES ===")
    print(context_only_offer_confirmation.to_string(index=False))
    print(
        "\nOfferID availability is kept separate from model suitability; "
        "availability alone does not make the raw identifier a useful predictor."
    )
    print("\nAll Section 3.3.2 assertions passed.")

    print("\n=== SECTION 3.4 -- IDENTIFIER ATTRIBUTE AUDIT ===")
    print(
        identifier_availability.to_string(
            index=False,
            formatters={
                "row_completeness_percentage": "{:.2f}%".format,
                "case_coverage_percentage": "{:.2f}%".format,
            },
        )
    )

    print("\n=== EVENTID UNIQUENESS ===")
    print(event_id_uniqueness_results.to_string())

    print("\n=== CASE-ID FREQUENCY VALIDATION ===")
    print(case_id_frequency_validation.to_string())
    print(
        "\nBoth identifiers are available at prediction time in prefix_k10. "
        "Their availability does not make them suitable as raw predictors."
    )
    print("\nAll Section 3.4 assertions passed.")

    print("\n=== SECTION 3.5.1 -- FINAL FEATURE AVAILABILITY MATRIX ===")
    concise_columns = [
        "feature",
        "feature_group",
        "availability_at_k",
        "availability_pct",
        "practically_available",
        "future_dependency",
        "availability_decision",
        "model_handling",
    ]
    print(
        final_feature_availability_matrix[concise_columns].to_string(
            index=False,
            formatters={"availability_pct": "{:.2%}".format},
        )
    )
    print("\n=== AVAILABILITY-DECISION SUMMARY ===")
    print(availability_decision_summary.to_string())
    print("\nAll Section 3.5.1 assertions passed.")


def main():
    """Run the validated boundary and audits through Section 3.5.1."""
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
    global event_process_completeness
    global event_process_missingness_by_event_position
    global eligible_log
    global sparse_feature_availability
    global sparse_first_seen_distributions
    global sparse_activity_context
    global sparse_event_origin_context
    global semantic_value_distributions
    global semantic_outcome_value_presence_crosstabs
    global semantic_outcome_value_pattern_crosstabs
    global semantic_multiplicity_summary
    global semantic_offer_linkage_summary
    global semantic_lifecycle_relationships
    global representative_offer_traces
    global context_only_offer_confirmation
    global identifier_availability
    global event_id_uniqueness_results
    global case_id_frequency_validation
    global final_feature_availability_matrix
    global availability_decision_summary

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
    (
        event_process_completeness,
        event_process_missingness_by_event_position,
    ) = audit_event_process_completeness(
        prefix_k10=prefix_k10,
        eligible_cases=eligible_cases,
    )
    eligible_log = df_work.loc[df_work[CASE_COL].isin(eligible_cases)].copy()
    (
        sparse_feature_availability,
        sparse_first_seen_distributions,
        sparse_activity_context,
        sparse_event_origin_context,
    ) = audit_sparse_feature_availability(
        df_work=df_work,
        eligible_cases=eligible_cases,
        eligible_log=eligible_log,
        prefix_k10=prefix_k10,
    )
    (
        semantic_value_distributions,
        semantic_outcome_value_presence_crosstabs,
        semantic_outcome_value_pattern_crosstabs,
        semantic_multiplicity_summary,
        semantic_offer_linkage_summary,
        semantic_lifecycle_relationships,
        representative_offer_traces,
        context_only_offer_confirmation,
    ) = audit_sparse_feature_semantics(
        eligible_log=eligible_log,
        prefix_k10=prefix_k10,
        case_target=case_target,
        sparse_feature_availability=sparse_feature_availability,
    )
    (
        identifier_availability,
        event_id_uniqueness_results,
        case_id_frequency_validation,
    ) = audit_identifier_attributes(
        prefix_k10=prefix_k10,
        eligible_cases=eligible_cases,
    )
    (
        final_feature_availability_matrix,
        availability_decision_summary,
    ) = build_final_feature_availability_matrix(
        case_level_feature_availability=case_level_feature_availability,
        event_process_completeness=event_process_completeness,
        sparse_feature_availability=sparse_feature_availability,
        identifier_availability=identifier_availability,
        eligible_cases=eligible_cases,
    )
    exported_matrix_path = export_final_feature_availability_matrix(
        final_feature_availability_matrix
    )
    print(f"\nExported final matrix: {exported_matrix_path}")
    print("CSV round-trip validation passed.")
    print_results(
        summary=k10_validation_summary,
        prefix_sizes=prefix_sizes,
        case_level_feature_availability=case_level_feature_availability,
        first_seen_distributions=case_level_first_seen_distributions,
        requested_amount_zero_cases_at_event_1=(
            requested_amount_zero_cases_at_event_1
        ),
        event_process_completeness=event_process_completeness,
        event_process_missingness_by_event_position=(
            event_process_missingness_by_event_position
        ),
        sparse_feature_availability=sparse_feature_availability,
        sparse_first_seen_distributions=sparse_first_seen_distributions,
        sparse_activity_context=sparse_activity_context,
        sparse_event_origin_context=sparse_event_origin_context,
        semantic_value_distributions=semantic_value_distributions,
        semantic_outcome_value_presence_crosstabs=(
            semantic_outcome_value_presence_crosstabs
        ),
        semantic_outcome_value_pattern_crosstabs=(
            semantic_outcome_value_pattern_crosstabs
        ),
        semantic_multiplicity_summary=semantic_multiplicity_summary,
        semantic_offer_linkage_summary=semantic_offer_linkage_summary,
        semantic_lifecycle_relationships=semantic_lifecycle_relationships,
        representative_offer_traces=representative_offer_traces,
        context_only_offer_confirmation=context_only_offer_confirmation,
        identifier_availability=identifier_availability,
        event_id_uniqueness_results=event_id_uniqueness_results,
        case_id_frequency_validation=case_id_frequency_validation,
        final_feature_availability_matrix=(
            final_feature_availability_matrix
        ),
        availability_decision_summary=availability_decision_summary,
    )


if __name__ == "__main__":
    main()
