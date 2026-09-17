"""Freeze Temporal Evaluation V1 Part 1 window membership.

The temporal boundaries are selected from the complete frozen TEST population
using only ``case_start_time`` and canonical TEST row order.  Frozen SHAP cases
then inherit those boundaries by ``case_id``.  This module does not load a
model, predictions, or SHAP values and does not fit or recompute anything.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = PROJECT_ROOT / "docs" / "temporal_evaluation_design_v1.md"
FEATURE_TABLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_k10"
    / "case_features_k10.parquet"
)
BASELINE_RESULTS_DIR = PROJECT_ROOT / "results" / "baseline_v1"
TEST_IDS_PATH = BASELINE_RESULTS_DIR / "test_case_ids.csv"
SPLIT_SUMMARY_PATH = BASELINE_RESULTS_DIR / "split_summary.json"
SHAP_RESULTS_DIR = PROJECT_ROOT / "results" / "shap_pilot_v1"
EXPLAINED_IDS_PATH = SHAP_RESULTS_DIR / "explained_case_ids.csv"
SAMPLE_MANIFEST_PATH = SHAP_RESULTS_DIR / "sample_manifest.json"
FULL_SHAP_METADATA_PATH = SHAP_RESULTS_DIR / "full_shap_case_metadata.csv"
FULL_SHAP_MANIFEST_PATH = SHAP_RESULTS_DIR / "full_shap_manifest.json"

OUTPUT_DIR = PROJECT_ROOT / "results" / "temporal_eval_v1"
TEMPORAL_WINDOWS_PATH = OUTPUT_DIR / "temporal_windows.csv"
TEMPORAL_SHAP_WINDOWS_PATH = OUTPUT_DIR / "temporal_shap_windows.csv"
SUMMARY_PATH = OUTPUT_DIR / "temporal_window_summary.json"

EXPECTED_TEST_CASES = 6_276
EXPECTED_SHAP_CASES = 1_000
WINDOW_NAMES = ("T1", "T2", "T3")
NOMINAL_CASES_PER_WINDOW = 2_092
NOMINAL_BOUNDARIES = (2_092, 4_184)
BOUNDARY_SELECTION_INPUTS = ("case_start_time", "canonical_test_row_index")


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


def _require(condition: bool, message: str) -> None:
    """Raise a stable validation error when an invariant is false."""
    if not condition:
        raise AssertionError(message)


def _protected_paths() -> list[Path]:
    """Collect frozen inputs whose bytes must remain unchanged."""
    protected_roots = (
        BASELINE_RESULTS_DIR,
        SHAP_RESULTS_DIR,
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
    paths.add(DESIGN_PATH)
    required_paths = {
        DESIGN_PATH,
        FEATURE_TABLE_PATH,
        TEST_IDS_PATH,
        SPLIT_SUMMARY_PATH,
        EXPLAINED_IDS_PATH,
        SAMPLE_MANIFEST_PATH,
        FULL_SHAP_METADATA_PATH,
        FULL_SHAP_MANIFEST_PATH,
        PROJECT_ROOT
        / "artifacts"
        / "shap_pilot_v1"
        / "random_forest_baseline.joblib",
        PROJECT_ROOT
        / "data"
        / "processed"
        / "shap_pilot_v1"
        / "shap_values_success.npy",
    }
    missing = sorted(
        (path for path in required_paths if not path.is_file()),
        key=lambda path: str(path),
    )
    if missing:
        raise FileNotFoundError(f"Protected artifacts are missing: {missing}")
    paths.update(required_paths)
    return sorted(paths, key=_relative)


def protect_artifact_hashes(paths: Iterable[Path]) -> dict[Path, str]:
    """Hash frozen artifacts so before/after identity can be enforced."""
    return {path: _sha256(path) for path in paths}


def load_canonical_test_population(
    test_ids_path: Path = TEST_IDS_PATH,
    feature_table_path: Path = FEATURE_TABLE_PATH,
    split_summary_path: Path = SPLIT_SUMMARY_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join canonical TEST identities to frozen case-level time and target."""
    split_summary = _load_json(split_summary_path)
    _require(
        split_summary.get("test_cases") == EXPECTED_TEST_CASES,
        "Frozen split summary does not declare 6,276 TEST cases",
    )
    _require(
        split_summary.get("case_id_column") == "case_id",
        "Frozen split case identifier is not case_id",
    )
    _require(
        split_summary.get("case_start_time_column") == "case_start_time",
        "Frozen split time variable is not case_start_time",
    )
    _require(
        split_summary.get("target_column") == "target",
        "Frozen split target column is not target",
    )
    recorded_source_hash = split_summary.get("source_feature_table_sha256")
    _require(
        recorded_source_hash == _sha256(feature_table_path),
        "Frozen case-feature table hash differs from split_summary.json",
    )

    canonical = pd.read_csv(
        test_ids_path,
        encoding="utf-8",
        dtype={"case_id": "string"},
    )
    _require(
        canonical.columns.tolist() == ["case_id"],
        f"Unexpected canonical TEST columns: {canonical.columns.tolist()}",
    )
    canonical = canonical.copy()
    canonical["canonical_test_row_index"] = range(len(canonical))

    features = pd.read_parquet(
        feature_table_path,
        columns=["case_id", "case_start_time", "target"],
        engine="pyarrow",
    )
    features["case_id"] = features["case_id"].astype("string")
    features["case_start_time"] = pd.to_datetime(
        features["case_start_time"], utc=True, errors="raise"
    )
    _require(features["case_id"].is_unique, "Case-feature case_id is not unique")

    joined = canonical.merge(
        features,
        on="case_id",
        how="left",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    _require(
        joined["_merge"].eq("both").all(),
        "At least one canonical TEST case is absent from the case-feature table",
    )
    joined = joined.drop(columns="_merge")
    return joined, split_summary


def validate_test_identity(
    test_population: pd.DataFrame,
    split_summary: dict[str, Any],
) -> None:
    """Validate the complete canonical TEST join before boundary selection."""
    _require(
        len(test_population) == EXPECTED_TEST_CASES,
        f"Expected 6,276 TEST rows, found {len(test_population)}",
    )
    _require(
        test_population["canonical_test_row_index"].tolist()
        == list(range(EXPECTED_TEST_CASES)),
        "Canonical TEST row order was not preserved",
    )
    _require(
        test_population["case_id"].notna().all(),
        "Canonical TEST contains a missing case_id",
    )
    _require(
        test_population["case_id"].is_unique,
        "Canonical TEST contains duplicate case_id values",
    )
    _require(
        test_population["case_start_time"].notna().all(),
        "TEST case_start_time contains missing values",
    )
    _require(
        test_population["target"].notna().all(),
        "TEST target contains missing values",
    )
    _require(
        set(test_population["target"].unique().tolist()) == {0, 1},
        "TEST target must contain exactly the labels 0 and 1",
    )
    target_counts = test_population["target"].value_counts().to_dict()
    _require(
        target_counts
        == {
            0: split_summary.get("test_unsuccessful"),
            1: split_summary.get("test_success"),
        },
        "Joined TEST target distribution differs from split_summary.json",
    )


def build_chronological_order(test_population: pd.DataFrame) -> pd.DataFrame:
    """Sort deterministically by time, using canonical order only for ties."""
    ordered = test_population.sort_values(
        ["case_start_time", "canonical_test_row_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    ordered.insert(0, "temporal_order", range(len(ordered)))
    _require(
        ordered["case_start_time"].is_monotonic_increasing,
        "Chronological ordering is not nondecreasing",
    )
    return ordered


def find_nearest_valid_boundary(
    ordered_times: pd.Series,
    nominal_position: int,
) -> int:
    """Choose the closest split between distinct timestamps.

    Absolute distance is minimized; an equal-distance tie selects the earlier
    split position.  Positions are cumulative row counts, not row indices.
    """
    population_size = len(ordered_times)
    if not 0 < nominal_position < population_size:
        raise ValueError(f"Invalid nominal boundary: {nominal_position}")
    valid_positions = [
        position
        for position in range(1, population_size)
        if ordered_times.iloc[position - 1] != ordered_times.iloc[position]
    ]
    if not valid_positions:
        raise ValueError(
            "Strict temporal windows are impossible because no valid timestamp "
            "split position exists"
        )
    return min(
        valid_positions,
        key=lambda position: (abs(position - nominal_position), position),
    )


def assign_temporal_windows(
    ordered: pd.DataFrame,
    boundaries: tuple[int, int],
) -> pd.DataFrame:
    """Assign exactly one T1/T2/T3 label using frozen cumulative positions."""
    first, second = boundaries
    if not 0 < first < second < len(ordered):
        raise ValueError(f"Invalid actual boundary positions: {boundaries}")
    assigned = ordered.copy()
    assigned["window"] = (
        ["T1"] * first
        + ["T2"] * (second - first)
        + ["T3"] * (len(assigned) - second)
    )
    assigned["window"] = pd.Categorical(
        assigned["window"], categories=WINDOW_NAMES, ordered=True
    )
    return assigned


def validate_window_boundaries(
    canonical_test: pd.DataFrame,
    temporal_windows: pd.DataFrame,
) -> dict[str, str]:
    """Run the required full-TEST Part 1 membership assertions A--Q."""
    results: dict[str, str] = {}
    canonical_ids = canonical_test["case_id"].tolist()
    canonical_id_set = set(canonical_ids)

    _require(len(temporal_windows) == EXPECTED_TEST_CASES, "A failed")
    results["A_full_test_population_exactly_6276"] = "PASS"

    observed_counts = temporal_windows["case_id"].value_counts().to_dict()
    _require(
        all(observed_counts.get(case_id) == 1 for case_id in canonical_ids),
        "B failed: a canonical TEST case does not appear exactly once",
    )
    results["B_every_canonical_test_case_appears_once"] = "PASS"

    _require(
        set(temporal_windows["case_id"]).issubset(canonical_id_set),
        "C failed: a non-TEST case appears",
    )
    results["C_no_non_test_case_appears"] = "PASS"

    _require(temporal_windows["case_id"].is_unique, "D failed")
    results["D_no_duplicate_case_id"] = "PASS"

    _require(temporal_windows["case_start_time"].notna().all(), "E failed")
    results["E_case_start_time_complete"] = "PASS"

    _require(temporal_windows["target"].notna().all(), "F failed")
    results["F_target_complete"] = "PASS"

    _require(set(temporal_windows["target"].unique()) == {0, 1}, "G failed")
    results["G_target_is_binary_0_1"] = "PASS"

    _require(
        temporal_windows["window"].notna().all()
        and set(temporal_windows["window"].astype(str)) == set(WINDOW_NAMES),
        "H failed: invalid or missing temporal window",
    )
    results["H_every_case_assigned_exactly_one_valid_window"] = "PASS"

    window_id_sets = {
        name: set(temporal_windows.loc[temporal_windows["window"] == name, "case_id"])
        for name in WINDOW_NAMES
    }
    _require(
        set().union(*window_id_sets.values()) == canonical_id_set,
        "I failed: window union differs from TEST population",
    )
    results["I_window_union_equals_complete_test"] = "PASS"

    _require(
        window_id_sets["T1"].isdisjoint(window_id_sets["T2"])
        and window_id_sets["T1"].isdisjoint(window_id_sets["T3"])
        and window_id_sets["T2"].isdisjoint(window_id_sets["T3"]),
        "J failed: temporal windows overlap",
    )
    results["J_windows_are_mutually_disjoint"] = "PASS"

    timestamp_window_counts = temporal_windows.groupby(
        "case_start_time", observed=True
    )["window"].nunique()
    _require(timestamp_window_counts.max() == 1, "K failed: timestamp split")
    results["K_no_timestamp_crosses_window_boundary"] = "PASS"

    grouped = {
        name: temporal_windows.loc[temporal_windows["window"] == name]
        for name in WINDOW_NAMES
    }
    _require(
        grouped["T1"]["case_start_time"].max()
        < grouped["T2"]["case_start_time"].min(),
        "L failed: T1 and T2 are not strictly separated",
    )
    results["L_max_T1_time_before_min_T2_time"] = "PASS"

    _require(
        grouped["T2"]["case_start_time"].max()
        < grouped["T3"]["case_start_time"].min(),
        "M failed: T2 and T3 are not strictly separated",
    )
    results["M_max_T2_time_before_min_T3_time"] = "PASS"

    _require(
        temporal_windows["case_start_time"].is_monotonic_increasing
        and temporal_windows["temporal_order"].tolist()
        == list(range(EXPECTED_TEST_CASES)),
        "N failed: windows are not chronologically ordered",
    )
    results["N_windows_chronologically_ordered"] = "PASS"

    _require("target" not in BOUNDARY_SELECTION_INPUTS, "O failed")
    results["O_target_not_used_to_select_boundaries"] = "PASS"

    _require(
        not any("prediction" in value for value in BOUNDARY_SELECTION_INPUTS),
        "P failed",
    )
    results["P_prediction_not_used_to_select_boundaries"] = "PASS"

    _require(
        not any("shap" in value for value in BOUNDARY_SELECTION_INPUTS),
        "Q failed",
    )
    results["Q_shap_not_used_to_select_boundaries"] = "PASS"
    return results


def assign_shap_cases_to_windows(
    temporal_windows: pd.DataFrame,
    explained_path: Path = EXPLAINED_IDS_PATH,
    full_metadata_path: Path = FULL_SHAP_METADATA_PATH,
    sample_manifest_path: Path = SAMPLE_MANIFEST_PATH,
    full_manifest_path: Path = FULL_SHAP_MANIFEST_PATH,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Join frozen SHAP identities to full-TEST temporal membership."""
    sample_manifest = _load_json(sample_manifest_path)
    full_manifest = _load_json(full_manifest_path)
    _require(
        sample_manifest.get("explained_source") == "test"
        and sample_manifest.get("explained_sample_size") == EXPECTED_SHAP_CASES,
        "SHAP sample manifest does not describe 1,000 frozen TEST cases",
    )
    _require(
        full_manifest.get("explained_cases") == EXPECTED_SHAP_CASES
        and full_manifest.get("metadata_rows") == EXPECTED_SHAP_CASES
        and full_manifest.get("resampled") is False,
        "Full SHAP manifest identity contract changed",
    )
    expected_metadata_hash = full_manifest.get("canonical_artifact_hashes", {}).get(
        "full_shap_case_metadata_csv_sha256"
    )
    _require(
        expected_metadata_hash == _sha256(full_metadata_path),
        "Full SHAP metadata hash differs from full_shap_manifest.json",
    )

    identity_columns = ["sample_order", "source_row_index", "case_id", "target"]
    explained = pd.read_csv(
        explained_path,
        encoding="utf-8",
        dtype={"case_id": "string"},
    )
    _require(
        explained.columns.tolist() == identity_columns,
        f"Unexpected explained-case columns: {explained.columns.tolist()}",
    )
    full_metadata = pd.read_csv(
        full_metadata_path,
        encoding="utf-8",
        dtype={"case_id": "string"},
    )
    _require(
        all(column in full_metadata.columns for column in identity_columns),
        "Full SHAP metadata is missing frozen identity columns",
    )
    pd.testing.assert_frame_equal(
        explained,
        full_metadata[identity_columns],
        check_exact=True,
    )

    membership = temporal_windows[
        ["case_id", "target", "case_start_time", "window"]
    ].rename(columns={"target": "canonical_target"})
    assigned = explained.merge(
        membership,
        on="case_id",
        how="left",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )

    results: dict[str, str] = {}
    _require(len(explained) == EXPECTED_SHAP_CASES, "SHAP A failed")
    results["A_explained_population_exactly_1000"] = "PASS"

    test_ids = set(temporal_windows["case_id"])
    _require(set(explained["case_id"]).issubset(test_ids), "SHAP B failed")
    results["B_every_explained_case_belongs_to_test"] = "PASS"

    _require(assigned["_merge"].eq("both").all(), "SHAP C failed")
    results["C_every_explained_case_in_temporal_windows"] = "PASS"

    _require(assigned["window"].notna().all(), "SHAP D failed")
    results["D_every_explained_case_has_one_window"] = "PASS"

    _require(explained["case_id"].is_unique, "SHAP E failed")
    results["E_no_explained_case_duplicated"] = "PASS"

    _require(
        assigned["sample_order"].tolist() == explained["sample_order"].tolist(),
        "SHAP F failed",
    )
    results["F_sample_order_unchanged"] = "PASS"

    _require(
        assigned["source_row_index"].tolist()
        == explained["source_row_index"].tolist(),
        "SHAP G failed",
    )
    results["G_source_row_index_unchanged"] = "PASS"

    _require(
        assigned["target"].tolist() == assigned["canonical_target"].tolist(),
        "SHAP H failed: frozen target differs from canonical TEST target",
    )
    results["H_original_target_unchanged"] = "PASS"

    _require(set(assigned["case_id"]) == set(explained["case_id"]), "SHAP I failed")
    results["I_no_shap_case_added"] = "PASS"

    _require(len(assigned) == len(explained), "SHAP J failed")
    results["J_no_shap_case_removed"] = "PASS"

    _require(
        assigned["case_id"].tolist() == explained["case_id"].tolist(),
        "SHAP K failed: frozen case order changed",
    )
    results["K_no_shap_case_resampled"] = "PASS"

    inherited = temporal_windows.set_index("case_id")["window"].astype(str)
    expected_windows = explained["case_id"].map(inherited)
    _require(
        assigned["window"].astype(str).tolist() == expected_windows.tolist(),
        "SHAP L failed: membership was not inherited from full TEST windows",
    )
    results["L_membership_inherited_from_full_test_boundaries"] = "PASS"

    source_indices = explained["source_row_index"].astype(int)
    _require(
        source_indices.between(0, EXPECTED_TEST_CASES - 1).all(),
        "Frozen SHAP source_row_index is outside canonical TEST bounds",
    )
    canonical_order = temporal_windows.sort_values("canonical_test_row_index")
    source_rows = canonical_order.iloc[source_indices.to_numpy()]
    _require(
        source_rows["case_id"].tolist() == explained["case_id"].tolist()
        and source_rows["target"].tolist() == explained["target"].tolist(),
        "Frozen SHAP source-row identity differs from canonical TEST order",
    )

    output = assigned[
        [
            "sample_order",
            "source_row_index",
            "case_id",
            "target",
            "case_start_time",
            "window",
        ]
    ].copy()
    return output, results


def _isoformat(value: pd.Timestamp) -> str:
    """Serialize UTC pandas timestamps deterministically."""
    return value.isoformat()


def summarize_windows(
    temporal_windows: pd.DataFrame,
    temporal_shap_windows: pd.DataFrame,
    boundaries: tuple[int, int],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Create descriptive TEST, boundary, and frozen-SHAP summaries."""
    window_summaries: dict[str, dict[str, Any]] = {}
    shap_summaries: dict[str, Any] = {}
    for name in WINDOW_NAMES:
        frame = temporal_windows.loc[temporal_windows["window"] == name]
        success = int(frame["target"].eq(1).sum())
        unsuccessful = int(frame["target"].eq(0).sum())
        window_summaries[name] = {
            "window": name,
            "case_count": len(frame),
            "first_temporal_order": int(frame["temporal_order"].iloc[0]),
            "last_temporal_order": int(frame["temporal_order"].iloc[-1]),
            "start_case_start_time": _isoformat(frame["case_start_time"].iloc[0]),
            "end_case_start_time": _isoformat(frame["case_start_time"].iloc[-1]),
            "success_count": success,
            "unsuccessful_count": unsuccessful,
        }

        shap_frame = temporal_shap_windows.loc[
            temporal_shap_windows["window"] == name
        ]
        shap_success = int(shap_frame["target"].eq(1).sum())
        shap_unsuccessful = int(shap_frame["target"].eq(0).sum())
        shap_summaries[name] = {
            "case_count": len(shap_frame),
            "success_count": shap_success,
            "unsuccessful_count": shap_unsuccessful,
            "success_rate": shap_success / len(shap_frame),
        }

    boundary_summaries: list[dict[str, Any]] = []
    for label, nominal, actual in zip(
        ("T1_T2", "T2_T3"), NOMINAL_BOUNDARIES, boundaries, strict=True
    ):
        boundary_summaries.append(
            {
                "boundary": label,
                "nominal_cumulative_position": nominal,
                "actual_cumulative_split_position": actual,
                "adjustment_from_nominal": actual - nominal,
                "timestamp_immediately_before_boundary": _isoformat(
                    temporal_windows.loc[actual - 1, "case_start_time"]
                ),
                "timestamp_immediately_after_boundary": _isoformat(
                    temporal_windows.loc[actual, "case_start_time"]
                ),
                "timestamp_tie_adjustment_required": actual != nominal,
            }
        )
    return window_summaries, boundary_summaries, shap_summaries


def _csv_ready(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a deterministic output copy with ISO-8601 timestamps."""
    output = frame.copy()
    output["case_start_time"] = output["case_start_time"].map(_isoformat)
    output["window"] = output["window"].astype(str)
    return output


def write_summary(summary: dict[str, Any], path: Path = SUMMARY_PATH) -> None:
    """Write stable, human-readable JSON with a final newline."""
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
        file.write("\n")


def _print_summary(summary: dict[str, Any]) -> None:
    """Print the measured Part 1 freeze result and artifact hashes."""
    print("=== TEMPORAL EVALUATION V1 PART 1 ===")
    print(f"Full TEST cases: {summary['total_test_cases']:,}")
    print(f"Ordering: {summary['chronological_ordering']}")
    for boundary in summary["actual_boundaries"]:
        print(
            f"{boundary['boundary']}: nominal "
            f"{boundary['nominal_cumulative_position']:,}; actual "
            f"{boundary['actual_cumulative_split_position']:,}; adjustment "
            f"{boundary['adjustment_from_nominal']:+,}; tie adjustment "
            f"{boundary['timestamp_tie_adjustment_required']}"
        )
    for name, values in summary["windows"].items():
        print(
            f"{name}: {values['case_count']:,} cases, "
            f"{values['start_case_start_time']} to "
            f"{values['end_case_start_time']}, "
            f"Success={values['success_count']:,}, "
            f"Unsuccessful={values['unsuccessful_count']:,}"
        )
    for name, values in summary["shap_target_distribution_per_window"].items():
        print(
            f"SHAP {name}: {values['case_count']:,} cases, "
            f"Success={values['success_count']:,}, "
            f"Unsuccessful={values['unsuccessful_count']:,}, "
            f"Success rate={values['success_rate']:.4%}"
        )
    print(f"Strict temporal separation: {summary['strict_temporal_separation']}")
    print(f"Protected artifacts unchanged: {summary['protected_artifacts']['unchanged']}")
    print("Generated artifacts:")
    for path, digest in summary["generated_artifact_sha256"].items():
        print(f"  {path}: {digest}")


def main() -> None:
    """Build, validate, write, and freeze Temporal Evaluation V1 Part 1."""
    protected_paths = _protected_paths()
    protected_before = protect_artifact_hashes(protected_paths)

    canonical_test, split_summary = load_canonical_test_population()
    validate_test_identity(canonical_test, split_summary)
    ordered = build_chronological_order(canonical_test)
    boundaries = tuple(
        find_nearest_valid_boundary(ordered["case_start_time"], nominal)
        for nominal in NOMINAL_BOUNDARIES
    )
    _require(len(boundaries) == 2, "Expected exactly two temporal boundaries")
    actual_boundaries = (int(boundaries[0]), int(boundaries[1]))
    temporal_windows = assign_temporal_windows(ordered, actual_boundaries)
    test_assertions = validate_window_boundaries(
        canonical_test, temporal_windows
    )
    temporal_shap_windows, shap_assertions = assign_shap_cases_to_windows(
        temporal_windows
    )
    windows, boundary_details, shap_windows = summarize_windows(
        temporal_windows, temporal_shap_windows, actual_boundaries
    )

    protected_after = protect_artifact_hashes(protected_paths)
    _require(
        protected_after == protected_before,
        "A protected frozen artifact changed during temporal window creation",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_output = _csv_ready(
        temporal_windows[
            [
                "temporal_order",
                "case_id",
                "case_start_time",
                "target",
                "window",
                "canonical_test_row_index",
            ]
        ]
    )
    shap_output = _csv_ready(temporal_shap_windows)
    full_output.to_csv(
        TEMPORAL_WINDOWS_PATH,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    shap_output.to_csv(
        TEMPORAL_SHAP_WINDOWS_PATH,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    generated_hashes = {
        _relative(TEMPORAL_WINDOWS_PATH): _sha256(TEMPORAL_WINDOWS_PATH),
        _relative(TEMPORAL_SHAP_WINDOWS_PATH): _sha256(TEMPORAL_SHAP_WINDOWS_PATH),
    }
    summary: dict[str, Any] = {
        "experiment": "temporal_evaluation_v1",
        "stage": "part_1_window_freeze",
        "time_variable": "case_start_time",
        "chronological_ordering": [
            "case_start_time ascending",
            "canonical_test_row_index ascending (ties only)",
        ],
        "boundary_selection_inputs": list(BOUNDARY_SELECTION_INPUTS),
        "window_design": "chronological_equal_case",
        "window_count": 3,
        "total_test_cases": EXPECTED_TEST_CASES,
        "nominal_cases_per_window": NOMINAL_CASES_PER_WINDOW,
        "nominal_boundaries": list(NOMINAL_BOUNDARIES),
        "actual_boundaries": boundary_details,
        "tie_policy": (
            "Select the valid split between distinct case_start_time values "
            "nearest to each nominal cumulative position by absolute distance; "
            "if distances tie, select the earlier split position."
        ),
        "strict_temporal_separation": True,
        "windows": windows,
        "shap_explained_cases": EXPECTED_SHAP_CASES,
        "shap_cases_per_window": {
            name: values["case_count"] for name, values in shap_windows.items()
        },
        "shap_target_distribution_per_window": shap_windows,
        "resampling": False,
        "model_retrained": False,
        "preprocessing_refitted": False,
        "shap_recomputed": False,
        "prediction_metrics_computed": False,
        "shap_importance_computed": False,
        "full_test_membership_assertions": test_assertions,
        "shap_membership_assertions": shap_assertions,
        "assertion_status": "PASS",
        "protected_artifacts": {
            "count": len(protected_after),
            "unchanged": True,
            "sha256": {
                _relative(path): digest
                for path, digest in protected_after.items()
            },
        },
        "generated_artifact_sha256": generated_hashes,
    }
    write_summary(summary)
    summary["generated_artifact_sha256"][_relative(SUMMARY_PATH)] = _sha256(
        SUMMARY_PATH
    )
    _print_summary(summary)
    print("All Temporal Evaluation V1 Part 1 assertions passed.")


if __name__ == "__main__":
    main()
