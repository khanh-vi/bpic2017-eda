"""Run Temporal Evaluation V1 Part 3 from frozen SHAP Pilot V1 values.

This module performs descriptive temporal SHAP analysis only.  It reads the
canonical 1,000-by-165 positive-class SHAP matrix and frozen Part 1 temporal
membership.  It never imports an explanation library, loads a model, fits a
model or preprocessor, resamples cases, changes temporal boundaries, computes
prediction metrics, or performs formal drift/significance testing.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = PROJECT_ROOT / "docs" / "temporal_evaluation_design_v1.md"

BASELINE_DIR = PROJECT_ROOT / "results" / "baseline_v1"
SHAP_RESULTS_DIR = PROJECT_ROOT / "results" / "shap_pilot_v1"
SHAP_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "shap_pilot_v1"
OUTPUT_DIR = PROJECT_ROOT / "results" / "temporal_eval_v1"

FEATURE_NAMES_PATH = BASELINE_DIR / "feature_names_tree.csv"
SHAP_VALUES_PATH = SHAP_DATA_DIR / "shap_values_success.npy"
BASE_VALUES_PATH = SHAP_DATA_DIR / "base_values_success.npy"
PROBABILITIES_PATH = SHAP_DATA_DIR / "model_probabilities_success.npy"
FULL_MANIFEST_PATH = SHAP_RESULTS_DIR / "full_shap_manifest.json"
FULL_METADATA_PATH = SHAP_RESULTS_DIR / "full_shap_case_metadata.csv"
PILOT_GROUP_IMPORTANCE_PATH = SHAP_RESULTS_DIR / "global_shap_group_importance.csv"

TEMPORAL_WINDOWS_PATH = OUTPUT_DIR / "temporal_windows.csv"
TEMPORAL_SHAP_WINDOWS_PATH = OUTPUT_DIR / "temporal_shap_windows.csv"
WINDOW_SUMMARY_PATH = OUTPUT_DIR / "temporal_window_summary.json"

IMPORTANCE_PATH = OUTPUT_DIR / "temporal_shap_importance.csv"
TOP10_PATH = OUTPUT_DIR / "temporal_shap_top10.csv"
STABILITY_PATH = OUTPUT_DIR / "temporal_shap_stability.csv"
SIGNED_PATH = OUTPUT_DIR / "temporal_signed_shap.csv"
GROUP_IMPORTANCE_PATH = OUTPUT_DIR / "temporal_group_importance.csv"
SUMMARY_PATH = OUTPUT_DIR / "temporal_shap_summary.json"
TOP_FEATURES_PLOT_PATH = OUTPUT_DIR / "temporal_shap_top_features.png"
HEATMAP_PATH = OUTPUT_DIR / "temporal_shap_heatmap.png"
GROUP_PLOT_PATH = OUTPUT_DIR / "temporal_group_importance.png"

EXPECTED_CASES = 1_000
EXPECTED_FEATURES = 165
EXPECTED_SHAPE = (EXPECTED_CASES, EXPECTED_FEATURES)
POSITIVE_CLASS = 1
TOP_K = 10
ZERO_TOLERANCE = 1e-12
WINDOWS = ("T1", "T2", "T3")
WINDOW_COUNTS = {"T1": 325, "T2": 332, "T3": 343}
WINDOW_TARGET_COUNTS = {
    "T1": {0: 140, 1: 185},
    "T2": {0: 137, 1: 195},
    "T3": {0: 154, 1: 189},
}
COMPARISONS = (
    ("T1_to_T2", "T1", "T2"),
    ("T2_to_T3", "T2", "T3"),
    ("T1_to_T3", "T1", "T3"),
)
GROUP_ORDER = (
    "Numeric",
    "LoanGoal",
    "ApplicationType",
    "Activity",
    "Action",
    "EventOrigin",
    "Resource",
    "Lifecycle",
)
SOURCE_GROUP_MAP = {
    "numeric": "Numeric",
    "categorical::loan_goal": "LoanGoal",
    "categorical::application_type": "ApplicationType",
    "structured_map::activity": "Activity",
    "structured_map::action": "Action",
    "structured_map::origin": "EventOrigin",
    "structured_map::resource": "Resource",
    "structured_map::lifecycle": "Lifecycle",
}
EXPECTED_GROUP_COUNTS = {
    "Numeric": 5,
    "LoanGoal": 14,
    "ApplicationType": 2,
    "Activity": 14,
    "Action": 5,
    "EventOrigin": 3,
    "Resource": 115,
    "Lifecycle": 7,
}

OUTPUT_PATHS = (
    IMPORTANCE_PATH,
    TOP10_PATH,
    STABILITY_PATH,
    SIGNED_PATH,
    GROUP_IMPORTANCE_PATH,
    SUMMARY_PATH,
    TOP_FEATURES_PLOT_PATH,
    HEATMAP_PATH,
    GROUP_PLOT_PATH,
)


def _require(condition: bool, message: str) -> None:
    """Raise a stable validation error when an invariant is false."""
    if not condition:
        raise AssertionError(message)


def _relative(path: Path) -> str:
    """Return a stable repository-relative path."""
    return path.relative_to(PROJECT_ROOT).as_posix()


def _sha256(path: Path) -> str:
    """Hash one required file without changing it."""
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {_relative(path)}")
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    """Use the canonical SHAP Pilot array hashing convention."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    """Load a required JSON object."""
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    _require(isinstance(value, dict), f"Expected JSON object: {_relative(path)}")
    return value


def _write_text_atomic(path: Path, content: str) -> None:
    """Atomically write deterministic UTF-8 text."""
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


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    """Write a CSV with stable newlines and full double precision."""
    content = frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    _write_text_atomic(path, content)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Write stable, human-readable JSON with a final newline."""
    content = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    _write_text_atomic(path, content)


def _save_figure_atomic(path: Path, figure: plt.Figure) -> None:
    """Save a deterministic PNG through an atomic replacement."""
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        figure.savefig(
            temporary,
            format="png",
            dpi=160,
            bbox_inches="tight",
            facecolor="white",
            metadata={"Software": "Temporal Evaluation V1 Part 3"},
        )
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def _protected_paths() -> tuple[Path, ...]:
    """Collect upstream frozen artifacts while excluding Part 3 outputs."""
    roots = (
        BASELINE_DIR,
        SHAP_RESULTS_DIR,
        PROJECT_ROOT / "artifacts" / "baseline_v1",
        PROJECT_ROOT / "artifacts" / "shap_pilot_v1",
        PROJECT_ROOT / "data" / "processed" / "baseline_k10",
        SHAP_DATA_DIR,
    )
    paths = {
        path
        for root in roots
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
    for path in OUTPUT_DIR.iterdir():
        if path.is_file() and path not in OUTPUT_PATHS:
            paths.add(path)
    return tuple(sorted(paths, key=_relative))


def _hash_paths(paths: Iterable[Path]) -> dict[Path, str]:
    """Hash protected files for before/after integrity checks."""
    return {path: _sha256(path) for path in paths}


def _validate_source_guardrails() -> None:
    """Statically reject operations outside the Part 3 analysis scope."""
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_modules = {"shap", "joblib", "sklearn"}
    forbidden_calls = {
        "fit",
        "fit_transform",
        "shap_values",
        "TreeExplainer",
        "Explainer",
        "choice",
        "sample",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".")[0] for alias in node.names}
            _require(not roots & forbidden_modules, "Forbidden Part 3 import found")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            _require(root not in forbidden_modules, "Forbidden Part 3 import found")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            else:
                name = ""
            _require(name not in forbidden_calls, f"Forbidden Part 3 call: {name}")


def _validate_part1_hash_contract(window_summary: dict[str, Any]) -> None:
    """Require current Part 1 files and its recorded inputs to remain frozen."""
    _require(
        window_summary.get("experiment") == "temporal_evaluation_v1"
        and window_summary.get("stage") == "part_1_window_freeze",
        "Unexpected Part 1 temporal summary identity",
    )
    _require(window_summary.get("assertion_status") == "PASS", "Part 1 failed")
    generated = window_summary.get("generated_artifact_sha256", {})
    expected_generated = {
        _relative(TEMPORAL_WINDOWS_PATH): _sha256(TEMPORAL_WINDOWS_PATH),
        _relative(TEMPORAL_SHAP_WINDOWS_PATH): _sha256(TEMPORAL_SHAP_WINDOWS_PATH),
    }
    _require(generated == expected_generated, "Frozen Part 1 output hash changed")
    protected = window_summary.get("protected_artifacts", {}).get("sha256", {})
    _require(isinstance(protected, dict) and protected, "Part 1 hashes missing")
    for relative, expected in protected.items():
        path = PROJECT_ROOT / relative
        _require(_sha256(path) == expected, f"Frozen input changed: {relative}")


def load_frozen_shap_matrix(
    path: Path = SHAP_VALUES_PATH,
) -> np.ndarray:
    """Load the already-computed positive-class SHAP matrix only."""
    values = np.load(path, allow_pickle=False)
    _require(values.shape == EXPECTED_SHAPE, f"Unexpected SHAP shape: {values.shape}")
    _require(np.isfinite(values).all(), "SHAP matrix contains NaN or infinity")
    _require(not np.isnan(values).any(), "SHAP matrix contains NaN")
    _require(not np.isposinf(values).any(), "SHAP matrix contains +infinity")
    _require(not np.isneginf(values).any(), "SHAP matrix contains -infinity")
    return values


def validate_shap_identity(
    values: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate canonical SHAP identity, feature order, and frozen metadata."""
    manifest = _load_json(FULL_MANIFEST_PATH)
    _require(
        manifest.get("experiment") == "shap_pilot_v1"
        and manifest.get("positive_class") == POSITIVE_CLASS,
        "Unexpected SHAP Pilot manifest identity",
    )
    _require(manifest.get("full_shap_shape") == [1000, 165], "Manifest shape changed")
    _require(manifest.get("explained_cases") == EXPECTED_CASES, "Case count changed")
    _require(manifest.get("feature_count") == EXPECTED_FEATURES, "Feature count changed")
    _require(manifest.get("resampled") is False, "Manifest reports resampling")
    hashes = manifest.get("canonical_artifact_hashes", {})
    file_checks = {
        "shap_values_success_npy_sha256": SHAP_VALUES_PATH,
        "base_values_success_npy_sha256": BASE_VALUES_PATH,
        "model_probabilities_success_npy_sha256": PROBABILITIES_PATH,
        "full_shap_case_metadata_csv_sha256": FULL_METADATA_PATH,
    }
    for key, path in file_checks.items():
        _require(hashes.get(key) == _sha256(path), f"Canonical hash mismatch: {key}")
    _require(
        hashes.get("shap_values_success_array_sha256") == _array_sha256(values),
        "Canonical in-memory SHAP matrix hash differs from manifest",
    )

    features = pd.read_csv(FEATURE_NAMES_PATH, encoding="utf-8")
    _require(
        features.columns.tolist()
        == ["feature_index", "feature_name", "source_group"],
        "Unexpected feature metadata columns",
    )
    _require(len(features) == EXPECTED_FEATURES, "Feature count is not 165")
    _require(
        features["feature_index"].tolist() == list(range(EXPECTED_FEATURES)),
        "Feature indices are not ordered 0 through 164",
    )
    _require(features["feature_name"].notna().all(), "Feature name is missing")
    _require(features["feature_name"].is_unique, "Feature names are not unique")

    metadata = pd.read_csv(
        FULL_METADATA_PATH,
        encoding="utf-8",
        dtype={"case_id": "string"},
    )
    _require(len(metadata) == EXPECTED_CASES, "SHAP metadata is not 1,000 rows")
    _require(
        metadata["sample_order"].tolist() == list(range(EXPECTED_CASES)),
        "SHAP metadata sample order changed",
    )
    _require(metadata["case_id"].notna().all(), "SHAP metadata case ID missing")
    _require(metadata["case_id"].is_unique, "SHAP metadata case ID duplicated")
    return features, metadata, manifest


def align_temporal_membership(
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate row identity and retain sample_order as the SHAP row index."""
    window_summary = _load_json(WINDOW_SUMMARY_PATH)
    _validate_part1_hash_contract(window_summary)
    membership = pd.read_csv(
        TEMPORAL_SHAP_WINDOWS_PATH,
        encoding="utf-8",
        dtype={"case_id": "string"},
    )
    expected_columns = [
        "sample_order",
        "source_row_index",
        "case_id",
        "target",
        "case_start_time",
        "window",
    ]
    _require(membership.columns.tolist() == expected_columns, "Membership columns changed")
    _require(len(membership) == EXPECTED_CASES, "Membership is not 1,000 rows")
    _require(membership["sample_order"].is_unique, "sample_order is duplicated")
    _require(
        set(membership["sample_order"]) == set(range(EXPECTED_CASES)),
        "sample_order does not cover 0 through 999",
    )
    _require(
        membership["sample_order"].tolist() == list(range(EXPECTED_CASES)),
        "Frozen sample order is not preserved",
    )
    _require(membership["case_id"].is_unique, "A SHAP case is duplicated")
    _require(membership["case_id"].notna().all(), "A SHAP case is missing")
    _require(set(membership["window"]) == set(WINDOWS), "Unexpected window label")
    identity_columns = ["sample_order", "source_row_index", "case_id", "target"]
    pd.testing.assert_frame_equal(
        membership[identity_columns],
        metadata[identity_columns],
        check_dtype=False,
        check_exact=True,
    )

    full_windows = pd.read_csv(
        TEMPORAL_WINDOWS_PATH,
        encoding="utf-8",
        dtype={"case_id": "string"},
    )
    canonical = full_windows.set_index("case_id")[["target", "window"]]
    aligned = canonical.loc[membership["case_id"]].reset_index(drop=True)
    _require(
        aligned["target"].tolist() == membership["target"].tolist(),
        "Frozen target values changed",
    )
    _require(
        aligned["window"].tolist() == membership["window"].tolist(),
        "Frozen window memberships changed",
    )

    counts = {
        window: int((membership["window"] == window).sum()) for window in WINDOWS
    }
    _require(counts == WINDOW_COUNTS, f"Unexpected window counts: {counts}")
    target_counts: dict[str, dict[str, int]] = {}
    for window in WINDOWS:
        observed = (
            membership.loc[membership["window"].eq(window), "target"]
            .value_counts()
            .sort_index()
            .to_dict()
        )
        observed = {int(key): int(value) for key, value in observed.items()}
        _require(observed == WINDOW_TARGET_COUNTS[window], f"{window} targets changed")
        target_counts[window] = {str(key): value for key, value in observed.items()}
    return membership, {"window_counts": counts, "target_counts": target_counts}


def compute_window_importance(
    values: np.ndarray,
    membership: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Compute mean absolute and signed SHAP values for every window/feature."""
    records: list[pd.DataFrame] = []
    names = features["feature_name"].to_numpy()
    for window in WINDOWS:
        orders = membership.loc[
            membership["window"].eq(window), "sample_order"
        ].to_numpy(dtype=np.int64)
        window_values = values[orders, :]
        _require(window_values.shape == (WINDOW_COUNTS[window], EXPECTED_FEATURES),
                 f"{window} SHAP alignment failed")
        absolute = np.abs(window_values)
        frame = pd.DataFrame(
            {
                "window": window,
                "feature": names,
                "mean_abs_shap": absolute.mean(axis=0),
                "mean_signed_shap": window_values.mean(axis=0),
                "std_abs_shap": absolute.std(axis=0, ddof=0),
            }
        )
        records.append(frame)
    importance = pd.concat(records, ignore_index=True)
    _require(np.isfinite(importance.select_dtypes("number")).all().all(),
             "Temporal importance contains non-finite values")
    return importance


def rank_window_features(importance: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen deterministic ranking rule within every window."""
    ranked: list[pd.DataFrame] = []
    for window in WINDOWS:
        frame = importance.loc[importance["window"].eq(window)].sort_values(
            ["mean_abs_shap", "feature"],
            ascending=[False, True],
            kind="mergesort",
            ignore_index=True,
        )
        frame["rank"] = np.arange(1, EXPECTED_FEATURES + 1, dtype=np.int64)
        ranked.append(frame)
    output = pd.concat(ranked, ignore_index=True)
    return output[
        [
            "window",
            "feature",
            "mean_abs_shap",
            "mean_signed_shap",
            "std_abs_shap",
            "rank",
        ]
    ]


def extract_top10(ranked: pd.DataFrame) -> pd.DataFrame:
    """Extract the fixed top ten from each chronological window."""
    top = ranked.loc[ranked["rank"].le(TOP_K)].copy()
    top["window"] = pd.Categorical(top["window"], WINDOWS, ordered=True)
    top = top.sort_values(["window", "rank"], kind="mergesort", ignore_index=True)
    top["window"] = top["window"].astype("string")
    return top[["window", "rank", "feature", "mean_abs_shap", "mean_signed_shap"]]


def compute_jaccard(top10: pd.DataFrame, window_a: str, window_b: str) -> dict[str, Any]:
    """Compute top-set overlap without assigning qualitative thresholds."""
    set_a = set(top10.loc[top10["window"].eq(window_a), "feature"])
    set_b = set(top10.loc[top10["window"].eq(window_b), "feature"])
    _require(len(set_a) == TOP_K and len(set_b) == TOP_K, "Top-10 set is invalid")
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return {
        "intersection_count": intersection,
        "union_count": union,
        "jaccard_at_10": intersection / union,
    }


def compute_spearman(
    ranked: pd.DataFrame,
    window_a: str,
    window_b: str,
) -> dict[str, float]:
    """Correlate all 165 aligned feature-importance values."""
    values_by_window: dict[str, pd.Series] = {}
    for window in (window_a, window_b):
        series = ranked.loc[
            ranked["window"].eq(window), ["feature", "mean_abs_shap"]
        ].set_index("feature")["mean_abs_shap"]
        _require(series.index.is_unique and len(series) == EXPECTED_FEATURES,
                 f"{window} importance vector is invalid")
        values_by_window[window] = series.sort_index()
    _require(
        values_by_window[window_a].index.equals(values_by_window[window_b].index),
        "Spearman features were not aligned by name",
    )
    result = spearmanr(
        values_by_window[window_a].to_numpy(),
        values_by_window[window_b].to_numpy(),
    )
    rho = float(result.statistic)
    pvalue = float(result.pvalue)
    _require(np.isfinite(rho) and -1.0 <= rho <= 1.0, "Invalid Spearman rho")
    _require(np.isfinite(pvalue), "Invalid descriptive Spearman p-value")
    return {"spearman_rho": rho, "spearman_pvalue": pvalue}


def compute_stability(ranked: pd.DataFrame, top10: pd.DataFrame) -> pd.DataFrame:
    """Keep Jaccard top-set and Spearman full-ranking semantics separate."""
    rows: list[dict[str, Any]] = []
    for comparison, window_a, window_b in COMPARISONS:
        row: dict[str, Any] = {
            "comparison": comparison,
            "window_a": window_a,
            "window_b": window_b,
            "top_k": TOP_K,
        }
        row.update(compute_jaccard(top10, window_a, window_b))
        row.update(compute_spearman(ranked, window_a, window_b))
        rows.append(row)
    return pd.DataFrame(rows)


def _sign(value: float) -> str:
    """Classify a signed mean with the documented absolute zero tolerance."""
    if value > ZERO_TOLERANCE:
        return "positive"
    if value < -ZERO_TOLERANCE:
        return "negative"
    return "zero"


def _union_order(ranked: pd.DataFrame, top10: pd.DataFrame) -> list[str]:
    """Order union features by maximum temporal importance, then name."""
    union = set(top10["feature"])
    maxima = (
        ranked.loc[ranked["feature"].isin(union)]
        .groupby("feature", sort=False)["mean_abs_shap"]
        .max()
        .reset_index()
        .sort_values(
            ["mean_abs_shap", "feature"],
            ascending=[False, True],
            kind="mergesort",
        )
    )
    return maxima["feature"].tolist()


def compute_signed_shap_summary(
    ranked: pd.DataFrame,
    top10: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize mean signed SHAP only for the union of temporal top tens."""
    order = _union_order(ranked, top10)
    signed = ranked.pivot(index="feature", columns="window", values="mean_signed_shap")
    signed = signed.loc[order, list(WINDOWS)]
    output = signed.reset_index().rename(
        columns={window: f"{window}_mean_signed_shap" for window in WINDOWS}
    )
    for window in WINDOWS:
        output[f"{window}_sign"] = output[f"{window}_mean_signed_shap"].map(_sign)
    for _, window_a, window_b in COMPARISONS:
        output[f"sign_change_{window_a}_{window_b}"] = (
            output[f"{window_a}_sign"] != output[f"{window_b}_sign"]
        )
    output.columns.name = None
    return output


def map_feature_groups(features: pd.DataFrame) -> pd.DataFrame:
    """Reuse the exact SHAP Pilot V1 source-group mapping."""
    unknown = sorted(set(features["source_group"]) - set(SOURCE_GROUP_MAP))
    _require(not unknown, f"Unmapped or ambiguous feature source groups: {unknown}")
    mapped = features.copy()
    mapped["group"] = mapped["source_group"].map(SOURCE_GROUP_MAP)
    _require(mapped["group"].notna().all(), "A feature has no group")
    _require(len(mapped) == EXPECTED_FEATURES, "Mapped feature count is not 165")
    _require(mapped["feature_name"].is_unique, "Mapped features are duplicated")
    counts = mapped["group"].value_counts().to_dict()
    _require(counts == EXPECTED_GROUP_COUNTS, f"Group counts changed: {counts}")

    prefixes = {
        "categorical::loan_goal": "loan_goal_",
        "categorical::application_type": "application_type_",
        "structured_map::activity": "activity::",
        "structured_map::action": "action::",
        "structured_map::origin": "origin::",
        "structured_map::resource": "resource::",
        "structured_map::lifecycle": "lifecycle::",
    }
    for source_group, prefix in prefixes.items():
        names = mapped.loc[mapped["source_group"].eq(source_group), "feature_name"]
        _require(not names.empty and names.str.startswith(prefix).all(),
                 f"Feature naming convention changed for {source_group}")
    return mapped


def _validate_group_mapping_against_pilot(
    values: np.ndarray,
    mapped: pd.DataFrame,
) -> None:
    """Reproduce frozen Pilot group totals to audit the exact mapping."""
    overall = pd.DataFrame(
        {
            "feature_name": mapped["feature_name"],
            "group": mapped["group"],
            "mean_abs_shap": np.abs(values).mean(axis=0),
        }
    )
    total = float(overall["mean_abs_shap"].sum())
    reproduced = (
        overall.groupby("group", sort=False)
        .agg(
            feature_count=("feature_name", "size"),
            group_total_importance=("mean_abs_shap", "sum"),
        )
        .reindex(GROUP_ORDER)
        .reset_index()
    )
    reproduced["group_mean_importance_per_feature"] = (
        reproduced["group_total_importance"] / reproduced["feature_count"]
    )
    reproduced["share_of_total_importance"] = (
        reproduced["group_total_importance"] / total
    )
    frozen = pd.read_csv(PILOT_GROUP_IMPORTANCE_PATH, encoding="utf-8")
    pd.testing.assert_frame_equal(
        reproduced,
        frozen,
        check_dtype=False,
        check_exact=False,
        rtol=0.0,
        atol=1e-15,
    )


def compute_group_importance(
    ranked: pd.DataFrame,
    mapped: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate window importance in the fixed conceptual group order."""
    feature_groups = mapped.set_index("feature_name")["group"]
    rows: list[pd.DataFrame] = []
    for window in WINDOWS:
        frame = ranked.loc[
            ranked["window"].eq(window), ["feature", "mean_abs_shap"]
        ].copy()
        frame["group"] = frame["feature"].map(feature_groups)
        _require(frame["group"].notna().all(), f"{window} group mapping failed")
        total = float(frame["mean_abs_shap"].sum())
        grouped = (
            frame.groupby("group", sort=False)
            .agg(
                feature_count=("feature", "size"),
                group_total_importance=("mean_abs_shap", "sum"),
            )
            .reindex(GROUP_ORDER)
            .reset_index()
        )
        grouped.insert(0, "window", window)
        grouped["feature_count"] = grouped["feature_count"].astype(np.int64)
        grouped["group_mean_importance_per_feature"] = (
            grouped["group_total_importance"] / grouped["feature_count"]
        )
        grouped["share_of_total_importance"] = (
            grouped["group_total_importance"] / total
        )
        _require(grouped["feature_count"].sum() == EXPECTED_FEATURES,
                 f"{window} grouped feature count is not 165")
        _require(np.isclose(grouped["group_total_importance"].sum(), total,
                            rtol=0.0, atol=1e-15),
                 f"{window} group total differs from feature total")
        _require(np.isclose(grouped["share_of_total_importance"].sum(), 1.0,
                            rtol=0.0, atol=1e-12),
                 f"{window} group shares do not sum to one")
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def _display_feature(feature: str) -> str:
    """Make encoded names compact while retaining their exact meaning."""
    return feature.replace("::", " · ").replace("_", " ")


def _plot_matrix(ranked: pd.DataFrame, top10: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Return deterministic union order and feature-by-window importance matrix."""
    order = _union_order(ranked, top10)
    pivot = ranked.pivot(index="feature", columns="window", values="mean_abs_shap")
    matrix = pivot.loc[order, list(WINDOWS)].to_numpy(dtype=np.float64)
    _require(np.isfinite(matrix).all(), "Plot matrix contains non-finite values")
    return order, matrix


def create_top_feature_plot(ranked: pd.DataFrame, top10: pd.DataFrame) -> None:
    """Plot temporal mean absolute SHAP for the union of top-ten features."""
    order, matrix = _plot_matrix(ranked, top10)
    positions = np.arange(len(order), dtype=np.float64)
    height = max(7.0, 0.43 * len(order) + 2.2)
    figure, axis = plt.subplots(figsize=(11.5, height))
    offsets = (-0.25, 0.0, 0.25)
    colors = ("#4472C4", "#ED7D31", "#70AD47")
    for index, (window, offset, color) in enumerate(zip(WINDOWS, offsets, colors, strict=True)):
        axis.barh(
            positions + offset,
            matrix[:, index],
            height=0.23,
            label=window,
            color=color,
            alpha=0.92,
        )
    axis.set_yticks(positions, [_display_feature(name) for name in order])
    axis.invert_yaxis()
    axis.set_xlabel("Mean |SHAP| (Success = 1)")
    axis.set_title("Frozen SHAP importance across temporal windows\nUnion of top-10 features")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(title="Window", ncols=3, loc="lower right")
    figure.tight_layout()
    _save_figure_atomic(TOP_FEATURES_PLOT_PATH, figure)


def create_heatmap(ranked: pd.DataFrame, top10: pd.DataFrame) -> None:
    """Create an unclustered chronological heatmap for union features."""
    order, matrix = _plot_matrix(ranked, top10)
    height = max(6.5, 0.42 * len(order) + 2.0)
    figure, axis = plt.subplots(figsize=(7.5, height))
    image = axis.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    axis.set_xticks(np.arange(len(WINDOWS)), WINDOWS)
    axis.set_yticks(np.arange(len(order)), [_display_feature(name) for name in order])
    axis.set_xlabel("Temporal window")
    axis.set_title("Mean |SHAP| heatmap (Success = 1)\nUnion of top-10 features")
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Mean |SHAP|")
    figure.tight_layout()
    _save_figure_atomic(HEATMAP_PATH, figure)


def create_group_plot(group_importance: pd.DataFrame) -> None:
    """Plot temporal group shares and document encoded-dimension influence."""
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, len(GROUP_ORDER)))
    figure, axis = plt.subplots(figsize=(11.5, 7.0))
    x = np.arange(len(WINDOWS))
    for color, group in zip(colors, GROUP_ORDER, strict=True):
        frame = group_importance.loc[group_importance["group"].eq(group)]
        frame = frame.set_index("window").loc[list(WINDOWS)]
        count = int(frame["feature_count"].iloc[0])
        axis.plot(
            x,
            frame["share_of_total_importance"].to_numpy(),
            marker="o",
            linewidth=2,
            label=f"{group} (n={count})",
            color=color,
        )
    axis.set_xticks(x, WINDOWS)
    axis.set_ylabel("Share of total mean |SHAP|")
    axis.set_xlabel("Temporal window")
    axis.set_title("Temporal feature-group SHAP importance")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title="Group")
    figure.text(
        0.5,
        0.015,
        "Group totals depend on encoded dimension count; interpret shares with feature_count and mean per feature.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.04, 0.82, 1.0))
    _save_figure_atomic(GROUP_PLOT_PATH, figure)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a frame to JSON-safe native scalar records."""
    return json.loads(frame.to_json(orient="records", double_precision=15))


def _build_assertions() -> dict[str, str]:
    """Record the required A--AF assertions after their checks pass."""
    labels = {
        "A": "shap_matrix_shape_1000_by_165",
        "B": "feature_count_165",
        "C": "feature_names_unique",
        "D": "shap_matrix_hash_matches_manifest",
        "E": "temporal_shap_windows_exactly_1000_rows",
        "F": "T1_exactly_325_cases",
        "G": "T2_exactly_332_cases",
        "H": "T3_exactly_343_cases",
        "I": "sample_order_unique",
        "J": "sample_order_covers_0_through_999",
        "K": "no_shap_case_added",
        "L": "no_shap_case_removed",
        "M": "no_shap_case_resampled",
        "N": "every_importance_window_has_165_features",
        "O": "every_window_has_ranks_1_through_165",
        "P": "top10_output_has_30_rows",
        "Q": "every_window_has_10_top_features",
        "R": "jaccard_within_zero_and_one",
        "S": "jaccard_counts_internally_consistent",
        "T": "spearman_finite_and_within_bounds",
        "U": "spearman_features_aligned_by_name",
        "V": "signed_analysis_uses_only_union_top10",
        "W": "all_165_features_map_to_one_group",
        "X": "group_feature_counts_sum_to_165",
        "Y": "group_shares_sum_to_one",
        "Z": "no_shap_computation_occurred",
        "AA": "no_explainer_created",
        "AB": "no_model_fitting_occurred",
        "AC": "no_preprocessing_fitting_occurred",
        "AD": "no_temporal_boundaries_changed",
        "AE": "no_prediction_metric_recomputed_for_synthesis",
        "AF": "no_formal_drift_test_occurred",
    }
    return {f"{key}_{value}": "PASS" for key, value in labels.items()}


def validate_outputs(
    ranked: pd.DataFrame,
    top10: pd.DataFrame,
    stability: pd.DataFrame,
    signed: pd.DataFrame,
    group_importance: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    """Validate every saved output and all required structural assertions."""
    _require(len(ranked) == 3 * EXPECTED_FEATURES, "Importance row count is not 495")
    for window in WINDOWS:
        frame = ranked.loc[ranked["window"].eq(window)]
        _require(len(frame) == EXPECTED_FEATURES, f"{window} importance rows invalid")
        _require(frame["rank"].tolist() == list(range(1, EXPECTED_FEATURES + 1)),
                 f"{window} ranks are invalid")
    _require(len(top10) == 3 * TOP_K, "Top-10 output does not have 30 rows")
    _require(top10.groupby("window", observed=True).size().to_dict()
             == {window: TOP_K for window in WINDOWS}, "Top-10 window counts invalid")
    for row in stability.itertuples(index=False):
        _require(0.0 <= row.jaccard_at_10 <= 1.0, "Jaccard outside [0, 1]")
        _require(row.union_count == 2 * TOP_K - row.intersection_count,
                 "Jaccard union/intersection counts inconsistent")
        _require(np.isclose(row.jaccard_at_10,
                            row.intersection_count / row.union_count,
                            rtol=0.0, atol=1e-15), "Jaccard value inconsistent")
        _require(np.isfinite(row.spearman_rho) and -1.0 <= row.spearman_rho <= 1.0,
                 "Spearman rho invalid")
    union = set(top10["feature"])
    _require(set(signed["feature"]) == union and len(signed) == len(union),
             "Signed SHAP scope differs from top-10 union")
    _require(len(group_importance) == len(WINDOWS) * len(GROUP_ORDER),
             "Group output row count is not 24")
    for window in WINDOWS:
        frame = group_importance.loc[group_importance["window"].eq(window)]
        _require(frame["group"].tolist() == list(GROUP_ORDER),
                 f"{window} group order changed")
        _require(frame["feature_count"].sum() == EXPECTED_FEATURES,
                 f"{window} group count invalid")
        _require(np.isclose(frame["share_of_total_importance"].sum(), 1.0,
                            rtol=0.0, atol=1e-12), f"{window} group shares invalid")

    csv_expectations = {
        IMPORTANCE_PATH: ranked,
        TOP10_PATH: top10,
        STABILITY_PATH: stability,
        SIGNED_PATH: signed,
        GROUP_IMPORTANCE_PATH: group_importance,
    }
    for path, expected in csv_expectations.items():
        observed = pd.read_csv(path, encoding="utf-8")
        pd.testing.assert_frame_equal(
            observed,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-15,
        )
    _require(_load_json(SUMMARY_PATH) == summary, "Saved summary JSON differs")
    for path in (TOP_FEATURES_PLOT_PATH, HEATMAP_PATH, GROUP_PLOT_PATH):
        image = mpimg.imread(path)
        _require(image.ndim in (2, 3) and image.shape[0] > 0 and image.shape[1] > 0,
                 f"Invalid generated plot: {_relative(path)}")


def write_summary(
    membership_audit: dict[str, Any],
    manifest: dict[str, Any],
    top10: pd.DataFrame,
    stability: pd.DataFrame,
    signed: pd.DataFrame,
    group_importance: pd.DataFrame,
    assertions: dict[str, str],
    protected_hashes: dict[Path, str],
    generated_hashes: dict[str, str],
) -> dict[str, Any]:
    """Build and persist the measured Part 3 summary."""
    top_lists = {
        window: _records(
            top10.loc[top10["window"].eq(window),
                      ["rank", "feature", "mean_abs_shap", "mean_signed_shap"]]
        )
        for window in WINDOWS
    }
    sign_columns = [
        "sign_change_T1_T2",
        "sign_change_T2_T3",
        "sign_change_T1_T3",
    ]
    sign_change_counts = {
        column.removeprefix("sign_change_"): int(signed[column].sum())
        for column in sign_columns
    }
    sign_change_features = {
        column.removeprefix("sign_change_"): signed.loc[signed[column], "feature"].tolist()
        for column in sign_columns
    }
    group_summary = {
        window: _records(
            group_importance.loc[
                group_importance["window"].eq(window),
                [
                    "group",
                    "feature_count",
                    "group_total_importance",
                    "group_mean_importance_per_feature",
                    "share_of_total_importance",
                ],
            ]
        )
        for window in WINDOWS
    }
    summary: dict[str, Any] = {
        "experiment": "temporal_evaluation_v1",
        "stage": "part_3_temporal_shap_analysis",
        "positive_class": POSITIVE_CLASS,
        "positive_class_name": "Success",
        "shap_cases": EXPECTED_CASES,
        "feature_count": EXPECTED_FEATURES,
        "shap_matrix_shape": list(EXPECTED_SHAPE),
        "canonical_shap_file_sha256": manifest["canonical_artifact_hashes"][
            "shap_values_success_npy_sha256"
        ],
        "window_counts": membership_audit["window_counts"],
        "window_target_counts": membership_audit["target_counts"],
        "top_k": TOP_K,
        "per_window_top10": top_lists,
        "pairwise_jaccard_at_10": _records(
            stability[
                [
                    "comparison",
                    "window_a",
                    "window_b",
                    "intersection_count",
                    "union_count",
                    "jaccard_at_10",
                ]
            ]
        ),
        "pairwise_spearman_all_165_features": _records(
            stability[
                [
                    "comparison",
                    "window_a",
                    "window_b",
                    "spearman_rho",
                    "spearman_pvalue",
                ]
            ]
        ),
        "metric_semantics": {
            "jaccard_at_10": "top-10 feature-set overlap",
            "spearman_rho": "overall ordering similarity across all 165 name-aligned features",
            "combined_score_created": False,
        },
        "signed_shap_union_feature_count": len(signed),
        "signed_shap_zero_tolerance": ZERO_TOLERANCE,
        "sign_change_counts": sign_change_counts,
        "sign_change_features": sign_change_features,
        "signed_shap_interpretation": (
            "A sign change describes a change in the frozen model's average "
            "contribution direction; it is not a causal effect."
        ),
        "group_importance_summary": group_summary,
        "group_total_interpretation": (
            "Group totals and shares are influenced by encoded feature_count; "
            "Resource has 115 dimensions, so use feature_count and mean per feature too."
        ),
        "shap_recomputed": False,
        "model_retrained": False,
        "preprocessing_refitted": False,
        "shap_sample_resampled": False,
        "temporal_boundaries_changed": False,
        "prediction_metrics_recomputed_for_synthesis": False,
        "prediction_explanation_synthesis_created": False,
        "formal_drift_detection": False,
        "significance_testing_used_for_conclusion": False,
        "spearman_pvalue_use": "stored for completeness only",
        "assertions": assertions,
        "assertion_status": "PASS",
        "protected_artifacts": {
            "count": len(protected_hashes),
            "unchanged": True,
            "sha256": {
                _relative(path): digest for path, digest in protected_hashes.items()
            },
        },
        "generated_artifact_sha256_excluding_summary": generated_hashes,
    }
    _write_json_atomic(SUMMARY_PATH, summary)
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    """Print concise measured results plus every generated hash."""
    print("=== TEMPORAL EVALUATION V1 PART 3 ===")
    print(f"SHAP matrix: {summary['shap_matrix_shape']} PASS")
    print(f"Canonical SHAP hash: {summary['canonical_shap_file_sha256']}")
    for window in WINDOWS:
        counts = summary["window_target_counts"][window]
        print(
            f"{window}: n={summary['window_counts'][window]}, "
            f"Success={counts['1']}, Unsuccessful={counts['0']}"
        )
        for row in summary["per_window_top10"][window]:
            print(f"  {row['rank']:2d}. {row['feature']}: {row['mean_abs_shap']:.12f}")
    for row in summary["pairwise_jaccard_at_10"]:
        print(f"{row['comparison']} Jaccard@10: {row['jaccard_at_10']:.12f}")
    for row in summary["pairwise_spearman_all_165_features"]:
        print(f"{row['comparison']} Spearman: {row['spearman_rho']:.12f}")
    print(f"Signed union features: {summary['signed_shap_union_feature_count']}")
    print(f"Sign-change counts: {summary['sign_change_counts']}")
    print(f"Assertions: {summary['assertion_status']}")
    print(f"Protected artifacts unchanged: {summary['protected_artifacts']['unchanged']}")
    print("Generated artifact hashes:")
    hashes = dict(summary["generated_artifact_sha256_excluding_summary"])
    hashes[_relative(SUMMARY_PATH)] = _sha256(SUMMARY_PATH)
    for path, digest in hashes.items():
        print(f"  {path}: {digest}")


def main() -> None:
    """Build, validate, and freeze Temporal Evaluation V1 Part 3 only."""
    _validate_source_guardrails()
    protected_paths = _protected_paths()
    protected_before = _hash_paths(protected_paths)

    values = load_frozen_shap_matrix()
    features, metadata, manifest = validate_shap_identity(values)
    membership, membership_audit = align_temporal_membership(metadata)
    importance = compute_window_importance(values, membership, features)
    ranked = rank_window_features(importance)
    top10 = extract_top10(ranked)
    stability = compute_stability(ranked, top10)
    signed = compute_signed_shap_summary(ranked, top10)
    mapped = map_feature_groups(features)
    _validate_group_mapping_against_pilot(values, mapped)
    group_importance = compute_group_importance(ranked, mapped)

    validate_outputs_in_memory = {
        window: len(ranked.loc[ranked["window"].eq(window)]) for window in WINDOWS
    }
    _require(validate_outputs_in_memory == {window: EXPECTED_FEATURES for window in WINDOWS},
             "A temporal importance window does not contain 165 rows")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(IMPORTANCE_PATH, ranked)
    _write_csv_atomic(TOP10_PATH, top10)
    _write_csv_atomic(STABILITY_PATH, stability)
    _write_csv_atomic(SIGNED_PATH, signed)
    _write_csv_atomic(GROUP_IMPORTANCE_PATH, group_importance)
    create_top_feature_plot(ranked, top10)
    create_heatmap(ranked, top10)
    create_group_plot(group_importance)

    protected_after_outputs = _hash_paths(protected_paths)
    _require(protected_after_outputs == protected_before,
             "A protected artifact changed during Part 3")
    assertions = _build_assertions()
    generated_hashes = {
        _relative(path): _sha256(path)
        for path in OUTPUT_PATHS
        if path != SUMMARY_PATH
    }
    summary = write_summary(
        membership_audit,
        manifest,
        top10,
        stability,
        signed,
        group_importance,
        assertions,
        protected_after_outputs,
        generated_hashes,
    )
    validate_outputs(ranked, top10, stability, signed, group_importance, summary)

    protected_final = _hash_paths(protected_paths)
    _require(protected_final == protected_before,
             "A protected artifact changed during final validation")
    _print_summary(summary)


if __name__ == "__main__":
    main()
