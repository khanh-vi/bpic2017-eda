"""Build SHAP Pilot V1 Part 4 reports from frozen Part 3 outputs only.

This reporting layer validates and consumes the canonical stored SHAP arrays.
It never loads the model, computes SHAP values, fits preprocessing, resamples,
or performs temporal or explanation-stability analysis.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "shap_pilot_v1"
BASELINE_RESULTS_DIR = PROJECT_ROOT / "results" / "baseline_v1"
SHAP_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "shap_pilot_v1"
BASELINE_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "baseline_k10"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "shap_pilot_v1" / "random_forest_baseline.joblib"

SHAP_VALUES_PATH = SHAP_DATA_DIR / "shap_values_success.npy"
BASE_VALUES_PATH = SHAP_DATA_DIR / "base_values_success.npy"
PROBABILITIES_PATH = SHAP_DATA_DIR / "model_probabilities_success.npy"
FULL_METADATA_PATH = RESULTS_DIR / "full_shap_case_metadata.csv"
FULL_MANIFEST_PATH = RESULTS_DIR / "full_shap_manifest.json"
EXPLAINED_IDS_PATH = RESULTS_DIR / "explained_case_ids.csv"
BACKGROUND_IDS_PATH = RESULTS_DIR / "background_case_ids.csv"
SAMPLE_MANIFEST_PATH = RESULTS_DIR / "sample_manifest.json"
MODEL_MANIFEST_PATH = RESULTS_DIR / "model_freeze_manifest.json"
CORE_MANIFEST_PATH = RESULTS_DIR / "shap_core_manifest.json"
CORE_RESOLUTION_PATH = RESULTS_DIR / "shap_core_resolution.json"

FEATURE_NAMES_PATH = BASELINE_RESULTS_DIR / "feature_names_tree.csv"
PREDICTIONS_PATH = BASELINE_RESULTS_DIR / "random_forest_predictions.csv"
PREPROCESSING_MANIFEST_PATH = BASELINE_RESULTS_DIR / "preprocessing_manifest.json"
FEATURE_MANIFEST_PATH = BASELINE_RESULTS_DIR / "feature_manifest.json"
TEST_IDS_PATH = BASELINE_RESULTS_DIR / "test_case_ids.csv"
X_TEST_PATH = BASELINE_DATA_DIR / "X_test_tree.npz"

GLOBAL_IMPORTANCE_PATH = RESULTS_DIR / "global_shap_importance.csv"
GROUP_IMPORTANCE_PATH = RESULTS_DIR / "global_shap_group_importance.csv"
GLOBAL_BAR_PATH = RESULTS_DIR / "shap_global_bar.png"
BEESWARM_PATH = RESULTS_DIR / "shap_beeswarm.png"
LOCAL_EXAMPLES_PATH = RESULTS_DIR / "local_examples.csv"
LOCAL_CONTRIBUTIONS_PATH = RESULTS_DIR / "local_feature_contributions.csv"
REPORT_PATH = RESULTS_DIR / "shap_pilot_v1.md"
REPORT_MANIFEST_PATH = RESULTS_DIR / "report_manifest.json"

EXPECTED_SHAP_VERSION = "0.52.0"
EXPECTED_SHAPE = (1_000, 165)
EXPECTED_FEATURES = 165
EXPECTED_CASES = 1_000
POSITIVE_CLASS = 1
ADDITIVITY_TOLERANCE = 1e-5
PROBABILITY_TOLERANCE = 5e-15
PLOT_SEED = 42
TOP_FEATURES = 20
LOCAL_DISPLAY_FEATURES = 15

KNOWN_CANONICAL_HASHES = {
    SHAP_VALUES_PATH: "a01994419b95950b5c1b6446c841d82d0715669ac0ff17e4b6fad9bfdf79da03",
    BASE_VALUES_PATH: "b833d9c5629af02f4ee2fb52c3338e15722526204238cca59e1ddc06285200fb",
    PROBABILITIES_PATH: "8cd77ce1b77fbaed8a37d5ed63e328f0fb5e6ed9c11fac98db019d47729f0c6c",
    FULL_METADATA_PATH: "4787f1696a0628514cf3c15da616f5da34c44312c86b0214f6a3abd54aee67e1",
}

CANONICAL_HASH_KEYS = {
    SHAP_VALUES_PATH: "shap_values_success_npy_sha256",
    BASE_VALUES_PATH: "base_values_success_npy_sha256",
    PROBABILITIES_PATH: "model_probabilities_success_npy_sha256",
    FULL_METADATA_PATH: "full_shap_case_metadata_csv_sha256",
}

GROUP_ORDER = [
    "Numeric",
    "LoanGoal",
    "ApplicationType",
    "Activity",
    "Action",
    "EventOrigin",
    "Resource",
    "Lifecycle",
]

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

LOCAL_CATEGORY_ORDER = ["TP", "TN", "FP", "FN"]

LOCAL_PLOT_PATHS = {
    "TP": RESULTS_DIR / "local_tp_waterfall.png",
    "TN": RESULTS_DIR / "local_tn_waterfall.png",
    "FP": RESULTS_DIR / "local_fp_waterfall.png",
    "FN": RESULTS_DIR / "local_fn_waterfall.png",
}

REPRODUCIBLE_OUTPUT_PATHS = (
    GLOBAL_IMPORTANCE_PATH,
    GROUP_IMPORTANCE_PATH,
    LOCAL_EXAMPLES_PATH,
    LOCAL_CONTRIBUTIONS_PATH,
    GLOBAL_BAR_PATH,
    BEESWARM_PATH,
    *LOCAL_PLOT_PATHS.values(),
)


def _relative(path: Path) -> str:
    """Return a stable project-relative POSIX path."""
    return path.relative_to(PROJECT_ROOT).as_posix()


def _sha256(path: Path) -> str:
    """Return a file SHA-256 digest without modifying the file."""
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {_relative(path)}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    """Hash an array using the Part 3 shape/dtype/bytes convention."""
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
    if not isinstance(value, dict):
        raise AssertionError(f"Expected a JSON object in {_relative(path)}")
    return value


def _write_text_atomic(path: Path, content: str) -> None:
    """Atomically replace a deterministic UTF-8 text output."""
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
    """Write a CSV deterministically with full double precision."""
    content = frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    _write_text_atomic(path, content)


def _save_figure_atomic(path: Path, figure: plt.Figure) -> None:
    """Save a PNG through a temporary file before atomic replacement."""
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
        )
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def _existing_hashes(paths: tuple[Path, ...]) -> dict[Path, str] | None:
    """Return hashes only when every deterministic output already exists."""
    if not all(path.is_file() for path in paths):
        return None
    return {path: _sha256(path) for path in paths}


def _validate_source_guardrails() -> None:
    """Reject accidental explainer construction or fitting in this script."""
    source = Path(__file__).read_text(encoding="utf-8")
    forbidden_patterns = {
        "SHAP explainer construction": r"shap\s*\.\s*(?:Tree)?Explainer\s*\(",
        "SHAP-value method call": r"\.\s*shap_values\s*\(",
        "model or preprocessing fitting": r"\.\s*fit(?:_transform)?\s*\(",
        "random sampling": r"\.\s*(?:choice|sample)\s*\(",
    }
    for label, pattern in forbidden_patterns.items():
        if re.search(pattern, source):
            raise AssertionError(f"Forbidden Part 4 operation found: {label}")


def _protected_paths(full_manifest: dict[str, Any]) -> tuple[Path, ...]:
    """Return the frozen Part 0--3 artifacts protected during Part 4."""
    recorded = full_manifest.get("protected_artifacts", {}).get("sha256", {})
    if not isinstance(recorded, dict) or not recorded:
        raise AssertionError("Part 3 protected-artifact hash map is missing")
    paths = {PROJECT_ROOT / relative for relative in recorded}
    paths.update(
        {
            MODEL_PATH,
            BACKGROUND_IDS_PATH,
            EXPLAINED_IDS_PATH,
            MODEL_MANIFEST_PATH,
            SAMPLE_MANIFEST_PATH,
            CORE_MANIFEST_PATH,
            CORE_RESOLUTION_PATH,
            FULL_MANIFEST_PATH,
            SHAP_VALUES_PATH,
            BASE_VALUES_PATH,
            PROBABILITIES_PATH,
            FULL_METADATA_PATH,
        }
    )
    return tuple(sorted(paths, key=_relative))


def _validate_recorded_input_hashes(
    full_manifest: dict[str, Any],
    sample_manifest: dict[str, Any],
) -> None:
    """Validate relevant source files against existing frozen manifests."""
    part3_hashes = full_manifest["protected_artifacts"]["sha256"]
    for relative, expected in part3_hashes.items():
        path = PROJECT_ROOT / relative
        if _sha256(path) != expected:
            raise AssertionError(f"Frozen input hash differs: {relative}")

    part1_hashes = sample_manifest["protected_artifacts"]["sha256"]
    additionally_relevant = (
        PREDICTIONS_PATH,
        PREPROCESSING_MANIFEST_PATH,
        FEATURE_MANIFEST_PATH,
    )
    for path in additionally_relevant:
        relative = _relative(path)
        expected = part1_hashes.get(relative)
        if expected is None or _sha256(path) != expected:
            raise AssertionError(f"Relevant frozen input hash differs: {relative}")


def validate_part3_hashes(full_manifest: dict[str, Any]) -> None:
    """Require both manifest and known canonical Part 3 file hashes."""
    manifest_hashes = full_manifest.get("canonical_artifact_hashes", {})
    for path, expected in KNOWN_CANONICAL_HASHES.items():
        actual = _sha256(path)
        if actual != expected:
            raise AssertionError(
                f"Canonical Part 3 hash mismatch for {_relative(path)}: {actual}"
            )
        key = CANONICAL_HASH_KEYS[path]
        if manifest_hashes.get(key) != expected:
            raise AssertionError(f"Part 3 manifest hash mismatch for {key}")


def load_canonical_shap_outputs(
    full_manifest: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load and validate canonical Part 3 arrays and aligned metadata."""
    shap_values = np.load(SHAP_VALUES_PATH, allow_pickle=False)
    base_values = np.load(BASE_VALUES_PATH, allow_pickle=False)
    probabilities = np.load(PROBABILITIES_PATH, allow_pickle=False)
    metadata = pd.read_csv(
        FULL_METADATA_PATH,
        encoding="utf-8",
        dtype={"case_id": "string"},
    )

    if shap_values.shape != EXPECTED_SHAPE:
        raise AssertionError(f"Unexpected SHAP shape: {shap_values.shape}")
    if base_values.shape != (EXPECTED_CASES,):
        raise AssertionError(f"Unexpected base-value shape: {base_values.shape}")
    if probabilities.shape != (EXPECTED_CASES,):
        raise AssertionError(f"Unexpected probability shape: {probabilities.shape}")
    if len(metadata) != EXPECTED_CASES:
        raise AssertionError(f"Unexpected metadata rows: {len(metadata)}")
    if not np.isfinite(shap_values).all():
        raise AssertionError("Frozen SHAP matrix contains NaN or infinity")
    if not np.isfinite(base_values).all() or not np.isfinite(probabilities).all():
        raise AssertionError("Frozen base values or probabilities are non-finite")

    expected_columns = [
        "sample_order",
        "source_row_index",
        "case_id",
        "target",
        "model_probability_success",
        "reconstructed_probability_success",
        "absolute_additivity_error",
    ]
    if metadata.columns.tolist() != expected_columns:
        raise AssertionError("Unexpected Part 3 metadata columns")
    if metadata["sample_order"].tolist() != list(range(EXPECTED_CASES)):
        raise AssertionError("Part 3 metadata sample order is not canonical")
    if metadata["case_id"].isna().any() or not metadata["case_id"].is_unique:
        raise AssertionError("Part 3 metadata case IDs must be non-null and unique")

    reconstructed = base_values + shap_values.sum(axis=1)
    saved_reconstructed = metadata[
        "reconstructed_probability_success"
    ].to_numpy(dtype=np.float64)
    saved_probabilities = metadata["model_probability_success"].to_numpy(
        dtype=np.float64
    )
    saved_errors = metadata["absolute_additivity_error"].to_numpy(dtype=np.float64)
    errors = np.abs(probabilities - reconstructed)
    if not np.allclose(
        probabilities, saved_probabilities, rtol=0.0, atol=PROBABILITY_TOLERANCE
    ):
        raise AssertionError("Part 3 probabilities do not align with metadata")
    if not np.allclose(
        reconstructed, saved_reconstructed, rtol=0.0, atol=PROBABILITY_TOLERANCE
    ):
        raise AssertionError("Reconstructed probabilities differ from metadata")
    if not np.allclose(errors, saved_errors, rtol=0.0, atol=PROBABILITY_TOLERANCE):
        raise AssertionError("Additivity errors differ from Part 3 metadata")
    if float(errors.max()) > ADDITIVITY_TOLERANCE:
        raise AssertionError("A Part 3 case exceeds the accepted 1e-5 tolerance")

    canonical_hashes = full_manifest["canonical_artifact_hashes"]
    array_hash_checks = {
        "shap_values_success_array_sha256": _array_sha256(shap_values),
        "base_values_success_array_sha256": _array_sha256(base_values),
        "model_probabilities_success_array_sha256": _array_sha256(probabilities),
    }
    for key, actual in array_hash_checks.items():
        if canonical_hashes.get(key) != actual:
            raise AssertionError(f"Canonical in-memory array hash mismatch: {key}")
    return shap_values, base_values, probabilities, metadata


def _load_feature_names(
    preprocessing_manifest: dict[str, Any],
) -> pd.DataFrame:
    """Load ordered feature metadata and validate preprocessing conventions."""
    features = pd.read_csv(FEATURE_NAMES_PATH, encoding="utf-8")
    expected_columns = ["feature_index", "feature_name", "source_group"]
    if features.columns.tolist() != expected_columns:
        raise AssertionError("Unexpected feature-name metadata columns")
    if len(features) != EXPECTED_FEATURES:
        raise AssertionError(f"Unexpected feature count: {len(features)}")
    if features["feature_index"].tolist() != list(range(EXPECTED_FEATURES)):
        raise AssertionError("Feature indices are not contiguous and ordered")
    if features["feature_name"].isna().any() or not features["feature_name"].is_unique:
        raise AssertionError("Feature names must be non-null and unique")

    configured_numeric = preprocessing_manifest["predictor_groups"]["numeric"]
    observed_numeric = features.loc[
        features["source_group"].eq("numeric"), "feature_name"
    ].tolist()
    if observed_numeric != configured_numeric:
        raise AssertionError("Numeric feature order differs from preprocessing metadata")
    if preprocessing_manifest["representations"]["tree"]["feature_dimension"] != 165:
        raise AssertionError("Preprocessing manifest TREE dimension is not 165")
    return features


def reconstruct_explained_features(
    feature_names: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Extract exact frozen TEST rows using stored source-row positions."""
    explained = pd.read_csv(
        EXPLAINED_IDS_PATH,
        encoding="utf-8",
        dtype={"case_id": "string"},
    )
    expected_columns = ["sample_order", "source_row_index", "case_id", "target"]
    if explained.columns.tolist() != expected_columns:
        raise AssertionError("Unexpected explained-case columns")
    if len(explained) != EXPECTED_CASES:
        raise AssertionError("Explained-case table does not have 1,000 rows")
    if explained["sample_order"].tolist() != list(range(EXPECTED_CASES)):
        raise AssertionError("Explained cases are not in frozen sample order")
    if explained["case_id"].isna().any() or not explained["case_id"].is_unique:
        raise AssertionError("Explained case IDs must be non-null and unique")
    if not metadata.iloc[:, :4].equals(explained):
        raise AssertionError("Part 3 metadata identity differs from explained cases")

    test_ids = pd.read_csv(
        TEST_IDS_PATH,
        encoding="utf-8",
        dtype={"case_id": "string"},
    )
    if test_ids.columns.tolist() != ["case_id"]:
        raise AssertionError("Unexpected canonical TEST ID columns")
    if test_ids["case_id"].isna().any() or not test_ids["case_id"].is_unique:
        raise AssertionError("Canonical TEST IDs must be non-null and unique")

    test_matrix = sparse.load_npz(X_TEST_PATH)
    if test_matrix.shape != (len(test_ids), EXPECTED_FEATURES):
        raise AssertionError(f"Unexpected canonical TEST matrix shape: {test_matrix.shape}")
    if not np.isfinite(test_matrix.data).all():
        raise AssertionError("Canonical TEST matrix contains NaN or infinity")
    if test_matrix.shape[1] != len(feature_names):
        raise AssertionError("TEST columns do not align with frozen feature metadata")

    indices = explained["source_row_index"].to_numpy(dtype=np.int64)
    if np.unique(indices).size != EXPECTED_CASES:
        raise AssertionError("Explained source-row indices are not unique")
    if (indices < 0).any() or (indices >= len(test_ids)).any():
        raise AssertionError("An explained source-row index is out of bounds")
    expected_ids = test_ids.iloc[indices]["case_id"].reset_index(drop=True)
    if not expected_ids.equals(explained["case_id"].reset_index(drop=True)):
        raise AssertionError("Explained IDs do not match canonical TEST row identity")

    explained_matrix = test_matrix[indices, :].toarray()
    if explained_matrix.shape != EXPECTED_SHAPE:
        raise AssertionError(
            f"Unexpected reconstructed feature shape: {explained_matrix.shape}"
        )
    if not np.isfinite(explained_matrix).all():
        raise AssertionError("Reconstructed explained features contain NaN or infinity")
    return explained_matrix, explained


def compute_global_importance(
    shap_values: np.ndarray,
    feature_names: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """Compute deterministic mean absolute SHAP ranking for all features."""
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    if mean_abs.shape != (EXPECTED_FEATURES,) or not np.isfinite(mean_abs).all():
        raise AssertionError("Invalid global mean absolute SHAP values")
    total_importance = float(mean_abs.sum())
    if not np.isfinite(total_importance) or total_importance <= 0.0:
        raise AssertionError("Total SHAP importance must be finite and positive")

    importance = pd.DataFrame(
        {
            "feature": feature_names["feature_name"].to_numpy(),
            "mean_abs_shap": mean_abs,
        }
    )
    importance = importance.sort_values(
        ["mean_abs_shap", "feature"],
        ascending=[False, True],
        kind="mergesort",
        ignore_index=True,
    )
    importance.insert(0, "rank", np.arange(1, EXPECTED_FEATURES + 1))
    importance["share_of_total_importance"] = (
        importance["mean_abs_shap"] / total_importance
    )
    if importance["rank"].tolist() != list(range(1, EXPECTED_FEATURES + 1)):
        raise AssertionError("Global importance ranks are invalid")
    if not np.isclose(
        importance["share_of_total_importance"].sum(), 1.0, rtol=0.0, atol=1e-12
    ):
        raise AssertionError("Feature importance shares do not sum to one")
    return importance, total_importance


def map_feature_groups(
    feature_names: pd.DataFrame,
    preprocessing_manifest: dict[str, Any],
) -> pd.DataFrame:
    """Map every feature through explicit stored source-group metadata."""
    unknown = sorted(set(feature_names["source_group"]) - set(SOURCE_GROUP_MAP))
    if unknown:
        raise AssertionError(f"Unmapped or ambiguous source groups: {unknown}")
    mapped = feature_names.copy()
    mapped["group"] = mapped["source_group"].map(SOURCE_GROUP_MAP)
    if mapped["group"].isna().any() or len(mapped) != EXPECTED_FEATURES:
        raise AssertionError("Every feature must map to exactly one group")

    prefix_rules = {
        "categorical::loan_goal": "loan_goal_",
        "categorical::application_type": "application_type_",
        "structured_map::activity": "activity::",
        "structured_map::action": "action::",
        "structured_map::origin": "origin::",
        "structured_map::resource": "resource::",
        "structured_map::lifecycle": "lifecycle::",
    }
    for source_group, prefix in prefix_rules.items():
        names = mapped.loc[
            mapped["source_group"].eq(source_group), "feature_name"
        ].astype(str)
        if names.empty or not names.str.startswith(prefix).all():
            raise AssertionError(
                f"Feature names do not match stored convention for {source_group}"
            )

    sizes = preprocessing_manifest["train_fitted_sizes"]
    expected_counts = {
        "Numeric": int(sizes["numeric_features"]),
        "LoanGoal": int(sizes["categorical"]["loan_goal"]),
        "ApplicationType": int(sizes["categorical"]["application_type"]),
        "Activity": int(sizes["structured"]["activity"]),
        "Action": int(sizes["structured"]["action"]),
        "EventOrigin": int(sizes["structured"]["origin"]),
        "Resource": int(sizes["structured"]["resource"]),
        "Lifecycle": int(sizes["structured"]["lifecycle"]),
    }
    actual_counts = mapped["group"].value_counts().to_dict()
    if actual_counts != expected_counts:
        raise AssertionError(
            f"Feature group counts differ from preprocessing metadata: {actual_counts}"
        )
    if sum(actual_counts.values()) != EXPECTED_FEATURES:
        raise AssertionError("Feature group counts do not sum to 165")
    return mapped


def compute_group_importance(
    importance: pd.DataFrame,
    mapped_features: pd.DataFrame,
    total_importance: float,
) -> pd.DataFrame:
    """Aggregate individual mean absolute SHAP values in conceptual order."""
    merged = mapped_features[["feature_name", "group"]].merge(
        importance[["feature", "mean_abs_shap"]],
        left_on="feature_name",
        right_on="feature",
        how="left",
        validate="one_to_one",
    )
    if merged["mean_abs_shap"].isna().any():
        raise AssertionError("Feature importance failed to align for group aggregation")

    grouped = (
        merged.groupby("group", sort=False, observed=True)
        .agg(
            feature_count=("feature_name", "size"),
            group_total_importance=("mean_abs_shap", "sum"),
        )
        .reindex(GROUP_ORDER)
        .reset_index()
    )
    if grouped.isna().any().any():
        raise AssertionError("At least one required feature group is absent")
    grouped["feature_count"] = grouped["feature_count"].astype(np.int64)
    grouped["group_mean_importance_per_feature"] = (
        grouped["group_total_importance"] / grouped["feature_count"]
    )
    grouped["share_of_total_importance"] = (
        grouped["group_total_importance"] / total_importance
    )
    if grouped["feature_count"].sum() != EXPECTED_FEATURES:
        raise AssertionError("Grouped feature counts do not sum to 165")
    if not np.isclose(
        grouped["group_total_importance"].sum(),
        total_importance,
        rtol=0.0,
        atol=1e-15,
    ):
        raise AssertionError("Group totals do not equal total feature importance")
    if not np.isclose(
        grouped["share_of_total_importance"].sum(),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("Group importance shares do not sum to one")
    return grouped


def load_frozen_predictions(
    explained: pd.DataFrame,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Join canonical RF labels and validate targets and probabilities."""
    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        encoding="utf-8",
        dtype={"case_id": "string"},
    )
    expected_columns = ["case_id", "y_true", "y_pred", "y_prob_success"]
    if predictions.columns.tolist() != expected_columns:
        raise AssertionError("Unexpected frozen RF prediction columns")
    if predictions["case_id"].isna().any() or not predictions["case_id"].is_unique:
        raise AssertionError("Frozen RF prediction IDs must be non-null and unique")

    joined = explained.merge(
        predictions,
        on="case_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(joined) != EXPECTED_CASES or joined["y_pred"].isna().any():
        raise AssertionError("Every explained case must have exactly one RF prediction")
    joined = joined.sort_values("sample_order", kind="stable", ignore_index=True)
    if joined["sample_order"].tolist() != list(range(EXPECTED_CASES)):
        raise AssertionError("Prediction join changed frozen sample order")
    if not np.array_equal(
        joined["target"].to_numpy(dtype=np.int64),
        joined["y_true"].to_numpy(dtype=np.int64),
    ):
        raise AssertionError("Frozen RF y_true differs from explained target")
    if not np.allclose(
        joined["y_prob_success"].to_numpy(dtype=np.float64),
        probabilities,
        rtol=0.0,
        atol=PROBABILITY_TOLERANCE,
    ):
        raise AssertionError("Part 3 probabilities differ from frozen RF probabilities")
    if not set(joined["y_true"].unique()).issubset({0, 1}):
        raise AssertionError("Unexpected true-label value")
    if not set(joined["y_pred"].unique()).issubset({0, 1}):
        raise AssertionError("Unexpected predicted-label value")
    return joined


def select_local_examples(
    joined: pd.DataFrame,
    metadata: pd.DataFrame,
    base_values: np.ndarray,
    probabilities: np.ndarray,
    shap_values: np.ndarray,
) -> pd.DataFrame:
    """Select the first frozen-sample TP, TN, FP, and FN deterministically."""
    true_values = joined["y_true"].to_numpy(dtype=np.int64)
    predicted_values = joined["y_pred"].to_numpy(dtype=np.int64)
    conditions = {
        "TP": (true_values == 1) & (predicted_values == 1),
        "TN": (true_values == 0) & (predicted_values == 0),
        "FP": (true_values == 0) & (predicted_values == 1),
        "FN": (true_values == 1) & (predicted_values == 0),
    }
    selected_rows: list[dict[str, Any]] = []
    for category in LOCAL_CATEGORY_ORDER:
        candidates = joined.loc[conditions[category]]
        if candidates.empty:
            raise AssertionError(f"No {category} case exists in the explained sample")
        selected = candidates.loc[candidates["sample_order"].idxmin()]
        sample_order = int(selected["sample_order"])
        if sample_order != int(candidates["sample_order"].min()):
            raise AssertionError(f"{category} selection is not minimum sample_order")
        reconstructed = float(base_values[sample_order] + shap_values[sample_order].sum())
        selected_rows.append(
            {
                "category": category,
                "sample_order": sample_order,
                "source_row_index": int(selected["source_row_index"]),
                "case_id": str(selected["case_id"]),
                "y_true": int(selected["y_true"]),
                "y_pred": int(selected["y_pred"]),
                "model_probability_success": float(probabilities[sample_order]),
                "base_value_success": float(base_values[sample_order]),
                "reconstructed_probability_success": reconstructed,
                "absolute_additivity_error": float(
                    abs(probabilities[sample_order] - reconstructed)
                ),
            }
        )
        if str(metadata.iloc[sample_order]["case_id"]) != str(selected["case_id"]):
            raise AssertionError(f"{category} local SHAP row is misaligned")

    local_examples = pd.DataFrame(selected_rows)
    if len(local_examples) != 4:
        raise AssertionError("Exactly four local examples must be selected")
    if local_examples["category"].tolist() != LOCAL_CATEGORY_ORDER:
        raise AssertionError("Local example category order is invalid")
    return local_examples


def build_local_explanations(
    local_examples: pd.DataFrame,
    shap_values: np.ndarray,
    explained_features: np.ndarray,
    feature_names: pd.DataFrame,
) -> pd.DataFrame:
    """Create a complete 4-by-165 machine-readable contribution table."""
    names = feature_names["feature_name"].astype(str).to_numpy()
    records: list[pd.DataFrame] = []
    for example in local_examples.itertuples(index=False):
        sample_order = int(example.sample_order)
        values = shap_values[sample_order]
        feature_values = explained_features[sample_order]
        direction = np.where(values > 0.0, "positive", np.where(values < 0.0, "negative", "zero"))
        records.append(
            pd.DataFrame(
                {
                    "category": example.category,
                    "sample_order": sample_order,
                    "case_id": example.case_id,
                    "feature": names,
                    "feature_value": feature_values,
                    "shap_value": values,
                    "abs_shap_value": np.abs(values),
                    "direction": direction,
                }
            )
        )
    contributions = pd.concat(records, ignore_index=True)
    if len(contributions) != 4 * EXPECTED_FEATURES:
        raise AssertionError("Local contribution table must contain 660 rows")
    counts = contributions.groupby("category", sort=False).size().to_dict()
    if counts != {category: EXPECTED_FEATURES for category in LOCAL_CATEGORY_ORDER}:
        raise AssertionError("Every local example must contain all 165 features")
    return contributions


def create_global_plots(
    importance: pd.DataFrame,
    shap_values: np.ndarray,
    base_values: np.ndarray,
    explained_features: np.ndarray,
    feature_names: pd.DataFrame,
) -> None:
    """Create top-20 global bar and beeswarm plots from frozen values."""
    top = importance.head(TOP_FEATURES).iloc[::-1]
    figure, axis = plt.subplots(figsize=(11, 8.5))
    axis.barh(top["feature"], top["mean_abs_shap"], color="#2f6f9f")
    axis.set_xlabel("Mean absolute SHAP value (Success probability output)")
    axis.set_ylabel("Transformed model feature")
    axis.set_title("Top 20 global SHAP features — Success class 1")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    _save_figure_atomic(GLOBAL_BAR_PATH, figure)

    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_values,
        data=explained_features,
        feature_names=feature_names["feature_name"].astype(str).tolist(),
    )
    feature_index_by_name = {
        name: index
        for index, name in enumerate(
            feature_names["feature_name"].astype(str).tolist()
        )
    }
    top_indices = [
        feature_index_by_name[name]
        for name in importance.head(TOP_FEATURES)["feature"].astype(str)
    ]
    top_explanation = explanation[:, top_indices]
    np.random.seed(PLOT_SEED)
    shap.plots.beeswarm(
        top_explanation,
        max_display=TOP_FEATURES,
        show=False,
        plot_size=(12, 8.5),
    )
    figure = plt.gcf()
    axes = figure.get_axes()
    if axes:
        axes[0].set_title("Top 20 SHAP distributions — Success class 1")
        axes[0].set_xlabel("SHAP value (contribution to Success probability output)")
    figure.tight_layout()
    _save_figure_atomic(BEESWARM_PATH, figure)


def create_local_plots(
    local_examples: pd.DataFrame,
    shap_values: np.ndarray,
    base_values: np.ndarray,
    explained_features: np.ndarray,
    feature_names: pd.DataFrame,
) -> None:
    """Create four aligned waterfall plots from stored local SHAP rows."""
    names = feature_names["feature_name"].astype(str).tolist()
    for example in local_examples.itertuples(index=False):
        sample_order = int(example.sample_order)
        explanation = shap.Explanation(
            values=shap_values[sample_order],
            base_values=base_values[sample_order],
            data=explained_features[sample_order],
            feature_names=names,
        )
        np.random.seed(PLOT_SEED)
        shap.plots.waterfall(
            explanation,
            max_display=LOCAL_DISPLAY_FEATURES,
            show=False,
        )
        figure = plt.gcf()
        figure.set_size_inches(12, 8.5)
        figure.suptitle(
            f"{example.category}: {example.case_id} | true={example.y_true}, "
            f"predicted={example.y_pred}, P(Success)="
            f"{example.model_probability_success:.6f}",
            y=1.02,
            fontsize=12,
        )
        _save_figure_atomic(LOCAL_PLOT_PATHS[example.category], figure)


def _markdown_table(frame: pd.DataFrame, formats: dict[str, str]) -> str:
    """Create a small Markdown table without optional tabulate dependencies."""
    columns = frame.columns.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for column, value in zip(columns, row, strict=True):
            if column in formats:
                cells.append(formats[column].format(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _local_contribution_lines(
    category: str,
    contributions: pd.DataFrame,
    limit: int = 5,
) -> tuple[str, str]:
    """Return concise top positive and negative model-contribution text."""
    subset = contributions.loc[contributions["category"].eq(category)]
    positive = subset.loc[subset["shap_value"] > 0].nlargest(limit, "shap_value")
    negative = subset.loc[subset["shap_value"] < 0].nsmallest(limit, "shap_value")

    def render(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "none"
        return "; ".join(
            f"`{row.feature}` ({row.shap_value:+.6f})"
            for row in frame.itertuples(index=False)
        )

    return render(positive), render(negative)


def write_shap_pilot_report(
    full_manifest: dict[str, Any],
    resolution: dict[str, Any],
    importance: pd.DataFrame,
    groups: pd.DataFrame,
    local_examples: pd.DataFrame,
    contributions: pd.DataFrame,
    deterministic_rerun_result: str,
) -> None:
    """Write the requested Part 4 scientific report from canonical tables."""
    top_table = importance.head(10)[["rank", "feature", "mean_abs_shap", "share_of_total_importance"]]
    group_table = groups.copy()
    top_markdown = _markdown_table(
        top_table,
        {
            "mean_abs_shap": "{:.9f}",
            "share_of_total_importance": "{:.6%}",
        },
    )
    group_markdown = _markdown_table(
        group_table,
        {
            "group_total_importance": "{:.9f}",
            "group_mean_importance_per_feature": "{:.9f}",
            "share_of_total_importance": "{:.6%}",
        },
    )

    local_sections: list[str] = []
    for example in local_examples.itertuples(index=False):
        positive, negative = _local_contribution_lines(
            example.category, contributions
        )
        local_sections.append(
            f"### {example.category}\n\n"
            f"- Case ID: `{example.case_id}`\n"
            f"- Frozen sample order: {example.sample_order}\n"
            f"- True class: {example.y_true}\n"
            f"- Predicted class: {example.y_pred}\n"
            f"- Predicted Success probability: {example.model_probability_success:.9f}\n"
            f"- Largest positive model contributions: {positive}\n"
            f"- Largest negative model contributions: {negative}\n"
        )
    local_text = "\n".join(local_sections)

    max_error = float(full_manifest["maximum_additivity_error"])
    mean_error = float(full_manifest["mean_additivity_error"])
    median_error = float(full_manifest["median_additivity_error"])
    original_error = float(resolution["original_max_error"])
    report = f"""# SHAP Pilot V1

## 1. Objective

This pilot validates and inspects SHAP explanations for the frozen, untuned Random Forest Baseline V1 at prediction point k=10. It is a reporting and interpretation layer over the canonical Part 3 values, not the final tuned-model SHAP evaluation.

## 2. Frozen Model and Data

- Model: Random Forest Baseline V1
- Model tuning: none
- Prediction point: k=10
- Feature count: 165
- SHAP background: 500 TRAIN cases
- Explained sample: 1,000 TEST cases
- Positive class: Success = 1
- SHAP version: 0.52.0

The explained rows were reconstructed directly from the canonical transformed TEST representation using the frozen `source_row_index` values. No preprocessing was fitted or applied in Part 4.

## 3. SHAP Configuration

Part 3 used TreeExplainer with interventional feature perturbation, probability model output, and `approximate=False`. The background contains only 500 frozen TRAIN cases. The 1,000-case TEST explained sample is frozen, and no resampling occurred. Part 4 only constructs `shap.Explanation` containers from saved values for plotting; it does not construct an explainer or recompute explanations.

## 4. Numerical Validation

The original project-defined absolute probability-space tolerance of 1e-6 failed, with a smoke-test maximum residual of {original_error:.12g} (approximately 6.67e-6). Diagnostic checks of model output, class mapping, base-value mapping, feature alignment, and frozen input identity passed. The project then adopted a revised absolute probability-space tolerance of 1e-5 before the full computation; this is a project-defined validation criterion, not a SHAP library default.

Across all 1,000 frozen cases, the maximum absolute additivity error was {max_error:.12g}, the mean was {mean_error:.12g}, and the median was {median_error:.12g}. All 1,000 cases passed 1e-5, and the SHAP values, base values, model probabilities, and reconstructed feature matrix contained no NaN or infinity.

## 5. Global Feature Importance

Global importance is the mean absolute SHAP value over all 1,000 frozen explained cases. The complete canonical ranking is in [`global_shap_importance.csv`](global_shap_importance.csv).

{top_markdown}

The top-20 magnitude ranking is visualized in [`shap_global_bar.png`](shap_global_bar.png), and the distribution of signed contributions is shown in [`shap_beeswarm.png`](shap_beeswarm.png). Beeswarm color represents the exact transformed feature value supplied to the frozen model. Many inputs are one-hot or count encoded, so these colors must not be read as raw business-unit values for every feature. These plots describe the frozen model and do not imply causality.

## 6. Feature-Group Importance

The complete grouped output is in [`global_shap_group_importance.csv`](global_shap_group_importance.csv).

{group_markdown}

Group totals sum individual feature magnitudes and are descriptive. In particular, Resource contains 115 of the 165 encoded dimensions. Its total importance must therefore be interpreted together with its mean importance per feature; total group importance alone does not establish that Resource is intrinsically the most influential feature type.

## 7. Local Examples

For each TP, TN, FP, and FN category, the selected example is the case with the smallest `sample_order` in the frozen 1,000-case sample. Selection did not use confidence, SHAP magnitude, probability extremes, interesting features, or visual appearance. Complete metadata is in [`local_examples.csv`](local_examples.csv), and all 660 feature contributions are in [`local_feature_contributions.csv`](local_feature_contributions.csv).

Positive SHAP values mean that a feature contribution moves the frozen model output toward a higher predicted Success probability relative to the SHAP reference expectation for that case. Negative values move it toward a lower predicted Success probability. They are model contributions, not causal effects.

{local_text}
The corresponding waterfall plots are `local_tp_waterfall.png`, `local_tn_waterfall.png`, `local_fp_waterfall.png`, and `local_fn_waterfall.png`. Each uses the same 15-feature display limit and is constructed from the aligned frozen SHAP row, base value, and transformed feature row.

## 8. Interpretation Boundaries

SHAP explains the frozen model prediction; it does not establish causality. The Random Forest is an untuned baseline, and explanations may change after appropriate model validation or tuning. This pilot does not establish temporal explanation stability and includes no temporal-window, drift, Jaccard, Spearman, or attribution-sign stability analysis.

## 9. Pilot Outcome

- Canonical SHAP computation: completed in Part 3 and hash-validated in Part 4
- Global explanations: produced
- Local explanations: produced
- Feature and class alignment: valid
- SHAP recomputation in Part 4: none
- Deterministic Part 4 rerun comparison: {deterministic_rerun_result}
- Protected Part 0--3 artifact integrity: PASS

## 10. Next Step

Temporal explanation-stability analysis is outside this Part 4 task and was not implemented.
"""
    _write_text_atomic(REPORT_PATH, report)


def _output_manifest(
    shap_values: np.ndarray,
    explained_features: np.ndarray,
    importance: pd.DataFrame,
    groups: pd.DataFrame,
    local_examples: pd.DataFrame,
    total_importance: float,
    protected_before: dict[Path, str],
    protected_after: dict[Path, str],
    deterministic_rerun_result: str,
    builder_source_sha256: str,
) -> dict[str, Any]:
    """Build the deterministic Part 4 validation and output manifest."""
    output_paths = [*REPRODUCIBLE_OUTPUT_PATHS, REPORT_PATH]
    output_hashes = {_relative(path): _sha256(path) for path in output_paths}
    group_records = []
    for row in groups.itertuples(index=False):
        group_records.append(
            {
                "group": str(row.group),
                "feature_count": int(row.feature_count),
                "group_total_importance": float(row.group_total_importance),
                "group_mean_importance_per_feature": float(
                    row.group_mean_importance_per_feature
                ),
                "share_of_total_importance": float(row.share_of_total_importance),
            }
        )
    local_records = []
    for row in local_examples.itertuples(index=False):
        local_records.append(
            {
                "category": str(row.category),
                "sample_order": int(row.sample_order),
                "case_id": str(row.case_id),
                "y_true": int(row.y_true),
                "y_pred": int(row.y_pred),
                "model_probability_success": float(row.model_probability_success),
            }
        )
    assertions = {
        "shap_matrix_shape_1000_by_165": "PASS",
        "base_values_shape_1000": "PASS",
        "model_probabilities_shape_1000": "PASS",
        "metadata_rows_1000": "PASS",
        "feature_names_count_165": "PASS",
        "feature_names_unique": "PASS",
        "part_3_canonical_hashes_match": "PASS",
        "explained_feature_matrix_shape_1000_by_165": "PASS",
        "explained_case_order_matches": "PASS",
        "no_missing_case_ids": "PASS",
        "no_duplicate_case_ids": "PASS",
        "all_shap_values_finite": "PASS",
        "all_features_map_to_exactly_one_group": "PASS",
        "group_feature_counts_sum_to_165": "PASS",
        "feature_importance_shares_sum_to_one": "PASS",
        "group_importance_shares_sum_to_one": "PASS",
        "tp_tn_fp_fn_examples_exist": "PASS",
        "exactly_four_local_examples": "PASS",
        "minimum_sample_order_selection": "PASS",
        "rf_y_true_matches_explained_target": "PASS",
        "part_3_probabilities_match_frozen_rf": "PASS",
        "local_shap_rows_match_sample_order": "PASS",
        "no_explainer_constructed": "PASS",
        "no_shap_value_computation": "PASS",
        "no_model_training": "PASS",
        "no_preprocessing_fitting": "PASS",
        "no_tuning": "PASS",
        "no_resampling": "PASS",
        "no_temporal_window_analysis": "PASS",
        "no_explanation_stability_metrics": "PASS",
        "protected_part_0_through_3_artifacts_unchanged": "PASS",
    }
    return {
        "experiment": "shap_pilot_v1",
        "stage": "part_4_reporting_and_interpretation",
        "builder_source_sha256": builder_source_sha256,
        "shap_version": str(shap.__version__),
        "positive_class": POSITIVE_CLASS,
        "shap_matrix_shape": list(shap_values.shape),
        "explained_feature_matrix_shape": list(explained_features.shape),
        "total_feature_importance": total_importance,
        "top_10_features": [
            {
                "rank": int(row.rank),
                "feature": str(row.feature),
                "mean_abs_shap": float(row.mean_abs_shap),
            }
            for row in importance.head(10).itertuples(index=False)
        ],
        "feature_groups": group_records,
        "local_examples": local_records,
        "plot_seed": PLOT_SEED,
        "global_plot_max_display": TOP_FEATURES,
        "local_plot_max_display": LOCAL_DISPLAY_FEATURES,
        "deterministic_rerun": {
            "compared_csv_and_png_outputs": len(REPRODUCIBLE_OUTPUT_PATHS),
            "result": deterministic_rerun_result,
        },
        "protected_artifacts": {
            "count": len(protected_before),
            "unchanged": protected_before == protected_after,
            "sha256": {
                _relative(path): digest
                for path, digest in sorted(
                    protected_after.items(), key=lambda item: _relative(item[0])
                )
            },
        },
        "outputs_sha256": output_hashes,
        "assertions": assertions,
        "assertion_status": "PASS",
    }


def validate_outputs(
    importance: pd.DataFrame,
    groups: pd.DataFrame,
    local_examples: pd.DataFrame,
    contributions: pd.DataFrame,
) -> None:
    """Reload all numerical CSVs and validate final output contracts."""
    reloaded_importance = pd.read_csv(GLOBAL_IMPORTANCE_PATH)
    reloaded_groups = pd.read_csv(GROUP_IMPORTANCE_PATH)
    reloaded_examples = pd.read_csv(LOCAL_EXAMPLES_PATH, dtype={"case_id": "string"})
    reloaded_contributions = pd.read_csv(
        LOCAL_CONTRIBUTIONS_PATH, dtype={"case_id": "string"}
    )
    if reloaded_importance.shape != importance.shape or len(reloaded_importance) != 165:
        raise AssertionError("Invalid saved global importance CSV")
    if reloaded_groups["group"].tolist() != GROUP_ORDER:
        raise AssertionError("Invalid saved feature-group order")
    if reloaded_groups["feature_count"].sum() != 165:
        raise AssertionError("Invalid saved feature-group counts")
    if reloaded_examples["category"].tolist() != LOCAL_CATEGORY_ORDER:
        raise AssertionError("Invalid saved local-example order")
    if len(reloaded_examples) != len(local_examples) or len(reloaded_examples) != 4:
        raise AssertionError("Invalid saved local-example row count")
    if len(reloaded_contributions) != len(contributions) or len(contributions) != 660:
        raise AssertionError("Invalid saved local-contribution row count")
    if not np.isclose(
        reloaded_importance["share_of_total_importance"].sum(),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("Saved feature shares do not sum to one")
    if not np.isclose(
        reloaded_groups["share_of_total_importance"].sum(),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("Saved group shares do not sum to one")
    required_outputs = [*REPRODUCIBLE_OUTPUT_PATHS, REPORT_PATH]
    missing = [path for path in required_outputs if not path.is_file()]
    if missing:
        raise AssertionError(f"Missing Part 4 outputs: {missing}")
    empty = [path for path in required_outputs if path.stat().st_size == 0]
    if empty:
        raise AssertionError(f"Empty Part 4 outputs: {empty}")


def main() -> None:
    """Validate frozen inputs and build SHAP Pilot V1 Part 4 outputs."""
    _validate_source_guardrails()
    if str(shap.__version__) != EXPECTED_SHAP_VERSION:
        raise AssertionError(
            f"SHAP version must remain {EXPECTED_SHAP_VERSION}; found {shap.__version__}"
        )

    full_manifest = _load_json(FULL_MANIFEST_PATH)
    sample_manifest = _load_json(SAMPLE_MANIFEST_PATH)
    model_manifest = _load_json(MODEL_MANIFEST_PATH)
    core_manifest = _load_json(CORE_MANIFEST_PATH)
    resolution = _load_json(CORE_RESOLUTION_PATH)
    preprocessing_manifest = _load_json(PREPROCESSING_MANIFEST_PATH)
    feature_manifest = _load_json(FEATURE_MANIFEST_PATH)

    if any(
        manifest.get("assertion_status") != "PASS"
        for manifest in (full_manifest, sample_manifest, model_manifest, resolution)
    ):
        raise AssertionError("A frozen handoff manifest does not have PASS status")
    if core_manifest.get("additivity_result") != "FAIL":
        raise AssertionError("The original Part 2 tolerance failure was not preserved")
    if full_manifest.get("stage") != "part_3_full_tree_shap_computation":
        raise AssertionError("Unexpected Part 3 manifest stage")
    expected_configuration = {
        "shap_version": EXPECTED_SHAP_VERSION,
        "feature_count": EXPECTED_FEATURES,
        "background_cases": 500,
        "explained_cases": EXPECTED_CASES,
        "positive_class": POSITIVE_CLASS,
        "feature_perturbation": "interventional",
        "model_output": "probability",
        "approximate": False,
        "additivity_tolerance": ADDITIVITY_TOLERANCE,
    }
    for key, expected in expected_configuration.items():
        if full_manifest.get(key) != expected:
            raise AssertionError(f"Unexpected Part 3 configuration for {key}")
    if feature_manifest.get("prediction_point") != 10:
        raise AssertionError("Feature manifest prediction point is not k=10")

    validate_part3_hashes(full_manifest)
    _validate_recorded_input_hashes(full_manifest, sample_manifest)
    protected_paths = _protected_paths(full_manifest)
    protected_before = {path: _sha256(path) for path in protected_paths}
    previous_output_hashes = _existing_hashes(REPRODUCIBLE_OUTPUT_PATHS)
    builder_source_sha256 = _sha256(Path(__file__))
    previous_builder_source_sha256 = None
    if REPORT_MANIFEST_PATH.is_file():
        previous_builder_source_sha256 = _load_json(REPORT_MANIFEST_PATH).get(
            "builder_source_sha256"
        )

    shap_values, base_values, probabilities, metadata = (
        load_canonical_shap_outputs(full_manifest)
    )
    feature_names = _load_feature_names(preprocessing_manifest)
    explained_features, explained = reconstruct_explained_features(
        feature_names, metadata
    )
    importance, total_importance = compute_global_importance(
        shap_values, feature_names
    )
    mapped_features = map_feature_groups(feature_names, preprocessing_manifest)
    groups = compute_group_importance(
        importance, mapped_features, total_importance
    )
    joined = load_frozen_predictions(explained, probabilities)
    local_examples = select_local_examples(
        joined, metadata, base_values, probabilities, shap_values
    )
    contributions = build_local_explanations(
        local_examples, shap_values, explained_features, feature_names
    )

    _write_csv_atomic(GLOBAL_IMPORTANCE_PATH, importance)
    _write_csv_atomic(GROUP_IMPORTANCE_PATH, groups)
    _write_csv_atomic(LOCAL_EXAMPLES_PATH, local_examples)
    _write_csv_atomic(LOCAL_CONTRIBUTIONS_PATH, contributions)
    create_global_plots(
        importance,
        shap_values,
        base_values,
        explained_features,
        feature_names,
    )
    create_local_plots(
        local_examples,
        shap_values,
        base_values,
        explained_features,
        feature_names,
    )

    current_output_hashes = {
        path: _sha256(path) for path in REPRODUCIBLE_OUTPUT_PATHS
    }
    if previous_output_hashes is None:
        deterministic_rerun_result = "FIRST_RUN_NO_PRIOR_OUTPUTS"
    elif previous_builder_source_sha256 != builder_source_sha256:
        deterministic_rerun_result = "SOURCE_CHANGED_REBASELINE"
    elif previous_output_hashes == current_output_hashes:
        deterministic_rerun_result = "PASS"
    else:
        differing = [
            _relative(path)
            for path in REPRODUCIBLE_OUTPUT_PATHS
            if previous_output_hashes[path] != current_output_hashes[path]
        ]
        raise AssertionError(
            f"Deterministic rerun changed CSV or PNG outputs: {differing}"
        )

    write_shap_pilot_report(
        full_manifest,
        resolution,
        importance,
        groups,
        local_examples,
        contributions,
        deterministic_rerun_result,
    )
    validate_outputs(importance, groups, local_examples, contributions)

    protected_after = {path: _sha256(path) for path in protected_paths}
    if protected_before != protected_after:
        changed = [
            _relative(path)
            for path in protected_paths
            if protected_before[path] != protected_after[path]
        ]
        raise AssertionError(f"Protected Part 0--3 artifacts changed: {changed}")

    report_manifest = _output_manifest(
        shap_values,
        explained_features,
        importance,
        groups,
        local_examples,
        total_importance,
        protected_before,
        protected_after,
        deterministic_rerun_result,
        builder_source_sha256,
    )
    _write_text_atomic(
        REPORT_MANIFEST_PATH,
        json.dumps(report_manifest, indent=2, ensure_ascii=False) + "\n",
    )

    print("SHAP Pilot V1 Part 4: PASS")
    print(f"SHAP matrix shape: {shap_values.shape}")
    print(f"Explained feature matrix shape: {explained_features.shape}")
    print(f"Total feature importance: {total_importance:.17g}")
    print(f"Deterministic rerun: {deterministic_rerun_result}")
    print(f"Protected artifacts unchanged: {protected_before == protected_after}")


if __name__ == "__main__":
    main()
