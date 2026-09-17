"""Compute canonical, checkpointed Part 3 TreeSHAP values for SHAP Pilot V1.

This script consumes only frozen Part 0--2b inputs. It computes Success-class
TreeSHAP values in fixed 50-case batches, validates each batch before an atomic
checkpoint write, resumes only from fully validated checkpoints, and assembles
the canonical Part 3 arrays and numerical reports. It does not fit, resample,
rank features, create plots, or perform temporal or stability analyses.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import shap
import sklearn
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "shap_pilot_v1"
BASELINE_RESULTS_DIR = PROJECT_ROOT / "results" / "baseline_v1"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "baseline_k10"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "shap_pilot_v1"
BATCH_DIR = OUTPUT_DIR / "batches"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "shap_pilot_v1"

MODEL_PATH = ARTIFACTS_DIR / "random_forest_baseline.joblib"
MODEL_MANIFEST_PATH = RESULTS_DIR / "model_freeze_manifest.json"
BACKGROUND_IDS_PATH = RESULTS_DIR / "background_case_ids.csv"
EXPLAINED_IDS_PATH = RESULTS_DIR / "explained_case_ids.csv"
SAMPLE_MANIFEST_PATH = RESULTS_DIR / "sample_manifest.json"
CORE_MANIFEST_PATH = RESULTS_DIR / "shap_core_manifest.json"
RESOLUTION_PATH = RESULTS_DIR / "shap_core_resolution.json"
FEATURE_NAMES_PATH = BASELINE_RESULTS_DIR / "feature_names_tree.csv"
TRAIN_IDS_PATH = BASELINE_RESULTS_DIR / "train_case_ids.csv"
TEST_IDS_PATH = BASELINE_RESULTS_DIR / "test_case_ids.csv"
X_TRAIN_PATH = PROCESSED_DIR / "X_train_tree.npz"
X_TEST_PATH = PROCESSED_DIR / "X_test_tree.npz"
Y_TRAIN_PATH = PROCESSED_DIR / "y_train.npy"
Y_TEST_PATH = PROCESSED_DIR / "y_test.npy"

SHAP_VALUES_PATH = OUTPUT_DIR / "shap_values_success.npy"
BASE_VALUES_PATH = OUTPUT_DIR / "base_values_success.npy"
PROBABILITIES_PATH = OUTPUT_DIR / "model_probabilities_success.npy"
FULL_METADATA_PATH = RESULTS_DIR / "full_shap_case_metadata.csv"
RUNTIME_PATH = RESULTS_DIR / "shap_batch_runtime.csv"
FULL_MANIFEST_PATH = RESULTS_DIR / "full_shap_manifest.json"

EXPECTED_SHAP_VERSION = "0.52.0"
EXPECTED_FEATURES = 165
EXPECTED_BACKGROUND_CASES = 500
EXPECTED_EXPLAINED_CASES = 1_000
EXPECTED_ESTIMATORS = 300
POSITIVE_CLASS = 1
FEATURE_PERTURBATION = "interventional"
MODEL_OUTPUT = "probability"
APPROXIMATE = False
ADDITIVITY_TOLERANCE = 1e-5
BATCH_SIZE = 50
EXPECTED_BATCH_COUNT = 20

SAMPLE_COLUMNS = ["sample_order", "source_row_index", "case_id", "target"]
METADATA_COLUMNS = SAMPLE_COLUMNS + [
    "model_probability_success",
    "reconstructed_probability_success",
    "absolute_additivity_error",
]
RUNTIME_COLUMNS = [
    "batch_number",
    "batch_size",
    "start_time",
    "elapsed_seconds",
    "seconds_per_case",
]
FORBIDDEN_FEATURES = {
    "case_id",
    "case_start_time",
    "target",
    "outcome_activity",
    "outcome_position",
    "A_Pending",
    "A_Cancelled",
    "A_Denied",
}


def _relative(path: Path) -> str:
    """Return a stable project-relative POSIX path."""
    return path.relative_to(PROJECT_ROOT).as_posix()


def _sha256(path: Path) -> str:
    """Return a file's SHA-256 digest without modifying it."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    """Hash an array's shape, dtype, and C-contiguous bytes."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    """Load and require a JSON object."""
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AssertionError(f"Expected a JSON object in {_relative(path)}")
    return value


def _write_text_atomic(path: Path, content: str) -> None:
    """Atomically replace a text output."""
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


def _write_npy_atomic(path: Path, array: np.ndarray) -> None:
    """Atomically replace a NumPy array output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    """Atomically create a compressed NumPy checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    if temporary.exists():
        temporary.unlink()
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_text(frame: pd.DataFrame) -> str:
    """Serialize a CSV with deterministic line endings."""
    return frame.to_csv(index=False, lineterminator="\n")


def _protected_paths() -> tuple[Path, ...]:
    """Return all frozen model, sample, Part 2, and canonical source inputs."""
    return (
        MODEL_PATH,
        MODEL_MANIFEST_PATH,
        BACKGROUND_IDS_PATH,
        EXPLAINED_IDS_PATH,
        SAMPLE_MANIFEST_PATH,
        CORE_MANIFEST_PATH,
        RESOLUTION_PATH,
        FEATURE_NAMES_PATH,
        TRAIN_IDS_PATH,
        TEST_IDS_PATH,
        X_TRAIN_PATH,
        X_TEST_PATH,
        Y_TRAIN_PATH,
        Y_TEST_PATH,
    )


def _artifact_hashes(paths: tuple[Path, ...]) -> dict[Path, str]:
    """Hash protected files after requiring every one to exist."""
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing protected inputs: {missing}")
    return {path: _sha256(path) for path in paths}


def _input_fingerprint(hashes: dict[Path, str]) -> str:
    """Return one deterministic fingerprint for checkpoint input identity."""
    relative_hashes = {
        _relative(path): digest
        for path, digest in sorted(hashes.items(), key=lambda item: str(item[0]))
    }
    encoded = json.dumps(relative_hashes, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_handoff_manifests(
    model_manifest: dict[str, Any],
    sample_manifest: dict[str, Any],
    core_manifest: dict[str, Any],
    resolution: dict[str, Any],
) -> None:
    """Validate all frozen Part 0--2b contracts before full computation."""
    for label, manifest in (
        ("Part 0", model_manifest),
        ("Part 1", sample_manifest),
        ("Part 2b", resolution),
    ):
        if manifest.get("assertion_status") != "PASS":
            raise AssertionError(f"{label} handoff does not have PASS status")
    if model_manifest.get("artifact_sha256") != _sha256(MODEL_PATH):
        raise AssertionError("Frozen model hash differs from Part 0 manifest")
    if model_manifest.get("feature_count") != EXPECTED_FEATURES:
        raise AssertionError("Part 0 feature count is not 165")
    if sample_manifest.get("feature_count") != EXPECTED_FEATURES:
        raise AssertionError("Part 1 feature count is not 165")
    if sample_manifest.get("background_sample_size") != EXPECTED_BACKGROUND_CASES:
        raise AssertionError("Part 1 background size is not 500")
    if sample_manifest.get("explained_sample_size") != EXPECTED_EXPLAINED_CASES:
        raise AssertionError("Part 1 explained size is not 1,000")
    if core_manifest.get("stage") != "part_2_tree_shap_core_validation":
        raise AssertionError("Unexpected Part 2 manifest stage")
    if core_manifest.get("additivity_tolerance") != 1e-6:
        raise AssertionError("Part 2 original tolerance is not 1e-6")
    if str(core_manifest.get("additivity_result", "")).upper() != "FAIL":
        raise AssertionError("Part 2 original numerical failure was not preserved")
    if resolution.get("stage") != "part_2b_additivity_tolerance_resolution":
        raise AssertionError("Unexpected Part 2b resolution stage")
    expected_resolution = {
        "shap_version": EXPECTED_SHAP_VERSION,
        "feature_count": EXPECTED_FEATURES,
        "background_cases": EXPECTED_BACKGROUND_CASES,
        "positive_class": POSITIVE_CLASS,
        "positive_class_index": 1,
        "feature_perturbation": FEATURE_PERTURBATION,
        "model_output": MODEL_OUTPUT,
        "approximate": APPROXIMATE,
        "revised_tolerance": ADDITIVITY_TOLERANCE,
        "revised_result": "pass",
        "effective_background_cases": EXPECTED_BACKGROUND_CASES,
        "tree_explainer_configuration_unchanged": True,
    }
    for key, expected in expected_resolution.items():
        if resolution.get(key) != expected:
            raise AssertionError(
                f"Part 2b {key!r} changed: expected {expected!r}, "
                f"found {resolution.get(key)!r}"
            )
    recorded = resolution.get("protected_artifacts", {}).get("sha256", {})
    for relative_path, expected_hash in recorded.items():
        path = PROJECT_ROOT / relative_path
        if not path.is_file() or _sha256(path) != expected_hash:
            raise AssertionError(f"Part 2b protected input changed: {relative_path}")


def _load_feature_names() -> list[str]:
    """Load and validate the canonical ordered TREE feature names."""
    frame = pd.read_csv(FEATURE_NAMES_PATH)
    if not {"feature_index", "feature_name"}.issubset(frame.columns):
        raise AssertionError("Feature-name file has an unexpected schema")
    if len(frame) != EXPECTED_FEATURES:
        raise AssertionError(f"Expected 165 features, found {len(frame)}")
    if not np.array_equal(
        frame["feature_index"].to_numpy(), np.arange(EXPECTED_FEATURES)
    ):
        raise AssertionError("Feature indices are not exactly 0 through 164")
    names = frame["feature_name"].astype(str).tolist()
    if len(set(names)) != EXPECTED_FEATURES:
        raise AssertionError("Feature names are not unique")
    forbidden = sorted(FORBIDDEN_FEATURES.intersection(names))
    if forbidden:
        raise AssertionError(f"Forbidden SHAP features found: {forbidden}")
    return names


def _load_sample_table(path: Path, expected_rows: int) -> pd.DataFrame:
    """Load one frozen sample table without changing row order."""
    frame = pd.read_csv(path, dtype={"case_id": "string"})
    if frame.columns.tolist() != SAMPLE_COLUMNS:
        raise AssertionError(f"Unexpected columns in {_relative(path)}")
    if len(frame) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} rows in {_relative(path)}, found {len(frame)}"
        )
    if not np.array_equal(frame["sample_order"], np.arange(expected_rows)):
        raise AssertionError(f"Invalid sample order in {_relative(path)}")
    if frame["case_id"].isna().any() or not frame["case_id"].is_unique:
        raise AssertionError(f"Invalid case IDs in {_relative(path)}")
    return frame


def _validate_sample_identity(
    sample: pd.DataFrame,
    canonical_ids: pd.DataFrame,
    target: np.ndarray,
    matrix: sparse.csr_matrix,
    sample_name: str,
) -> np.ndarray:
    """Validate frozen indices against canonical IDs, targets, and matrix."""
    if "case_id" not in canonical_ids.columns:
        raise AssertionError(f"Canonical {sample_name} IDs lack case_id")
    if matrix.shape[0] != target.shape[0] or len(canonical_ids) != target.shape[0]:
        raise AssertionError(f"Canonical {sample_name} sources are misaligned")
    indices = sample["source_row_index"].to_numpy(dtype=np.int64)
    if (indices < 0).any() or (indices >= matrix.shape[0]).any():
        raise AssertionError(f"Frozen {sample_name} row index is out of bounds")
    expected_ids = canonical_ids.iloc[indices]["case_id"].astype(str).to_numpy()
    actual_ids = sample["case_id"].astype(str).to_numpy()
    if not np.array_equal(actual_ids, expected_ids):
        raise AssertionError(f"Frozen {sample_name} case IDs do not match")
    if not np.array_equal(sample["target"].to_numpy(), target[indices]):
        raise AssertionError(f"Frozen {sample_name} targets do not match")
    return indices


def _dense_copy_is_identical(
    sparse_matrix: sparse.csr_matrix, dense_matrix: np.ndarray
) -> bool:
    """Verify exact sparse-to-dense numerical identity."""
    if dense_matrix.shape != sparse_matrix.shape:
        return False
    return (sparse_matrix - sparse.csr_matrix(dense_matrix)).nnz == 0


def load_frozen_inputs() -> dict[str, Any]:
    """Load and validate frozen inputs without fitting or resampling."""
    if shap.__version__ != EXPECTED_SHAP_VERSION:
        raise AssertionError(
            f"SHAP must be {EXPECTED_SHAP_VERSION}, found {shap.__version__}"
        )
    protected_paths = _protected_paths()
    protected_hashes = _artifact_hashes(protected_paths)
    model_manifest = _load_json(MODEL_MANIFEST_PATH)
    sample_manifest = _load_json(SAMPLE_MANIFEST_PATH)
    core_manifest = _load_json(CORE_MANIFEST_PATH)
    resolution = _load_json(RESOLUTION_PATH)
    _validate_handoff_manifests(
        model_manifest, sample_manifest, core_manifest, resolution
    )
    feature_names = _load_feature_names()

    background = _load_sample_table(BACKGROUND_IDS_PATH, EXPECTED_BACKGROUND_CASES)
    explained = _load_sample_table(EXPLAINED_IDS_PATH, EXPECTED_EXPLAINED_CASES)
    train_ids = pd.read_csv(TRAIN_IDS_PATH, dtype={"case_id": "string"})
    test_ids = pd.read_csv(TEST_IDS_PATH, dtype={"case_id": "string"})
    X_train = sparse.load_npz(X_TRAIN_PATH).tocsr()
    X_test = sparse.load_npz(X_TEST_PATH).tocsr()
    y_train = np.load(Y_TRAIN_PATH, allow_pickle=False)
    y_test = np.load(Y_TEST_PATH, allow_pickle=False)
    if X_train.shape[1] != EXPECTED_FEATURES or X_test.shape[1] != EXPECTED_FEATURES:
        raise AssertionError("Canonical TREE feature count is not 165")

    background_indices = _validate_sample_identity(
        background, train_ids, y_train, X_train, "background"
    )
    explained_indices = _validate_sample_identity(
        explained, test_ids, y_test, X_test, "explained"
    )
    background_sparse = X_train[background_indices].tocsr()
    explained_sparse = X_test[explained_indices].tocsr()
    if background_sparse.shape != (EXPECTED_BACKGROUND_CASES, EXPECTED_FEATURES):
        raise AssertionError(f"Unexpected background shape: {background_sparse.shape}")
    if explained_sparse.shape != (EXPECTED_EXPLAINED_CASES, EXPECTED_FEATURES):
        raise AssertionError(f"Unexpected explained shape: {explained_sparse.shape}")

    background_dense = background_sparse.toarray()
    explained_dense = explained_sparse.toarray()
    if not _dense_copy_is_identical(background_sparse, background_dense):
        raise AssertionError("Background dense conversion changed values")
    if not _dense_copy_is_identical(explained_sparse, explained_dense):
        raise AssertionError("Explained dense conversion changed values")
    if not np.isfinite(background_dense).all():
        raise AssertionError("Dense background contains NaN or infinity")
    if not np.isfinite(explained_dense).all():
        raise AssertionError("Dense explained matrix contains NaN or infinity")

    model = joblib.load(MODEL_PATH)
    if not isinstance(model, RandomForestClassifier):
        raise AssertionError(f"Unexpected frozen model type: {type(model)!r}")
    if model.n_estimators != EXPECTED_ESTIMATORS:
        raise AssertionError(f"Expected 300 estimators, found {model.n_estimators}")
    if int(model.n_features_in_) != EXPECTED_FEATURES:
        raise AssertionError("Frozen model feature count is not 165")
    classes = np.asarray(model.classes_)
    if not np.array_equal(classes, np.array([0, 1])):
        raise AssertionError(f"Frozen model classes are not [0, 1]: {classes!r}")
    positive_matches = np.flatnonzero(classes == POSITIVE_CLASS)
    if positive_matches.size != 1:
        raise AssertionError("Positive class label 1 does not resolve uniquely")
    positive_class_index = int(positive_matches[0])

    probabilities = np.asarray(model.predict_proba(explained_dense), dtype=float)
    if probabilities.shape != (EXPECTED_EXPLAINED_CASES, classes.size):
        raise AssertionError(f"Unexpected predict_proba shape: {probabilities.shape}")
    success_probabilities = probabilities[:, positive_class_index]
    _validate_probabilities(success_probabilities, "model probabilities")
    return {
        "protected_hashes": protected_hashes,
        "input_fingerprint": _input_fingerprint(protected_hashes),
        "feature_names": feature_names,
        "feature_names_sha256": _sha256(FEATURE_NAMES_PATH),
        "background": background,
        "explained": explained,
        "background_dense": background_dense,
        "explained_dense": explained_dense,
        "model": model,
        "classes": classes,
        "positive_class_index": positive_class_index,
        "success_probabilities": success_probabilities,
    }


def _validate_probabilities(values: np.ndarray, label: str) -> None:
    """Require finite probabilities in the closed unit interval."""
    if not np.isfinite(values).all():
        raise AssertionError(f"{label} contain NaN or infinity")
    if ((values < 0.0) | (values > 1.0)).any():
        raise AssertionError(f"{label} fall outside [0, 1]")


def build_explainer(
    model: RandomForestClassifier,
    background: np.ndarray,
    feature_names: list[str],
) -> shap.TreeExplainer:
    """Build the exact Part 2 TreeExplainer with all 500 background rows."""
    masker = shap.maskers.Independent(
        background,
        max_samples=EXPECTED_BACKGROUND_CASES,
    )
    if masker.data.shape != background.shape:
        raise AssertionError("SHAP masker did not retain all 500 background rows")
    if not np.array_equal(masker.data, background):
        raise AssertionError("SHAP masker changed or reordered the background")
    explainer = shap.TreeExplainer(
        model,
        data=masker,
        feature_perturbation=FEATURE_PERTURBATION,
        model_output=MODEL_OUTPUT,
        feature_names=feature_names,
    )
    if explainer.feature_perturbation != FEATURE_PERTURBATION:
        raise AssertionError("TreeExplainer did not retain interventional semantics")
    if explainer.model.model_output != MODEL_OUTPUT:
        raise AssertionError("TreeExplainer did not retain probability output")
    return explainer


def identify_batches() -> list[tuple[int, int, int]]:
    """Return deterministic (batch number, start, stop) definitions."""
    batch_count = (EXPECTED_EXPLAINED_CASES + BATCH_SIZE - 1) // BATCH_SIZE
    if batch_count != EXPECTED_BATCH_COUNT:
        raise AssertionError(f"Expected 20 batches, resolved {batch_count}")
    batches = []
    for batch_number in range(batch_count):
        start = batch_number * BATCH_SIZE
        stop = min(start + BATCH_SIZE, EXPECTED_EXPLAINED_CASES)
        batches.append((batch_number, start, stop))
    if batches[-1] != (19, 950, 1000):
        raise AssertionError("Final batch membership is not sample_order 950--999")
    return batches


def _checkpoint_paths(batch_number: int) -> tuple[Path, Path]:
    stem = f"batch_{batch_number:03d}"
    return BATCH_DIR / f"{stem}.npz", BATCH_DIR / f"{stem}.json"


def _check_checkpoint_directory(batches: list[tuple[int, int, int]]) -> None:
    """Reject unexpected or half-written named checkpoints."""
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    expected_names = {
        path.name
        for batch_number, _, _ in batches
        for path in _checkpoint_paths(batch_number)
    }
    existing = {
        path.name
        for path in BATCH_DIR.iterdir()
        if path.name.startswith("batch_") and path.suffix in {".npz", ".json"}
    }
    unexpected = sorted(existing - expected_names)
    if unexpected:
        raise AssertionError(f"Unexpected checkpoint files: {unexpected}")
    for batch_number, _, _ in batches:
        npz_path, json_path = _checkpoint_paths(batch_number)
        if npz_path.exists() != json_path.exists():
            raise AssertionError(
                f"Incomplete checkpoint pair for batch {batch_number:03d}; "
                "refusing to replace it"
            )


def _extract_success_output(
    explanation: shap.Explanation,
    batch_rows: int,
    positive_class_index: int,
    class_count: int,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Map the modern SHAP output axis explicitly to class label 1."""
    if not isinstance(explanation, shap.Explanation):
        raise AssertionError(f"Unexpected SHAP return type: {type(explanation)!r}")
    raw_names = list(explanation.feature_names or [])
    if raw_names != feature_names:
        raise AssertionError("SHAP feature order differs from canonical feature order")
    values = np.asarray(explanation.values, dtype=float)
    base_values = np.asarray(explanation.base_values, dtype=float)
    expected_shape = (batch_rows, EXPECTED_FEATURES, class_count)
    if values.shape != expected_shape:
        raise AssertionError(
            f"Cannot map SHAP output to classes: got {values.shape}, "
            f"expected {expected_shape}"
        )
    success_values = values[:, :, positive_class_index]
    if base_values.shape == (batch_rows, class_count):
        success_base = base_values[:, positive_class_index]
        base_representation = "per_case_per_class"
    elif base_values.shape == (class_count,):
        success_base = np.full(
            batch_rows, base_values[positive_class_index], dtype=float
        )
        base_representation = "per_class_scalar_broadcast_to_cases"
    else:
        raise AssertionError(f"Cannot map SHAP base values: {base_values.shape}")
    mapping = {
        "raw_output_shape": list(values.shape),
        "raw_base_value_shape": list(base_values.shape),
        "output_axis": 2,
        "class_axis_order": list(range(class_count)),
        "positive_class": POSITIVE_CLASS,
        "positive_class_index": positive_class_index,
        "base_value_representation": base_representation,
    }
    return success_values, success_base, mapping


def validate_additivity(
    sample: pd.DataFrame,
    shap_values: np.ndarray,
    base_values: np.ndarray,
    model_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate all numerical requirements and return reconstruction/errors."""
    rows = len(sample)
    if shap_values.shape != (rows, EXPECTED_FEATURES):
        raise AssertionError(f"Unexpected Success SHAP shape: {shap_values.shape}")
    if base_values.shape != (rows,):
        raise AssertionError(f"Unexpected base-value shape: {base_values.shape}")
    if model_probabilities.shape != (rows,):
        raise AssertionError(
            f"Unexpected model-probability shape: {model_probabilities.shape}"
        )
    if not np.isfinite(shap_values).all():
        raise AssertionError("Success SHAP values contain NaN or infinity")
    if not np.isfinite(base_values).all():
        raise AssertionError("Success base values contain NaN or infinity")
    _validate_probabilities(model_probabilities, "model probabilities")
    reconstructed = base_values + shap_values.sum(axis=1)
    _validate_probabilities(reconstructed, "reconstructed probabilities")
    errors = np.abs(reconstructed - model_probabilities)
    if not np.isfinite(errors).all():
        raise AssertionError("Additivity errors contain NaN or infinity")
    failing = np.flatnonzero(errors > ADDITIVITY_TOLERANCE)
    if failing.size:
        row = int(failing[0])
        case = sample.iloc[row]
        raise AssertionError(
            "Additivity tolerance exceeded: "
            f"case_id={case['case_id']}, sample_order={int(case['sample_order'])}, "
            f"model_probability={model_probabilities[row]:.17g}, "
            f"reconstructed_probability={reconstructed[row]:.17g}, "
            f"absolute_error={errors[row]:.17g}"
        )
    return reconstructed, errors


def compute_one_batch(
    explainer: shap.TreeExplainer,
    batch_number: int,
    sample: pd.DataFrame,
    matrix: np.ndarray,
    model_probabilities: np.ndarray,
    positive_class_index: int,
    class_count: int,
    feature_names: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Compute and validate one batch without persisting partial results."""
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    raw = explainer(
        matrix,
        check_additivity=True,
        approximate=APPROXIMATE,
    )
    elapsed = time.perf_counter() - started
    values, base_values, class_mapping = _extract_success_output(
        raw,
        len(sample),
        positive_class_index,
        class_count,
        feature_names,
    )
    reconstructed, errors = validate_additivity(
        sample, values, base_values, model_probabilities
    )
    arrays = {
        "sample_order": sample["sample_order"].to_numpy(dtype=np.int64),
        "source_row_index": sample["source_row_index"].to_numpy(dtype=np.int64),
        "case_id": sample["case_id"].astype(str).to_numpy(dtype=str),
        "target": sample["target"].to_numpy(dtype=np.int64),
        "model_probability_success": model_probabilities.astype(float),
        "base_value_success": base_values.astype(float),
        "shap_values_success": values.astype(float),
        "reconstructed_probability_success": reconstructed.astype(float),
        "absolute_additivity_error": errors.astype(float),
    }
    runtime = {
        "batch_number": batch_number,
        "batch_size": len(sample),
        "start_time": started_at,
        "elapsed_seconds": elapsed,
        "seconds_per_case": elapsed / len(sample),
        "class_mapping": class_mapping,
    }
    return arrays, runtime


def save_one_batch(
    batch_number: int,
    arrays: dict[str, np.ndarray],
    runtime: dict[str, Any],
    input_fingerprint: str,
    feature_names_sha256: str,
) -> None:
    """Atomically save one already validated checkpoint and its metadata."""
    npz_path, json_path = _checkpoint_paths(batch_number)
    if npz_path.exists() or json_path.exists():
        raise AssertionError(
            f"Checkpoint batch {batch_number:03d} appeared during computation"
        )
    _write_npz_atomic(npz_path, **arrays)
    metadata = {
        "experiment": "shap_pilot_v1",
        "stage": "part_3_full_tree_shap_batch",
        "batch_number": batch_number,
        "batch_size": int(arrays["sample_order"].size),
        "sample_order_start": int(arrays["sample_order"][0]),
        "sample_order_stop_exclusive": int(arrays["sample_order"][-1]) + 1,
        "feature_count": EXPECTED_FEATURES,
        "shap_version": shap.__version__,
        "positive_class": POSITIVE_CLASS,
        "positive_class_index": runtime["class_mapping"]["positive_class_index"],
        "feature_perturbation": FEATURE_PERTURBATION,
        "model_output": MODEL_OUTPUT,
        "approximate": APPROXIMATE,
        "additivity_tolerance": ADDITIVITY_TOLERANCE,
        "maximum_additivity_error": float(
            arrays["absolute_additivity_error"].max()
        ),
        "input_fingerprint_sha256": input_fingerprint,
        "feature_names_sha256": feature_names_sha256,
        "checkpoint_npz": _relative(npz_path),
        "checkpoint_npz_sha256": _sha256(npz_path),
        "runtime": {
            key: runtime[key]
            for key in (
                "batch_number",
                "batch_size",
                "start_time",
                "elapsed_seconds",
                "seconds_per_case",
            )
        },
        "class_mapping": runtime["class_mapping"],
        "validation_status": "PASS",
    }
    try:
        _write_text_atomic(json_path, json.dumps(metadata, indent=2) + "\n")
    except BaseException:
        npz_path.unlink(missing_ok=True)
        raise


def validate_existing_checkpoint(
    batch_number: int,
    sample: pd.DataFrame,
    expected_probabilities: np.ndarray,
    input_fingerprint: str,
    feature_names_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fully validate an existing checkpoint before allowing resume to skip it."""
    npz_path, json_path = _checkpoint_paths(batch_number)
    metadata = _load_json(json_path)
    expected_metadata = {
        "experiment": "shap_pilot_v1",
        "stage": "part_3_full_tree_shap_batch",
        "batch_number": batch_number,
        "batch_size": len(sample),
        "sample_order_start": int(sample["sample_order"].iloc[0]),
        "sample_order_stop_exclusive": int(sample["sample_order"].iloc[-1]) + 1,
        "feature_count": EXPECTED_FEATURES,
        "shap_version": EXPECTED_SHAP_VERSION,
        "positive_class": POSITIVE_CLASS,
        "positive_class_index": 1,
        "feature_perturbation": FEATURE_PERTURBATION,
        "model_output": MODEL_OUTPUT,
        "approximate": APPROXIMATE,
        "additivity_tolerance": ADDITIVITY_TOLERANCE,
        "input_fingerprint_sha256": input_fingerprint,
        "feature_names_sha256": feature_names_sha256,
        "checkpoint_npz": _relative(npz_path),
        "validation_status": "PASS",
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise AssertionError(
                f"Checkpoint {batch_number:03d} metadata mismatch for {key}: "
                f"expected {expected!r}, found {metadata.get(key)!r}"
            )
    if metadata.get("checkpoint_npz_sha256") != _sha256(npz_path):
        raise AssertionError(f"Checkpoint {batch_number:03d} NPZ hash mismatch")

    required_arrays = {
        "sample_order",
        "source_row_index",
        "case_id",
        "target",
        "model_probability_success",
        "base_value_success",
        "shap_values_success",
        "reconstructed_probability_success",
        "absolute_additivity_error",
    }
    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != required_arrays:
            raise AssertionError(
                f"Checkpoint {batch_number:03d} array schema mismatch"
            )
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    identity_checks = {
        "sample_order": sample["sample_order"].to_numpy(dtype=np.int64),
        "source_row_index": sample["source_row_index"].to_numpy(dtype=np.int64),
        "case_id": sample["case_id"].astype(str).to_numpy(dtype=str),
        "target": sample["target"].to_numpy(dtype=np.int64),
    }
    for name, expected in identity_checks.items():
        if not np.array_equal(arrays[name], expected):
            raise AssertionError(
                f"Checkpoint {batch_number:03d} identity mismatch in {name}"
            )
    if not np.allclose(
        arrays["model_probability_success"],
        expected_probabilities,
        rtol=0.0,
        atol=1e-15,
    ):
        raise AssertionError(
            f"Checkpoint {batch_number:03d} model probabilities changed"
        )
    reconstructed, errors = validate_additivity(
        sample,
        arrays["shap_values_success"],
        arrays["base_value_success"],
        arrays["model_probability_success"],
    )
    if not np.allclose(
        arrays["reconstructed_probability_success"],
        reconstructed,
        rtol=0.0,
        atol=1e-15,
    ):
        raise AssertionError(
            f"Checkpoint {batch_number:03d} reconstruction mismatch"
        )
    if not np.allclose(
        arrays["absolute_additivity_error"], errors, rtol=0.0, atol=1e-15
    ):
        raise AssertionError(f"Checkpoint {batch_number:03d} error mismatch")
    if not np.isclose(
        metadata.get("maximum_additivity_error"),
        float(errors.max()),
        rtol=0.0,
        atol=1e-15,
    ):
        raise AssertionError(
            f"Checkpoint {batch_number:03d} maximum-error metadata mismatch"
        )
    runtime = metadata.get("runtime")
    if not isinstance(runtime, dict):
        raise AssertionError(f"Checkpoint {batch_number:03d} lacks runtime metadata")
    for name in RUNTIME_COLUMNS:
        if name not in runtime:
            raise AssertionError(
                f"Checkpoint {batch_number:03d} runtime lacks {name}"
            )
    if runtime["batch_number"] != batch_number or runtime["batch_size"] != len(sample):
        raise AssertionError(f"Checkpoint {batch_number:03d} runtime identity mismatch")
    elapsed = float(runtime["elapsed_seconds"])
    seconds_per_case = float(runtime["seconds_per_case"])
    if not np.isfinite(elapsed) or elapsed <= 0.0:
        raise AssertionError(f"Checkpoint {batch_number:03d} elapsed time is invalid")
    if not np.isclose(
        seconds_per_case, elapsed / len(sample), rtol=0.0, atol=1e-12
    ):
        raise AssertionError(
            f"Checkpoint {batch_number:03d} seconds-per-case mismatch"
        )
    return arrays, runtime


def assemble_canonical_output(
    batches: list[tuple[dict[str, np.ndarray], dict[str, Any]]],
    explained: pd.DataFrame,
) -> dict[str, Any]:
    """Assemble, validate, and atomically write canonical Part 3 outputs."""
    arrays_by_name = {
        name: np.concatenate([batch[0][name] for batch in batches], axis=0)
        for name in (
            "sample_order",
            "source_row_index",
            "case_id",
            "target",
            "model_probability_success",
            "base_value_success",
            "shap_values_success",
            "reconstructed_probability_success",
            "absolute_additivity_error",
        )
    }
    shap_values = arrays_by_name["shap_values_success"]
    base_values = arrays_by_name["base_value_success"]
    probabilities = arrays_by_name["model_probability_success"]
    reconstructed = arrays_by_name["reconstructed_probability_success"]
    errors = arrays_by_name["absolute_additivity_error"]
    if shap_values.shape != (EXPECTED_EXPLAINED_CASES, EXPECTED_FEATURES):
        raise AssertionError(f"Unexpected full SHAP shape: {shap_values.shape}")
    if base_values.shape != (EXPECTED_EXPLAINED_CASES,):
        raise AssertionError(f"Unexpected full base shape: {base_values.shape}")
    if probabilities.shape != (EXPECTED_EXPLAINED_CASES,):
        raise AssertionError(f"Unexpected full probability shape: {probabilities.shape}")
    if not np.array_equal(
        arrays_by_name["sample_order"], np.arange(EXPECTED_EXPLAINED_CASES)
    ):
        raise AssertionError("Full sample_order is not exactly 0 through 999")
    expected_identity = {
        "source_row_index": explained["source_row_index"].to_numpy(dtype=np.int64),
        "case_id": explained["case_id"].astype(str).to_numpy(dtype=str),
        "target": explained["target"].to_numpy(dtype=np.int64),
    }
    for name, expected in expected_identity.items():
        if not np.array_equal(arrays_by_name[name], expected):
            raise AssertionError(f"Full assembled {name} differs from frozen sample")
    if len(np.unique(arrays_by_name["sample_order"])) != EXPECTED_EXPLAINED_CASES:
        raise AssertionError("Duplicate full sample_order values found")
    if len(np.unique(arrays_by_name["case_id"])) != EXPECTED_EXPLAINED_CASES:
        raise AssertionError("Duplicate full case IDs found")
    for label, values in (
        ("SHAP values", shap_values),
        ("base values", base_values),
        ("model probabilities", probabilities),
        ("reconstructed probabilities", reconstructed),
        ("additivity errors", errors),
    ):
        if not np.isfinite(values).all():
            raise AssertionError(f"Full {label} contain NaN or infinity")
    _validate_probabilities(probabilities, "full model probabilities")
    _validate_probabilities(reconstructed, "full reconstructed probabilities")
    if float(errors.max()) > ADDITIVITY_TOLERANCE:
        raise AssertionError("Full maximum additivity error exceeds 1e-5")

    metadata = pd.DataFrame(
        {
            "sample_order": arrays_by_name["sample_order"],
            "source_row_index": arrays_by_name["source_row_index"],
            "case_id": arrays_by_name["case_id"],
            "target": arrays_by_name["target"],
            "model_probability_success": probabilities,
            "reconstructed_probability_success": reconstructed,
            "absolute_additivity_error": errors,
        },
        columns=METADATA_COLUMNS,
    )
    if len(metadata) != EXPECTED_EXPLAINED_CASES:
        raise AssertionError("Full metadata row count is not 1,000")
    if not np.array_equal(
        metadata["case_id"].astype(str).to_numpy(),
        explained["case_id"].astype(str).to_numpy(),
    ):
        raise AssertionError("Full metadata case order differs from explained IDs")
    if not np.array_equal(metadata["target"], explained["target"]):
        raise AssertionError("Full metadata targets differ from explained IDs")

    runtime = pd.DataFrame([batch[1] for batch in batches], columns=RUNTIME_COLUMNS)
    if len(runtime) != EXPECTED_BATCH_COUNT:
        raise AssertionError("Runtime table does not contain exactly 20 batches")
    if not np.array_equal(runtime["batch_number"], np.arange(EXPECTED_BATCH_COUNT)):
        raise AssertionError("Runtime batches are not in order 0 through 19")
    if not (runtime["batch_size"] == BATCH_SIZE).all():
        raise AssertionError("Runtime table contains a non-50-case batch")

    _write_npy_atomic(SHAP_VALUES_PATH, shap_values)
    _write_npy_atomic(BASE_VALUES_PATH, base_values)
    _write_npy_atomic(PROBABILITIES_PATH, probabilities)
    _write_text_atomic(FULL_METADATA_PATH, _csv_text(metadata))
    _write_text_atomic(RUNTIME_PATH, _csv_text(runtime))

    error_summary = {
        "maximum": float(errors.max()),
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "minimum": float(errors.min()),
        "count_above_1e_6": int(np.count_nonzero(errors > 1e-6)),
        "count_above_5e_6": int(np.count_nonzero(errors > 5e-6)),
        "count_above_1e_5": int(np.count_nonzero(errors > 1e-5)),
    }
    numerical = {
        "minimum": float(shap_values.min()),
        "maximum": float(shap_values.max()),
        "overall_mean_absolute_value": float(np.abs(shap_values).mean()),
        "nan_count": int(np.isnan(shap_values).sum()),
        "positive_infinity_count": int(np.isposinf(shap_values).sum()),
        "negative_infinity_count": int(np.isneginf(shap_values).sum()),
        "all_finite": bool(np.isfinite(shap_values).all()),
    }
    total_seconds = float(runtime["elapsed_seconds"].sum())
    fastest_index = int(runtime["seconds_per_case"].idxmin())
    slowest_index = int(runtime["seconds_per_case"].idxmax())
    runtime_summary = {
        "total_shap_computation_seconds": total_seconds,
        "average_seconds_per_case": total_seconds / EXPECTED_EXPLAINED_CASES,
        "fastest_batch": runtime.loc[fastest_index].to_dict(),
        "slowest_batch": runtime.loc[slowest_index].to_dict(),
        "scientific_runtime_source": "successful validated checkpoint batches only",
    }
    hashes = {
        "shap_values_success_npy_sha256": _sha256(SHAP_VALUES_PATH),
        "base_values_success_npy_sha256": _sha256(BASE_VALUES_PATH),
        "model_probabilities_success_npy_sha256": _sha256(PROBABILITIES_PATH),
        "full_shap_case_metadata_csv_sha256": _sha256(FULL_METADATA_PATH),
        "shap_values_success_array_sha256": _array_sha256(shap_values),
        "base_values_success_array_sha256": _array_sha256(base_values),
        "model_probabilities_success_array_sha256": _array_sha256(probabilities),
    }
    return {
        "shap_values": shap_values,
        "base_values": base_values,
        "probabilities": probabilities,
        "metadata": metadata,
        "runtime": runtime,
        "error_summary": error_summary,
        "numerical": numerical,
        "runtime_summary": runtime_summary,
        "hashes": hashes,
    }


def _json_safe(value: Any) -> Any:
    """Convert NumPy scalar values recursively for JSON output."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_manifest(
    inputs: dict[str, Any],
    assembled: dict[str, Any],
    computed_batches: list[int],
    skipped_batches: list[int],
    protected_hashes_after: dict[Path, str],
) -> dict[str, Any]:
    """Write the canonical full-computation manifest with measured values."""
    if protected_hashes_after != inputs["protected_hashes"]:
        raise AssertionError("A protected frozen input changed during Part 3")
    errors = assembled["error_summary"]
    numerical = assembled["numerical"]
    assertions = {
        "model_artifact_unchanged": "PASS",
        "background_ids_unchanged": "PASS",
        "explained_ids_unchanged": "PASS",
        "part_0_through_2b_artifacts_unchanged": "PASS",
        "background_shape_500_by_165": "PASS",
        "explained_shape_1000_by_165": "PASS",
        "dense_conversion_preserves_all_values": "PASS",
        "model_feature_count_165": "PASS",
        "positive_class_is_1": "PASS",
        "shap_version_0_52_0": "PASS",
        "tree_explainer_configuration_matches_part_2": "PASS",
        "approximate_false": "PASS",
        "batch_size_50": "PASS",
        "batch_count_20": "PASS",
        "every_batch_follows_frozen_sample_order": "PASS",
        "every_batch_has_correct_case_ids": "PASS",
        "all_shap_values_finite": "PASS",
        "every_case_additivity_error_at_most_1e_5": "PASS",
        "full_matrix_shape_1000_by_165": "PASS",
        "full_metadata_rows_1000": "PASS",
        "no_missing_samples": "PASS",
        "no_duplicate_samples": "PASS",
        "feature_order_unchanged": "PASS",
        "no_model_fitting": "PASS",
        "no_preprocessing_fitting": "PASS",
        "no_resampling": "PASS",
        "no_global_ranking_created": "PASS",
        "no_feature_group_ranking_created": "PASS",
        "no_shap_plot_created": "PASS",
        "no_temporal_analysis": "PASS",
        "no_explanation_stability_calculation": "PASS",
    }
    manifest = {
        "experiment": "shap_pilot_v1",
        "stage": "part_3_full_tree_shap_computation",
        "environment": {
            "python_version": platform.python_version(),
            "shap_version": shap.__version__,
            "scikit_learn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "pandas_version": pd.__version__,
        },
        "shap_version": shap.__version__,
        "prediction_point": 10,
        "model": type(inputs["model"]).__name__,
        "model_path": _relative(MODEL_PATH),
        "positive_class": POSITIVE_CLASS,
        "positive_class_index": inputs["positive_class_index"],
        "feature_count": EXPECTED_FEATURES,
        "background_cases": EXPECTED_BACKGROUND_CASES,
        "explained_cases": EXPECTED_EXPLAINED_CASES,
        "background_shape": list(inputs["background_dense"].shape),
        "explained_shape": list(inputs["explained_dense"].shape),
        "batch_size": BATCH_SIZE,
        "batch_count": EXPECTED_BATCH_COUNT,
        "completed_batches": EXPECTED_BATCH_COUNT,
        "computed_batches_this_invocation": computed_batches,
        "resumed_skipped_batches_this_invocation": skipped_batches,
        "feature_perturbation": FEATURE_PERTURBATION,
        "model_output": MODEL_OUTPUT,
        "approximate": APPROXIMATE,
        "additivity_tolerance": ADDITIVITY_TOLERANCE,
        "full_shap_shape": list(assembled["shap_values"].shape),
        "base_values_shape": list(assembled["base_values"].shape),
        "model_probabilities_shape": list(assembled["probabilities"].shape),
        "metadata_rows": len(assembled["metadata"]),
        "maximum_additivity_error": errors["maximum"],
        "mean_additivity_error": errors["mean"],
        "median_additivity_error": errors["median"],
        "minimum_additivity_error": errors["minimum"],
        "additivity_error_threshold_counts": {
            "above_1e_6": errors["count_above_1e_6"],
            "above_5e_6": errors["count_above_5e_6"],
            "above_1e_5": errors["count_above_1e_5"],
        },
        "shap_numerical_summary": numerical,
        "numerical_finiteness_result": "PASS" if numerical["all_finite"] else "FAIL",
        "runtime_summary": assembled["runtime_summary"],
        "resume_support_enabled": True,
        "resume_validation_result": "PASS",
        "model_retrained": False,
        "preprocessing_refitted": False,
        "background_changed": False,
        "explained_sample_changed": False,
        "resampled": False,
        "global_ranking_created": False,
        "feature_group_ranking_created": False,
        "plot_created": False,
        "temporal_analysis_performed": False,
        "stability_calculated": False,
        "canonical_artifact_hashes": assembled["hashes"],
        "protected_artifacts": {
            "count": len(protected_hashes_after),
            "unchanged": True,
            "input_fingerprint_sha256": inputs["input_fingerprint"],
            "sha256": {
                _relative(path): digest
                for path, digest in sorted(
                    protected_hashes_after.items(), key=lambda item: str(item[0])
                )
            },
        },
        "outputs": {
            "shap_values_success": _relative(SHAP_VALUES_PATH),
            "base_values_success": _relative(BASE_VALUES_PATH),
            "model_probabilities_success": _relative(PROBABILITIES_PATH),
            "full_case_metadata": _relative(FULL_METADATA_PATH),
            "batch_runtime": _relative(RUNTIME_PATH),
            "full_manifest": _relative(FULL_MANIFEST_PATH),
            "checkpoint_directory": _relative(BATCH_DIR),
        },
        "assertions": assertions,
        "assertion_status": "PASS",
    }
    manifest = _json_safe(manifest)
    _write_text_atomic(FULL_MANIFEST_PATH, json.dumps(manifest, indent=2) + "\n")
    return manifest


def run_full_shap_pilot() -> dict[str, Any]:
    """Run or resume Part 3 and return its completed manifest."""
    inputs = load_frozen_inputs()
    batches = identify_batches()
    _check_checkpoint_directory(batches)
    completed: list[tuple[dict[str, np.ndarray], dict[str, Any]]] = []
    computed_batches: list[int] = []
    skipped_batches: list[int] = []
    explainer: shap.TreeExplainer | None = None

    for batch_number, start, stop in batches:
        sample = inputs["explained"].iloc[start:stop].reset_index(drop=True)
        expected_orders = np.arange(start, stop)
        if not np.array_equal(sample["sample_order"], expected_orders):
            raise AssertionError(
                f"Batch {batch_number:03d} does not follow frozen sample_order"
            )
        batch_probabilities = inputs["success_probabilities"][start:stop]
        npz_path, json_path = _checkpoint_paths(batch_number)
        if npz_path.exists() and json_path.exists():
            arrays, runtime = validate_existing_checkpoint(
                batch_number,
                sample,
                batch_probabilities,
                inputs["input_fingerprint"],
                inputs["feature_names_sha256"],
            )
            skipped_batches.append(batch_number)
            print(
                f"Batch {batch_number:03d}: validated checkpoint; skipped SHAP",
                flush=True,
            )
        else:
            if explainer is None:
                explainer = build_explainer(
                    inputs["model"],
                    inputs["background_dense"],
                    inputs["feature_names"],
                )
            arrays, runtime = compute_one_batch(
                explainer,
                batch_number,
                sample,
                inputs["explained_dense"][start:stop],
                batch_probabilities,
                inputs["positive_class_index"],
                len(inputs["classes"]),
                inputs["feature_names"],
            )
            save_one_batch(
                batch_number,
                arrays,
                runtime,
                inputs["input_fingerprint"],
                inputs["feature_names_sha256"],
            )
            computed_batches.append(batch_number)
            print(
                f"Batch {batch_number:03d}: computed and checkpointed "
                f"({runtime['elapsed_seconds']:.3f} seconds, "
                f"max error {arrays['absolute_additivity_error'].max():.3e})",
                flush=True,
            )
        completed.append((arrays, runtime))

    assembled = assemble_canonical_output(completed, inputs["explained"])
    protected_after = _artifact_hashes(_protected_paths())
    manifest = write_manifest(
        inputs,
        assembled,
        computed_batches,
        skipped_batches,
        protected_after,
    )
    return manifest


def print_report(manifest: dict[str, Any]) -> None:
    """Print the Part 3 completion report."""
    thresholds = manifest["additivity_error_threshold_counts"]
    numerical = manifest["shap_numerical_summary"]
    runtime = manifest["runtime_summary"]
    print("SHAP Pilot V1 Part 3: PASS")
    print(f"SHAP version: {manifest['shap_version']}")
    print(f"Background shape: {manifest['background_shape']}")
    print(f"Explained shape: {manifest['explained_shape']}")
    print(f"Batch size/count: {manifest['batch_size']}/{manifest['batch_count']}")
    print(f"Completed batches: {manifest['completed_batches']}")
    print(f"Computed this invocation: {manifest['computed_batches_this_invocation']}")
    print(
        "Resumed/skipped this invocation: "
        f"{manifest['resumed_skipped_batches_this_invocation']}"
    )
    print(f"Full SHAP shape: {manifest['full_shap_shape']}")
    print(f"Base-value shape: {manifest['base_values_shape']}")
    print(f"Probability shape: {manifest['model_probabilities_shape']}")
    print(
        "Additivity error min/mean/median/max: "
        f"{manifest['minimum_additivity_error']:.17g} / "
        f"{manifest['mean_additivity_error']:.17g} / "
        f"{manifest['median_additivity_error']:.17g} / "
        f"{manifest['maximum_additivity_error']:.17g}"
    )
    print(
        "Error counts above 1e-6 / 5e-6 / 1e-5: "
        f"{thresholds['above_1e_6']} / {thresholds['above_5e_6']} / "
        f"{thresholds['above_1e_5']}"
    )
    print(
        "SHAP min/max/overall mean absolute: "
        f"{numerical['minimum']:.17g} / {numerical['maximum']:.17g} / "
        f"{numerical['overall_mean_absolute_value']:.17g}"
    )
    print(
        "SHAP NaN/+inf/-inf: "
        f"{numerical['nan_count']}/{numerical['positive_infinity_count']}/"
        f"{numerical['negative_infinity_count']}"
    )
    print(
        "Successful SHAP runtime total/seconds per case: "
        f"{runtime['total_shap_computation_seconds']:.3f} / "
        f"{runtime['average_seconds_per_case']:.6f}"
    )
    print(
        f"Fastest batch: {int(runtime['fastest_batch']['batch_number']):03d}; "
        f"slowest batch: {int(runtime['slowest_batch']['batch_number']):03d}"
    )
    print(f"Canonical hashes: {manifest['canonical_artifact_hashes']}")
    print(
        "Protected artifacts: "
        f"{manifest['protected_artifacts']['count']} unchanged"
    )


def main() -> None:
    """CLI entry point."""
    manifest = run_full_shap_pilot()
    print_report(manifest)


if __name__ == "__main__":
    main()
