"""Freeze deterministic input populations for SHAP Pilot V1 Part 1.

This script samples canonical TRAIN and TEST row positions without using
targets, feature values, or predictions. It extracts and validates the
corresponding TREE-matrix rows, audits the frozen Random Forest on the fixed
explained sample, and writes only case-ID tables plus a sampling manifest.
It does not fit preprocessing, fit a model, import SHAP, or execute SHAP.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_RESULTS_DIR = PROJECT_ROOT / "results" / "baseline_v1"
BASELINE_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "baseline_k10"
BASELINE_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "baseline_v1"
PILOT_RESULTS_DIR = PROJECT_ROOT / "results" / "shap_pilot_v1"
PILOT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "shap_pilot_v1"

TRAIN_IDS_PATH = BASELINE_RESULTS_DIR / "train_case_ids.csv"
TEST_IDS_PATH = BASELINE_RESULTS_DIR / "test_case_ids.csv"
FEATURE_NAMES_PATH = BASELINE_RESULTS_DIR / "feature_names_tree.csv"
X_TRAIN_PATH = BASELINE_PROCESSED_DIR / "X_train_tree.npz"
X_TEST_PATH = BASELINE_PROCESSED_DIR / "X_test_tree.npz"
Y_TRAIN_PATH = BASELINE_PROCESSED_DIR / "y_train.npy"
Y_TEST_PATH = BASELINE_PROCESSED_DIR / "y_test.npy"
MODEL_PATH = PILOT_ARTIFACTS_DIR / "random_forest_baseline.joblib"
MODEL_FREEZE_MANIFEST_PATH = PILOT_RESULTS_DIR / "model_freeze_manifest.json"

BACKGROUND_IDS_PATH = PILOT_RESULTS_DIR / "background_case_ids.csv"
EXPLAINED_IDS_PATH = PILOT_RESULTS_DIR / "explained_case_ids.csv"
SAMPLE_MANIFEST_PATH = PILOT_RESULTS_DIR / "sample_manifest.json"

EXPECTED_TRAIN_CASES = 25_100
EXPECTED_TEST_CASES = 6_276
EXPECTED_FEATURES = 165
EXPECTED_TARGET_COUNTS = {
    "train": {0: 11_322, 1: 13_778},
    "test": {0: 2_826, 1: 3_450},
}
BACKGROUND_SAMPLE_SIZE = 500
EXPLAINED_SAMPLE_SIZE = 1_000
RANDOM_SEED = 42
POSITIVE_CLASS = 1


def _relative(path: Path) -> str:
    """Return a repository-relative path with stable separators."""
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


def _protected_paths() -> list[Path]:
    """Collect frozen Baseline V1 and SHAP Part 0 artifacts for hashing."""
    roots = [BASELINE_RESULTS_DIR, BASELINE_PROCESSED_DIR, BASELINE_ARTIFACTS_DIR]
    paths = {
        path
        for root in roots
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_file()
    }
    paths.update({MODEL_PATH, MODEL_FREEZE_MANIFEST_PATH})
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Protected artifacts are missing: {missing}")
    return sorted(paths, key=_relative)


def _artifact_hashes(paths: list[Path]) -> dict[Path, str]:
    """Hash protected artifacts without modifying them."""
    return {path: _sha256(path) for path in paths}


def load_canonical_split(
    ids_path: Path,
    matrix_path: Path,
    target_path: Path,
) -> tuple[pd.Series, sparse.csr_matrix, np.ndarray]:
    """Load an ordered canonical ID list and its aligned frozen arrays."""
    ids_frame = pd.read_csv(ids_path, encoding="utf-8", dtype={"case_id": "string"})
    if ids_frame.columns.tolist() != ["case_id"]:
        raise AssertionError(
            f"Unexpected canonical ID columns in {ids_path}: "
            f"{ids_frame.columns.tolist()}"
        )
    case_ids = ids_frame["case_id"]
    if case_ids.isna().any() or not case_ids.is_unique:
        raise AssertionError(f"Canonical case IDs must be non-null and unique: {ids_path}")

    matrix = sparse.load_npz(matrix_path)
    if not sparse.isspmatrix_csr(matrix):
        raise AssertionError(f"Expected a CSR TREE matrix: {matrix_path}")
    target = np.load(target_path, allow_pickle=False)
    return case_ids, matrix, target


def _target_counts(target: np.ndarray) -> dict[int, int]:
    """Return integer counts for all labels in a target vector."""
    labels, counts = np.unique(target, return_counts=True)
    return {
        int(label): int(count)
        for label, count in zip(labels, counts, strict=True)
    }


def validate_alignment(
    split_name: str,
    case_ids: pd.Series,
    matrix: sparse.csr_matrix,
    target: np.ndarray,
    expected_rows: int,
) -> None:
    """Validate the canonical row alignment contract before sampling."""
    if len(case_ids) != expected_rows:
        raise AssertionError(
            f"Unexpected {split_name} case-ID count: {len(case_ids)}"
        )
    if matrix.shape != (expected_rows, EXPECTED_FEATURES):
        raise AssertionError(f"Unexpected {split_name} TREE shape: {matrix.shape}")
    if target.shape != (expected_rows,) or target.ndim != 1:
        raise AssertionError(f"Unexpected {split_name} target shape: {target.shape}")
    if _target_counts(target) != EXPECTED_TARGET_COUNTS[split_name]:
        raise AssertionError(f"Unexpected {split_name} target distribution")
    if not np.isfinite(matrix.data).all():
        raise AssertionError(f"{split_name} TREE matrix contains NaN or infinity")
    if not np.isfinite(target).all():
        raise AssertionError(f"{split_name} target contains NaN or infinity")
    if set(np.unique(target).tolist()) != {0, 1}:
        raise AssertionError(f"{split_name} target labels differ from {{0, 1}}")


def load_and_validate_feature_names() -> pd.DataFrame:
    """Load ordered TREE feature metadata and validate its frozen identity."""
    frame = pd.read_csv(FEATURE_NAMES_PATH, encoding="utf-8")
    expected_columns = ["feature_index", "feature_name", "source_group"]
    if frame.columns.tolist() != expected_columns:
        raise AssertionError(
            f"Unexpected feature-name columns: {frame.columns.tolist()}"
        )
    if len(frame) != EXPECTED_FEATURES:
        raise AssertionError(f"Unexpected feature-name count: {len(frame)}")
    if frame["feature_index"].tolist() != list(range(EXPECTED_FEATURES)):
        raise AssertionError("Feature indices are not contiguous and ordered")
    if frame["feature_name"].isna().any() or not frame["feature_name"].is_unique:
        raise AssertionError("TREE feature names must be non-null and unique")
    return frame


def deterministic_sample_indices(
    population_size: int,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    """Sample ordered row positions uniformly without replacement."""
    if sample_size > population_size:
        raise ValueError("Sample size cannot exceed population size")
    rng = np.random.default_rng(seed)
    indices = rng.choice(population_size, size=sample_size, replace=False)
    if indices.shape != (sample_size,) or np.unique(indices).size != sample_size:
        raise AssertionError("Sampling did not produce the required unique indices")
    if (indices < 0).any() or (indices >= population_size).any():
        raise AssertionError("Sampled source row index is out of bounds")
    return indices


def extract_sample(
    case_ids: pd.Series,
    matrix: sparse.csr_matrix,
    target: np.ndarray,
    source_row_indices: np.ndarray,
) -> tuple[pd.DataFrame, sparse.csr_matrix]:
    """Extract an ID/target table and pure matrix-row subset in sample order."""
    sampled_ids = case_ids.iloc[source_row_indices].astype(str).to_numpy()
    sampled_target = target[source_row_indices]
    sample_frame = pd.DataFrame(
        {
            "sample_order": np.arange(source_row_indices.size, dtype=np.int64),
            "source_row_index": source_row_indices,
            "case_id": sampled_ids,
            "target": sampled_target.astype(np.int8, copy=False),
        }
    )
    sample_matrix = matrix[source_row_indices, :].tocsr(copy=True)
    return sample_frame, sample_matrix


def validate_sample_identity(
    sample_name: str,
    sample_frame: pd.DataFrame,
    sample_matrix: sparse.csr_matrix,
    canonical_ids: pd.Series,
    canonical_target: np.ndarray,
    expected_size: int,
) -> None:
    """Validate exact positional identity for one sampled population."""
    expected_columns = ["sample_order", "source_row_index", "case_id", "target"]
    if sample_frame.columns.tolist() != expected_columns:
        raise AssertionError(f"Unexpected {sample_name} output columns")
    if len(sample_frame) != expected_size:
        raise AssertionError(f"Unexpected {sample_name} sample size")
    if sample_frame["sample_order"].tolist() != list(range(expected_size)):
        raise AssertionError(f"Unexpected {sample_name} sample order")
    if not sample_frame["source_row_index"].is_unique:
        raise AssertionError(f"{sample_name} source row indices are not unique")
    indices = sample_frame["source_row_index"].to_numpy(dtype=np.int64)
    if (indices < 0).any() or (indices >= len(canonical_ids)).any():
        raise AssertionError(f"{sample_name} source row index is out of bounds")
    expected_ids = canonical_ids.iloc[indices].astype(str).to_numpy()
    if not np.array_equal(sample_frame["case_id"].to_numpy(), expected_ids):
        raise AssertionError(f"{sample_name} case IDs do not match source rows")
    if not np.array_equal(sample_frame["target"].to_numpy(), canonical_target[indices]):
        raise AssertionError(f"{sample_name} targets do not match source rows")
    if sample_frame["case_id"].isna().any() or not sample_frame["case_id"].is_unique:
        raise AssertionError(f"{sample_name} case IDs must be non-null and unique")
    expected_shape = (expected_size, EXPECTED_FEATURES)
    if sample_matrix.shape != expected_shape:
        raise AssertionError(
            f"Unexpected {sample_name} matrix shape: {sample_matrix.shape}"
        )
    if not np.isfinite(sample_matrix.data).all():
        raise AssertionError(f"{sample_name} matrix contains NaN or infinity")


def _distribution(target: np.ndarray) -> dict[str, Any]:
    """Return the requested binary target counts and rates."""
    counts = _target_counts(target)
    total = int(target.size)
    success = counts.get(1, 0)
    unsuccessful = counts.get(0, 0)
    return {
        "success_count": success,
        "unsuccessful_count": unsuccessful,
        "success_rate": success / total,
        "unsuccessful_rate": unsuccessful / total,
    }


def audit_explained_predictions(
    model: RandomForestClassifier,
    matrix: sparse.csr_matrix,
    target: np.ndarray,
) -> dict[str, Any]:
    """Characterize predictions on the already-frozen explained sample."""
    if not hasattr(model, "classes_") or not hasattr(model, "n_features_in_"):
        raise AssertionError("Frozen model is not fitted")
    if int(model.n_features_in_) != EXPECTED_FEATURES:
        raise AssertionError(
            f"Frozen model expects {model.n_features_in_} features, not 165"
        )
    classes = np.asarray(model.classes_)
    positive_columns = np.flatnonzero(classes == POSITIVE_CLASS)
    if positive_columns.size != 1:
        raise AssertionError("Class 1 has no unique predict_proba column")
    positive_column = int(positive_columns[0])

    y_pred = np.asarray(model.predict(matrix))
    probabilities = np.asarray(model.predict_proba(matrix))
    y_prob_success = probabilities[:, positive_column]
    if y_pred.shape != target.shape or y_prob_success.shape != target.shape:
        raise AssertionError("Explained-sample prediction shapes are invalid")
    if not np.isfinite(y_prob_success).all():
        raise AssertionError("Explained probabilities contain NaN or infinity")
    if ((y_prob_success < 0) | (y_prob_success > 1)).any():
        raise AssertionError("Explained probabilities fall outside [0, 1]")

    confusion = {
        "tp": int(np.sum((target == 1) & (y_pred == 1))),
        "tn": int(np.sum((target == 0) & (y_pred == 0))),
        "fp": int(np.sum((target == 0) & (y_pred == 1))),
        "fn": int(np.sum((target == 1) & (y_pred == 0))),
    }
    if sum(confusion.values()) != target.size:
        raise AssertionError("Confusion counts do not cover the explained sample")
    coverage = {name: count > 0 for name, count in confusion.items()}
    return {
        "positive_class": POSITIVE_CLASS,
        "model_classes": [int(value) for value in classes],
        "positive_probability_column_index": positive_column,
        "confusion_counts": confusion,
        "category_coverage": coverage,
        "probability_summary": {
            "minimum": float(y_prob_success.min()),
            "mean": float(y_prob_success.mean()),
            "maximum": float(y_prob_success.max()),
        },
    }


def _sparse_exact(left: sparse.csr_matrix, right: sparse.csr_matrix) -> bool:
    """Return whether two CSR matrices have identical structure and data."""
    return (
        left.shape == right.shape
        and np.array_equal(left.indptr, right.indptr)
        and np.array_equal(left.indices, right.indices)
        and np.array_equal(left.data, right.data)
    )


def _csv_text(frame: pd.DataFrame) -> str:
    """Serialize a sample table deterministically."""
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue()


def _write_text_atomic(path: Path, content: str) -> None:
    """Write text through a sibling temporary file and atomically replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    try:
        temporary_path.write_text(content, encoding="utf-8", newline="")
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_sample_manifest(manifest: dict[str, Any]) -> None:
    """Serialize the deterministic sampling manifest."""
    content = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    _write_text_atomic(SAMPLE_MANIFEST_PATH, content)


def _assert_no_split_leakage(
    train_ids: pd.Series,
    test_ids: pd.Series,
    background: pd.DataFrame,
    explained: pd.DataFrame,
) -> dict[str, int]:
    """Assert membership, exclusion, uniqueness, and cross-sample isolation."""
    train_set = set(train_ids.astype(str))
    test_set = set(test_ids.astype(str))
    background_set = set(background["case_id"])
    explained_set = set(explained["case_id"])
    if not train_set.isdisjoint(test_set):
        raise AssertionError("Canonical TRAIN and TEST case IDs overlap")
    if not background_set.issubset(train_set):
        raise AssertionError("A background case is outside canonical TRAIN")
    if not background_set.isdisjoint(test_set):
        raise AssertionError("A background case belongs to canonical TEST")
    if not explained_set.issubset(test_set):
        raise AssertionError("An explained case is outside canonical TEST")
    if not explained_set.isdisjoint(train_set):
        raise AssertionError("An explained case belongs to canonical TRAIN")
    if not background_set.isdisjoint(explained_set):
        raise AssertionError("Background and explained samples overlap")
    return {
        "canonical_train_test_overlap": len(train_set & test_set),
        "background_test_overlap": len(background_set & test_set),
        "explained_train_overlap": len(explained_set & train_set),
        "background_explained_overlap": len(background_set & explained_set),
    }


def prepare_shap_samples() -> dict[str, Any]:
    """Create and validate the frozen SHAP Pilot V1 Part 1 samples."""
    if any(name == "shap" or name.startswith("shap.") for name in sys.modules):
        raise AssertionError("SHAP was unexpectedly imported before Part 1")

    protected_paths = _protected_paths()
    protected_hashes_before = _artifact_hashes(protected_paths)
    model_freeze_manifest = _load_json(MODEL_FREEZE_MANIFEST_PATH)
    if model_freeze_manifest.get("assertion_status") != "PASS":
        raise AssertionError("SHAP Part 0 manifest is not validated")
    if model_freeze_manifest.get("artifact_sha256") != _sha256(MODEL_PATH):
        raise AssertionError("Frozen model hash differs from the Part 0 manifest")

    train_ids, X_train, y_train = load_canonical_split(
        TRAIN_IDS_PATH, X_TRAIN_PATH, Y_TRAIN_PATH
    )
    test_ids, X_test, y_test = load_canonical_split(
        TEST_IDS_PATH, X_TEST_PATH, Y_TEST_PATH
    )
    validate_alignment(
        "train", train_ids, X_train, y_train, EXPECTED_TRAIN_CASES
    )
    validate_alignment("test", test_ids, X_test, y_test, EXPECTED_TEST_CASES)
    feature_names = load_and_validate_feature_names()
    if len(feature_names) != X_train.shape[1] or len(feature_names) != X_test.shape[1]:
        raise AssertionError("Feature-name count differs from TREE matrix columns")

    background_indices = deterministic_sample_indices(
        EXPECTED_TRAIN_CASES, BACKGROUND_SAMPLE_SIZE, RANDOM_SEED
    )
    explained_indices = deterministic_sample_indices(
        EXPECTED_TEST_CASES, EXPLAINED_SAMPLE_SIZE, RANDOM_SEED
    )
    background, X_background = extract_sample(
        train_ids, X_train, y_train, background_indices
    )
    explained, X_explained = extract_sample(
        test_ids, X_test, y_test, explained_indices
    )
    validate_sample_identity(
        "background",
        background,
        X_background,
        train_ids,
        y_train,
        BACKGROUND_SAMPLE_SIZE,
    )
    validate_sample_identity(
        "explained",
        explained,
        X_explained,
        test_ids,
        y_test,
        EXPLAINED_SAMPLE_SIZE,
    )
    overlap_checks = _assert_no_split_leakage(
        train_ids, test_ids, background, explained
    )

    model = joblib.load(MODEL_PATH)
    if not isinstance(model, RandomForestClassifier):
        raise AssertionError(f"Unexpected frozen model type: {type(model)!r}")
    if X_background.shape[1] != int(model.n_features_in_):
        raise AssertionError("Background feature dimension differs from frozen RF")
    if X_explained.shape[1] != int(model.n_features_in_):
        raise AssertionError("Explained feature dimension differs from frozen RF")
    prediction_audit = audit_explained_predictions(
        model,
        X_explained,
        explained["target"].to_numpy(),
    )

    # Repeat the full selection/extraction path from fresh RNG instances.
    background_indices_rerun = deterministic_sample_indices(
        EXPECTED_TRAIN_CASES, BACKGROUND_SAMPLE_SIZE, RANDOM_SEED
    )
    explained_indices_rerun = deterministic_sample_indices(
        EXPECTED_TEST_CASES, EXPLAINED_SAMPLE_SIZE, RANDOM_SEED
    )
    background_rerun, X_background_rerun = extract_sample(
        train_ids, X_train, y_train, background_indices_rerun
    )
    explained_rerun, X_explained_rerun = extract_sample(
        test_ids, X_test, y_test, explained_indices_rerun
    )
    background_csv = _csv_text(background)
    explained_csv = _csv_text(explained)
    background_csv_rerun = _csv_text(background_rerun)
    explained_csv_rerun = _csv_text(explained_rerun)
    determinism = {
        "method": "two independent numpy.random.default_rng(42) initializations per source",
        "background_indices_identical": bool(
            np.array_equal(background_indices, background_indices_rerun)
        ),
        "explained_indices_identical": bool(
            np.array_equal(explained_indices, explained_indices_rerun)
        ),
        "background_csv_bytes_identical": background_csv == background_csv_rerun,
        "explained_csv_bytes_identical": explained_csv == explained_csv_rerun,
        "background_matrix_rows_identical": _sparse_exact(
            X_background, X_background_rerun
        ),
        "explained_matrix_rows_identical": _sparse_exact(
            X_explained, X_explained_rerun
        ),
    }
    if not all(value for key, value in determinism.items() if key != "method"):
        raise AssertionError(f"Deterministic rerun failed: {determinism}")
    determinism["result"] = "PASS"

    if _artifact_hashes(protected_paths) != protected_hashes_before:
        raise AssertionError("A protected artifact changed before output writing")

    _write_text_atomic(BACKGROUND_IDS_PATH, background_csv)
    _write_text_atomic(EXPLAINED_IDS_PATH, explained_csv)

    background_target = background["target"].to_numpy()
    explained_target = explained["target"].to_numpy()
    assertions = {
        "A_every_background_case_belongs_to_train": "PASS",
        "B_no_background_case_belongs_to_test": "PASS",
        "C_every_explained_case_belongs_to_test": "PASS",
        "D_no_explained_case_belongs_to_train": "PASS",
        "E_background_ids_unique": "PASS",
        "F_explained_ids_unique": "PASS",
        "G_background_explained_overlap_zero": "PASS",
        "H_background_sample_size_500": "PASS",
        "I_explained_sample_size_1000": "PASS",
        "J_source_row_indices_valid": "PASS",
        "K_case_ids_match_canonical_source_rows": "PASS",
        "L_targets_match_canonical_source_rows": "PASS",
        "M_sampling_did_not_use_targets": "PASS",
        "N_sampling_did_not_use_predictions": "PASS",
        "O_sampling_did_not_use_feature_values": "PASS",
        "P_no_preprocessing_fitting": "PASS",
        "Q_no_model_fitting": "PASS",
        "R_no_shap_import_or_execution": "PASS",
        "feature_name_count_165": "PASS",
        "feature_names_unique": "PASS",
        "background_shape_500_by_165": "PASS",
        "explained_shape_1000_by_165": "PASS",
        "sample_matrices_finite": "PASS",
        "feature_columns_unchanged": "PASS",
        "sample_dimensions_match_frozen_model": "PASS",
        "protected_artifacts_unchanged": "PASS",
        "deterministic_rerun": "PASS",
    }
    manifest: dict[str, Any] = {
        "experiment": "shap_pilot_v1",
        "stage": "part_1_sampling",
        "prediction_point": 10,
        "feature_count": EXPECTED_FEATURES,
        "canonical_source_counts": {
            "train": EXPECTED_TRAIN_CASES,
            "test": EXPECTED_TEST_CASES,
        },
        "full_source_target_distribution": {
            "train": _distribution(y_train),
            "test": _distribution(y_test),
        },
        "background_source": "train",
        "background_sample_size": BACKGROUND_SAMPLE_SIZE,
        "background_sampling": "uniform_without_replacement",
        "background_seed": RANDOM_SEED,
        "explained_source": "test",
        "explained_sample_size": EXPLAINED_SAMPLE_SIZE,
        "explained_sampling": "uniform_without_replacement",
        "explained_seed": RANDOM_SEED,
        "rng": "numpy.random.default_rng",
        "stratified": False,
        "selection_inputs": ["population_size", "sample_size", "seed"],
        "selection_used_targets": False,
        "selection_used_predictions": False,
        "selection_used_features": False,
        "background_target_distribution": _distribution(background_target),
        "explained_target_distribution": _distribution(explained_target),
        "matrix_shapes": {
            "background": list(X_background.shape),
            "explained": list(X_explained.shape),
        },
        "matrix_subsets_saved": False,
        "overlap_checks": overlap_checks,
        "background_test_overlap": overlap_checks["background_test_overlap"],
        "prediction_audit": prediction_audit,
        "deterministic_rerun": determinism,
        "model_retrained": False,
        "preprocessing_refitted": False,
        "feature_space_changed": False,
        "shap_imported": False,
        "shap_executed": False,
        "protected_artifacts": {
            "count": len(protected_paths),
            "unchanged": True,
            "sha256": {
                _relative(path): digest
                for path, digest in protected_hashes_before.items()
            },
        },
        "outputs": {
            "background_case_ids": _relative(BACKGROUND_IDS_PATH),
            "explained_case_ids": _relative(EXPLAINED_IDS_PATH),
            "sample_manifest": _relative(SAMPLE_MANIFEST_PATH),
        },
        "assertions": assertions,
        "assertion_status": "PASS",
    }
    write_sample_manifest(manifest)

    if any(name == "shap" or name.startswith("shap.") for name in sys.modules):
        raise AssertionError("SHAP was unexpectedly imported during Part 1")
    if _artifact_hashes(protected_paths) != protected_hashes_before:
        raise AssertionError("A protected artifact changed during Part 1")
    print_report(manifest)
    return manifest


def print_report(manifest: dict[str, Any]) -> None:
    """Print measured Part 1 sampling and audit results."""
    print("\n=== SHAP PILOT V1 PART 1: FROZEN SAMPLES ===")
    counts = manifest["canonical_source_counts"]
    print(f"Canonical counts: train={counts['train']}, test={counts['test']}")
    print(
        "Sampling: independent numpy.random.default_rng(42), "
        "uniform without replacement"
    )
    print(f"Background sample size: {manifest['background_sample_size']}")
    print(f"Explained sample size: {manifest['explained_sample_size']}")
    print(
        "Background target distribution: "
        f"{manifest['background_target_distribution']}"
    )
    print(
        "Explained target distribution: "
        f"{manifest['explained_target_distribution']}"
    )
    print(
        "Full-source target distributions: "
        f"{manifest['full_source_target_distribution']}"
    )
    print(f"Matrix shapes: {manifest['matrix_shapes']}")
    print(f"Overlap checks: {manifest['overlap_checks']}")
    audit = manifest["prediction_audit"]
    print(f"Explained confusion counts: {audit['confusion_counts']}")
    print(f"Prediction-category coverage: {audit['category_coverage']}")
    print(f"Explained probability summary: {audit['probability_summary']}")
    print(f"Deterministic rerun: {manifest['deterministic_rerun']['result']}")
    print(
        "Protected artifacts: "
        f"{manifest['protected_artifacts']['count']} unchanged"
    )
    print(f"Required assertions: {manifest['assertion_status']}")
    print(f"Background IDs: {_relative(BACKGROUND_IDS_PATH)}")
    print(f"Explained IDs: {_relative(EXPLAINED_IDS_PATH)}")
    print(f"Manifest: {_relative(SAMPLE_MANIFEST_PATH)}")
    print("Subset matrices saved: no (extracted and validated in memory)")
    print("SHAP executed: no")


def main() -> None:
    """Run SHAP Pilot V1 Part 1 and stop."""
    prepare_shap_samples()


if __name__ == "__main__":
    main()
