"""Build train-only Baseline V1 feature spaces at prefix k=10.

This is Part 4 only. It uses the frozen Part 3 case-ID files, fits every
preprocessing component on training cases, and creates reproducible sparse
linear and tree representations. It does not fit or evaluate a classifier.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import inspect
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_string_dtype,
)
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_TABLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_k10"
    / "case_features_k10.parquet"
)
FEATURE_MANIFEST_PATH = (
    PROJECT_ROOT / "results" / "baseline_v1" / "feature_manifest.json"
)
SPLIT_SUMMARY_PATH = (
    PROJECT_ROOT / "results" / "baseline_v1" / "split_summary.json"
)
TRAIN_IDS_PATH = (
    PROJECT_ROOT / "results" / "baseline_v1" / "train_case_ids.csv"
)
TEST_IDS_PATH = (
    PROJECT_ROOT / "results" / "baseline_v1" / "test_case_ids.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "baseline_v1"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "baseline_k10"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "baseline_v1"
PREPROCESSING_MANIFEST_PATH = RESULTS_DIR / "preprocessing_manifest.json"
LINEAR_FEATURE_NAMES_PATH = RESULTS_DIR / "feature_names_linear.csv"
TREE_FEATURE_NAMES_PATH = RESULTS_DIR / "feature_names_tree.csv"
PREPROCESSING_COMPONENTS_PATH = (
    ARTIFACTS_DIR / "preprocessing_components.joblib"
)

MATRIX_PATHS = {
    "X_train_linear": PROCESSED_DIR / "X_train_linear.npz",
    "X_test_linear": PROCESSED_DIR / "X_test_linear.npz",
    "X_train_tree": PROCESSED_DIR / "X_train_tree.npz",
    "X_test_tree": PROCESSED_DIR / "X_test_tree.npz",
    "y_train": PROCESSED_DIR / "y_train.npy",
    "y_test": PROCESSED_DIR / "y_test.npy",
}

EXPECTED_TRAIN_CASES = 25_100
EXPECTED_TEST_CASES = 6_276
EXPECTED_TARGET_COUNTS = {
    "train": {0: 11_322, 1: 13_778},
    "test": {0: 2_826, 1: 3_450},
}
EXPECTED_NUMERIC_COLUMNS = [
    "requested_amount",
    "n_unique_activities",
    "n_unique_resources",
    "prefix_duration_seconds",
    "mean_event_gap_seconds",
]
EXPECTED_STATIC_CATEGORICAL_COLUMNS = ["loan_goal", "application_type"]
MAP_COLUMN_NAMESPACES = {
    "activity_counts": "activity",
    "action_counts": "action",
    "event_origin_counts": "origin",
    "resource_counts": "resource",
    "lifecycle_transition_counts": "lifecycle",
}
OUTCOME_ACTIVITIES = frozenset({"A_Pending", "A_Cancelled", "A_Denied"})
FORBIDDEN_SOURCE_COLUMNS = frozenset(
    {
        "case_id",
        "case_start_time",
        "target",
        "outcome_activity",
        "outcome_position",
    }
)


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load a required JSON object."""
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    """Return a stable content hash for an input artifact."""
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _relative(path: Path) -> str:
    """Return a platform-independent repository-relative path."""
    return path.relative_to(PROJECT_ROOT).as_posix()


def _load_canonical_ids(path: Path) -> list[str]:
    """Load one frozen ID file without changing its row order."""
    if not path.is_file():
        raise FileNotFoundError(f"Canonical split file not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8", dtype={"case_id": "string"})
    if frame.columns.tolist() != ["case_id"]:
        raise ValueError(
            f"Canonical split file must contain only case_id: {path}"
        )
    if frame["case_id"].isna().any() or not frame["case_id"].is_unique:
        raise ValueError(f"Canonical case IDs must be non-null and unique: {path}")
    return frame["case_id"].astype(str).tolist()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Construct a JSON object while rejecting duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate key in count-map JSON: {key!r}")
        result[key] = value
    return result


def parse_count_map(value: Any, *, column: str, case_id: str) -> dict[str, float]:
    """Parse and validate one dictionary/map or JSON count-map value."""
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value, object_pairs_hook=_reject_duplicate_json_keys)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(
                f"Invalid count-map JSON in {column!r} for case {case_id!r}"
            ) from error
    elif hasattr(value, "as_py"):
        parsed = value.as_py()

    if not isinstance(parsed, Mapping):
        raise TypeError(
            f"Count map {column!r} for case {case_id!r} is not a map/object: "
            f"{type(parsed).__name__}"
        )

    validated: dict[str, float] = {}
    for key, count in parsed.items():
        if not isinstance(key, str):
            raise TypeError(
                f"Count-map key in {column!r} for case {case_id!r} "
                "is not a string."
            )
        if not key:
            raise ValueError(
                f"Count-map key in {column!r} for case {case_id!r} is empty."
            )
        if isinstance(count, bool) or not isinstance(count, Real):
            raise TypeError(
                f"Count for {key!r} in {column!r}, case {case_id!r}, "
                "is not numeric."
            )
        numeric_count = float(count)
        if not math.isfinite(numeric_count) or numeric_count < 0:
            raise ValueError(
                f"Count for {key!r} in {column!r}, case {case_id!r}, "
                "must be finite and non-negative."
            )
        validated[key] = numeric_count
    return validated


def _validate_feature_contract(
    feature_table: pd.DataFrame,
    feature_manifest: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    """Confirm exact Part 2 predictor names, groups, and physical dtypes."""
    numeric_columns = feature_manifest.get("numeric_feature_columns")
    if numeric_columns != EXPECTED_NUMERIC_COLUMNS:
        raise AssertionError(
            "Part 2 numeric feature contract changed: "
            f"{numeric_columns!r}"
        )

    static_mapping = feature_manifest.get("static_source_features")
    if not isinstance(static_mapping, dict):
        raise ValueError("Part 2 manifest has no static source-feature mapping.")
    static_categorical_columns = [
        output
        for output in static_mapping.values()
        if output not in numeric_columns
    ]
    if static_categorical_columns != EXPECTED_STATIC_CATEGORICAL_COLUMNS:
        raise AssertionError(
            "Part 2 static categorical feature contract changed: "
            f"{static_categorical_columns!r}"
        )

    derived = feature_manifest.get("derived_prefix_features")
    if not isinstance(derived, dict):
        raise ValueError("Part 2 manifest has no derived-prefix feature mapping.")
    source_to_map_column = derived.get("per_case_count_maps")
    if not isinstance(source_to_map_column, dict):
        raise ValueError("Part 2 manifest has no per-case count-map mapping.")
    map_columns = list(source_to_map_column.values())
    if map_columns != list(MAP_COLUMN_NAMESPACES):
        raise AssertionError(
            f"Part 2 count-map feature contract changed: {map_columns!r}"
        )

    predictive_columns = feature_manifest.get("predictive_feature_columns")
    expected_predictors = (
        static_categorical_columns + [numeric_columns[0]] + map_columns
        + numeric_columns[1:]
    )
    if predictive_columns != expected_predictors:
        raise AssertionError(
            "Part 2 predictive feature order/content changed: "
            f"{predictive_columns!r}"
        )

    required_columns = [
        "case_id",
        "case_start_time",
        "target",
        *predictive_columns,
    ]
    if feature_table.columns.tolist() != required_columns:
        raise AssertionError(
            "Parquet columns do not match the Part 2 manifest exactly: "
            f"{feature_table.columns.tolist()!r}"
        )
    if not is_string_dtype(feature_table["case_id"].dtype):
        raise TypeError("case_id must have a string dtype.")
    if not is_datetime64_any_dtype(feature_table["case_start_time"].dtype):
        raise TypeError("case_start_time must have a datetime dtype.")
    if not is_numeric_dtype(feature_table["target"].dtype):
        raise TypeError("target must have a numeric dtype.")
    for column in numeric_columns:
        if not is_numeric_dtype(feature_table[column].dtype):
            raise TypeError(f"Numeric predictor {column!r} is not numeric.")
    for column in static_categorical_columns:
        if not is_string_dtype(feature_table[column].dtype):
            raise TypeError(f"Categorical predictor {column!r} is not string.")
    for column in map_columns:
        if not (
            is_string_dtype(feature_table[column].dtype)
            or feature_table[column].dtype == object
        ):
            raise TypeError(
                f"Count-map predictor {column!r} is neither string nor object."
            )
    return numeric_columns, static_categorical_columns, map_columns


def load_frozen_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    list[str],
    list[str],
    list[str],
]:
    """Load Part 2 data and select Part 3 splits by canonical IDs only."""
    feature_manifest = _load_json_object(FEATURE_MANIFEST_PATH)
    split_summary = _load_json_object(SPLIT_SUMMARY_PATH)
    if not FEATURE_TABLE_PATH.is_file():
        raise FileNotFoundError(
            f"Part 2 feature table not found: {FEATURE_TABLE_PATH}"
        )
    feature_table = pd.read_parquet(FEATURE_TABLE_PATH, engine="pyarrow")
    groups = _validate_feature_contract(feature_table, feature_manifest)
    numeric_columns, categorical_columns, map_columns = groups

    train_ids = _load_canonical_ids(TRAIN_IDS_PATH)
    test_ids = _load_canonical_ids(TEST_IDS_PATH)
    if len(train_ids) != EXPECTED_TRAIN_CASES:
        raise AssertionError(f"Unexpected canonical train size: {len(train_ids)}")
    if len(test_ids) != EXPECTED_TEST_CASES:
        raise AssertionError(f"Unexpected canonical test size: {len(test_ids)}")
    if not set(train_ids).isdisjoint(test_ids):
        raise AssertionError("Canonical train and test case IDs overlap.")
    if feature_table["case_id"].isna().any() or not feature_table["case_id"].is_unique:
        raise AssertionError("Part 2 case IDs must be non-null and unique.")
    if set(train_ids).union(test_ids) != set(feature_table["case_id"]):
        raise AssertionError(
            "Canonical Part 3 IDs do not exactly cover the Part 2 population."
        )

    indexed = feature_table.set_index("case_id", drop=False)
    train = indexed.loc[train_ids].reset_index(drop=True)
    test = indexed.loc[test_ids].reset_index(drop=True)
    if train["case_id"].tolist() != train_ids:
        raise AssertionError("Training rows do not preserve canonical ID order.")
    if test["case_id"].tolist() != test_ids:
        raise AssertionError("Test rows do not preserve canonical ID order.")

    if split_summary.get("train_cases") != len(train):
        raise AssertionError("Frozen split summary train count does not match IDs.")
    if split_summary.get("test_cases") != len(test):
        raise AssertionError("Frozen split summary test count does not match IDs.")
    if split_summary.get("canonical_train_ids") != _relative(TRAIN_IDS_PATH).replace(
        "/", "\\"
    ):
        raise AssertionError("Split summary points to a different train-ID file.")
    if split_summary.get("canonical_test_ids") != _relative(TEST_IDS_PATH).replace(
        "/", "\\"
    ):
        raise AssertionError("Split summary points to a different test-ID file.")

    predictor_columns = numeric_columns + categorical_columns + map_columns
    missing = pd.concat(
        [train[predictor_columns], test[predictor_columns]], ignore_index=True
    ).isna().sum()
    if missing.any():
        raise ValueError(
            "Unexpected missing predictor values; preprocessing stopped without "
            f"adding an imputer: {missing[missing.gt(0)].to_dict()}"
        )
    return (
        train,
        test,
        feature_manifest,
        split_summary,
        numeric_columns,
        categorical_columns,
        map_columns,
    )


def parse_maps(
    frame: pd.DataFrame, map_columns: list[str]
) -> dict[str, list[dict[str, float]]]:
    """Parse every structured column without constructing a global vocabulary."""
    parsed: dict[str, list[dict[str, float]]] = {}
    for column in map_columns:
        column_maps = [
            parse_count_map(value, column=column, case_id=case_id)
            for value, case_id in zip(frame[column], frame["case_id"], strict=True)
        ]
        invalid_totals = [
            index
            for index, counts in enumerate(column_maps)
            if not math.isclose(sum(counts.values()), 10.0)
        ]
        if invalid_totals:
            raise AssertionError(
                f"Count map {column!r} does not total 10 for "
                f"{len(invalid_totals)} cases."
            )
        parsed[column] = column_maps

    activity_keys = set().union(
        *(counts.keys() for counts in parsed["activity_counts"])
    )
    leaking_outcomes = sorted(activity_keys.intersection(OUTCOME_ACTIVITIES))
    if leaking_outcomes:
        raise AssertionError(
            "Outcome activities occur in prefix activity maps: "
            f"{leaking_outcomes}"
        )
    return parsed


def merge_namespaced_maps(
    parsed: dict[str, list[dict[str, float]]], row_count: int
) -> list[dict[str, float]]:
    """Merge the five maps per case using collision-proof namespaces."""
    merged_rows: list[dict[str, float]] = []
    for row_index in range(row_count):
        merged: dict[str, float] = {}
        for column, namespace in MAP_COLUMN_NAMESPACES.items():
            for key, count in parsed[column][row_index].items():
                namespaced_key = f"{namespace}::{key}"
                if namespaced_key in merged:
                    raise AssertionError(
                        f"Duplicate namespaced map key: {namespaced_key!r}"
                    )
                merged[namespaced_key] = count
        merged_rows.append(merged)
    return merged_rows


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create a sparse unknown-safe encoder across supported sklearn APIs."""
    parameters = inspect.signature(OneHotEncoder).parameters
    kwargs: dict[str, Any] = {
        "handle_unknown": "ignore",
        "dtype": np.float64,
    }
    if "sparse_output" in parameters:
        kwargs["sparse_output"] = True
    else:  # pragma: no cover - compatibility with older sklearn releases
        kwargs["sparse"] = True
    return OneHotEncoder(**kwargs)


def _vocabularies(
    parsed: dict[str, list[dict[str, float]]]
) -> dict[str, set[str]]:
    """Return observed raw keys for each structured feature group."""
    return {
        MAP_COLUMN_NAMESPACES[column]: set().union(
            *(counts.keys() for counts in rows)
        )
        for column, rows in parsed.items()
    }


def _unknown_audit(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical_columns: list[str],
    train_maps: dict[str, list[dict[str, float]]],
    test_maps: dict[str, list[dict[str, float]]],
) -> dict[str, dict[str, Any]]:
    """Audit test values and keys absent from training, including affected rows."""
    audit: dict[str, dict[str, Any]] = {}
    for column in categorical_columns:
        train_values = set(train[column])
        test_only = sorted(set(test[column]).difference(train_values))
        audit[column] = {
            "train_vocabulary_size": len(train_values),
            "test_only_count": len(test_only),
            "test_only_values": test_only,
            "test_rows_affected": int(test[column].isin(test_only).sum()),
        }

    train_vocabularies = _vocabularies(train_maps)
    test_vocabularies = _vocabularies(test_maps)
    for column, namespace in MAP_COLUMN_NAMESPACES.items():
        train_values = train_vocabularies[namespace]
        test_only = sorted(test_vocabularies[namespace].difference(train_values))
        affected = sum(
            bool(set(counts).intersection(test_only))
            for counts in test_maps[column]
        )
        audit[namespace] = {
            "train_vocabulary_size": len(train_values),
            "test_only_count": len(test_only),
            "test_only_values": test_only,
            "test_rows_affected": affected,
        }
    return audit


def _feature_metadata(
    numeric_columns: list[str],
    categorical_columns: list[str],
    encoder: OneHotEncoder,
    vectorizer: DictVectorizer,
) -> pd.DataFrame:
    """Build deterministic, unique feature names and source groups."""
    records: list[dict[str, Any]] = []
    for name in numeric_columns:
        records.append({"feature_name": name, "source_group": "numeric"})

    encoded_names = encoder.get_feature_names_out(categorical_columns).tolist()
    encoded_index = 0
    for column, categories in zip(
        categorical_columns, encoder.categories_, strict=True
    ):
        for _ in categories:
            records.append(
                {
                    "feature_name": encoded_names[encoded_index],
                    "source_group": f"categorical::{column}",
                }
            )
            encoded_index += 1

    for name in vectorizer.get_feature_names_out().tolist():
        namespace = name.split("::", maxsplit=1)[0]
        records.append(
            {
                "feature_name": name,
                "source_group": f"structured_map::{namespace}",
            }
        )
    metadata = pd.DataFrame.from_records(records)
    metadata.insert(0, "feature_index", np.arange(len(metadata), dtype=int))
    if not metadata["feature_name"].is_unique:
        duplicates = metadata.loc[
            metadata["feature_name"].duplicated(keep=False), "feature_name"
        ].tolist()
        raise AssertionError(f"Transformed feature names are not unique: {duplicates}")
    return metadata


def _matrix_stats(matrix: sparse.csr_matrix) -> dict[str, Any]:
    """Return shape and practical sparsity statistics for a CSR matrix."""
    entries = matrix.shape[0] * matrix.shape[1]
    density = matrix.nnz / entries if entries else 0.0
    return {
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "nonzero_entries": int(matrix.nnz),
        "density": density,
        "sparsity": 1.0 - density,
        "format": matrix.format,
        "dtype": str(matrix.dtype),
    }


def _assert_finite(matrix: sparse.csr_matrix, label: str) -> None:
    """Reject NaN and infinity in stored sparse values."""
    if np.isnan(matrix.data).any():
        raise AssertionError(f"{label} contains NaN values.")
    if np.isinf(matrix.data).any():
        raise AssertionError(f"{label} contains infinite values.")


def _constant_features(
    matrix: sparse.csr_matrix, feature_metadata: pd.DataFrame
) -> list[dict[str, str]]:
    """Find exactly constant training columns without removing them."""
    csc = matrix.tocsc(copy=True)
    csc.eliminate_zeros()
    constants: list[dict[str, str]] = []
    for column_index in range(csc.shape[1]):
        start = csc.indptr[column_index]
        stop = csc.indptr[column_index + 1]
        values = csc.data[start:stop]
        is_constant_zero = len(values) == 0
        is_constant_nonzero = (
            len(values) == csc.shape[0]
            and np.all(values == values[0])
        )
        if is_constant_zero or is_constant_nonzero:
            row = feature_metadata.iloc[column_index]
            constants.append(
                {
                    "feature_name": str(row["feature_name"]),
                    "source_group": str(row["source_group"]),
                }
            )
    return constants


def build_preprocessing() -> dict[str, Any]:
    """Fit train-only components, validate leakage controls, and save outputs."""
    (
        train,
        test,
        feature_manifest,
        split_summary,
        numeric_columns,
        categorical_columns,
        map_columns,
    ) = load_frozen_data()
    predictor_columns = numeric_columns + categorical_columns + map_columns

    train_maps = parse_maps(train, map_columns)
    test_maps = parse_maps(test, map_columns)
    train_dicts = merge_namespaced_maps(train_maps, len(train))
    test_dicts = merge_namespaced_maps(test_maps, len(test))
    unknown_audit = _unknown_audit(
        train,
        test,
        categorical_columns,
        train_maps,
        test_maps,
    )

    # These are the only fit calls in Part 4. Every input is train-derived.
    map_vectorizer = DictVectorizer(sparse=True, sort=True, dtype=np.float64)
    X_train_maps = map_vectorizer.fit_transform(train_dicts).tocsr()
    X_test_maps = map_vectorizer.transform(test_dicts).tocsr()

    categorical_encoder = _make_one_hot_encoder()
    X_train_categorical = categorical_encoder.fit_transform(
        train[categorical_columns]
    ).tocsr()
    X_test_categorical = categorical_encoder.transform(
        test[categorical_columns]
    ).tocsr()

    train_numeric = train[numeric_columns].to_numpy(dtype=np.float64)
    test_numeric = test[numeric_columns].to_numpy(dtype=np.float64)
    numeric_scaler = StandardScaler()
    X_train_numeric_linear = numeric_scaler.fit_transform(train_numeric)
    X_test_numeric_linear = numeric_scaler.transform(test_numeric)

    X_train_linear = sparse.hstack(
        [X_train_numeric_linear, X_train_categorical, X_train_maps],
        format="csr",
        dtype=np.float64,
    )
    X_test_linear = sparse.hstack(
        [X_test_numeric_linear, X_test_categorical, X_test_maps],
        format="csr",
        dtype=np.float64,
    )
    X_train_tree = sparse.hstack(
        [train_numeric, X_train_categorical, X_train_maps],
        format="csr",
        dtype=np.float64,
    )
    X_test_tree = sparse.hstack(
        [test_numeric, X_test_categorical, X_test_maps],
        format="csr",
        dtype=np.float64,
    )
    matrices = {
        "X_train_linear": X_train_linear,
        "X_test_linear": X_test_linear,
        "X_train_tree": X_train_tree,
        "X_test_tree": X_test_tree,
    }
    feature_metadata = _feature_metadata(
        numeric_columns,
        categorical_columns,
        categorical_encoder,
        map_vectorizer,
    )
    feature_names = feature_metadata["feature_name"].tolist()
    expected_dimension = len(feature_names)

    assertions: dict[str, str] = {}
    assert len(train) == EXPECTED_TRAIN_CASES
    assertions["A_train_row_count"] = "PASS"
    assert len(test) == EXPECTED_TEST_CASES
    assertions["B_test_row_count"] = "PASS"
    assert train["case_id"].tolist() == _load_canonical_ids(TRAIN_IDS_PATH)
    assert test["case_id"].tolist() == _load_canonical_ids(TEST_IDS_PATH)
    assertions["C_frozen_case_ids_exact"] = "PASS"
    assert train["target"].value_counts().sort_index().to_dict() == (
        EXPECTED_TARGET_COUNTS["train"]
    )
    assert test["target"].value_counts().sort_index().to_dict() == (
        EXPECTED_TARGET_COUNTS["test"]
    )
    assertions["D_target_distributions"] = "PASS"
    assert X_train_linear.shape[1] == X_test_linear.shape[1]
    assert X_train_tree.shape[1] == X_test_tree.shape[1]
    assert X_train_linear.shape[1] == X_train_tree.shape[1]
    assertions["E_identical_train_test_dimensions"] = "PASS"
    assert len(feature_names) == len(set(feature_names))
    assertions["F_unique_feature_names"] = "PASS"
    for label, matrix in matrices.items():
        _assert_finite(matrix, label)
    assertions["G_no_nan"] = "PASS"
    assertions["H_no_infinity"] = "PASS"
    assert "case_id" not in predictor_columns
    assertions["I_case_id_excluded"] = "PASS"
    assert "case_start_time" not in predictor_columns
    assertions["J_case_start_time_excluded"] = "PASS"
    assert "target" not in predictor_columns
    assertions["K_target_excluded"] = "PASS"
    assert not any(
        column in FORBIDDEN_SOURCE_COLUMNS
        or "outcome_activity" in column.lower()
        or "outcome_position" in column.lower()
        or "audit" in column.lower()
        for column in predictor_columns
    )
    assertions["L_outcome_and_audit_fields_excluded"] = "PASS"

    train_vocabularies = _vocabularies(train_maps)
    for activity, assertion_name in [
        ("A_Pending", "M_A_Pending_absent"),
        ("A_Cancelled", "N_A_Cancelled_absent"),
        ("A_Denied", "O_A_Denied_absent"),
    ]:
        assert activity not in train_vocabularies["activity"]
        assertions[assertion_name] = "PASS"

    for column, fitted_categories in zip(
        categorical_columns, categorical_encoder.categories_, strict=True
    ):
        assert set(fitted_categories) == set(train[column])
    assertions["P_all_categories_occur_in_train"] = "PASS"
    expected_map_features = {
        f"{namespace}::{key}"
        for namespace, keys in train_vocabularies.items()
        for key in keys
    }
    assert set(map_vectorizer.vocabulary_) == expected_map_features
    assertions["Q_all_structured_keys_occur_in_train"] = "PASS"
    for column, fitted_categories in zip(
        categorical_columns, categorical_encoder.categories_, strict=True
    ):
        test_only = set(test[column]).difference(train[column])
        assert test_only.isdisjoint(fitted_categories)
    assertions["R_test_only_categories_not_fitted"] = "PASS"
    test_vocabularies = _vocabularies(test_maps)
    test_only_map_features = {
        f"{namespace}::{key}"
        for namespace, keys in test_vocabularies.items()
        for key in keys.difference(train_vocabularies[namespace])
    }
    assert test_only_map_features.isdisjoint(map_vectorizer.vocabulary_)
    assertions["S_test_only_structured_keys_not_fitted"] = "PASS"
    np.testing.assert_allclose(numeric_scaler.mean_, train_numeric.mean(axis=0))
    np.testing.assert_allclose(numeric_scaler.var_, train_numeric.var(axis=0))
    assert np.all(np.asarray(numeric_scaler.n_samples_seen_) == len(train))
    assertions["T_scaler_statistics_from_train_only"] = "PASS"
    assert X_train_maps.shape[0] == len(train)
    assert X_train_categorical.shape[0] == len(train)
    assert int(np.asarray(numeric_scaler.n_samples_seen_).max()) == len(train)
    assertions["U_all_components_fitted_on_train_only"] = "PASS"

    assert expected_dimension == X_train_linear.shape[1]
    np.testing.assert_allclose(
        X_train_linear[:, : len(numeric_columns)].toarray(),
        numeric_scaler.transform(train_numeric),
    )
    np.testing.assert_allclose(
        X_train_tree[:, : len(numeric_columns)].toarray(), train_numeric
    )

    linear_constants = _constant_features(X_train_linear, feature_metadata)
    tree_constants = _constant_features(X_train_tree, feature_metadata)
    if linear_constants != tree_constants:
        raise AssertionError(
            "Linear and tree representations disagree on constant columns."
        )

    y_train = train["target"].to_numpy(dtype=np.int8, copy=True)
    y_test = test["target"].to_numpy(dtype=np.int8, copy=True)
    source_dtypes = {
        column: str(feature_table_dtype)
        for column, feature_table_dtype in zip(
            train.columns, train.dtypes, strict=True
        )
    }
    category_sizes = {
        column: len(categories)
        for column, categories in zip(
            categorical_columns, categorical_encoder.categories_, strict=True
        )
    }
    vocabulary_sizes = {
        namespace: len(keys)
        for namespace, keys in train_vocabularies.items()
    }
    matrix_stats = {
        label: _matrix_stats(matrix) for label, matrix in matrices.items()
    }

    manifest: dict[str, Any] = {
        "baseline_version": "v1_part4",
        "prediction_point": 10,
        "preprocessing_fit_scope": "train_only",
        "classifier_trained": False,
        "split_sizes": {"train": len(train), "test": len(test)},
        "target_distributions": {
            "train": {str(key): value for key, value in EXPECTED_TARGET_COUNTS["train"].items()},
            "test": {str(key): value for key, value in EXPECTED_TARGET_COUNTS["test"].items()},
        },
        "inputs": {
            "feature_table": _relative(FEATURE_TABLE_PATH),
            "feature_manifest": _relative(FEATURE_MANIFEST_PATH),
            "split_summary": _relative(SPLIT_SUMMARY_PATH),
            "canonical_train_ids": _relative(TRAIN_IDS_PATH),
            "canonical_test_ids": _relative(TEST_IDS_PATH),
            "sha256": {
                "feature_table": _sha256(FEATURE_TABLE_PATH),
                "feature_manifest": _sha256(FEATURE_MANIFEST_PATH),
                "split_summary": _sha256(SPLIT_SUMMARY_PATH),
                "canonical_train_ids": _sha256(TRAIN_IDS_PATH),
                "canonical_test_ids": _sha256(TEST_IDS_PATH),
            },
        },
        "source_column_dtypes": source_dtypes,
        "count_map_storage_and_parser": {
            "parquet_storage": "JSON string",
            "parser": "json.loads with duplicate-key rejection",
        },
        "software_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "predictor_groups": {
            "numeric": numeric_columns,
            "static_categorical": categorical_columns,
            "structured_count_maps": {
                column: namespace
                for column, namespace in MAP_COLUMN_NAMESPACES.items()
            },
            "excluded_metadata": ["case_id", "case_start_time"],
            "excluded_target": "target",
        },
        "train_fitted_sizes": {
            "numeric_features": len(numeric_columns),
            "categorical": category_sizes,
            "structured": vocabulary_sizes,
        },
        "test_only_unknowns": unknown_audit,
        "representations": {
            "linear": {
                "numeric_processing": "StandardScaler fitted on train only",
                "feature_dimension": X_train_linear.shape[1],
                "train_matrix": matrix_stats["X_train_linear"],
                "test_matrix": matrix_stats["X_test_linear"],
            },
            "tree": {
                "numeric_processing": "unscaled",
                "feature_dimension": X_train_tree.shape[1],
                "train_matrix": matrix_stats["X_train_tree"],
                "test_matrix": matrix_stats["X_test_tree"],
            },
            "shared_processing": (
                "train-fitted OneHotEncoder and DictVectorizer"
            ),
        },
        "linear_numeric_scaler": {
            "feature_order": numeric_columns,
            "mean": numeric_scaler.mean_.tolist(),
            "variance": numeric_scaler.var_.tolist(),
            "scale": numeric_scaler.scale_.tolist(),
            "samples_seen": int(np.asarray(numeric_scaler.n_samples_seen_).max()),
        },
        "zero_variance_train_features": {
            "count": len(linear_constants),
            "features": linear_constants,
            "removed": False,
        },
        "feature_names": {
            "linear": _relative(LINEAR_FEATURE_NAMES_PATH),
            "tree": _relative(TREE_FEATURE_NAMES_PATH),
            "deterministic": True,
            "unique": True,
        },
        "saved_artifacts": {
            "shared_preprocessing_components": _relative(
                PREPROCESSING_COMPONENTS_PATH
            ),
            **{label: _relative(path) for label, path in MATRIX_PATHS.items()},
        },
        "component_fit_rows": {
            "OneHotEncoder": len(train),
            "DictVectorizer": len(train),
            "StandardScaler": len(train),
        },
        "assertions": assertions,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    feature_metadata.to_csv(
        LINEAR_FEATURE_NAMES_PATH, index=False, encoding="utf-8"
    )
    feature_metadata.to_csv(
        TREE_FEATURE_NAMES_PATH, index=False, encoding="utf-8"
    )
    for label, matrix in matrices.items():
        sparse.save_npz(MATRIX_PATHS[label], matrix, compressed=True)
    np.save(MATRIX_PATHS["y_train"], y_train, allow_pickle=False)
    np.save(MATRIX_PATHS["y_test"], y_test, allow_pickle=False)
    joblib.dump(
        {
            "preprocessing_fit_scope": "train_only",
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "map_columns_to_namespaces": MAP_COLUMN_NAMESPACES,
            "numeric_scaler": numeric_scaler,
            "categorical_encoder": categorical_encoder,
            "map_vectorizer": map_vectorizer,
            "feature_names": feature_names,
            "canonical_train_ids_sha256": _sha256(TRAIN_IDS_PATH),
        },
        PREPROCESSING_COMPONENTS_PATH,
        compress=3,
    )
    with PREPROCESSING_MANIFEST_PATH.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print_report(manifest)
    return manifest


def print_report(manifest: dict[str, Any]) -> None:
    """Print the compact Part 4 audit required for review."""
    print("\n=== BASELINE V1 PART 4: TRAIN-ONLY PREPROCESSING ===")
    print(f"Train cases: {manifest['split_sizes']['train']:,}")
    print(f"Test cases: {manifest['split_sizes']['test']:,}")
    print("Preprocessing fit scope: train_only")

    print("\n=== TRAIN-FITTED FEATURE GROUP SIZES ===")
    sizes = manifest["train_fitted_sizes"]
    print(f"Numeric features: {sizes['numeric_features']}")
    for group, size in sizes["categorical"].items():
        print(f"{group}: {size}")
    for group, size in sizes["structured"].items():
        print(f"{group}: {size}")

    print("\n=== TEST-ONLY UNKNOWN AUDIT ===")
    for group, audit in manifest["test_only_unknowns"].items():
        print(
            f"{group}: train size={audit['train_vocabulary_size']}, "
            f"test-only={audit['test_only_count']}, "
            f"affected rows={audit['test_rows_affected']}"
        )
        if audit["test_only_values"]:
            print(f"  values: {audit['test_only_values']}")

    print("\n=== FEATURE SPACES ===")
    for representation in ["linear", "tree"]:
        report = manifest["representations"][representation]
        print(
            f"{representation.upper()} dimension: "
            f"{report['feature_dimension']}"
        )
        for split in ["train_matrix", "test_matrix"]:
            stats = report[split]
            print(
                f"  {split}: {stats['rows']:,} x {stats['columns']:,}; "
                f"density={stats['density']:.6%}; "
                f"sparsity={stats['sparsity']:.6%}"
            )

    constants = manifest["zero_variance_train_features"]
    print("\n=== ZERO-VARIANCE TRAIN FEATURES ===")
    print(f"Count: {constants['count']}")
    for feature in constants["features"]:
        print(f"{feature['feature_name']} [{feature['source_group']}]")
    print("Removed: no")

    print("\n=== LEAKAGE ASSERTIONS ===")
    for name, result in manifest["assertions"].items():
        print(f"{name}: {result}")
    print("\nAll Baseline V1 Part 4 assertions passed.")
    print(f"Saved manifest: {PREPROCESSING_MANIFEST_PATH}")


def main() -> None:
    """Run Baseline V1 Part 4 and stop before model training."""
    build_preprocessing()


if __name__ == "__main__":
    main()
