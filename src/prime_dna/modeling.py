"""Family-disjoint modeling and statistical evaluation for PRIME-DNA."""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import (
    BRANCH_CONTEXT_FEATURES,
    DEGREE_FEATURES,
    LOCAL_FEATURE_NAMES,
    STRUCTURE_FEATURE_NAMES,
    STRUCTURE_FEATURE_GROUPS,
    CachePaths,
    assert_feature_contract,
    feature_indices,
    make_split_manifest,
)


EPSILON = 1e-7

LEAVE_ONE_GROUP_MODELS = {
    "topology_position": "lineage_without_topology",
    "descendant_scale": "lineage_without_scale",
    "branch_context": "lineage_without_branch",
    "child_composition": "lineage_without_composition",
}

GROUP_ONLY_MODELS = {
    "topology_position": "lineage_topology_only",
    "descendant_scale": "lineage_scale_only",
    "branch_context": "lineage_branch",
    "child_composition": "lineage_composition_only",
}


@dataclass
class ExperimentResult:
    metrics: pd.DataFrame
    family_metrics: pd.DataFrame
    criteria: dict[str, object]
    run_manifest: dict[str, object]


def _clip_probability(probability: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)


def expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, n_bins: int = 15
) -> float:
    """Return equal-width expected calibration error for binary probabilities."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    y_true = np.asarray(y_true, dtype=np.uint8)
    probability = _clip_probability(probability)
    if len(y_true) != len(probability) or not len(y_true):
        raise ValueError("Calibration inputs must be non-empty and equally sized")
    bin_index = np.minimum((probability * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for index in range(n_bins):
        mask = bin_index == index
        if not mask.any():
            continue
        error += float(mask.mean()) * abs(
            float(probability[mask].mean()) - float(y_true[mask].mean())
        )
    return float(error)


def select_macro_f1_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Select a threshold from validation data without a quadratic grid scan."""
    y_true = np.asarray(y_true, dtype=np.uint8)
    probability = _clip_probability(probability)
    if len(y_true) == 0:
        raise ValueError("Cannot select a threshold from an empty validation set")
    order = np.argsort(-probability, kind="mergesort")
    sorted_y = y_true[order]
    sorted_p = probability[order]
    cumulative_positive = np.cumsum(sorted_y)
    positions = np.arange(1, len(sorted_y) + 1)
    boundaries = np.flatnonzero(
        np.r_[sorted_p[:-1] != sorted_p[1:], True]
    )
    tp = cumulative_positive[boundaries].astype(float)
    fp = positions[boundaries].astype(float) - tp
    total_positive = float(sorted_y.sum())
    total_negative = float(len(sorted_y) - total_positive)
    fn = total_positive - tp
    tn = total_negative - fp
    f1_positive = np.divide(
        2.0 * tp,
        2.0 * tp + fp + fn,
        out=np.zeros_like(tp),
        where=(2.0 * tp + fp + fn) > 0,
    )
    f1_negative = np.divide(
        2.0 * tn,
        2.0 * tn + fp + fn,
        out=np.zeros_like(tn),
        where=(2.0 * tn + fp + fn) > 0,
    )
    macro = 0.5 * (f1_positive + f1_negative)

    # Include the all-negative candidate.
    all_negative_macro = 0.5 * (
        0.0
        + (2.0 * total_negative)
        / max(2.0 * total_negative + total_positive, 1.0)
    )
    best = int(np.argmax(macro))
    if all_negative_macro > float(macro[best]):
        return float(np.nextafter(sorted_p[0], np.inf))
    return float(sorted_p[boundaries[best]])


def classification_metrics(
    y_true: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, float]:
    """Return threshold-free, calibrated, and thresholded binary metrics."""
    y_true = np.asarray(y_true, dtype=np.uint8)
    probability = _clip_probability(probability)
    prediction = (probability >= threshold).astype(np.uint8)
    both_classes = len(np.unique(y_true)) == 2
    tp = int(np.sum((y_true == 1) & (prediction == 1)))
    tn = int(np.sum((y_true == 0) & (prediction == 0)))
    fp = int(np.sum((y_true == 0) & (prediction == 1)))
    fn = int(np.sum((y_true == 1) & (prediction == 0)))

    def safe_ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    return {
        "AUROC": float(roc_auc_score(y_true, probability)) if both_classes else float("nan"),
        "AUPRC": float(average_precision_score(y_true, probability))
        if both_classes
        else float("nan"),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, probability)),
        "ece_15": expected_calibration_error(y_true, probability, 15),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction))
        if both_classes
        else float("nan"),
        "macro_f1": float(
            f1_score(
                y_true,
                prediction,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "precision_duplication": safe_ratio(tp, tp + fp),
        "recall_duplication": safe_ratio(tp, tp + fn),
        "specificity_speciation": safe_ratio(tn, tn + fp),
        "negative_predictive_value": safe_ratio(tn, tn + fn),
        "mcc": float(matthews_corrcoef(y_true, prediction))
        if both_classes
        else float("nan"),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()) if len(y_true) else float("nan"),
    }


def select_fusion_weight(
    y_validation: np.ndarray,
    baseline_probability: np.ndarray,
    lineage_probability: np.ndarray,
    *,
    maximum_weight: float = 0.5,
) -> tuple[float, pd.DataFrame]:
    """Select the conservative PRIME expert weight on validation log loss."""
    rows = []
    for weight in np.linspace(0.0, maximum_weight, 51):
        probability = (1.0 - weight) * baseline_probability + weight * lineage_probability
        rows.append(
            {
                "weight": float(weight),
                "validation_log_loss": float(
                    log_loss(y_validation, _clip_probability(probability), labels=[0, 1])
                ),
            }
        )
    frame = pd.DataFrame(rows)
    selected = float(
        frame.sort_values(["validation_log_loss", "weight"]).iloc[0]["weight"]
    )
    if selected > maximum_weight + 1e-12:
        raise AssertionError("Fusion weight escaped its declared cap")
    return selected, frame


def _sample_weights(y_train: np.ndarray) -> np.ndarray:
    positive = max(int(y_train.sum()), 1)
    negative = max(int(len(y_train) - positive), 1)
    return np.where(
        y_train == 1,
        len(y_train) / (2.0 * positive),
        len(y_train) / (2.0 * negative),
    ).astype(np.float32)


def _hgb(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=31,
        learning_rate=0.08,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


def _logistic(seed: int) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=500,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        ),
    )


def _fit_models(
    X_local: np.ndarray,
    X_structure: np.ndarray,
    y: np.ndarray,
    train_index: np.ndarray,
    seed: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    assert_feature_contract((*LOCAL_FEATURE_NAMES, *STRUCTURE_FEATURE_NAMES))
    sample_weight = _sample_weights(y[train_index])
    degree_index = feature_indices(STRUCTURE_FEATURE_NAMES, DEGREE_FEATURES)
    branch_index = feature_indices(
        STRUCTURE_FEATURE_NAMES, BRANCH_CONTEXT_FEATURES
    )
    matrices = {
        "local_logistic": X_local,
        "local_hgb": X_local,
        "lineage_logistic": X_structure,
        "lineage_degree": X_structure[:, degree_index],
        "lineage_branch": X_structure[:, branch_index],
        "lineage_full": X_structure,
        "prime_joint": np.column_stack([X_local, X_structure]),
    }
    for group_name, group_features in STRUCTURE_FEATURE_GROUPS.items():
        matrices[GROUP_ONLY_MODELS[group_name]] = X_structure[
            :, feature_indices(STRUCTURE_FEATURE_NAMES, group_features)
        ]
    for group_name, excluded_features in STRUCTURE_FEATURE_GROUPS.items():
        keep_features = [
            name for name in STRUCTURE_FEATURE_NAMES if name not in excluded_features
        ]
        matrices[LEAVE_ONE_GROUP_MODELS[group_name]] = X_structure[
            :, feature_indices(STRUCTURE_FEATURE_NAMES, keep_features)
        ]
    models: dict[str, object] = {}
    for offset, name in enumerate(["local_logistic", "lineage_logistic"]):
        logistic = _logistic(seed + offset)
        logistic.fit(matrices[name][train_index], y[train_index])
        models[name] = logistic
    # Preserve the original five model seeds so adding comparisons cannot move
    # the pre-existing primary results. New models use a separate offset range.
    hgb_seed_offsets = {
        "local_hgb": 0,
        "lineage_degree": 1,
        "lineage_branch": 2,
        "lineage_full": 3,
        "prime_joint": 4,
        "lineage_topology_only": 10,
        "lineage_scale_only": 11,
        "lineage_composition_only": 12,
        "lineage_without_topology": 20,
        "lineage_without_scale": 21,
        "lineage_without_branch": 22,
        "lineage_without_composition": 23,
    }
    for name, offset in hgb_seed_offsets.items():
        model = _hgb(seed + offset)
        model.fit(
            matrices[name][train_index],
            y[train_index],
            sample_weight=sample_weight,
        )
        models[name] = model
    return models, matrices


def _model_probabilities(
    models: Mapping[str, object],
    matrices: Mapping[str, np.ndarray],
    row_index: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        name: _clip_probability(model.predict_proba(matrices[name][row_index])[:, 1])
        for name, model in models.items()
    }


def _family_metrics(
    y_test: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
    test_family_index: np.ndarray,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    unique_families, starts, counts = np.unique(
        test_family_index, return_index=True, return_counts=True
    )
    for family_index, start, count in zip(unique_families, starts, counts):
        segment = slice(int(start), int(start + count))
        family = manifest.iloc[int(family_index)]
        for model_name, probability in probabilities.items():
            metrics = classification_metrics(
                y_test[segment], probability[segment], thresholds[model_name]
            )
            rows.append(
                {
                    "family_id": family["family_id"],
                    "source_file": family["source_file"],
                    "model": model_name,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _paired_bootstrap(
    family_metrics: pd.DataFrame,
    baseline_model: str,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    pivot_log = family_metrics.pivot(index="family_id", columns="model", values="log_loss")
    pivot_f1 = family_metrics.pivot(index="family_id", columns="model", values="macro_f1")
    log_gain = (
        pivot_log[baseline_model] - pivot_log["prime_fusion"]
    ).dropna().to_numpy(float)
    f1_gain = (
        pivot_f1["prime_fusion"] - pivot_f1[baseline_model]
    ).dropna().to_numpy(float)
    if not len(log_gain) or not len(f1_gain):
        raise ValueError("No paired test-family metrics available for bootstrap")
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in range(repeats):
        log_sample = rng.integers(0, len(log_gain), len(log_gain))
        f1_sample = rng.integers(0, len(f1_gain), len(f1_gain))
        rows.append(
            {
                "repeat": repeat,
                "mean_log_loss_gain": float(log_gain[log_sample].mean()),
                "mean_macro_f1_gain": float(f1_gain[f1_sample].mean()),
            }
        )
    distribution = pd.DataFrame(rows)
    summary = {
        "families": int(len(log_gain)),
        "mean_log_loss_gain": float(log_gain.mean()),
        "log_loss_gain_ci_low": float(
            distribution["mean_log_loss_gain"].quantile(0.025)
        ),
        "log_loss_gain_ci_high": float(
            distribution["mean_log_loss_gain"].quantile(0.975)
        ),
        "mean_macro_f1_gain": float(f1_gain.mean()),
        "macro_f1_gain_ci_low": float(
            distribution["mean_macro_f1_gain"].quantile(0.025)
        ),
        "macro_f1_gain_ci_high": float(
            distribution["mean_macro_f1_gain"].quantile(0.975)
        ),
    }
    return distribution, summary


def _placebo_distribution(
    y_test: np.ndarray,
    baseline_probability: np.ndarray,
    lineage_probability: np.ndarray,
    test_family_index: np.ndarray,
    fusion_weight: float,
    repeats: int,
    seed: int,
    *,
    within_family: bool = True,
) -> pd.DataFrame:
    """Break node/context alignment within families or across the full test set."""
    unique_families, starts, counts = np.unique(
        test_family_index, return_index=True, return_counts=True
    )
    del unique_families
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in range(repeats):
        shuffled = lineage_probability.copy()
        if within_family:
            for start, count in zip(starts, counts):
                rng.shuffle(shuffled[int(start) : int(start + count)])
        else:
            rng.shuffle(shuffled)
        probability = (1.0 - fusion_weight) * baseline_probability + fusion_weight * shuffled
        rows.append(
            {
                "repeat": repeat,
                "placebo_type": "within_family" if within_family else "global",
                "AUROC": float(roc_auc_score(y_test, probability)),
            }
        )
    return pd.DataFrame(rows)


def _calibration_table(
    y_true: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    model_names: Sequence[str],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Create equal-count reliability bins without changing model probabilities."""
    rows: list[dict[str, object]] = []
    y_true = np.asarray(y_true, dtype=np.uint8)
    for model_name in model_names:
        probability = _clip_probability(probabilities[model_name])
        order = np.argsort(probability, kind="mergesort")
        for bin_index, indices in enumerate(np.array_split(order, n_bins)):
            if not len(indices):
                continue
            observed = float(y_true[indices].mean())
            predicted = float(probability[indices].mean())
            rows.append(
                {
                    "model": model_name,
                    "bin": int(bin_index + 1),
                    "n": int(len(indices)),
                    "probability_min": float(probability[indices].min()),
                    "probability_max": float(probability[indices].max()),
                    "mean_predicted_probability": predicted,
                    "observed_duplication_rate": observed,
                    "absolute_calibration_gap": abs(predicted - observed),
                }
            )
    return pd.DataFrame(rows)


def _subgroup_metrics(
    y_test: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
    test_family_index: np.ndarray,
    manifest: pd.DataFrame,
    X_local_test: np.ndarray,
    X_structure_test: np.ndarray,
    model_names: Sequence[str],
) -> pd.DataFrame:
    """Evaluate frozen models over family and node strata without retuning."""
    family_size_bin = manifest["size_bin"].to_numpy(dtype=int)[test_family_index]
    duplication_bin = manifest["duplication_bin"].to_numpy(dtype=int)[test_family_index]
    normalized_depth = X_structure_test[
        :, int(feature_indices(STRUCTURE_FEATURE_NAMES, ["normalized_depth"])[0])
    ]
    depth_bin = np.select(
        [
            np.isclose(normalized_depth, 0.0),
            normalized_depth <= 0.25,
            normalized_depth <= 0.50,
            normalized_depth <= 0.75,
        ],
        [0, 1, 2, 3],
        default=4,
    ).astype(int)
    root_bin = (X_local_test[:, 1] < 0.5).astype(int)

    size_ranges = (
        manifest.groupby("size_bin")["n_nodes"].agg(["min", "max"]).to_dict("index")
    )
    duplication_ranges = (
        manifest.groupby("duplication_bin")["duplication_rate"]
        .agg(["min", "max"])
        .to_dict("index")
    )
    dimensions: list[tuple[str, np.ndarray, dict[int, str]]] = [
        (
            "family_size_quintile",
            family_size_bin,
            {
                int(level): (
                    f"Q{int(level) + 1}: {int(bounds['min'])}-{int(bounds['max'])} nodes"
                )
                for level, bounds in size_ranges.items()
            },
        ),
        (
            "family_duplication_quintile",
            duplication_bin,
            {
                int(level): (
                    f"Q{int(level) + 1}: {float(bounds['min']):.3f}-"
                    f"{float(bounds['max']):.3f} duplication rate"
                )
                for level, bounds in duplication_ranges.items()
            },
        ),
        (
            "normalized_depth",
            depth_bin,
            {
                0: "root (0)",
                1: "shallow (0, 0.25]",
                2: "middle (0.25, 0.50]",
                3: "deep (0.50, 0.75]",
                4: "deepest (0.75, 1.00]",
            },
        ),
        ("root_status", root_bin, {0: "root", 1: "non-root"}),
    ]
    rows: list[dict[str, object]] = []
    for dimension, labels, label_names in dimensions:
        for level in sorted(np.unique(labels)):
            mask = labels == level
            if not mask.any():
                continue
            for model_name in model_names:
                rows.append(
                    {
                        "dimension": dimension,
                        "level": int(level),
                        "level_label": label_names.get(int(level), str(level)),
                        "model": model_name,
                        "n_families": int(np.unique(test_family_index[mask]).size),
                        **classification_metrics(
                            y_test[mask],
                            probabilities[model_name][mask],
                            thresholds[model_name],
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _within_family_permutation_importance(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_family_index: np.ndarray,
    feature_names: Sequence[str],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Measure AUROC loss after shuffling one feature within each test family."""
    if repeats < 1:
        raise ValueError("Permutation repeats must be positive")
    X_work = np.asarray(X_test, dtype=np.float32).copy()
    baseline_probability = _clip_probability(model.predict_proba(X_work)[:, 1])
    baseline_auroc = float(roc_auc_score(y_test, baseline_probability))
    _, starts, counts = np.unique(
        test_family_index, return_index=True, return_counts=True
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for feature_index, feature_name in enumerate(feature_names):
        original = X_work[:, feature_index].copy()
        for repeat in range(repeats):
            shuffled = original.copy()
            for start, count in zip(starts, counts):
                rng.shuffle(shuffled[int(start) : int(start + count)])
            X_work[:, feature_index] = shuffled
            probability = _clip_probability(model.predict_proba(X_work)[:, 1])
            permuted_auroc = float(roc_auc_score(y_test, probability))
            rows.append(
                {
                    "feature": feature_name,
                    "repeat": int(repeat),
                    "permutation_scope": "within_family",
                    "baseline_auroc": baseline_auroc,
                    "permuted_auroc": permuted_auroc,
                    "auroc_drop": baseline_auroc - permuted_auroc,
                }
            )
        X_work[:, feature_index] = original
    return pd.DataFrame(rows)


def _stratified_training_families(
    manifest: pd.DataFrame, fraction: float, seed: int
) -> np.ndarray:
    """Select a deterministic fraction of training families within split strata."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Training fraction must be in (0, 1]")
    train = manifest.loc[manifest["split"] == "train"]
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for _, group in train.groupby(["size_bin", "duplication_bin"], sort=True):
        positions = group.index.to_numpy(dtype=int, copy=True)
        rng.shuffle(positions)
        count = len(positions) if fraction == 1.0 else max(1, int(round(fraction * len(positions))))
        selected.extend(positions[:count].tolist())
    return np.asarray(sorted(selected), dtype=int)


def _learning_curve(
    X_local: np.ndarray,
    X_structure: np.ndarray,
    y: np.ndarray,
    family_index: np.ndarray,
    manifest: pd.DataFrame,
    validation_index: np.ndarray,
    test_index: np.ndarray,
    full_models: Mapping[str, object],
    full_validation_probability: Mapping[str, np.ndarray],
    full_test_probability: Mapping[str, np.ndarray],
    fractions: Sequence[float],
    seed: int,
) -> pd.DataFrame:
    """Estimate family-level data efficiency for local, lineage, and fusion models."""
    rows: list[dict[str, object]] = []
    for fraction_index, fraction in enumerate(sorted(set(float(v) for v in fractions))):
        selected_families = _stratified_training_families(
            manifest, fraction, seed + 30_000 + fraction_index
        )
        train_index = np.flatnonzero(np.isin(family_index, selected_families))
        if fraction == 1.0:
            local_model = full_models["local_hgb"]
            lineage_model = full_models["lineage_full"]
            validation_local = full_validation_probability["local_hgb"]
            validation_lineage = full_validation_probability["lineage_full"]
            test_local = full_test_probability["local_hgb"]
            test_lineage = full_test_probability["lineage_full"]
        else:
            sample_weight = _sample_weights(y[train_index])
            local_model = _hgb(seed)
            lineage_model = _hgb(seed + 3)
            local_model.fit(X_local[train_index], y[train_index], sample_weight=sample_weight)
            lineage_model.fit(
                X_structure[train_index], y[train_index], sample_weight=sample_weight
            )
            validation_local = _clip_probability(
                local_model.predict_proba(X_local[validation_index])[:, 1]
            )
            validation_lineage = _clip_probability(
                lineage_model.predict_proba(X_structure[validation_index])[:, 1]
            )
            test_local = _clip_probability(
                local_model.predict_proba(X_local[test_index])[:, 1]
            )
            test_lineage = _clip_probability(
                lineage_model.predict_proba(X_structure[test_index])[:, 1]
            )
        fusion_weight, _ = select_fusion_weight(
            y[validation_index], validation_local, validation_lineage
        )
        validation_fusion = _clip_probability(
            (1.0 - fusion_weight) * validation_local + fusion_weight * validation_lineage
        )
        test_fusion = _clip_probability(
            (1.0 - fusion_weight) * test_local + fusion_weight * test_lineage
        )
        validation_map = {
            "local_hgb": validation_local,
            "lineage_full": validation_lineage,
            "prime_fusion": validation_fusion,
        }
        test_map = {
            "local_hgb": test_local,
            "lineage_full": test_lineage,
            "prime_fusion": test_fusion,
        }
        for model_name, probability in test_map.items():
            threshold = select_macro_f1_threshold(
                y[validation_index], validation_map[model_name]
            )
            rows.append(
                {
                    "training_fraction_requested": fraction,
                    "training_families": int(len(selected_families)),
                    "training_nodes": int(len(train_index)),
                    "model": model_name,
                    "fusion_weight": fusion_weight if model_name == "prime_fusion" else float("nan"),
                    **classification_metrics(y[test_index], probability, threshold),
                }
            )
    return pd.DataFrame(rows)


def _split_sensitivity(
    X_local: np.ndarray,
    X_structure: np.ndarray,
    y: np.ndarray,
    family_index: np.ndarray,
    family_frame: pd.DataFrame,
    primary_manifest: pd.DataFrame,
    primary_validation_probability: Mapping[str, np.ndarray],
    primary_test_probability: Mapping[str, np.ndarray],
    primary_thresholds: Mapping[str, float],
    primary_weight: float,
    seeds: Sequence[int],
) -> pd.DataFrame:
    """Repeat the core local/lineage/fusion comparison over family split seeds."""
    rows: list[dict[str, object]] = []
    primary_seed = int(seeds[0])
    for split_seed in dict.fromkeys(int(v) for v in seeds):
        if split_seed == primary_seed:
            manifest = primary_manifest
            split_code = manifest["split_code"].to_numpy(dtype=np.int8)[family_index]
            validation_index = np.flatnonzero(split_code == 1)
            test_index = np.flatnonzero(split_code == 2)
            validation_probability = {
                name: primary_validation_probability[name]
                for name in ("local_hgb", "lineage_full", "prime_fusion")
            }
            test_probability = {
                name: primary_test_probability[name]
                for name in ("local_hgb", "lineage_full", "prime_fusion")
            }
            thresholds = {
                name: primary_thresholds[name]
                for name in ("local_hgb", "lineage_full", "prime_fusion")
            }
            fusion_weight = primary_weight
        else:
            manifest = make_split_manifest(family_frame, split_seed)
            split_code = manifest["split_code"].to_numpy(dtype=np.int8)[family_index]
            train_index = np.flatnonzero(split_code == 0)
            validation_index = np.flatnonzero(split_code == 1)
            test_index = np.flatnonzero(split_code == 2)
            sample_weight = _sample_weights(y[train_index])
            # Hold model randomness fixed so this analysis isolates family split changes.
            local_model = _hgb(primary_seed)
            lineage_model = _hgb(primary_seed + 3)
            local_model.fit(X_local[train_index], y[train_index], sample_weight=sample_weight)
            lineage_model.fit(
                X_structure[train_index], y[train_index], sample_weight=sample_weight
            )
            validation_local = _clip_probability(
                local_model.predict_proba(X_local[validation_index])[:, 1]
            )
            validation_lineage = _clip_probability(
                lineage_model.predict_proba(X_structure[validation_index])[:, 1]
            )
            test_local = _clip_probability(
                local_model.predict_proba(X_local[test_index])[:, 1]
            )
            test_lineage = _clip_probability(
                lineage_model.predict_proba(X_structure[test_index])[:, 1]
            )
            fusion_weight, _ = select_fusion_weight(
                y[validation_index], validation_local, validation_lineage
            )
            validation_probability = {
                "local_hgb": validation_local,
                "lineage_full": validation_lineage,
                "prime_fusion": _clip_probability(
                    (1.0 - fusion_weight) * validation_local
                    + fusion_weight * validation_lineage
                ),
            }
            test_probability = {
                "local_hgb": test_local,
                "lineage_full": test_lineage,
                "prime_fusion": _clip_probability(
                    (1.0 - fusion_weight) * test_local
                    + fusion_weight * test_lineage
                ),
            }
            thresholds = {
                name: select_macro_f1_threshold(y[validation_index], probability)
                for name, probability in validation_probability.items()
            }
        test_family_index = family_index[test_index]
        family_metrics = _family_metrics(
            y[test_index], test_probability, thresholds, test_family_index, manifest
        )
        family_macro = family_metrics.groupby("model")["macro_f1"].mean()
        for model_name, probability in test_probability.items():
            rows.append(
                {
                    "split_seed": split_seed,
                    "model": model_name,
                    "fusion_weight": fusion_weight if model_name == "prime_fusion" else float("nan"),
                    "training_families": int((manifest["split"] == "train").sum()),
                    "validation_families": int((manifest["split"] == "validation").sum()),
                    "test_families": int((manifest["split"] == "test").sum()),
                    "family_macro_f1": float(family_macro.loc[model_name]),
                    **classification_metrics(
                        y[test_index], probability, thresholds[model_name]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _split_audit(manifest: pd.DataFrame) -> dict[str, object]:
    families = {
        split: set(manifest.loc[manifest["split"] == split, "family_id"])
        for split in ("train", "validation", "test")
    }
    overlap = (
        (families["train"] & families["validation"])
        | (families["train"] & families["test"])
        | (families["validation"] & families["test"])
    )
    return {
        "families_by_split": {key: len(value) for key, value in families.items()},
        "family_overlap_count": len(overlap),
        "passed": len(overlap) == 0 and all(families.values()),
    }


def run_experiment(
    cache: CachePaths,
    manifest: pd.DataFrame,
    audit: Mapping[str, object],
    output_dir: Path,
    *,
    seed: int = 42,
    bootstrap_repeats: int = 2_000,
    placebo_repeats: int = 100,
    permutation_repeats: int = 5,
    sensitivity_seeds: Sequence[int] | None = None,
    learning_fractions: Sequence[float] = (0.10, 0.25, 0.50, 1.00),
) -> ExperimentResult:
    """Train, evaluate, and persist the complete family-disjoint experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(cache.features, allow_pickle=False) as arrays:
        X_local = arrays["X_local"]
        X_structure = arrays["X_structure"]
        y = arrays["y"].astype(np.uint8, copy=False)
        family_index = arrays["family_index"].astype(np.int64, copy=False)
        node_ids = arrays["node_id"]
    if family_index.max(initial=-1) >= len(manifest):
        raise ValueError("Feature cache family indices do not match the split manifest")
    split_code = manifest["split_code"].to_numpy(dtype=np.int8)[family_index]
    train_index = np.flatnonzero(split_code == 0)
    validation_index = np.flatnonzero(split_code == 1)
    test_index = np.flatnonzero(split_code == 2)
    if not all(len(index) for index in (train_index, validation_index, test_index)):
        raise ValueError("Every split must contain target nodes")

    models, matrices = _fit_models(
        X_local, X_structure, y, train_index, seed
    )
    validation_probability = _model_probabilities(
        models, matrices, validation_index
    )
    test_probability = _model_probabilities(models, matrices, test_index)

    train_prior = float(y[train_index].mean())
    validation_probability["training_prior"] = np.full(
        len(validation_index), train_prior
    )
    test_probability["training_prior"] = np.full(len(test_index), train_prior)
    local_candidates = ["local_logistic", "local_hgb"]
    best_local = min(
        local_candidates,
        key=lambda name: log_loss(
            y[validation_index], validation_probability[name], labels=[0, 1]
        ),
    )
    fusion_weight, weight_scan = select_fusion_weight(
        y[validation_index],
        validation_probability[best_local],
        validation_probability["lineage_full"],
    )
    validation_probability["prime_fusion"] = _clip_probability(
        (1.0 - fusion_weight) * validation_probability[best_local]
        + fusion_weight * validation_probability["lineage_full"]
    )
    test_probability["prime_fusion"] = _clip_probability(
        (1.0 - fusion_weight) * test_probability[best_local]
        + fusion_weight * test_probability["lineage_full"]
    )
    weight_scan.to_csv(output_dir / "fusion_weight_scan.csv", index=False)

    thresholds = {
        name: select_macro_f1_threshold(y[validation_index], probability)
        for name, probability in validation_probability.items()
    }
    comparison_models = [
        best_local,
        "lineage_logistic",
        "lineage_full",
        "prime_fusion",
        "prime_joint",
    ]
    calibration = _calibration_table(
        y[test_index], test_probability, comparison_models, n_bins=10
    )
    calibration.to_csv(output_dir / "calibration_table.csv", index=False)
    metric_rows = []
    for name, probability in test_probability.items():
        metric_rows.append(
            {
                "model": name,
                **classification_metrics(y[test_index], probability, thresholds[name]),
            }
        )
    metrics = pd.DataFrame(metric_rows)

    test_family_index = family_index[test_index]
    family_metrics = _family_metrics(
        y[test_index],
        test_probability,
        thresholds,
        test_family_index,
        manifest,
    )
    family_macro = (
        family_metrics.groupby("model", as_index=False)
        .agg(
            family_macro_f1=("macro_f1", "mean"),
            family_macro_log_loss=("log_loss", "mean"),
            families=("family_id", "nunique"),
        )
    )
    metrics = metrics.merge(family_macro, on="model", how="left")
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    family_metrics.to_csv(output_dir / "family_metrics.csv", index=False)

    subgroup_metrics = _subgroup_metrics(
        y[test_index],
        test_probability,
        thresholds,
        test_family_index,
        manifest,
        X_local[test_index],
        X_structure[test_index],
        [best_local, "lineage_full", "prime_fusion", "prime_joint"],
    )
    subgroup_metrics.to_csv(output_dir / "subgroup_metrics.csv", index=False)

    bootstrap_distribution, bootstrap_summary = _paired_bootstrap(
        family_metrics, best_local, bootstrap_repeats, seed + 10_000
    )
    bootstrap_distribution.to_csv(
        output_dir / "bootstrap_distribution.csv", index=False
    )
    (output_dir / "bootstrap_summary.json").write_text(
        json.dumps(bootstrap_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    placebo = _placebo_distribution(
        y[test_index],
        test_probability[best_local],
        test_probability["lineage_full"],
        test_family_index,
        fusion_weight,
        placebo_repeats,
        seed + 20_000,
        within_family=True,
    )
    placebo.to_csv(output_dir / "placebo_distribution.csv", index=False)

    global_placebo = _placebo_distribution(
        y[test_index],
        test_probability[best_local],
        test_probability["lineage_full"],
        test_family_index,
        fusion_weight,
        placebo_repeats,
        seed + 21_000,
        within_family=False,
    )
    global_placebo.to_csv(
        output_dir / "global_placebo_distribution.csv", index=False
    )

    permutation_importance = _within_family_permutation_importance(
        models["lineage_full"],
        X_structure[test_index],
        y[test_index],
        test_family_index,
        STRUCTURE_FEATURE_NAMES,
        permutation_repeats,
        seed + 22_000,
    )
    permutation_importance.to_csv(
        output_dir / "feature_permutation_importance.csv", index=False
    )

    learning_curve = _learning_curve(
        X_local,
        X_structure,
        y,
        family_index,
        manifest,
        validation_index,
        test_index,
        models,
        validation_probability,
        test_probability,
        learning_fractions,
        seed,
    )
    learning_curve.to_csv(output_dir / "learning_curve.csv", index=False)

    if sensitivity_seeds is None:
        sensitivity_seeds = (seed, seed + 1, seed + 2)
    sensitivity_seeds = tuple(dict.fromkeys([seed, *map(int, sensitivity_seeds)]))
    family_frame = pd.read_csv(cache.families)
    split_sensitivity = _split_sensitivity(
        X_local,
        X_structure,
        y,
        family_index,
        family_frame,
        manifest,
        validation_probability,
        test_probability,
        thresholds,
        fusion_weight,
        sensitivity_seeds,
    )
    split_sensitivity.to_csv(output_dir / "split_sensitivity.csv", index=False)

    prediction_frame = pd.DataFrame(
        {
            "node_id": np.char.decode(node_ids[test_index], "utf-8"),
            "family_id": manifest["family_id"].to_numpy()[test_family_index],
            "source_file": manifest["source_file"].to_numpy()[test_family_index],
            "split": "test",
            "target_event": np.where(y[test_index] == 1, "duplication", "speciation"),
        }
    )
    for name, probability in test_probability.items():
        prediction_frame[f"{name}_probability"] = probability.astype(np.float32)
        prediction_frame[f"{name}_prediction"] = np.where(
            probability >= thresholds[name], "duplication", "speciation"
        )
    prediction_frame.to_csv(
        output_dir / "test_predictions.csv.gz", index=False, compression="gzip"
    )

    split_audit = _split_audit(manifest)
    metric_lookup = metrics.set_index("model")
    auroc_gain = float(
        metric_lookup.loc["prime_fusion", "AUROC"]
        - metric_lookup.loc[best_local, "AUROC"]
    )
    family_f1_gain = float(
        metric_lookup.loc["prime_fusion", "family_macro_f1"]
        - metric_lookup.loc[best_local, "family_macro_f1"]
    )
    placebo_mean = float(placebo["AUROC"].mean())
    placebo_gap = float(metric_lookup.loc["prime_fusion", "AUROC"] - placebo_mean)
    audit_passed = bool(audit.get("critical_checks_passed"))
    leakage_passed = bool(audit.get("leakage_check_passed"))
    passed = {
        "auroc_gain_at_least_0_05": auroc_gain >= 0.05,
        "family_macro_f1_gain_at_least_0_05": family_f1_gain >= 0.05,
        "bootstrap_log_loss_ci_above_zero": bootstrap_summary[
            "log_loss_gain_ci_low"
        ]
        > 0.0,
        "aligned_auroc_at_least_0_03_above_placebo": placebo_gap >= 0.03,
        "data_audit_passed": audit_passed,
        "leakage_audit_passed": leakage_passed,
        "family_split_audit_passed": bool(split_audit["passed"]),
    }
    if all(passed.values()):
        conclusion = "supported_bounded_generalization"
    elif audit_passed and leakage_passed and auroc_gain > 0 and family_f1_gain > 0:
        conclusion = "directional_but_inconclusive"
    elif not (audit_passed and leakage_passed and split_audit["passed"]):
        conclusion = "invalid_due_to_audit_failure"
    else:
        conclusion = "not_supported"
    criteria: dict[str, object] = {
        "conclusion": conclusion,
        "best_local_baseline": best_local,
        "selected_fusion_weight": fusion_weight,
        "test_auroc_gain": auroc_gain,
        "family_macro_f1_gain": family_f1_gain,
        "placebo_mean_auroc": placebo_mean,
        "aligned_minus_placebo_auroc": placebo_gap,
        "bootstrap": bootstrap_summary,
        "split_audit": split_audit,
        "criteria": passed,
    }

    subgroup_pivot = subgroup_metrics.pivot_table(
        index=["dimension", "level", "level_label"],
        columns="model",
        values="AUROC",
    )
    subgroup_gain = (
        subgroup_pivot["prime_fusion"] - subgroup_pivot[best_local]
    ).dropna()
    sensitivity_pivot = split_sensitivity.pivot(
        index="split_seed", columns="model", values="AUROC"
    )
    sensitivity_gain = (
        sensitivity_pivot["prime_fusion"] - sensitivity_pivot["local_hgb"]
    )
    permutation_summary = (
        permutation_importance.groupby("feature", as_index=False)
        .agg(mean_auroc_drop=("auroc_drop", "mean"), std_auroc_drop=("auroc_drop", "std"))
        .sort_values("mean_auroc_drop", ascending=False)
    )
    permutation_summary["std_auroc_drop"] = permutation_summary[
        "std_auroc_drop"
    ].fillna(0.0)
    leave_one_out = {
        group_name: {
            "model": model_name,
            "AUROC": float(metric_lookup.loc[model_name, "AUROC"]),
            "drop_from_full_lineage": float(
                metric_lookup.loc["lineage_full", "AUROC"]
                - metric_lookup.loc[model_name, "AUROC"]
            ),
        }
        for group_name, model_name in LEAVE_ONE_GROUP_MODELS.items()
    }
    group_only = {
        group_name: {
            "model": model_name,
            "AUROC": float(metric_lookup.loc[model_name, "AUROC"]),
            "AUPRC": float(metric_lookup.loc[model_name, "AUPRC"]),
        }
        for group_name, model_name in GROUP_ONLY_MODELS.items()
    }
    extended_summary: dict[str, object] = {
        "status": "posthoc_robustness_not_used_for_primary_model_selection",
        "global_placebo": {
            "repeats": int(len(global_placebo)),
            "mean_AUROC": float(global_placebo["AUROC"].mean()),
            "aligned_minus_mean_AUROC": float(
                metric_lookup.loc["prime_fusion", "AUROC"]
                - global_placebo["AUROC"].mean()
            ),
        },
        "subgroups": {
            "evaluable_groups": int(len(subgroup_gain)),
            "minimum_prime_auroc_gain": float(subgroup_gain.min()),
            "median_prime_auroc_gain": float(subgroup_gain.median()),
            "all_evaluable_groups_positive": bool((subgroup_gain > 0).all()),
        },
        "split_sensitivity": {
            "seeds": [int(value) for value in sensitivity_gain.index],
            "minimum_prime_auroc_gain": float(sensitivity_gain.min()),
            "maximum_prime_auroc_gain": float(sensitivity_gain.max()),
            "all_seeds_positive": bool((sensitivity_gain > 0).all()),
        },
        "calibration": {
            model_name: float(metric_lookup.loc[model_name, "ece_15"])
            for model_name in comparison_models
        },
        "leave_one_group_out": leave_one_out,
        "group_only_models": group_only,
        "top_permutation_features": permutation_summary.head(5).to_dict("records"),
    }
    criteria["extended_robustness"] = extended_summary
    (output_dir / "extended_analysis_summary.json").write_text(
        json.dumps(extended_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "acceptance_criteria.json").write_text(
        json.dumps(criteria, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    run_manifest: dict[str, object] = {
        "seed": seed,
        "bootstrap_repeats": bootstrap_repeats,
        "placebo_repeats": placebo_repeats,
        "rows_by_split": {
            "train": int(len(train_index)),
            "validation": int(len(validation_index)),
            "test": int(len(test_index)),
        },
        "families_by_split": split_audit["families_by_split"],
        "features": {
            "local": list(LOCAL_FEATURE_NAMES),
            "structure": list(STRUCTURE_FEATURE_NAMES),
            "topology_subset": list(DEGREE_FEATURES),
            "branch_subset": list(BRANCH_CONTEXT_FEATURES),
            "leave_one_group_out": {
                group_name: list(features)
                for group_name, features in STRUCTURE_FEATURE_GROUPS.items()
            },
            "group_only_models": dict(GROUP_ONLY_MODELS),
        },
        "leakage_fields_excluded": list(audit.get("leakage_blacklist", [])),
        "source": {
            "nodes_path": audit.get("nodes_path"),
            "edges_path": audit.get("edges_path"),
            "sha256": audit.get("source_sha256", {}),
            "audit_scope": audit.get("scope"),
            "critical_checks_passed": audit.get("critical_checks_passed"),
        },
        "model": {
            "hist_gradient_boosting": {
                "max_iter": 100,
                "max_leaf_nodes": 31,
                "learning_rate": 0.08,
                "l2_regularization": 1.0,
            },
            "fusion_weight_range": [0.0, 0.5],
            "fusion_weight_step": 0.01,
            "best_local_baseline": best_local,
            "selected_fusion_weight": fusion_weight,
        },
        "extended_analysis": {
            "status": "posthoc_robustness_not_used_for_primary_model_selection",
            "permutation_scope": "within_family",
            "permutation_repeats": int(permutation_repeats),
            "learning_fractions": [float(value) for value in learning_fractions],
            "split_sensitivity_seeds": [int(value) for value in sensitivity_seeds],
            "calibration_bins": 10,
            "ece_equal_width_bins": 15,
            "subgroup_thresholds": "frozen_validation_thresholds",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ExperimentResult(metrics, family_metrics, criteria, run_manifest)
