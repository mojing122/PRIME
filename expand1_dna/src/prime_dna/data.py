"""Streaming PANTHER audit and leakage-safe feature extraction.

The raw files are close to one gigabyte, so this module processes one protein
family at a time.  Event annotations are targets only.  No event field, NHX
event code, raw node number, or edge-side copy of the event annotation enters
the model feature matrices.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


TARGET_EVENTS = {"speciation": 0, "duplication": 1}
LEAKAGE_FIELDS = {
    "event_type",
    "event_type_raw",
    "parent_event_type",
    "duplication_flag",
    "nhx_attributes",
    "raw_node_id",
}

LOCAL_FEATURE_NAMES = (
    "incoming_branch_length",
    "is_root",
    "incoming_branch_missing",
)

STRUCTURE_FEATURE_NAMES = (
    "log_outdegree",
    "normalized_depth",
    "log_sibling_count",
    "log_family_nodes",
    "log_subtree_nodes",
    "log_descendant_leaves",
    "child_branch_mean",
    "child_branch_std",
    "child_branch_min",
    "child_branch_max",
    "child_branch_zero_fraction",
    "child_ancestral_fraction",
    "child_taxon_diversity",
    "child_taxon_missing_fraction",
)

DEGREE_FEATURES = (
    "log_outdegree",
    "normalized_depth",
    "log_sibling_count",
    "log_family_nodes",
    "log_subtree_nodes",
    "log_descendant_leaves",
    "child_ancestral_fraction",
)

BRANCH_CONTEXT_FEATURES = (
    "child_branch_mean",
    "child_branch_std",
    "child_branch_min",
    "child_branch_max",
    "child_branch_zero_fraction",
)

TOPOLOGY_POSITION_FEATURES = (
    "log_outdegree",
    "normalized_depth",
    "log_sibling_count",
)

DESCENDANT_SCALE_FEATURES = (
    "log_family_nodes",
    "log_subtree_nodes",
    "log_descendant_leaves",
)

CHILD_COMPOSITION_FEATURES = (
    "child_ancestral_fraction",
    "child_taxon_diversity",
    "child_taxon_missing_fraction",
)

STRUCTURE_FEATURE_GROUPS = {
    "topology_position": TOPOLOGY_POSITION_FEATURES,
    "descendant_scale": DESCENDANT_SCALE_FEATURES,
    "branch_context": BRANCH_CONTEXT_FEATURES,
    "child_composition": CHILD_COMPOSITION_FEATURES,
}


@dataclass(frozen=True)
class CachePaths:
    """Files produced by one full or quick extraction profile."""

    root: Path

    @property
    def features(self) -> Path:
        return self.root / "features.npz"

    @property
    def families(self) -> Path:
        return self.root / "families.csv"

    @property
    def audit(self) -> Path:
        return self.root / "data_audit.json"

    @property
    def metadata(self) -> Path:
        return self.root / "cache_metadata.json"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading a source file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def grouped_csv(path: Path) -> Iterator[tuple[str, list[dict[str, str]]]]:
    """Yield consecutive ``family_id`` groups from a CSV file."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "family_id" not in reader.fieldnames:
            raise ValueError(f"{path} has no family_id column")
        previous: str | None = None
        for family_id, rows in groupby(reader, key=lambda row: row["family_id"]):
            if previous is not None and family_id <= previous:
                raise ValueError(
                    f"{path} must be strictly grouped by family_id; "
                    f"saw {family_id!r} after {previous!r}"
                )
            previous = family_id
            yield family_id, list(rows)


def _float_or_nan(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def _safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def assert_feature_contract(feature_names: Sequence[str]) -> None:
    """Fail closed when a target-derived field enters a model interface."""
    lowered = {name.lower() for name in feature_names}
    direct = lowered.intersection(LEAKAGE_FIELDS)
    suspicious = {
        name
        for name in lowered
        if any(token in name for token in ("event_type", "parent_event", "nhx", "raw_node"))
    }
    if direct or suspicious:
        raise ValueError(f"Leakage-prone features rejected: {sorted(direct | suspicious)}")


def _tree_features(
    family_id: str,
    nodes: Sequence[Mapping[str, str]],
    edges: Sequence[Mapping[str, str]],
) -> tuple[list[tuple[str, int, list[float], list[float]]], dict[str, object]]:
    """Validate one family tree and derive per-target feature rows."""
    node_by_id = {row["node_id"]: row for row in nodes}
    duplicate_nodes = len(nodes) - len(node_by_id)
    children: dict[str, list[tuple[str, float]]] = defaultdict(list)
    parent: dict[str, str] = {}
    endpoint_missing = 0
    self_edges = 0
    duplicate_parent = 0
    family_mismatch = 0
    relation_mismatch = 0
    event_copy_mismatch = 0
    branch_copy_mismatch = 0

    for edge in edges:
        source = edge["source_id"]
        target = edge["target_id"]
        if source not in node_by_id or target not in node_by_id:
            endpoint_missing += 1
            continue
        if source == target:
            self_edges += 1
        if target in parent:
            duplicate_parent += 1
        parent[target] = source
        if edge["family_id"] != family_id:
            family_mismatch += 1
        if edge["relation"] != "parent_of":
            relation_mismatch += 1
        if edge["parent_event_type"] != node_by_id[source]["event_type"]:
            event_copy_mismatch += 1
        child_branch = _float_or_nan(edge["child_branch_length"])
        node_branch = _float_or_nan(node_by_id[target]["branch_length"])
        if not (math.isclose(child_branch, node_branch, rel_tol=1e-7, abs_tol=1e-9)):
            branch_copy_mismatch += 1
        children[source].append((target, child_branch))

    declared_roots = [
        row["node_id"] for row in nodes if row["is_root"].strip().lower() == "true"
    ]
    degree_roots = [node_id for node_id in node_by_id if node_id not in parent]
    root = degree_roots[0] if len(degree_roots) == 1 else None

    depth: dict[str, int] = {}
    traversal: list[str] = []
    if root is not None:
        stack = [(root, 0)]
        while stack:
            node_id, node_depth = stack.pop()
            if node_id in depth:
                continue
            depth[node_id] = node_depth
            traversal.append(node_id)
            stack.extend((child, node_depth + 1) for child, _ in children[node_id])

    subtree_nodes = {node_id: 1 for node_id in node_by_id}
    descendant_leaves = {
        node_id: int(len(children[node_id]) == 0) for node_id in node_by_id
    }
    for node_id in reversed(traversal):
        for child, _ in children[node_id]:
            subtree_nodes[node_id] += subtree_nodes[child]
            descendant_leaves[node_id] += descendant_leaves[child]

    critical_errors = {
        "duplicate_nodes": duplicate_nodes,
        "endpoint_missing": endpoint_missing,
        "self_edges": self_edges,
        "duplicate_parent": duplicate_parent,
        "family_mismatch": family_mismatch,
        "relation_mismatch": relation_mismatch,
        "event_copy_mismatch": event_copy_mismatch,
        "branch_copy_mismatch": branch_copy_mismatch,
        "edge_count_mismatch": int(len(edges) != len(nodes) - 1),
        "declared_root_mismatch": int(
            len(declared_roots) != 1 or root is None or declared_roots[0] != root
        ),
        "unreachable_nodes": len(nodes) - len(depth),
    }

    max_depth = max(depth.values(), default=0)
    feature_rows: list[tuple[str, int, list[float], list[float]]] = []
    for row in nodes:
        event_type = row["event_type"]
        if event_type not in TARGET_EVENTS:
            continue
        node_id = row["node_id"]
        node_children = children[node_id]
        outdegree = len(node_children)
        incoming = _float_or_nan(row["branch_length"])
        is_root = float(node_id == root)
        local = [
            0.0 if math.isnan(incoming) else incoming,
            is_root,
            float(math.isnan(incoming)),
        ]

        child_lengths = np.asarray([length for _, length in node_children], dtype=float)
        if len(child_lengths):
            branch_mean = float(child_lengths.mean())
            branch_std = float(child_lengths.std())
            branch_min = float(child_lengths.min())
            branch_max = float(child_lengths.max())
            zero_fraction = float(np.mean(child_lengths == 0.0))
        else:
            branch_mean = branch_std = branch_min = branch_max = zero_fraction = 0.0

        child_rows = [node_by_id[child] for child, _ in node_children]
        ancestral_fraction = _safe_fraction(
            sum(child["node_type"] == "ancestral_gene" for child in child_rows),
            outdegree,
        )
        anchors = [
            (child.get("species") or child.get("organism_code") or "").strip()
            for child in child_rows
        ]
        nonempty_anchors = [anchor for anchor in anchors if anchor]
        taxon_diversity = _safe_fraction(len(set(nonempty_anchors)), outdegree)
        taxon_missing = _safe_fraction(
            sum(not anchor for anchor in anchors), outdegree
        )
        sibling_count = (
            max(len(children[parent[node_id]]) - 1, 0) if node_id in parent else 0
        )
        structure = [
            math.log1p(outdegree),
            _safe_fraction(depth.get(node_id, 0), max_depth),
            math.log1p(sibling_count),
            math.log1p(len(nodes)),
            math.log1p(subtree_nodes[node_id]),
            math.log1p(descendant_leaves[node_id]),
            branch_mean,
            branch_std,
            branch_min,
            branch_max,
            zero_fraction,
            ancestral_fraction,
            taxon_diversity,
            taxon_missing,
        ]
        feature_rows.append((node_id, TARGET_EVENTS[event_type], local, structure))

    source_files = {row["source_file"] for row in nodes}
    edge_source_files = {row["source_file"] for row in edges}
    family_summary: dict[str, object] = {
        "family_id": family_id,
        "source_file": next(iter(source_files)) if len(source_files) == 1 else "",
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_targets": len(feature_rows),
        "n_speciation": sum(row["event_type"] == "speciation" for row in nodes),
        "n_duplication": sum(row["event_type"] == "duplication" for row in nodes),
        "n_coded_event": sum(row["event_type"] == "coded_event" for row in nodes),
        "max_depth": max_depth,
        "node_source_file_count": len(source_files),
        "edge_source_file_count": len(edge_source_files),
        **critical_errors,
    }
    family_summary["duplication_rate"] = _safe_fraction(
        int(family_summary["n_duplication"]), int(family_summary["n_targets"])
    )
    return feature_rows, family_summary


def _quantile_summary(values: Iterable[int | float]) -> dict[str, float]:
    data = np.asarray(list(values), dtype=float)
    if not len(data):
        return {}
    return {
        "min": float(data.min()),
        "q25": float(np.quantile(data, 0.25)),
        "median": float(np.quantile(data, 0.50)),
        "q75": float(np.quantile(data, 0.75)),
        "q95": float(np.quantile(data, 0.95)),
        "q99": float(np.quantile(data, 0.99)),
        "max": float(data.max()),
        "mean": float(data.mean()),
    }


def build_feature_cache(
    nodes_path: Path,
    edges_path: Path,
    cache: CachePaths,
    *,
    quick_families: int | None = None,
    write_features: bool = True,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Audit raw files and optionally persist leakage-safe model arrays."""
    assert_feature_contract((*LOCAL_FEATURE_NAMES, *STRUCTURE_FEATURE_NAMES))
    cache.root.mkdir(parents=True, exist_ok=True)

    local_values = array("f")
    structure_values = array("f")
    labels = bytearray()
    family_indices = array("I")
    node_ids: list[bytes] = []
    max_node_id_bytes = 1
    family_rows: list[dict[str, object]] = []
    node_nulls: Counter[str] = Counter()
    edge_nulls: Counter[str] = Counter()
    node_types: Counter[str] = Counter()
    event_types: Counter[str] = Counter()
    relation_types: Counter[str] = Counter()
    node_fields: list[str] = []
    edge_fields: list[str] = []

    node_groups = grouped_csv(nodes_path)
    edge_groups = grouped_csv(edges_path)
    for family_index, ((node_family, nodes), (edge_family, edges)) in enumerate(
        zip(node_groups, edge_groups, strict=True)
    ):
        if node_family != edge_family:
            raise ValueError(
                f"Node/edge family streams diverged: {node_family} != {edge_family}"
            )
        if quick_families is not None and family_index >= quick_families:
            break
        if not node_fields:
            node_fields = list(nodes[0])
            edge_fields = list(edges[0])
        for row in nodes:
            node_types[row["node_type"] or "<NULL>"] += 1
            event_types[row["event_type"] or "<NULL>"] += 1
            for field, value in row.items():
                if value is None or value == "":
                    node_nulls[field] += 1
        for row in edges:
            relation_types[row["relation"] or "<NULL>"] += 1
            for field, value in row.items():
                if value is None or value == "":
                    edge_nulls[field] += 1

        extracted, family_summary = _tree_features(node_family, nodes, edges)
        family_rows.append(family_summary)
        if write_features:
            for node_id, target, local, structure in extracted:
                encoded = node_id.encode("utf-8")
                max_node_id_bytes = max(max_node_id_bytes, len(encoded))
                node_ids.append(encoded)
                labels.append(target)
                family_indices.append(family_index)
                local_values.extend(local)
                structure_values.extend(structure)

    family_frame = pd.DataFrame(family_rows)
    if family_frame.empty:
        raise ValueError("No PANTHER families were read")

    critical_columns = [
        "duplicate_nodes",
        "endpoint_missing",
        "self_edges",
        "duplicate_parent",
        "family_mismatch",
        "relation_mismatch",
        "event_copy_mismatch",
        "branch_copy_mismatch",
        "edge_count_mismatch",
        "declared_root_mismatch",
        "unreachable_nodes",
    ]
    critical_totals = {
        column: int(family_frame[column].sum()) for column in critical_columns
    }
    source_consistency = int(
        ((family_frame["node_source_file_count"] == 1)
         & (family_frame["edge_source_file_count"] == 1)).all()
    )
    node_count = int(family_frame["n_nodes"].sum())
    edge_count = int(family_frame["n_edges"].sum())
    target_count = int(family_frame["n_targets"].sum())
    audit: dict[str, object] = {
        "scope": "quick" if quick_families is not None else "full",
        "nodes_path": str(nodes_path.resolve()),
        "edges_path": str(edges_path.resolve()),
        "node_fields": node_fields,
        "edge_fields": edge_fields,
        "node_rows": node_count,
        "edge_rows": edge_count,
        "families": int(len(family_frame)),
        "target_rows": target_count,
        "node_types": dict(node_types),
        "event_types": dict(event_types),
        "relation_types": dict(relation_types),
        "node_null_rates": {
            field: _safe_fraction(node_nulls[field], node_count) for field in node_fields
        },
        "edge_null_rates": {
            field: _safe_fraction(edge_nulls[field], edge_count) for field in edge_fields
        },
        "family_node_size": _quantile_summary(family_frame["n_nodes"]),
        "family_target_size": _quantile_summary(family_frame["n_targets"]),
        "critical_totals": critical_totals,
        "source_file_consistency": bool(source_consistency),
        "leakage_blacklist": sorted(LEAKAGE_FIELDS),
        "model_features": [*LOCAL_FEATURE_NAMES, *STRUCTURE_FEATURE_NAMES],
        "leakage_check_passed": True,
        "critical_checks_passed": bool(
            not any(critical_totals.values()) and source_consistency
        ),
    }
    if quick_families is None:
        audit["source_sha256"] = {
            "nodes": sha256_file(nodes_path),
            "edges": sha256_file(edges_path),
        }

    family_frame.to_csv(cache.families, index=False)
    cache.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    if write_features:
        n_rows = len(labels)
        local_array = np.frombuffer(local_values, dtype=np.float32).reshape(
            n_rows, len(LOCAL_FEATURE_NAMES)
        )
        structure_array = np.frombuffer(structure_values, dtype=np.float32).reshape(
            n_rows, len(STRUCTURE_FEATURE_NAMES)
        )
        node_id_array = np.asarray(node_ids, dtype=f"S{max_node_id_bytes}")
        np.savez_compressed(
            cache.features,
            X_local=local_array,
            X_structure=structure_array,
            y=np.frombuffer(labels, dtype=np.uint8),
            family_index=np.frombuffer(family_indices, dtype=np.uint32),
            node_id=node_id_array,
        )
        metadata = {
            "rows": n_rows,
            "families": len(family_frame),
            "local_features": list(LOCAL_FEATURE_NAMES),
            "structure_features": list(STRUCTURE_FEATURE_NAMES),
            "scope": audit["scope"],
            "source_size": {
                "nodes": nodes_path.stat().st_size,
                "edges": edges_path.stat().st_size,
            },
        }
        cache.metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return audit, family_frame


def cache_is_current(
    nodes_path: Path, edges_path: Path, cache: CachePaths, scope: str
) -> bool:
    """Check cheap cache invariants; full SHA-256 remains in the audit artifact."""
    if not all(path.exists() for path in (cache.features, cache.families, cache.audit, cache.metadata)):
        return False
    metadata = json.loads(cache.metadata.read_text(encoding="utf-8"))
    return bool(
        metadata.get("scope") == scope
        and metadata.get("source_size", {}).get("nodes") == nodes_path.stat().st_size
        and metadata.get("source_size", {}).get("edges") == edges_path.stat().st_size
        and metadata.get("local_features") == list(LOCAL_FEATURE_NAMES)
        and metadata.get("structure_features") == list(STRUCTURE_FEATURE_NAMES)
    )


def make_split_manifest(family_frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Create a deterministic 70/15/15 family-disjoint stratified split."""
    frame = family_frame.copy().sort_values("family_id").reset_index(drop=True)
    size_edges = np.unique(np.quantile(frame["n_nodes"], [0.2, 0.4, 0.6, 0.8]))
    rate_edges = np.unique(
        np.quantile(frame["duplication_rate"], [0.2, 0.4, 0.6, 0.8])
    )
    frame["size_bin"] = np.searchsorted(size_edges, frame["n_nodes"], side="right")
    frame["duplication_bin"] = np.searchsorted(
        rate_edges, frame["duplication_rate"], side="right"
    )
    frame["split"] = ""
    rng = np.random.default_rng(seed)
    for _, group in frame.groupby(["size_bin", "duplication_bin"], sort=True):
        positions = group.index.to_numpy(copy=True)
        rng.shuffle(positions)
        n_group = len(positions)
        if n_group >= 3:
            n_train = max(1, int(round(0.70 * n_group)))
            n_val = max(1, int(round(0.15 * n_group)))
            if n_train + n_val >= n_group:
                n_train = n_group - 2
                n_val = 1
        else:
            n_train = n_group
            n_val = 0
        frame.loc[positions[:n_train], "split"] = "train"
        frame.loc[positions[n_train : n_train + n_val], "split"] = "validation"
        frame.loc[positions[n_train + n_val :], "split"] = "test"

    if (frame["split"] == "").any():
        raise AssertionError("Every family must receive a split")
    if set(frame["split"]) != {"train", "validation", "test"}:
        raise ValueError("Dataset is too small to create all three family splits")
    frame["split_code"] = frame["split"].map(
        {"train": 0, "validation": 1, "test": 2}
    ).astype("int8")
    return frame


def feature_indices(names: Sequence[str], requested: Sequence[str]) -> np.ndarray:
    """Resolve a named, auditable feature subset."""
    lookup = {name: index for index, name in enumerate(names)}
    missing = sorted(set(requested) - set(lookup))
    if missing:
        raise KeyError(f"Unknown features: {missing}")
    return np.asarray([lookup[name] for name in requested], dtype=int)
