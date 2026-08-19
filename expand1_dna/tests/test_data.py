from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prime_dna.data import (  # noqa: E402
    CachePaths,
    STRUCTURE_FEATURE_GROUPS,
    STRUCTURE_FEATURE_NAMES,
    _tree_features,
    assert_feature_contract,
    build_feature_cache,
    grouped_csv,
    make_split_manifest,
)


NODE_FIELDS = [
    "node_id", "family_id", "raw_node_id", "label", "node_type", "is_root",
    "event_type", "event_type_raw", "duplication_flag", "species",
    "organism_code", "taxon_id", "external_ids", "branch_length", "confidence",
    "nhx_attributes", "source_file",
]
EDGE_FIELDS = [
    "source_id", "target_id", "family_id", "relation", "parent_event_type",
    "child_branch_length", "source_file",
]


def synthetic_family(family_id: str):
    source_file = f"{family_id}.tree"
    common = {
        "family_id": family_id,
        "duplication_flag": "",
        "taxon_id": "",
        "confidence": "",
        "source_file": source_file,
    }
    nodes = [
        {
            **common, "node_id": f"{family_id}:AN0", "raw_node_id": "AN0",
            "label": "", "node_type": "ancestral_gene", "is_root": "True",
            "event_type": "speciation", "event_type_raw": "0>1",
            "species": "Eukaryota", "organism_code": "", "external_ids": "{}",
            "branch_length": "", "nhx_attributes": '{"Ev": "0>1", "ID": "AN0"}',
        },
        {
            **common, "node_id": f"{family_id}:AN1", "raw_node_id": "AN1",
            "label": "", "node_type": "ancestral_gene", "is_root": "False",
            "event_type": "duplication", "event_type_raw": "1>0",
            "species": "Mammalia", "organism_code": "", "external_ids": "{}",
            "branch_length": "0.2", "nhx_attributes": '{"Ev": "1>0", "ID": "AN1"}',
        },
    ]
    for suffix, organism, branch in [("L1", "HUMAN", "0.3"), ("L2", "MOUSE", "0.4"), ("L3", "YEAST", "0.5")]:
        nodes.append(
            {
                **common, "node_id": f"{family_id}:{suffix}", "raw_node_id": suffix,
                "label": f"gene_{suffix}", "node_type": "extant_gene",
                "is_root": "False", "event_type": "", "event_type_raw": "",
                "species": "", "organism_code": organism,
                "external_ids": f'{{"UniProt": "{suffix}"}}',
                "branch_length": branch, "nhx_attributes": f'{{"ID": "{suffix}"}}',
            }
        )
    edges = [
        {
            "source_id": f"{family_id}:AN0", "target_id": f"{family_id}:AN1",
            "family_id": family_id, "relation": "parent_of",
            "parent_event_type": "speciation", "child_branch_length": "0.2",
            "source_file": source_file,
        },
        {
            "source_id": f"{family_id}:AN0", "target_id": f"{family_id}:L3",
            "family_id": family_id, "relation": "parent_of",
            "parent_event_type": "speciation", "child_branch_length": "0.5",
            "source_file": source_file,
        },
        {
            "source_id": f"{family_id}:AN1", "target_id": f"{family_id}:L1",
            "family_id": family_id, "relation": "parent_of",
            "parent_event_type": "duplication", "child_branch_length": "0.3",
            "source_file": source_file,
        },
        {
            "source_id": f"{family_id}:AN1", "target_id": f"{family_id}:L2",
            "family_id": family_id, "relation": "parent_of",
            "parent_event_type": "duplication", "child_branch_length": "0.4",
            "source_file": source_file,
        },
    ]
    return nodes, edges


class DataTests(unittest.TestCase):
    def test_tree_features_have_expected_depth_and_subtree_values(self):
        nodes, edges = synthetic_family("PTHR00001")
        rows, summary = _tree_features("PTHR00001", nodes, edges)
        by_node = {node_id: (target, local, structure) for node_id, target, local, structure in rows}
        root = by_node["PTHR00001:AN0"]
        child = by_node["PTHR00001:AN1"]
        self.assertEqual(root[0], 0)
        self.assertEqual(root[1], [0.0, 1.0, 1.0])
        self.assertEqual(root[2][1], 0.0)
        self.assertAlmostEqual(root[2][4], np.log1p(5))
        self.assertAlmostEqual(root[2][5], np.log1p(3))
        self.assertEqual(child[0], 1)
        self.assertAlmostEqual(child[2][1], 0.5)
        self.assertEqual(child[2][12], 1.0)
        for name in ["duplicate_nodes", "endpoint_missing", "edge_count_mismatch", "declared_root_mismatch", "unreachable_nodes"]:
            self.assertEqual(summary[name], 0)

    def test_grouped_csv_preserves_quoted_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nodes.csv"
            nodes, _ = synthetic_family("PTHR00001")
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=NODE_FIELDS)
                writer.writeheader()
                writer.writerows(nodes)
            groups = list(grouped_csv(path))
            self.assertEqual(groups[0][1][0]["nhx_attributes"], '{"Ev": "0>1", "ID": "AN0"}')

    def test_leakage_contract_fails_closed(self):
        assert_feature_contract(["incoming_branch_length", "log_outdegree"])
        with self.assertRaises(ValueError):
            assert_feature_contract(["incoming_branch_length", "parent_event_type"])

    def test_structure_feature_groups_are_complete_and_disjoint(self):
        flattened = [
            feature
            for group in STRUCTURE_FEATURE_GROUPS.values()
            for feature in group
        ]
        self.assertEqual(set(flattened), set(STRUCTURE_FEATURE_NAMES))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_family_split_is_deterministic_and_disjoint(self):
        frame = pd.DataFrame(
            {
                "family_id": [f"PTHR{i:05d}" for i in range(100)],
                "n_nodes": np.arange(6, 106),
                "duplication_rate": np.linspace(0.05, 0.8, 100),
            }
        )
        first = make_split_manifest(frame, 42)
        second = make_split_manifest(frame, 42)
        self.assertTrue(first[["family_id", "split"]].equals(second[["family_id", "split"]]))
        sets = [set(first.loc[first["split"] == split, "family_id"]) for split in ["train", "validation", "test"]]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_feature_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nodes_path = root / "nodes.csv"
            edges_path = root / "edges.csv"
            all_nodes, all_edges = [], []
            for index in range(5):
                nodes, edges = synthetic_family(f"PTHR{index:05d}")
                all_nodes.extend(nodes)
                all_edges.extend(edges)
            with nodes_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=NODE_FIELDS)
                writer.writeheader()
                writer.writerows(all_nodes)
            with edges_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=EDGE_FIELDS)
                writer.writeheader()
                writer.writerows(all_edges)
            cache = CachePaths(root / "cache")
            audit, families = build_feature_cache(nodes_path, edges_path, cache)
            with np.load(cache.features, allow_pickle=False) as arrays:
                self.assertTrue(audit["critical_checks_passed"])
                self.assertEqual(len(families), 5)
                self.assertEqual(arrays["X_local"].shape, (10, 3))
                self.assertEqual(arrays["X_structure"].shape[0], 10)
                self.assertEqual(arrays["y"].tolist(), [0, 1] * 5)


if __name__ == "__main__":
    unittest.main()
