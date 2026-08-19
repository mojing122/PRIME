from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prime_dna.data import CachePaths, build_feature_cache, make_split_manifest  # noqa: E402
from prime_dna.figures import (  # noqa: E402
    save_additional_figures,
    save_main_figures,
    write_figure_index,
)
from prime_dna.modeling import run_experiment  # noqa: E402
from test_data import EDGE_FIELDS, NODE_FIELDS, synthetic_family  # noqa: E402


class IntegrationTests(unittest.TestCase):
    def test_extended_pipeline_writes_metrics_controls_and_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nodes_path = root / "nodes.csv"
            edges_path = root / "edges.csv"
            all_nodes, all_edges = [], []
            for index in range(50):
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
            manifest = make_split_manifest(families, 42)
            result_dir = root / "results"
            figure_dir = root / "figures"
            result = run_experiment(
                cache,
                manifest,
                audit,
                result_dir,
                seed=42,
                bootstrap_repeats=20,
                placebo_repeats=5,
                permutation_repeats=1,
                sensitivity_seeds=(42,),
                learning_fractions=(1.0,),
            )

            placebo = pd.read_csv(result_dir / "placebo_distribution.csv")
            global_placebo = pd.read_csv(
                result_dir / "global_placebo_distribution.csv"
            )
            subgroup = pd.read_csv(result_dir / "subgroup_metrics.csv")
            calibration = pd.read_csv(result_dir / "calibration_table.csv")
            save_main_figures(
                result.metrics,
                result.family_metrics,
                result.criteria,
                placebo,
                global_placebo,
                calibration,
                figure_dir,
            )
            save_additional_figures(
                result.metrics,
                pd.read_csv(result_dir / "learning_curve.csv"),
                pd.read_csv(result_dir / "feature_permutation_importance.csv"),
                pd.read_csv(result_dir / "split_sensitivity.csv"),
                subgroup,
                pd.read_csv(result_dir / "fusion_weight_scan.csv"),
                pd.read_csv(result_dir / "bootstrap_distribution.csv"),
                placebo,
                global_placebo,
                result.criteria,
                figure_dir,
            )
            write_figure_index(figure_dir)

            expected = [
                "metrics.csv",
                "family_metrics.csv",
                "subgroup_metrics.csv",
                "calibration_table.csv",
                "feature_permutation_importance.csv",
                "learning_curve.csv",
                "split_sensitivity.csv",
                "global_placebo_distribution.csv",
                "extended_analysis_summary.json",
            ]
            for filename in expected:
                self.assertTrue((result_dir / filename).exists(), filename)
            for extension in ("png", "pdf", "svg", "emf"):
                self.assertTrue(
                    (figure_dir / "main_panels" / f"a_performance_scatter.{extension}").exists()
                )
                self.assertTrue(
                    (figure_dir / "additional_panels" / f"f_learning_curve.{extension}").exists()
                )
            self.assertTrue((figure_dir / "FIGURE_INDEX.md").exists())
            summary = json.loads(
                (result_dir / "extended_analysis_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                summary["status"],
                "posthoc_robustness_not_used_for_primary_model_selection",
            )


if __name__ == "__main__":
    unittest.main()
