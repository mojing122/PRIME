"""Run the PRIME-to-PANTHER bounded-generalization feasibility experiment."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prime_dna.data import (  # noqa: E402
    CachePaths,
    build_feature_cache,
    cache_is_current,
    make_split_manifest,
)
from prime_dna.modeling import run_experiment  # noqa: E402
from prime_dna.figures import (  # noqa: E402
    save_additional_figures,
    save_main_figures,
    write_figure_index,
)
from prime_dna.reporting import (  # noqa: E402
    write_data_audit_markdown,
    write_results_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit PANTHER trees and test leakage-safe PRIME lineage augmentation "
            "on masked speciation/duplication annotations."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["audit", "experiment", "all"],
        default="all",
        help="audit: build/check data cache; experiment: use cache; all: both",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use the first 400 families and reduced resampling for a smoke run.",
    )
    parser.add_argument(
        "--rebuild-cache", action="store_true", help="Ignore an existing valid cache."
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        default=PROJECT_ROOT / "data" / "PANTHER_nodes.csv",
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=PROJECT_ROOT / "data" / "PANTHER_edges.csv",
    )
    parser.add_argument(
        "--sensitivity-seeds",
        type=int,
        default=None,
        help="Number of consecutive family-split seeds, including --seed.",
    )
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=None,
        help="Within-family feature permutation repeats (quick: 2; full: 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = "quick" if args.quick else "full"
    cache = CachePaths(PROJECT_ROOT / "cache" / profile)
    result_dir = PROJECT_ROOT / "results" / profile
    figure_dir = PROJECT_ROOT / "figures" / profile
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    scope = "quick" if args.quick else "full"
    current = cache_is_current(args.nodes, args.edges, cache, scope)
    if args.mode in {"audit", "all"} and (args.rebuild_cache or not current):
        print(f"Building {profile} leakage-safe feature cache...")
        build_feature_cache(
            args.nodes,
            args.edges,
            cache,
            quick_families=400 if args.quick else None,
            write_features=True,
        )
        current = True
    if args.mode == "experiment" and not current:
        raise SystemExit("No current cache. Run with --mode all or --mode audit first.")

    audit = json.loads(cache.audit.read_text(encoding="utf-8"))
    families = __import__("pandas").read_csv(cache.families)
    manifest = make_split_manifest(families, args.seed)
    if manifest["family_id"].tolist() != families["family_id"].tolist():
        raise AssertionError("Sorted family order changed between cache and split manifest")
    manifest.to_csv(result_dir / "split_manifest.csv", index=False)
    shutil.copy2(cache.audit, result_dir / "data_audit.json")
    write_data_audit_markdown(audit, result_dir / "DATA_AUDIT.md")
    print(
        f"Audit complete: {audit['families']:,} families, "
        f"{audit['target_rows']:,} target nodes."
    )
    if args.mode == "audit":
        return

    bootstrap_repeats = 200 if args.quick else 2_000
    placebo_repeats = 10 if args.quick else 100
    permutation_repeats = (
        args.permutation_repeats
        if args.permutation_repeats is not None
        else (2 if args.quick else 5)
    )
    sensitivity_count = (
        args.sensitivity_seeds
        if args.sensitivity_seeds is not None
        else (2 if args.quick else 3)
    )
    if sensitivity_count < 1:
        raise SystemExit("--sensitivity-seeds must be at least 1")
    if permutation_repeats < 1:
        raise SystemExit("--permutation-repeats must be at least 1")
    sensitivity_seeds = tuple(args.seed + offset for offset in range(sensitivity_count))
    learning_fractions = (0.25, 0.50, 1.00) if args.quick else (0.10, 0.25, 0.50, 1.00)
    print("Training family-disjoint baselines and PRIME models...")
    result = run_experiment(
        cache,
        manifest,
        audit,
        result_dir,
        seed=args.seed,
        bootstrap_repeats=bootstrap_repeats,
        placebo_repeats=placebo_repeats,
        permutation_repeats=permutation_repeats,
        sensitivity_seeds=sensitivity_seeds,
        learning_fractions=learning_fractions,
    )
    placebo = __import__("pandas").read_csv(result_dir / "placebo_distribution.csv")
    global_placebo = __import__("pandas").read_csv(
        result_dir / "global_placebo_distribution.csv"
    )
    subgroup_metrics = __import__("pandas").read_csv(
        result_dir / "subgroup_metrics.csv"
    )
    calibration = __import__("pandas").read_csv(result_dir / "calibration_table.csv")
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
        __import__("pandas").read_csv(result_dir / "learning_curve.csv"),
        __import__("pandas").read_csv(
            result_dir / "feature_permutation_importance.csv"
        ),
        __import__("pandas").read_csv(result_dir / "split_sensitivity.csv"),
        subgroup_metrics,
        __import__("pandas").read_csv(result_dir / "fusion_weight_scan.csv"),
        __import__("pandas").read_csv(result_dir / "bootstrap_distribution.csv"),
        placebo,
        global_placebo,
        result.criteria,
        figure_dir,
    )
    write_figure_index(figure_dir)
    write_results_markdown(
        audit,
        result.metrics,
        result.criteria,
        result.run_manifest,
        result_dir / "RESULTS.md",
    )
    print(f"Conclusion: {result.criteria['conclusion']}")
    print(f"Results: {result_dir}")


if __name__ == "__main__":
    main()
