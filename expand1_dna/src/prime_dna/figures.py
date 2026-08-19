"""Standalone publication figures for the PRIME–PANTHER experiment."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MM_PER_INCH = 25.4
FIGURE_HEIGHT_MM = 50.0
FORMATS = ("png", "pdf", "svg", "emf")

DISPLAY_NAMES = {
    "training_prior": "Training prior",
    "local_logistic": "Local logistic",
    "local_hgb": "Local HGB",
    "lineage_logistic": "Lineage logistic",
    "lineage_topology_only": "Topology only",
    "lineage_scale_only": "Descendant scale",
    "lineage_branch": "Branch context",
    "lineage_composition_only": "Child composition",
    "lineage_without_topology": "Without topology",
    "lineage_without_scale": "Without scale",
    "lineage_without_branch": "Without branch",
    "lineage_without_composition": "Without composition",
    "lineage_full": "Lineage expert",
    "prime_fusion": "PRIME fusion",
    "prime_joint": "Local + lineage",
}

COLORS = {
    "neutral": "#666666",
    "neutral_light": "#B9B9B9",
    "lineage": "#6F93C3",
    "lineage_light": "#A9BBDD",
    "prime": "#B64342",
    "gain": "#6BA67A",
    "teal": "#2A9D8F",
    "orange": "#D28B32",
}


def _apply_style() -> None:
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.5,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 5.2,
        }
    )


def _new_figure(width_mm: float) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(
        figsize=(width_mm / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH),
        layout="constrained",
    )
    return fig, ax


def _inkscape_path() -> str:
    executable = shutil.which("inkscape")
    if executable:
        return executable
    fallback = Path(r"H:\Inkscape\bin\inkscape.exe")
    if fallback.exists():
        return str(fallback)
    raise RuntimeError(
        "EMF export requires Inkscape; add inkscape.exe to PATH or install it at "
        r"H:\Inkscape\bin\inkscape.exe"
    )


def _export_bundle(
    fig: plt.Figure,
    output_base: Path,
    *,
    width_mm: float,
    source_data: str,
    purpose: str,
) -> dict[str, object]:
    """Save one exact-height figure and convert its SVG to vector EMF."""
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    svg_path = output_base.with_suffix(".svg")
    emf_path = output_base.with_suffix(".emf")
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    plt.close(fig)
    subprocess.run(
        [
            _inkscape_path(),
            str(svg_path.resolve()),
            "--export-type=emf",
            f"--export-filename={emf_path.resolve()}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if not emf_path.exists() or emf_path.stat().st_size < 100:
        raise RuntimeError(f"Inkscape did not create a valid EMF: {emf_path}")
    return {
        "figure": output_base.name,
        "width_mm": float(width_mm),
        "height_mm": FIGURE_HEIGHT_MM,
        "formats": ",".join(FORMATS),
        "source_data": source_data,
        "purpose": purpose,
    }


def _write_manifest(rows: list[dict[str, object]], output_dir: Path) -> None:
    pd.DataFrame(rows).to_csv(output_dir / "FIGURE_MANIFEST.csv", index=False)


def save_main_figures(
    metrics: pd.DataFrame,
    family_metrics: pd.DataFrame,
    criteria: Mapping[str, object],
    placebo: pd.DataFrame,
    global_placebo: pd.DataFrame,
    calibration: pd.DataFrame,
    figure_dir: Path,
) -> list[dict[str, object]]:
    """Export the five main evidence panels as independent 50-mm figures."""
    _apply_style()
    output_dir = figure_dir / "main_panels"
    output_dir.mkdir(parents=True, exist_ok=True)
    lookup = metrics.set_index("model")
    best_local = str(criteria["best_local_baseline"])
    rows: list[dict[str, object]] = []

    # a | AUROC–AUPRC performance map. PRIME is the only filled point.
    models = [
        "training_prior",
        "local_logistic",
        "local_hgb",
        "lineage_branch",
        "lineage_topology_only",
        "lineage_scale_only",
        "prime_fusion",
    ]
    label_specs = {
        "training_prior": (4, -8, "Prior"),
        "local_logistic": (5, 7, "Local LR"),
        "local_hgb": (5, -9, "Local HGB"),
        "lineage_branch": (5, -8, "Branch"),
        "lineage_topology_only": (5, 6, "Topology"),
        "lineage_scale_only": (5, -9, "Scale"),
        "prime_fusion": (-5, -10, "PRIME"),
    }
    fig, ax = _new_figure(76)
    for model_name in models:
        x_value = float(lookup.loc[model_name, "AUROC"])
        y_value = float(lookup.loc[model_name, "AUPRC"])
        if model_name == "prime_fusion":
            ax.scatter(
                x_value,
                y_value,
                s=38,
                marker="D",
                facecolor=COLORS["prime"],
                edgecolor=COLORS["prime"],
                linewidth=0.8,
                zorder=4,
            )
        else:
            edge = (
                COLORS["neutral"]
                if model_name in {"training_prior", "local_logistic", "local_hgb"}
                else COLORS["lineage"]
            )
            marker = "s" if model_name == "local_hgb" else "o"
            ax.scatter(
                x_value,
                y_value,
                s=28,
                marker=marker,
                facecolor="none",
                edgecolor=edge,
                linewidth=1.0,
                zorder=3,
            )
        dx, dy, label = label_specs[model_name]
        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="right" if dx < 0 else "left",
            va="center",
            fontsize=5.3,
            color=COLORS["prime"] if model_name == "prime_fusion" else "#3F3F3F",
        )
    ax.set_xlim(0.46, 1.015)
    ax.set_ylim(0.20, 1.015)
    ax.set_xlabel("AUROC")
    ax.set_ylabel("Duplication AUPRC")
    ax.grid(color="#E6E6E6", linewidth=0.55)
    rows.append(
        _export_bundle(
            fig,
            output_dir / "a_performance_scatter",
            width_mm=76,
            source_data="metrics.csv",
            purpose="AUROC–AUPRC comparison across local, lineage-subset and PRIME models",
        )
    )

    # b | Per-family primary effect.
    family_pivot = family_metrics.pivot(
        index="family_id", columns="model", values="macro_f1"
    )
    family_gain = (family_pivot["prime_fusion"] - family_pivot[best_local]).dropna()
    bootstrap = criteria["bootstrap"]
    fig, ax = _new_figure(66)
    ax.hist(
        family_gain,
        bins=35,
        color="#A9C9B2",
        edgecolor="white",
        linewidth=0.35,
    )
    ax.axvline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.8)
    ax.axvline(family_gain.mean(), color=COLORS["prime"], linewidth=1.3)
    ax.axvspan(
        float(bootstrap["macro_f1_gain_ci_low"]),
        float(bootstrap["macro_f1_gain_ci_high"]),
        color=COLORS["prime"],
        alpha=0.14,
    )
    ax.text(
        0.03,
        0.96,
        f"mean={family_gain.mean():.3f}\n"
        f"95% CI [{float(bootstrap['macro_f1_gain_ci_low']):.3f}, "
        f"{float(bootstrap['macro_f1_gain_ci_high']):.3f}]\n"
        f"positive={(family_gain > 0).mean():.1%}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.2,
    )
    ax.set_xlabel("PRIME − local Macro-F1 per family")
    ax.set_ylabel("Held-out families")
    rows.append(
        _export_bundle(
            fig,
            output_dir / "b_family_macro_f1_gain",
            width_mm=66,
            source_data="family_metrics.csv; bootstrap_summary.json",
            purpose="Family-equal distribution and paired-bootstrap interval of the PRIME gain",
        )
    )

    # c | Correct alignment versus two negative controls.
    aligned_auc = float(lookup.loc["prime_fusion", "AUROC"])
    control_rows = [
        ("Aligned PRIME", aligned_auc, aligned_auc, aligned_auc, COLORS["prime"]),
        (
            "Within-family mismatch",
            float(placebo["AUROC"].mean()),
            float(placebo["AUROC"].quantile(0.025)),
            float(placebo["AUROC"].quantile(0.975)),
            COLORS["neutral"],
        ),
        (
            "Global mismatch",
            float(global_placebo["AUROC"].mean()),
            float(global_placebo["AUROC"].quantile(0.025)),
            float(global_placebo["AUROC"].quantile(0.975)),
            COLORS["lineage"],
        ),
    ]
    fig, ax = _new_figure(78)
    y_values = np.arange(len(control_rows))[::-1]
    for y_value, (label, estimate, low, high, color) in zip(y_values, control_rows):
        ax.plot([low, high], [y_value, y_value], color=color, linewidth=1.5)
        ax.scatter(estimate, y_value, color=color, s=26, zorder=3)
        ax.text(
            estimate + (0.012 if estimate < 0.95 else -0.012),
            y_value,
            f"{estimate:.3f}",
            ha="left" if estimate < 0.95 else "right",
            va="center",
            fontsize=5.2,
            color=color,
        )
    ax.set_yticks(y_values, [item[0] for item in control_rows])
    ax.set_xlim(0.50, 1.015)
    ax.set_xlabel("AUROC (mean and empirical 95% interval)")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.55)
    rows.append(
        _export_bundle(
            fig,
            output_dir / "c_context_alignment_control",
            width_mm=78,
            source_data="placebo_distribution.csv; global_placebo_distribution.csv; metrics.csv",
            purpose="Necessity of correct node–lineage alignment",
        )
    )

    # d | Feature-group sufficiency and redundancy.
    group_specs = [
        ("Topology", "lineage_topology_only", "lineage_without_topology"),
        ("Descendant scale", "lineage_scale_only", "lineage_without_scale"),
        ("Branch context", "lineage_branch", "lineage_without_branch"),
        ("Child composition", "lineage_composition_only", "lineage_without_composition"),
    ]
    matrix = np.asarray(
        [
            [
                float(lookup.loc[only_model, "AUROC"]),
                float(lookup.loc[without_model, "AUROC"]),
            ]
            for _, only_model, without_model in group_specs
        ]
    )
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "mechanism_blues", ["#F4F7FA", "#AFC5DE", "#376AA3"]
    )
    fig, ax = _new_figure(78)
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0.65, vmax=1.0)
    ax.set_xticks([0, 1], ["Only this group\n(sufficiency)", "All except group\n(redundancy)"])
    ax.set_yticks(np.arange(len(group_specs)), [item[0] for item in group_specs])
    ax.tick_params(length=0)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=6,
                fontweight="bold" if row_index == 3 else "normal",
                color="white" if value > 0.88 else "#222222",
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    rows.append(
        _export_bundle(
            fig,
            output_dir / "d_feature_group_mechanism",
            width_mm=78,
            source_data="metrics.csv",
            purpose="Feature-group sufficiency and conditional redundancy",
        )
    )

    # e | Probability reliability.
    fig, ax = _new_figure(60)
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=0.8)
    for model_name, color in [
        (best_local, COLORS["neutral"]),
        ("lineage_full", "#376AA3"),
        ("prime_fusion", COLORS["prime"]),
    ]:
        model_rows = calibration.loc[calibration["model"] == model_name].sort_values("bin")
        ax.plot(
            model_rows["mean_predicted_probability"],
            model_rows["observed_duplication_rate"],
            marker="o",
            markersize=2.2,
            linewidth=1.0,
            color=color,
            label=f"{DISPLAY_NAMES[model_name]} (ECE={float(lookup.loc[model_name, 'ece_15']):.3f})",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted duplication probability")
    ax.set_ylabel("Observed duplication rate")
    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        edgecolor="none",
        fontsize=4.7,
    )
    rows.append(
        _export_bundle(
            fig,
            output_dir / "e_probability_calibration",
            width_mm=60,
            source_data="calibration_table.csv; metrics.csv",
            purpose="Reliability and calibration cost of conservative fusion",
        )
    )

    _write_manifest(rows, output_dir)
    return rows


def save_additional_figures(
    metrics: pd.DataFrame,
    learning_curve: pd.DataFrame,
    permutation_importance: pd.DataFrame,
    split_sensitivity: pd.DataFrame,
    subgroup_metrics: pd.DataFrame,
    fusion_weight_scan: pd.DataFrame,
    bootstrap_distribution: pd.DataFrame,
    placebo: pd.DataFrame,
    global_placebo: pd.DataFrame,
    criteria: Mapping[str, object],
    figure_dir: Path,
) -> list[dict[str, object]]:
    """Export supplementary analytical charts as independent 50-mm figures."""
    _apply_style()
    output_dir = figure_dir / "additional_panels"
    output_dir.mkdir(parents=True, exist_ok=True)
    lookup = metrics.set_index("model")
    best_local = str(criteria["best_local_baseline"])
    rows: list[dict[str, object]] = []

    # Learning curve.
    fig, ax = _new_figure(70)
    for model_name, color in [
        ("local_hgb", COLORS["neutral"]),
        ("lineage_full", "#376AA3"),
        ("prime_fusion", COLORS["prime"]),
    ]:
        model_rows = learning_curve.loc[learning_curve["model"] == model_name].sort_values(
            "training_fraction_requested"
        )
        ax.plot(
            100 * model_rows["training_fraction_requested"],
            model_rows["AUROC"],
            marker="o",
            markersize=2.5,
            linewidth=1.0,
            color=color,
            label=DISPLAY_NAMES[model_name],
        )
    ax.set_xlabel("Training families used (%)")
    ax.set_ylabel("Test AUROC")
    ax.set_ylim(0.45, 1.015)
    ax.legend(loc="lower right")
    ax.grid(color="#E6E6E6", linewidth=0.5)
    rows.append(
        _export_bundle(
            fig,
            output_dir / "f_learning_curve",
            width_mm=70,
            source_data="learning_curve.csv",
            purpose="Family-level data efficiency",
        )
    )

    # Within-family permutation importance.
    importance = (
        permutation_importance.groupby("feature", as_index=False)
        .agg(mean_drop=("auroc_drop", "mean"), std_drop=("auroc_drop", "std"))
        .fillna(0.0)
        .sort_values("mean_drop")
    )
    fig, ax = _new_figure(88)
    ax.barh(
        importance["feature"],
        importance["mean_drop"],
        xerr=importance["std_drop"],
        color="#8FAAD0",
        error_kw={"elinewidth": 0.6, "capsize": 1.5},
    )
    ax.axvline(0, color="#555555", linewidth=0.6)
    ax.set_xlabel("Within-family permutation AUROC drop")
    ax.tick_params(axis="y", labelsize=4.5)
    rows.append(
        _export_bundle(
            fig,
            output_dir / "g_permutation_importance",
            width_mm=88,
            source_data="feature_permutation_importance.csv",
            purpose="Node–feature alignment importance for all lineage features",
        )
    )

    # Split-seed sensitivity.
    fig, ax = _new_figure(65)
    for model_name, color in [
        ("local_hgb", COLORS["neutral"]),
        ("lineage_full", "#376AA3"),
        ("prime_fusion", COLORS["prime"]),
    ]:
        model_rows = split_sensitivity.loc[split_sensitivity["model"] == model_name].sort_values(
            "split_seed"
        )
        ax.plot(
            model_rows["split_seed"].astype(str),
            model_rows["AUROC"],
            marker="o",
            markersize=2.5,
            linewidth=1.0,
            color=color,
            label=DISPLAY_NAMES[model_name],
        )
    ax.set_xlabel("Family split seed")
    ax.set_ylabel("Test AUROC")
    ax.set_ylim(0.45, 1.015)
    ax.legend(loc="lower right")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)
    rows.append(
        _export_bundle(
            fig,
            output_dir / "h_split_sensitivity",
            width_mm=65,
            source_data="split_sensitivity.csv",
            purpose="Robustness to family split seed",
        )
    )

    # All evaluable subgroup gains in one dot plot.
    subgroup_pivot = subgroup_metrics.pivot_table(
        index=["dimension", "level", "level_label"], columns="model", values="AUROC"
    )
    subgroup_gain = (
        subgroup_pivot["prime_fusion"] - subgroup_pivot[best_local]
    ).dropna().reset_index(name="gain")
    dimension_order = {
        "family_size_quintile": 0,
        "family_duplication_quintile": 1,
        "normalized_depth": 2,
        "root_status": 3,
    }
    dimension_names = {
        "family_size_quintile": "Size",
        "family_duplication_quintile": "Dup. rate",
        "normalized_depth": "Depth",
        "root_status": "Root status",
    }
    subgroup_gain["dimension_order"] = subgroup_gain["dimension"].map(dimension_order).fillna(99)
    subgroup_gain = subgroup_gain.sort_values(["dimension_order", "level"])
    subgroup_gain["display"] = subgroup_gain.apply(
        lambda row: f"{dimension_names.get(row['dimension'], row['dimension'])}: {row['level_label']}",
        axis=1,
    )
    fig, ax = _new_figure(90)
    y_values = np.arange(len(subgroup_gain))[::-1]
    color_map = {
        "family_size_quintile": "#6F93C3",
        "family_duplication_quintile": "#D28B32",
        "normalized_depth": "#6BA67A",
        "root_status": "#8C6BB1",
    }
    for y_value, (_, row) in zip(y_values, subgroup_gain.iterrows()):
        ax.scatter(
            float(row["gain"]),
            y_value,
            s=18,
            color=color_map.get(str(row["dimension"]), COLORS["neutral"]),
        )
    ax.set_yticks(y_values, subgroup_gain["display"])
    ax.tick_params(axis="y", labelsize=4.2)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=0.7)
    ax.set_xlabel("PRIME − local AUROC")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.5)
    rows.append(
        _export_bundle(
            fig,
            output_dir / "i_subgroup_generalization",
            width_mm=90,
            source_data="subgroup_metrics.csv",
            purpose="PRIME gain across every evaluable structural subgroup",
        )
    )

    # Class-specific error balance.
    comparison_models = ["local_hgb", "lineage_full", "prime_fusion", "prime_joint"]
    x_values = np.arange(len(comparison_models))
    width = 0.34
    fig, ax = _new_figure(78)
    ax.bar(
        x_values - width / 2,
        [float(lookup.loc[name, "recall_duplication"]) for name in comparison_models],
        width,
        color="#D28B32",
        label="Duplication recall",
    )
    ax.bar(
        x_values + width / 2,
        [float(lookup.loc[name, "specificity_speciation"]) for name in comparison_models],
        width,
        color="#6F93C3",
        label="Speciation specificity",
    )
    ax.set_xticks(x_values, [DISPLAY_NAMES[name] for name in comparison_models], rotation=18, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Rate")
    ax.legend(loc="lower right")
    rows.append(
        _export_bundle(
            fig,
            output_dir / "j_class_error_balance",
            width_mm=78,
            source_data="metrics.csv",
            purpose="Class-specific recall and specificity",
        )
    )

    # Fusion-weight validation scan.
    selected_weight = float(criteria["selected_fusion_weight"])
    selected_row = fusion_weight_scan.loc[
        np.isclose(fusion_weight_scan["weight"], selected_weight)
    ].iloc[0]
    fig, ax = _new_figure(66)
    ax.plot(
        fusion_weight_scan["weight"],
        fusion_weight_scan["validation_log_loss"],
        color="#6F93C3",
        linewidth=1.1,
    )
    ax.scatter(
        selected_weight,
        float(selected_row["validation_log_loss"]),
        s=28,
        color=COLORS["prime"],
        zorder=3,
    )
    ax.axvline(selected_weight, color=COLORS["prime"], linestyle="--", linewidth=0.7)
    ax.set_xlabel("Lineage fusion weight")
    ax.set_ylabel("Validation log loss")
    ax.grid(color="#E6E6E6", linewidth=0.5)
    rows.append(
        _export_bundle(
            fig,
            output_dir / "k_fusion_weight_selection",
            width_mm=66,
            source_data="fusion_weight_scan.csv",
            purpose="Validation-only selection of the conservative fusion weight",
        )
    )

    # Paired-bootstrap distributions, one metric per standalone figure.
    for filename, column, xlabel, color in [
        (
            "l_bootstrap_macro_f1_gain",
            "mean_macro_f1_gain",
            "Bootstrap mean family Macro-F1 gain",
            COLORS["gain"],
        ),
        (
            "m_bootstrap_log_loss_gain",
            "mean_log_loss_gain",
            "Bootstrap mean family log-loss improvement",
            COLORS["teal"],
        ),
    ]:
        values = bootstrap_distribution[column]
        fig, ax = _new_figure(66)
        ax.hist(values, bins=35, color=color, edgecolor="white", linewidth=0.35)
        ax.axvline(values.mean(), color=COLORS["prime"], linewidth=1.1)
        ax.axvline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Bootstrap repetitions")
        rows.append(
            _export_bundle(
                fig,
                output_dir / filename,
                width_mm=66,
                source_data="bootstrap_distribution.csv",
                purpose="Paired family-bootstrap uncertainty distribution",
            )
        )

    # Raw alignment-placebo distributions.
    fig, ax = _new_figure(68)
    low = min(float(placebo["AUROC"].min()), float(global_placebo["AUROC"].min()))
    high = max(float(placebo["AUROC"].max()), float(global_placebo["AUROC"].max()))
    bins = np.linspace(low - 0.002, high + 0.002, 24)
    ax.hist(
        global_placebo["AUROC"],
        bins=bins,
        alpha=0.72,
        color="#8FAAD0",
        label="Global mismatch",
    )
    ax.hist(
        placebo["AUROC"],
        bins=bins,
        alpha=0.72,
        color="#777777",
        label="Within-family mismatch",
    )
    ax.set_xlabel("Placebo AUROC")
    ax.set_ylabel("Repetitions")
    ax.legend(loc="upper left")
    rows.append(
        _export_bundle(
            fig,
            output_dir / "n_placebo_distributions",
            width_mm=68,
            source_data="placebo_distribution.csv; global_placebo_distribution.csv",
            purpose="Full distributions of alignment negative controls",
        )
    )

    _write_manifest(rows, output_dir)
    return rows


def write_figure_index(figure_dir: Path) -> None:
    """Write a compact index across main and supplementary standalone figures."""
    main_manifest = pd.read_csv(figure_dir / "main_panels" / "FIGURE_MANIFEST.csv")
    additional_manifest = pd.read_csv(
        figure_dir / "additional_panels" / "FIGURE_MANIFEST.csv"
    )
    lines = [
        "# PRIME × PANTHER 独立图件索引",
        "",
        "所有图件高度均为 50 mm，无图内标题，并导出 PNG（600 dpi）、PDF、SVG 和 EMF。",
        "EMF 由 Inkscape 从 Matplotlib 生成的 SVG 转换，SVG/PDF 保留可编辑矢量文字。",
        "",
        "| 分组 | 图件 | 宽度 (mm) | 数据来源 | 含义 |",
        "|---|---|---:|---|---|",
    ]
    for group_name, frame in [("主图", main_manifest), ("扩展", additional_manifest)]:
        for row in frame.itertuples(index=False):
            lines.append(
                f"| {group_name} | `{row.figure}` | {float(row.width_mm):.0f} | "
                f"`{row.source_data}` | {row.purpose} |"
            )
    lines.append("")
    (figure_dir / "FIGURE_INDEX.md").write_text("\n".join(lines), encoding="utf-8")
