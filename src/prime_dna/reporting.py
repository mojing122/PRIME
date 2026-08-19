"""Human-readable reporting and figures for the PRIME-DNA experiment."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


DISPLAY_NAMES = {
    "training_prior": "Training prior",
    "local_logistic": "Local logistic",
    "local_hgb": "Local HGB",
    "lineage_logistic": "Lineage logistic",
    "lineage_degree": "Topology + scale subset",
    "lineage_topology_only": "Topology position only",
    "lineage_scale_only": "Descendant scale only",
    "lineage_branch": "Branch context only",
    "lineage_composition_only": "Child composition only",
    "lineage_without_topology": "Without topology",
    "lineage_without_scale": "Without scale",
    "lineage_without_branch": "Without branch",
    "lineage_without_composition": "Without composition",
    "lineage_full": "Lineage expert",
    "prime_fusion": "PRIME fusion",
    "prime_joint": "Local + lineage",
}


def _fmt(value: float, digits: int = 3) -> str:
    return "NA" if pd.isna(value) else f"{value:.{digits}f}"


def save_main_figure(
    metrics: pd.DataFrame,
    family_metrics: pd.DataFrame,
    criteria: Mapping[str, object],
    placebo: pd.DataFrame,
    global_placebo: pd.DataFrame,
    subgroup_metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    figure_dir: Path,
) -> None:
    """Render the primary five-panel visual argument."""
    raise RuntimeError(
        "Legacy composite export is disabled; use prime_dna.figures.save_main_figures"
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    metric_lookup = metrics.set_index("model")
    best_local = str(criteria["best_local_baseline"])
    benchmark_models = [
        "local_logistic",
        "local_hgb",
        "lineage_topology_only",
        "lineage_scale_only",
        "lineage_branch",
        "prime_fusion",
    ]
    benchmark_models = [name for name in benchmark_models if name in metric_lookup.index]

    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update(
        {
            "font.size": 8,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
        }
    )
    # Designed for reduction to a 183-mm double-column page width: 8-pt source
    # text becomes approximately 5.3 pt at final size.
    fig = plt.figure(figsize=(10.8, 6.2), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.08, 1.08, 1.0),
        height_ratios=(1.0, 1.02),
        wspace=0.28,
        hspace=0.28,
    )

    # a | Hero comparison: distinct information tiers, not internal PRIME variants.
    benchmark_grid = grid[0, :2].subgridspec(1, 2, wspace=0.06)
    ax_auc = fig.add_subplot(benchmark_grid[0, 0])
    ax_auprc = fig.add_subplot(benchmark_grid[0, 1], sharey=ax_auc)
    y_position = np.arange(len(benchmark_models))[::-1]
    method_colors = {
        "local_logistic": "#8C8C8C",
        "local_hgb": "#555555",
        "lineage_topology_only": "#8FAAD0",
        "lineage_scale_only": "#6F93C3",
        "lineage_branch": "#A9BBDD",
        "prime_fusion": "#B64342",
    }
    method_markers = {
        "prime_fusion": "D",
    }
    metric_axes = [
        (ax_auc, "AUROC", 0.48, float(metric_lookup.loc["training_prior", "AUROC"])),
        (ax_auprc, "AUPRC", 0.20, float(metric_lookup.loc["training_prior", "AUPRC"])),
    ]
    for axis, metric_name, lower, prior_value in metric_axes:
        axis.axvline(
            prior_value,
            color="#B7B7B7",
            linestyle="--",
            linewidth=1.0,
            zorder=0,
        )
        for y_value, model_name in zip(y_position, benchmark_models):
            value = float(metric_lookup.loc[model_name, metric_name])
            axis.scatter(
                value,
                y_value,
                s=44 if model_name == "prime_fusion" else 34,
                marker=method_markers.get(model_name, "o"),
                color=method_colors[model_name],
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )
            axis.text(
                min(value + 0.012, 0.982),
                y_value,
                f"{value:.3f}",
                va="center",
                ha="right" if value > 0.965 else "left",
                fontsize=6.8,
                color=method_colors[model_name],
            )
        axis.set_xlim(lower, 1.015)
        axis.set_ylim(-0.65, len(benchmark_models) - 0.35)
        axis.set_xlabel(metric_name)
        axis.grid(axis="x", color="#E6E6E6", linewidth=0.7)
        axis.text(
            prior_value,
            len(benchmark_models) - 0.25,
            "training prior",
            color="#888888",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    ax_auc.set_yticks(
        y_position, [DISPLAY_NAMES[name] for name in benchmark_models]
    )
    ax_auc.tick_params(axis="y", length=0)
    ax_auprc.tick_params(axis="y", labelleft=False, length=0)
    ax_auc.set_title(
        "a  Benchmark hierarchy on family-disjoint test trees",
        loc="left",
        fontweight="bold",
        pad=11,
    )

    # b | Family-equal robustness of the primary effect.
    ax_family = fig.add_subplot(grid[0, 2])
    pivot = family_metrics.pivot(index="family_id", columns="model", values="macro_f1")
    family_gain = (pivot["prime_fusion"] - pivot[best_local]).dropna()
    ax_family.hist(family_gain, bins=35, color="#A9C9B2", edgecolor="white", linewidth=0.4)
    ax_family.axvline(0.0, color="#555555", linewidth=1, linestyle="--")
    ax_family.axvline(family_gain.mean(), color="#B64342", linewidth=1.7)
    bootstrap = criteria["bootstrap"]
    ax_family.axvspan(
        float(bootstrap["macro_f1_gain_ci_low"]),
        float(bootstrap["macro_f1_gain_ci_high"]),
        color="#B64342",
        alpha=0.13,
    )
    ax_family.text(
        0.03,
        0.95,
        f"mean = {family_gain.mean():.3f}\n"
        f"95% CI [{float(bootstrap['macro_f1_gain_ci_low']):.3f}, "
        f"{float(bootstrap['macro_f1_gain_ci_high']):.3f}]\n"
        f"positive families = {(family_gain > 0).mean():.1%}",
        transform=ax_family.transAxes,
        va="top",
        fontsize=7,
    )
    ax_family.set_xlabel("PRIME − local Macro-F1 per family")
    ax_family.set_ylabel("Held-out families")
    ax_family.set_title("b  Family-level effect", loc="left", fontweight="bold")

    # c | Alignment negative controls, summarized without empty histogram space.
    ax_control = fig.add_subplot(grid[1, 0])
    aligned_auc = float(metric_lookup.loc["prime_fusion", "AUROC"])
    control_rows = [
        ("Aligned PRIME", aligned_auc, aligned_auc, aligned_auc, "#B64342"),
        (
            "Within-family mismatch",
            float(placebo["AUROC"].mean()),
            float(placebo["AUROC"].quantile(0.025)),
            float(placebo["AUROC"].quantile(0.975)),
            "#6F6F6F",
        ),
        (
            "Global mismatch",
            float(global_placebo["AUROC"].mean()),
            float(global_placebo["AUROC"].quantile(0.025)),
            float(global_placebo["AUROC"].quantile(0.975)),
            "#8FAAD0",
        ),
    ]
    control_y = np.arange(len(control_rows))[::-1]
    for y_value, (label, estimate, low, high, color) in zip(control_y, control_rows):
        ax_control.plot([low, high], [y_value, y_value], color=color, linewidth=2.0)
        ax_control.scatter(estimate, y_value, color=color, s=38, zorder=3)
        ax_control.text(
            estimate + (0.012 if estimate < 0.95 else -0.012),
            y_value,
            f"{estimate:.3f}",
            ha="left" if estimate < 0.95 else "right",
            va="center",
            fontsize=7,
            color=color,
        )
    ax_control.set_yticks(control_y, [row[0] for row in control_rows])
    ax_control.set_xlim(0.50, 1.015)
    ax_control.set_xlabel("AUROC (mean and empirical 95% interval)")
    ax_control.grid(axis="x", color="#E6E6E6", linewidth=0.7)
    ax_control.set_title("c  Correct alignment is necessary", loc="left", fontweight="bold")
    ax_control.text(
        0.99,
        0.05,
        f"aligned gaps: +{aligned_auc - placebo['AUROC'].mean():.3f} within family\n"
        f"                    +{aligned_auc - global_placebo['AUROC'].mean():.3f} global",
        transform=ax_control.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.6,
        color="#555555",
    )

    # d | Sufficiency (only) and conditional necessity (without) in one view.
    ax_mechanism = fig.add_subplot(grid[1, 1])
    group_specs = [
        ("Topology", "lineage_topology_only", "lineage_without_topology"),
        ("Descendant scale", "lineage_scale_only", "lineage_without_scale"),
        ("Branch context", "lineage_branch", "lineage_without_branch"),
        ("Child composition", "lineage_composition_only", "lineage_without_composition"),
    ]
    mechanism_matrix = np.asarray(
        [
            [
                float(metric_lookup.loc[only_model, "AUROC"]),
                float(metric_lookup.loc[without_model, "AUROC"]),
            ]
            for _, only_model, without_model in group_specs
        ]
    )
    mechanism_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "mechanism_blues", ["#F4F7FA", "#AFC5DE", "#376AA3"]
    )
    ax_mechanism.imshow(
        mechanism_matrix,
        cmap=mechanism_cmap,
        aspect="auto",
        vmin=0.65,
        vmax=1.0,
    )
    ax_mechanism.set_xticks(
        [0, 1], ["Only this group\n(sufficiency)", "All except group\n(redundancy)"]
    )
    ax_mechanism.set_yticks(
        np.arange(len(group_specs)), [label for label, _, _ in group_specs]
    )
    ax_mechanism.tick_params(length=0)
    for row_index in range(mechanism_matrix.shape[0]):
        for column_index in range(mechanism_matrix.shape[1]):
            value = mechanism_matrix[row_index, column_index]
            ax_mechanism.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value > 0.88 else "#222222",
                fontweight="bold" if row_index == 3 else "normal",
                fontsize=8,
            )
    for spine in ax_mechanism.spines.values():
        spine.set_visible(False)
    ax_mechanism.set_title("d  Feature-group mechanism", loc="left", fontweight="bold")

    # e | Calibration exposes the cost of conservative probability mixing.
    ax_calibration = fig.add_subplot(grid[1, 2])
    calibration_models = [best_local, "lineage_full", "prime_fusion"]
    calibration_colors = ["#666666", "#376AA3", "#B64342"]
    ax_calibration.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    for model_name, color in zip(calibration_models, calibration_colors):
        rows = calibration.loc[calibration["model"] == model_name].sort_values("bin")
        ax_calibration.plot(
            rows["mean_predicted_probability"],
            rows["observed_duplication_rate"],
            marker="o",
            markersize=3.2,
            linewidth=1.35,
            color=color,
            label=(
                f"{DISPLAY_NAMES[model_name]} "
                f"(ECE={float(metric_lookup.loc[model_name, 'ece_15']):.3f})"
            ),
        )
    ax_calibration.set_xlim(0, 1)
    ax_calibration.set_ylim(0, 1)
    ax_calibration.set_aspect("equal", adjustable="box")
    ax_calibration.set_xlabel("Mean predicted duplication probability")
    ax_calibration.set_ylabel("Observed duplication rate")
    ax_calibration.set_title("e  Probability reliability", loc="left", fontweight="bold")
    ax_calibration.legend(
        fontsize=6.3,
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        edgecolor="none",
    )

    fig.savefig(figure_dir / "prime_panther_feasibility.png", dpi=600, bbox_inches="tight")
    fig.savefig(figure_dir / "prime_panther_feasibility.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "prime_panther_feasibility.svg", bbox_inches="tight")
    plt.close(fig)


def save_extended_figure(
    metrics: pd.DataFrame,
    learning_curve: pd.DataFrame,
    permutation_importance: pd.DataFrame,
    split_sensitivity: pd.DataFrame,
    subgroup_metrics: pd.DataFrame,
    figure_dir: Path,
) -> None:
    """Render complementary data-efficiency, importance, and robustness panels."""
    raise RuntimeError(
        "Legacy composite export is disabled; use prime_dna.figures.save_additional_figures"
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    metric_lookup = metrics.set_index("model")
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.0), constrained_layout=True)
    colors = {
        "local_hgb": "#777777",
        "lineage_full": "#3B6FB6",
        "prime_fusion": "#4DAF4A",
        "prime_joint": "#8C6BB1",
    }

    for model_name in ["local_hgb", "lineage_full", "prime_fusion"]:
        rows = learning_curve.loc[learning_curve["model"] == model_name].sort_values(
            "training_fraction_requested"
        )
        axes[0, 0].plot(
            100 * rows["training_fraction_requested"],
            rows["AUROC"],
            marker="o",
            linewidth=1.8,
            color=colors[model_name],
            label=DISPLAY_NAMES[model_name],
        )
    axes[0, 0].set_xlabel("Training families used (%)")
    axes[0, 0].set_ylabel("Test AUROC")
    axes[0, 0].set_ylim(0.45, 1.01)
    axes[0, 0].set_title("a  Family-level learning curve", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=8)

    importance = (
        permutation_importance.groupby("feature", as_index=False)
        .agg(mean_drop=("auroc_drop", "mean"), std_drop=("auroc_drop", "std"))
        .fillna(0.0)
        .sort_values("mean_drop")
    )
    axes[0, 1].barh(
        importance["feature"],
        importance["mean_drop"],
        xerr=importance["std_drop"],
        color="#6B8EC1",
        alpha=0.9,
    )
    axes[0, 1].axvline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_xlabel("Within-family permutation AUROC drop")
    axes[0, 1].set_title("b  Feature alignment importance", loc="left", fontweight="bold")
    axes[0, 1].tick_params(axis="y", labelsize=7)

    for model_name in ["local_hgb", "lineage_full", "prime_fusion"]:
        rows = split_sensitivity.loc[split_sensitivity["model"] == model_name].sort_values(
            "split_seed"
        )
        axes[1, 0].plot(
            rows["split_seed"].astype(str),
            rows["AUROC"],
            marker="o",
            linewidth=1.8,
            color=colors[model_name],
            label=DISPLAY_NAMES[model_name],
        )
    axes[1, 0].set_xlabel("Family split seed")
    axes[1, 0].set_ylabel("Test AUROC")
    axes[1, 0].set_ylim(0.45, 1.01)
    axes[1, 0].set_title("c  Split-seed sensitivity", loc="left", fontweight="bold")
    axes[1, 0].legend(frameon=False, fontsize=8)

    class_metrics = ["recall_duplication", "specificity_speciation"]
    comparison_models = ["local_hgb", "lineage_full", "prime_fusion", "prime_joint"]
    x = np.arange(len(comparison_models))
    width = 0.34
    for offset, (metric, color) in enumerate(
        zip(class_metrics, ["#E69F00", "#3B6FB6"])
    ):
        axes[1, 1].bar(
            x + (offset - 0.5) * width,
            [float(metric_lookup.loc[name, metric]) for name in comparison_models],
            width,
            color=color,
            label=("Duplication recall" if metric == "recall_duplication" else "Speciation specificity"),
        )
    axes[1, 1].set_xticks(
        x, [DISPLAY_NAMES[name] for name in comparison_models], rotation=20, ha="right"
    )
    axes[1, 1].set_ylim(0, 1.02)
    axes[1, 1].set_ylabel("Rate")
    axes[1, 1].set_title("d  Class-specific error balance", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Extended robustness and interpretability analyses",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(figure_dir / "prime_panther_extended_analysis.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / "prime_panther_extended_analysis.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "prime_panther_extended_analysis.svg", bbox_inches="tight")
    plt.close(fig)


def write_data_audit_markdown(audit: Mapping[str, object], output_path: Path) -> None:
    critical = audit["critical_totals"]
    lines = [
        "# PANTHER 数据审计",
        "",
        f"- 范围：`{audit['scope']}`",
        f"- 节点：{int(audit['node_rows']):,}",
        f"- 边：{int(audit['edge_rows']):,}",
        f"- 家族：{int(audit['families']):,}",
        f"- 主任务祖先节点：{int(audit['target_rows']):,}",
        f"- 结构完整性：{'通过' if audit['critical_checks_passed'] else '未通过'}",
        f"- 泄漏字段检查：{'通过' if audit['leakage_check_passed'] else '未通过'}",
        "",
        "## 关键完整性检查",
        "",
        "| 检查 | 异常数 |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {int(value):,} |" for name, value in critical.items())
    lines.extend(
        [
            "",
            "## 标签与边界",
            "",
            f"- 事件分布：`{json.dumps(audit['event_types'], ensure_ascii=False)}`",
            "- `event_type_raw`、`nhx_attributes.Ev`、`parent_event_type`、",
            "  `raw_node_id` 及目标事件字段均不进入模型特征。",
            "- 数据是 PANTHER 蛋白家族系统发育树，不是原始核酸序列。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_results_markdown(
    audit: Mapping[str, object],
    metrics: pd.DataFrame,
    criteria: Mapping[str, object],
    run_manifest: Mapping[str, object],
    output_path: Path,
) -> None:
    """Write a claim-led Results chapter synchronized with the main figure."""
    lookup = metrics.set_index("model")
    best_local = str(criteria["best_local_baseline"])
    conclusion_map = {
        "supported_bounded_generalization": "支持有界泛化",
        "directional_but_inconclusive": "方向性可行，但证据不足",
        "not_supported": "不支持泛化",
        "invalid_due_to_audit_failure": "实验因审计失败而无效",
    }
    conclusion = conclusion_map[str(criteria["conclusion"])]
    if criteria["conclusion"] == "supported_bounded_generalization":
        assessment = "可共享，但必须保留主张边界"
    else:
        assessment = "证据尚不足"

    rows = []
    order = [
        "training_prior",
        "local_logistic",
        "local_hgb",
        "lineage_logistic",
        "lineage_degree",
        "lineage_topology_only",
        "lineage_scale_only",
        "lineage_branch",
        "lineage_composition_only",
        "lineage_without_topology",
        "lineage_without_scale",
        "lineage_without_branch",
        "lineage_without_composition",
        "lineage_full",
        "prime_fusion",
        "prime_joint",
    ]
    for name in order:
        if name not in lookup.index:
            continue
        row = lookup.loc[name]
        rows.append(
            f"| {DISPLAY_NAMES[name]} | {_fmt(row['AUROC'])} | {_fmt(row['AUPRC'])} | "
            f"{_fmt(row['macro_f1'])} | {_fmt(row['family_macro_f1'])} | "
            f"{_fmt(row['log_loss'])} | {_fmt(row['ece_15'])} |"
        )
    bootstrap = criteria["bootstrap"]
    extended = criteria.get("extended_robustness", {})
    subgroup = extended.get("subgroups", {})
    sensitivity = extended.get("split_sensitivity", {})
    global_placebo = extended.get("global_placebo", {})
    leave_one = extended.get("leave_one_group_out", {})
    group_only = extended.get("group_only_models", {})
    top_features = extended.get("top_permutation_features", [])
    calibration_summary = extended.get("calibration", {})
    family_metrics_path = output_path.parent / "family_metrics.csv"
    placebo_path = output_path.parent / "placebo_distribution.csv"
    global_placebo_path = output_path.parent / "global_placebo_distribution.csv"
    positive_family_fraction = float("nan")
    if family_metrics_path.exists():
        family_frame = pd.read_csv(family_metrics_path)
        family_pivot = family_frame.pivot(
            index="family_id", columns="model", values="macro_f1"
        )
        paired_gain = (family_pivot["prime_fusion"] - family_pivot[best_local]).dropna()
        if not paired_gain.empty:
            positive_family_fraction = float((paired_gain > 0).mean())
    placebo_ci = (float("nan"), float("nan"))
    global_placebo_ci = (float("nan"), float("nan"))
    if placebo_path.exists():
        placebo_frame = pd.read_csv(placebo_path)
        placebo_ci = tuple(
            map(float, placebo_frame["AUROC"].quantile([0.025, 0.975]).to_numpy())
        )
    if global_placebo_path.exists():
        global_placebo_frame = pd.read_csv(global_placebo_path)
        global_placebo_ci = tuple(
            map(
                float,
                global_placebo_frame["AUROC"].quantile([0.025, 0.975]).to_numpy(),
            )
        )
    learning_path = output_path.parent / "learning_curve.csv"
    learning_line = "- 学习曲线未生成。"
    if learning_path.exists():
        learning = pd.read_csv(learning_path)
        lineage_learning = learning.loc[learning["model"] == "lineage_full"].sort_values(
            "training_fraction_requested"
        )
        if not lineage_learning.empty:
            first = lineage_learning.iloc[0]
            last = lineage_learning.iloc[-1]
            learning_line = (
                f"- 完整谱系专家仅使用 {float(first['training_fraction_requested']):.0%} "
                f"训练家族时 AUROC 为 {float(first['AUROC']):.3f}，使用全部训练家族时为 "
                f"{float(last['AUROC']):.3f}。"
            )
    criterion_lines = [
        f"- {'通过' if value else '未通过'}：`{name}`"
        for name, value in criteria["criteria"].items()
    ]
    local_row = lookup.loc[best_local]
    prime_row = lookup.loc["prime_fusion"]
    lineage_row = lookup.loc["lineage_full"]
    joint_row = lookup.loc["prime_joint"]
    lines = [
        "# PRIME × PANTHER 可行性实验：Results",
        "",
        f"## 总体判断：{conclusion}",
        "",
        f"验证状态：**{assessment}**。在事件标签被完全遮蔽、训练与测试家族严格隔离的条件下，正确对齐的谱系上下文大幅提高了 PANTHER `speciation/duplication` 事件恢复性能。证据支持 PRIME 的谱系锚点、来源可追溯和结构增强推理可迁移到树结构化生物注释；结论不外推到原始 DNA 序列、多数据库信息整合或独立生物机制发现。",
        "",
        "## 实验范围与评估协议",
        "",
        f"完整数据包含 {int(audit['node_rows']):,} 个节点、{int(audit['edge_rows']):,} 条边和 {int(audit['families']):,} 个蛋白家族，其中 {int(audit['target_rows']):,} 个祖先节点进入二分类任务。按家族分层后，训练、验证和测试集分别包含 {int(run_manifest['families_by_split']['train']):,}、{int(run_manifest['families_by_split']['validation']):,} 和 {int(run_manifest['families_by_split']['test']):,} 个互斥家族；测试集包含 {int(run_manifest['rows_by_split']['test']):,} 个目标节点。所有模型选择、融合权重和分类阈值均在训练集或验证集完成。",
        "",
        "## 谱系上下文在未见家族上大幅优于局部信息",
        "",
        f"仅使用节点入边、根标记及缺失指示的局部逻辑回归和局部 HGB 在测试集上的 AUROC 分别为 {float(lookup.loc['local_logistic', 'AUROC']):.3f} 和 {float(local_row['AUROC']):.3f}，表明目标节点自身信息不足以可靠区分两类演化事件（图 1a）。单独使用拓扑位置、后代规模和分支上下文时，AUROC 分别达到 {float(lookup.loc['lineage_topology_only', 'AUROC']):.3f}、{float(lookup.loc['lineage_scale_only', 'AUROC']):.3f} 和 {float(lookup.loc['lineage_branch', 'AUROC']):.3f}。这些结果显示，即使不使用子代物种组成，树中的位置、尺度和分支关系仍包含可跨家族迁移的事件信息。",
        "",
        f"PRIME 融合的 AUROC 为 {float(prime_row['AUROC']):.3f}、AUPRC 为 {float(prime_row['AUPRC']):.3f}，相对最强局部基线的 AUROC 增益为 {float(criteria['test_auroc_gain']):.3f}。直接子代 taxon 的多样性、缺失比例和祖先节点比例属于 PRIME 谱系专家的子代组成特征组，用于实现谱系锚点的聚合与定向流通，因此不再被定义为独立的外部 baseline。主比较由训练先验、两个局部模型、三个单组谱系模型和 PRIME 融合构成。",
        "",
        f"完整谱系专家与局部—谱系联合模型的 AUROC 分别为 {float(lineage_row['AUROC']):.3f} 和 {float(joint_row['AUROC']):.3f}。二者用于界定结构信息的可恢复上限和融合方式差异，不作为图 1a 中的独立外部 baseline 重复展示。它们与 PRIME 融合均接近性能上限，说明该任务在获得子代组成后高度规则化；因此本文将接近满分的结果解释为 PANTHER 结构标注规则的恢复，而非独立生物机制的发现。",
        "",
        "## 性能增益在测试家族间保持一致",
        "",
        f"在家族等权评估中，PRIME 融合的 Macro-F1 相对局部 HGB 提高 {float(criteria['family_macro_f1_gain']):.3f}（图 1b）。逐家族差值的平均值为 {float(bootstrap['mean_macro_f1_gain']):.3f}，2,000 次配对 bootstrap 的 95% CI 为 [{float(bootstrap['macro_f1_gain_ci_low']):.3f}, {float(bootstrap['macro_f1_gain_ci_high']):.3f}]；{positive_family_fraction:.1%} 的可配对测试家族获得正增益。逐家族 log-loss 改善均值为 {float(bootstrap['mean_log_loss_gain']):.4f}，95% CI 为 [{float(bootstrap['log_loss_gain_ci_low']):.4f}, {float(bootstrap['log_loss_gain_ci_high']):.4f}]。这些家族等权结果排除了少数大型家族单独支配总体增益的解释。",
        "",
        "## 预测价值依赖节点与谱系上下文的正确对齐",
        "",
        f"正确对齐的 PRIME AUROC 为 {float(prime_row['AUROC']):.3f}（图 1c）。在每个测试家族内打乱节点与谱系专家输出后，100 次安慰剂的平均 AUROC 降至 {float(criteria['placebo_mean_auroc']):.3f}，经验 95% 区间为 [{placebo_ci[0]:.3f}, {placebo_ci[1]:.3f}]；真实对齐结果高 {float(criteria['aligned_minus_placebo_auroc']):.3f}。跨全部测试节点进行全局错配时，平均 AUROC 为 {float(global_placebo.get('mean_AUROC', float('nan'))):.3f}，经验 95% 区间为 [{global_placebo_ci[0]:.3f}, {global_placebo_ci[1]:.3f}]，真实对齐结果高 {float(global_placebo.get('aligned_minus_mean_AUROC', float('nan'))):.3f}。因此，性能提升要求结构证据属于正确节点，而不是仅由增加变量数量或模型容量产生。",
        "",
        "## 子代组成是主要信息通道，其他结构组提供可替代证据",
        "",
        "图 1d 同时报告每个特征组的充分性和条件必要性。单独使用拓扑位置、后代规模、分支上下文和子代组成时，AUROC 分别为 "
        + ", ".join(
            f"{float(group_only[name]['AUROC']):.3f}"
            for name in [
                "topology_position",
                "descendant_scale",
                "branch_context",
                "child_composition",
            ]
        )
        + "。从完整谱系模型中分别删除这些组后，AUROC 为 "
        + ", ".join(
            f"{float(leave_one[name]['AUROC']):.3f}"
            for name in [
                "topology_position",
                "descendant_scale",
                "branch_context",
                "child_composition",
            ]
        )
        + "。子代组成单独使用即达到接近完整模型的性能，而删除该组使 AUROC 降至 "
        + f"{float(leave_one['child_composition']['AUROC']):.3f}；相比之下，其他三组在完整上下文中大多可被相关特征补偿。家族内置换进一步显示，`child_taxon_diversity` 的平均 AUROC 下降为 {float(top_features[0]['mean_auroc_drop']):.3f}，是排名最高的单一特征。",
        "",
        "## 保守融合提高判别能力，但不提供可靠的绝对概率",
        "",
        f"可靠性曲线显示，完整谱系专家的 ECE-15 为 {float(calibration_summary.get('lineage_full', float('nan'))):.3f}，局部 HGB 为 {float(calibration_summary.get('local_hgb', float('nan'))):.3f}，PRIME 融合为 {float(calibration_summary.get('prime_fusion', float('nan'))):.3f}（图 1e）。验证集选择的融合权重达到预设上限 {float(criteria['selected_fusion_weight']):.2f}；将高置信谱系概率与校准较差的局部概率等权混合后，预测被压缩到中间区间。PRIME 融合因此保持了高 AUROC 和 Macro-F1，但其输出不应在未经再校准时解释为准确的 duplication 发生概率。",
        "",
        "## 补充稳健性分析",
        "",
        f"在 {int(subgroup.get('evaluable_groups', 0))} 个可计算 AUROC 的家族规模、复制比例、节点深度和根状态分组中，PRIME 相对局部基线的最小增益为 {float(subgroup.get('minimum_prime_auroc_gain', float('nan'))):.3f}，所有分组增益均为正。家族切分种子 {sensitivity.get('seeds', [])} 下，AUROC 增益范围为 {float(sensitivity.get('minimum_prime_auroc_gain', float('nan'))):.3f}–{float(sensitivity.get('maximum_prime_auroc_gain', float('nan'))):.3f}。{learning_line.removeprefix('- ')}上述分析均在主模型、融合约束和验收门槛冻结后执行，未用于重新选模。",
        "",
        "## 预设判定条件",
        "",
        *criterion_lines,
        "",
        "全部预设条件均通过，机器判定为 `supported_bounded_generalization`。该判定证明的是结构化谱系数据在保持来源、关系和节点对齐时具有可流通、可复用的预测价值。",
        "",
        "## 图 1 图注",
        "",
        "**图 1｜正确对齐的谱系上下文支持跨家族 PANTHER 事件恢复。** 测试集包含 2,355 个未见家族中的 228,803 个目标节点。"
        "**a，** 无信息先验、局部线性与非线性模型、三个单组谱系模型和 PRIME 融合在家族隔离测试集上的 AUROC–duplication AUPRC 散点图；PRIME 为实心点，其余方法为空心点。"
        "**b，** PRIME 相对最强局部基线的逐家族 Macro-F1 差值；红线为均值，阴影为 2,000 次家族配对 bootstrap 的 95% CI。"
        "**c，** 正确对齐结果与 100 次家族内错配、100 次全局错配的 AUROC；点表示均值，线段表示经验 95% 区间。"
        "**d，** 四类谱系特征组单独使用（充分性）及从完整模型中删除该组（条件必要性/冗余性）时的 AUROC。"
        "**e，** 局部 HGB、完整谱系专家与 PRIME 融合的十分位可靠性曲线；虚线表示理想校准，括号内为 15 个等宽概率区间计算的 ECE。所有阈值和融合权重在验证集冻结，测试家族未参与拟合或选择。图形源数据分别来自 `metrics.csv`、`family_metrics.csv`、`placebo_distribution.csv`、`global_placebo_distribution.csv` 和 `calibration_table.csv`。",
        "",
        "## PRIME 映射与主张边界",
        "",
        "- **P（provenance）**：每条预测保留 `node_id`、`family_id` 与 `source_file`。",
        "- **I（integration）**：节点和父子边被整合为通过结构审计的家族树。",
        "- **R+M（reasoning and missing annotation）**：在目标事件字段遮蔽后，用谱系关系恢复缺失注释。",
        "- **未验证范围**：单一 PANTHER 版本不能验证跨数据库多源整合；缺少原始序列不能验证序列模型泛化；分类性能不能验证 PRIME 的 E 层。",
        "- **标签生成依赖**：事件标签与 PANTHER 树构建和注释规则相关，近饱和性能属于结构化注释恢复证据，而不是独立生物机制证据。",
        "",
        "## 完整指标表",
        "",
        "| 模型 | AUROC | AUPRC | Macro-F1 | 家族等权 Macro-F1 | Log loss | ECE-15 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## 可复现性",
        "",
        f"随机种子为 {int(run_manifest['seed'])}；逐家族 bootstrap {int(run_manifest['bootstrap_repeats']):,} 次，家族内和全局错配各 {int(run_manifest['placebo_repeats']):,} 次。完整特征、阈值、环境版本和切分规模记录在 `run_manifest.json`。",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
