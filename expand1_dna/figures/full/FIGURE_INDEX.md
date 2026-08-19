# PRIME × PANTHER 独立图件索引

所有图件高度均为 50 mm，无图内标题，并导出 PNG（600 dpi）、PDF、SVG 和 EMF。
EMF 由 Inkscape 从 Matplotlib 生成的 SVG 转换，SVG/PDF 保留可编辑矢量文字。

| 分组 | 图件 | 宽度 (mm) | 数据来源 | 含义 |
|---|---|---:|---|---|
| 主图 | `a_performance_scatter` | 76 | `metrics.csv` | AUROC–AUPRC comparison across local, lineage-subset and PRIME models |
| 主图 | `b_family_macro_f1_gain` | 66 | `family_metrics.csv; bootstrap_summary.json` | Family-equal distribution and paired-bootstrap interval of the PRIME gain |
| 主图 | `c_context_alignment_control` | 78 | `placebo_distribution.csv; global_placebo_distribution.csv; metrics.csv` | Necessity of correct node–lineage alignment |
| 主图 | `d_feature_group_mechanism` | 78 | `metrics.csv` | Feature-group sufficiency and conditional redundancy |
| 主图 | `e_probability_calibration` | 60 | `calibration_table.csv; metrics.csv` | Reliability and calibration cost of conservative fusion |
| 扩展 | `f_learning_curve` | 70 | `learning_curve.csv` | Family-level data efficiency |
| 扩展 | `g_permutation_importance` | 88 | `feature_permutation_importance.csv` | Node–feature alignment importance for all lineage features |
| 扩展 | `h_split_sensitivity` | 65 | `split_sensitivity.csv` | Robustness to family split seed |
| 扩展 | `i_subgroup_generalization` | 90 | `subgroup_metrics.csv` | PRIME gain across every evaluable structural subgroup |
| 扩展 | `j_class_error_balance` | 78 | `metrics.csv` | Class-specific recall and specificity |
| 扩展 | `k_fusion_weight_selection` | 66 | `fusion_weight_scan.csv` | Validation-only selection of the conservative fusion weight |
| 扩展 | `l_bootstrap_macro_f1_gain` | 66 | `bootstrap_distribution.csv` | Paired family-bootstrap uncertainty distribution |
| 扩展 | `m_bootstrap_log_loss_gain` | 66 | `bootstrap_distribution.csv` | Paired family-bootstrap uncertainty distribution |
| 扩展 | `n_placebo_distributions` | 68 | `placebo_distribution.csv; global_placebo_distribution.csv` | Full distributions of alignment negative controls |
