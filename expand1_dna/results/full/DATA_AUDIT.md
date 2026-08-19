# PANTHER 数据审计

- 范围：`full`
- 节点：3,564,945
- 边：3,549,262
- 家族：15,683
- 主任务祖先节点：1,540,340
- 结构完整性：通过
- 泄漏字段检查：通过

## 关键完整性检查

| 检查 | 异常数 |
|---|---:|
| duplicate_nodes | 0 |
| endpoint_missing | 0 |
| self_edges | 0 |
| duplicate_parent | 0 |
| family_mismatch | 0 |
| relation_mismatch | 0 |
| event_copy_mismatch | 0 |
| branch_copy_mismatch | 0 |
| edge_count_mismatch | 0 |
| declared_root_mismatch | 0 |
| unreachable_nodes | 0 |

## 标签与边界

- 事件分布：`{"speciation": 1158521, "duplication": 381819, "<NULL>": 2017190, "coded_event": 7415}`
- `event_type_raw`、`nhx_attributes.Ev`、`parent_event_type`、
  `raw_node_id` 及目标事件字段均不进入模型特征。
- 数据是 PANTHER 蛋白家族系统发育树，不是原始核酸序列。
