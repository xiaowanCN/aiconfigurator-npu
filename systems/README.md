# 归档副本 —— 请勿当作运行时数据使用

这个目录是 **0.18.0 时期的遗留副本**，不是当前生效的数据。

- 真正的运行时数据在包内：`src/aiconfigurator_npu/systems/data/<system>/<backend>/<version>/`
- 生效版本现在是 `ascend_910b / vllm-ascend / 0.23.0`

## 为什么它不会被加载

加载器取 `${systems_root}/<system>.yaml`（`perf_database.get_database`：`system_yaml_path = os.path.join(systems_root, f"{system}.yaml")`）。
而这里把 YAML 放在了 `ascend_910b_aiconfigurator/ascend_910b.yaml` —— 路径不匹配，
所以只有在显式传 `--systems-paths systems/ascend_910b_aiconfigurator` 时才会命中。

## 为什么仍然保留

`data/ascend_910b/vllm-ascend/0.18.0/` 下保存了**唯一留存的 TensorCast 原始 CSV**
（`GroupedMatmul_MoE_BF16.csv`、`MatMulV2.csv`、`FusedInferAttentionScore.csv` 等），
是数据溯源资产，故不删除。

但请注意：**它与 0.23.0 的表不是同一次测量**。逐单元格比对显示 91 个重合单元格全部不一致，
最大相对偏差 461%（例：experts=256、tokens=1 时，0.23.0 表为
`[0.04376, 0.12383, 0.18358, 0.18393]`，此处的原始 CSV 为 `[0.03101, 0.06574, 0.11581, 0.14002]`）。

因此**不可**用它去补齐 0.23.0 缺的行；`tools/repair_moe_distribution.py` 会检测同源性并拒绝。
详见 `docs/DATA_COLLECTION_PLAN.md` 中 P0-2 一节。

（`docs/DSA_COLLECTION_COMMANDS.md`、`docs/GLM5_ADAPTATION_DESIGN.md` 里引用的
`systems/data/ascend_910b/vllm-ascend/0.18.0/` 路径属于历史文档，请改用上文的包内路径。）
