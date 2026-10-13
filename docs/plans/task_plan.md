# 三模型适配：数据补采说明 + 纯代码任务

## 目标
1. 产出可执行的数据补采规格文档：为适配 Qwen3.8-27B / DeepSeek-R1 / DeepSeek-V4-Pro，
   明确需要采集哪些表、每个表的完整维度网格、用哪个脚本、命令行怎么写。
2. 完成剩余纯代码任务，每项改动必须与 NVIDIA 侧 aic-core 逻辑逐一校对。

## 阶段
- [x] P1 摸清采集器现状（collector/npu/*）：支持的表、参数、输出格式
- [x] P2 撰写 docs/DATA_COLLECTION_PLAN.md（含三模型全量网格 + 缺失重点标记）
- [x] P3 T9：仓库根 systems/ 残留副本 -> 判定保留 + 新增 systems/README.md 说明
- [x] P4 T12：README / V4 / R1 适配文档与数据现状同步
- [x] P5 T10：搜索维度对齐 NVIDIA（enable_wideep 废弃 + moe_backend；cp_list 保留分歧并写明理由）
- [x] P6 T8 收尾：按实测回填 support_matrix（含新增 Qwen3-32B）
- [x] P7 回归验证三模型 + V4 基线不回退

## 关键决策与证据
- **P0-2 不能靠仓库根的 0.18.0 CSV 免硬件修复**：实测它与 0.23.0 表不同源，
  91 个重合单元格全部不一致（最大相对偏差 461%）。因此新增
  `tools/repair_moe_distribution.py`，内置同源性校验并对不同源直接拒绝写入。
- **根 systems/ 不删除**：虽永不被加载（YAML 路径不匹配），但保留了唯一留存的原始 TensorCast CSV，
  是溯源资产；且两篇历史文档仍引用该路径。改为加 README 说明其性质。
- **不新增 cp_list**：NPU 无 CP 算子建模，增加维度只会产出无法定价的方案；
  已 code 注释写明分歧理由。cp=1 时两侧宽度约束完全等价（已用
  `tools/check_search_space_parity.py` 在 7 个用例上验证一致）。
- **T11 不盲移植**：上游 vLLM 显存上限模型（gpu_memory_utilization, of_total 语义）
  依赖 5 个 NPU 侧不存在的钩子，且默认值需按 vllm-ascend 标定，故记录待立项。

## 新增产物
- `docs/DATA_COLLECTION_PLAN.md` — 全量补采规格
- `tools/repair_moe_distribution.py` — MoE distribution 修复（带同源校验，dry-run 默认）
- `tools/check_search_space_parity.py` — 与 NVIDIA 侧搜索空间一致性比对
- `tools/check_model_readiness.py` — 模型静态就绪度体检
- `systems/README.md` — 归档副本说明

## 回归结果（每次改动后）
- V4-Pro SILICON 64 卡：312.70 tokens/s / 4.89 tokens/s/gpu / tp8pp1dp8etp1ep64 —— 始终未变
- Qwen3-32B：13,181.57 tokens/s（disagg）—— 始终未变
- R1 / Qwen3.8-27B：如预期仍无结果（数据缺口，非代码可解）
- `check_search_space_parity.py`：7/7 PASS
