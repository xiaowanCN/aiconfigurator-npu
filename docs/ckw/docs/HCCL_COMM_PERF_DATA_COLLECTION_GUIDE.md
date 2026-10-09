# HCCL 集合通信性能数据采集与接入完整指南

> 本文档覆盖 AIConfigurator-NPU 框架中 **全部集合通信原语**（`all_reduce` / `all_gather` / `reduce_scatter` / `alltoall`）的：
> 1. 数据文件在框架中的加载与使用链路
> 2. 需要采集哪些数据（维度、范围、格式）
> 3. 基于 `mpirun + all_reduce_test` 等测试工具的采集方案
> 4. 原始数据 → 框架格式的转换与放置
>
> 相对 `docs/HCCL_ALLREDUCE_PERF_DATA_GUIDE.md`（只覆盖 all_reduce），本文档补齐了 `all_gather` / `reduce_scatter` / `alltoall` 的采集与语义说明。

## 目录

- [1. 背景：框架中的两套通信数据文件](#1-背景框架中的两套通信数据文件)
- [2. 数据文件在框架中的完整使用链路](#2-数据文件在框架中的完整使用链路)
- [3. 四个通信原语的使用场景与采集需求](#3-四个通信原语的使用场景与采集需求)
- [4. 关键语义约定（必须先读懂）](#4-关键语义约定必须先读懂)
- [5. 采集方案 A：mpirun + HCCL 测试工具](#5-采集方案-ampirun--hccl-测试工具)
- [6. 采集方案 B：torchrun + torch_npu Python 脚本](#6-采集方案-btorchrun--torch_npu-python-脚本)
- [7. 原始数据 → 框架格式转换](#7-原始数据--框架格式转换)
- [8. 数据放置、验证与常见问题](#8-数据放置验证与常见问题)
- [附录 A：完整采集 Checklist](#附录-a完整采集-checklist)
- [附录 B：相关源码索引](#附录-b相关源码索引)

---

## 1. 背景：框架中的两套通信数据文件

备份目录 `backup/aic_npu_v0920_workspace_backup/workspace/data/hccl_data/` 下有两个文件，它们是**两套独立的通信性能表**，服务不同算子：

| 文件 | 作用 | 加载函数 | 查询接口 | 框架调用方 |
|------|------|----------|----------|-----------|
| `custom_allreduce_perf.txt` | **vLLM/SGLang custom allreduce 内核** 的 AllReduce 性能 | `load_custom_allreduce_data` | `query_custom_allreduce` | `ops.CustomAllReduce` |
| `nccl_perf.txt` | **HCCL/NCCL 集合通信原语** 性能（多 op 通用表） | `load_nccl_data` | `query_nccl` | `ops.NCCL` |

### 1.1 backup 样例数据现状

`custom_allreduce_perf.txt`（18 行）：只有 `half, num_gpus=8`，message_size 524288~536870912。

`nccl_perf.txt`（37 行）：

| op_name | 行数 | 备注 |
|---------|------|------|
| `all_reduce` | 18 | 与 `custom_allreduce_perf.txt` **完全相同** |
| `all_gather` | 18 | 同一 size 网格 |
| `reduce_scatter` | **0** | **缺失** |
| `alltoall` | **0** | **缺失** |

> **重要**：backup 数据是**临时存放**的，框架不会从 `data/hccl_data/` 读取。必须放到 `systems/data/...` 正确路径（见 [§8](#8-数据放置验证与常见问题)）才能被 `PerfDatabase` 加载。当前 `src/aiconfigurator_npu/systems/data/` 下**这两个文件均不存在**，SILICON 模式查通信会直接 raise，HYBRID 模式回退 SOL。

---

## 2. 数据文件在框架中的完整使用链路

### 2.1 加载链路（`perf_database.py`）

```
PerfDatabase.__init__  (perf_database.py:2102)
  │
  ├─ data_dir       = {systems_root}/{system_spec.data_dir}/{backend}/{version}
  │                 = systems/data/ascend_910b/vllm-ascend/0.18.0/
  │
  ├─ nccl_data_dir  = {systems_root}/{system_spec.data_dir}/nccl/{system_spec.misc.nccl_version}
  │                 = systems/data/ascend_910b/nccl/2.26.0/
  │
  ├─ _load_op_data(PerfDataFilename.custom_allreduce)
  │     → load_custom_allreduce_data(data_dir / "custom_allreduce_perf.txt")     # :2131, :2151
  │     → self._custom_allreduce_data                                            # :2175
  │
  └─ _load_op_data(PerfDataFilename.nccl)
        → load_nccl_data(nccl_data_dir / "nccl_perf.txt")                        # :2132, :2152-2153
        → self._nccl_data                                                        # :2176
```

**路径分流是硬编码的**（`perf_database.py:2151-2153`）：只有 `nccl_perf.txt` 走 `nccl/{nccl_version}/` 目录，其余（含 `custom_allreduce_perf.txt`）走 backend 目录。这与 `ascend_910b.yaml` 中 `misc.nccl_version: '2.26.0'` 对应——HCCL 数据按**通信库版本**组织，与推理后端版本解耦。

### 2.2 数据结构

**`custom_allreduce_perf.txt`**（`perf_database.py:366-438`）：

```text
custom_allreduce_data[CommQuantMode][tp_size:int]["AUTO"][message_size:int] = {
    "latency": float,   # ms
    "power":   float,   # W
    "energy":  float,   # power * latency, W·ms
}
```

- `dtype` 列被**硬编码为 `CommQuantMode.half`**（`perf_database.py:416`，TODO）
- `strategy` 固定 `"AUTO"`
- 可选扩展列 `kernel_source` / `backend`：`*_eager` 行会被跳过，只保留 graph 模式（b60 除外）

**`nccl_perf.txt`**（`perf_database.py:441-493`）：

```text
nccl_data[CommQuantMode][op_name:str][num_gpus:int][message_size:int] = {
    "latency": float,
    "power":   float,
    "energy":  float,
}
```

- `dtype` 从 `nccl_dtype` 列解析 `CommQuantMode[name]`，支持 `half` / `int8` / `fp8`
- `op_name` 取值：`all_reduce` / `all_gather` / `reduce_scatter` / `alltoall`

### 2.3 查询接口与 DatabaseMode

两个查询接口都支持 4 种模式（`common.py:528-537`）：

| DatabaseMode | 行为 |
|--------------|------|
| `SILICON`（默认） | **纯查表 + 1D 线性插值**，表缺失直接 raise |
| `HYBRID` | 优先查表，失败回退 `SOL / 0.8` |
| `EMPIRICAL` | `SOL / 0.8` |
| `SOL` / `SOL_FULL` | 纯理论带宽模型 |

**SILICON 查表逻辑**（两接口同构）：

1. `tp_size/num_gpus == 1` → 返回 0
2. 用 `min(num_gpus, max_available)` 取表（节点内上限 `num_gpus_per_node=8`）
3. `_nearest_1d_point_helper` + `_interp_1d` 对 `message_size` 做**线性插值**
4. 若查询规模 > 表内最大规模（跨节点），按 `(N-1)/N` 因子 + 带宽比缩放（`perf_database.py:4612-4624` / `:4727-4736`）

**SOL 公式**（`perf_database.py:4554-4567` / `:4666-4680`）：

| operation | SOL 时间（ms） |
|-----------|----------------|
| `all_reduce` | `2 * elem_bytes * size * (N-1)/N / p2p_bw * 1000` |
| `all_gather` / `reduce_scatter` / `alltoall` | `elem_bytes * size * (N-1)/N / p2p_bw * 1000` |

其中 `p2p_bw` 来自 `system_spec.node`（Ascend 910B：节点内 HCCS `56GB/s`，跨节点 RoCE `25GB/s`）。

### 2.4 调用链（谁在用这些数据）

```
models.py  (各模型的 context/generation 算子序列)
  │
  ├─ ops.CustomAllReduce(name, scale, h, tp_size)          operations.py:40
  │     query(): size = x * h                               # 元素数
  │     → database.query_custom_allreduce(half, tp_size, size)
  │     场景：TP 下 attention/FFN/embedding 输出 AllReduce
  │           （GPT/LLAMA/MOE/DEEPSEEKV32/NEMOTRONH/QWEN35 全系）
  │
  ├─ ops.NCCL(name, scale, nccl_op, h, num_gpus, quant)    operations.py:93
  │     query(): message_size = x * h                       # 元素数
  │     → database.query_nccl(quant, num_gpus, nccl_op, message_size)
  │     场景：DSA / WideEP MLA 的 "AllGather + Compute + ReduceScatter"
  │           替代 2×AllReduce 的 TP 通信模式（仅 context/prefill 侧）
  │           models.py:2135-2165 (DEEPSEEKV32)
  │           models.py:2857-2889 (WideEP DEEPSEEKV32)
  │
  └─ ops.TrtLLMWideEPMoEDispatch(...)                      operations.py:383
        → database.query_trtllm_alltoall(...)              # 不走 query_nccl！
        场景：MoE token dispatch/combine 的 alltoall
        对应文件：trtllm_alltoall_perf.txt（独立格式，见 §3.4）
```

---

## 3. 四个通信原语的使用场景与采集需求

### 3.1 总览

| 原语 | 框架查询路径 | 数据文件 | backup 现状 | 采集优先级 |
|------|-------------|----------|-------------|-----------|
| **all_reduce** | `query_custom_allreduce`（主）<br>`query_nccl("all_reduce")`（fallback） | `custom_allreduce_perf.txt`<br>+ `nccl_perf.txt` | 有（仅 half/8卡） | **P0 必须** |
| **all_gather** | `query_nccl("all_gather")` | `nccl_perf.txt` | 有（仅 half/8卡） | **P0 必须**（DSA/MLA） |
| **reduce_scatter** | `query_nccl("reduce_scatter")` | `nccl_perf.txt` | **无** | **P0 必须**（DSA/MLA） |
| **alltoall** | `query_nccl("alltoall")`（仅 SOL 公式）<br>`query_trtllm_alltoall`（MoE 主路径） | `nccl_perf.txt`（通用）<br>`trtllm_alltoall_perf.txt`（MoE） | **无** | P1 / P2（见下） |

### 3.2 all_reduce —— TP 输出同步

**何时需要**：只要 `tp_size > 1`，**每一个** Transformer 层的 attention 输出、FFN 输出都要各做一次 AllReduce。这是**最高频**的通信原语。

**每 token 通信量**：`h` 个元素（`h` = hidden_size），即 `message_size = num_tokens * h`。

**采集要求**：
- `num_gpus` ∈ {2, 4, 8}（覆盖常见 TP 配置；有跨节点 TP 需求再加 16/32/64）
- `dtype` = `half`（框架硬编码；如后续放开再补 `int8`/`fp8`）
- `message_size` 覆盖 decode 小 batch 到 prefill 大 batch，见 [§4.3](#43-message_size-的推荐覆盖范围)

**两份数据的关系**：`custom_allreduce_perf.txt` 走 vLLM/SGLang 的 custom allreduce 内核（NVLink/HCCS 直连优化路径），`nccl_perf.txt` 的 `all_reduce` 走 HCCL 标准路径。二者性能不同，**都应采集**。查询优先级：

1. 非 GB200 → `query_custom_allreduce`（custom 内核）
2. GB200 且 `tp_size>4` → 自动 fallback `query_nccl("all_reduce")`
3. `custom_allreduce_perf.txt` 缺失 → SILICON raise / HYBRID 回退 SOL

### 3.3 all_gather + reduce_scatter —— MLA/DSA 的 TP 通信

**何时需要**：DeepSeek-V3.2 / WideEP MLA 等采用 **"AllGather 收齐 hidden → 计算 → ReduceScatter 归约"** 替代 2×AllReduce 的模型。仅 `tp_size > 1` 时启用，且当前**只在 context（prefill）侧**加入。

**每 token 通信量**：两者都是 `h` 个元素（`h` = **完整** hidden_size，非分片），即 `message_size = num_tokens * h`。语义见 [§4.2](#42-message_size-的语义按原语区分)。

**采集要求**：
- 两个原语**必须成对采集**（缺 `reduce_scatter` 会导致 SILICON 模式查表 raise）
- 维度同 all_reduce

### 3.4 alltoall —— MoE 的 token dispatch/combine

**注意：MoE 的 alltoall 不走 `nccl_perf.txt`！**

| 路径 | 数据文件 | 格式维度 | 使用场景 |
|------|----------|----------|----------|
| `query_trtllm_alltoall`（主） | `trtllm_alltoall_perf.txt` | `kernel_source, op_name, quant_mode, num_nodes, hidden_size, topk, num_experts, moe_ep_size, latency, power` | MoE WideEP dispatch/combine |
| `query_nccl("alltoall")`（备用/通用） | `nccl_perf.txt` | 与 all_reduce 同构 | 当前 models 层**无调用点**，SOL 公式支持 |

**MoE 场景采集（P1）**：若要用实测数据替代 MoE alltoall 的 SOL 估算，需要按 `trtllm_alltoall_perf.txt` 格式采集，维度远多于普通集合通信：

- `op_name` ∈ {`alltoall_prepare`, `alltoall_dispatch`, `alltoall_combine`, `alltoall_combine_low_precision`}
- 维度：`moe_ep_size` × `hidden_size` × `topk` × `num_experts` × `num_tokens` × `quant_mode` × `num_nodes`
- 这是**多维表**（`perf_database.py:1908-2010`），采集成本高，建议按目标模型的实际配置裁剪

**通用 alltoall（P2）**：如果只是想让 `nccl_perf.txt` 表完整（便于 SOL/EMPIRICAL 校准或未来扩展），按普通集合通信格式采集即可。本文档的采集方案覆盖这一类。

---

## 4. 关键语义约定（必须先读懂）

### 4.1 `message_size` 是元素数，不是字节

框架查询接口的 `size` / `message_size` 参数**一律是元素个数（element count）**：

| 接口 | 参数 | 单位 | 证据 |
|------|------|------|------|
| `query_custom_allreduce(size)` | `size` | **元素数** | `operations.py:55` 注释 "count, not size in bytes"；`size = x * h` |
| `query_nccl(message_size)` | `message_size` | **元素数** | `perf_database.py:4641` 注释 `# element number` |
| `query_p2p(message_bytes)` | `message_bytes` | **字节** | `operations.py:84` `p2p_bytes = size * 2` |

**转字节只发生在 SOL 公式中**（`* dtype.value.memory`，half=2）。

> **采集陷阱**：`nccl-tests` / `hccl-test` 系工具的 `-b/-e` 参数与输出通常是**字节（bytes）**。转换公式：
>
> ```text
> message_size(elements) = raw_size(bytes) / dtype_bytes
> dtype_bytes: half/bf16=2, int8/fp8=1
> ```
>
> 例：工具测 `1 MiB` 的 FP16 all_reduce → 框架 `message_size = 1048576 / 2 = 524288`（正好对上 backup 首行）。

### 4.2 `message_size` 的语义（按原语区分）

框架对不同原语传入的 `message_size` 含义不同，**采集时必须对齐**，否则插值结果错位：

| 原语 | 框架传入的 `message_size` | 采集时本地张量 `numel` | 备注 |
|------|--------------------------|----------------------|------|
| `all_reduce` | 参与归约的张量元素数（in = out = `x*h`） | `message_size` | 直接对应 |
| `all_gather` | **完整 gathered 输出** 的元素数（`x*h`，h 为完整 hidden） | 输入张量 = `message_size / N`，输出 = `message_size` | nccl-tests 的 sendcount 是**每 rank 输入**，需 ×N |
| `reduce_scatter` | **完整输入** 的元素数（`x*h`） | 输入张量 = `message_size`，输出 = `message_size / N` | nccl-tests 的 recvcount 是**每 rank 输出**，需 ×N |
| `alltoall` | 按框架 SOL 公式，等价于"完整张量"元素数 | 每 rank 输入 = `message_size / N` | 与 all_gather 同构 |

**一句话记忆**：`message_size` 始终是 **"未分片的完整张量的元素数"**（`num_tokens * hidden_size`），与 TP 分片无关。

### 4.3 `message_size` 的推荐覆盖范围

需要同时覆盖 **decode 小消息** 和 **prefill 大消息**。backup 只覆盖了 512K~512M 元素（偏 prefill），decode 侧是空的。

以 `h=4096` 为例：

| 场景 | 典型 token 数 | message_size (元素) | FP16 字节 |
|------|--------------|---------------------|-----------|
| decode bs=1 | 1 | 4 K | 8 KB |
| decode bs=128 | 128 | 512 K | 1 MB |
| prefill isl=1024 | 1024 | 4 M | 8 MB |
| prefill isl=4096 | 4096 | 16 M | 32 MB |
| prefill isl=8192 | 8192 | 32 M | 64 MB |
| 长文/大 h | — | 256 M~512 M | 512 MB~1 GB |

**推荐采集网格**（元素数，几何递增，密度足够线性插值）：

```text
小消息（decode，每 2~4× 一点）:
  1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144

中消息（短 prefill，每 1.25~1.5× 一点）:
  524288, 655360, 786432, 917504, 1048576,
  1310720, 1572864, 1835008, 2097152,
  2621440, 3145728, 3670016, 4194304, 5242880, 6291456, 8388608

大消息（长 prefill，每 2× 一点）:
  16777216, 33554432, 67108864, 134217728, 268435456, 536870912
```

> 小消息段尤其重要：decode 阶段 TP 通信延迟占比高，且是线性插值的外推区，点太稀会严重失真。

### 4.4 `num_gpus` 维度

- 至少采集 **2, 4, 8** 三个点（覆盖 TP=2/4/8）
- 框架对 `num_gpus > 表内最大值` 有缩放外推（跨节点），但精度有限；**有跨节点 TP/EP 需求时务必补采 16/32 卡数据**
- `num_gpus=1` 不需要采集（框架直接返回 0）

### 4.5 dtype 维度

| 文件 | 当前框架支持 | 采集建议 |
|------|-------------|---------|
| `custom_allreduce_perf.txt` | 加载端**硬编码 half**（`perf_database.py:416`） | 只采 `half` 即可；若未来放开需补 `int8`/`fp8` |
| `nccl_perf.txt` | `half` / `int8` / `fp8`（`CommQuantMode`） | 优先 `half`；用 FP8 通信的模型补 `fp8` |

实际所有 models 调用点都传 `CommQuantMode.half`（`models.py:2141` 等），**最小可用集 = 仅 half**。

---

## 5. 采集方案 A：mpirun + HCCL 测试工具

### 5.1 工具来源

Ascend CANN / HCCL 生态提供与 `nccl-tests` 同构的 C++ 集合通信基准工具，典型可执行文件名：

| 原语 | 可执行文件 |
|------|-----------|
| all_reduce | `all_reduce_test` |
| all_gather | `all_gather_test` |
| reduce_scatter | `reduce_scatter_test` |
| alltoall | `alltoall_test` / `all_to_all_test` |

获取途径（按环境二选一）：

```bash
# 途径 1：CANN toolkit 自带的 hccl_test 样例（需编译）
cd /usr/local/Ascend/${ASCEND_TOOLKIT_VERSION}/tools/hccl_test   # 路径以实际安装为准
make

# 途径 2：独立 hccl-test / ascend-benchmark 包
# 安装后将可执行目录加入 PATH
export PATH=/path/to/hccl_test/bin:$PATH
```

> 若环境里没有现成 C++ 测试程序，直接用 [§6](#6-采集方案-btorchrun--torch_npu-python-脚本) 的 Python 方案，**输出格式与框架完全对齐**，免转换。

### 5.2 通用命令模板

以用户给出的 `all_reduce_test` 为例：

```bash
mpirun -n 8 all_reduce_test \
  -p 8 \
  -b 8K \
  -e 16K \
  -f 2 \
  -w 20 \
  -n 100 \
  -c 0
```

**参数说明**（nccl-tests 系约定，以所用工具的 `-h` 为准）：

| 参数 | 含义 | 采集建议值 |
|------|------|-----------|
| `-n <N>` (mpirun) | 启动的 rank 数 = 参与通信的卡数 | 2 / 4 / 8（分别跑） |
| `-p <N>` | 进程/并行度（部分工具为 dtype 位宽） | 与 rank 数一致 |
| `-b <size>` | **b**egin：起始消息大小 | 小消息：`2K`；大消息：`512K` |
| `-e <size>` | **e**nd：结束消息大小 | 小消息：`256K`；大消息：`1G` |
| `-f <factor>` | **f**actor：相邻尺寸倍率 | 小段 `2`，中段可用 `1.25` |
| `-w <N>` | **w**armup：预热迭代数 | `20`（≥20，排除首迭代建链开销） |
| `-n <N>` | **n**um-iters：计时迭代数 | `100`（≥100，降方差） |
| `-c <0/1>` | **c**heck：结果正确性校验 | 采集时 `0`（省开销）；抽样校验时 `1` |

> **单位陷阱**：`-b/-e` 的 `8K/16K/1M` 几乎都是**字节**。转换到框架 `message_size` 时必须除以 dtype 字节数（见 [§4.1](#41-message_size-是元素数不是字节)）。

### 5.3 四个原语的完整采集命令

#### 5.3.1 all_reduce（→ `custom_allreduce_perf.txt` + `nccl_perf.txt`）

```bash
#!/bin/bash
# collect_all_reduce.sh
set -e
export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
OUT=./raw_comm; mkdir -p ${OUT}

for NP in 2 4 8; do
  echo "=== all_reduce  np=${NP} ==="

  # A) 小消息段（decode）：2K~256K bytes = 1K~128K elements (FP16)
  mpirun -n ${NP} all_reduce_test \
    -p ${NP} -b 2K -e 256K -f 2 -w 20 -n 100 -c 0 \
    | tee ${OUT}/all_reduce_np${NP}_small.txt

  # B) 中消息段：512K~8M bytes = 256K~4M elements
  mpirun -n ${NP} all_reduce_test \
    -p ${NP} -b 512K -e 8M -f 1.25 -w 20 -n 100 -c 0 \
    | tee ${OUT}/all_reduce_np${NP}_mid.txt

  # C) 大消息段：16M~1G bytes = 8M~512M elements
  mpirun -n ${NP} all_reduce_test \
    -p ${NP} -b 16M -e 1G -f 2 -w 20 -n 100 -c 0 \
    | tee ${OUT}/all_reduce_np${NP}_large.txt
done
```

#### 5.3.2 all_gather（→ `nccl_perf.txt`）

```bash
#!/bin/bash
# collect_all_gather.sh
# 注意：all_gather_test 的 size 参数通常是「每 rank 输入字节数」
# 框架 message_size = 完整输出元素数 = (每 rank 输入字节数 * NP) / dtype_bytes
set -e
OUT=./raw_comm; mkdir -p ${OUT}

for NP in 2 4 8; do
  echo "=== all_gather  np=${NP} ==="

  # 小消息：目标完整输出 2K~256K elements → 每 rank 输入字节 = elements*2/NP
  mpirun -n ${NP} all_gather_test \
    -p ${NP} -b 2K -e 256K -f 2 -w 20 -n 100 -c 0 \
    | tee ${OUT}/all_gather_np${NP}_small.txt

  mpirun -n ${NP} all_gather_test \
    -p ${NP} -b 512K -e 8M -f 1.25 -w 20 -n 100 -c 0 \
    | tee ${OUT}/all_gather_np${NP}_mid.txt

  mpirun -n ${NP} all_gather_test \
    -p ${NP} -b 16M -e 1G -f 2 -w 20 -n 100 -c 0 \
    | tee ${OUT}/all_gather_np${NP}_large.txt
done
```

#### 5.3.3 reduce_scatter（→ `nccl_perf.txt`）

```bash
#!/bin/bash
# collect_reduce_scatter.sh
# reduce_scatter_test 的 size 参数通常是「每 rank 输入字节数」或「每 rank 输出字节数」，
# 务必用 -h 确认。框架 message_size = 完整输入元素数。
set -e
OUT=./raw_comm; mkdir -p ${OUT}

for NP in 2 4 8; do
  echo "=== reduce_scatter  np=${NP} ==="

  mpirun -n ${NP} reduce_scatter_test \
    -p ${NP} -b 2K -e 256K -f 2 -w 20 -n 100 -c 0 \
    | tee ${OUT}/reduce_scatter_np${NP}_small.txt

  mpirun -n ${NP} reduce_scatter_test \
    -p ${NP} -b 512K -e 8M -f 1.25 -w 20 -n 100 -c 0 \
    | tee ${OUT}/reduce_scatter_np${NP}_mid.txt

  mpirun -n ${NP} reduce_scatter_test \
    -p ${NP} -b 16M -e 1G -f 2 -w 20 -n 100 -c 0 \
    | tee ${OUT}/reduce_scatter_np${NP}_large.txt
done
```

#### 5.3.4 alltoall（→ `nccl_perf.txt`，通用表）

```bash
#!/bin/bash
# collect_alltoall.sh
set -e
OUT=./raw_comm; mkdir -p ${OUT}

for NP in 2 4 8; do
  echo "=== alltoall  np=${NP} ==="

  mpirun -n ${NP} alltoall_test \
    -p ${NP} -b 2K -e 256K -f 2 -w 20 -n 100 -c 0 \
    | tee ${OUT}/alltoall_np${NP}_small.txt

  mpirun -n ${NP} alltoall_test \
    -p ${NP} -b 512K -e 8M -f 1.25 -w 20 -n 100 -c 0 \
    | tee ${OUT}/alltoall_np${NP}_mid.txt

  mpirun -n ${NP} alltoall_test \
    -p ${NP} -b 16M -e 1G -f 2 -w 20 -n 100 -c 0 \
    | tee ${OUT}/alltoall_np${NP}_large.txt
done
```

### 5.4 一键总控脚本

```bash
#!/bin/bash
# collect_all_comm.sh —— 采集全部 4 个原语
set -e

bash collect_all_reduce.sh
bash collect_all_gather.sh
bash collect_reduce_scatter.sh
bash collect_alltoall.sh

echo "全部原始日志位于 ./raw_comm/"
echo "下一步：python parse_hccl_test_logs.py --input ./raw_comm --output ./hccl_data"
```

nohup bash collect_all_comm.sh > collect_all_comm_log.txt 2>&1 &

### 5.5 输出解析要点

不同工具的输出格式略有差异，典型输出片段：

```text
# size(B)    count      type      red    time(us)  algbw(GB/s)  busbw(GB/s)  #wrong
   1048576    524288    float16     sum     322.72       3.25        2.17    0
```

解析时提取三列即可：

| 输出列 | 对应框架字段 | 换算 |
|--------|-------------|------|
| `size(B)` / `count` | `message_size` | 优先用 `count`（已是元素数）；若只有 `size(B)` 则 `/ dtype_bytes` |
| `time(us)` | `latency` | `/ 1000` → ms |
| — | `power` | 默认 `0.0`（见 [§8.3](#83-功耗列如何填)） |

> **all_gather / reduce_scatter 额外注意**：确认工具的 `size/count` 是"每 rank"还是"全局"，按 [§4.2](#42-message_size-的语义按原语区分) 换算成"完整张量元素数"。

---

## 6. 采集方案 B：torchrun + torch_npu Python 脚本

Python 方案**直接输出框架 CSV**，无需解析转换，且语义与 `query_*` 接口严格对齐。推荐优先使用。

### 6.1 统一采集脚本

```python
#!/usr/bin/env python3
"""
collect_hccl_comm.py —— HCCL 集合通信性能统一采集脚本
输出：custom_allreduce_perf.txt + nccl_perf.txt（框架可直接加载）

用法:
  torchrun --nproc_per_node=8 collect_hccl_comm.py \
      --ops all_reduce all_gather reduce_scatter alltoall \
      --tp-sizes 2 4 8 --dtype half \
      --out-dir ./hccl_data
"""

import argparse
import csv
import os
from typing import Dict, List

import torch
import torch.distributed as dist

# 框架 message_size 网格（元素数）—— 见 §4.3
DEFAULT_SIZES = [
    # decode 小消息
    1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
    # 中消息
    524288, 655360, 786432, 917504, 1048576,
    1310720, 1572864, 1835008, 2097152,
    2621440, 3145728, 3670016, 4194304, 5242880, 6291456, 8388608,
    # 大消息
    16777216, 33554432, 67108864, 134217728, 268435456, 536870912,
]

DTYPE_MAP = {
    "half": (torch.float16, 2),
    "bf16": (torch.bfloat16, 2),
    "fp8": (torch.float8_e4m3fn, 1),
    "int8": (torch.int8, 1),
}


def setup(rank: int, world_size: int) -> None:
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    torch.npu.set_device(rank)


def bench_op(op: str, numel: int, dtype: torch.dtype, iters: int, warmup: int) -> float:
    """
    测量单个 (op, numel) 点的平均延迟 (ms)。

    语义对齐框架 query 接口（§4.2）：
      - all_reduce      : 本地张量 numel = message_size（in = out）
      - all_gather      : 本地输入 numel = message_size / N，gather 后 out = message_size
      - reduce_scatter  : 本地输入 numel = message_size，scatter 后 out = message_size / N
      - alltoall        : 每 rank 输入/输出均为 message_size / N
    """
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device(f"npu:{rank}")

    def local_numel() -> int:
        if op == "all_reduce":
            return numel
        if op == "reduce_scatter":
            return numel
        # all_gather / alltoall: 本地输入 = 完整 / N
        return numel // world

    n_local = local_numel()
    if n_local <= 0:
        return 0.0

    send = torch.randn(n_local, dtype=dtype, device=device)

    if op == "all_reduce":
        def once():
            t = send.clone()
            dist.all_reduce(t)
        out_shape_note = "in=out=message_size"

    elif op == "all_gather":
        # 框架 message_size = 完整输出元素数
        recv = torch.empty(numel, dtype=dtype, device=device)
        recv_list = list(recv.chunk(world))
        def once():
            dist.all_gather(recv_list, send)
        out_shape_note = "in=message_size/N, out=message_size"

    elif op == "reduce_scatter":
        recv = torch.empty(n_local // world, dtype=dtype, device=device)
        send_list = list(send.chunk(world))
        def once():
            dist.reduce_scatter(recv, send_list)
        out_shape_note = "in=message_size, out=message_size/N"

    elif op == "alltoall":
        recv = torch.empty(n_local, dtype=dtype, device=device)
        def once():
            dist.all_to_all_single(recv, send)
        out_shape_note = "in=out=message_size/N"

    else:
        raise ValueError(f"unsupported op: {op}")

    # warmup
    for _ in range(warmup):
        once()
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        once()
    end.record()
    torch.npu.synchronize()

    latency_ms = start.elapsed_time(end) / iters
    if rank == 0:
        print(f"  [{op}] message_size={numel:>10}  local={n_local:>10}  "
              f"({out_shape_note})  latency={latency_ms:.6f} ms")
    return latency_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", nargs="+",
                        default=["all_reduce", "all_gather", "reduce_scatter", "alltoall"])
    parser.add_argument("--tp-sizes", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--dtype", default="half", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--message-sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--out-dir", default="./hccl_data")
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 8))
    setup(rank, world_size)

    dtype, dtype_bytes = DTYPE_MAP[args.dtype]
    # 框架硬编码 custom_allreduce 为 half（perf_database.py:416）
    framework_dtype_str = "half"

    all_reduce_rows: List[Dict] = []
    nccl_rows: List[Dict] = []

    for tp in args.tp_sizes:
        if tp > world_size:
            continue
        group = dist.new_group(ranks=list(range(tp)))
        for op in args.ops:
            if rank == 0:
                print(f"\n===== op={op}  tp={tp} =====")
            for numel in args.message_sizes:
                # all_gather/alltoall 的本地张量必须整除
                if op in ("all_gather", "alltoall") and (numel % tp != 0):
                    continue
                if op == "reduce_scatter" and (numel % tp != 0):
                    continue

                lat = bench_op(op, numel, dtype, args.iters, args.warmup)
                if rank != 0:
                    continue

                if op == "all_reduce":
                    # 同一份数据写两个文件（见 §3.2）
                    all_reduce_rows.append({
                        "allreduce_dtype": framework_dtype_str,
                        "num_gpus": tp,
                        "message_size": numel,
                        "latency": f"{lat:.6f}",
                        "power": "0.0",
                    })
                    nccl_rows.append({
                        "nccl_dtype": framework_dtype_str,
                        "op_name": "all_reduce",
                        "num_gpus": tp,
                        "message_size": numel,
                        "latency": f"{lat:.6f}",
                        "power": "0.0",
                    })
                else:
                    nccl_rows.append({
                        "nccl_dtype": framework_dtype_str,
                        "op_name": op,
                        "num_gpus": tp,
                        "message_size": numel,
                        "latency": f"{lat:.6f}",
                        "power": "0.0",
                    })
        dist.destroy_process_group(group)

    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)

        car_path = os.path.join(args.out_dir, "custom_allreduce_perf.txt")
        with open(car_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["allreduce_dtype", "num_gpus", "message_size", "latency", "power"],
            )
            w.writeheader()
            w.writerows(all_reduce_rows)
        print(f"\n[Saved] {car_path}  ({len(all_reduce_rows)} rows)")

        nccl_path = os.path.join(args.out_dir, "nccl_perf.txt")
        with open(nccl_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["nccl_dtype", "op_name", "num_gpus", "message_size", "latency", "power"],
            )
            w.writeheader()
            w.writerows(nccl_rows)
        print(f"[Saved] {nccl_path}  ({len(nccl_rows)} rows)")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

### 6.2 启动命令

```bash
export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# 全量采集（4 个原语 × tp 2/4/8 × 全 size 网格）
torchrun --nproc_per_node=8 collect_hccl_comm.py \
    --ops all_reduce all_gather reduce_scatter alltoall \
    --tp-sizes 2 4 8 \
    --dtype half \
    --iters 100 --warmup 20 \
    --out-dir ./hccl_data

# 只补采 reduce_scatter（其余已有）
torchrun --nproc_per_node=8 collect_hccl_comm.py \
    --ops reduce_scatter --tp-sizes 2 4 8 --out-dir ./hccl_data
```

> **子通信组注意**：脚本用 `dist.new_group(ranks=range(tp))` 测小 TP。HCCL 对子组支持因版本而异，若 `new_group` 失败，改为**每个 tp_size 单独用 `torchrun --nproc_per_node=${tp}` 启动一次**，结果手工合并。

### 6.3 mpirun 启动 Python 脚本（等价方式）

若希望与 `all_reduce_test` 一样用 `mpirun` 拉起：

```bash
mpirun -n 8 \
  -x MASTER_ADDR=localhost \
  -x MASTER_PORT=29500 \
  -x LOCAL_RANK=0 \
  python -m torch.distributed.run \
    --nproc_per_node=8 \
    --master_port=29500 \
    collect_hccl_comm.py --ops all_reduce --tp-sizes 8
```

或直接让脚本读 `PMI_RANK` / `OMPI_COMM_WORLD_RANK`（需改 `setup()` 里的 rank/world_size 来源）。**实践上 `torchrun` 更省事**，`mpirun` 更适合与 C++ 测试工具统一调度。

---

## 7. 原始数据 → 框架格式转换

若用方案 A（C++ 工具）得到日志，用仓库内脚本转换：

```bash
python tools/parse_hccl_test_logs.py \
    --input /mnt/caikaiwei/projects/aic-collect-hccl/raw_comm \
    --output ./hccl_data_0923
```

脚本源码见 `tools/parse_hccl_test_logs.py`，支持三类日志：

1. **HCCL 管道分隔表格**（`all_reduce_test` 等工具的典型输出，`.log` / `.txt` 均可）：

   ```text
   the minbytes is 524288, maxbytes is 8388608, iters is 100, warmup_iters is 20
   data_size(Bytes) : | avg_time(us) : | alg_bandwidth(GB/s) : | check_result:
   524288        | 258.65    | 2.02703    | NULL
   ```

2. **nccl-tests 空格分隔表格**（含 `size / count / type / time(us)` 列）
3. **CSV 输出**（`size_bytes,latency_us`）

文件名必须符合 `{op}_np{N}_*.txt` 约定（如 `all_reduce_np8_mid.txt`），扩展名 `.log` / `.txt` / `.csv` 均可。

**关键换算再强调一次**：

| 原语 | 工具 `count/size` 含义 | 框架 `message_size` |
|------|------------------------|---------------------|
| all_reduce | 本地张量元素数 | `count`（直用） |
| all_gather | 每 rank **输入** 元素数 | `count * num_gpus` |
| reduce_scatter | 每 rank **输入** 或 **输出**（看工具） | 输入侧直用；输出侧 `count * num_gpus` |
| alltoall | 每 rank 元素数 | `count * num_gpus` |

---

## 8. 数据放置、验证与常见问题

### 8.1 正确的放置路径

```text
src/aiconfigurator_npu/systems/data/ascend_910b/
├── vllm-ascend/                       # ← backend 目录
│   └── 0.18.0/                        # ← version 目录
│       ├── custom_allreduce_perf.txt  # ← 放这里
│       ├── gemm_perf.txt
│       ├── moe_perf.txt
│       └── ...
└── nccl/                              # ← 固定目录名 "nccl"
    └── 2.26.0/                        # ← ascend_910b.yaml 的 misc.nccl_version
        └── nccl_perf.txt              # ← 放这里
```

复制命令：

```bash
DATA_ROOT=src/aiconfigurator_npu/systems/data/ascend_910b

mkdir -p ${DATA_ROOT}/nccl/2.26.0

cp ./hccl_data/custom_allreduce_perf.txt \
   ${DATA_ROOT}/vllm-ascend/0.18.0/

cp ./hccl_data/nccl_perf.txt \
   ${DATA_ROOT}/nccl/2.26.0/
```

> **注意**：`nccl_perf.txt` **不要**放到 `vllm-ascend/0.18.0/` 下——`perf_database.py:2152-2153` 会硬编码改走 `nccl/{nccl_version}/` 目录，放错位置将加载不到。backup 里 `data/systems/data/ascend_910b/vllm-ascend/0.18.0/nccl_perf.txt` 就属于放错位置的遗留。

### 8.2 加载验证

```python
# verify_comm_data.py
from aiconfigurator_npu.sdk.perf_database import get_database
from aiconfigurator_npu.sdk.common import CommQuantMode, DatabaseMode


def main():
    db = get_database(system="ascend_910b", backend="vllm-ascend", version="0.18.0")

    print("custom_allreduce loaded:", db._custom_allreduce_data.is_loaded)
    print("nccl loaded:           ", db._nccl_data.is_loaded)

    cases = [
        ("custom_allreduce", lambda: db.query_custom_allreduce(
            CommQuantMode.half, 8, 524288, DatabaseMode.SILICON)),
        ("custom_allreduce", lambda: db.query_custom_allreduce(
            CommQuantMode.half, 4, 4096 * 4096, DatabaseMode.SILICON)),
        ("nccl all_gather",  lambda: db.query_nccl(
            CommQuantMode.half, 8, "all_gather", 524288, DatabaseMode.SILICON)),
        ("nccl reduce_scatter", lambda: db.query_nccl(
            CommQuantMode.half, 8, "reduce_scatter", 524288, DatabaseMode.SILICON)),
        ("nccl alltoall", lambda: db.query_nccl(
            CommQuantMode.half, 8, "alltoall", 524288, DatabaseMode.SILICON)),
    ]
    for name, fn in cases:
        try:
            r = fn()
            print(f"{name:<24} latency={float(r):.6f} ms  energy={r.energy:.6f} W·ms")
        except Exception as e:
            print(f"{name:<24} FAILED: {e}")


if __name__ == "__main__":
    main()
```

**验证要点**：
1. 两个 `is_loaded` 均为 `True`
2. `reduce_scatter` 查询不 raise（这是 backup 数据最常缺的一环）
3. 小消息（如 4096）与大消息（如 536870912）都在表内或插值区间内
4. 对比 SILICON 与 SOL 结果量级：`latency_SILICON ≈ latency_SOL / 0.5~0.9` 属合理区间

### 8.3 功耗列如何填

框架按 `energy = power * latency` 计算能耗（`perf_database.py:422` / `:478`）。

- **不测功耗**：填 `0.0`，能耗恒为 0（当前 backup 即如此），不影响延迟预测
- **测功耗**：采集期间用 `npu-smi info -t power` 周期采样，取均值填入

```bash
# 采集期间后台采功耗（每 100ms 一次）
while true; do
  npu-smi info -t power -i 0 | grep -Eo '[0-9]+' | head -1 >> power.log
  sleep 0.1
done &
POWER_PID=$!
# ... 跑 collect_hccl_comm.py ...
kill ${POWER_PID}
awk '{s+=$1; n++} END {print s/n}' power.log   # 平均功率 (W)
```

### 8.4 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `Failed to query nccl data ...` (SILICON) | 文件缺失 / 路径放错 / 缺 `reduce_scatter` 行 | 按 §8.1 放置；确认 `op_name` 覆盖 4 类 |
| `KeyError: 'nccl_dtype'` | 表头拼写错误 | 表头必须严格为 `nccl_dtype,op_name,num_gpus,message_size,latency,power` |
| 插值严重失真 | size 点太稀，尤其小消息段 | 按 §4.3 网格补点；decode 段每 2× 一点 |
| 查 `tp=16` 结果离谱 | 表内只有 8 卡数据，靠缩放外推 | 补采 16/32 卡实测；或改用 HYBRID 让 SOL 兜底 |
| all_gather 延迟 ≈ all_reduce 一半但模型偏慢 | `message_size` 语义搞反（每 rank vs 全局） | 按 §4.2 统一为"完整张量元素数" |
| Python 采集首点异常慢 | 首迭代建链/内存池未热 | `warmup ≥ 20`；丢弃首个 size 点 |
| `new_group` 报 HCCL 错 | 子通信组不支持 | 每个 tp_size 单独 `torchrun --nproc_per_node=tp` |
| MoE alltoall 查不到 | MoE 走 `trtllm_alltoall_perf.txt`，不是 `nccl_perf.txt` | 见 §3.4，按 `query_trtllm_alltoall` 的多维格式单独采集 |

---

## 附录 A：完整采集 Checklist

**P0（最小可用集，必须）**

- [ ] `all_reduce` × `num_gpus` ∈ {2,4,8} × 全 size 网格 → `custom_allreduce_perf.txt` + `nccl_perf.txt`
- [ ] `all_gather` × `num_gpus` ∈ {2,4,8} × 全 size 网格 → `nccl_perf.txt`
- [ ] `reduce_scatter` × `num_gpus` ∈ {2,4,8} × 全 size 网格 → `nccl_perf.txt`
- [ ] dtype = `half`
- [ ] 放置到 `systems/data/ascend_910b/vllm-ascend/0.18.0/` 与 `.../nccl/2.26.0/`
- [ ] 跑 §8.2 验证脚本，SILICON 模式全绿

**P1（MoE / WideEP 场景）**

- [ ] `alltoall` × `num_gpus` ∈ {2,4,8} × 全 size 网格 → `nccl_perf.txt`
- [ ] MoE dispatch/combine → `trtllm_alltoall_perf.txt`（按 `moe_ep_size × hidden × topk × num_experts × num_tokens` 裁剪）
- [ ] 跨节点：`num_gpus` ∈ {16,32} 的 4 原语

**P2（增强）**

- [ ] `fp8` / `int8` 通信量化数据（`nccl_perf.txt` 的 `nccl_dtype` 列）
- [ ] 功耗列实测（`power` → `energy = power * latency`）
- [ ] 与 SOL 模型交叉校准，更新 `ascend_910b.yaml` 的 `intra_node_bw` / `inter_node_bw`

**规模估算（P0）**：

| 项 | 数量 |
|----|------|
| 原语 | 3（AR + AG + RS） |
| num_gpus | 3（2/4/8） |
| size 点 | ~31 |
| 合计测量点 | 3 × 3 × 31 ≈ **280** |
| 每点耗时（warmup 20 + iters 100） | 秒级 |
| 总时长 | 约十几分钟到 1 小时（视大消息段） |

---

## 附录 B：相关源码索引

| 文件 | 行号 | 说明 |
|------|------|------|
| `src/aiconfigurator_npu/sdk/common.py` | 551-583 | `PerfDataFilename`（文件名枚举） |
| `src/aiconfigurator_npu/sdk/common.py` | 528-537 | `DatabaseMode`（SILICON/HYBRID/SOL/...） |
| `src/aiconfigurator_npu/sdk/common.py` | 645-652 | `CommQuantMode`（half/int8/fp8） |
| `src/aiconfigurator_npu/sdk/perf_database.py` | 366-438 | `load_custom_allreduce_data` |
| `src/aiconfigurator_npu/sdk/perf_database.py` | 441-493 | `load_nccl_data` |
| `src/aiconfigurator_npu/sdk/perf_database.py` | 2102-2176 | `PerfDatabase.__init__` 加载与路径分流 |
| `src/aiconfigurator_npu/sdk/perf_database.py` | 4531-4633 | `query_custom_allreduce` |
| `src/aiconfigurator_npu/sdk/perf_database.py` | 4635-4745 | `query_nccl` |
| `src/aiconfigurator_npu/sdk/perf_database.py` | 1908-2010 | `load_trtllm_alltoall_data`（MoE alltoall） |
| `src/aiconfigurator_npu/sdk/operations.py` | 40-62 | `CustomAllReduce`（all_reduce 场景） |
| `src/aiconfigurator_npu/sdk/operations.py` | 93-122 | `NCCL`（all_gather/reduce_scatter 场景） |
| `src/aiconfigurator_npu/sdk/operations.py` | 383-447 | `TrtLLMWideEPMoEDispatch`（MoE alltoall 场景） |
| `src/aiconfigurator_npu/sdk/models.py` | 2135-2165 | DEEPSEEKV32 的 AG + RS |
| `src/aiconfigurator_npu/sdk/models.py` | 2857-2889 | WideEP DEEPSEEKV32 的 AG + RS |
| `src/aiconfigurator_npu/systems/ascend_910b.yaml` | 17-31 | 带宽 / `nccl_version` / `nccl_mem` |
| `docs/HCCL_ALLREDUCE_PERF_DATA_GUIDE.md` | — | 旧版（仅 all_reduce），本文档的子集 |
