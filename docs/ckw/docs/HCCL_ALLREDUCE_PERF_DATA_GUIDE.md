# HCCL AllReduce 性能数据采集与使用指南

## 目录

- [1. 概述](#1-概述)
- [2. 数据文件说明](#2-数据文件说明)
- [3. 数据采集方法](#3-数据采集方法)
  - [3.1 采集脚本编写](#31-采集脚本编写)
  - [3.2 采集命令](#32-采集命令)
  - [3.3 数据格式转换](#33-数据格式转换)
- [4. 数据使用方式](#4-数据使用方式)
  - [4.1 数据加载](#41-数据加载)
  - [4.2 数据查询接口](#42-数据查询接口)
  - [4.3 在模型推理中使用](#43-在模型推理中使用)
- [5. 完整示例](#5-完整示例)
- [6. 常见问题](#6-常见问题)

---

## 1. 概述

HCCL (Huawei Collective Communication Library) 是华为 Ascend NPU 的集合通信库，类似于 NVIDIA 的 NCCL。在 AIConfigurator 中，HCCL AllReduce 性能数据用于：

1. **预测 Tensor Parallelism 通信开销** - 当 TP > 1 时，需要 AllReduce 操作同步梯度
2. **优化模型并行配置** - 评估不同 TP/PP/DP 配置下的通信延迟
3. **生成最优部署配置** - 在满足 SLA 约束下找到最优的并行策略

### 相关文件

| 文件名 | 说明 | 数据位置 |
|--------|------|----------|
| `custom_allreduce_perf.txt` | 自定义 AllReduce 性能数据 (vLLM/SGLang 后端) | `systems/data/ascend_910b/vllm-ascend/0.18.0/` |
| `nccl_perf.txt` | NCCL/HCCL 集合通信性能数据 | `systems/data/ascend_910b/nccl/2.26.0/` |

---

## 2. 数据文件说明

### 2.1 custom_allreduce_perf.txt 格式

```csv
allreduce_dtype,num_gpus,message_size,latency,power
half,8,524288,0.322721,0.0
half,8,655360,0.322460,0.0
half,8,786432,0.337648,0.0
half,8,917504,0.318719,0.0
half,8,1048576,0.345686,0.0
half,8,1310720,0.414392,0.0
half,8,1572864,0.354691,0.0
half,8,1835008,0.400689,0.0
half,8,2097152,0.300129,0.0
half,8,2621440,0.319911,0.0
half,8,3145728,0.225305,0.0
half,8,3670016,0.235057,0.0
half,8,8388608,0.346606,0.0
half,8,33554432,1.032035,0.0
half,8,67108864,1.948760,0.0
half,8,134217728,4.135939,0.0
half,8,268435456,7.874923,0.0
half,8,536870912,15.032148,0.0
```

**字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `allreduce_dtype` | string | 数据类型，如 `half` (FP16), `fp8`, `int8` |
| `num_gpus` | int | 参与 AllReduce 的 GPU/NPU 数量 (即 TP size) |
| `message_size` | int | 消息大小 (元素数量，非字节数) |
| `latency` | float | 延迟 (毫秒, ms) |
| `power` | float | 功耗 (瓦特, W)，可选字段，默认为 0.0 |

**扩展格式 (vLLM/SGLang)**:

```csv
allreduce_dtype,num_gpus,message_size,latency,power,kernel_source,backend
half,8,524288,0.322721,0.0,vLLM_custom_graph,vllm_graph
half,8,524288,0.450000,0.0,vLLM_custom_eager,vllm_eager
```

| 字段 | 说明 |
|------|------|
| `kernel_source` | 内核来源标识，如 `vLLM_custom_graph`, `SGLang_CustomAllReduce_graph` |
| `backend` | 后端标识，如 `vllm_graph`, `sglang_graph` |

**注意**: 系统会自动过滤 eager 模式数据，仅保留 graph 模式数据（性能更好）。

### 2.2 nccl_perf.txt 格式

```csv
nccl_dtype,op_name,num_gpus,message_size,latency,power
half,all_reduce,8,524288,0.322721,0.0
half,all_reduce,8,655360,0.322460,0.0
half,all_gather,8,524288,0.406493,0.0
half,all_gather,8,655360,0.434424,0.0
half,reduce_scatter,8,524288,0.350000,0.0
```

**字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `nccl_dtype` | string | 数据类型 |
| `op_name` | string | 操作类型: `all_reduce`, `all_gather`, `reduce_scatter`, `alltoall` |
| `num_gpus` | int | 参与通信的 GPU/NPU 数量 |
| `message_size` | int | 消息大小 (元素数量) |
| `latency` | float | 延迟 (毫秒, ms) |
| `power` | float | 功耗 (瓦特, W) |

---

## 3. 数据采集方法

### 3.1 采集脚本编写

在 Ascend 910B 平台上采集 HCCL 性能数据需要使用 PyTorch NPU 和 HCCL 库。以下是一个完整的采集脚本示例：

```python
#!/usr/bin/env python3
"""
HCCL AllReduce 性能数据采集脚本
用于采集 Ascend 910B 平台上的 AllReduce 通信性能数据
"""

import torch
import torch_npu
import csv
import time
from typing import List, Tuple


def setup_distributed(rank: int, world_size: int):
    """初始化分布式环境"""
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    torch.distributed.init_process_group(
        backend='hccl',
        rank=rank,
        world_size=world_size
    )
    torch.npu.set_device(rank)


def benchmark_allreduce(
    tensor_size: int,
    num_iters: int = 100,
    warmup_iters: int = 20,
    dtype: torch.dtype = torch.float16
) -> float:
    """
    Benchmark AllReduce 操作

    Args:
        tensor_size: 张量元素数量
        num_iters: 测试迭代次数
        warmup_iters: 预热迭代次数
        dtype: 数据类型

    Returns:
        平均延迟 (毫秒)
    """
    device = torch.device(f'npu:{torch.distributed.get_rank()}')

    # 创建输入张量
    tensor = torch.randn(tensor_size, dtype=dtype, device=device)

    # 预热
    for _ in range(warmup_iters):
        torch.distributed.all_reduce(tensor)
    torch.npu.synchronize()

    # 计时
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        torch.distributed.all_reduce(tensor)
    end_event.record()
    torch.npu.synchronize()

    # 计算平均延迟 (ms)
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_latency_ms = elapsed_time_ms / num_iters

    return avg_latency_ms


def collect_allreduce_data(
    message_sizes: List[int],
    tp_sizes: List[int],
    output_file: str,
    dtype: torch.dtype = torch.float16
):
    """
    采集 AllReduce 性能数据

    Args:
        message_sizes: 消息大小列表 (元素数量)
        tp_sizes: TP size 列表
        output_file: 输出文件路径
        dtype: 数据类型
    """
    rank = torch.distributed.get_rank()

    results = []

    for tp_size in tp_sizes:
        if tp_size > torch.distributed.get_world_size():
            continue

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Testing TP size: {tp_size}")
            print(f"{'='*60}")

        for msg_size in message_sizes:
            if rank == 0:
                print(f"  Message size: {msg_size:>12} elements ({msg_size * 2 / 1024 / 1024:.2f} MB for FP16)")

            # 只有前 tp_size 个 rank 参与测试
            if rank < tp_size:
                # 创建子进程组
                group = torch.distributed.new_group(ranks=list(range(tp_size)))

                latency = benchmark_allreduce(
                    tensor_size=msg_size,
                    num_iters=100,
                    warmup_iters=20,
                    dtype=dtype
                )

                if rank == 0:
                    results.append({
                        'allreduce_dtype': 'half' if dtype == torch.float16 else 'fp8',
                        'num_gpus': tp_size,
                        'message_size': msg_size,
                        'latency': f'{latency:.6f}',
                        'power': '0.0'  # 功耗测量需要额外工具
                    })
                    print(f"    Latency: {latency:.6f} ms")

            torch.distributed.barrier()

    # 保存结果
    if rank == 0:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'allreduce_dtype', 'num_gpus', 'message_size', 'latency', 'power'
            ])
            writer.writeheader()
            writer.writerows(results)

        print(f"\nResults saved to {output_file}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Collect HCCL AllReduce performance data')
    parser.add_argument('--output', type=str, default='custom_allreduce_perf.txt',
                        help='Output file path')
    parser.add_argument('--tp-sizes', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='TP sizes to test')
    parser.add_argument('--message-sizes', type=int, nargs='+',
                        default=[
                            512 * 1024,      # 512K
                            640 * 1024,      # 640K
                            768 * 1024,      # 768K
                            896 * 1024,      # 896K
                            1024 * 1024,     # 1M
                            1280 * 1024,     # 1.25M
                            1536 * 1024,     # 1.5M
                            1792 * 1024,     # 1.75M
                            2048 * 1024,     # 2M
                            2560 * 1024,     # 2.5M
                            3072 * 1024,     # 3M
                            3584 * 1024,     # 3.5M
                            8192 * 1024,     # 8M
                            32 * 1024 * 1024, # 32M
                            64 * 1024 * 1024, # 64M
                            128 * 1024 * 1024, # 128M
                            256 * 1024 * 1024, # 256M
                            512 * 1024 * 1024, # 512M
                        ],
                        help='Message sizes to test (in elements)')
    parser.add_argument('--dtype', type=str, default='half', choices=['half', 'fp8'],
                        help='Data type to test')

    args = parser.parse_args()

    # 初始化分布式环境
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 8))

    setup_distributed(rank, world_size)

    dtype = torch.float16 if args.dtype == 'half' else torch.float8_e4m3fn

    # 采集数据
    collect_allreduce_data(
        message_sizes=args.message_sizes,
        tp_sizes=args.tp_sizes,
        output_file=args.output,
        dtype=dtype
    )

    # 清理
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    import os
    main()
```

### 3.2 采集命令

#### 方法 1: 使用 torchrun 启动

```bash
# 在 8 卡 Ascend 910B 节点上采集
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    collect_hccl_allreduce.py \
    --output custom_allreduce_perf.txt \
    --tp-sizes 1 2 4 8 \
    --dtype half
```

#### 方法 2: 使用 mpirun 启动

```bash
# 使用 OpenMPI 启动
mpirun -np 8 \
    -x MASTER_ADDR=localhost \
    -x MASTER_PORT=29500 \
    python collect_hccl_allreduce.py \
    --output custom_allreduce_perf.txt \
    --tp-sizes 1 2 4 8
```

#### 方法 3: 使用 Ascend npu-smi 工具配合

```bash
# 设置 NPU 环境
export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 运行采集脚本
torchrun --nproc_per_node=8 collect_hccl_allreduce.py
```

### 3.3 数据格式转换

如果采集的原始数据格式与 AIConfigurator 要求的格式不同，可以使用以下脚本转换：

```python
#!/usr/bin/env python3
"""
将原始 HCCL 采集数据转换为 AIConfigurator 格式
"""

import csv
from typing import List, Dict


def convert_raw_to_aic_format(
    input_file: str,
    output_file: str,
    dtype: str = 'half'
):
    """
    转换原始数据为 AIConfigurator 格式

    原始格式 (假设):
        tp_size, message_bytes, latency_ms

    AIConfigurator 格式:
        allreduce_dtype, num_gpus, message_size, latency, power
    """
    results = []

    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头

        for row in reader:
            tp_size = int(row[0])
            message_bytes = int(row[1])
            latency_ms = float(row[2])

            # 将字节数转换为元素数量 (FP16 = 2 bytes per element)
            message_elements = message_bytes // 2

            results.append({
                'allreduce_dtype': dtype,
                'num_gpus': tp_size,
                'message_size': message_elements,
                'latency': f'{latency_ms:.6f}',
                'power': '0.0'
            })

    # 保存转换后的数据
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'allreduce_dtype', 'num_gpus', 'message_size', 'latency', 'power'
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"Converted {len(results)} records to {output_file}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input file path')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--dtype', default='half', help='Data type')

    args = parser.parse_args()

    convert_raw_to_aic_format(args.input, args.output, args.dtype)
```

---

## 4. 数据使用方式

### 4.1 数据加载

AIConfigurator 在初始化 `PerfDatabase` 时自动加载性能数据：

```python
# perf_database.py 中的加载逻辑

def load_custom_allreduce_data(custom_allreduce_file):
    """
    加载 custom allreduce 性能数据

    数据结构:
        custom_allreduce_data[dtype][tp_size][strategy][message_size] = {
            'latency': float,  # ms
            'power': float,    # W
            'energy': float    # W*ms
        }
    """
    if not os.path.exists(custom_allreduce_file):
        return None

    custom_allreduce_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))

    with open(custom_allreduce_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        # 解析数据
        dtype = common.CommQuantMode.half  # 目前只支持 half
        tp_size = int(row['num_gpus'])
        message_size = int(row['message_size'])
        latency = float(row['latency'])
        power = float(row.get('power', 0.0))
        energy = power * latency

        # 存储数据
        custom_allreduce_data[dtype][tp_size]['AUTO'][message_size] = {
            'latency': latency,
            'power': power,
            'energy': energy
        }

    return custom_allreduce_data
```

### 4.2 数据查询接口

```python
# perf_database.py 中的查询接口

class PerfDatabase:
    @functools.lru_cache(maxsize=32768)
    def query_custom_allreduce(
        self,
        quant_mode: common.CommQuantMode,
        tp_size: int,
        size: int,  # 元素数量
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult:
        """
        查询 AllReduce 操作延迟

        Args:
            quant_mode: 通信量化模式
            tp_size: Tensor Parallelism 大小
            size: 元素数量
            database_mode: 数据库模式 (SILICON, HYBRID, SOL, EMPIRICAL)

        Returns:
            PerformanceResult: 包含延迟和能耗的结果对象
        """

        # SOL (Speed of Light) 理论计算
        def get_sol(quant_mode, tp_size, size):
            if tp_size == 1:
                return 0, 0, 0

            p2p_bw = self._get_p2p_bandwidth(tp_size)
            # Ring AllReduce: 2 * (N-1)/N * data_size / bandwidth
            sol_time = 2 * size * 2 / tp_size * (tp_size - 1) / p2p_bw
            return sol_time * 1000, 0, 0  # 转换为 ms

        # SILICON 模式: 使用实测数据
        def get_silicon():
            if tp_size == 1:
                return PerformanceResult(0.0, energy=0.0)

            self._custom_allreduce_data.raise_if_not_loaded()

            # 获取该 tp_size 下的所有数据点
            comm_dict = self._custom_allreduce_data[quant_mode][tp_size]['AUTO']

            # 线性插值查询
            size_left, size_right = self._nearest_1d_point_helper(
                size, list(comm_dict.keys()), inner_only=False
            )
            result = self._interp_1d(
                [size_left, size_right],
                [comm_dict[size_left], comm_dict[size_right]],
                size
            )

            lat = result['latency'] if isinstance(result, dict) else result
            energy = result.get('energy', 0.0) if isinstance(result, dict) else 0.0

            return PerformanceResult(lat, energy=energy)

        # 根据模式返回结果
        if database_mode == common.DatabaseMode.SOL:
            return PerformanceResult(get_sol(quant_mode, tp_size, size)[0], energy=0.0)
        elif database_mode == common.DatabaseMode.SILICON:
            return get_silicon()
        elif database_mode == common.DatabaseMode.HYBRID:
            # HYBRID 模式: 优先使用实测数据，缺失时使用 SOL
            try:
                return get_silicon()
            except:
                return PerformanceResult(get_sol(quant_mode, tp_size, size)[0], energy=0.0)
```

### 4.3 在模型推理中使用

在 `operations.py` 中，AllReduce 延迟被用于计算 Tensor Parallelism 的通信开销：

```python
# operations.py 中的使用示例

class MoEDispatch:
    def analyze(self, database, num_tokens, ...):
        """
        分析 MoE 模块的延迟，包括通信开销
        """
        comm_latency = 0

        # 计算 AllReduce 通信延迟
        if self._attention_tp_size > 1:
            # 计算通信量 (元素数量)
            volume = num_tokens * self._hidden_size

            # 查询 AllReduce 延迟
            comm_latency = database.query_custom_allreduce(
                common.CommQuantMode.half,  # 通信数据类型
                self.num_gpus,              # TP size
                volume                      # 元素数量
            )

        return compute_latency + comm_latency


class LinearOp:
    def analyze(self, database, ...):
        """
        分析 Linear 层的延迟
        """
        # 计算 GEMM 延迟
        gemm_latency = database.query_gemm(...)

        # 计算通信延迟 (如果 TP > 1)
        comm_latency = 0
        if self.tp_size > 1:
            # 输出张量需要 AllReduce
            output_size = batch_size * seq_len * self.output_dim
            comm_latency = database.query_custom_allreduce(
                common.CommQuantMode.half,
                self.tp_size,
                output_size
            )

        return gemm_latency + comm_latency
```

---

## 5. 完整示例

### 5.1 采集 Qwen3-8B 在 Ascend 910B 上的 AllReduce 数据

```bash
#!/bin/bash
# collect_qwen3_8b_allreduce.sh
# 采集 Qwen3-8B 模型在 Ascend 910B 上的 AllReduce 性能数据

set -e

# 配置环境
export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# 输出目录
OUTPUT_DIR="./data/hccl_data"
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "Collecting HCCL AllReduce Performance Data"
echo "=========================================="
echo "Platform: Ascend 910B"
echo "Output: ${OUTPUT_DIR}/custom_allreduce_perf.txt"
echo ""

# 运行采集脚本
torchrun \
    --nproc_per_node=8 \
    --master_port=${MASTER_PORT} \
    collect_hccl_allreduce.py \
    --output ${OUTPUT_DIR}/custom_allreduce_perf.txt \
    --tp-sizes 1 2 4 8 \
    --dtype half \
    --message-sizes \
        524288 655360 786432 917504 1048576 \
        1310720 1572864 1835008 2097152 2621440 \
        3145728 3670016 8388608 \
        33554432 67108864 134217728 268435456 536870912

echo ""
echo "=========================================="
echo "Collection Complete!"
echo "=========================================="
echo "Output file: ${OUTPUT_DIR}/custom_allreduce_perf.txt"
echo ""
echo "To use this data, copy to:"
echo "  systems/data/ascend_910b/vllm-ascend/0.18.0/custom_allreduce_perf.txt"
```

### 5.2 复制数据到 AIConfigurator

```bash
# 复制到系统数据目录
cp data/hccl_data/custom_allreduce_perf.txt \
   src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.18.0/

# 或者复制到自定义系统路径
cp data/hccl_data/custom_allreduce_perf.txt \
   systems/data/ascend_910b/vllm-ascend/0.18.0/
```

### 5.3 验证数据

```python
#!/usr/bin/env python3
"""
验证 AllReduce 性能数据是否正确加载
"""

from aiconfigurator_npu.sdk.perf_database import get_database
from aiconfigurator_npu.sdk.common import CommQuantMode, DatabaseMode


def verify_allreduce_data():
    """验证 AllReduce 数据"""

    # 加载数据库
    db = get_database(
        system='ascend_910b',
        backend='vllm-ascend',
        version='0.18.0'
    )

    if db is None:
        print("ERROR: Failed to load database")
        return

    print("Database loaded successfully!")
    print(f"System: {db.system}")
    print(f"Backend: {db.backend}")
    print(f"Version: {db.version}")
    print()

    # 测试查询
    test_cases = [
        (8, 1024 * 1024, "1M elements, TP=8"),
        (8, 8 * 1024 * 1024, "8M elements, TP=8"),
        (4, 1024 * 1024, "1M elements, TP=4"),
        (2, 1024 * 1024, "1M elements, TP=2"),
    ]

    print("Query Results:")
    print("-" * 60)
    print(f"{'Test Case':<30} {'Latency (ms)':<15} {'Energy (W·ms)':<15}")
    print("-" * 60)

    for tp_size, size, desc in test_cases:
        result = db.query_custom_allreduce(
            quant_mode=CommQuantMode.half,
            tp_size=tp_size,
            size=size,
            database_mode=DatabaseMode.SILICON
        )
        print(f"{desc:<30} {float(result):<15.6f} {result.energy:<15.6f}")

    print("-" * 60)


if __name__ == '__main__':
    verify_allreduce_data()
```

---

## 6. 常见问题

### 6.1 数据文件找不到

**问题**: `Custom allreduce data file not found`

**解决**: 检查数据文件路径是否正确：
```python
# 正确的路径结构
systems/
└── data/
    └── ascend_910b/
        └── vllm-ascend/
            └── 0.18.0/
                └── custom_allreduce_perf.txt
```

### 6.2 数据格式错误

**问题**: `KeyError: 'allreduce_dtype'`

**解决**: 确保 CSV 文件包含正确的列名：
```csv
allreduce_dtype,num_gpus,message_size,latency,power
half,8,524288,0.322721,0.0
```

### 6.3 插值精度问题

**问题**: 查询的消息大小不在数据点中

**解决**: AIConfigurator 使用线性插值，建议采集足够密集的数据点：
- 小消息 (1K-1M): 每 128K 一个点
- 大消息 (1M-100M): 每 8M 一个点
- 超大消息 (100M+): 每 64M 一个点

### 6.4 TP size 超出数据范围

**问题**: 查询的 TP size 大于采集的最大 TP size

**解决**: AIConfigurator 会自动使用 SOL 估算进行外推：
```python
# 当 tp_size > max_tp_size 时
# 使用带宽模型进行外推
scale_factor = (tp_size - 1) / tp_size * max_tp / (max_tp - 1) * base_bw / target_bw
extrapolated_latency = base_latency * scale_factor
```

### 6.5 功耗数据缺失

**问题**: 功耗列显示为 0.0

**解决**: 功耗测量需要额外工具支持：
```bash
# 使用 npu-smi 监控功耗
npu-smi info -d 0 -t power

# 或使用 torch_npu 的功耗接口
import torch_npu
power = torch_npu.npu_power_stats(0)  # 获取 NPU 0 的功耗
```

---

## 附录: 相关源码文件

| 文件 | 说明 |
|------|------|
| `src/aiconfigurator_npu/sdk/perf_database.py` | 性能数据库，包含数据加载和查询逻辑 |
| `src/aiconfigurator_npu/sdk/common.py` | 通用常量和枚举定义 |
| `src/aiconfigurator_npu/sdk/operations.py` | 算子定义，使用 AllReduce 数据 |
| `src/aiconfigurator_npu/systems/ascend_910b.yaml` | 系统规格定义 |
| `collector/bench_engine.py` | NPU 基准测试引擎 |
