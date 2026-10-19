# throughput_optimizer 执行链条与"跑 45 分钟无结果"分析

> 分析对象命令：
>
> ```bash
> python -m cli.inference.throughput_optimizer model_configs/msmodeling/GLM5.3-Flash \
>   --device ATLAS_800_A2_376T_64G --num-devices 128 \
>   --tp-sizes 1 2 4 8 16 32 64 \
>   --ep-sizes 1 2 4 8 16 32 64 \
>   --moe-dp-sizes 1 2 4 8 \
>   --input-length 8196 --output-length 1024 \
>   --compile \
>   --quantize-linear-action DISABLED --quantize-attention-action DISABLED \
>   --tpot-limit 100 --ttft-limit 20000 --jobs 32
> ```

## 0. TL;DR（先给结论）

| 现象 | 根因 |
|---|---|
| 跑了 45 分钟还没结果 | 该命令实际要评估 **175 个并行候选**，每个候选都要**重建一遍整模 + 重新 torch.compile + 约 10~30 次按新 batch shape 的重编译/前向追踪**。结果表在**所有候选全部跑完后**才统一汇总打印，因此中途"看不到任何结果"是正常表现 |
| 多个 python 进程 100% CPU | `--jobs 32` 起的是 **spawn 进程池**（`ProcessPoolExecutor`），32 个 worker 各自独立建图 + 跑 Dynamo/FX，全部是 **CPU 侧 Python/FX 开销**，与 NPU 无关 |
| 有 NPU 显存占用 | **msmodeling 仿真链路本身不占 NPU 显存、不下发 NPU kernel**：模型参数建在 `meta` device，`load_weights()` 是空实现，前向被 `Runtime` 以 `__torch_dispatch__` 拦截做算子记账。观察到的占用来自进程级 device context 或机器上其它进程（第 7 节给出核验方法） |

五个耗时放大因子（按贡献排序）：

1. **候选数爆炸**：`TP(7) × [EP×MOE-DP 合法对(25)] = 175` 个候选，`--max-search-combinations` **只告警不剪枝**。
2. **`--compile` + `dynamic_shapes=False`**：batch size 每变一次就重新触发 Dynamo trace + 一整套 FX pass（aot_autograd → 量化 → 融合 → SP → freezing → 常量折叠），GLM5.3-Flash 是 45 层 + 288 专家 + MLA/线性注意力混合 + 稀疏 indexer 的超大图。
3. **每个候选全量重建**：spawn worker 不继承任何状态，且 `_submit_task` 开头就 `torch.compiler.reset()`，候选之间**零复用**。
4. **`workload_cache` 单设备不启用**：只有 `--device` 传多个硬件 profile 时才开启跨进程 workload 复用，本命令是单 device，完全走冷路径。
5. **并发搜索每次评估 ~1 次全图追踪**：指数搜索(≤10) + 二分(~9~18) + 探针(1)，每个新 batch 档位 ≈ 1 次新 shape 的前向（meta 前向，但需重编译）。

---

## 1. 命令参数解析后到底确定了什么

入口：`cli/inference/throughput_optimizer.py`

| 阶段 | 位置 | 说明 |
|---|---|---|
| 声明式建 parser | `cli/registry/modules/throughput_optimizer.py:431` `SPEC` | 所有参数来自 registry；`--compile` → `do_compile` |
| 解析 + 归一化 | `cli/inference/throughput_optimizer.py:50` `arg_parse()` | 去重/校验 TP/EP/MOE-DP ≤ num_devices |
| 候选规模计算 | `cli/inference/throughput_optimizer.py:168-212` | 调 `resolve_parallel_search_candidates` + `count_search_combinations` |

关键默认补齐（命令里没写的）：

- `--performance-model` 未给 → `["analytic"]`（`throughput_optimizer.py:79`），即**解析 roofline 模型**，不读 profiling DB。
- `--num-mtp-tokens` 未给 → 0，MTP 候选 `[0]`（1 个）。
- `--dcp-sizes` 未给 → `[1]`（1 个）。
- `--pp-sizes` 未给 → **走 legacy 候选生成路径**（`parallel_runner.py:384`），PP=1。
- `--batch-range` 未给 → 并发搜索区间 `[1, 512]`（`base_throughput_optimizer.py:60 DEFAULT_MAX_BATCH_SIZE`）。
- `--concurrency-search-strategy` 未给 → `exponential`。
- `--max-batched-tokens` 未给 → **自动候选 `[32784, 16392, 8196]`**（`input_length × (4,2,1)`，`utils.py:46 AUTO_MAX_BATCHED_TOKENS_FACTORS`）。
- `--max-search-combinations` 未给 → 100（**仅告警阈值，不剪枝**）。

量化两个 `DISABLED` 意味着 `quantize_model` 基本不改写模块，但**不影响**"是否 compile / 是否重建模型"这两条主耗时链路。

---

## 2. 完整执行链条

```
python -m cli.inference.throughput_optimizer
│
├─ arg_parse()                              cli/inference/throughput_optimizer.py:50
│    ├─ build_argparser / parse_module_args  cli/registry/argparse_adapter.py
│    ├─ resolve_parallel_search_candidates   serving_cast/service/utils.py:596
│    └─ count_search_combinations            serving_cast/service/utils.py:612   → 196（告警 196）
│
├─ main()                                   cli/inference/throughput_optimizer.py:217
│    ├─ apply_compilation_config
│    ├─ check_device_targets                 cli/utils.py:70   （校验 comm grid ≥ 128）
│    └─ run_multi_device_loop                serving_cast/service/optimizer_curve_plots.py:675
│         │
│         ├─ 单 device → workload_cache = None     optimizer_curve_plots.py:692-696  ★冷路径
│         └─ ParallelRunner(run_args).run_agg()    serving_cast/parallel_runner.py:165
│              │
│              └─ _get_df_list()                  parallel_runner.py:660
│                   ├─ configs = list(_get_user_config())      → 175 个 UserInputConfig
│                   │    └─ _get_user_config_legacy()          parallel_runner.py:495
│                   │         └─ 五重循环 TP/EP/MOE-DP/MTP/DCP + 整除过滤  :611-637
│                   │
│                   └─ ProcessPoolExecutor(max_workers=32, mp_context=spawn)
│                        └─ executor.map(_submit_task, configs)   parallel_runner.py:702-710
│                             │
│                             │  ★ 以下对 175 个候选各执行一遍（32 路并发）
│                             └─ _submit_task()               parallel_runner.py:768
│                                  ├─ torch.compiler.reset() + recompile_limit=1e8   :785-788
│                                  ├─ _build_static_runner()  :347   (dynamic_shapes=False)
│                                  │    └─ _build_model_runner → ModelRunner(user_input)
│                                  │         ├─ create_performance_models (analytic)  model_runner.py:163
│                                  │         └─ build_model()                        model_builder.py:174
│                                  │              ├─ TransformerModel()  meta 建图     transformers/model.py:328
│                                  │              │    ├─ init_on_device_without_buffers("meta") + no_init_weights()
│                                  │              │    ├─ wrap_model / maybe_* / patch_* / quantize_model / shard_model
│                                  │              │    └─ load_weights()  ← 空实现
│                                  │              └─ torch.compile(backend=CompilerBackend)   model_builder.py:197
│                                  │
│                                  ├─ OptimizerFactory.create_strategy → AggThroughputOptimizer
│                                  └─ strategy.run(optimizer_data, batch_range)      parallel_runner.py:840
│                                       └─ BaseThroughputOptimizer.run()             base_throughput_optimizer.py:177
│                                            ├─ P0 自动 max_batched_tokens 重试       :196
│                                            ├─ P1 左端点探针 batch=1                 :245
│                                            ├─ P2 _exponential_search（≤10 轮）      :301
│                                            ├─ P3 二分精化（~log2 轮）               :269
│                                            └─ P4 concat + 排序 → OptimizerSummary   :285
│                                                 每次 get_inference_info:
│                                                   agg_throughput_optimizer.py:93
│                                                   ├─ _get_full_prefill_metrics()   :342
│                                                   │    ├─ prefill 整波 latency（跨 batch 命中缓存）
│                                                   │    ├─ prefill 余数波 latency
│                                                   │    └─ decode latency（每档 1 次新 shape）:425
│                                                   └─ early stop 判定（tpot/ttft/显存）:216
│
├─ 全部候选返回后：res.report_final_result(...)        optimizer_curve_plots.py:711  ★结果在此才打印
└─ render_cross_hardware_summary（单 device 为空操作）  optimizer_curve_plots.py:732
```

---

## 3. 候选数量：为什么是 175

`--pp-sizes` 未给 → 走 `_get_user_config_legacy`（`parallel_runner.py:495`）的五重循环，过滤规则在 `parallel_runner.py:611-637`：

```611:637:serving_cast/parallel_runner.py
        for tp in tp_list:
            if target_devices % tp != 0:
                continue
            dp = target_devices // tp
            token_domain = target_devices
            for ep in ep_list:
                if target_devices % ep != 0:
                    continue
                for moe_dp in moe_dp_list:
                    if target_devices % (ep * moe_dp) != 0:
                        continue
                    ...
                    for num_mtp_tokens in mtp_list:
                        for dcp in dcp_list:
                            if tp % dcp != 0:
                                continue
                            yield _build_user_input(...)
```

对 `num_devices = 128`：

- TP：`1,2,4,8,16,32,64` 全部整除 128 → **7**
- MTP：`[0]` → **1**
- DCP：`[1]` → **1**
- `(EP, MOE-DP)`：需 `EP × MOE-DP | 128`

| EP | 合法 MOE-DP | 个数 |
|---|---|---|
| 1 | 1,2,4,8 | 4 |
| 2 | 1,2,4,8 | 4 |
| 4 | 1,2,4,8 | 4 |
| 8 | 1,2,4,8 | 4 |
| 16 | 1,2,4,8 | 4 |
| 32 | 1,2,4（8 → 256 ✗） | 3 |
| 64 | 1,2（4/8 → 256/512 ✗） | 2 |
| **合计** | | **25** |

**实际候选 = 7 × 25 × 1 × 1 = 175**

而启动时打印的告警数字是 **196**（`7×7×4×1`，未做整除过滤），所以告警数字与实际执行数不一致是正常的：

```
[WARNING] Large number of parallel search combinations (196 = TP:7 x EP:7 x MOE-DP:4 x MTP:1 x DCP:1)
```

> ⚠️ `--max-search-combinations` **只是告警阈值**，超过后仍然照跑（`cli/inference/throughput_optimizer.py:200-212` 只 `print` 一行 WARNING）。不要误以为它能限制搜索规模。

另外：`is_valid_ep_domain`（EP token-domain 守恒，`utils.py:562`）只在开启 sequence-parallel / dispatch-ffn-combine 时才生效（`parallel_runner.py:607-610`），本命令两个开关都是默认 `False`（`user_config.py:114,116`），所以像 `TP=2 × EP=4 × MOE-DP=1` 这类 vLLM 语义下不成立的组合**也会被评估**。

---

## 4. 单个候选里到底做了什么（耗时构成）

### 4.1 建模型（CPU，每候选一次）

`ModelRunner.__init__` → `build_model()`：

- `TransformerModel.__init__` 在 `init_on_device_without_buffers("meta") + no_init_weights()` 下按 `config.json` 建结构（`transformers/model.py:351`），**不读 safetensors**，只下 `config.json`（`model_hub.py:28-52` 显式排除权重文件）。
- `load_weights()` 是空实现（`transformers/model.py:422`）。
- 依次执行 `wrap_model / maybe_enable_mtp / maybe_enable_dflash / maybe_enable_dspark / maybe_reuse_layers / patch_model / patch_rotary_emb / patch_attention / patch_mla / patch_moe / quantize_model / shard_model`。
- GLM5.3-Flash 规模（`models/GLM5.3-Flash/config.json`）：45 个 hidden layer + 1 个 MTP layer、`n_routed_experts=288`、MoE intermediate 2048、MLA（`kv_lora_rank=512`）+ 线性注意力混合、稀疏 attention indexer（`index_topk=2048`）、外加 24 层 vision tower。模块数量巨大，纯 Python 建图开销不可忽略。

### 4.2 torch.compile（CPU，每候选一次 + 每 shape 一次）

```193:202:tensor_cast/core/model_builder.py
    if user_input.do_compile:
        config.compilation.fusion_patterns.enable_dispatch_ffn_combine = bool(user_input.enable_dispatch_ffn_combine)
        model = torch.compile(
            model,
            backend=get_backend(device_name=user_input.device),
            dynamic=user_input.dynamic_shapes,
            fullgraph=use_full_graph,
        )
```

- backend 是自研 `CompilerBackend`（`tensor_cast/compilation/compile_backend.py:49`）：`aot_autograd` + 十余个 FX 改写 pass（量化 / 融合 / SP / `torch._inductor.freezing.freeze` / merge linear / 常量折叠 / multistream）。**没有 codegen、不编译 kernel**。
- 因为是惰性编译，**构建阶段不触发**；真正触发在第一次 forward。
- `dynamic_shapes=False` 被三处强制（`parallel_runner.py:354 / 439 / 547`）→ **按 shape 特化**：每换一个 batch/并发档位就重新 trace 一次全图并重跑全部 FX pass。这也是 `--compile` 时把 `recompile_limit` 抬到 `1e8` 的原因（`parallel_runner.py:785`，`LIMIT_COUNT = 1e8`）。
- GLM5.3-Flash 是 VL 模型（`hasattr(hf_config, "vision_config")`）→ `_prepare_vl_compile` 把 visual encoder 用 `torch._dynamo.disable` 包住并返回 `use_full_graph=False`（`model_builder.py:53-82`），即**允许 graph break**，编译更稳但图更碎。

### 4.3 并发搜索（每候选 ~10~30 次评估）

`BaseThroughputOptimizer.run` → `_run_with_auto_max_batched_tokens` → `_run_once`：

| 阶段 | 位置 | 评估次数 |
|---|---|---|
| P0 自动 max_batched_tokens | `base_throughput_optimizer.py:196` | `[32784, 16392, 8196]` 从大到小，**第一个非 prefill-OOM 就返回**（:217），正常只跑 1 轮；prefill OOM 才最多 ×3 |
| P1 左端点探针（batch=1） | `:245` | 1 |
| P2 指数搜索 | `:301` `_exponential_search` | `for _ in range(MAX_ITER_NUMS=10)`，512→1024→…→524288，命中 early stop 即 break |
| P3 二分精化 | `:269-283` | `~log2(right-left+1)`，≈9~18 |
| **合计** | | **约 11（SLO 立即触发）~29（最坏）次 `get_inference_info`** |

`--tpot-limit 100` 很紧（`e2el = ttft + decode_latency × 1024`，`tpot = e2el / output_length`），大概率在 batch=512 的第一轮指数探测就 early stop，因此**实际偏向 ~11 次/候选**。

### 4.4 一次 `get_inference_info` 内部

`--input-length 8196` + 自动 `max_batched_tokens=32784` → `ceil(8196/32784)=1` chunk → 走 `_get_full_prefill_metrics`（`agg_throughput_optimizer.py:342`，历史 wave 公式），**不走 chunked prefill 仿真**（那条路径 `_simulate_chunked_prefill` 会按去重 shape 批量 prefetch，是最贵路径，本命令没触发）。

```354:428:serving_cast/service/agg_throughput_optimizer.py
        prefill_batch_size = global_batched_token_limit // effective_input_length
        calc_nums_for_ttft = concurrency // prefill_batch_size
        left_calc_num = concurrency % prefill_batch_size
        ...
            ) = self._get_or_compute_latency(prefill_batch_size, optimizer_data, is_decode=False)   # 整波
            ...  = self._get_or_compute_latency(left_calc_num,     optimizer_data, is_decode=False)  # 余数波
        ...
        decode_latency, ... = self._get_or_compute_latency(batch_size, optimizer_data, is_decode=True)
```

- prefill 整波 shape `(query_len=8196, concurrency=global_limit//8196)` **与 batch 无关** → 跨所有 batch 档位命中 `_forward_record_cache`，摊销为 0。
- 余数波 shape 随 batch 变化 → 少量额外 shape。
- **decode shape 每档位 1 个新 key → 1 次新前向 + 1 次重编译**。
- 缓存 `_forward_record_cache` 是 **per strategy 实例 = per 候选**（`base_throughput_optimizer.py:138-144` 注释明确 "each parallel candidate gets an isolated strategy instance"），**候选之间完全不共享**。
- `output_length=1024` 只作为公式乘子（`e2el = ttft + decode_latency × 1024`），**不会**循环跑 1024 次 forward。

### 4.5 前向本身

```634:652:tensor_cast/core/model_runner.py
        with Runtime(self.perf_models, self.device_profile, memory_tracker=MemoryTracker(...)) as runtime, torch.no_grad():
            ...
            primary = _primary_model_output(self.model.forward(**input_kwargs))
        ...
        all_execution_time_s = runtime.total_execution_time_s()
```

`Runtime` 用 `__torch_dispatch__` 拦截每个 op，交给 `AnalyticPerformanceModel`（roofline 解析估算）累加时间 + `MemoryTracker` 累加显存。**没有真实 NPU 计算、没有 warmup/benchmark 循环**（`op_benchmark.py` 的 warmup1/bench10 是离线采集工具，不在本链路）。

---

## 5. 慢在哪：量化估算

| 项 | 数量 |
|---|---|
| 并行候选 | 175 |
| 每候选建模 + 首次编译 | 1 次 |
| 每候选额外重编译/追踪 | ~10~30 次（取 ~11） |
| **全图 Dynamo trace + FX pass 次数** | **175 × ~12 ≈ 2100 次**（量级 10³） |
| 并发度 | 32 |

即：**约两千次对 45 层 / 288 专家 / MLA+线性注意力混合图的完整 trace + FX 改写**，全部是单线程 Python。按每次数十秒量级估算，32 路并行下总墙钟时间落在**数十分钟到小时级**——与"45 分钟没结果"完全吻合。

三个"看起来像优化、实际不存在"的点：

- **`workload_cache` / `WorkloadReuseModelRunner` 不生效**：只在 `len(device_targets) > 1` 时创建（`optimizer_curve_plots.py:692-696`），单 device 时 `workload_cache=None` → `parallel_runner.py:315` 直接走普通 `ModelRunner`。即便生效，key 含 `tp/pp/ep/moe_dp/...`（`workload_cache.py:47-87`），同一硬件内不同并行候选 key 也不同，**仍不会跨候选复用**。
- **spawn worker 不继承任何缓存**：`mp_context = mp.get_context("spawn")`（`parallel_runner.py:700`），每个 worker 还要重新 import torch/transformers，冷启动 + 内存开销按 32 份计。
- **`torch.compiler.reset()` 每候选调用一次**（`parallel_runner.py:788`），Dynamo 缓存在候选间被显式清空。

---

## 6. 为什么"看不到任何结果"

结果汇总发生在**所有候选返回之后**：

```705:721:serving_cast/service/optimizer_curve_plots.py
            results = (
                tasks.run_agg()
                if not run_args.enable_optimize_prefill_decode_ratio and not run_args.disagg
                else tasks.run_disagg()
            )

            for res in results:
                res.report_final_result(run_args, silent=False)
```

`executor.map` 返回的是惰性迭代器，`_get_df_list` 里 `[r for r in results if r is not None]` 会一直阻塞到最后一个候选完成。因此：

- 中途**不会输出任何结果表**；
- 唯一的进度信号是 worker 进程的 `Start processing TP size: N` / `Finish processing TP size: N`（`parallel_runner.py:791 / 852`）以及 `Auto max_batched_tokens candidate`；
- 默认日志级别是 `info`（`cli/registry/shared.py:31`），这些日志应该可见；若被重定向/缓冲就会表现为"完全静默"。

---

## 7. 关于"NPU 有显存占用"

**代码链路结论：本命令的仿真不分配 NPU 显存、不下发 NPU kernel。**

证据：

1. 参数在 `meta` device：`init_on_device_without_buffers("meta") + no_init_weights()`（`transformers/model.py:351`），`load_weights()` 空实现（`:422`）。
2. 权重文件根本不下载：`CONFIG_JSON_ALLOW_PATTERNS` 只要 `config.json`，`MODELSCOPE_WEIGHT_IGNORE_PATTERNS` 显式排除 `*.safetensors/*.bin/*.pt`（`tensor_cast/model_hub.py:28-52`）。
3. KV cache / 输入张量也是 meta：`input_generator.py:1449-1455`，`runtime.py:82-93` 把 CPU 张量也转成 meta。
4. 全仓 `tensor_cast/` 里 `torch.npu` 只出现在注释中（`ops/mla.py:78` 说明是 meta 实现）。`CompilerBackend` 只做 FX 改写，无 codegen。
5. 显存是**算出来的**：`kv_cache_bytes` / `device_memory_available_gb = total - peak - reserved`（`model_runner.py:697-760`），`MemoryTracker` 记账。

所以观察到的 NPU 占用，来源只可能是：

- 机器上其它进程（其它人的仿真/推理/服务）；
- `torch_npu` / CANN 在 worker 进程初始化时创建的 device context（32 个 python 进程各一份，与 msmodeling 逻辑无关）。

现场核验：

```bash
npu-smi info                      # 看整卡占用
npu-smi info -t usages -i 0       # 看进程级占用与 PID
ps -eo pid,ppid,rss,etime,cmd --sort=-rss | grep throughput_optimizer | head -40
# 对比：这些 python 进程的 RSS 是否在涨（说明是吃主机内存，不是 NPU 显存）
```

如果 `npu-smi` 里的 PID 与本次命令的 python 进程**不重合**，那占用的就是别的进程。

---

## 8. 优化建议（按性价比排序）

### 8.1 首轮去掉 `--compile`

`--compile` 只改变 FX 图改写（融合/SP/multistream/常量折叠）后进入性能模型的算子集合，不改变"哪个并行策略更好"的量级结论。**先无 compile 粗筛，再对 Top-N 带 compile 复算**，可省掉绝大部分重编译开销：

```bash
python -m cli.inference.throughput_optimizer model_configs/msmodeling/GLM5.3-Flash \
  --device ATLAS_800_A2_376T_64G --num-devices 128 \
  --tp-sizes 1 2 4 8 16 32 64 --ep-sizes 1 2 4 8 16 32 64 --moe-dp-sizes 1 2 4 8 \
  --input-length 8196 --output-length 1024 \
  --quantize-linear-action DISABLED --quantize-attention-action DISABLED \
  --tpot-limit 100 --ttft-limit 20000 --jobs 32
```

### 8.2 用并行语义收敛搜索空间（最大收益）

175 个候选里有大量物理上无意义的组合。建议**分两阶段**：

阶段 A：固定 MOE-DP=1，只扫 TP × EP（7×7=49，且按 `EP|128`、`EP≥TP` 直觉再筛）；
阶段 B：在阶段 A 的最优 TP 附近，再扫 MOE-DP。

或直接使用业务上可能的部署形态（例如 EP=TP×DP 的 vLLM 语义）来给定组合，而不是笛卡尔积。

### 8.3 显式给 `--batch-range` 和 `--max-batched-tokens`

```bash
--batch-range 1 512 --max-batched-tokens 32784
```

- 跳过 `_exponential_search` 的 10 轮倍增（`base_throughput_optimizer.py:251` 判空才进指数搜索）；
- 跳过自动 `max_batched_tokens` 的 OOM 重试分支。

### 8.4 调整 `--jobs`

- spawn worker 每份都要独立 import torch + 建图 + 存 Dynamo cache，**主机内存峰值 ≈ jobs × 单份开销**。32 并发下若内存吃紧会触发 swap，表现为 CPU 100% 但进度极慢。
- 建议 `jobs = min(候选数, 物理核数 / 2 ~ 物理核数)`，并用 `free -g` / `vmstat 1` 确认没有 swap。

### 8.5 分批提交，换取可见进度

把 TP 拆成多次运行（每次一个或几个 TP），每次几分钟内出一张表，便于中途判断趋势、及时止损。

### 8.6 单点交叉验证用 `text_generate`

`python -m cli.inference.text_generate` 只评估一个给定并行配置，秒级~分钟级，适合先确认单个 TP/EP 配置的时延与显存是否合理，再决定要不要大范围搜索。

---

## 9. 关键常量速查

| 常量 | 值 | 位置 |
|---|---|---|
| `LIMIT_COUNT`（recompile limit） | `1e8` | `serving_cast/service/utils.py:41` |
| `DEFAULT_MAX_SEARCH_COMBINATIONS` | `100`（仅告警） | `serving_cast/service/utils.py:42` |
| `MAX_ITER_NUMS`（指数搜索轮数） | `10` | `serving_cast/service/utils.py:44` |
| `HISTORICAL_DEFAULT_MAX_BATCHED_TOKENS` | `8192` | `serving_cast/service/utils.py:45` |
| `AUTO_MAX_BATCHED_TOKENS_FACTORS` | `(4, 2, 1)` | `serving_cast/service/utils.py:46` |
| `DEFAULT_MAX_BATCH_SIZE` | `512` | `base_throughput_optimizer.py:60` |
| `_DEFAULT_MAX_BATCH_ESTIMATE` | `2**(10+9)-1` | `base_throughput_optimizer.py:382` |
| `config.compilation.enable_freezing` | `True` | `tensor_cast/config.py:31` |
| `enable_sequence_parallel` 默认 | `False` | `user_config.py:114`（`config.py:51`） |

---

## 10. 一句话复盘

**这不是"NPU 在算"，而是"32 个 CPU 进程在做两千多次超大图的 Dynamo 追踪与 FX 改写"**；搜索空间因为 TP×EP×MOE-DP 笛卡尔积被放大到 175 个候选，且候选之间零复用、结果又在最后统一打印，于是表现为"跑了 45 分钟、CPU 打满、看不到任何输出"。
