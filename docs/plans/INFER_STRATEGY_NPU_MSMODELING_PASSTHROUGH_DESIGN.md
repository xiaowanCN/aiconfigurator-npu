# msmodeling 参数透传设计（含 batch-range / max-batched-tokens 默认下发）

> 状态：**设计稿（待评审）**。评审通过后再改代码。
> 关联：`src/infer_strategy_npu/{schema,routing,cli,api/schemas}.py`、`runners/msmodeling.py`、`ui/`。
> 参数权威来源：`GITCODEProject/msmodeling/docs/zh/user_guide/msmodeling_throughput_optimizer_user_guide.md` §4。

---

## 1. 问题

1. `--batch-range`（默认 `4 256`）和 `--max-batched-tokens`（默认 `4096`）在 `infer-strategy-npu` CLI 与前端表单里**都没有位置**，无法下发到 msmodeling。
2. 更普遍的问题：每新增一个 msmodeling 参数（`--pp-sizes`、`--ep-sizes`、`--prefix-cache-hit-rate`、`--target-qps`、`--enable-decode-kv-offload`、`--performance-model` …）都要改 4~5 处（CLI parser → `RecommendRequest` → `ms_extra_flags`/`build_msmodeling_command` → `TaskCreateRequest` → 前端表单），后续 msmodeling 升级时兼容成本高。

目标：
- 给 `--batch-range`、`--max-batched-tokens` 留结构化位置并默认下发；
- 提供**通用透传通道**，让任意未在 wrapper 中建模的 msmodeling 参数无需改代码即可传入（CLI 级 / 请求级）。

---

## 2. 现状接线（改动前必读）

参数落地的链路：

```
RecommendRequest (schema.py, dataclass)
   ├─ CLI:  cli.py build_parser() + main() 构造 RecommendRequest
   └─ HTTP: TaskCreateRequest (api/schemas.py) → to_request_dict() → RecommendRequest
        ↓
routing.build_msmodeling_command()  # dry-run 字符串（/tasks/preview 用）
routing.ms_extra_flags(request, mode) # 真实 runner 拼 argv 也走它
        ↓
runners/msmodeling.py：argv = [model, --device, --num-devices, ...] + ms_extra_flags() + slo_flags()
        ↓
cli.inference.throughput_optimizer.arg_parse() 解析执行
```

现状已建模的 msmodeling 透传字段（共 11 个）：
`quantize_linear_action` / `quantize_non_expert_linear_action` / `quantize_attention_action` /
`speculative_method` / `num_speculative_tokens` / `acceptance_length` / `num_draft_layers` /
`tp_sizes` / `jobs` / `max_search_combinations` / `compile_model`。

**没有** `batch_range` / `max_batched_tokens`，也没有通用逃生舱。

---

## 3. 方案

### 3.1 两个具体字段（typed / 校验 / 文档化 / UI 可见）

`RecommendRequest`（`schema.py`）新增：

```python
batch_range: list[int] = field(default_factory=lambda: [4, 256])
"""msmodeling --batch-range，1~2 个正整数：[min max] 或 [max]；默认 [4, 256]。"""

max_batched_tokens: int = 4096
"""msmodeling --max-batched-tokens；默认 4096（用户指定）。"""
```

注入点（两处保持一致，保证 dry-run 与真实命令等价）：

- `routing.ms_extra_flags()` 末尾追加：
  ```python
  if request.batch_range:
      extra += ["--batch-range", *request.batch_range]
  extra += ["--max-batched-tokens", request.max_batched_tokens]
  ```
- `routing.build_msmodeling_command()` 同步追加（已复用 `ms_extra_flags`，实际只需在前者加即可）。

HTTP `TaskCreateRequest`（`api/schemas.py`）新增同名字段，配正整数校验（沿用现有 `positive_int` 风格）：
```python
batch_range: list[int] = Field(default=[4, 256], description="msmodeling --batch-range：[min max] 或 [max]")
max_batched_tokens: int = Field(4096, gt=0, description="msmodeling --max-batched-tokens")
```
`to_request_dict()` 用 `exclude_none=True`，默认 4096 非 None 会恒传 → 符合「默认下发」诉求。

CLI `build_parser()` 在 `msmodeling passthrough` 组追加：
```python
ms.add_argument("--batch-range", type=positive_int, nargs="+", default=[4, 256], help="msmodeling --batch-range：批大小搜索范围 [min max] / [max]")
ms.add_argument("--max-batched-tokens", type=positive_int, default=4096, help="msmodeling --max-batched-tokens：单 DP 副本 Prefill/混合步的最大 batched token 预算")
```
`main()` 构造 `RecommendRequest` 时补 `batch_range=args.batch_range, max_batched_tokens=args.max_batched_tokens`。

前端 `ui/index.html` msmodeling 透传区新增两个输入，并在 `ui/app.js` 提交 payload 与 `/tasks/preview` 展示里带上。

### 3.2 通用透传通道 `extra_msmodeling_args`（逃生舱）

提供**不受 wrapper 建模约束**的通道，原样追加到 msmodeling argv（仅 agg/pd_ratio 命令，**不影响 aic-npu**）。

类型 `list[str]`（每个元素是一个 argv token；token 内部可含空格，不被拆分）：

- 例：要传 `--pp-layer-partitions '[[31,30],[16,15,15,15]]'`，调用方传
  `["--pp-layer-partitions", "[[31,30],[16,15,15,15]]"]`（两元素，第二个含空格但作为一个 token）。
- 为什么用 `list[str]` 而非 `str`：避免 shell 风格 split 破坏带空格的值（如 JSON 分区串）。

CLI 入口（`nargs="*"` 收成 token 列表，前导 `--` 也一并保留）：
```python
ms.add_argument("--extra-msmodeling-args", nargs="*", default=[],
                help="原样透传给 msmodeling 的额外参数 token 列表（逃生舱，优先级最高，可覆盖其它参数）")
```
HTTP：
```python
extra_msmodeling_args: list[str] | None = Field(None, description="原样透传给 msmodeling 的额外 argv token（逃生舱）")
```

注入点：`routing.ms_extra_flags()` **最末尾**附加，使透传参数可覆盖前面的结构化 flag（argparse 同名参数 last-wins）：
```python
extra += sanitize_extra_args(request.extra_msmodeling_args)
```

最小校验 `sanitize_extra_args(tokens)`（仅防注入，不做语义校验）：
- 丢弃空串；
- 拒绝含 `\n` / `\r` / `\x00` 的 token（JSON 分区串里的换行除外，正常不会带）；
- 非法 → 抛 `ValueError`（CLI/HTTP 层转成 400）。

文档定位：明确标注「power-user 逃生舱，优先级最高，可能覆盖其它参数；出错由 msmodeling 自行报错」。

### 3.3 argv 顺序与优先级

拼装顺序（最后出现的同名参数生效）：
```
[model, --device, --num-devices, --input-length, --output-length,
 --compile?,
 (pd-ratio: --enable-optimize-prefill-decode-ratio --prefill-devices-per-instance N --decode-devices-per-instance N),
 quantize*/spec*/tp_sizes/jobs/max_search_combinations,        # ms_extra_flags 前半
 --batch-range 4 256 --max-batched-tokens 4096,               # 3.1 结构化
 slo_flags (--ttft-limit/--tpot-limit),
 *extra_msmodeling_args]                                       # 3.2 逃生舱（最后，可覆盖）
```
- 结构化字段恒传；`extra_msmodeling_args` 在后 → 同名覆盖。
- 若透传里再写 `--max-batched-tokens 8192`，则以 8192 为准（文档注明）。

### 3.4 不改动的范围

- **aic-npu 路径完全不受影响**：透传只进 `ms_extra_flags()`，aic-npu 走 `build_aic_npu_command()`。
- `RecommendRequest.to_dict()` 用 `asdict`，新字段自动带上，无需改序列化。
- `slo_flags`、`pd_ratio_grid` 等不变。

---

## 4. 实现阶段影响的文件

| 文件 | 改动 |
|---|---|
| `src/infer_strategy_npu/schema.py` | `RecommendRequest` +`batch_range` / `max_batched_tokens` / `extra_msmodeling_args` |
| `src/infer_strategy_npu/routing.py` | `ms_extra_flags()` 追加结构化两字段 + 透传；`build_msmodeling_command()` 同步 |
| `src/infer_strategy_npu/api/schemas.py` | `TaskCreateRequest` +3 字段 + 校验；`to_request_dict()` 已 `exclude_none` |
| `src/infer_strategy_npu/cli.py` | parser 加 2 结构化 + 1 透传；`main()` 构造 `RecommendRequest` 补字段 |
| `src/infer_strategy_npu/runners/msmodeling.py` | 已用 `ms_extra_flags` 自动继承；加 `sanitize_extra_args` 兜底 |
| `ui/index.html` + `ui/app.js` | 表单新增字段 + 预览/提交 payload |
| `docs/plans/infer-strategy-npu-http-api.md` | HTTP 字段表 + 透传说明 |

---

## 5. 验证

- CLI：`infer-strategy-npu -recommend ... --dry-run` 打印的命令应含 `--batch-range 4 256 --max-batched-tokens 4096`。
- 透传：`--extra-msmodeling-args --tp-sizes 8 --prefix-cache-hit-rate 0.5` → dry-run 命令含对应 token。
- 非法透传：含 `\n` 的 token → 启动/提交即 400。
- 优先级：透传带 `--max-batched-tokens 8192` → 实际命令尾部为 `... --max-batched-tokens 8192`。
- HTTP：`POST /tasks` 带 `batch_range` / `max_batched_tokens` / `extra_msmodeling_args`；`POST /tasks/preview` 的 `commands` 体现；`GET /tasks/{id}/result` 的 `request` 回显。

---

## 6. 待评审决策点（Decisions）

1. **`max_batched_tokens` 默认策略**：
   - 方案 A（按你原话）：默认 `4096`，**恒传**，覆盖 msmodeling 的自动 `4*input_length`（OOM 降级）。
   - 方案 B（更稳）：默认 `None`（不传，交给 msmodeling 自适应），仅当用户显式设置才下发。
   - ⚠️ 权衡：isl 较大时（如 4096 → 4*isl=16384），恒传 4096 会偏小，可能抬高 TTFT。建议默认 `None`（方案 B），由调用方按需设 4096；但若你就是要 4096 当全局默认，选 A。

2. **`batch_range` 默认策略**：同理，默认 `[4,256]` 恒传（A）还是 `None` 时由 msmodeling 自适应（B）。

3. **通用透传载体**：`list[str]`（推荐，避免空格拆分问题） vs `str`（shell 风格 split）。

4. **CLI 透传入口**：`--extra-msmodeling-args` 用 `nargs="*"`（推荐，简单）还是 `nargs=argparse.REMAINDER`（吞掉其后所有 token，但位置必须最后）。

> 默认建议：**结构化两字段采用「None=自适应、显式才下发」（Decision 1/2 选 B），通用透传用 `list[str]` + `nargs="*"`**（Decision 3/4 选推荐项）——既满足你"给这两个参数留位置 + 默认 4096/4 256 可传"的诉求（显式设即可），又避免恒传覆盖 msmodeling 的自适应导致长输入场景 TTFT 变差。如果你坚持恒传 4096，我按 A 实现。
