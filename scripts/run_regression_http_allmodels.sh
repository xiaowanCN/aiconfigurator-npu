#!/usr/bin/env bash
# run_regression_http_allmodels.sh
#
# 全量回归（HTTP 版）：所有支持的模型 x 卡数(32/64/128) x 量化(2) x 设备(2) = 108 用例
#   - 通过 POST /api/v1/tasks 创建异步任务
#   - 轮询 GET  /api/v1/tasks/{id} 直到终态
#   - 最后整理进度与测试情况报告（含路由正确性校验）
#
# 固定场景：
#   输入长度 4096 / 输出长度 1024 / TTFT 20000 / TPOT 100
#   framework=auto / serving_modes=[agg, disagg]
#   max_batched_tokens=8192 / batch_range=[4, 128]
#   量化变体：
#     DISABLE                  -> quantize=DISABLE
#     ENABLE_MXFP4_FP8         -> quantize=ENABLE
#                                 + quantize_linear_action=MXFP4
#                                 + quantize_attention_action=FP8
#
# 预期路由（供报告校验）：
#   aic-npu  <=> (model ∈ {DeepSeek-R1, Qwen3.8-27B, DeepSeek-V4-Pro})
#                 AND quantize=DISABLE AND device=ATLAS_800_A2_376T_64G
#   其余            -> msmodeling
#
# 用法:
#   bash scripts/run_regression_http_allmodels.sh                # 真实跑 108 用例
#   BASE_URL=http://10.1.2.3:18123 bash scripts/run_regression_http_allmodels.sh                # 指向远端服务
#   DRY_RUN=1 bash scripts/run_regression_http_allmodels.sh                                    # 只打印请求体, 不真正下发
#   nohup bash scripts/run_regression_http_allmodels.sh  > itn-regression-http/run_all.log 2>&1 &
#
# 前置: 先启动服务  infer-strategy-npu-server --host 0.0.0.0 --port 18123
# nohup infer-strategy-npu-server --host 0.0.0.0 --port 18123 > infer-strategy.log 2>&1 &
# 依赖: curl 与 python3（容器内均可用）

set -u

# ---- 可调参数(环境变量可覆盖) ----
BASE_URL="${BASE_URL:-http://127.0.0.1:18123}"
OUT_DIR="${OUT_DIR:-./itn-regression-http}"
POLL_INTERVAL="${POLL_INTERVAL:-20}"        # 轮询间隔(秒)
MAX_WAIT="${MAX_WAIT:-7200}"                # 单任务最长等待(秒)
INPUT_LENGTH="${INPUT_LENGTH:-4096}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-1024}"
TTFT_LIMIT="${TTFT_LIMIT:-20000}"
TPOT_LIMIT="${TPOT_LIMIT:-100}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
BATCH_RANGE_MIN="${BATCH_RANGE_MIN:-4}"
BATCH_RANGE_MAX="${BATCH_RANGE_MAX:-128}"
PD_RATIO_SIZES="${PD_RATIO_SIZES:-8 16}"   # 每实例卡数 P/D 枚举空间(对应 --pd-ratio-sizes)
DRY_RUN="${DRY_RUN:-0}"

# 固定参数导出给内嵌 python 读取
export INPUT_LENGTH OUTPUT_LENGTH TTFT_LIMIT TPOT_LIMIT MAX_BATCHED_TOKENS BATCH_RANGE_MIN BATCH_RANGE_MAX PD_RATIO_SIZES

LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"
CASES_TSV="$OUT_DIR/cases.tsv"
REPORT_MD="$OUT_DIR/report.md"
: > "$CASES_TSV"

# ---- 矩阵维度 ----
# 所有支持的模型(与 src/infer_strategy_npu/models.py MODELS 保持一致)
MODELS=(
  "ZhipuAI/GLM-5.3"
  "deepseek-ai/DeepSeek-V4.1-Flash"
  "deepseek-ai/DeepSeek-R1"
  "Qwen/Qwen3.8-27B"
  "deepseek-ai/DeepSeek-V4-Pro"
  "ZhipuAI/GLM-5.3-Flash"
  "ZhipuAI/GLM-5.2"
  "moonshotai/Kimi-K3"
  "MiniMax/MiniMax-M3"
)
DEVICES=("ATLAS_800_A2_376T_64G" "ATLAS_950_486T_96G")
NUM_DEVICES=(32 64 128)

# 量化变体: 标签|quantize|quantize_linear_action|quantize_attention_action
QUANT_VARIANTS=(
  "DISABLE|DISABLE|||"
  "ENABLE_MXFP4_FP8|ENABLE|MXFP4|FP8"
)

# 设备短名(仅用于日志/用例 id 可读性)
declare -A DEV_SHORT=(
  ["ATLAS_800_A2_376T_64G"]="A800"
  ["ATLAS_950_486T_96G"]="A950"
)

# aic-npu 白名单(供报告校验; 与 routing.AIC_NPU_MODEL_IDS 保持一致)
AIC_MODELS=("deepseek-ai/DeepSeek-R1" "Qwen/Qwen3.8-27B" "deepseek-ai/DeepSeek-V4-Pro")
AIC_DEVICE="ATLAS_800_A2_376T_64G"

# ---- 统计 ----
total=0; n_success=0; n_failed=0; n_timeout=0; n_cancelled=0; n_error=0; n_skip=0

# ----------------------------------------------------------------- 工具函数
sanitize() { echo "$1" | tr '/.' '__' | tr -d ' '; }

# 构造请求体(JSON) -> 输出到 stdout
build_body() {
  # $1 model $2 device $3 num_devices $4 quantize $5 linear_action $6 attention_action
  python3 - "$1" "$2" "$3" "$4" "$5" "$6" <<'PY'
import sys, json, os
model, device, nd, q, lin, att = sys.argv[1:7]
def e(k):
    v = os.environ.get(k)
    return int(v) if v not in (None, "") else None
body = {
    "model": model,
    "device": device,
    "num_devices": int(nd),
    "input_length": e("INPUT_LENGTH"),
    "output_length": e("OUTPUT_LENGTH"),
    "ttft_limit": e("TTFT_LIMIT"),
    "tpot_limit": e("TPOT_LIMIT"),
    "quantize": q,
    "framework": "auto",
    "serving_modes": ["agg", "disagg"],
    "max_batched_tokens": e("MAX_BATCHED_TOKENS"),
    "batch_range": [e("BATCH_RANGE_MIN"), e("BATCH_RANGE_MAX")],
    "pd_ratio_sizes": [int(x) for x in os.environ.get("PD_RATIO_SIZES", "").split() if x.strip()],
}
if q == "ENABLE":
    if lin:
        body["quantize_linear_action"] = lin
    if att:
        body["quantize_attention_action"] = att
print(json.dumps(body, ensure_ascii=False))
PY
}

# 解析创建响应: 成功打印 "OK\t<task_id>", 失败打印 "ERR\t<detail>"
parse_create() {
  python3 - "$1" <<'PY'
import sys, json
try:
    d = json.load(open(sys.argv[1]))
except Exception as exc:
    print("ERR\tunparseable response: %s" % exc)
    sys.exit()
code = d.get("code")
data = d.get("data")
if isinstance(data, dict) and data.get("task_id"):
    print("OK\t" + str(data["task_id"]))
else:
    print("ERR\tHTTP %s %s" % (code, json.dumps(data, ensure_ascii=False)))
PY
}

# 解析轮询响应: 打印 "status\tpercent\tstage\tframework\terror"
parse_poll() {
  python3 - "$1" <<'PY'
import sys, json
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("ERROR\t-\t-\t-\tunparseable response")
    sys.exit()
data = d.get("data") or {}
status = data.get("status") or "UNKNOWN"
prog = data.get("progress") or {}
percent = prog.get("percent")
stage = prog.get("stage") or ""
res = data.get("result") or {}
routing = (res.get("routing") or {}) if isinstance(res, dict) else {}
framework = routing.get("framework") or ""
err = data.get("error") or ""
if percent is None:
    percent = "-"
print("%s\t%s\t%s\t%s\t%s" % (status, percent, stage, framework, err))
PY
}

is_terminal() {
  case "$1" in
    SUCCESS|FAILED|TIMEOUT|CANCELLED|ERROR) return 0 ;;
    *) return 1 ;;
  esac
}

# ----------------------------------------------------------------- 单用例
run_case() {
  local model="$1" device="$2" nd="$3" qlabel="$4" quantize="$5" lin="$6" att="$7"
  local dev_short="${DEV_SHORT[$device]:-DEV}"
  local id="$(sanitize "$model")-${dev_short}-${nd}-${qlabel}"
  local body_file="$LOG_DIR/${id}.req.json"
  local create_file="$LOG_DIR/${id}.create.json"
  local poll_file="$LOG_DIR/${id}.poll.json"
  local task_id="-" status="-" framework="" elapsed=0 err=""

  build_body "$model" "$device" "$nd" "$quantize" "$lin" "$att" > "$body_file"

  if [ "$DRY_RUN" = "1" ]; then
    echo "  [DRY] $id  -> $(cat "$body_file")"
    status="SKIPPED"; n_skip=$((n_skip+1))
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$id" "$model" "$device" "$nd" "$qlabel" "$quantize" "$task_id" "$status" "$framework" "$elapsed" "$err" >> "$CASES_TSV"
    total=$((total+1))
    return
  fi

  local t0 t1
  t0=$(date +%s)

  # --- 创建任务 ---
  local http_code
  http_code=$(curl -s -o "$create_file" -w '%{http_code}' -X POST \
    "$BASE_URL/api/v1/tasks" \
    -H 'Content-Type: application/json' \
    --data-binary @"$body_file")
  local parsed
  parsed=$(parse_create "$create_file")
  local ptag pval
  IFS=$'\t' read -r ptag pval <<< "$parsed"
  if [ "$ptag" != "OK" ]; then
    status="ERROR"; err="$pval (http=$http_code)"
    t1=$(date +%s); elapsed=$((t1-t0))
    n_error=$((n_error+1))
    printf '  [ERROR] %-48s create failed: %s\n' "$id" "$err"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$id" "$model" "$device" "$nd" "$qlabel" "$quantize" "$task_id" "$status" "$framework" "$elapsed" "$err" >> "$CASES_TSV"
    total=$((total+1))
    return
  fi
  task_id="$pval"
  printf '  [START] %-48s task=%s\n' "$id" "$task_id"

  # --- 轮询直到终态 ---
  local waited=0
  while true; do
    curl -s "$BASE_URL/api/v1/tasks/$task_id" -o "$poll_file"
    local pstatus ppercent pstage pframework perr
    IFS=$'\t' read -r pstatus ppercent pstage pframework perr <<< "$(parse_poll "$poll_file")"
    status="$pstatus"; framework="$pframework"; err="$perr"
    if is_terminal "$status"; then
      break
    fi
    sleep "$POLL_INTERVAL"
    waited=$((waited+POLL_INTERVAL))
    if [ "$waited" -ge "$MAX_WAIT" ]; then
      status="TIMEOUT"; err="client-side wait exceeded ${MAX_WAIT}s"
      break
    fi
  done

  t1=$(date +%s); elapsed=$((t1-t0))
  case "$status" in
    SUCCESS)  n_success=$((n_success+1))  ;;
    FAILED)   n_failed=$((n_failed+1))    ;;
    TIMEOUT)  n_timeout=$((n_timeout+1))   ;;
    CANCELLED)n_cancelled=$((n_cancelled+1));;
    ERROR)    n_error=$((n_error+1))       ;;
  esac

  printf '  [%-7s] %-48s (%4ds) fw=%-9s task=%s\n' "$status" "$id" "$elapsed" "${framework:-?}" "$task_id"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$id" "$model" "$device" "$nd" "$qlabel" "$quantize" "$task_id" "$status" "$framework" "$elapsed" "$err" >> "$CASES_TSV"
  total=$((total+1))
}

# ----------------------------------------------------------------- 健康检查
echo "============= infer-strategy-npu 全量回归 (HTTP) ============="
echo " base_url=$BASE_URL  out_dir=$OUT_DIR  poll_interval=${POLL_INTERVAL}s  max_wait=${MAX_WAIT}s"
echo " input_length=$INPUT_LENGTH  output_length=$OUTPUT_LENGTH  ttft=$TTFT_LIMIT  tpot=$TPOT_LIMIT"
echo " max_batched_tokens=$MAX_BATCHED_TOKENS  batch_range=[$BATCH_RANGE_MIN $BATCH_RANGE_MAX]  dry_run=$DRY_RUN"
echo " models=${#MODELS[@]}  devices=${#DEVICES[@]}  num_devices=${NUM_DEVICES[*]}  quant=${#QUANT_VARIANTS[@]}"
echo " 预计用例数 = ${#MODELS[@]} x ${#DEVICES[@]} x ${#NUM_DEVICES[@]} x ${#QUANT_VARIANTS[@]} = $(( ${#MODELS[@]} * ${#DEVICES[@]} * ${#NUM_DEVICES[@]} * ${#QUANT_VARIANTS[@]} ))"
echo "============================================================"

if [ "$DRY_RUN" != "1" ]; then
  health_file="$LOG_DIR/health.json"
  hc=$(curl -s -o "$health_file" -w '%{http_code}' "$BASE_URL/api/v1/health")
  if [ "$hc" != "200" ]; then
    echo "!! 服务不可达 (HTTP $hc)。请先启动: infer-strategy-npu-server --host 0.0.0.0 --port 18123"
    exit 2
  fi
  echo "  健康检查: HTTP $hc -> $(python3 -c 'import sys,json;d=json.load(open(sys.argv[1]));print(d.get("data",{}).get("version","?"))' "$health_file")"
fi

# ----------------------------------------------------------------- 主循环
for model in "${MODELS[@]}"; do
  for device in "${DEVICES[@]}"; do
    for nd in "${NUM_DEVICES[@]}"; do
      for qv in "${QUANT_VARIANTS[@]}"; do
        IFS='|' read -r qlabel quantize lin att <<< "$qv"
        run_case "$model" "$device" "$nd" "$qlabel" "$quantize" "$lin" "$att"
      done
    done
  done
done

# ----------------------------------------------------------------- 报告
python3 - "$CASES_TSV" "$REPORT_MD" "${BASE_URL}" <<'PY'
import sys, os
tsv, report_md, base_url = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
with open(tsv, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        # id model device nd qlabel quantize task_id status framework elapsed error
        while len(parts) < 11:
            parts.append("")
        rows.append(parts)

if not rows:
    print("无用例记录。")
    sys.exit(0)

ALLOW = {"deepseek-ai/DeepSeek-R1", "Qwen/Qwen3.8-27B", "deepseek-ai/DeepSeek-V4-Pro"}
AIC_DEV = "ATLAS_800_A2_376T_64G"

def expected(model, device, quantize):
    if model in ALLOW and quantize == "DISABLE" and device == AIC_DEV:
        return "aic-npu"
    return "msmodeling"

status_count = {}
fw_count = {}
mismatch = []
failures = []
for r in rows:
    _id, model, device, nd, qlabel, quantize, task_id, status, framework, elapsed, err = r
    status_count[status] = status_count.get(status, 0) + 1
    fw = framework or "-"
    fw_count[fw] = fw_count.get(fw, 0) + 1
    # 路由校验(仅对真正执行且有框架结论的用例)
    if status in ("SUCCESS", "FAILED") and framework:
        exp = expected(model, device, quantize)
        if exp != framework:
            mismatch.append((_id, model, device, nd, qlabel, exp, framework))
    if status in ("FAILED", "TIMEOUT", "ERROR", "CANCELLED"):
        failures.append((_id, status, task_id, err))

total = len(rows)
lines = []
lines.append("# 全量回归测试报告 (HTTP)")
lines.append("")
lines.append(f"- 服务: `{base_url}`")
lines.append(f"- 用例总数: **{total}**")
lines.append("")
lines.append("## 1. 状态汇总")
lines.append("")
lines.append("| 状态 | 数量 |")
lines.append("|------|------|")
for s in ["SUCCESS", "FAILED", "TIMEOUT", "CANCELLED", "ERROR", "SKIPPED"]:
    if status_count.get(s):
        lines.append(f"| {s} | {status_count[s]} |")
lines.append(f"| **合计** | **{total}** |")
lines.append("")
lines.append("## 2. 实际路由分布 (framework)")
lines.append("")
lines.append("| 框架 | 用例数 |")
lines.append("|------|--------|")
for fw, c in sorted(fw_count.items(), key=lambda kv: -kv[1]):
    lines.append(f"| {fw} | {c} |")
lines.append("")
lines.append("## 3. 路由正确性校验 (auto 路由)")
lines.append("")
if mismatch:
    lines.append(f"⚠️ 发现 **{len(mismatch)}** 个路由与预期不符的用例：")
    lines.append("")
    lines.append("| 用例 | 模型 | 设备 | 卡数 | 量化 | 预期 | 实际 |")
    lines.append("|------|------|------|------|------|------|------|")
    for m in mismatch:
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % m)
else:
    lines.append("✅ 所有已得出结论的用例路由均与预期一致（aic-npu 仅限 DeepSeek-R1 / Qwen3.8-27B / DeepSeek-V4-Pro + DISABLE + ATLAS_800_A2_376T_64G，其余 msmodeling）。")
lines.append("")
lines.append("## 4. 失败 / 异常用例")
lines.append("")
if failures:
    lines.append(f"共 **{len(failures)}** 个非成功用例：")
    lines.append("")
    lines.append("| 用例 | 状态 | task_id | 错误/说明 |")
    lines.append("|------|------|---------|-----------|")
    for _id, st, tid, err in failures:
        e = (err or "").replace("|", "\\|")[:200]
        lines.append(f"| {_id} | {st} | {tid} | {e} |")
else:
    lines.append("✅ 无失败 / 异常用例。")
lines.append("")
lines.append("## 5. 全部用例明细")
lines.append("")
lines.append("| 用例 | 模型 | 设备 | 卡数 | 量化 | task_id | 状态 | 框架 | 耗时(s) |")
lines.append("|------|------|------|------|------|---------|------|------|---------|")
for r in rows:
    _id, model, device, nd, qlabel, quantize, task_id, status, framework, elapsed, err = r
    lines.append(f"| {_id} | {model} | {device} | {nd} | {qlabel} | {task_id} | {status} | {framework or '-'} | {elapsed} |")
lines.append("")

report_text = "\n".join(lines)
with open(report_md, "w", encoding="utf-8") as f:
    f.write(report_text + "\n")

# 同时打印到 stdout
print(report_text)
print("")
print(f"报告已写入: {report_md}")
print(f"用例明细 TSV: {tsv}")
PY

# ----------------------------------------------------------------- 退出码
if [ "$n_failed" -gt 0 ] || [ "$n_error" -gt 0 ] || [ "$n_timeout" -gt 0 ]; then
  exit 1
fi
exit 0
