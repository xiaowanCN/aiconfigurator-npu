#!/usr/bin/env bash
# test_infer_strategy_task_enter_container.sh
# 在 Linux 容器里对 infer-strategy-npu-task-enter.py 做黑盒回归：
# 逐个执行形如
#   python infer-strategy-npu-task-enter.py --model ... --device ... --num_devices ... \
#     --input-length ... --output-length ... --tpot-limit ... --ttft-limit ... \
#     --serving_modes 1 --log-file .../02.log --result-file .../02.json
# 的真实命令，校验契约 JSON / 日志 / 退出码，最后生成测试报告。
#
# 用法（容器内 CWD=/workspace，或仓库根目录）：
#   bash scripts/test_infer_strategy_task_enter_container.sh --mode quick
#   bash scripts/test_infer_strategy_task_enter_container.sh --mode smoke
#   bash scripts/test_infer_strategy_task_enter_container.sh --mode all \
#        --out-dir /apps/suyantest/outputs/infer/infer_strategy_npu
#   nohup bash scripts/test_infer_strategy_task_enter_container.sh --mode all \
#        > /apps/suyantest/outputs/infer/infer_strategy_npu/run_all.log 2>&1 &
#
# 模式：
#   quick     只跑瞬时失败用例（不需要 msmodeling，建议先跑这个验证环境）
#   invalid   同 quick
#   smoke     invalid + 三种 serving_modes 基础寻优
#   defaults  invalid + 默认值 / 别名
#   matrix    invalid + 模型 / 设备 / PD 卡数矩阵
#   boundary  invalid + 值域边界与极端 SLO
#   contract  invalid + 运行期落盘契约
#   all|full  全部（默认）
#
# 其他选项：
#   --only SUBSTR     只跑 ID 含 SUBSTR 的用例，便于单用例调试
#   --timeout SEC     单个真实寻优用例超时（默认 600）
#   --quick-timeout SEC  单个瞬时用例超时（默认 60）
#   --python BIN      Python 解释器（默认 python）
#   --entry PATH      入口脚本路径
#   --out-dir DIR     产物目录（路径中不要含空格）
#   --help            打印本说明
#
# 环境变量覆盖：OUT_DIR / PYTHON_BIN / ENTRY / MODE / ONLY / TIMEOUT / QUICK_TIMEOUT
#
# 产物（OUT_DIR 下）：
#   report.md        测试报告（汇总、用例明细、失败明细、覆盖说明）
#   summary.json     机器可读汇总
#   failures.txt     失败清单
#   results.jsonl    逐用例原始记录
#   cases/<id>.*     单用例的 json / log / stdout / stderr
#
# 退出码：0=全部通过（SKIP 不计失败），1=有用例失败，2=用法/环境错误。

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENTRY="${ENTRY:-$ROOT/infer-strategy-npu-task-enter.py}"
HELPER="${HELPER:-$SCRIPT_DIR/test_infer_strategy_task_enter_checks.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-/apps/suyantest/outputs/infer/infer_strategy_npu}"
MODE="${MODE:-all}"
ONLY="${ONLY:-}"
TIMEOUT="${TIMEOUT:-600}"
QUICK_TIMEOUT="${QUICK_TIMEOUT:-60}"

while [ $# -gt 0 ]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --quick-timeout) QUICK_TIMEOUT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --entry) ENTRY="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --help|-h)
      awk 'NR == 1 { next } /^#/ { print; next } { exit }' "$0"
      exit 0 ;;
    *) echo "未知参数：$1（用 --help 查看用法）" >&2; exit 2 ;;
  esac
done

case "$MODE" in
  quick|invalid|smoke|defaults|matrix|boundary|contract|all|full) ;;
  *) echo "未知模式：$MODE" >&2; exit 2 ;;
esac

for required in "$ENTRY" "$HELPER"; do
  if [ ! -f "$required" ]; then
    echo "缺少文件：$required" >&2
    exit 2
  fi
done
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "找不到 Python 解释器：$PYTHON_BIN" >&2
  exit 2
fi

HAS_TIMEOUT=0
command -v timeout >/dev/null 2>&1 && HAS_TIMEOUT=1

# 沿用统一镜像约定：避免无 NPU 驱动时 torch 启动即探测设备
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
# 保证入口脚本与校验器在任意 locale 下都输出 UTF-8（results.jsonl / 报告用 UTF-8 读取）
export PYTHONIOENCODING=utf-8

CASE_DIR="$OUT_DIR/cases"
RESULTS="$OUT_DIR/results.jsonl"
mkdir -p "$CASE_DIR"
: > "$RESULTS"
mkdir -p "$CASE_DIR/log_dir_target"   # 给「--log-file 指向目录」用例用
STARTED_AT="$(date '+%Y-%m-%dT%H:%M:%S%z')"

# --------------------------------------------------------------------------- 基准参数
# 与用户示例一致；重复传入同一 option 时 argparse 取最后一次，故用例只需追加覆盖项。
BASE="--model DeepSeek-V4.1-Flash-MXFP4 --device 950DT --num_devices 32 --input-length 4096 --output-length 1024 --tpot-limit 100 --ttft-limit 20 --serving_modes 1"
COMMON="--model DeepSeek-V4.1-Flash-MXFP4 --device 950DT --num_devices 32 --input-length 4096 --output-length 1024"
BASE_NO_TTFT="$COMMON --tpot-limit 100 --serving_modes 0"
BASE_NO_TPOT="$COMMON --ttft-limit 20 --serving_modes 0"
BASE_NO_SLO="$COMMON --serving_modes 0"
BASE_NO_MODES="$COMMON --tpot-limit 100 --ttft-limit 20"
BASE_NO_MODEL="--device 950DT --num_devices 32 --input-length 4096 --output-length 1024 --tpot-limit 100 --ttft-limit 20 --serving_modes 0"
BASE_NO_DEVICE="--model DeepSeek-V4.1-Flash-MXFP4 --num_devices 32 --input-length 4096 --output-length 1024 --tpot-limit 100 --ttft-limit 20 --serving_modes 0"
ALIAS_UNDERSCORE="--model DeepSeek-V4.1-Flash-MXFP4 --device 950DT --num_devices 32 --input_length 4096 --output_length 1024 --tpot_limit 100 --ttft_limit 20 --serving_modes 0"
ALIAS_HYPHEN="--model DeepSeek-V4.1-Flash-MXFP4 --device 950DT --num-devices 32 --input-length 4096 --output-length 1024 --tpot-limit 100 --ttft-limit 20 --serving-modes 0"

# --------------------------------------------------------------------------- 调度
should_run() {
  local category="$1"
  case "$MODE" in
    all|full) return 0 ;;
    quick|invalid) [ "$category" = "invalid" ] && return 0; return 1 ;;
    smoke) case "$category" in invalid|smoke) return 0 ;; *) return 1 ;; esac ;;
    defaults) case "$category" in invalid|defaults|alias) return 0 ;; *) return 1 ;; esac ;;
    matrix) case "$category" in invalid|matrix) return 0 ;; *) return 1 ;; esac ;;
    boundary) case "$category" in invalid|boundary) return 0 ;; *) return 1 ;; esac ;;
    contract) case "$category" in invalid|contract) return 0 ;; *) return 1 ;; esac ;;
  esac
}

emit_skip() {
  local id="$1" name="$2" category="$3" reason="$4"
  "$PYTHON_BIN" "$HELPER" skip --id "$id" --name "$name" \
    --category "$category" --group "$category" --reason "$reason" >> "$RESULTS"
  printf '  [SKIP] %-34s %s\n' "$id" "$name"
}

emit_check() {
  # 把 helper 输出追加到 results.jsonl；helper 自身异常时给出可见告警
  if ! "$PYTHON_BIN" "$HELPER" check "$@" >> "$RESULTS"; then
    echo "  [WARN] 校验器执行失败，请检查 $RESULTS 是否缺少记录" >&2
    return
  fi
  if tail -n 1 "$RESULTS" | grep -q '"status": "PASS"'; then
    printf '  [PASS] %-34s %s\n' "$2" "$4"
  else
    printf '  [FAIL] %-34s %s\n' "$2" "$4"
    tail -n 1 "$RESULTS" | "$PYTHON_BIN" -c \
      'import json,sys; print("    " + " ; ".join(json.loads(sys.stdin.read()).get("errors", [])))' \
      || true
  fi
}
# emit_check 的参数顺序固定为 --id ID --name NAME ...，故 $2/$4 即 ID / NAME

run_case() {
  local id="" name="" category="invalid" argv="" expect_global="0" expect_inner="0,3"
  local expect_exit="0" expect_result="auto" expect_message="" expect_request=""
  local expect_running=0 expect_validate=0 expect_no_result=0 check_contract=1
  local case_timeout="$TIMEOUT" result_suffix=".json" no_log=0 no_result=0
  local log_path="" result_path="" notes="" runtime_observed=""

  while [ $# -gt 0 ]; do
    case "$1" in
      --id) id="$2"; shift 2 ;;
      --name) name="$2"; shift 2 ;;
      --category) category="$2"; shift 2 ;;
      --argv) argv="$2"; shift 2 ;;
      --expect-global) expect_global="$2"; shift 2 ;;
      --expect-inner) expect_inner="$2"; shift 2 ;;
      --expect-exit) expect_exit="$2"; shift 2 ;;
      --expect-result) expect_result="$2"; shift 2 ;;
      --expect-message) expect_message="$2"; shift 2 ;;
      --expect-request) expect_request="$2"; shift 2 ;;
      --expect-running) expect_running="$2"; shift 2 ;;
      --expect-validate) expect_validate="$2"; shift 2 ;;
      --expect-no-result) expect_no_result="$2"; shift 2 ;;
      --check-contract) check_contract="$2"; shift 2 ;;
      --timeout) case_timeout="$2"; shift 2 ;;
      --result-suffix) result_suffix="$2"; shift 2 ;;
      --no-log) no_log=1; shift ;;
      --no-result) no_result=1; shift ;;
      --log-path) log_path="$2"; shift 2 ;;
      --result-path) result_path="$2"; shift 2 ;;
      --notes) notes="$2"; shift 2 ;;
      --runtime-observed) runtime_observed="$2"; shift 2 ;;
      *) echo "run_case 未知参数：$1" >&2; return 1 ;;
    esac
  done

  if [ -n "$ONLY" ] && [[ "$id" != *"$ONLY"* ]]; then
    emit_skip "$id" "$name" "$category" "被 --only=$ONLY 过滤"
    return 0
  fi
  if ! should_run "$category"; then
    emit_skip "$id" "$name" "$category" "被 --mode=$MODE 过滤"
    return 0
  fi

  [ -z "$log_path" ] && log_path="$CASE_DIR/$id.log"
  [ -z "$result_path" ] && result_path="$CASE_DIR/$id$result_suffix"
  local stdout_f="$CASE_DIR/$id.stdout" stderr_f="$CASE_DIR/$id.stderr"
  if [ -n "$log_path" ] && [ -f "$log_path" ]; then rm -f "$log_path"; fi
  rm -f "$result_path" "$stdout_f" "$stderr_f"

  local -a cmd=("$PYTHON_BIN" "$ENTRY")
  local -a parts=()
  read -r -a parts <<< "$argv"
  cmd+=("${parts[@]}")
  [ "$no_log" -eq 0 ] && cmd+=(--log-file "$log_path")
  [ "$no_result" -eq 0 ] && cmd+=(--result-file "$result_path")

  local cmd_text
  printf -v cmd_text '%q ' "${cmd[@]}"

  local t0 t1 rc=0 timed_out=0 elapsed
  t0="$(date +%s)"
  if [ "$HAS_TIMEOUT" -eq 1 ] && [ "$case_timeout" -gt 0 ]; then
    timeout "$case_timeout" "${cmd[@]}" >"$stdout_f" 2>"$stderr_f"
    rc=$?
    [ "$rc" -eq 124 ] && timed_out=1
  else
    "${cmd[@]}" >"$stdout_f" 2>"$stderr_f"
    rc=$?
  fi
  t1="$(date +%s)"
  elapsed=$((t1 - t0))

  emit_check \
    --id "$id" --name "$name" --category "$category" --group "$category" \
    --command "$cmd_text" --result "$result_path" --log "$log_path" \
    --stdout-file "$stdout_f" --stderr-file "$stderr_f" \
    --elapsed "$elapsed" --exit-code "$rc" --timed-out "$timed_out" \
    --expect-global "$expect_global" --expect-inner "$expect_inner" \
    --expect-exit "$expect_exit" --expect-result "$expect_result" \
    --expect-message "$expect_message" --expect-request "$expect_request" \
    --expect-running "$expect_running" --expect-validate "$expect_validate" \
    --expect-no-result "$expect_no_result" --check-contract "$check_contract" \
    --runtime-observed "$runtime_observed" --timeout "$case_timeout" --notes "$notes"
}

# 瞬时失败用例（不需要 msmodeling / NPU）
quick_fail() {
  local id="$1" name="$2" argv="$3"
  shift 3
  run_case --id "$id" --name "$name" --category invalid --argv "$argv" \
    --expect-global 1 --expect-inner 1 --expect-exit 1 --expect-result empty \
    --expect-validate 1 --timeout "$QUICK_TIMEOUT" "$@"
}

# 真实寻优用例（需要 msmodeling / NPU）
real_case() {
  local id="$1" name="$2" category="$3" argv="$4"
  shift 4
  run_case --id "$id" --name "$name" --category "$category" --argv "$argv" \
    --expect-global 0 --expect-inner 0,3 --expect-exit 0 --expect-running 1 \
    --timeout "$TIMEOUT" "$@"
}

# 运行期「结果文件不得提前出现」的异步检查
runtime_check() {
  local id="contract_runtime_no_result_file" name="运行期不写结果文件" category="contract"
  if [ -n "$ONLY" ] && [[ "$id" != *"$ONLY"* ]]; then
    emit_skip "$id" "$name" "$category" "被 --only=$ONLY 过滤"
    return 0
  fi
  if ! should_run "$category"; then
    emit_skip "$id" "$name" "$category" "被 --mode=$MODE 过滤"
    return 0
  fi

  local log_path="$CASE_DIR/$id.log" result_path="$CASE_DIR/$id.json"
  local stdout_f="$CASE_DIR/$id.stdout" stderr_f="$CASE_DIR/$id.stderr"
  rm -f "$log_path" "$result_path" "$stdout_f" "$stderr_f"

  local -a cmd=("$PYTHON_BIN" "$ENTRY")
  local -a parts=()
  read -r -a parts <<< "$BASE --serving_modes 0"
  cmd+=("${parts[@]}" --log-file "$log_path" --result-file "$result_path")
  local cmd_text
  printf -v cmd_text '%q ' "${cmd[@]}"

  local t0 t1 rc=0 elapsed observed=1
  t0="$(date +%s)"
  "${cmd[@]}" >"$stdout_f" 2>"$stderr_f" &
  local pid=$!
  local i
  for i in 1 2 3 4 5; do
    sleep 1
    # 只有「进程仍在运行且结果文件已出现」才算违反契约
    if [ -f "$result_path" ] && kill -0 "$pid" 2>/dev/null; then
      observed=0
      break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
  done
  wait "$pid"
  rc=$?
  t1="$(date +%s)"
  elapsed=$((t1 - t0))

  emit_check \
    --id "$id" --name "$name" --category "$category" --group "$category" \
    --command "$cmd_text" --result "$result_path" --log "$log_path" \
    --stdout-file "$stdout_f" --stderr-file "$stderr_f" \
    --elapsed "$elapsed" --exit-code "$rc" --timed-out 0 \
    --expect-global 0 --expect-inner 0,3 --expect-exit 0 --expect-running 1 \
    --expect-result auto --expect-message "" --expect-request "" \
    --expect-validate 0 --expect-no-result 0 --check-contract 1 \
    --runtime-observed "$observed" --timeout "$TIMEOUT" \
    --notes "后台轮询 5s：检查终态之前 --result-file 是否存在"
}

echo "=========================================================================="
echo "入口脚本 : $ENTRY"
echo "校验器   : $HELPER"
echo "解释器   : $PYTHON_BIN"
echo "产物目录 : $OUT_DIR"
echo "模式     : $MODE    过滤: ${ONLY:-无}    超时: ${TIMEOUT}s / ${QUICK_TIMEOUT}s"
echo "开始时间 : $STARTED_AT"
echo "=========================================================================="

# --------------------------------------------------------------------------- A. 基础状态
real_case smoke_agg        "基础：serving_modes=0（仅 agg）"      smoke "$BASE --serving_modes 0"
real_case smoke_disagg     "基础：serving_modes=1（仅 disagg）"   smoke "$BASE --serving_modes 1"
real_case smoke_both       "基础：serving_modes=2（agg+disagg）"  smoke "$BASE --serving_modes 2"

# --------------------------------------------------------------------------- B. 默认值与别名
real_case default_no_ttft  "省略 ttft-limit（默认 20s）"          defaults "$BASE_NO_TTFT" --expect-request "ttft_limit=20.0"
real_case default_no_tpot  "省略 tpot-limit（默认 100ms）"        defaults "$BASE_NO_TPOT" --expect-request "tpot_limit=100"
real_case default_no_slo   "同时省略 ttft/tpot"                    defaults "$BASE_NO_SLO" --expect-request "ttft_limit=20.0;tpot_limit=100"
real_case default_no_modes "省略 serving_modes（默认 1）"          defaults "$BASE_NO_MODES" --expect-request "serving_modes=1"
real_case alias_underscore "下划线别名参数"                        alias "$ALIAS_UNDERSCORE"
real_case alias_hyphen     "连字符别名参数"                        alias "$ALIAS_HYPHEN"
real_case alias_logfile    "--log_file / --result_file 别名"       alias \
  "$COMMON --tpot-limit 100 --ttft-limit 20 --serving_modes 0 --log_file $CASE_DIR/alias_logfile.log --result_file $CASE_DIR/alias_logfile.json" \
  --no-log --no-result --log-path "$CASE_DIR/alias_logfile.log" --result-path "$CASE_DIR/alias_logfile.json"

# --------------------------------------------------------------------------- C. 模型 / 设备 / PD 矩阵
i=0
for model in \
  "GLM-5.2-w4a4c8-mxfp4" "GLM-5.2-BF16" "GLM-5.3-BF16" "GLM-5.2-FP8" \
  "GLM-5.3-FP8" "DeepSeek-R1-FP8" "DeepSeek-R1-0528-W8A8" \
  "DeepSeek-V4.1-Flash-MXFP4" "DeepSeek-V4-Pro-W8A8"; do
  i=$((i + 1))
  real_case "matrix_model_$(printf '%02d' "$i")" "模型方案：$model" matrix \
    "$BASE --model $model --serving_modes 0"
done
i=0
for device in 910B2 910B4 950DT; do
  i=$((i + 1))
  real_case "matrix_device_$(printf '%02d' "$i")" "设备：$device" matrix \
    "$BASE --device $device --serving_modes 0" --expect-request "device=$device"
done
i=0
for cards in 8 16 32 64; do
  i=$((i + 1))
  if [ "$cards" -eq 8 ]; then
    # 8 卡时三种对称组合全部被框架过滤 → 内层必为 3/empty，result 双空字典
    real_case "matrix_pd_$(printf '%02d' "$i")" "PD 组合：${cards} 卡（全过滤）" matrix \
      "$BASE --num_devices $cards --serving_modes 1" \
      --expect-inner 3 --expect-result empty
  else
    real_case "matrix_pd_$(printf '%02d' "$i")" "PD 组合：${cards} 卡" matrix \
      "$BASE --num_devices $cards --serving_modes 1"
  fi
done

# --------------------------------------------------------------------------- D. 值域边界
real_case boundary_nd_min     "num_devices 下界 8"            boundary "$BASE --num_devices 8 --serving_modes 0" --expect-request "num_devices=8;num_nodes=1"
real_case boundary_nd_max     "num_devices 上界 256"          boundary "$BASE --num_devices 256 --serving_modes 0" --expect-request "num_devices=256;num_nodes=32"
real_case boundary_in_min     "input_length 下界 128"         boundary "$BASE --input-length 128 --serving_modes 0" --expect-request "input_length=128"
real_case boundary_in_max     "input_length 上界 131072"      boundary "$BASE --input-length 131072 --serving_modes 0" --expect-request "input_length=131072"
real_case boundary_out_min    "output_length 下界 128"        boundary "$BASE --output-length 128 --serving_modes 0" --expect-request "output_length=128"
real_case boundary_out_max    "output_length 上界 131072"     boundary "$BASE --output-length 131072 --serving_modes 0" --expect-request "output_length=131072"
real_case boundary_ttft_min   "ttft_limit 下界 0.001"         boundary "$BASE --ttft-limit 0.001 --serving_modes 0" --expect-request "ttft_limit=0.001"
real_case boundary_ttft_max   "ttft_limit 上界 60"            boundary "$BASE --ttft-limit 60 --serving_modes 0" --expect-request "ttft_limit=60.0"
real_case boundary_tpot_min   "tpot_limit 下界 1"             boundary "$BASE --tpot-limit 1 --serving_modes 0" --expect-request "tpot_limit=1"
real_case boundary_tpot_max   "tpot_limit 上界 1000"          boundary "$BASE --tpot-limit 1000 --serving_modes 0" --expect-request "tpot_limit=1000"
real_case boundary_extreme    "极端 SLO（ttft 0.001/tpot 1）" boundary "$BASE --ttft-limit 0.001 --tpot-limit 1 --serving_modes 0" --notes "通常无可行解 → 内层 3/empty；环境差异允许 0/success"
real_case boundary_group_size "非 MXFP4 方案回显默认 group-size" boundary "$BASE --model GLM-5.2-BF16 --serving_modes 0" --expect-request "mxfp4-group-size=32"

# --------------------------------------------------------------------------- E. 失败分支（瞬时）
quick_fail invalid_model_unknown   "模型不在白名单"          "$BASE --model Unknown-Model" --expect-message "不在白名单内"
quick_fail invalid_model_case      "模型大小写不匹配"        "$BASE --model deepseek-v4.1-flash-mxfp4" --expect-message "不在白名单内"
quick_fail invalid_device_unknown  "设备代号非法"            "$BASE --device 950X" --expect-message "不在支持范围内"
quick_fail invalid_device_case     "设备大小写不匹配"        "$BASE --device 950dt" --expect-message "不在支持范围内"
quick_fail invalid_nd_multiple     "卡数非 8 倍数（12）"     "$BASE --num_devices 12" --expect-message "必须是 8 的倍数"
quick_fail invalid_nd_zero         "卡数为 0"                "$BASE --num_devices 0" --expect-message "超出取值范围"
quick_fail invalid_nd_above        "卡数越界（264）"         "$BASE --num_devices 264" --expect-message "超出取值范围"
quick_fail invalid_nd_nonint       "卡数非整数"              "$BASE --num_devices abc" --expect-message "必须是整数"
quick_fail invalid_input_below     "输入长度 127"            "$BASE --input-length 127" --expect-message "超出取值范围"
quick_fail invalid_input_above     "输入长度 131073"         "$BASE --input-length 131073" --expect-message "超出取值范围"
quick_fail invalid_output_below    "输出长度 127"            "$BASE --output-length 127" --expect-message "超出取值范围"
quick_fail invalid_output_above    "输出长度 131073"         "$BASE --output-length 131073" --expect-message "超出取值范围"
quick_fail invalid_ttft_zero       "ttft_limit 为 0"         "$BASE --ttft-limit 0" --expect-message "超出取值范围"
quick_fail invalid_ttft_above      "ttft_limit 60.001"       "$BASE --ttft-limit 60.001" --expect-message "超出取值范围"
quick_fail invalid_ttft_nonnum     "ttft_limit 非数值"        "$BASE --ttft-limit abc" --expect-message "必须是浮点数"
quick_fail invalid_ttft_nan        "ttft_limit NaN"          "$BASE --ttft-limit nan" --expect-message "超出取值范围"
quick_fail invalid_tpot_zero       "tpot_limit 为 0"         "$BASE --tpot-limit 0" --expect-message "超出取值范围"
quick_fail invalid_tpot_above      "tpot_limit 1001"         "$BASE --tpot-limit 1001" --expect-message "超出取值范围"
quick_fail invalid_modes_3         "serving_modes=3"         "$BASE --serving_modes 3" --expect-message "超出取值范围"
quick_fail invalid_modes_neg       "serving_modes=-1"        "$BASE --serving_modes -1" --expect-message "超出取值范围"
quick_fail invalid_result_ext      "结果文件非 .json"         "$BASE" --result-suffix .txt --expect-message "必须是 .json" --notes "文件名为 .txt，内容仍是完整契约 JSON"
quick_fail invalid_log_dir         "日志路径指向目录"         "$BASE" --log-path "$CASE_DIR/log_dir_target" --expect-message "指向了目录" --notes "日志退化为 stdout，仍应写出失败结果 JSON"
quick_fail missing_log_file        "缺少 --log-file"          "$BASE" --no-log --expect-message "required"
quick_fail missing_result_file     "缺少 --result-file"       "$BASE" --no-result --expect-message "required" --expect-no-result 1 --check-contract 0 --notes "无法确定结果路径：不落盘、退出 1"
quick_fail missing_model           "缺少 --model"            "$BASE_NO_MODEL" --expect-message "required"
quick_fail missing_device          "缺少 --device"           "$BASE_NO_DEVICE" --expect-message "required"

# --------------------------------------------------------------------------- F. 落盘契约
runtime_check

# --------------------------------------------------------------------------- 报告
echo "=========================================================================="
echo "生成测试报告 ..."
"$PYTHON_BIN" "$HELPER" report \
  --results "$RESULTS" --out-dir "$OUT_DIR" --entry "$ENTRY" \
  --python-bin "$PYTHON_BIN" --mode "$MODE" --timeout "$TIMEOUT" \
  --started-at "$STARTED_AT" --base-args "$BASE"

fail_count="$(grep -c '"status": "FAIL"' "$RESULTS" || true)"
echo "=========================================================================="
if [ "$fail_count" -gt 0 ]; then
  echo "存在 $fail_count 个失败用例，详见 $OUT_DIR/report.md 与 $OUT_DIR/failures.txt"
  exit 1
fi
echo "全部通过（SKIP 不计失败），报告见 $OUT_DIR/report.md"
exit 0
