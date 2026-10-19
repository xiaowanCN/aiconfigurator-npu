#!/usr/bin/env bash
# run_infer_strategy_tests.sh
# 自动跑 infer-strategy-npu 的全部测试命令，逐条记录日志并打印汇总。
#
# 用法:
#   bash run_infer_strategy_tests.sh                            # 默认: A(dry-run 全量) + B(真实 aic-npu)
#   bash run_infer_strategy_tests.sh --with-msmodeling-real     # 额外跑 C(真实 msmodeling, 需 NPU 主机)
#   nohup bash run_infer_strategy_tests.sh --with-msmodeling-real > itn-test-results/run_all.log 2>&1 &
#
# 从 /workspace(统一镜像默认 CWD) 或仓库根目录运行, 以便 model_configs/msmodeling 相对路径可被解析。
# 环境变量可覆盖: DEVICE / NUM_DEVICES / OUT_DIR / BIN / WITH_MS
#   / INPUT_LENGTH / OUTPUT_LENGTH / TPOT_LIMIT / TTFT_LIMIT / JOBS

set -u

# ---- 可调参数(可用环境变量覆盖) ----
DEVICE="${DEVICE:-ATLAS_800_A2_376T_64G}"
NUM_DEVICES="${NUM_DEVICES:-64}"
OUT_DIR="${OUT_DIR:-./itn-test-results}"
BIN="${BIN:-infer-strategy-npu}"
WITH_MS="${WITH_MS:-0}"
[ "${1:-}" = "--with-msmodeling-real" ] && WITH_MS=1

# 推荐请求默认参数(与 CLI 内置默认值一致; 留空则使用 CLI 默认)
INPUT_LENGTH="${INPUT_LENGTH:-8196}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-1024}"
TPOT_LIMIT="${TPOT_LIMIT:-100}"
TTFT_LIMIT="${TTFT_LIMIT:-20000}"
# JOBS 留空 -> 不传 --jobs, 由 msmodeling 自行决定搜索并发度
JOBS="${JOBS:-64}"

# 真实 msmodeling 导入链会显式 import torch_npu; 关闭后端自动加载可避免无驱动环境下
# torch 启动时去探测 NPU(在真实 NPU 主机上不影响功能)。
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

LOG_DIR="$OUT_DIR/logs"
SUMMARY="$OUT_DIR/summary.txt"
mkdir -p "$LOG_DIR"
: > "$SUMMARY"

pass=0
fail=0

run_test() {
  local id="$1" desc="$2"; shift 2
  local log="$LOG_DIR/${id}.log"
  local t0 t1 rc
  t0=$(date +%s)
  if "$@" >"$log" 2>&1; then rc=0; else rc=$?; fi
  t1=$(date +%s)
  if [ "$rc" -eq 0 ]; then
    printf '  [PASS] %-6s (%3ds) %s\n' "$id" "$((t1-t0))" "$desc"
    printf 'PASS %-6s %s\n' "$id" "$desc" >>"$SUMMARY"
    pass=$((pass+1))
  else
    printf '  [FAIL] %-6s rc=%d (%3ds) %s\n' "$id" "$rc" "$((t1-t0))" "$desc"
    printf 'FAIL %-6s %s\n' "$id" "$desc" >>"$SUMMARY"
    echo "    ---- tail of $log ----"
    tail -n 20 "$log" | sed 's/^/    /'
    fail=$((fail+1))
  fi
}

# 公共参数(注意: 调用处保持不引号, 以便拆成多个独立参数)
C="--device $DEVICE --num-devices $NUM_DEVICES"
# 推荐请求公共参数(序列长度/时延约束/并发; 不引号以便拆词)
X="--input-length $INPUT_LENGTH --output-length $OUTPUT_LENGTH --tpot-limit $TPOT_LIMIT --ttft-limit $TTFT_LIMIT"
[ -n "$JOBS" ] && X="$X --jobs $JOBS"

echo "================================================================"
echo " infer-strategy-npu 测试套件"
echo " device=$DEVICE  num_devices=$NUM_DEVICES  out_dir=$OUT_DIR  bin=$BIN  with_msmodeling_real=$WITH_MS"
echo " input_length=$INPUT_LENGTH  output_length=$OUTPUT_LENGTH  tpot_limit=$TPOT_LIMIT  ttft_limit=$TTFT_LIMIT  jobs=$JOBS"
echo "================================================================"

# ---------- 0. 自检 ----------
run_test T00-version "version"       "$BIN" --version
run_test T00-models  "list-models"   "$BIN" --list-models
run_test T00-devices "list-devices"  "$BIN" --list-devices

# ---------- A. dry-run 全量(安全, 任何环境可跑) ----------
# pass
run_test T01 "DeepSeek-R1     DISABLE auto -> aic-npu"    "$BIN" -recommend --model deepseek-ai/DeepSeek-R1     $C $X --quantize DISABLE --out-dir "$OUT_DIR/T01" --dry-run
# pass
run_test T02 "DeepSeek-R1     W8A8    auto -> msmodeling" "$BIN" -recommend --model deepseek-ai/DeepSeek-R1     $C $X --quantize W8A8   --out-dir "$OUT_DIR/T02" --dry-run
# pass
run_test T03 "Qwen3.8-27B     DISABLE auto -> aic-npu"    "$BIN" -recommend --model Qwen/Qwen3.8-27B            $C $X --quantize DISABLE --out-dir "$OUT_DIR/T03" --dry-run
# pass
run_test T04 "Qwen3.8-27B     W8A8    auto -> msmodeling" "$BIN" -recommend --model Qwen/Qwen3.8-27B            $C $X --quantize W8A8   --out-dir "$OUT_DIR/T04" --dry-run
# pass
run_test T05 "DeepSeek-V4-Pro DISABLE auto -> aic-npu"    "$BIN" -recommend --model deepseek-ai/DeepSeek-V4-Pro $C $X --quantize DISABLE --out-dir "$OUT_DIR/T05" --dry-run
run_test T06 "DeepSeek-V4-Pro W8A8    auto -> msmodeling" "$BIN" -recommend --model deepseek-ai/DeepSeek-V4-Pro $C $X --quantize W8A8   --out-dir "$OUT_DIR/T06" --dry-run
run_test T07 "GLM-5.3-Flash   DISABLE auto -> msmodeling" "$BIN" -recommend --model ZhipuAI/GLM-5.3-Flash       $C $X --quantize DISABLE --out-dir "$OUT_DIR/T07" --dry-run
# run_test T08 "Kimi-K3         W8A8    auto -> msmodeling" "$BIN" -recommend --model moonshotai/Kimi-K3          $C $X --quantize W8A8   --out-dir "$OUT_DIR/T08" --dry-run
# run_test T09 "MiniMax-M3      W8A8    auto -> msmodeling" "$BIN" -recommend --model MiniMax/MiniMax-M3          $C $X --quantize W8A8   --out-dir "$OUT_DIR/T09" --dry-run
run_test T10 "GLM-5.3-Flash   forced msmodeling"          "$BIN" -recommend --model ZhipuAI/GLM-5.3-Flash       $C $X --framework msmodeling --out-dir "$OUT_DIR/T10" --dry-run
run_test T11 "DeepSeek-R1     forced aic-npu"             "$BIN" -recommend --model deepseek-ai/DeepSeek-R1     $C $X --framework aic-npu    --out-dir "$OUT_DIR/T11" --dry-run

# ---------- B. 真实 aic-npu(纯 Python 估算, CPU 即可跑) ----------
run_test R01 "DeepSeek-R1     aic-npu real"  "$BIN" -recommend --model deepseek-ai/DeepSeek-R1     $C $X --framework aic-npu --out-dir "$OUT_DIR/R01"
run_test R02 "Qwen3.8-27B     aic-npu real"  "$BIN" -recommend --model Qwen/Qwen3.8-27B            $C $X --framework aic-npu --out-dir "$OUT_DIR/R02"
run_test R03 "DeepSeek-V4-Pro aic-npu real"  "$BIN" -recommend --model deepseek-ai/DeepSeek-V4-Pro $C $X --framework aic-npu --out-dir "$OUT_DIR/R03"

# ---------- C. 真实 msmodeling(需 NPU 主机; transformers 会 import torch_npu) ----------
if [ "$WITH_MS" -eq 1 ]; then
  run_test R04 "DeepSeek-R1     msmodeling real (W8A8)" "$BIN" -recommend --model deepseek-ai/DeepSeek-R1     $C $X --quantize W8A8 --out-dir "$OUT_DIR/R04"
  run_test R05 "Qwen3.8-27B     msmodeling real (W8A8)" "$BIN" -recommend --model Qwen/Qwen3.8-27B            $C $X --quantize W8A8 --out-dir "$OUT_DIR/R05"
  run_test R06 "DeepSeek-V4-Pro msmodeling real (W8A8)" "$BIN" -recommend --model deepseek-ai/DeepSeek-V4-Pro $C $X --quantize W8A8 --out-dir "$OUT_DIR/R06"
  # 以下三个模型目前是占位 config, 真实 msmodeling 可能失败, 属预期
  run_test R07 "GLM-5.3-Flash   msmodeling real (占位config, 可能失败)" "$BIN" -recommend --model ZhipuAI/GLM-5.3-Flash $C $X --out-dir "$OUT_DIR/R07"
  # run_test R08 "Kimi-K3         msmodeling real (占位config, 可能失败)" "$BIN" -recommend --model moonshotai/Kimi-K3          $C $X --quantize W8A8 --out-dir "$OUT_DIR/R08"
  # run_test R09 "MiniMax-M3      msmodeling real (占位config, 可能失败)" "$BIN" -recommend --model MiniMax/MiniMax-M3          $C $X --quantize W8A8 --out-dir "$OUT_DIR/R09"
fi

echo "================================================================"
echo " 汇总: PASS=$pass  FAIL=$fail"
echo " 明细见 $SUMMARY"
echo " 各用例日志见 $LOG_DIR/<id>.log"
echo "================================================================"
exit $fail
