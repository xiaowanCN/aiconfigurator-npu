#!/usr/bin/env bash
# run_regression_6models_300npu.sh
# 全量回归：6 个模型 x 2 种量化 = 12 个用例
#   固定场景：300 卡 / 8K 输入 / 1K 输出 / ttft 30000 / tpot 200
#
# 模型（4）：ZhipuAI/GLM-5.3-Flash、Qwen/Qwen3.8-27B、deepseek-ai/DeepSeek-R1、deepseek-ai/DeepSeek-V4-Pro
# 量化（2）：DISABLE、W8A8
#
# 预期路由（便于核对结果）：
#   DISABLE -> GLM-5.3-Flash 走 msmodeling；另外 3 个走 aic-npu
#   W8A8    -> 4 个全部走 msmodeling
#   即本回归同时覆盖 aic-npu(3 例) 与 msmodeling(5 例) 两条下发路径。
#
# 用法:
#   bash scripts/run_regression_6models_300npu.sh                 # 真实跑 8 个用例
#   bash scripts/run_regression_6models_300npu.sh --dry-run       # 只渲染命令, 不实际执行(安全预览)
#   nohup bash scripts/run_regression_6models_300npu.sh > itn-regression-300npu/run_all.log 2>&1 &
#
# 从 /workspace(统一镜像默认 CWD) 或仓库根目录运行, 以便 model_configs/msmodeling 相对路径可被解析。
# 环境变量可覆盖: DEVICE / NUM_DEVICES / INPUT_LENGTH / OUTPUT_LENGTH / TTFT_LIMIT / TPOT_LIMIT
#   / OUT_DIR / BIN / TP_SIZES / JOBS / DRY_RUN

set -u

# ---- 可调参数(可用环境变量覆盖) ----
DEVICE="${DEVICE:-ATLAS_800_A2_376T_64G}"
NUM_DEVICES="${NUM_DEVICES:-300}"
INPUT_LENGTH="${INPUT_LENGTH:-8192}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-1024}"
TTFT_LIMIT="${TTFT_LIMIT:-30000}"
TPOT_LIMIT="${TPOT_LIMIT:-200}"
OUT_DIR="${OUT_DIR:-./itn-regression-300npu}"
BIN="${BIN:-infer-strategy-npu}"
# TP_SIZES 留空 -> 不传, 由 infer-strategy-npu 按模型与卡数自动推导
TP_SIZES="${TP_SIZES:-}"
JOBS="${JOBS:-32}"
# PD 均衡搜索(xPyD)默认开启; PD_OFF=1 关闭, PD_MAX=N 限制 P/D 节点数上限(默认 3)
# PD_SIZES 显式指定组合(如 "1P1D 2P1D 3P1D"), 给定时忽略 PD_MAX
PD_OFF="${PD_OFF:-0}"
PD_MAX="${PD_MAX:-}"
PD_SIZES="${PD_SIZES:-}"
DRY_RUN="${DRY_RUN:-0}"
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

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
    printf '  [PASS] %-16s (%4ds) %s\n' "$id" "$((t1-t0))" "$desc"
    printf 'PASS %-16s %s\n' "$id" "$desc" >>"$SUMMARY"
    pass=$((pass+1))
  else
    printf '  [FAIL] %-16s rc=%d (%4ds) %s\n' "$id" "$rc" "$((t1-t0))" "$desc"
    printf 'FAIL %-16s %s\n' "$id" "$desc" >>"$SUMMARY"
    echo "    ---- tail of $log ----"
    tail -n 20 "$log" | sed 's/^/    /'
    fail=$((fail+1))
  fi
}

# 公共参数(注意: 调用处保持不引号, 以便拆成多个独立参数)
C="--device $DEVICE --num-devices $NUM_DEVICES"
X="--input-length $INPUT_LENGTH --output-length $OUTPUT_LENGTH --tpot-limit $TPOT_LIMIT --ttft-limit $TTFT_LIMIT"
[ -n "$TP_SIZES" ] && X="$X --tp-sizes $TP_SIZES"
[ -n "$JOBS" ] && X="$X --jobs $JOBS"
[ "$PD_OFF" -eq 1 ] && X="$X --serving-modes agg"
[ "$DISAGG_ONLY" -eq 1 ] && X="$X --serving-modes disagg"
[ -n "$PD_SIZES" ] && X="$X --pd-ratio-sizes $PD_SIZES"
[ "$DRY_RUN" -eq 1 ] && X="$X --dry-run"

echo "================================================================"
echo " infer-strategy-npu 全量回归 (6 模型 x 2 量化 = 12 用例)"
echo " device=$DEVICE  num_devices=$NUM_DEVICES  out_dir=$OUT_DIR  bin=$BIN"
echo " input_length=$INPUT_LENGTH  output_length=$OUTPUT_LENGTH  tpot_limit=$TPOT_LIMIT  ttft_limit=$TTFT_LIMIT"
echo " tp_sizes=${TP_SIZES:-<自动推导>}  jobs=${JOBS:-<cli默认>}  dry_run=$DRY_RUN"
echo "================================================================"

# ---------- 0. 自检 ----------
run_test T00-version "version" "$BIN" --version

# ---------- 8 个回归用例 ----------
# id 规则: <模型缩写>-<量化>, 便于按名字定位日志与结果目录
run_test glm53flash-DISABLE "GLM-5.3-Flash  DISABLE -> msmodeling" "$BIN" -recommend --model ZhipuAI/GLM-5.3-Flash       $C $X --quantize DISABLE --out-dir "$OUT_DIR/glm53flash-DISABLE"
run_test glm53flash-W8A8    "GLM-5.3-Flash  W8A8    -> msmodeling" "$BIN" -recommend --model ZhipuAI/GLM-5.3-Flash       $C $X --quantize W8A8    --out-dir "$OUT_DIR/glm53flash-W8A8"
run_test qwen38b-DISABLE    "Qwen3.8-27B    DISABLE -> aic-npu"    "$BIN" -recommend --model Qwen/Qwen3.8-27B            $C $X --quantize DISABLE --out-dir "$OUT_DIR/qwen38b-DISABLE"
run_test qwen38b-W8A8       "Qwen3.8-27B    W8A8    -> msmodeling" "$BIN" -recommend --model Qwen/Qwen3.8-27B            $C $X --quantize W8A8    --out-dir "$OUT_DIR/qwen38b-W8A8"
run_test deepseekR1-DISABLE "DeepSeek-R1    DISABLE -> aic-npu"    "$BIN" -recommend --model deepseek-ai/DeepSeek-R1     $C $X --quantize DISABLE --out-dir "$OUT_DIR/deepseekR1-DISABLE"
run_test deepseekR1-W8A8    "DeepSeek-R1    W8A8    -> msmodeling" "$BIN" -recommend --model deepseek-ai/DeepSeek-R1     $C $X --quantize W8A8    --out-dir "$OUT_DIR/deepseekR1-W8A8"
run_test deepseekV4-DISABLE "DeepSeek-V4-Pro DISABLE -> aic-npu"   "$BIN" -recommend --model deepseek-ai/DeepSeek-V4-Pro $C $X --quantize DISABLE --out-dir "$OUT_DIR/deepseekV4-DISABLE"
run_test deepseekV4-W8A8    "DeepSeek-V4-Pro W8A8    -> msmodeling" "$BIN" -recommend --model deepseek-ai/DeepSeek-V4-Pro $C $X --quantize W8A8    --out-dir "$OUT_DIR/deepseekV4-W8A8"
run_test kimik3-DISABLE     "Kimi-K3 DISABLE -> aic-npu"   "$BIN" -recommend --model moonshotai/Kimi-K3 $C $X --quantize DISABLE --out-dir "$OUT_DIR/kimik3-DISABLE"
run_test kimik3-W8A8        "Kimi-K3 W8A8    -> msmodeling" "$BIN" -recommend --model moonshotai/Kimi-K3 $C $X --quantize W8A8    --out-dir "$OUT_DIR/kimik3-W8A8"
run_test minimaxm3-DISABLE     "MiniMax-M3 DISABLE -> aic-npu"   "$BIN" -recommend --model MiniMax/MiniMax-M3 $C $X --quantize DISABLE --out-dir "$OUT_DIR/minimaxm3-DISABLE"
run_test minimaxm3-W8A8        "MiniMax-M3 W8A8    -> msmodeling" "$BIN" -recommend --model MiniMax/MiniMax-M3 $C $X --quantize W8A8    --out-dir "$OUT_DIR/minimaxm3-W8A8"



echo "================================================================"
echo " 汇总: PASS=$pass  FAIL=$fail"
echo " 明细见 $SUMMARY"
echo " 各用例日志见 $LOG_DIR/<id>.log"
echo " 各用例结果见 $OUT_DIR/<id>/  (run.json / summary.csv)"
echo "================================================================"
exit $fail
