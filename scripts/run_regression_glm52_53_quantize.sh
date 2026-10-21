#!/usr/bin/env bash
# run_regression_glm52_53_quantize.sh
# GLM5.2 / GLM5.3 量化回归: 2 模型 x 2 卡数 = 4 用例
#   固定场景: device=ATLAS_950_486T_96G / isl=8196 / osl=1024 / ttft=30000 / tpot=200
#             quantize=ENABLE 且显式覆盖 linear=MXFP4 / attention=FP8 / tp-sizes=2 4 8 16
#             (non-expert-linear 未指定 -> 默认 DISABLED)
#
# 预期路由(便于核对): GLM5.2 / GLM5.3 均非 aic-npu 白名单 -> 一律走 msmodeling
#
# 用法:
#   bash scripts/run_regression_glm52_53_quantize.sh                 # 真实跑 4 个用例
#   bash scripts/run_regression_glm52_53_quantize.sh --dry-run       # 只渲染命令, 不实际执行(安全预览)
#   nohup bash scripts/run_regression_glm52_53_quantize.sh > itn-regression-glm52-53-quantize/run_all.log 2>&1 &
#
# 从 /workspace(统一镜像默认 CWD) 或仓库根目录运行, 以便 model_configs/msmodeling 相对路径可被解析。
# 环境变量可覆盖: DEVICE / NUM_DEVICES(空格分隔列表) / MODELS(空格分隔列表) /
#   INPUT_LENGTH / OUTPUT_LENGTH / TTFT_LIMIT / TPOT_LIMIT / QUANTIZE /
#   LINEAR_ACTION / ATTENTION_ACTION / TP_SIZES / OUT_DIR / BIN / JOBS / DRY_RUN

set -u

# ---- 可调参数(可用环境变量覆盖) ----
DEVICE="${DEVICE:-ATLAS_950_486T_96G}"
# 卡数列表: 16 与 32
NUM_DEVICES="${NUM_DEVICES:-16 32}"
# 目标模型列表
MODELS="${MODELS:-ZhipuAI/GLM-5.2 ZhipuAI/GLM-5.3}"
INPUT_LENGTH="${INPUT_LENGTH:-8196}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-1024}"
TTFT_LIMIT="${TTFT_LIMIT:-30000}"
TPOT_LIMIT="${TPOT_LIMIT:-200}"
QUANTIZE="${QUANTIZE:-ENABLE}"
LINEAR_ACTION="${LINEAR_ACTION:-MXFP4}"
ATTENTION_ACTION="${ATTENTION_ACTION:-FP8}"
# 显式 tp-sizes, 不自动推导
TP_SIZES="${TP_SIZES:-2 4 8 16}"
OUT_DIR="${OUT_DIR:-./itn-regression-glm52-53-quantize}"
BIN="${BIN:-infer-strategy-npu}"
JOBS="${JOBS:-32}"
# PD 均衡搜索默认开启
PD_OFF="${PD_OFF:-0}"
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
C="--device $DEVICE"
X="--input-length $INPUT_LENGTH --output-length $OUTPUT_LENGTH --tpot-limit $TPOT_LIMIT --ttft-limit $TTFT_LIMIT"
X="$X --quantize $QUANTIZE --quantize-linear-action $LINEAR_ACTION --quantize-attention-action $ATTENTION_ACTION"
[ -n "$TP_SIZES" ] && X="$X --tp-sizes $TP_SIZES"
[ -n "$JOBS" ] && X="$X --jobs $JOBS"
[ "$PD_OFF" -eq 1 ] && X="$X --no-pd-balance-search"
[ "$DRY_RUN" -eq 1 ] && X="$X --dry-run"

echo "================================================================"
echo " infer-strategy-npu GLM5.2/GLM5.3 量化回归 (2 模型 x 2 卡数 = 4 用例)"
echo " device=$DEVICE  num_devices=[$NUM_DEVICES]  out_dir=$OUT_DIR  bin=$BIN"
echo " input_length=$INPUT_LENGTH  output_length=$OUTPUT_LENGTH  tpot_limit=$TPOT_LIMIT  ttft_limit=$TTFT_LIMIT"
echo " quantize=$QUANTIZE  linear=$LINEAR_ACTION  attention=$ATTENTION_ACTION  tp_sizes=[$TP_SIZES]  dry_run=$DRY_RUN"
echo "================================================================"

# ---------- 0. 自检 ----------
run_test T00-version "version" "$BIN" --version

# ---------- 回归用例: 模型 x 卡数 ----------
for m in $MODELS; do
  # 模型缩写, 用于 id 与结果目录
  case "$m" in
    ZhipuAI/GLM-5.2) mid="glm52" ;;
    ZhipuAI/GLM-5.3) mid="glm53" ;;
    *) mid=$(echo "$m" | tr '/A-Z' '_a-z') ;;
  esac
  for nd in $NUM_DEVICES; do
    id="${mid}-${nd}"
    run_test "$id" "$m  ${nd}卡  quantize=$QUANTIZE($LINEAR_ACTION/$ATTENTION_ACTION) tp=[$TP_SIZES]" \
      "$BIN" -recommend --model "$m" --num-devices "$nd" $C $X --out-dir "$OUT_DIR/$id"
  done
done

echo "================================================================"
echo " 汇总: PASS=$pass  FAIL=$fail"
echo " 明细见 $SUMMARY"
echo " 各用例日志见 $LOG_DIR/<id>.log"
echo " 各用例结果见 $OUT_DIR/<id>/  (run.json / summary.csv)"
echo "================================================================"
exit $fail
