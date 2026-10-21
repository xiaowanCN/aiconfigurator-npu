#!/usr/bin/env bash
# run_regression_allmodels_quantize.sh
# 全量模型回归: 8 模型 x 2 量化 x 2 卡数 = 32 用例
#   固定场景: device=ATLAS_800_A2_376T_64G / isl=8196 / osl=1024 / ttft=30000 / tpot=200
#
# 量化:
#   DISABLE -> 三 action 强制 DISABLED（aic-npu 白名单的 3 个模型走 aic-npu，其余走 msmodeling）
#   ENABLE  -> 走 msmodeling，action 取原生默认（linear=W8A8_DYNAMIC / non_expert=DISABLED / attention=DISABLED）
#             可用环境变量 LINEAR_ACTION / NON_EXPERT_LINEAR_ACTION / ATTENTION_ACTION 显式覆盖
#
# 设备说明:
#   aic-npu 只支持 ATLAS_800_A2_376T_64G（仅该设备能映射到 aic-npu 的 --system），其余设备一律走 msmodeling。
#   因此本脚本默认 DEVICE=ATLAS_800_A2_376T_64G，可让 8 模型 x 2 量化 全 32 用例都跑得通
#   （aic-npu 白名单的 3 个模型 DISABLE 走 aic-npu，其余走 msmodeling）。
#   若把 DEVICE 覆盖成 ATLAS_950_486T_96G 等其它设备，则那 3 个模型的 DISABLE 用例会被 SKIP（aic-npu 不支持该设备）。
#
# 预期路由(便于核对, 默认 DEVICE=ATLAS_800_A2_376T_64G):
#   aic-npu : Qwen3.8-27B / DeepSeek-R1 / DeepSeek-V4-Pro 的 DISABLE
#   msmodeling: 其余全部（含这 3 个模型的 ENABLE，以及 GLM/Kimi/MiniMax 的 DISABLE 与 ENABLE）
#
# 用法:
#   bash scripts/run_regression_allmodels_quantize.sh                 # 真实跑 32 个用例
#   bash scripts/run_regression_allmodels_quantize.sh --dry-run       # 只渲染命令, 不实际执行(安全预览)
#   LINEAR_ACTION=MXFP4 ATTENTION_ACTION=FP8 bash scripts/run_regression_allmodels_quantize.sh
#   DEVICE=ATLAS_950_486T_96G bash scripts/run_regression_allmodels_quantize.sh   # 仅 msmodeling 路径(aic-npu DISABLE 用例 SKIP)
#
# 从 /workspace(统一镜像默认 CWD) 或仓库根目录运行, 以便 model_configs/msmodeling 相对路径可被解析。
# 环境变量可覆盖: DEVICE / MODELS(空格分隔) / QUANTIZES(空格分隔) / NUM_DEVICES(空格分隔) /
#   INPUT_LENGTH / OUTPUT_LENGTH / TTFT_LIMIT / TPOT_LIMIT / LINEAR_ACTION /
#   NON_EXPERT_LINEAR_ACTION / ATTENTION_ACTION / TP_SIZES / OUT_DIR / BIN / JOBS / DRY_RUN

set -u

# ---- 可调参数(可用环境变量覆盖) ----
DEVICE="${DEVICE:-ATLAS_800_A2_376T_64G}"
# aic-npu 仅支持该设备; 其余设备一律走 msmodeling
AIC_NPU_DEVICE="ATLAS_800_A2_376T_64G"
AIC_NPU_MODEL_IDS="Qwen/Qwen3.8-27B deepseek-ai/DeepSeek-R1 deepseek-ai/DeepSeek-V4-Pro"
MODELS="${MODELS:-ZhipuAI/GLM-5.3-Flash ZhipuAI/GLM-5.3 ZhipuAI/GLM-5.2 Qwen/Qwen3.8-27B deepseek-ai/DeepSeek-R1 deepseek-ai/DeepSeek-V4-Pro moonshotai/Kimi-K3 MiniMax/MiniMax-M3}"
QUANTIZES="${QUANTIZES:-DISABLE ENABLE}"
NUM_DEVICES="${NUM_DEVICES:-16 32}"
INPUT_LENGTH="${INPUT_LENGTH:-8196}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-1024}"
TTFT_LIMIT="${TTFT_LIMIT:-30000}"
TPOT_LIMIT="${TPOT_LIMIT:-200}"
# ENABLE 时的显式 action 覆盖（留空 -> 取 msmodeling 原生默认）
LINEAR_ACTION="${LINEAR_ACTION:-}"
NON_EXPERT_LINEAR_ACTION="${NON_EXPERT_LINEAR_ACTION:-}"
ATTENTION_ACTION="${ATTENTION_ACTION:-}"
# 留空 -> 不传 --tp-sizes, 由 infer-strategy-npu 按模型与卡数自动推导
TP_SIZES="${TP_SIZES:-}"
OUT_DIR="${OUT_DIR:-./itn-regression-allmodels-quantize}"
BIN="${BIN:-infer-strategy-npu}"
JOBS="${JOBS:-}"
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
skip=0

run_test() {
  local id="$1" desc="$2"; shift 2
  local log="$LOG_DIR/${id}.log"
  local t0 t1 rc
  t0=$(date +%s)
  if "$@" >"$log" 2>&1; then rc=0; else rc=$?; fi
  t1=$(date +%s)
  if [ "$rc" -eq 0 ]; then
    printf '  [PASS] %-24s (%4ds) %s\n' "$id" "$((t1-t0))" "$desc"
    printf 'PASS %-24s %s\n' "$id" "$desc" >>"$SUMMARY"
    pass=$((pass+1))
  else
    printf '  [FAIL] %-24s rc=%d (%4ds) %s\n' "$id" "$rc" "$((t1-t0))" "$desc"
    printf 'FAIL %-24s %s\n' "$id" "$desc" >>"$SUMMARY"
    echo "    ---- tail of $log ----"
    tail -n 20 "$log" | sed 's/^/    /'
    fail=$((fail+1))
  fi
}

skip_test() {
  # 因环境/设备原因主动跳过（非失败），写入 SUMMARY 并计数
  local id="$1" desc="$2"
  printf '  [SKIP] %-24s %s\n' "$id" "$desc"
  printf 'SKIP %-24s %s\n' "$id" "$desc" >>"$SUMMARY"
  skip=$((skip+1))
}

# 判断某模型是否属于 aic-npu 白名单（仅这些模型的 DISABLE 才会路由到 aic-npu）
is_aic_npu_model() {
  local m="$1"
  for a in $AIC_NPU_MODEL_IDS; do
    [ "$m" = "$a" ] && return 0
  done
  return 1
}

# 公共参数(注意: 调用处保持不引号, 以便拆成多个独立参数)
C="--device $DEVICE"
X="--input-length $INPUT_LENGTH --output-length $OUTPUT_LENGTH --tpot-limit $TPOT_LIMIT --ttft-limit $TTFT_LIMIT"
[ -n "$TP_SIZES" ] && X="$X --tp-sizes $TP_SIZES"
[ -n "$JOBS" ] && X="$X --jobs $JOBS"
[ "$PD_OFF" -eq 1 ] && X="$X --no-pd-balance-search"
[ "$DRY_RUN" -eq 1 ] && X="$X --dry-run"

echo "================================================================"
echo " infer-strategy-npu 全量模型量化回归 (8 模型 x 2 量化 x 2 卡数 = 32 用例)"
echo " device=$DEVICE  num_devices=[$NUM_DEVICES]  out_dir=$OUT_DIR  bin=$BIN"
echo " input_length=$INPUT_LENGTH  output_length=$OUTPUT_LENGTH  tpot_limit=$TPOT_LIMIT  ttft_limit=$TTFT_LIMIT"
echo " quantizes=[$QUANTIZES]  linear=${LINEAR_ACTION:-<默认>} non_expert=${NON_EXPERT_LINEAR_ACTION:-<默认>} attention=${ATTENTION_ACTION:-<默认>}"
echo " tp_sizes=${TP_SIZES:-<自动推导>}  dry_run=$DRY_RUN"
echo "================================================================"

# ---------- 0. 自检 ----------
run_test T00-version "version" "$BIN" --version

# ---------- 回归用例: 模型 x 量化 x 卡数 ----------
for m in $MODELS; do
  # 模型缩写, 用于 id 与结果目录
  case "$m" in
    ZhipuAI/GLM-5.3-Flash) mid="glm53flash" ;;
    ZhipuAI/GLM-5.3)       mid="glm53" ;;
    ZhipuAI/GLM-5.2)       mid="glm52" ;;
    Qwen/Qwen3.8-27B)      mid="qwen38b" ;;
    deepseek-ai/DeepSeek-R1)     mid="deepseekR1" ;;
    deepseek-ai/DeepSeek-V4-Pro) mid="deepseekV4" ;;
    moonshotai/Kimi-K3)     mid="kimik3" ;;
    MiniMax/MiniMax-M3)     mid="minimaxm3" ;;
    *) mid=$(echo "$m" | tr '/A-Z' '_a-z') ;;
  esac
  for q in $QUANTIZES; do
    # 量化参数(ENABLE 时按需追加显式 action 覆盖)
    QP="--quantize $q"
    if [ "$q" = "ENABLE" ]; then
      [ -n "$LINEAR_ACTION" ] && QP="$QP --quantize-linear-action $LINEAR_ACTION"
      [ -n "$NON_EXPERT_LINEAR_ACTION" ] && QP="$QP --quantize-non-expert-linear-action $NON_EXPERT_LINEAR_ACTION"
      [ -n "$ATTENTION_ACTION" ] && QP="$QP --quantize-attention-action $ATTENTION_ACTION"
    fi
    for nd in $NUM_DEVICES; do
      id="${mid}-${q}-${nd}"
      # aic-npu 白名单模型且 DISABLE 时, 若当前 DEVICE 不是 aic-npu 支持的设备 -> SKIP
      if [ "$q" = "DISABLE" ] && is_aic_npu_model "$m" && [ "$DEVICE" != "$AIC_NPU_DEVICE" ]; then
        skip_test "$id" "$m  ${nd}卡  quantize=$q (aic-npu 不支持设备 $DEVICE, 需在 $AIC_NPU_DEVICE 上跑)"
        continue
      fi
      run_test "$id" "$m  ${nd}卡  quantize=$q" \
        "$BIN" -recommend --model "$m" --num-devices "$nd" $C $X $QP --out-dir "$OUT_DIR/$id"
    done
  done
done

echo "================================================================"
echo " 汇总: PASS=$pass  FAIL=$fail  SKIP=$skip"
echo " 明细见 $SUMMARY"
echo " 各用例日志见 $LOG_DIR/<id>.log"
echo " 各用例结果见 $OUT_DIR/<id>/  (run.json / summary.csv)"
echo "================================================================"
# SKIP 不算失败; 仅有 FAIL 时返回非 0
[ "$fail" -eq 0 ] && exit 0 || exit "$fail"
