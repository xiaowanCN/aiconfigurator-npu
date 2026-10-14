#!/usr/bin/env bash
# ============================================================================
# 三模型性能数据「从头全量采集」一键脚本
#   DeepSeek-R1 / DeepSeek-V4-Pro / Qwen3.8-27B
#   系统： ascend_910b / vllm-ascend / 0.23.0
#
# 依据文档：docs/DATA_FROM_SCRATCH_3_MODELS.md（§2 环境 / §4 采集 / §5 落位 / §6 校验）
# 覆盖范围：文档 §4.1、§4.3 ~ §4.8 的全部采集点
#   §4.1 gemm_perf.txt                                  （三模型共用）
#   §4.3 moe_perf.txt                                   （R1 + V4）
#   §4.4 context_mla / generation_mla / mla_bmm         （R1）
#   §4.5 dsv4_csa_* / dsv4_hca_*                        （V4-Pro）
#   §4.6 mhc_module_perf.txt                            （V4-Pro）
#   §4.7 context_attention / generation_attention hd256 （Qwen3.8-27B）
#   §4.8 gdn_perf.txt                                   （Qwen3.8-27B）
# 不含：§4.2 custom_allreduce_perf.txt（三模型共用，按需求排除）
#
# 所有采集产物写入 staging/，与 0.23.0/ 隔离；默认不做落位（merge），
# 需要落位时显式加 LANDING=1（会先备份 0.23.0/）。
#
# ---------------------------------------------------------------------------
# 运行命令（NPU 机器、任意目录，脚本自己 cd 到仓库根目录）：
# 完整说明见 docs/COLLECT_3MODELS_FROM_SCRATCH_RUNBOOK.md
#
#   1) 常规全量采集（后台跑 + 另开终端 tail 看日志）：
#        nohup bash tools/collect_3models_from_scratch.sh &
#        tail -f staging/collect_3models_log.txt
#
#   2) 前台跑（脚本内部已 tee：终端实时看 + 同步落日志）：
#        bash tools/collect_3models_from_scratch.sh
#      断连风险的机器建议套 tmux/screen：
#        tmux new -s collect 'bash tools/collect_3models_from_scratch.sh'
#      （各采集器均带 --resume，中断后原样重跑即可续采）
#
#   3) 先冒烟（GDN 2 点 + attention hd256 1 点），确认 kernel 可用再全采：
#        SMOKE=1 bash tools/collect_3models_from_scratch.sh
#
#   4) 只跑某些阶段（阶段名子串匹配，逗号分隔，忽略大小写）：
#        ONLY="moe,mla" bash tools/collect_3models_from_scratch.sh
#        ONLY="gdn"     bash tools/collect_3models_from_scratch.sh
#
#   5) 跳过某些阶段：
#        SKIP="gemm" bash tools/collect_3models_from_scratch.sh
#
#   6) 采集完立即落位（merge 进 0.23.0 + 校验）：
#        LANDING=1 bash tools/collect_3models_from_scratch.sh
#
#   7) 只看计划不执行：
#        DRY_RUN=1 bash tools/collect_3models_from_scratch.sh
#
# 可选环境变量：
#   STG        staging 根目录                默认 $REPO/staging
#   DATA_DIR   落位目标目录                   默认 src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.23.0
#   LOG        整体日志文件                   默认 $STG/collect_3models_log.txt
#   LOG_APPEND=1  追加到旧日志（默认每次新建）
#   OPP_PATH   vendor 自定义算子包路径         默认 /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
# ============================================================================
set -u

# ---------------------------------------------------------------------------
# 0. 路径与环境
# ---------------------------------------------------------------------------
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

DATA_DIR="${DATA_DIR:-$REPO/src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.23.0}"
STG="${STG:-$REPO/staging}"
OUT="$STG/out"
LOG="${LOG:-$STG/collect_3models_log.txt}"
OPP_PATH="${OPP_PATH:-/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer}"

# vendor 自定义算子包必须在加载路径上（DSv4 / GDN 依赖）
export ASCEND_CUSTOM_OPP_PATH="$OPP_PATH"
export LD_LIBRARY_PATH="$OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO/collector:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1          # 配合 tee 实时看到采集进度
export PYTHONFAULTHANDLER=1

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null

mkdir -p "$OUT" "$STG"/{gemm,moe_r1,moe_v4,mla_r1_ctx,mla_r1_gen,mla_bmm_r1,\
dsv4_csa_ctx,dsv4_csa_gen,dsv4_hca_ctx,dsv4_hca_gen,mhc,attn_hd256,gdn} 2>/dev/null
mkdir -p "$(dirname "$LOG")"
if [[ "${LOG_APPEND:-0}" != "1" ]]; then : > "$LOG"; fi

PY=("${PYTHON:-python}" -u)

# ---------------------------------------------------------------------------
# 1. 日志 / 阶段进度工具
# ---------------------------------------------------------------------------
say() { printf '%s\n' "$*" | tee -a "$LOG"; }

# 把采集器直出的 *_perf.txt 拷进 staging/out；缺失只告警，不改变阶段返回码
copy_to_out() {
  if [[ -f "$1" ]]; then
    cp "$1" "$OUT/$2" && say "  -> $OUT/$2"
  else
    say "  WARN: 未产出 $1"
  fi
  return 0
}

# 执行一条命令：stdout+stderr 同时进 LOG 和终端（tee），返回命令真实退出码
# 需要切换工作目录时，先设置 RUN_CWD（如 collect_attn.py / collect_gdn.py 需在
# collector/npu 下运行）
RUN_CWD=""
run_cmd() {
  local t0=$SECONDS rc cwd="$RUN_CWD"
  if [[ -n "$cwd" ]]; then
    say "\$ (cd $cwd && $*)"
    ( cd "$cwd" && "$@" ) 2>&1 | tee -a "$LOG"
  else
    say "\$ $*"
    "$@" 2>&1 | tee -a "$LOG"
  fi
  rc=${PIPESTATUS[0]}
  CMD_RC=$rc
  CMD_SECS=$((SECONDS - t0))
  return $rc
}

say "=============================================================================="
say "三模型全量采集开始  $(date '+%F %T')"
say "  仓库根目录 : $REPO"
say "  staging    : $STG"
say "  落位目录   : $DATA_DIR (仅 LANDING=1 时写入)"
say "  日志文件   : $LOG"
say "=============================================================================="

# ---------------------------------------------------------------------------
# 2. 各采集阶段（顺序按文档 §7 的推荐执行顺序：先难后易）
# ---------------------------------------------------------------------------

# ---- §4.8 gdn_perf.txt（Qwen3.8-27B，新采集器，风险最高，先验证） ----------
stage_gdn() {
  local rc
  RUN_CWD="$REPO/collector/npu"
  run_cmd "${PY[@]}" collect_gdn.py \
    --model-name qwen3_8_27B \
    --d-model 5120 --d-conv 4 \
    --num-k-heads 16 --head-k-dim 128 --num-v-heads 48 --head-v-dim 128 \
    --tp-sizes 1 2 4 8 \
    --max-context-tokens 1000000 \
    --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
    --output-dir "$STG/gdn" --timing-mode graph --resume
  rc=$?
  RUN_CWD=""
  copy_to_out "$STG/gdn/gdn_perf.txt" "gdn_perf.txt"
  return $rc
}

stage_smoke_gdn() {
  RUN_CWD="$REPO/collector/npu"
  run_cmd "${PY[@]}" collect_gdn.py \
    --tp-sizes 1 --warmup-iters 1 --bench-iters 3 --timing-mode graph \
    --output-dir "$STG/gdn_smoke"
  local rc=$?
  RUN_CWD=""
  say "  冒烟自检：延迟应随 seq_len / batch_size / (nk,nv) 单调变化；"
  say "  若不同头数的值几乎相同，说明测到的是主机开销而非 kernel 时间。"
  return $rc
}

# ---- §4.5 DSv4 CSA / HCA（V4-Pro；四个组合各用独立 output-dir） -----------
stage_dsv4_csa_ctx() {
  run_cmd "${PY[@]}" collector/npu/collect_dsv4_attn.py \
    --op-types dsv4_context --attn-kinds csa \
    --batch-list 1 2 4 8 16 32 \
    --seq-len-list 128 256 512 1024 2048 4096 8192 16384 \
    --num-heads-list 128 64 32 16 8 4 \
    --module-source framework \
    --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
    --output-dir "$STG/dsv4_csa_ctx" --timing-mode graph --resume
}

stage_dsv4_csa_gen() {
  run_cmd "${PY[@]}" collector/npu/collect_dsv4_attn.py \
    --op-types dsv4_generation --attn-kinds csa \
    --batch-list 1 2 4 8 16 32 64 128 256 384 512 \
    --seq-len-list 128 256 512 1024 2048 4096 8192 16384 \
    --num-heads-list 128 64 32 16 8 4 \
    --module-source framework \
    --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
    --output-dir "$STG/dsv4_csa_gen" --timing-mode graph --resume
}

stage_dsv4_hca_ctx() {
  run_cmd "${PY[@]}" collector/npu/collect_dsv4_attn.py \
    --op-types dsv4_context --attn-kinds hca \
    --batch-list 1 2 4 8 16 32 \
    --seq-len-list 128 256 512 1024 2048 4096 8192 16384 \
    --num-heads-list 128 64 32 16 8 4 \
    --module-source framework \
    --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
    --output-dir "$STG/dsv4_hca_ctx" --timing-mode graph --resume
}

stage_dsv4_hca_gen() {
  run_cmd "${PY[@]}" collector/npu/collect_dsv4_attn.py \
    --op-types dsv4_generation --attn-kinds hca \
    --batch-list 1 2 4 8 16 32 64 128 256 384 512 \
    --seq-len-list 128 256 512 1024 2048 4096 8192 16384 \
    --num-heads-list 128 64 32 16 8 4 \
    --module-source framework \
    --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
    --output-dir "$STG/dsv4_hca_gen" --timing-mode graph --resume
}

stage_dsv4_convert() {
  local d
  for d in dsv4_csa_ctx dsv4_csa_gen dsv4_hca_ctx dsv4_hca_gen; do
    run_cmd "${PY[@]}" tools/convert_to_aiconfigurator.py \
      --input-dir "$STG/$d" --output-dir "$OUT" \
      --device "Ascend 910B" --framework vllm-ascend --version 0.23.0 \
      --dsv4-output
  done
}

# ---- §4.6 mhc_module_perf.txt（V4-Pro） -----------------------------------
stage_mhc() {
  local rc
  run_cmd "${PY[@]}" collector/npu/collect_mhc.py \
    --op-types pre post \
    --num-tokens-list 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 \
    --hc-mult 4 --hidden-size 7168 --sinkhorn-iters 20 \
    --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
    --output-dir "$STG/mhc" --timing-mode graph --resume
  rc=$?
  copy_to_out "$STG/mhc/mhc_module_perf.txt" "mhc_module_perf.txt"
  return $rc
}

# ---- §4.3 moe_perf.txt（R1 + V4，共享表） ---------------------------------
stage_moe_r1() {
  run_cmd "${PY[@]}" collector/npu/collect_moe.py \
    --models deepseek-v3 deepseek-v3-ep2 deepseek-v3-ep4 deepseek-v3-ep16 \
            deepseek-v3-ep32 deepseek-v3-ep64 \
    --token-list 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 \
    --quant-types bf16 --output-dir "$STG/moe_r1" --timing-mode graph --resume
}

stage_moe_v4() {
  run_cmd "${PY[@]}" collector/npu/collect_moe.py \
    --models deepseek-v4-pro deepseek-v4-pro-ep2 deepseek-v4-pro-ep4 \
            deepseek-v4-pro-ep16 deepseek-v4-pro-ep32 deepseek-v4-pro-ep64 \
    --token-list 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 \
    --quant-types bf16 --output-dir "$STG/moe_v4" --timing-mode graph --resume
}

stage_moe_convert() {
  # R1 先建表（只写 staging，不碰 0.23.0）
  run_cmd "${PY[@]}" tools/convert_to_aiconfigurator.py \
    --input-dir "$STG/moe_r1" --output-dir "$OUT" \
    --device "Ascend 910B" --framework vllm-ascend --version 0.23.0 \
    --moe-distribution both
  # V4 追加到同一份 staging 表
  run_cmd "${PY[@]}" tools/convert_to_aiconfigurator.py \
    --input-dir "$STG/moe_v4" --output-dir "$OUT" \
    --device "Ascend 910B" --framework vllm-ascend --version 0.23.0 \
    --moe-distribution both --append
}

# ---- §4.4 MLA（R1） --------------------------------------------------------
stage_mla_ctx() {
  local rc
  run_cmd "${PY[@]}" collector/npu/collect_mla.py --output-format mla \
    --architecture DeepseekV3ForCausalLM --model deepseek-ai/DeepSeek-R1 \
    --op-types context \
    --batch-list 1 2 4 8 16 32 64 128 256 \
    --seq-len-list 128 256 512 1024 2048 4096 8192 16384 32768 \
    --num-heads-list 128 64 32 16 8 4 \
    --kv-lora-rank 512 --qk-nope-head-dim 128 --qk-rope-head-dim 64 --v-head-dim 128 \
    --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
    --mla-dtype float16 --kv-cache-dtype float16 \
    --output-dir "$STG/mla_r1_ctx" --timing-mode graph --resume
  rc=$?
  copy_to_out "$STG/mla_r1_ctx/context_mla_perf.txt" "context_mla_perf.txt"
  return $rc
}

stage_mla_gen() {
  local rc
  run_cmd "${PY[@]}" collector/npu/collect_mla.py --output-format mla \
    --architecture DeepseekV3ForCausalLM --model deepseek-ai/DeepSeek-R1 \
    --op-types generation \
    --batch-list 1 2 4 8 16 32 64 128 256 \
    --seq-len-list 128 256 512 1024 2048 4096 8192 16384 32768 \
    --num-heads-list 128 64 32 16 8 4 \
    --kv-lora-rank 512 --qk-nope-head-dim 128 --qk-rope-head-dim 64 --v-head-dim 128 \
    --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
    --mla-dtype float16 --kv-cache-dtype float16 \
    --output-dir "$STG/mla_r1_gen" --timing-mode graph --resume
  rc=$?
  copy_to_out "$STG/mla_r1_gen/generation_mla_perf.txt" "generation_mla_perf.txt"
  return $rc
}

stage_mla_bmm() {
  local rc
  run_cmd "${PY[@]}" collector/npu/collect_mla_bmm.py \
    --num-tokens-list 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 \
    --num-heads-list 128 64 32 16 8 4 \
    --kv-lora-rank 512 --head-dim 128 \
    --op-types pre post \
    --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
    --bmm-dtype float16 --output-dir "$STG/mla_bmm_r1" --timing-mode graph --resume
  rc=$?
  copy_to_out "$STG/mla_bmm_r1/mla_bmm_perf.txt" "mla_bmm_perf.txt"
  return $rc
}

# ---- §4.7 dense attention head_dim=256（Qwen3.8-27B，共享表） --------------
stage_smoke_attn_hd256() {
  RUN_CWD="$REPO/collector/npu"
  run_cmd "${PY[@]}" collect_attn.py \
    --head-size 256 --num-heads-list 24 --num-kv-heads-list 4 \
    --seq-len-list 4096 --timing-mode graph --output-dir "$STG/attn_hd256_smoke"
  local rc=$?
  RUN_CWD=""
  say "  冒烟自检：确认 910B FIA 支持 head_dim=256（部分实现上限 128/192）。"
  return $rc
}

stage_attn_hd256() {
  RUN_CWD="$REPO/collector/npu"
  run_cmd "${PY[@]}" collect_attn.py \
    --op-types context generation \
    --head-size 256 \
    --num-heads-list 24 12 6 3 \
    --num-kv-heads-list 4 2 1 \
    --batch-list 1 2 4 8 16 32 \
    --seq-len-list 128 256 512 1024 2048 4096 8192 16384 \
    --output-dir "$STG/attn_hd256" --timing-mode graph --resume
  local rc=$?
  RUN_CWD=""
  return $rc
}

stage_attn_convert() {
  run_cmd "${PY[@]}" tools/convert_to_aiconfigurator.py \
    --input-dir "$STG/attn_hd256" --output-dir "$OUT" \
    --device "Ascend 910B" --framework vllm-ascend --version 0.23.0
}

# ---- §4.1 gemm_perf.txt（三模型共用，共享表） ------------------------------
stage_gemm() {
  run_cmd "${PY[@]}" collector/npu/collect_gemm.py \
    --quant-types bf16 \
    --m-list 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 \
    --n-list 2048 3072 4096 5120 7168 8192 12288 16384 17408 \
    --k-list 2048 3072 4096 5120 7168 8192 12288 16384 17408 \
    --output-dir "$STG/gemm" --timing-mode graph --resume
}

stage_gemm_convert() {
  run_cmd "${PY[@]}" tools/convert_to_aiconfigurator.py \
    --input-dir "$STG/gemm" --output-dir "$OUT" \
    --device "Ascend 910B" --framework vllm-ascend --version 0.23.0
}

# ---- §5 落位（merge，不是 append；默认关闭，需 LANDING=1） ----------------
stage_landing_merge() {
  local bak="/tmp/0.23.0.bak.$(date +%s)"
  cp -r "$DATA_DIR" "$bak" && say "  已备份 $DATA_DIR -> $bak"
  local t
  # 注意：不含 custom_allreduce_perf.txt（本次未采集 §4.2）
  for t in gemm_perf moe_perf \
           context_mla_perf generation_mla_perf mla_bmm_perf \
           dsv4_csa_context_module_perf dsv4_csa_generation_module_perf \
           dsv4_hca_context_module_perf dsv4_hca_generation_module_perf \
           mhc_module_perf context_attention_perf generation_attention_perf gdn_perf; do
    local src="$OUT/$t.txt"
    if [[ ! -f "$src" ]]; then say "  skip (staging 无此表): $t"; continue; fi
    run_cmd "${PY[@]}" tools/merge_perf_tables.py "$DATA_DIR/$t.txt" "$src" --in-place
  done
}

# ---- §6 落位后校验 ---------------------------------------------------------
stage_validate() {
  run_cmd "${PY[@]}" tools/check_model_readiness.py \
    model_configs/deepseek-ai--DeepSeek-R1_config.json \
    model_configs/deepseek-ai--DeepSeek-V4-Pro_config.json \
    model_configs/qwen3_8_27B_config.json \
    --system ascend_910b --backend vllm-ascend --version 0.23.0
  run_cmd "${PY[@]}" tools/check_dsv4_data.py --isl 4096 --osl 1024
}

# ---------------------------------------------------------------------------
# 3. 组装执行计划（阶段名 -> 函数）
# ---------------------------------------------------------------------------
declare -a PLAN_FUNC=()
declare -a PLAN_NAME=()
add_stage() { PLAN_FUNC+=("$1"); PLAN_NAME+=("$2"); }

if [[ "${SMOKE:-0}" == "1" ]]; then
  add_stage stage_smoke_gdn         "冒烟: GDN(2点)"
  add_stage stage_smoke_attn_hd256  "冒烟: attention head_dim=256"
fi

# 文档 §7 推荐顺序：1 GDN → 2 DSv4+mHC → 3 MoE → 4 MLA → 5 attention → 6 GEMM
add_stage stage_gdn             "§4.8 gdn_perf (Qwen3.8-27B)"
add_stage stage_dsv4_csa_ctx    "§4.5 dsv4 CSA context (V4-Pro)"
add_stage stage_dsv4_csa_gen    "§4.5 dsv4 CSA generation (V4-Pro)"
add_stage stage_dsv4_hca_ctx    "§4.5 dsv4 HCA context (V4-Pro)"
add_stage stage_dsv4_hca_gen    "§4.5 dsv4 HCA generation (V4-Pro)"
add_stage stage_dsv4_convert    "§4.5 dsv4 四表转换 -> staging/out"
add_stage stage_mhc             "§4.6 mhc_module_perf (V4-Pro)"
add_stage stage_moe_r1          "§4.3 MoE R1 (ep1..64)"
add_stage stage_moe_v4          "§4.3 MoE V4-Pro (ep1..64)"
add_stage stage_moe_convert     "§4.3 MoE 转换 -> staging/out"
add_stage stage_mla_ctx         "§4.4 context_mla (R1)"
add_stage stage_mla_gen         "§4.4 generation_mla (R1)"
add_stage stage_mla_bmm         "§4.4 mla_bmm (R1)"
add_stage stage_attn_hd256      "§4.7 attention head_dim=256 (Qwen3.8-27B)"
add_stage stage_attn_convert    "§4.7 attention 转换 -> staging/out"
add_stage stage_gemm            "§4.1 gemm (三模型共用)"
add_stage stage_gemm_convert    "§4.1 gemm 转换 -> staging/out"

if [[ "${LANDING:-0}" == "1" ]]; then
  add_stage stage_landing_merge "§5 落位 merge -> 0.23.0"
  add_stage stage_validate      "§6 落位后校验"
fi

# 过滤：ONLY / SKIP（阶段名子串匹配，忽略大小写）
match_any() {            # match_any <阶段名> <逗号分隔的关键字串>
  local name_lc tok hit=0
  name_lc="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  local IFS=','
  for tok in $2; do
    tok="$(printf '%s' "$tok" | tr '[:upper:]' '[:lower:]' | tr -d ' ')"
    [[ -z "$tok" ]] && continue
    [[ "$name_lc" == *"$tok"* ]] && { hit=1; break; }
  done
  return $((1 - hit))
}

declare -a FUNC=()
declare -a NAME=()
for ((i = 0; i < ${#PLAN_FUNC[@]}; i++)); do
  n="${PLAN_NAME[i]}"
  if [[ -n "${ONLY:-}" ]] && ! match_any "$n" "${ONLY}"; then continue; fi
  if [[ -n "${SKIP:-}" ]] && match_any "$n" "${SKIP}"; then continue; fi
  FUNC+=("${PLAN_FUNC[i]}"); NAME+=("$n")
done

TOTAL=${#FUNC[@]}
say "执行计划共 ${TOTAL} 个阶段："
for ((i = 0; i < TOTAL; i++)); do say "  $((i + 1)). ${NAME[i]}"; done
say "=============================================================================="

if [[ "${DRY_RUN:-0}" == "1" ]]; then say "DRY_RUN=1，仅打印计划，退出。"; exit 0; fi
if (( TOTAL == 0 )); then say "没有匹配的阶段（检查 ONLY/SKIP），退出。"; exit 1; fi

# ---------------------------------------------------------------------------
# 4. 逐阶段执行 + 进度日志
# ---------------------------------------------------------------------------
declare -a RC=()
declare -a SECS=()
SCRIPT_T0=$SECONDS

for ((i = 0; i < TOTAL; i++)); do
  idx=$((i + 1))
  remain_n=$((TOTAL - idx))
  remain_list=""
  for ((j = idx; j < TOTAL; j++)); do
    remain_list+="${NAME[j]}"
    ((j < TOTAL - 1)) && remain_list+=" | "
  done
  [[ -z "$remain_list" ]] && remain_list="（无，这是最后一个阶段）"

  say ""
  say "=============================================================================="
  say "[${idx}/${TOTAL}] $(date '+%F %T') >>> 当前阶段：${NAME[i]}"
  say "  已完成 ${i} / 共 ${TOTAL}；剩余 ${remain_n} 个待采集：${remain_list}"
  say "  已耗时 $(( (SECONDS - SCRIPT_T0) / 60 )) 分钟"
  say "------------------------------------------------------------------------------"

  "${FUNC[i]}"
  rc=$?

  RC+=("$rc"); SECS+=("${CMD_SECS:-0}")
  if (( rc == 0 )); then
    say "[${idx}/${TOTAL}] <<< 完成：${NAME[i]}（耗时 ${CMD_SECS}s）"
  else
    say "[${idx}/${TOTAL}] <<< 失败 rc=${rc}：${NAME[i]}（耗时 ${CMD_SECS}s）—— 继续下一个阶段"
  fi
done

# ---------------------------------------------------------------------------
# 5. 汇总
# ---------------------------------------------------------------------------
say ""
say "=============================================================================="
say "全部阶段结束 $(date '+%F %T')，总耗时 $(( (SECONDS - SCRIPT_T0) / 60 )) 分钟"
say "------------------------------------------------------"
fail=0
for ((i = 0; i < TOTAL; i++)); do
  st="OK"
  (( RC[i] != 0 )) && { st="FAILED rc=${RC[i]}"; fail=$((fail + 1)); }
  say "  $(printf '%-2d' $((i + 1))). ${NAME[i]}  ... ${st} (${SECS[i]}s)"
done
say "------------------------------------------------------"
say "失败阶段数：${fail} / ${TOTAL}"
say "staging 干净表目录：$OUT"
say "  当前产出："
for f in "$OUT"/*.txt; do [[ -f "$f" ]] && say "    $(wc -l < "$f") 行  $f"; done
say "整体日志：$LOG"
if [[ "${LANDING:-0}" != "1" ]]; then
  say "提示：本次只采集未落位。确认无误后执行落位（先备份再 merge）："
  say "  LANDING=1 bash tools/collect_3models_from_scratch.sh"
fi
say "=============================================================================="

(( fail == 0 )) && exit 0
exit 1
