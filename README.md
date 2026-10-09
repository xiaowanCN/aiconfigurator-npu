# AIConfigurator for NPU

Operator microbenchmark collector for Ascend NPU (CANN + vLLM Ascend).

Adapted from [NVIDIA AIConfigurator](https://github.com/ai-dynamo/aiconfigurator) collector design, targeting Ascend NPU with vllm-ascend framework alignment.

## Structure

```
collector/
  bench_engine.py          # NPU Event timing engine with NPU Graph capture + replay
  npu/
    gemm_factory.py        # GEMM operator factory (BF16 / W8A8_DYNAMIC)
    collect_gemm.py        # GEMM microbenchmark collector
    attn_factory.py        # Attention operator factory (Context / Decode)
    collect_attn.py        # Attention microbenchmark collector
    moe_factory.py         # MoE operator factory (BF16 / W8A8_DYNAMIC)
    collect_moe.py         # MoE microbenchmark collector
```

## Features

- NPU Graph capture + replay to eliminate Python dispatch overhead (~30-40us → ~10us/op)
- Real kernel path via vllm-ascend framework layer (not raw torch_npu calls)
- FRACTAL_NZ weight format support for W8A8 quantized models
- 6-op L2 cache flush rotation (GEMM) aligned with AIConfigurator GPU version
- CSV output compatible with TensorCast profiling database format
- Checkpoint/resume support for large parameter sweeps

## Requirements

- Ascend NPU with CANN 8.5+
- vLLM 0.18.0+ with vllm-ascend
- torch-npu

## Quick Start

```bash
# GEMM
python collector/npu/collect_gemm.py --quant-types bf16 w8a8_dynamic --output-dir ./gemm_data

# Attention
python collector/npu/collect_attn.py --op-types context generation --output-dir ./attn_data

# MoE
python collector/npu/collect_moe.py --quant-types bf16 w8a8_dynamic --output-dir ./moe_data
```

## Docker Image

The image is based on the official `quay.io/ascend/vllm-ascend:v0.18.0` base
(CANN 8.5+ / torch-npu / vLLM 0.18.0 / vllm-ascend pre-installed) and contains
`collector/`, `tools/`, `model_configs/` plus the search engine (`aic-npu` CLI).
Offline mode is enabled by default (`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`);
the GLM-5, DeepSeek-V3 and DeepSeek-R1 model configs ship in `model_configs/`.

### Build via GitHub Actions (recommended)

1. Push this repository to GitHub.
2. Configure repository secrets (**Settings → Secrets and variables → Actions**):
   - `QUAY_USERNAME` — your quay.io username
   - `QUAY_PASSWORD` — your quay.io password or Robot Account token
3. Go to **Actions → Build and Push Aiconfigurator NPU Image → Run workflow**,
   enter an image tag (e.g. `v0.1.0` or `latest`), and run.

The workflow builds an **arm64 image** (`linux/arm64`, for Ascend 910B
Kunpeng hosts, QEMU-emulated on x86 runners) and pushes two tags, e.g.:

```
quay.io/18896723947/aiconfigurator-npu:v0.1.0
quay.io/18896723947/aiconfigurator-npu:<short-sha>
```

### Build locally

```bash
docker build -t quay.io/18896723947/aiconfigurator-npu:v0.1.0 -f docker/Dockerfile .
docker push quay.io/18896723947/aiconfigurator-npu:v0.1.0
```

> Note: building on GitHub/x86 runners only installs dependencies; NPU
> functionality must be verified on real hardware (run
> `tools/check_vllm_compat.py` and `npu-smi info` inside the container).

### Run on Ascend 910B

Ascend containers require explicit device and driver mounts:

```bash
docker run -it --name aic-npu \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  --device /dev/davinci0 \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /data/aic_output:/workspace/data \
  --shm-size 32g \
  quay.io/18896723947/aiconfigurator-npu:v0.1.0
```

- Add one `--device /dev/davinciN` per NPU card (davinci0–davinci7 for 8 cards).
- If the host has Ascend Docker Runtime installed, simplify with
  `-e ASCEND_VISIBLE_DEVICES=0 --runtime ascend` instead of manual `--device` mounts.
- Mount a volume at `/workspace/data` to persist benchmark output.

Verify inside the container, then start collecting:

```bash
npu-smi info
python -c "import torch_npu; print(torch_npu.npu.is_available())"
python tools/check_vllm_compat.py

# DSA module collection (GLM-5), see docs/DSA_COLLECTION_COMMANDS.md
python collector/npu/collect_mla_module.py --mode context --quick \
  --batch-size 4 --seq-len 2048 --output-dir /workspace/data/glm5_dsa_module
```

## Model Support

| Model | Architecture | Family | Attention perf data |
|-------|--------------|--------|---------------------|
| `deepseek-ai/DeepSeek-R1` | `DeepseekV3ForCausalLM` | `DEEPSEEK` | `context_mla_perf.txt` / `generation_mla_perf.txt` — pending collection |
| `deepseek-ai/DeepSeek-V3` | `DeepseekV3ForCausalLM` | `DEEPSEEK` | same as R1 |
| `zai-org/GLM-5` | `GlmMoeDsaForCausalLM` | `DEEPSEEKV32` | `dsa_*_module_perf.txt` — pending collection |
| `Qwen/Qwen3-235B-A22B` | `Qwen3MoeForCausalLM` | `MOE` | ready (`*_attention_perf.txt`) |

GEMM / MoE / communication tables for `ascend_910b` are already in
`src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.18.0/`.

DeepSeek-R1 reuses `DeepSeekModel` (same architecture and dimensions as V3), so no
model code changes are needed — see `docs/DeepSeek-R1_ADAPTATION.md` for the full
onboarding checklist and MLA collection commands.

Until the MLA tables are collected, run the search in HYBRID mode so the attention
ops fall back to the analytic (SOL) model:

```bash
aic-npu default --model deepseek-ai/DeepSeek-R1 --system ascend_910b \
  --backend vllm --database-mode hybrid --total-gpus 16
```
