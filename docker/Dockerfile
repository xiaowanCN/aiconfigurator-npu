# Ascend NPU runtime image for aiconfigurator-npu
# Base: vllm-ascend official image (CANN 8.5+ / torch-npu / vLLM 0.18.0 / vllm-ascend)
# Usage on 910B: see README.md "Docker Image" section.
FROM quay.io/ascend/vllm-ascend:v0.23.0.post1

WORKDIR /workspace

# Copy only what is needed (keep context small, maximize layer caching)
COPY collector/ ./collector/
COPY tools/ ./tools/
COPY model_configs/ ./model_configs/
COPY src/ ./src/
COPY pyproject.toml README.md ./

# Install the search-engine package (registers the `aic-npu` CLI).
# Pure-Python deps (numpy/pandas/scipy/...) resolve inside the base image.
RUN pip install --no-cache-dir -e .

# Offline mode: GLM-5 model config ships in model_configs/, no HuggingFace Hub access needed.
# PYTHONPATH: collector scripts import bench_engine/*_factory from collector/ root.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=/workspace/collector

# Pre-create benchmark output dir (mount a volume here when running collectors)
RUN mkdir -p /workspace/data

CMD ["/bin/bash"]
