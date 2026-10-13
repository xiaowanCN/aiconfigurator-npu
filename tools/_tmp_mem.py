import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aiconfigurator_npu.sdk import common, config, models, utils  # noqa: E402
from aiconfigurator_npu.sdk.backends.trtllm_backend import TRTLLMBackend  # noqa: E402
from aiconfigurator_npu.sdk.perf_database import PerfDatabase  # noqa: E402

MODEL = "model_configs/Qwen--Qwen3-32B_config.json"
db = PerfDatabase("ascend_910b", "vllm-ascend", "0.23.0", "src/aiconfigurator_npu/systems")
mem_cap_gib = db.system_spec["gpu"]["mem_capacity"] / (1 << 30)
print(f"mem_capacity = {mem_cap_gib:.1f} GiB")

backend = TRTLLMBackend()

# 1) correct per-tp rebuild
print("=== per-tp memory (model rebuilt with correct tp) ===")
for tp in (1, 2, 4, 8, 16, 32):
    mc = config.ModelConfig()
    mc.tp_size = tp
    model = models.get_model(MODEL, mc, common.BackendName.vllm.value)
    mem = backend._get_memory_usage(model, db, batch_size=1, beam_width=0, isl=4096, osl=1024, num_tokens=4096)
    oom = mem["total"] >= mem_cap_gib
    print(f"tp={tp:2d} weights={mem['weights']:7.2f} total={mem['total']:7.2f} OOM={oom} family={model.model_family}")

# 2) enumerate agg configs at total_gpus=64 (mirror pareto_analysis)
print("=== agg parallel configs @64 GPUs, tp-family check ===")
pcfg = utils.enumerate_parallel_config(
    num_gpu_list=[1, 2, 4, 8, 16, 32, 64],
    tp_list=[1, 2, 4, 8, 16, 32, 64],
    pp_list=[1],
    dp_list=[1, 2, 4, 8, 16, 32, 64],
    moe_tp_list=[1],
    moe_ep_list=[1],
    is_moe=False,
    backend=common.BackendName.vllm,
)
kept = []
for tp, pp, dp, mtp, mep in pcfg:
    if tp * pp * dp > 64:
        continue
    kept.append((tp, pp, dp, mtp, mep))
print(f"configs with tp*pp*dp<=64: {len(kept)}")
oom_count = 0
for tp, pp, dp, mtp, mep in kept:
    mc = config.ModelConfig()
    mc.tp_size = tp
    model = models.get_model(MODEL, mc, common.BackendName.vllm.value)
    mem = backend._get_memory_usage(model, db, batch_size=1, beam_width=0, isl=4096, osl=1024, num_tokens=4096)
    if mem["total"] >= mem_cap_gib:
        oom_count += 1
print(f"OOM configs: {oom_count}/{len(kept)}")
# show the non-OOM tp values available
non_oom_tps = set()
for tp, pp, dp, mtp, mep in kept:
    mc = config.ModelConfig()
    mc.tp_size = tp
    model = models.get_model(MODEL, mc, common.BackendName.vllm.value)
    mem = backend._get_memory_usage(model, db, 1, 0, 4096, 1024, num_tokens=4096)
    if mem["total"] < mem_cap_gib:
        non_oom_tps.add(tp)
print(f"non-OOM tp values: {sorted(non_oom_tps)}")
