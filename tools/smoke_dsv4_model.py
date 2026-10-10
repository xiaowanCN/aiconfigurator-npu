"""Smoke check for the DeepSeekV4Model op graph (no NPU required)."""

from aiconfigurator_npu.sdk import config
from aiconfigurator_npu.sdk.models import check_is_moe, get_model, get_model_family

MODEL = "deepseek-ai/DeepSeek-V4-Pro"

print("family:", get_model_family(MODEL), "is_moe:", check_is_moe(MODEL))

for tp in (1, 2, 4, 8):
    mc = config.ModelConfig(
        tp_size=tp,
        pp_size=1,
        moe_tp_size=1,
        moe_ep_size=tp,
        attention_dp_size=1,
        nextn=1,
        nextn_accept_rates=[0.85, 0.3, 0.0, 0.0, 0.0],
    )
    m = get_model(MODEL, mc, "vllm-ascend")
    print(f"\n=== tp={tp} ===")
    print("  context ops :", [o._name for o in m.context_ops])
    print("  generation  :", [o._name for o in m.generation_ops])
    print("  kv/token    :", m.get_kvcache_elements_per_token())
    print("  kv win/seq  :", m.get_kvcache_sliding_window_elements())
    print("  weights GiB :", round(sum(o.get_weights() for o in m.context_ops) / 2**30, 2))
    print("  mtp scale   :", round(m._mtp_scale_factor, 4))
