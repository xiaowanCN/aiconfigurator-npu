#!/usr/bin/env python3
"""Inspect staging/out granularity and per-model readiness (replicates
tools/check_model_readiness.py but against an arbitrary data_dir)."""
import csv
import os
import sys

STAGING_OUT = r"D:\Projects\ExpProjects\GITProject\aic-npu-cn-ascend\staging-1014\1014\staging\out"
REPO = r"D:\Projects\ExpProjects\GITProject\aic-npu-cn-ascend\aiconfigurator-npu"
sys.path.insert(0, os.path.join(REPO, "src"))

from aiconfigurator_npu.sdk.utils import get_model_config_from_model_path  # noqa: E402

TABLES = {
    "gemm": ("gemm_perf.txt", ("gemm_dtype",)),
    "attention": ("context_attention_perf.txt", ("num_heads", "head_dim", "attn_dtype")),
    "attention_gen": ("generation_attention_perf.txt", ("num_heads", "head_dim")),
    "mla": ("context_mla_perf.txt", ("num_heads", "mla_dtype", "kv_cache_dtype")),
    "mla_gen": ("generation_mla_perf.txt", ("num_heads", "mla_dtype")),
    "mla_bmm": ("mla_bmm_perf.txt", ("num_heads", "bmm_dtype")),
    "moe": ("moe_perf.txt", ("moe_dtype", "topk", "num_experts", "hidden_size", "inter_size", "moe_ep_size", "distribution")),
    "dsv4_csa": ("dsv4_csa_context_module_perf.txt", ("num_heads",)),
    "dsv4_hca": ("dsv4_hca_context_module_perf.txt", ("num_heads",)),
    "dsv4_gen": ("dsv4_csa_generation_module_perf.txt", ("num_heads",)),
    "mhc": ("mhc_module_perf.txt", ("hc_mult",)),
    "gdn": ("gdn_perf.txt", ()),
}
FAMILY_REQUIREMENTS = {
    "DEEPSEEK": ["gemm", "mla", "mla_gen", "mla_bmm", "moe"],
    "DEEPSEEKV4": ["gemm", "dsv4_csa", "dsv4_hca", "dsv4_gen", "mhc", "moe"],
    "QWEN35": ["gemm", "attention", "attention_gen", "gdn"],
}
FAMILY_ATTENTION = {"DEEPSEEK": "mla", "DEEPSEEKV4": "dsv4", "QWEN35": "dense"}
ARCH_TO_FAMILY = {
    "DeepseekV3ForCausalLM": "DEEPSEEK",
    "DeepseekV4ForCausalLM": "DEEPSEEKV4",
    "Qwen3_5ForConditionalGeneration": "QWEN35",
}
MODELS = [
    os.path.join(REPO, "model_configs", "deepseek-ai--DeepSeek-R1_config.json"),
    os.path.join(REPO, "model_configs", "deepseek-ai--DeepSeek-V4-Pro_config.json"),
    os.path.join(REPO, "model_configs", "qwen3_8_27B_config.json"),
]


def _read_rows(path):
    if not os.path.isfile(path):
        return None
    with open(path, newline="", encoding="utf-8") as h:
        return list(csv.DictReader(h))


def _int_values(rows, col):
    out = set()
    for r in rows:
        if col not in r:
            continue
        try:
            out.add(int(float(r[col])))
        except (TypeError, ValueError):
            pass
    return sorted(out)


def _all_values(rows, col):
    return sorted({r[col] for r in rows if r.get(col)})


def _powers(max_tp):
    v, out = 1, []
    while v <= max_tp:
        out.append(v)
        v *= 2
    return out


def check(model_path, data_dir, max_tp=64):
    info = get_model_config_from_model_path(model_path)
    arch = info.get("architecture") or "?"
    family = ARCH_TO_FAMILY.get(arch)
    if family is None:
        from aiconfigurator_npu.sdk import models as m
        family = m.get_model_family(model_path)
    model = {
        "heads": int(info.get("n", 0)),
        "head_dim": int(info.get("d", 0)),
        "topk": int(info.get("topk", 0)),
        "experts": int(info.get("num_experts", 0)),
        "hidden": int(info.get("hidden_size", 0)),
        "moe_inter": int(info.get("moe_inter_size", 0)),
    }
    rep = {"missing": [], "blocking": [], "warnings": [], "grids": {}}
    for t in FAMILY_REQUIREMENTS[family]:
        fn, cols = TABLES[t]
        rows = _read_rows(os.path.join(data_dir, fn))
        if rows is None:
            rep["missing"].append(fn)
            rep["blocking"].append(f"{family} requires {fn} (not collected)")
            continue
        grid = {c: _int_values(rows, c) or _all_values(rows, c) for c in cols}
        rep["grids"][t] = {c: grid[c] for c in list(grid)[:6]}
        rep["grids"][t]["__rows"] = len(rows)
    kind = FAMILY_ATTENTION[family]
    if kind == "mla":
        g = rep["grids"].get("mla", {}).get("num_heads", [])
        sup = [tp for tp in _powers(max_tp) if model["heads"] % tp == 0 and model["heads"] // tp in g]
        un = [tp for tp in _powers(max_tp) if model["heads"] % tp == 0 and model["heads"] // tp not in g]
        if not sup:
            rep["blocking"].append(f"MLA no priceable tp (heads={model['heads']}, grid={g})")
        elif un:
            rep["warnings"].append(f"MLA tp {un} unpriceable (need rank-local heads {sorted({model['heads']//tp for tp in un})}, have {g})")
        rep["grids"]["mla_supported_tp"] = sup
    elif kind == "dsv4":
        for t in ("dsv4_csa", "dsv4_hca"):
            g = rep["grids"].get(t, {}).get("num_heads", [])
            if not g:
                continue
            miss = [tp for tp in _powers(max_tp) if model["heads"] % tp == 0 and model["heads"] // tp not in g]
            if miss:
                rep["warnings"].append(f"{t}: rank-local heads for tp {miss} missing (have {g})")
    elif kind == "dense":
        hg = rep["grids"].get("attention", {}).get("num_heads", [])
        dims = rep["grids"].get("attention", {}).get("head_dim", [])
        if model["head_dim"] and dims and model["head_dim"] not in dims:
            rep["blocking"].append(f"dense head_dim needed {model['head_dim']} but collected {dims}")
        if model["heads"] and hg:
            sup = [tp for tp in _powers(max_tp) if model["heads"] % tp == 0 and model["heads"] // tp in hg]
            un = [tp for tp in _powers(max_tp) if model["heads"] % tp == 0 and model["heads"] // tp not in hg]
            if un:
                rep["warnings"].append(f"dense tp {un} unpriceable (need rank-local {sorted({model['heads']//tp for tp in un})}, have {hg})")
            rep["grids"]["attention_allowed_tp"] = sup
            bad = [tp for tp in _powers(max_tp) if model["heads"] % tp != 0]
            if bad:
                rep["warnings"].append(f"tp {bad} rejected: not divisor of heads={model['heads']}")
    if "moe" in FAMILY_REQUIREMENTS[family] and model["experts"]:
        rows = _read_rows(os.path.join(data_dir, "moe_perf.txt")) or []
        m = [r for r in rows if int(r.get("num_experts", 0)) == model["experts"]
             and int(r.get("topk", 0)) == model["topk"]
             and int(r.get("hidden_size", 0)) == model["hidden"]
             and int(r.get("inter_size", 0)) == model["moe_inter"]]
        ep = _int_values(m, "moe_ep_size")
        dist = _all_values(m, "distribution")
        rep["grids"]["moe_shape"] = {"rows": len(m), "ep": ep, "dist": dist}
        if not m:
            rep["blocking"].append(f"MoE no row for experts={model['experts']} topk={model['topk']} hidden={model['hidden']} inter={model['moe_inter']}")
        elif len(ep) <= 1:
            rep["warnings"].append(f"MoE only ep={ep}; dp*tp==moe_ep pins worker width")
        if m and "attention_allowed_tp" in rep["grids"]:
            atp = rep["grids"]["attention_allowed_tp"]
        elif m and "mla_supported_tp" in rep["grids"]:
            atp = rep["grids"]["mla_supported_tp"]
        else:
            atp = None
        if atp:
            vw = [w for w in ep if any(w % tp == 0 for tp in atp)]
            rep["grids"]["viable_worker_widths"] = vw
            if not vw:
                rep["blocking"].append(f"no viable worker width: attn tp {atp}, moe ep {ep}")
    return arch, family, model, rep


def main():
    print("=" * 100)
    print(f"STAGING_OUT = {STAGING_OUT}")
    print("=" * 100)
    # per-table summary
    print("\n### Table row counts & key grids (staging/out) ###")
    for t, (fn, cols) in TABLES.items():
        rows = _read_rows(os.path.join(STAGING_OUT, fn))
        if rows is None:
            print(f"  [MISSING] {fn}")
            continue
        summary = {c: _int_values(rows, c) or _all_values(rows, c) for c in cols}
        print(f"  {fn:42s} rows={len(rows):5d} grids={ {c: (v[:12] + ['...'] if len(v) > 12 else v) for c, v in summary.items()} }")
    # per-model readiness
    print("\n### Per-model readiness against staging/out ###")
    total_block = 0
    for mp in MODELS:
        arch, family, model, rep = check(mp, STAGING_OUT)
        status = "BLOCKED" if rep["blocking"] else ("DEGRADED" if rep["warnings"] else "READY")
        total_block += bool(rep["blocking"])
        print(f"\n[{status}] {os.path.basename(mp)}  arch={arch} family={family}")
        print(f"    geometry: heads={model['heads']} head_dim={model['head_dim']} experts={model['experts']} topk={model['topk']} hidden={model['hidden']} moe_inter={model['moe_inter']}")
        for b in rep["blocking"]:
            print(f"    BLOCKING  {b}")
        for w in rep["warnings"]:
            print(f"    WARNING   {w}")
        if not rep["blocking"] and not rep["warnings"]:
            print("    all required tables + grids present")
        for k, v in rep["grids"].items():
            print(f"    grid[{k}] = {v}")
    print(f"\n=> {total_block} model(s) BLOCKED on staging/out data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
