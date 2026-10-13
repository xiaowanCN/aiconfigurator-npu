#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check whether a model can actually be priced from the ascend_910b silicon tables.

Why this exists
---------------
``aic-npu default`` reports failures only after a full sweep, and in HYBRID mode
it silently substitutes SOL estimates, so a model can look "supported" while
every interesting parallel config is unpriceable. This tool answers "is this
model really adapted?" BEFORE the sweep, straight from the shipped tables.

What it checks
--------------
1. every perf table the model's operator set needs exists;
2. the operator's key grid actually contains the model geometry:
   - attention/MLA: rank-local ``num_heads`` for every candidate TP
     (and ``head_dim`` for dense attention);
   - MoE: ``(topk, num_experts, hidden, inter)`` shape under some distribution,
     plus how many distinct ``moe_ep_size`` values were collected;
   - DSv4 CSA/HCA modules and mHC;
3. cluster-wide comm coverage (``num_gpus`` in the collective tables).

Usage
-----
    python tools/check_model_readiness.py                       # all model_configs
    python tools/check_model_readiness.py model_configs/x.json  # one model
    python tools/check_model_readiness.py --system ascend_910b \
        --backend vllm-ascend --version 0.23.0

Exit code is 1 when any model has a blocking gap.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

DEFAULT_SYSTEM = "ascend_910b"
DEFAULT_BACKEND = "vllm-ascend"
DEFAULT_VERSION = "0.23.0"

# operator: perf file name -> columns whose grid we report
TABLES = {
    "gemm": ("gemm_perf.txt", ("gemm_dtype",)),
    "attention": ("context_attention_perf.txt", ("num_heads", "head_dim", "attn_dtype")),
    "attention_gen": ("generation_attention_perf.txt", ("num_heads", "head_dim")),
    "mla": ("context_mla_perf.txt", ("num_heads", "mla_dtype", "kv_cache_dtype")),
    "mla_gen": ("generation_mla_perf.txt", ("num_heads", "mla_dtype")),
    "mla_bmm": ("mla_bmm_perf.txt", ("num_heads", "bmm_dtype")),
    "moe": (
        "moe_perf.txt",
        ("moe_dtype", "topk", "num_experts", "hidden_size", "inter_size", "moe_ep_size", "distribution"),
    ),
    "dsv4_csa": ("dsv4_csa_context_module_perf.txt", ("num_heads",)),
    "dsv4_hca": ("dsv4_hca_context_module_perf.txt", ("num_heads",)),
    "dsv4_gen": ("dsv4_csa_generation_module_perf.txt", ("num_heads",)),
    "dsa": ("dsa_context_module_perf.txt", ("num_heads",)),
    "mhc": ("mhc_module_perf.txt", ("hc_mult",)),
    "gdn": ("gdn_perf.txt", ()),
    "mamba2": ("mamba2_perf.txt", ()),
}

# family -> tables its operator graph reads (see aiconfigurator_npu/sdk/models.py)
FAMILY_REQUIREMENTS = {
    "DEEPSEEK": ["gemm", "mla", "mla_gen", "mla_bmm", "moe"],
    "DEEPSEEKV4": ["gemm", "dsv4_csa", "dsv4_hca", "dsv4_gen", "mhc", "moe"],
    "DEEPSEEKV32": ["gemm", "dsa", "moe"],
    "MOE": ["gemm", "attention", "attention_gen", "moe"],
    "GPT": ["gemm", "attention", "attention_gen"],
    "LLAMA": ["gemm", "attention", "attention_gen"],
    "QWEN35": ["gemm", "attention", "attention_gen", "gdn"],
    "NEMOTRONH": ["gemm", "attention", "attention_gen", "mamba2"],
}

# how a family shards attention across tp, and what it needs from the table
FAMILY_ATTENTION = {
    "DEEPSEEK": "mla",
    "DEEPSEEKV4": "dsv4",
    "DEEPSEEKV32": "dsa",
    "MOE": "dense",
    "GPT": "dense",
    "LLAMA": "dense",
    "QWEN35": "dense",
    "NEMOTRONH": "dense",
}

ARCH_TO_FAMILY = {
    "DeepseekV3ForCausalLM": "DEEPSEEK",
    "DeepseekV4ForCausalLM": "DEEPSEEKV4",
    "DeepseekV32ForCausalLM": "DEEPSEEKV32",
    "GlmMoeDsaForCausalLM": "DEEPSEEKV32",
    "Qwen3MoeForCausalLM": "MOE",
    "Qwen3_5ForCausalLM": "QWEN35",
    "Qwen3_5ForConditionalGeneration": "QWEN35",
    "NemotronHForCausalLM": "NEMOTRONH",
    "LlamaForCausalLM": "LLAMA",
    "GPTForCausalLM": "GPT",
}


def _read_rows(path: str) -> list[dict] | None:
    if not os.path.isfile(path):
        return None
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _int_values(rows: list[dict], column: str) -> list[int]:
    out = set()
    for row in rows:
        if column not in row:
            continue
        try:
            out.add(int(float(row[column])))
        except (TypeError, ValueError):
            pass
    return sorted(out)


def _all_values(rows: list[dict], column: str) -> list[str]:
    return sorted({row[column] for row in rows if row.get(column)})


def _load_model(model_path: str) -> dict:
    """Pull the geometry the perf queries use, straight from the HF config."""
    try:
        from aiconfigurator_npu.sdk.utils import get_model_config_from_model_path
    except Exception:  # noqa: BLE001 - sdk not importable: fall back to raw json
        import json

        with open(model_path, encoding="utf-8") as handle:
            config = json.load(handle)
        text = config.get("text_config", config)
        return {
            "architecture": (config.get("architectures") or ["?"])[0],
            "heads": int(text.get("num_attention_heads", 0)),
            "head_dim": int(text.get("head_dim") or 0),
            "topk": int(text.get("num_experts_per_tok", 0)),
            "experts": int(text.get("n_routed_experts") or text.get("num_experts") or 0),
            "hidden": int(text.get("hidden_size", 0)),
            "moe_inter": int(text.get("moe_intermediate_size", 0)),
        }

    info = get_model_config_from_model_path(model_path)
    return {
        "architecture": info.get("architecture") or "?",
        "heads": int(info.get("n", 0)),
        "head_dim": int(info.get("d", 0)),
        "topk": int(info.get("topk", 0)),
        "experts": int(info.get("num_experts", 0)),
        "hidden": int(info.get("hidden_size", 0)),
        "moe_inter": int(info.get("moe_inter_size", 0)),
    }


def _moe_coverage(rows: list[dict], model: dict) -> dict:
    """Distinct distributions / ep values collected for this expert shape."""
    matches = [
        r
        for r in rows
        if int(r.get("num_experts", 0)) == model["experts"]
        and int(r.get("topk", 0)) == model["topk"]
        and int(r.get("hidden_size", 0)) == model["hidden"]
        and int(r.get("inter_size", 0)) == model["moe_inter"]
    ]
    return {
        "rows": len(matches),
        "distributions": _all_values(matches, "distribution"),
        "ep": _int_values(matches, "moe_ep_size"),
    }


def check_model(model_path: str, data_dir: str, max_tp: int = 64) -> dict:
    model = _load_model(model_path)
    family = ARCH_TO_FAMILY.get(model["architecture"])
    if family is None:
        from aiconfigurator_npu.sdk import models as models_sdk

        try:
            family = models_sdk.get_model_family(model_path)
        except Exception:  # noqa: BLE001
            family = None

    report: dict = {
        "model": os.path.basename(model_path),
        "architecture": model["architecture"],
        "family": family or "?",
        "missing_tables": [],
        "warnings": [],
        "blocking": [],
        "grids": {},
    }

    if family not in FAMILY_REQUIREMENTS:
        report["blocking"].append(f"unknown architecture '{model['architecture']}' -> no family mapping")
        return report

    # 1. table presence
    for table in FAMILY_REQUIREMENTS[family]:
        filename, columns = TABLES[table]
        rows = _read_rows(os.path.join(data_dir, filename))
        if rows is None:
            report["missing_tables"].append(filename)
            report["blocking"].append(f"{family} requires {filename}, which is not collected")
            continue
        grid = {column: _int_values(rows, column) or _all_values(rows, column) for column in columns}
        # only report grids whose values beyond the first few matter
        report["grids"][table] = dict(list(grid.items())[:6])

    if report["missing_tables"]:
        # Several later checks need these tables; keep going where possible.
        pass

    # 2. attention / MLA head coverage per candidate tp
    attention_kind = FAMILY_ATTENTION.get(family)
    if attention_kind == "mla":
        grid = report["grids"].get("mla", {}).get("num_heads", [])
        supported = [tp for tp in _powers(max_tp) if tp and model["heads"] % tp == 0 and model["heads"] // tp in grid]
        unsupported = [
            tp for tp in _powers(max_tp) if tp and model["heads"] % tp == 0 and model["heads"] // tp not in grid
        ]
        if not supported:
            report["blocking"].append(
                f"MLA table has num_heads={grid}; tp adds to rank-local heads, so no tp is priceable "
                f"for num_attention_heads={model['heads']}"
            )
        elif unsupported:
            report["warnings"].append(
                f"attention tp {unsupported} unpriceable (need num_heads "
                f"{sorted({model['heads'] // tp for tp in unsupported})}), collected {grid}"
            )
        report["grids"]["mla_supported_tp"] = supported
    elif attention_kind == "dsv4":
        for table in ("dsv4_csa", "dsv4_hca"):
            grid = report["grids"].get(table, {}).get("num_heads", [])
            if not grid:
                continue
            missing = [tp for tp in _powers(max_tp) if tp and model["heads"] % tp == 0 and model["heads"] // tp not in grid]
            if missing:
                report["warnings"].append(
                    f"{table}: rank-local heads for tp {missing} not collected (have {grid})"
                )
    elif attention_kind == "dense" and family != "NEMOTRONH":
        heads_grid = report["grids"].get("attention", {}).get("num_heads", [])
        dims = report["grids"].get("attention", {}).get("head_dim", [])
        if model["head_dim"] and dims and model["head_dim"] not in dims:
            report["blocking"].append(
                f"dense attention collected head_dim={dims} but the model needs {model['head_dim']}"
            )
        if model["heads"] and heads_grid:
            report["grids"]["attention_allowed_tp"] = [
                tp for tp in _powers(max_tp) if tp and model["heads"] % tp == 0 and model["heads"] // tp in heads_grid
            ]
            unsupported = [
                tp
                for tp in _powers(max_tp)
                if tp and model["heads"] % tp == 0 and model["heads"] // tp not in heads_grid
            ]
            if unsupported:
                report["warnings"].append(
                    f"dense attention: rank-local heads for tp {unsupported} not collected (have {heads_grid})"
                )
        unsupported_tp = [tp for tp in _powers(max_tp) if tp and model["heads"] and model["heads"] % tp != 0]
        if unsupported_tp:
            report["warnings"].append(
                f"tp {unsupported_tp} rejected outright: not a divisor of num_heads={model['heads']}"
            )

    # 3. MoE expert-shape + ep coverage
    if "moe" in FAMILY_REQUIREMENTS[family] and model["experts"]:
        moe_rows = _read_rows(os.path.join(data_dir, "moe_perf.txt")) or []
        coverage = _moe_coverage(moe_rows, model)
        report["grids"]["moe_shape"] = coverage
        if coverage["rows"] == 0:
            report["blocking"].append(
                f"MoE table has no row for experts={model['experts']} topk={model['topk']} "
                f"hidden={model['hidden']} inter={model['moe_inter']}"
            )
        elif len(coverage["ep"]) <= 1:
            report["warnings"].append(
                f"MoE collected the experts={model['experts']} shape at ep={coverage['ep']} only; "
                "dp*tp == moe_ep pins the worker width, so other worker sizes need ESTIMATED transfer"
            )

    # 4.5 Can any worker width actually be priced? For MoE with moe_tp pinned to
    # 1 the sweep enforces dp*tp == moe_ep, so the worker width IS the expert
    # count per worker: a single collected ep collapses the whole space.
    if "moe" in FAMILY_REQUIREMENTS[family] and model["experts"]:
        # Keep the heads the attention tables can price, whichever attention kind applies.
        allowed_tp = report["grids"].get("attention_allowed_tp")
        if allowed_tp is None:
            allowed_tp = report["grids"].get("mla_supported_tp")
        if allowed_tp:
            available_ep = report["grids"].get("moe_shape", {}).get("ep", [])
            # A worker of width W needs some acceptable tp that divides it
            # (dp = W // tp), because the sweep enforces dp*tp == moe_ep.
            report["grids"]["viable_worker_widths"] = [
                width for width in available_ep if any(width % tp == 0 for tp in allowed_tp)
            ]
            viable = report["grids"]["viable_worker_widths"]
            if not viable:
                report["blocking"].append(
                    f"no priceable worker width: attention accepts tp in {allowed_tp} while MoE collected "
                    f"only ep={available_ep}, and dp*tp must equal moe_ep"
                )

    # 5. cluster-wide collectives
    for filename, grid_key in (("custom_allreduce_perf.txt", "custom_allreduce_gpus"),):
        allreduce_rows = _read_rows(os.path.join(data_dir, filename))
        if allreduce_rows is None:
            continue
        gpus = _int_values(allreduce_rows, "num_gpus")
        report["grids"][grid_key] = gpus
        if gpus and max(gpus) < 16:
            report["warnings"].append(
                f"custom_allreduce covers num_gpus up to {max(gpus)}; larger collectives are extrapolated"
            )

    nccl_root = os.path.abspath(os.path.join(data_dir, "..", "..", "nccl"))
    if os.path.isdir(nccl_root):
        newest = sorted(os.listdir(nccl_root))[-1]
        rows = _read_rows(os.path.join(nccl_root, newest, "nccl_perf.txt")) or []
        gpus = _int_values(rows, "num_gpus")
        report["grids"]["nccl_gpus"] = gpus
        if gpus and max(gpus) < 16:
            report["warnings"].append(f"nccl covers num_gpus up to {max(gpus)}; larger rings are extrapolated")

    return report


def _powers(max_tp: int) -> list[int]:
    out = []
    value = 1
    while value <= max_tp:
        out.append(value)
        value *= 2
    return out


def _default_data_dir(system: str, backend: str, version: str) -> str:
    if os.path.isdir(os.path.join("systems", "data", system, backend, version)):
        return os.path.join("systems", "data", system, backend, version)
    try:
        import aiconfigurator_npu  # noqa: F401

        package_path = os.path.dirname(aiconfigurator_npu.__file__)
        return os.path.join(str(package_path), "systems", "data", system, backend, version)
    except Exception:  # noqa: BLE001
        return os.path.join("src", "aiconfigurator_npu", "systems", "data", system, backend, version)


def _discover_models() -> list[str]:
    roots = [os.path.join("model_configs"), os.path.join("src", "aiconfigurator_npu", "model_configs")]
    root = next((r for r in roots if os.path.isdir(r)), None)
    if root is None:
        return []
    return sorted(os.path.join(root, name) for name in os.listdir(root) if name.endswith(".json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help="model config json files (default: all in model_configs/)")
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--max-tp", type=int, default=64, help="largest attention TP to consider")
    args = parser.parse_args()

    models = args.models or _discover_models()
    if not models:
        print("no model configs found (run from the repo root, or pass paths)")
        return 2

    data_dir = _default_data_dir(args.system, args.backend, args.version)
    if not os.path.isdir(data_dir):
        print(f"perf data directory not found: {data_dir}")
        return 2

    print("=" * 100)
    print(f"system={args.system} backend={args.backend} version={args.version}")
    print(f"data_dir={data_dir}")
    print("=" * 100)

    failures = 0
    for model_path in models:
        report = check_model(model_path, data_dir, max_tp=args.max_tp)
        status = "BLOCKED" if report["blocking"] else ("DEGRADED" if report["warnings"] else "READY")
        failures += bool(report["blocking"])
        print(f"\n[{status}] {report['model']}  (family={report['family']}, arch={report['architecture']})")
        for item in report["blocking"]:
            print(f"    BLOCKING  {item}")
        for item in report["warnings"]:
            print(f"    WARNING   {item}")
        if not report["blocking"] and not report["warnings"]:
            print("    all required tables and grids present")

    print()
    print("=" * 100)
    print(
        f"{len(models)} model(s) checked, {failures} blocked. "
        "Blocked models cannot be priced in SILICON mode until the missing sweeps are collected; "
        "in HYBRID mode they silently fall back to SOL estimates."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
