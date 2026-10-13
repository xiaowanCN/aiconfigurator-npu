#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare the NPU search-space enumeration against NVIDIA aic-core.

``enumerate_parallel_config`` decides which deployments ever get priced, so a
silent divergence from upstream changes every recommendation. This tool runs
structurally identical inputs through both implementations and diffs the result.

Notes on the comparison
-----------------------
* Upstream emits 6-field configs ``[tp, pp, dp, moe_tp, moe_ep, cp]``; this fork
  emits 5 fields (documented deliberate divergence: context parallelism needs
  perf modeling we do not have). With ``cp=1`` the two must agree on the first
  five fields, which is exactly what is asserted here.
* ``vllm-ascend`` has no upstream counterpart; it is compared against upstream's
  ``vllm`` branch, whose filter ("MoE TP and EP cannot both be > 1") is the rule
  this fork extends to vllm-ascend.
* The upstream package cannot be imported normally -- its ``__init__`` pulls in a
  Rust extension (``_aiconfigurator_core``) that is usually not built in a source
  checkout. We register stub parents and load the two pure-python modules by path.

Usage
-----
    python tools/check_search_space_parity.py
    python tools/check_search_space_parity.py --nvidia-root ../aiconfigurator

Exit code is 1 when any case differs.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types

NPU_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NUM_GPU_LIST = [1, 2, 4, 8, 16, 32, 64]

# name, tp, pp, dp, moe_tp, moe_ep, backend, is_moe, moe_backend
CASES: list[tuple] = [
    ("vllm-ascend MoE", [1, 2, 4, 8], [1], [1, 2, 4, 8], [1], [1, 2, 4, 8], "vllm-ascend", True, None),
    ("vllm MoE", [1, 2, 4, 8], [1], [1, 2, 4, 8], [1, 2, 4], [1, 2, 4, 8], "vllm", True, None),
    ("trtllm MoE", [1, 2, 4, 8], [1, 2], [1, 2, 4], [1], [1, 2, 4, 8], "trtllm", True, None),
    ("sglang MoE default", [1, 2, 4, 8], [1], [1, 2, 4, 8], [1, 2, 4], [1, 2, 4, 8], "sglang", True, None),
    ("sglang MoE megamoe", [1, 2, 4, 8], [1], [1, 2, 4, 8], [1, 2, 4], [1, 2, 4, 8], "sglang", True, "megamoe"),
    ("vllm-ascend dense", [1, 2, 4, 8], [1, 2], [1], [1], [1], "vllm-ascend", False, None),
    ("trtllm dense", [1, 2, 4, 8], [1, 2], [1], [1], [1], "trtllm", False, None),
]


def _stub_package(name: str, location: str) -> types.ModuleType:
    stub = types.ModuleType(name)
    stub.__path__ = [location]
    sys.modules[name] = stub
    return stub


def _load_module(name: str, path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_upstream(nvidia_root: str):
    nv_src = os.path.join(nvidia_root, "aic-core", "src")
    nv_pkg = os.path.join(nv_src, "aiconfigurator_core")
    nv_sdk = os.path.join(nv_pkg, "sdk")
    for candidate in (nv_src, nv_sdk):
        if not os.path.isdir(candidate):
            raise SystemExit(f"upstream path not found: {candidate}")
    sdk_stub = _stub_package("aiconfigurator_core", nv_src)
    _stub_package("aiconfigurator_core.sdk", nv_sdk)
    common = _load_module("aiconfigurator_core.sdk.common", os.path.join(nv_sdk, "common.py"))
    setattr(sdk_stub, "common", common)
    utils = _load_module("aiconfigurator_core.sdk.utils", os.path.join(nv_sdk, "utils.py"))
    return common, utils


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--nvidia-root",
        default=os.path.join(NPU_ROOT, "..", "aiconfigurator"),
        help="path to the NVIDIA aiconfigurator checkout to compare against",
    )
    args = parser.parse_args()

    nvidia_root = os.path.abspath(args.nvidia_root)
    sys.path.insert(0, os.path.join(NPU_ROOT, "src"))

    nv_common, nv_utils = load_upstream(nvidia_root)
    from aiconfigurator_npu.sdk import common as npu_common  # noqa: PLC0415
    from aiconfigurator_npu.sdk import utils as npu_utils  # noqa: PLC0415

    print(f"npu     : {NPU_ROOT}")
    print(f"upstream: {nvidia_root}")
    print(f"cases   : {len(CASES)}")

    failures = 0
    for name, tp, pp, dp, mtp, mep, backend, is_moe, moe_backend in CASES:
        npu_out = npu_utils.enumerate_parallel_config(
            num_gpu_list=NUM_GPU_LIST,
            tp_list=tp, pp_list=pp, dp_list=dp, moe_tp_list=mtp, moe_ep_list=mep,
            is_moe=is_moe,
            backend=npu_common.BackendName(backend),
            moe_backend=moe_backend,
        )
        nv_backend = "vllm" if backend == "vllm-ascend" else backend
        nv_out = nv_utils.enumerate_parallel_config(
            num_gpu_list=NUM_GPU_LIST,
            tp_list=tp, pp_list=pp, dp_list=dp, moe_tp_list=mtp, moe_ep_list=mep,
            is_moe=is_moe,
            backend=nv_common.BackendName(nv_backend),
            moe_backend=moe_backend,
        )
        nv_trimmed = [cfg[:5] for cfg in nv_out]
        ok = npu_out == nv_trimmed
        failures += not ok
        print(f"  {'OK  ' if ok else 'DIFF'} {name:22s} npu={len(npu_out):3d} upstream={len(nv_trimmed):3d}")
        if not ok:
            for label, source, target in (("npu-only", npu_out, nv_trimmed), ("upstream-only", nv_trimmed, npu_out)):
                extra = [c for c in source if c not in target]
                if extra:
                    print(f"        {label}: {extra[:4]}")

    print()
    print("FAIL" if failures else "PASS", f"- {len(CASES)} cases, {failures} differing")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
