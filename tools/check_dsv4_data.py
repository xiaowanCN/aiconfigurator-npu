"""Check DeepSeek-V4 silicon-data coverage for aic-npu (no NPU required).

Verifies that every perf table / exact query key the DSv4 path needs is actually
reachable in SILICON mode, so a missing table is caught before spending an hour
on ``aic-npu default``.

What it probes (see docs/DeepSeek-V4-Pro_ADAPTATION.md §3 for the key matrix):

  * dsv4_{csa,hca}_{context,generation}_module_perf.txt
      keys: (fmha=float16, kvcache=float16, gemm=float16) x num_heads x (b, s)
  * mhc_module_perf.txt
      keys: op_name in {pre, post} x hc_mult=4 x hidden_size=7168 x num_tokens
  * moe_perf.txt
      keys: (float16, distribution) x topk=6 x num_experts=384 x 7168/3072
            x moe_tp=1 x moe_ep

Every probe runs with ``DatabaseMode.SILICON`` — anything that raises is a hard
miss (in HYBRID it would silently fall back to the SOL estimate instead).

Usage:
    python tools/check_dsv4_data.py --isl 4096 --osl 1024
    python tools/check_dsv4_data.py --only attn
    python tools/check_dsv4_data.py --only moe --moe-ep-list 2 4 8

Exit codes: 0 = all probes hit, 1 = at least one miss, 2 = environment error.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from aiconfigurator_npu.sdk import common  # noqa: E402
from aiconfigurator_npu.sdk.perf_database import get_database  # noqa: E402

# DeepSeek-V4-Pro structural constants (model_configs/...DeepSeek-V4-Pro_config.json)
V4_HIDDEN = 7168
V4_INTER = 3072
V4_TOPK = 6
V4_NUM_EXPERTS = 384
V4_HC_MULT = 4
V4_SINKHORN_ITERS = 20
# num_heads = 128 // tp. tp_list is widened with 16/32/64/... up to --total-gpus
# (TaskConfig._widen_gpu_sweep_lists), so --total-gpus 64 reaches tp=64.
DEFAULT_HEADS_LIST = [128, 64, 32, 16, 8, 4, 2]
# On 64 GB 910B cards only ep=64 fits (~45-57 GiB/card); 16/32 are cheap to
# collect and kept as tolerated extras, <=8 cannot fit the model at all.
DEFAULT_MOE_EP_LIST = [16, 32, 64]
DEFAULT_MOE_EP_OPTIONAL = [16, 32]

CONTEXT_BATCH_PROBES = (1, 2, 4, 8, 16, 32)
GENERATION_BATCH_PROBES = (1, 8, 64, 128, 256, 512)
MHC_TOKEN_PROBES = (1, 512, 4096, 16384, 32768)
MOE_TOKEN_PROBES = (1, 128, 1024, 4096, 16384)

ATTN_KINDS = {
    "csa": 4,    # compress_ratio 4  -> dsv4_csa_*
    "hca": 128,  # compress_ratio 128 (SWA folds in) -> dsv4_hca_*
}


class Report:
    """Collects probe outcomes so we can print a compact summary at the end.

    Misses are grouped: one missing table/key normally fails every probe of that
    group, so 40 identical ``IndexError`` lines would only bury the signal.
    """

    def __init__(self) -> None:
        self.hits = 0
        self.misses: dict[str, list[str]] = {}
        self.warnings: dict[str, list[str]] = {}
        self.notes: list[str] = []

    def ok(self) -> None:
        self.hits += 1

    def miss(self, group: str, detail: str) -> None:
        self.misses.setdefault(group, []).append(detail)

    def warn(self, group: str, detail: str) -> None:
        self.warnings.setdefault(group, []).append(detail)

    def note(self, text: str) -> None:
        self.notes.append(text)

    @property
    def miss_count(self) -> int:
        return sum(len(v) for v in self.misses.values())

    @property
    def warn_count(self) -> int:
        return sum(len(v) for v in self.warnings.values())


def _read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _int_bounds(rows: list[dict], column: str) -> tuple[int, int] | None:
    values = []
    for row in rows:
        try:
            values.append(int(row[column]))
        except (KeyError, TypeError, ValueError):
            continue
    if not values:
        return None
    return min(values), max(values)


def _fmt_bounds(bounds: tuple[int, int] | None) -> str:
    return f"{bounds[0]}..{bounds[1]}" if bounds else "n/a"


def _probe(report: Report, group: str, label: str, fn, optional: bool = False) -> None:
    """Run one SILICON query; any exception is a miss (or a warning if optional)."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - any failure means "falls back to SOL"
        detail = f"{label}: {type(exc).__name__}: {str(exc).splitlines()[0][:110]}"
        if optional:
            report.warn(group, detail)
        else:
            report.miss(group, detail)
    else:
        report.ok()


def _path(text: str) -> str:
    """Normalise the SDK's mixed-separator data paths for readable output."""
    return os.path.normpath(text)


def _axis_values(rows: list[dict], key: str) -> dict[str, set[int]]:
    """head -> {values of ``key``} for that head."""
    out: dict[str, set[int]] = {}
    for row in rows:
        try:
            out.setdefault(row["num_heads"], set()).add(int(row[key]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _grid_diagnostics(report: Report, table: str, rows: list[dict], probe_s: int) -> None:
    """Flag grids that make the SDK's 2-D cubic interpolation impossible.

    ``PerfDatabase._interp_2d_1d`` builds a 2-D (s, batch) Delaunay triangulation
    per num_heads and interpolates with method="cubic". When an axis collapses to
    a single value the points are collinear and scipy raises
    ``QhullError: Initial simplex is flat`` — which kills the WHOLE table, even
    queries at the one batch value that does exist.
    """
    per_head_b = _axis_values(rows, "batch_size")
    per_head_s = _axis_values(rows, "isl")

    flat_b = sorted(h for h, v in per_head_b.items() if len(v) < 2)
    flat_s = sorted(h for h, v in per_head_s.items() if len(v) < 2)
    if flat_b:
        report.miss(
            table,
            f"batch axis degenerate for num_heads={flat_b}: only one batch value "
            f"-> 2-D cubic interpolation raises QhullError and the ENTIRE table is "
            f"unusable (even b=1 queries fail). Collect >= 2 batch values.",
        )
    if flat_s:
        report.miss(
            table,
            f"isl axis degenerate for num_heads={flat_s}: only one seq value "
            f"-> same QhullError. Collect >= 2 seq values.",
        )

    # Holes: a (head, batch) cell holding a single (or no) seq point cannot serve
    # any other s. This is the classic "large batch x long seq all OOM'd" pattern.
    cells: dict[tuple[str, int], set[int]] = {}
    for row in rows:
        try:
            cells.setdefault((row["num_heads"], int(row["batch_size"])), set()).add(int(row["isl"]))
        except (KeyError, TypeError, ValueError):
            continue
    thin = {k: sorted(v) for k, v in cells.items() if len(v) < 2}
    if thin:
        sample = sorted(thin)[:5]
        report.note(
            f"{table}: {len(thin)} (num_heads,batch) cell(s) hold a single seq point "
            f"-> queries at other s fail there (e.g. {sample}, s={sorted(thin[sample[0]])}). "
            f"Target s={probe_s}."
        )
    unreachable = sorted(k for k, v in cells.items() if v and max(v) < probe_s)
    if unreachable:
        report.note(
            f"{table}: {len(unreachable)} (num_heads,batch) cell(s) never reach s={probe_s} "
            f"(max isl {max(max(v) for v in (cells[k] for k in unreachable))}) "
            f"-> those cells can only be reached by extrapolation."
        )


def check_attention(report: Report, db, args) -> None:
    fmha = common.FMHAQuantMode.float16
    kvcache = common.KVCacheQuantMode.float16
    gemm = common.GEMMQuantMode.float16

    for kind, ratio in ATTN_KINDS.items():
        for phase in ("context", "generation"):
            attr = f"_{phase}_dsv4_{kind}_module_data"
            loaded = getattr(db, attr, None)
            table = f"dsv4_{kind}_{phase}_module_perf.txt"
            if loaded is None or not getattr(loaded, "loaded", False):
                report.miss(
                    table,
                    f"table not loaded (expected {_path(getattr(loaded, 'filepath', '?'))})",
                )
                continue

            rows = _read_csv(loaded.filepath)
            heads = _int_bounds(rows, "num_heads")
            batch = _int_bounds(rows, "batch_size")
            seq = _int_bounds(rows, "isl")
            report.note(
                f"{table}: {len(rows)} rows | num_heads {_fmt_bounds(heads)} "
                f"| batch {_fmt_bounds(batch)} | isl {_fmt_bounds(seq)}"
            )

            if phase == "context":
                s = args.isl
                batches = CONTEXT_BATCH_PROBES
            else:
                s = args.isl + args.osl
                batches = GENERATION_BATCH_PROBES

            _grid_diagnostics(report, table, rows, s)

            for num_heads in args.num_heads_list:
                for b in batches:
                    label = f"{table} heads={num_heads} b={b} s={s}"
                    if phase == "context":
                        _probe(
                            report,
                            table,
                            label,
                            lambda b=b, s=s, num_heads=num_heads, kind=kind, ratio=ratio: db.query_context_dsv4_module(
                                b=b,
                                s=s,
                                num_heads=num_heads,
                                kvcache_quant_mode=kvcache,
                                fmha_quant_mode=fmha,
                                gemm_quant_mode=gemm,
                                database_mode=common.DatabaseMode.SILICON,
                                attn_kind=kind,
                                compress_ratio=ratio,
                                hidden_size=V4_HIDDEN,
                                o_lora_rank=1024,
                                o_groups=16,
                                head_dim=512,
                                qk_rope_head_dim=64,
                                index_n_heads=64,
                                index_head_dim=128,
                                index_topk=1024,
                                sliding_window=128,
                            ),
                        )
                    else:
                        _probe(
                            report,
                            table,
                            label,
                            lambda b=b, s=s, num_heads=num_heads, kind=kind, ratio=ratio: db.query_generation_dsv4_module(
                                b=b,
                                s=s,
                                num_heads=num_heads,
                                kv_cache_dtype=kvcache,
                                gemm_quant_mode=gemm,
                                database_mode=common.DatabaseMode.SILICON,
                                attn_kind=kind,
                                compress_ratio=ratio,
                                hidden_size=V4_HIDDEN,
                                o_lora_rank=1024,
                                o_groups=16,
                                head_dim=512,
                                qk_rope_head_dim=64,
                                index_n_heads=64,
                                index_head_dim=128,
                                index_topk=1024,
                                sliding_window=128,
                            ),
                        )


def check_mhc(report: Report, db, args) -> None:
    loaded = getattr(db, "_mhc_module_data", None)
    if loaded is None or not getattr(loaded, "loaded", False):
        report.miss(
            "mhc_module_perf.txt",
            f"table not loaded (expected {_path(getattr(loaded, 'filepath', '?'))})",
        )
        return

    rows = _read_csv(loaded.filepath)
    ops = sorted({r.get("op_name") for r in rows if r.get("op_name")})
    report.note(
        f"mhc_module_perf.txt: {len(rows)} rows | op_name {ops} | num_tokens "
        f"{_fmt_bounds(_int_bounds(rows, 'num_tokens'))} | hc_mult "
        f"{sorted({r.get('hc_mult') for r in rows})} | hidden "
        f"{sorted({r.get('hidden_size') for r in rows})}"
    )
    missing_ops = [op for op in ("pre", "post") if op not in ops]
    if missing_ops:
        report.miss(
            "mhc_module_perf.txt",
            f"op_name {missing_ops} not collected (query_mhc_module keys on op_name; "
            f"both 'pre' and 'post' are required) -> run collect_mhc.py --op-types pre post",
        )

    for op in ("pre", "post"):
        for num_tokens in MHC_TOKEN_PROBES:
            _probe(
                report,
                "mhc_module_perf.txt",
                f"op={op} num_tokens={num_tokens}",
                lambda op=op, num_tokens=num_tokens: db.query_mhc_module(
                    op=op,
                    num_tokens=num_tokens,
                    hc_mult=V4_HC_MULT,
                    hidden_size=V4_HIDDEN,
                    gemm_quant_mode=common.GEMMQuantMode.float16,
                    database_mode=common.DatabaseMode.SILICON,
                    sinkhorn_iters=V4_SINKHORN_ITERS,
                ),
            )


def check_moe(report: Report, db, args) -> None:
    loaded = getattr(db, "_moe_data", None)
    if loaded is None or not getattr(loaded, "loaded", False):
        report.miss(
            "moe_perf.txt",
            f"table not loaded (expected {_path(getattr(loaded, 'filepath', '?'))})",
        )
        return

    rows = _read_csv(loaded.filepath)
    v4_rows = [
        r
        for r in rows
        if r.get("topk") == str(V4_TOPK)
        and r.get("num_experts") == str(V4_NUM_EXPERTS)
        and r.get("hidden_size") == str(V4_HIDDEN)
        and r.get("inter_size") == str(V4_INTER)
    ]
    distributions = sorted({r.get("distribution") for r in v4_rows if r.get("distribution")})
    eps = sorted({int(r["moe_ep_size"]) for r in v4_rows if r.get("moe_ep_size")})
    report.note(
        f"moe_perf.txt: {len(rows)} rows total, {len(v4_rows)} V4 rows "
        f"(topk=6/384/7168/3072) | distribution {distributions or 'n/a'} "
        f"| moe_ep_size {eps or 'n/a'} | num_tokens {_fmt_bounds(_int_bounds(v4_rows, 'num_tokens'))}"
    )
    if not v4_rows:
        report.miss(
            "moe_perf.txt (V4 rows)",
            "no rows for topk=6 / num_experts=384 / hidden 7168 / inter 3072 (DeepSeek-V4-Pro)",
        )

    # --append is not idempotent: re-running the landing script duplicates rows.
    key_fields = (
        "moe_dtype", "distribution", "topk", "num_experts", "hidden_size",
        "inter_size", "moe_tp_size", "moe_ep_size", "num_tokens",
    )
    seen: set[tuple] = set()
    dupes = 0
    for row in v4_rows:
        key = tuple(row.get(f) for f in key_fields)
        if key in seen:
            dupes += 1
        seen.add(key)
    if dupes:
        report.warn(
            "moe_perf.txt (duplicates)",
            f"{dupes} duplicate V4 row(s) out of {len(v4_rows)} — --append was probably run "
            f"more than once. load_moe_data keeps the first hit and only logs at debug, so "
            f"results are correct but the table is bloated; drop the file and re-append once.",
        )

    quant = common.MoEQuantMode.float16
    optional_eps = set(args.moe_ep_optional)
    for ep in args.moe_ep_list:
        for is_context in (True, False):
            for num_tokens in MOE_TOKEN_PROBES:
                _probe(
                    report,
                    f"moe_perf.txt (ep={ep})",
                    f"ctx={is_context} num_tokens={num_tokens}",
                    lambda ep=ep, is_context=is_context, num_tokens=num_tokens: db.query_moe(
                        num_tokens=num_tokens,
                        hidden_size=V4_HIDDEN,
                        inter_size=V4_INTER,
                        topk=V4_TOPK,
                        num_experts=V4_NUM_EXPERTS,
                        moe_tp_size=1,
                        moe_ep_size=ep,
                        quant_mode=quant,
                        workload_distribution=args.moe_distribution,
                        is_context=is_context,
                        database_mode=common.DatabaseMode.SILICON,
                    ),
                    optional=ep in optional_eps,
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", default="ascend_910b")
    parser.add_argument("--backend", default="vllm-ascend")
    parser.add_argument("--version", default="0.23.0")
    parser.add_argument(
        "--systems-path", default=None,
        help=(
            "Systems root to load from (defaults to the installed search path). "
            "Point it at a staging copy containing '<system>.yaml' + 'data/' to "
            "validate freshly collected tables BEFORE copying them into the repo."
        ),
    )
    parser.add_argument("--isl", type=int, default=4096, help="Context sequence length to probe")
    parser.add_argument("--osl", type=int, default=1024, help="Output length; decode probes s = isl + osl")
    parser.add_argument(
        "--num-heads-list", type=int, nargs="+", default=DEFAULT_HEADS_LIST,
        help="num_heads values to probe (128 // tp; tp is widened up to --total-gpus)",
    )
    parser.add_argument(
        "--moe-ep-list", type=int, nargs="+", default=DEFAULT_MOE_EP_LIST,
        help="moe_ep_size values to probe. Defaults to the EP range that can actually "
             "hold DeepSeek-V4-Pro on 64 GB cards (only ep=64 fits).",
    )
    parser.add_argument(
        "--moe-ep-optional", type=int, nargs="*", default=DEFAULT_MOE_EP_OPTIONAL,
        help=(
            "EP values whose absence is downgraded to a warning instead of a failure. "
            "Defaults to 16/32: they are cheap to collect but the corresponding "
            "deployments OOM on 64 GB cards (>=90 GiB/card), so they never win anyway."
        ),
    )
    parser.add_argument(
        "--moe-distribution", default="power_law_1.01",
        help="distribution key DeepSeekV4Model queries (_power_law_alpha = 1.01)",
    )
    parser.add_argument("--only", default="all", choices=["all", "attn", "mhc", "moe"])
    parser.add_argument("--log-level", default="ERROR")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.ERROR))

    db = get_database(
        system=args.system,
        backend=args.backend,
        version=args.version,
        systems_paths=args.systems_path,
    )
    if db is None:
        print(f"ERROR: no database for system={args.system} backend={args.backend} version={args.version}")
        return 2

    print(f"Database : {args.system} / {args.backend} / {args.version}")
    print(f"Root     : {os.path.normpath(db.systems_root)}")
    print(f"Probes   : isl={args.isl} osl={args.osl} (decode s={args.isl + args.osl})")
    print(f"           num_heads={args.num_heads_list}")
    print(f"           moe_ep={args.moe_ep_list} (optional {args.moe_ep_optional}) "
          f"distribution={args.moe_distribution}")
    print()

    report = Report()
    if args.only in ("all", "attn"):
        check_attention(report, db, args)
    if args.only in ("all", "mhc"):
        check_mhc(report, db, args)
    if args.only in ("all", "moe"):
        check_moe(report, db, args)

    print("--- table inventory ---")
    for note in report.notes:
        print(f"  {note}")

    if report.warnings:
        print(f"--- WARNINGS ({report.warn_count} probes, {len(report.warnings)} group(s)) ---")
        for group, details in report.warnings.items():
            print(f"  ! {group}  [{len(details)} probe(s)] — tolerated, will fall back to SOL")
            print(f"      e.g. {details[0]}")
        print()

    if report.misses:
        print(f"--- MISSES ({report.miss_count} probes, {len(report.misses)} group(s)) ---")
        for group, details in report.misses.items():
            print(f"  x {group}  [{len(details)} probe(s)]")
            print(f"      e.g. {details[0]}")
        print()
        print(f"FAILED: {report.miss_count} probe(s) miss silicon data ({report.hits} hit).")
        print("The DSv4 path will fall back to SOL estimates (HYBRID) or raise (SILICON).")
        print("See docs/DeepSeek-V4-Pro_ADAPTATION.md §4 for the collection commands.")
        return 1

    print(f"OK: all {report.hits} probes hit silicon data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
