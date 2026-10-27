"""结果落盘与终端渲染。

CLI 与未来的 HTTP 服务共用同一套产物：
``<out-dir>/<run_id>/{run.json, request.json, summary.txt, aic_npu/, msmodeling/, logs/}``
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from infer_strategy_npu.config import DEFAULT_OUT_DIR, slugify
from infer_strategy_npu.schema import Candidate, ServingModeResult, UnifiedRunResult, to_float


def make_run_id(model: str, device: str, num_devices: int) -> str:
    """生成运行标识：``{时间戳}-{模型}-{设备}-{卡数}-{8 位唯一后缀}``。

    - 时间戳前缀：目录按创建时间自然排序，肉眼可按时间定位
    - 8 位后缀：避免同一秒发起的同参数任务重名（CLI 与 HTTP 共用此函数）
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{slugify(model)}-{slugify(device)}-{num_devices}-{uuid.uuid4().hex[:8]}"


@dataclass
class RunArtifacts:
    """一次运行的产物目录。"""

    run_id: str
    run_dir: Path
    logs_dir: Path
    aic_npu_dir: Path
    msmodeling_dir: Path
    files: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        out_dir: str | None,
        *,
        model: str,
        device: str,
        num_devices: int,
        run_id: str | None = None,
    ) -> RunArtifacts:
        """创建一次运行的产物目录。

        ``run_id`` 可由调用方（HTTP 服务的 task_id）指定，保证"任务 id == 结果目录名"；
        不指定时按 ``{时间戳}-{模型}-{设备}-{卡数}`` 自动生成（CLI 行为不变）。
        """
        base = Path(out_dir or DEFAULT_OUT_DIR).expanduser()
        if not run_id:
            run_id = make_run_id(model, device, num_devices)
        run_dir = base / run_id
        logs_dir = run_dir / "logs"
        aic_dir = run_dir / "aic_npu"
        ms_dir = run_dir / "msmodeling"
        for path in (run_dir, logs_dir, aic_dir, ms_dir):
            path.mkdir(parents=True, exist_ok=True)
        return cls(run_id=run_id, run_dir=run_dir, logs_dir=logs_dir, aic_npu_dir=aic_dir, msmodeling_dir=ms_dir)

    def write_json(self, relative: str, payload: Any) -> Path:
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        self.files.append(str(path))
        return path

    def write_text(self, relative: str, text: str) -> Path:
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self.files.append(str(path))
        return path

    def write_dataframe(self, relative: str, df) -> Path | None:
        if df is None:
            return None
        try:
            if hasattr(df, "empty") and df.empty:
                return None
        except Exception:  # pragma: no cover
            pass
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        self.files.append(str(path))
        return path

    def log_path(self, name: str) -> Path:
        return self.logs_dir / f"{name}.log"


def _fmt(value: float | int | None, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value}{suffix}"
    return f"{value:.{digits}f}{suffix}"


def _clip(value: str | None, width: int = 58) -> str:
    if not value:
        return "-"
    text = str(value)
    return text if len(text) <= width else text[: width - 3] + "..."


def _candidate_rows(candidates: list[Candidate], framework: str, serving_mode: str, mode: str) -> list[list[str]]:
    rows = []
    for idx, c in enumerate(candidates, start=1):
        rows.append(
            [
                str(idx),
                framework,
                mode,
                _fmt(to_float(c.throughput_tokens_per_s)),
                _fmt(to_float(c.throughput_tokens_per_s_per_device)),
                _fmt(to_float(c.ttft_ms)),
                _fmt(to_float(c.tpot_ms)),
                _fmt(to_float(c.throughput_req_per_s), 3),
                "-" if c.concurrency is None else str(c.concurrency),
                _clip(c.parallel_str),
            ]
        )
    return rows


_TABLE_HEADER = [
    "Top",
    "Framework",
    "Mode",
    "tokens/s",
    "tokens/s/dev",
    "TTFT(ms)",
    "TPOT(ms)",
    "req/s",
    "Conc.",
    "Parallel",
]

# PD 分离（PD-ratio）候选汇总表，按"单 D 实例 decode 吞吐"降序
_PD_TABLE_HEADER = [
    "Rank",
    "PxD",
    "P inst",
    "D inst",
    "P dev/inst",
    "D dev/inst",
    "P Parallel",
    "D Parallel",
    "P bs",
    "D bs",
    "P conc",
    "D conc",
    "TTFT(ms)",
    "TPOT(ms)",
    "decode tps/inst",
    "decode tps/cluster",
    "cluster tokens/s",
    "tokens/s/dev",
    "req/s",
    "Devices",
]


def _make_table(rows: list[list[str]], header: list[str] | None = None) -> str:
    columns = header or _TABLE_HEADER
    try:
        from prettytable import PrettyTable

        table = PrettyTable(columns)
        for row in rows:
            table.add_row(row)
        return str(table)
    except Exception:  # pragma: no cover - prettytable 缺失时退化为简单表格
        widths = [max(len(str(r[i])) for r in [*rows, columns]) for i in range(len(columns))]
        line = " | ".join(h.ljust(w) for h, w in zip(columns, widths))
        body = "\n".join(" | ".join(str(c).ljust(w) for c, w in zip(row, widths)) for row in rows)
        return f"{line}\n{'-' * len(line)}\n{body}"


def _pd_candidate_rows(candidates: list[dict]) -> list[list[str]]:
    """把 run.json 里的 PD 候选字典渲染成表格行。"""
    rows = []
    for item in candidates:
        prefill = item.get("prefill") or {}
        decode = item.get("decode") or {}
        rows.append(
            [
                str(item.get("rank") or len(rows) + 1),
                item.get("label") or "-",
                str(prefill.get("instances") or "-"),
                str(decode.get("instances") or "-"),
                str(prefill.get("devices_per_instance") or "-"),
                str(decode.get("devices_per_instance") or "-"),
                _clip(prefill.get("parallel_str"), 40),
                _clip(decode.get("parallel_str"), 40),
                str(prefill.get("batch_size") or "-"),
                str(decode.get("batch_size") or "-"),
                str(prefill.get("concurrency") or "-"),
                str(decode.get("concurrency") or "-"),
                _fmt(to_float(item.get("ttft_ms"))),
                _fmt(to_float(item.get("tpot_ms"))),
                _fmt(to_float(item.get("decode_tps_per_instance"))),
                _fmt(to_float(item.get("decode_tps_cluster"))),
                _fmt(to_float(item.get("throughput_tokens_per_s"))),
                _fmt(to_float(item.get("throughput_tokens_per_s_per_device"))),
                _fmt(to_float(item.get("throughput_req_per_s")), 3),
                f"{item.get('devices_used') or '-'}/{item.get('num_devices') or '-'}",
            ]
        )
    return rows


def _variant_headline(variant: ServingModeResult) -> str:
    """PD-ratio 变体的一行标题：P/D 两侧的实例与卡数分布。"""
    combo = variant.pd_ratio or {}
    if not combo:
        return "[PD 分离] P/D 两侧均用满总卡数"
    extra = (variant.best.extra if variant.best else None) or {}
    prefill = extra.get("prefill") or {}
    decode = extra.get("decode") or {}
    p_inst, d_inst = prefill.get("instances"), decode.get("instances")
    p_dev, d_dev = combo.get("prefill_devices"), combo.get("decode_devices")

    def _side(tag: str, instances, devices) -> str:
        if instances is None or devices is None:
            return f"{tag} 侧 {devices} 卡/实例"
        return f"{tag} 侧 {instances} 实例 × {devices} 卡（共 {instances * devices} 卡）"

    return f"[{variant.label}]  {_side('P', p_inst, p_dev)} | {_side('D', d_inst, d_dev)}"


def render_summary(result: UnifiedRunResult) -> str:
    """渲染人类可读摘要（同时写入 summary.txt）。"""
    lines: list[str] = []
    req = result.request
    lines.append("=" * 100)
    lines.append("infer-strategy-npu - recommend summary")
    lines.append("=" * 100)
    lines.append(f"run_id      : {result.run_id}")
    lines.append(f"status      : {result.status}")
    lines.append(f"model       : {req.get('model')}")
    lines.append(f"device      : {req.get('device')}  x{req.get('num_devices')}")
    lines.append(
        f"isl/osl     : {req.get('input_length')}/{req.get('output_length')}   "
        f"tpot-limit={req.get('tpot_limit')} ms   ttft-limit={req.get('ttft_limit')} ms"
    )
    lines.append(f"quantize    : {req.get('quantize')}")
    lines.append(
        f"serving     : {' '.join(req.get('serving_modes') or [])}"
        f"   PD 每实例卡数枚举: {' '.join(str(v) for v in (req.get('pd_ratio_sizes') or []))}"
    )
    lines.append(f"framework   : {result.routing.get('framework')}  ({result.routing.get('reason')})")
    if result.routing.get("aic_model_path"):
        lines.append(f"model-path  : {result.routing['aic_model_path']}")
    if result.routing.get("ms_model_source"):
        lines.append(f"model-source: {result.routing['ms_model_source']}")
    lines.append("")

    if not result.frameworks:
        lines.append("下发命令（dry-run / 未执行）:")
        for cmd in result.routing.get("commands", []):
            lines.append(f"  $ {cmd}")
        lines.append("")

    for fw in result.frameworks:
        lines.append(f"[{fw.framework}] status={fw.status}")
        for cmd in fw.commands:
            lines.append(f"  $ {cmd}")
        if fw.message:
            lines.append(f"  ! {fw.message}")
        if fw.agg and fw.agg.available:
            lines.append("")
            lines.append("  PD 混部 (agg):")
            rows = _candidate_rows(fw.agg.candidates or ([fw.agg.best] if fw.agg.best else []), fw.framework, "agg", "agg")
            lines.extend("  " + line for line in _make_table(rows).splitlines())
        variants = list(getattr(fw, "disagg_variants", []) or [])
        if variants:
            is_pd_ratio = any(v.pd_ratio for v in variants)
            lines.append("")
            lines.append(
                f"  PD 分离 (disagg) - PD-ratio 组合 {len(variants)} 组:"
                if is_pd_ratio
                else f"  PD 分离 (disagg) - 变体 {len(variants)} 组:"
            )
            for variant in variants:
                lines.append("")
                lines.append("    " + _variant_headline(variant))
                rows: list[list[str]] = []
                tag = variant.label or "disagg"
                if variant.prefill:
                    rows += _candidate_rows([variant.prefill], fw.framework, "disagg", f"{tag}_prefill")
                if variant.decode:
                    rows += _candidate_rows([variant.decode], fw.framework, "disagg", f"{tag}_decode")
                if variant.best:
                    rows += _candidate_rows([variant.best], fw.framework, "disagg", f"{tag}_overall")
                if not rows:
                    rows = _candidate_rows(variant.candidates, fw.framework, "disagg", tag)
                lines.extend("    " + line for line in _make_table(rows).splitlines())
            pd_candidates = list(getattr(fw, "pd_ratio_candidates", []) or [])
            if pd_candidates:
                lines.append("")
                lines.append(
                    f"  PD 分离候选汇总（按单 D 实例 decode 吞吐降序，共 {len(pd_candidates)} 条）:"
                )
                lines.extend(
                    "    " + line
                    for line in _make_table(_pd_candidate_rows(pd_candidates), _PD_TABLE_HEADER).splitlines()
                )
        elif fw.disagg and fw.disagg.available:
            lines.append("")
            lines.append("  PD 分离 (disagg):")
            rows = []
            if fw.disagg.prefill:
                rows += _candidate_rows([fw.disagg.prefill], fw.framework, "disagg", "disagg_prefill")
            if fw.disagg.decode:
                rows += _candidate_rows([fw.disagg.decode], fw.framework, "disagg", "disagg_decode")
            if fw.disagg.best:
                rows += _candidate_rows([fw.disagg.best], fw.framework, "disagg", "disagg_overall")
            if not rows:
                rows = _candidate_rows(fw.disagg.candidates, fw.framework, "disagg", "disagg")
            lines.extend("  " + line for line in _make_table(rows).splitlines())
        lines.append("")

    rec = result.recommendation
    if rec and rec.get("best"):
        best = rec["best"]
        lines.append("-" * 100)
        lines.append("RECOMMENDATION")
        lines.append(
            f"  serving_mode={rec.get('serving_mode')}  mode={rec.get('mode')}  framework={rec.get('framework')}"
        )
        lines.append(f"  throughput={_fmt(to_float(best.get('throughput_tokens_per_s')))} tokens/s")
        lines.append(f"  ttft={_fmt(to_float(best.get('ttft_ms')))} ms   tpot={_fmt(to_float(best.get('tpot_ms')))} ms")
        lines.append(f"  parallel={best.get('parallel_str') or '-'}   backend={best.get('backend') or '-'}")
        lines.append(f"  reason={rec.get('reason')}")
    else:
        lines.append("-" * 100)
        lines.append("RECOMMENDATION: 无可用结果")

    if result.errors:
        lines.append("")
        lines.append("ERRORS:")
        lines.extend(f"  - {e}" for e in result.errors)

    lines.append("")
    lines.append(f"artifacts   : {result.artifacts.get('run_dir')}")
    lines.append(f"run.json    : {result.artifacts.get('run_json')}")
    lines.append("=" * 100)
    return "\n".join(lines)


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """读取已落盘的统一结果（供 HTTP / 后续解析复用）。"""
    path = Path(run_dir) / "run.json"
    return json.loads(path.read_text(encoding="utf-8"))
