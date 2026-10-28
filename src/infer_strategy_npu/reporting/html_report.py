"""把单次寻优结果渲染成**单文件静态 HTML 报告**。

设计目标：三方平台只调我们的任务提交接口，任务完成后想直接看「推理策略寻优 / 任务详情」，
因此产物必须满足：

- 单文件：CSS / JS / 数据全部内联，双击即可打开，不依赖任何服务与网络；
- 与在线详情页同构：直接复用 ``ui/styles.css`` 与 ``ui/app.js``（唯一渲染实现），
  只把数据以 ``window.__REPORT_DATA__`` 注入，前端进入 report-mode 不再发请求；
- 体积小：剔除 ``raw``（每条候选 48 字段的原始行）与 ``disagg_variants[].pd_ratio_candidates``
  （与 framework 级完全重复的两份副本）。

用法::

    from infer_strategy_npu.reporting import write_report_from_run_json
    write_report_from_run_json(Path("_verify_results/<task_id>/run.json"))
    # -> _verify_results/<task_id>/report.html
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPORT_NAME = "report.html"


def ui_dir() -> Path:
    """ui/ 目录：源码运行时在仓库根，wheel 安装时可能不存在。"""
    return Path(__file__).resolve().parents[3] / "ui"


# ------------------------------------------------------------------ 瘦身


def slim_result(result: dict | None) -> dict | None:
    """剔掉体积大且页面不再渲染的数据，同时保证报告与在线页面渲染一致。

    - ``raw``：每条候选挂着的 48 字段原始行，页面不展示，一律剔除；
    - framework 级 ``pd_ratio_candidates``：候选汇总块已下线，若各
      ``disagg_variants[]`` 都自带候选（正常情况下都有），这份副本就是纯冗余。
    """
    if not result:
        return result
    data = _drop_raw(result)
    for fw in data.get("frameworks") or []:
        variants = fw.get("disagg_variants") or []
        if variants and all((v.get("pd_ratio_candidates") or v.get("candidates")) for v in variants):
            fw.pop("pd_ratio_candidates", None)
    return data


def _drop_raw(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _drop_raw(v) for k, v in obj.items() if k != "raw"}
    if isinstance(obj, list):
        return [_drop_raw(v) for v in obj]
    return obj


# ------------------------------------------------------------------ 渲染


def render_report_html(result: dict, *, base: Path | None = None, title: str | None = None) -> str:
    """把 run.json 结构渲染成完整 HTML 字符串。

    ``base`` 为 ui/ 目录，缺省用仓库内的 ``ui/``；找不到时退化为最小骨架（仍然可用）。
    """
    base = base or ui_dir()
    index_html = _read(base / "index.html")
    css = _read(base / "styles.css")
    js = _read(base / "app.js")

    payload = json.dumps(slim_result(result), ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")

    if index_html and css and js:
        html = index_html.replace(
            '<link rel="stylesheet" href="styles.css">', f"<style>\n{css}\n</style>"
        )
        html = html.replace(
            '<script src="app.js"></script>',
            f"<script>window.__REPORT_DATA__ = {payload};</script>\n<script>\n{js}\n</script>",
        )
        if title:
            html = re.sub(
                r"<title>.*?</title>",
                f"<title>{_esc(title)} · 推理策略寻优报告</title>",
                html,
                count=1,
                flags=re.S,
            )
        # 静态报告没有「任务记录」列表可返回，直接剔除右上角的按钮组
        # （静态报告 / 返回任务记录），避免它们出现在产物里（实时界面仍保留）。
        html = re.sub(
            r'<div[^>]*>\s*<button[^>]*id="btn-report"[\s\S]*?</div>',
            "",
            html,
            count=1,
        )
        return html

    # ui/ 缺失（如 wheel 安装）：退化成最小可打开页面 + 原始 JSON
    return (
        "<!DOCTYPE html><html lang='zh-CN'><meta charset='utf-8'>"
        f"<title>{_esc(title or '推理策略寻优报告')}</title>"
        "<body style='font-family:Consolas,monospace;padding:20px'>"
        "<p>未找到 ui/ 静态资源，以下为原始结果（run.json）：</p>"
        f"<pre id='data' style='white-space:pre-wrap'></pre>"
        f"<script>window.__REPORT_DATA__ = {payload};"
        "document.getElementById('data').textContent = JSON.stringify(window.__REPORT_DATA__, null, 2);"
        "</script></body></html>"
    )


def write_report(result: dict, run_dir: Path, *, base: Path | None = None) -> Path:
    """把报告写到 ``<run_dir>/report.html``，返回文件路径。"""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / REPORT_NAME
    target.write_text(render_report_html(result, base=base, title=str(result.get("run_id") or run_dir.name)), encoding="utf-8")
    return target


def write_report_from_run_json(run_json: Path, *, base: Path | None = None) -> Path | None:
    """从 run.json 直接产出报告；文件不存在/解析失败时返回 None。"""
    run_json = Path(run_json)
    try:
        result = json.loads(run_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return write_report(result, run_json.parent, base=base)


# ------------------------------------------------------------------ 内部


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
