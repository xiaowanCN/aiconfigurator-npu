"""结果产出物：静态报告等。"""

from infer_strategy_npu.reporting.html_report import (
    render_report_html,
    write_report,
    write_report_from_run_json,
)

__all__ = ["render_report_html", "write_report", "write_report_from_run_json"]
