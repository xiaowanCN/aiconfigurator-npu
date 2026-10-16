"""infer-strategy-npu: 统一的 NPU 推理策略推荐 CLI（整合 aic-npu 与 msmodeling）。"""

from __future__ import annotations

__version__ = "0.1.0"

from infer_strategy_npu.schema import (
    Candidate,
    FrameworkRun,
    RecommendRequest,
    ServingModeResult,
    UnifiedRunResult,
)
from infer_strategy_npu.service import recommend

__all__ = [
    "__version__",
    "recommend",
    "Candidate",
    "FrameworkRun",
    "RecommendRequest",
    "ServingModeResult",
    "UnifiedRunResult",
]
