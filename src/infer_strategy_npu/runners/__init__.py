"""框架 Runner 集合。"""

from infer_strategy_npu.runners.aic_npu import AicNpuRunner
from infer_strategy_npu.runners.msmodeling import MsmodelingRunner

__all__ = ["AicNpuRunner", "MsmodelingRunner"]
