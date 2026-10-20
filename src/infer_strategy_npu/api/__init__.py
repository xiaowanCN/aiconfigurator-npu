"""HTTP 服务（可选依赖 fastapi/uvicorn/pydantic）。

仅在以 ``infer-strategy-npu-server`` 或 ``uvicorn infer_strategy_npu.api.app:app``
启动时才会被导入，CLI 路径不加载本包，避免强依赖 Web 框架。
"""
