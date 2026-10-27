"""PD 分离（PD-ratio）结果的吞吐派生公式。

msmodeling 的 PD-ratio 结果表本身**不带**吞吐列，其终端图
``<device>_<model>_pd_decode_tps: Throughput vs concurrency`` 的纵轴是按
``concurrency_d / tpot_d * 1000`` 现场复算的：

- 原生计算点在 ``serving_cast/service/disagg_throughput_optimizer.py``::

      output_throughput = concurrency / tpot * 1000
      token_s_device    = output_throughput / dp / pp / tp

- PD-ratio 图在 ``serving_cast/service/optimizer_curve_plots.py::_pd_tps_curve_df``::

      work["token/s"] = concurrency / tpot * 1000

本模块在整合层按**同一公式**逐行复算，并把单实例口径放大到集群口径，
使 PD-ratio 结果可以直接与 aic-npu 的 ``tokens/s`` / ``tokens/s/gpu`` 比较。

所有函数对缺失输入返回 ``None``，不做除零。

注意：msmodeling 的 ``tpot`` 已经过 ``_fold_decode_latency_ms()`` 折叠，
投机解码（dflash/dspark/mtp）的加速已包含在内，因此这里无需再乘接受率。
"""

from __future__ import annotations


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    try:
        if float(denominator) == 0:
            return None
    except (TypeError, ValueError):
        return None
    return float(numerator) / float(denominator)


def decode_tps_per_instance(concurrency_d: float | None, tpot_d: float | None) -> float | None:
    """单 D 实例的 decode 吞吐（token/s）= ``concurrency_d / tpot_d * 1000``。

    与 msmodeling ``pd_decode_tps`` 图纵轴同口径，也是候选排序键。
    """
    return None if concurrency_d is None or tpot_d is None else _safe_div(concurrency_d, tpot_d) * 1000.0


def tokens_per_s_per_device(tokens_per_s: float | None, devices: float | None) -> float | None:
    """单卡吞吐 = 该级 token/s / 该级卡数（对应 msmodeling 原生 ``token/s/device``）。"""
    return _safe_div(tokens_per_s, devices)


def prefill_tps_per_instance(p_qps: float | None, input_length: int | None) -> float | None:
    """单 P 实例的 prefill 吞吐（token/s）= ``p_qps * input_length``。"""
    if p_qps is None or input_length is None:
        return None
    return float(p_qps) * float(input_length)


def cluster_tokens_per_s(req_per_s: float | None, output_length: int | None) -> float | None:
    """集群端到端 token 吞吐（token/s）= ``balanced_qps * output_length``。

    与 aic-npu 的 ``tokens/s`` 同口径。
    """
    if req_per_s is None or output_length is None:
        return None
    return float(req_per_s) * float(output_length)


def request_latency_ms(ttft_ms: float | None, tpot_ms: float | None, output_length: int | None) -> float | None:
    """单请求时延（ms）= ``ttft + tpot * (output_length - 1)``，与 aic-npu 一致。"""
    if tpot_ms is None or output_length is None:
        return None
    steps = max(int(output_length) - 1, 1)
    total = float(tpot_ms) * steps
    if ttft_ms is not None:
        total += float(ttft_ms)
    return total


def tokens_per_s_per_user(output_length: int | None, latency_ms: float | None) -> float | None:
    """每用户吞吐（token/s/user）= ``output_length / request_latency``。

    与 aic-npu 的 ``tokens/s/user`` 同口径。
    """
    if latency_ms is None or output_length is None:
        return None
    return _safe_div(float(output_length) * 1000.0, latency_ms)
