"""NPU Event-based timing engine for operator benchmarking.

Aligned with AIConfigurator's benchmark_with_power() timing rules:
- NPU Graph capture + replay to eliminate Python dispatch overhead
- Single Event pair wrapping multiple runs (no per-iteration synchronize)
- latency = total_elapsed / num_runs / repeat_n
- Fallback to eager execution if graph capture fails

Timing modes (selectable via --timing-mode):
- auto  (default): runs both graph and eager, takes the minimum latency.
- graph           : prefers NPU graph; uses graph when capture succeeds,
                    falls back to eager only if capture fails. Mirrors the
                    upstream AIConfigurator use_cuda_graph=True convention.
- eager           : never captures; times eager execution only.
"""

import logging
from dataclasses import dataclass
from typing import Callable

import torch

logger = logging.getLogger(__name__)

# Set once the first graph-capture failure has been reported, so a systematic
# failure doesn't emit the same warning thousands of times.
_GRAPH_CAPTURE_ERROR_LOGGED = False

# Iterations run on the capture stream before capturing, so that all workspace
# and memory-pool allocation is done before capture begins.
_CAPTURE_WARMUP_ITERS = 3


@dataclass(frozen=True)
class BenchResult:
    """Immutable benchmark result."""

    avg_us: float
    num_runs: int
    repeat_n: int
    used_graph: bool = False
    graph_us: float = 0.0
    eager_us: float = 0.0
    graph_attempted: bool = True


def describe_mode(result: BenchResult) -> str:
    """Human-readable timing mode for a result.

    ``used_graph=False`` is ambiguous: it covers "graph disabled by the caller",
    "capture raised" and "capture worked but replay was not faster than eager".
    Distinguishing them matters — the first two mean the graph path is either
    intentionally skipped or broken, the last is a deliberate choice of the
    faster measurement.
    """
    if result.used_graph:
        return "graph"
    if not result.graph_attempted:
        return "eager (NPU graph disabled)"
    if result.graph_us:
        return (
            f"eager (graph {result.graph_us:.2f}us "
            f"> eager {result.eager_us:.2f}us)"
        )
    return "eager (NPU graph capture failed)"


def _timed_run(
    kernel_func: Callable[[], None],
    num_runs: int,
    repeat_n: int,
    graph=None,
) -> float:
    """Run kernel_func num_runs times and return average latency in us.

    If graph is provided, uses graph.replay() instead of calling kernel_func.
    """
    start_evt = torch.npu.Event(enable_timing=True)
    end_evt = torch.npu.Event(enable_timing=True)

    start_evt.record()
    for _ in range(num_runs):
        if graph is not None:
            graph.replay()
        else:
            for _ in range(repeat_n):
                kernel_func()
    end_evt.record()
    torch.npu.synchronize()

    return start_evt.elapsed_time(end_evt) * 1000.0 / num_runs / repeat_n


def benchmark_npu(
    kernel_func: Callable[[], None],
    warmup_iters: int = 20,
    num_runs: int = 100,
    repeat_n: int = 1,
    timing_mode: str = "auto",
) -> BenchResult:
    """Warmup + time kernel_func on NPU. Timing mode selects graph/eager policy.

    ``timing_mode`` controls how the NPU graph and eager paths are combined:
      - "auto"  (default): capture graph, time both graph and eager, take the
                 minimum latency. Best achievable device-side latency.
      - "graph"           : capture graph; use the graph number when capture
                 succeeds, and fall back to eager only if capture fails.
                 Mirrors the upstream AIConfigurator
                 ``benchmark_with_power(..., use_cuda_graph=True)`` convention
                 (graph is preferred, not compared-and-min'd).
      - "eager"           : never capture; time eager execution only.

    Graph capture only works if the timed closure is purely device-side: vLLM
    metadata and forward context must be built OUTSIDE ``kernel_func``. Anything
    that copies host->device inside the closure (e.g. building a shape tensor
    per call) makes capture fail with ``aclrtMemcpy ... error 107030``.
    """
    global _GRAPH_CAPTURE_ERROR_LOGGED

    for _ in range(warmup_iters):
        kernel_func()
    torch.npu.synchronize()

    if timing_mode == "eager":
        eager_us = _timed_run(kernel_func, num_runs, repeat_n)
        return BenchResult(
            avg_us=eager_us,
            num_runs=num_runs,
            repeat_n=repeat_n,
            used_graph=False,
            eager_us=eager_us,
            graph_attempted=False,
        )

    graph = None
    graph_us = 0.0
    try:
        graph = torch.npu.NPUGraph()

        # NPU graphs refuse to be captured on the default/legacy stream
        # ("NPU graphs must be captured on a non-default stream"), so use a
        # dedicated side stream. Two things are required:
        #   1. the side stream must wait for the default one, otherwise the
        #      inputs produced during warmup are not visible to the capture;
        #   2. the op must be warmed up *on that same stream*, so all
        #      workspace / memory-pool allocation happens before capture —
        #      allocating during capture fails as well.
        # Replaying afterwards on the default stream is explicitly allowed.
        side_stream = torch.npu.Stream()
        side_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(side_stream):
            for _ in range(_CAPTURE_WARMUP_ITERS):
                kernel_func()
        torch.npu.synchronize()

        with torch.npu.graph(graph, stream=side_stream):
            for _ in range(repeat_n):
                kernel_func()

        # Capture ran on side_stream; let the default stream catch up before
        # the timed replay / eager runs below.
        torch.npu.current_stream().wait_stream(side_stream)

        # Warm up the ACTUAL execution path (replay), not just the eager one.
        for _ in range(_CAPTURE_WARMUP_ITERS):
            graph.replay()
        torch.npu.synchronize()

        graph_us = _timed_run(kernel_func, num_runs, repeat_n, graph=graph)
    except Exception as e:
        # Log the real cause once (with traceback at DEBUG) instead of
        # repeating a context-free warning for every single shape.
        if not _GRAPH_CAPTURE_ERROR_LOGGED:
            _GRAPH_CAPTURE_ERROR_LOGGED = True
            logger.warning(
                "NPU graph capture failed (%s: %s); falling back to eager mode "
                "for this and all remaining shapes. "
                "Set LOG_LEVEL=DEBUG for the full traceback.",
                type(e).__name__,
                e,
            )
            logger.debug("NPU graph capture traceback:", exc_info=True)
        else:
            logger.debug("NPU graph capture failed again: %s: %s", type(e).__name__, e)
        graph_us = 0.0

    # Release the graph — and crucially its private memory pool — BEFORE timing
    # the eager path. A captured graph sequesters intermediate tensors into a
    # pool that is not always reclaimed when the object is garbage collected;
    # the NVIDIA collector (collector/helper.py) documents this explicitly and
    # tears the graph down in a finally block because otherwise scratch memory
    # accumulates across shapes until the worker OOMs. Keeping the graph alive
    # while also running eager would additionally double the peak footprint,
    # which is why large (batch x seq) points used to OOM.
    if graph is not None:
        graph = None
        torch.npu.empty_cache()

    eager_us = _timed_run(kernel_func, num_runs, repeat_n)

    if timing_mode == "graph":
        # Prefer graph; fall back to eager only when capture failed.
        used_graph = bool(graph_us)
    else:  # "auto": take the faster of the two measurements.
        used_graph = bool(graph_us) and graph_us <= eager_us

    return BenchResult(
        avg_us=graph_us if used_graph else eager_us,
        num_runs=num_runs,
        repeat_n=repeat_n,
        used_graph=used_graph,
        graph_us=graph_us,
        eager_us=eager_us,
    )
