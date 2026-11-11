#!/usr/bin/env python3
"""infer-strategy-npu-task-enter.py 的离线单元测试（不需要 msmodeling / NPU）。

运行：python scripts/run_infer_strategy_task_enter_tests.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "infer-strategy-npu-task-enter.py"
_spec = importlib.util.spec_from_file_location("infer_strategy_task_enter", ENTRY)
mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = mod  # dataclass 解析字符串注解时需要能找到本模块
_spec.loader.exec_module(mod)


def agg_candidate(tps: float) -> dict:
    return {
        "throughput_tokens_per_s": tps,
        "throughput_tokens_per_s_per_device": tps / 32,
        "throughput_req_per_s": 1.0,
        "ttft_ms": 100.0,
        "tpot_ms": 20.0,
        "meets_slo": True,
        "concurrency": 8,
        "batch_size": 4,
        "num_devices": 32,
        "parallel": {"tp": 8},  # ParallelConfig.to_dict() 会剔除 None，投影层需补齐
        "framework": "msmodeling",
        "raw": {"internal": 1},
    }


def pd_candidate(decode_tps: float, cluster_tps: float, label: str) -> dict:
    side = {
        "instances": 1,
        "devices_per_instance": 8,
        "devices_total": 8,
        "parallel": {"tp": 8},
        "parallel_str": "tp8",
        "batch_size": 4,
        "concurrency": 8,
        "qps_per_instance": 1.0,
        "qps_allocated": 1.0,
        "ttft_ms": 100.0,
        "tpot_ms": 20.0,
        "tokens_per_s_per_instance": decode_tps,
        "tokens_per_s_per_device": decode_tps / 8,
        "tokens_per_s_cluster": cluster_tps,
    }
    return {
        "framework": "msmodeling",
        "label": label,
        "decode_tps_per_instance": decode_tps,
        "throughput_tokens_per_s": cluster_tps,
        "prefill": side,
        "decode": side,
        "raw": {"internal": 1},
    }


def fake_result(
    run_dir: Path,
    *,
    agg_candidates: list[dict] | None = None,
    pd_candidates: list[dict] | None = None,
    grid: bool = True,
    message: str | None = None,
) -> dict:
    return {
        "run_id": "unit-test-run",
        "status": "ok",
        "routing": {"framework": "msmodeling", "commands": []},
        "frameworks": [
            {
                "framework": "msmodeling",
                "status": "ok",
                "message": message,
                "commands": ["python -m cli.inference.throughput_optimizer models/GLM5.2 --device ATLAS_950_486T_96G"],
                "artifacts": {"pd_ratio_grid": [{"label": "P8D8"}] if grid else []},
                "agg": {"candidates": agg_candidates} if agg_candidates is not None else None,
                "disagg": None,
                "disagg_variants": [],
                "pd_ratio_candidates": pd_candidates or [],
            }
        ],
        "errors": [],
        "artifacts": {"run_dir": str(run_dir)},
    }


class TaskEnterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.log_file = self.root / "outputs" / "run.log"
        self.result_file = self.root / "outputs" / "result.json"

    def tearDown(self) -> None:
        import logging

        logger = logging.getLogger(mod._LOGGER_NAME)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
        self.tmp.cleanup()

    def args(self, **overrides) -> list[str]:
        params = {
            "--model": "DeepSeek-V4.1-Flash-MXFP4",
            "--device": "950DT",
            "--num_devices": "32",
            "--input_length": "4096",
            "--output_length": "1024",
            "--ttft_limit": "20",
            "--tpot_limit": "100",
            "--serving_modes": "2",
            "--log-file": str(self.log_file),
            "--result-file": str(self.result_file),
        }
        params.update(overrides)
        argv: list[str] = []
        for key, value in params.items():
            if value is None:
                continue
            argv.extend([key, str(value)])
        return argv

    # ---- 配置表 ----

    def test_model_plans(self) -> None:
        self.assertEqual(len(mod.MODEL_PLANS), 9)
        mxfp4 = [name for name, plan in mod.MODEL_PLANS.items() if plan.mxfp4_group_size]
        self.assertEqual(set(mxfp4), {"GLM-5.2-w4a4c8-mxfp4", "DeepSeek-V4.1-Flash-MXFP4"})
        for plan in mod.MODEL_PLANS.values():
            self.assertIn(plan.linear, mod._LINEAR_VALUE_DOMAIN)
            self.assertIn(plan.non_expert_linear, mod._LINEAR_VALUE_DOMAIN)
            self.assertIn(plan.attention, mod._ATTENTION_VALUE_DOMAIN)
        self.assertEqual(mod.PD_RATIO_SIZES, (8, 16, 32))
        self.assertEqual(mod.TOPK, 1)

    def test_r1_0528_w8a8_plan(self) -> None:
        plan = mod.MODEL_PLANS["DeepSeek-R1-0528-W8A8"]
        self.assertEqual(plan.model_id, "deepseek-ai/DeepSeek-R1")
        self.assertEqual(plan.local_dir, "DeepSeek-R1")
        self.assertEqual(
            (plan.linear, plan.non_expert_linear, plan.attention),
            ("W8A8_DYNAMIC", "W8A8_DYNAMIC", "DISABLED"),
        )
        self.assertIsNone(plan.mxfp4_group_size)

        config = mod.parse_and_validate(self.args(**{"--model": "DeepSeek-R1-0528-W8A8"}))
        echo = mod.build_request_echo(config)
        self.assertEqual(echo["quantize-linear-action"], "W8A8_DYNAMIC")
        self.assertEqual(echo["quantize-non-expert-linear-action"], "W8A8_DYNAMIC")
        self.assertEqual(echo["quantize-attention-action"], "DISABLED")
        self.assertEqual(echo["mxfp4-group-size"], "32")  # 仅回显默认值，不下发

        request = mod.build_request(config, ["disagg"], 16)
        self.assertEqual(request.model, "deepseek-ai/DeepSeek-R1")
        self.assertEqual(request.quantize, "ENABLE")
        self.assertEqual(request.quantize_linear_action, "W8A8_DYNAMIC")
        self.assertEqual(request.quantize_non_expert_linear_action, "W8A8_DYNAMIC")
        self.assertEqual(request.quantize_attention_action, "DISABLED")
        self.assertIsNone(request.extra_msmodeling_args)
        self.assertEqual(request.pd_ratio_sizes, [16])

    def test_device_specs(self) -> None:
        self.assertEqual(mod.DEVICE_SPECS["910B2"].ms_device, "ATLAS_800_A2_376T_64G")
        self.assertEqual(mod.DEVICE_SPECS["910B4"].ms_device, "ATLAS_800_A2_280T_64G")
        self.assertEqual(mod.DEVICE_SPECS["950DT"].ms_device, "ATLAS_950_486T_96G")
        self.assertTrue(all(spec.gpus_per_node == 8 for spec in mod.DEVICE_SPECS.values()))

    # ---- 入参校验 ----

    def test_defaults_units_and_num_nodes(self) -> None:
        config = mod.parse_and_validate(
            self.args(**{"--ttft_limit": None, "--tpot_limit": None, "--serving_modes": None})
        )
        self.assertEqual(config.ttft_limit_s, 20.0)
        self.assertEqual(config.ttft_limit_ms, 20000)
        self.assertEqual(config.tpot_limit_ms, 100)
        self.assertEqual(config.num_nodes, 4)
        self.assertEqual(config.serving_modes, 1)  # 默认值
        self.assertTrue(config.ttft_defaulted)
        self.assertTrue(config.tpot_defaulted)

    def test_ttft_floor_conversion(self) -> None:
        config = mod.parse_and_validate(self.args(**{"--ttft_limit": "0.001"}))
        self.assertEqual(config.ttft_limit_ms, 1)

    def test_aliases(self) -> None:
        argv = [
            "--model", "GLM-5.2-BF16", "--device", "910B2",
            "--num-devices", "64", "--input-length", "128", "--output-length", "131072",
            "--ttft-limit", "60", "--tpot-limit", "1000", "--serving-modes", "0",
            "--log_file", str(self.log_file), "--result_file", str(self.result_file),
        ]
        config = mod.parse_and_validate(argv)
        self.assertEqual(config.num_devices, 64)
        self.assertEqual(config.num_nodes, 8)
        self.assertEqual(config.ttft_limit_ms, 60000)
        self.assertEqual(config.serving_modes, 0)

    def test_invalid_values(self) -> None:
        cases = [
            {"--model": "GLM-5.2-FP16"},
            {"--device": "950X"},
            {"--num_devices": "12"},
            {"--num_devices": "264"},
            {"--input_length": "127"},
            {"--ttft_limit": "0"},
            {"--ttft_limit": "61"},
            {"--tpot_limit": "0"},
            {"--serving_modes": "3"},
            {"--result-file": str(self.root / "result.txt")},
        ]
        for override in cases:
            with self.subTest(override=override), self.assertRaises(ValueError):
                mod.parse_and_validate(self.args(**override))

    # ---- request 类型（严格按 4.1.3 参数说明表） ----

    def test_request_echo_types(self) -> None:
        config = mod.parse_and_validate(self.args())
        echo = mod.build_request_echo(config)
        for key in ("num_devices", "num_nodes", "input_length", "output_length",
                    "tpot_limit", "serving_modes", "topk"):
            self.assertIsInstance(echo[key], int)
            self.assertNotIsInstance(echo[key], bool)
        self.assertIsInstance(echo["ttft_limit"], float)
        for key in ("model", "device", "quantize-linear-action",
                    "quantize-non-expert-linear-action", "mxfp4-group-size",
                    "quantize-attention-action"):
            self.assertIsInstance(echo[key], str)
        self.assertEqual(echo["mxfp4-group-size"], "32")
        self.assertEqual(echo["topk"], 1)

    def test_group_size_defaults_to_32(self) -> None:
        """方案表未解析到配对 group-size 时，request 也按默认值 32 回显。"""
        mxfp4 = mod.build_request_echo(mod.parse_and_validate(self.args()))
        bf16 = mod.build_request_echo(
            mod.parse_and_validate(self.args(**{"--model": "GLM-5.2-BF16"}))
        )
        self.assertEqual(mxfp4["mxfp4-group-size"], "32")
        self.assertEqual(bf16["mxfp4-group-size"], "32")
        self.assertEqual(list(bf16), list(mod._REQUEST_KEY_ORDER))
        self.assertEqual(len(bf16), 14)
        self.assertEqual(mod._empty_echo()["mxfp4-group-size"], "32")
        self.assertEqual(
            mod.best_effort_request_echo(self.args(**{"--model": "NOT-ALLOWED"}))[
                "mxfp4-group-size"
            ],
            "32",
        )

    # ---- 下发给 msmodeling 的参数 ----

    def test_build_request_fixed_flags(self) -> None:
        config = mod.parse_and_validate(self.args())
        request = mod.build_request(config, ["disagg"], 8)
        self.assertEqual(request.framework, "msmodeling")
        self.assertEqual(request.quantize, "ENABLE")
        self.assertEqual(request.quantize_linear_action, "MXFP4")
        self.assertEqual(request.quantize_non_expert_linear_action, "FP8")
        self.assertEqual(request.quantize_attention_action, "DISABLED")
        self.assertEqual(request.extra_msmodeling_args, ["--mxfp4-group-size", "32"])
        self.assertEqual(request.pd_ratio_sizes, [8])
        self.assertEqual(request.serving_modes, ["disagg"])
        self.assertEqual(request.jobs, 32)
        self.assertEqual(request.batch_range, [4, 512])
        self.assertEqual(request.max_batched_tokens, 8192)
        self.assertEqual(request.ttft_limit, 20000)
        self.assertIsNone(request.speculative_method)
        self.assertIsNone(request.out_dir)

        bf16 = mod.parse_and_validate(self.args(**{"--model": "GLM-5.2-BF16"}))
        self.assertIsNone(mod.build_request(bf16, ["disagg"], 16).extra_msmodeling_args)
        self.assertEqual(mod.build_request(bf16, ["disagg"], 16).pd_ratio_sizes, [16])

    # ---- 逐组调用编排 ----

    def test_only_three_symmetric_calls(self) -> None:
        config = mod.parse_and_validate(self.args())
        calls: list[object] = []
        run_dir = self.root / "artifacts"

        def fake_recommend(request, *, log=None, run_id=None, on_progress=None):
            calls.append(request)
            return SimpleNamespace(to_dict=lambda: fake_result(run_dir, grid=True))

        with patch.object(mod, "load_recommend", return_value=fake_recommend):
            mod.run_all(config, _silent_logger())

        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[0].serving_modes, ["agg"])
        sizes = []
        for request in calls[1:]:
            self.assertEqual(request.serving_modes, ["disagg"])
            self.assertEqual(len(request.pd_ratio_sizes), 1)
            sizes.append(request.pd_ratio_sizes[0])
        self.assertEqual(sizes, [8, 16, 32])

    # ---- 结果组装与状态机 ----

    def _run_main(self, results: list[dict], argv: list[str] | None = None) -> tuple[int, dict]:
        self.result_file.unlink(missing_ok=True)  # 与上一个子用例隔离
        queue = list(results)
        observed: list[bool] = []

        def fake_recommend(request, *, log=None, run_id=None, on_progress=None):
            observed.append(not self.result_file.exists())  # 运行期不得写结果文件
            data = queue.pop(0) if queue else results[-1]
            return SimpleNamespace(to_dict=lambda: data)

        with patch.object(mod, "load_recommend", return_value=fake_recommend):
            exit_code = mod.main(self.args() if argv is None else argv)
        self.assertTrue(all(observed), "运行期 --result-file 不应存在")
        payload = json.loads(self.result_file.read_text(encoding="utf-8"))
        return exit_code, payload

    def test_success_payload_fields_and_d_side_sort(self) -> None:
        run_dir = self.root / "artifacts"
        results = [
            fake_result(run_dir, agg_candidates=[agg_candidate(100.0), agg_candidate(50.0)]),
            fake_result(run_dir, pd_candidates=[pd_candidate(5.0, 9999.0, "P8D8")]),
            fake_result(run_dir, pd_candidates=[pd_candidate(30.0, 100.0, "P16D16")]),
            fake_result(run_dir, pd_candidates=[pd_candidate(10.0, 8888.0, "P32D32")]),
        ]
        exit_code, payload = self._run_main(results)
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["code"], 0)
        inner = payload["common"]["infer_strategy_npu"]
        self.assertEqual((inner["code"], inner["status"]), (0, "success"))
        data = inner["data"]
        self.assertEqual(set(data), {"created_at", "started_at", "finished_at", "request", "result"})
        self.assertTrue(all("+08:00" in data[key] for key in ("created_at", "started_at", "finished_at")))

        agg = data["result"]["agg"]["candidates"]
        self.assertEqual(len(agg), 1)
        self.assertEqual(agg[0]["throughput_tokens_per_s"], 100.0)
        self.assertEqual(set(agg[0]["parallel"]), {"tp", "pp", "dp", "ep"})
        self.assertIsNone(agg[0]["parallel"]["pp"])
        self.assertNotIn("framework", agg[0])

        disagg = data["result"]["disagg"]["candidates"]
        self.assertEqual(len(disagg), 1)
        # 按 D 侧吞吐取最高（30），而不是按集群吞吐（9999/8888）
        self.assertEqual(disagg[0]["decode"]["tokens_per_s_per_instance"], 30.0)
        self.assertEqual(set(disagg[0]), {"prefill", "decode"})
        for side in ("prefill", "decode"):
            self.assertEqual(set(disagg[0][side]), set(mod.SIDE_FIELDS) | {"parallel"})
            self.assertEqual(set(disagg[0][side]["parallel"]), {"tp", "pp", "dp", "ep"})
            self.assertNotIn("qps_per_instance", disagg[0][side])

    def test_empty_shapes(self) -> None:
        empty = fake_result(self.root / "artifacts", grid=False)
        _, payload_agg = self._run_main([empty], self.args(**{"--serving_modes": "0"}))
        inner_agg = payload_agg["common"]["infer_strategy_npu"]
        self.assertEqual((inner_agg["code"], inner_agg["status"]), (3, "empty"))
        self.assertEqual(inner_agg["data"]["result"], {"agg": {}, "disagg": {}})

        _, payload_disagg = self._run_main([empty] * 3, self.args(**{"--serving_modes": "1"}))
        inner_disagg = payload_disagg["common"]["infer_strategy_npu"]
        self.assertEqual((inner_disagg["code"], inner_disagg["status"]), (3, "empty"))
        self.assertEqual(inner_disagg["data"]["result"], {"agg": {}, "disagg": {}})

    def test_no_result_is_empty_not_failed(self) -> None:
        empty = fake_result(self.root / "artifacts", grid=False)
        exit_code, payload = self._run_main([empty] * 4)
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["code"], 0)
        self.assertEqual(payload["status"], "success")
        inner = payload["common"]["infer_strategy_npu"]
        self.assertEqual((inner["code"], inner["status"]), (3, "empty"))
        self.assertEqual(inner["data"]["result"], {"agg": {}, "disagg": {}})

    def test_fatal_marker_is_failed(self) -> None:
        fatal = fake_result(
            self.root / "artifacts", grid=False, message="failed to import msmodeling: No module named msmodeling"
        )
        exit_code, payload = self._run_main([fatal] * 4)
        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["code"], 1)
        self.assertEqual(payload["common"]["infer_strategy_npu"]["code"], 1)
        self.assertEqual(
            payload["common"]["infer_strategy_npu"]["data"]["result"],
            {"agg": {}, "disagg": {}},
        )

    def test_invalid_input_still_writes_full_json(self) -> None:
        stamp = "2026-11-09T09:06:16+08:00"
        with patch.object(mod, "_now_iso", return_value=stamp):
            exit_code = mod.main(self.args(**{"--model": "NOT-ALLOWED"}))
        self.assertEqual(exit_code, 1)
        payload = json.loads(self.result_file.read_text(encoding="utf-8"))
        self.assertEqual(payload["code"], 1)
        self.assertTrue(self.log_file.exists())
        request = payload["common"]["infer_strategy_npu"]["data"]["request"]
        self.assertEqual(set(request), set(mod._empty_echo()))
        self.assertEqual(request["topk"], 1)
        result = payload["common"]["infer_strategy_npu"]["data"]["result"]
        self.assertEqual(result, {"agg": {}, "disagg": {}})
        data = payload["common"]["infer_strategy_npu"]["data"]
        # 瞬时报错：三个时刻允许同为某一秒
        self.assertEqual(
            [data[key] for key in ("created_at", "started_at", "finished_at")],
            [stamp, stamp, stamp],
        )

    def test_result_floats_rounded_half_up(self) -> None:
        run_dir = self.root / "artifacts"
        agg = agg_candidate(12.345)
        agg["ttft_ms"] = 2.675
        agg["tpot_ms"] = 1.004
        pd_item = pd_candidate(30.0, 100.0, "P8D8")
        pd_item["prefill"]["tokens_per_s_per_device"] = 3.755
        results = [
            fake_result(run_dir, agg_candidates=[agg]),
            fake_result(run_dir, pd_candidates=[pd_item]),
            fake_result(run_dir, grid=False),
            fake_result(run_dir, grid=False),
        ]
        _, payload = self._run_main(results)
        data = payload["common"]["infer_strategy_npu"]["data"]
        agg_out = data["result"]["agg"]["candidates"][0]
        self.assertEqual(agg_out["throughput_tokens_per_s"], 12.35)
        self.assertEqual(agg_out["ttft_ms"], 2.68)
        self.assertEqual(agg_out["tpot_ms"], 1.0)
        self.assertEqual(agg_out["num_devices"], 32)  # 整数不变
        self.assertIs(agg_out["meets_slo"], True)  # bool 不变
        self.assertIsNone(agg_out["parallel"]["pp"])  # null 不变
        disagg_out = data["result"]["disagg"]["candidates"][0]
        self.assertEqual(disagg_out["prefill"]["tokens_per_s_per_device"], 3.76)
        self.assertIsInstance(data["request"]["ttft_limit"], float)

        self.assertEqual(
            mod.round_json_floats({"a": 0.005, "b": -1.005, "c": 2, "d": None, "e": "x"}),
            {"a": 0.01, "b": -1.01, "c": 2, "d": None, "e": "x"},
        )
        self.assertEqual(mod.round_json_floats(0.001), 0.0)  # 边界：秒值也会被格式化

    def test_request_ttft_limit_keeps_precision(self) -> None:
        """入参回显不做 2 位取整，避免 0.001 秒被格式化成 0.0 越出 4.1.2 值域。"""
        config = mod.parse_and_validate(self.args(**{"--ttft-limit": "0.001"}))
        echo = mod.build_request_echo(config)
        payload = mod.assemble_payload(
            echo,
            mod.RunOutcome(),
            ("t0", "t1", "t2"),
            mod.Status(0, "success", "ok", 3, "empty", "no result"),
        )
        request = payload["common"]["infer_strategy_npu"]["data"]["request"]
        self.assertEqual(request["ttft_limit"], 0.001)
        self.assertEqual(config.ttft_limit_ms, 1)

    def test_timestamps_seconds_precision_with_elapsed(self) -> None:
        """秒精度 ISO8601；十几秒耗时的任务自然错开 finished_at，不做人为顺延。"""
        run_dir = self.root / "artifacts"
        results = [fake_result(run_dir, agg_candidates=[agg_candidate(100.0)])]
        results += [fake_result(run_dir, grid=False)] * 3
        created = "2026-11-09T09:06:16+08:00"
        finished = "2026-11-09T09:06:29+08:00"
        stamps = iter([created, created, finished])
        with patch.object(mod, "_now_iso", side_effect=lambda: next(stamps, finished)):
            _, payload = self._run_main(results)
        data = payload["common"]["infer_strategy_npu"]["data"]
        self.assertEqual(data["created_at"], created)
        self.assertEqual(data["started_at"], created)
        self.assertEqual(data["finished_at"], finished)
        self.assertLess(data["created_at"], data["finished_at"])
        self.assertTrue(
            all(
                "." not in stamp and stamp.endswith("+08:00")
                for stamp in (data["created_at"], data["started_at"], data["finished_at"])
            )
        )

    def test_atomic_write(self) -> None:
        path = self.root / "deep" / "result.json"
        mod.write_result_file(path, {"code": 1})
        mod.write_result_file(path, {"code": 0})
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"code": 0})
        self.assertEqual(list(path.parent.glob("*.tmp")), [])


def _silent_logger():
    import logging

    logger = logging.getLogger("task_enter_test_silent")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


if __name__ == "__main__":
    unittest.main(verbosity=2)
