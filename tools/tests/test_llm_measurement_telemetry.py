"""Observation regressions: NVML views overlap and unavailable bytes stay unknown."""
import ctypes
import copy
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.llm_baseline.measure_native_baseline import NvmlSampler
from tools.llm_baseline.measure_native_capacity import bound, canonical_provider_maximum, proof_scheduling_estimate
from tools.llm_baseline import measure_native_capacity as capacity
from tools.llm_baseline import measure_native_baseline as telemetry
from tools.llm_baseline import measure_portable_attention as attention
from tools.llm_baseline import run_browser_multimodal_series as browser


class TelemetryTests(unittest.TestCase):
    def test_fake_cycles_retain_warmups_and_reject_fifth_id_drift(self):
        fixed = [30, 198, 198, 504, 8085]
        for drift in (None, "cold-peer", -2, -1, 0, 4):
            calls = []; receipt = {"cycles": []}
            def fake_cycle(ordinal):
                calls.append(ordinal)
                ids = [list(fixed), list(fixed)]
                if drift == ordinal or drift == "cold-peer" and ordinal == "cold":
                    ids[1][-1] += 1
                return {"ordinal": ordinal, "selected_ids_by_owner": ids}
            with self.subTest(drift=drift):
                if drift is None:
                    capacity.collect_repeatable_cycles(receipt, fake_cycle, 2, 5)
                    self.assertEqual(calls, ["cold", -2, -1, 0, 1, 2, 3, 4])
                    self.assertEqual([x["ordinal"] for x in receipt["excluded_warmup_cycles"]], [-2, -1])
                    self.assertEqual(len(receipt["cycles"]), 5)
                else:
                    with self.assertRaisesRegex(RuntimeError, "all five returned IDs"):
                        capacity.collect_repeatable_cycles(receipt, fake_cycle, 2, 5)
                    self.assertEqual(calls[-1], "cold" if drift == "cold-peer" else drift)
                    retained = [receipt["cold_first"], *receipt.get("excluded_warmup_cycles", []), *receipt["cycles"]]
                    self.assertTrue(any(x["selected_ids_by_owner"][1][-1] != fixed[-1] for x in retained))

    def sampler(self, compute, graphics):
        sampler = object.__new__(NvmlSampler)
        sampler.devices = [ctypes.c_void_p(1)]
        class Library:
            def nvmlDeviceGetMemoryInfo(self, handle, output):
                output._obj.total = 4096; output._obj.free = 1024; output._obj.used = 3072
                return 0
        sampler.lib = Library()
        def query(rows):
            def read(handle, count, output):
                if output is None:
                    count._obj.value = len(rows)
                    return 7 if rows else 0
                for index, (pid, used) in enumerate(rows):
                    output[index].pid = pid; output[index].usedGpuMemory = used
                count._obj.value = len(rows)
                return 0
            return read
        sampler.processes = query(compute); sampler.graphics_processes = query(graphics)
        return sampler

    def test_overlapping_memberships_are_never_added(self):
        row = self.sampler([(os.getpid(), 100), (999999, 300)], [(os.getpid(), 120)]).sample()[0]
        self.assertEqual(row["worker_process_used_bytes"], 120)
        self.assertEqual(len(row["compute_processes"]), 2)
        self.assertEqual(len(row["graphics_processes"]), 1)

    def test_graphics_only_worker_is_observed(self):
        row = self.sampler([], [(os.getpid(), 96)]).sample()[0]
        self.assertEqual(row["worker_process_used_bytes"], 96)

    def test_unavailable_or_absent_worker_remains_unknown(self):
        row = self.sampler([(os.getpid(), (1 << 64) - 1)], [(999999, 32)]).sample()[0]
        self.assertIsNone(row["worker_process_used_bytes"])
        self.assertIsNone(row["compute_processes"][0]["used_bytes"])

    def test_context_estimate_counts_capacity_outputs_and_kv(self):
        graph = {"nodes": [{"outputs": {
            "k": {"tensor": "layer.0.k_rope", "dtype": "float32", "shape": [1, "S", 2, 4]},
            "v": {"tensor": "layer.0.v", "dtype": "float32", "shape": [1, "S", 2, 4]},
            "logits": {"tensor": "logits", "dtype": "float32", "shape": [1, "S", 11]}}}]}
        value = bound(graph, 512, 1000, 2, "vulkan")
        self.assertEqual(value["output_domain_bytes"], (8 + 8 + 11) * 512 * 4)
        self.assertEqual(value["kv_domain_bytes"], 16 * 512 * 4)
        self.assertEqual(value["largest_output_bytes"], 11 * 512 * 4)
        cpu = bound(graph, 512, 1000, 2, "cpu")
        self.assertEqual(cpu["device_scheduling_bytes"], 0)
        self.assertEqual(cpu["host_scheduling_bytes"] - value["host_scheduling_bytes"], 1000)
        self.assertEqual(value["device_scheduling_bytes"],
                         2 * value["per_context_scheduling_bytes"] + (1 << 30))

    def test_future_unknown_dimensions_and_dtypes_fail_closed(self):
        graph = {"nodes": [{"outputs": {"x": {"tensor": "x", "dtype": "float16", "shape": [1, "S"]}}}]}
        with self.assertRaisesRegex(RuntimeError, "F32/I32"):
            bound(graph, 128, 1000, 1, "cpu")
        graph["nodes"][0]["outputs"]["x"] = {"tensor": "x", "dtype": "float32", "shape": [1, "T"]}
        with self.assertRaises(TypeError):
            bound(graph, 128, 1000, 1, "cpu")

    def proof(self):
        return {"proofProtocol": "canonical-symbolic-domain-proof/v1",
                "resourceProtocol": "bounded-resource-maxima/v1", "graphFingerprint": "graph",
                "shapeDomainProofIdentity": "domain", "bounds": [
                    {"budgetDomainId": "maximum-tensor", "kind": 1, "maximumBytes": {"bytes": 50}},
                    {"budgetDomainId": "provider-resident", "kind": 3, "maximumBytes": {"bytes": 100},
                     "limitBytes": {"bytes": 200}, "cases": [
                         {"caseId": "ordinary", "terms": [
                             {"termId": "immutable", "upperBoundBytes": {"bytes": 40}},
                             {"termId": "context", "upperBoundBytes": {"bytes": 60}}],
                          "totalBytes": {"bytes": 100}}]}]}

    def test_provider_proof_checks_sum_max_limit_and_presence(self):
        self.assertEqual(canonical_provider_maximum(self.proof()), 100)
        for fault in ("sum", "maximum", "limit", "presence", "duplicate"):
            p = copy.deepcopy(self.proof()); b = p["bounds"][1]
            if fault == "sum": b["cases"][0]["terms"][0]["upperBoundBytes"]["bytes"] = 41
            if fault == "maximum": b["maximumBytes"]["bytes"] = 101
            if fault == "limit": b["limitBytes"]["bytes"] = 99
            if fault == "presence": del b["maximumBytes"]
            if fault == "duplicate": p["bounds"].append(copy.deepcopy(b))
            with self.subTest(fault=fault), self.assertRaises(RuntimeError):
                canonical_provider_maximum(p)

    def source_backed_plan(self):
        return {"live_contexts": 2,
                "pins": {"backend": "vulkan", "source": dict(capacity.ADMISSION_SOURCES),
                         "storage_inventory": {"source_file_logical_bytes": 30, "unique_payload_bytes": 20}},
                "source_backed_admission": {"protocol": capacity.ADMISSION_PROTOCOL,
                    "source_sha256": dict(capacity.ADMISSION_SOURCES),
                    "private_weights": {"cpu_packed_upper_bound_bytes": 12}}}

    def test_coarse_provider_is_whole_per_space_with_only_extra_cpu_packs(self):
        plan = self.source_backed_plan()
        p = self.proof(); estimate = proof_scheduling_estimate(plan, p)
        self.assertEqual(estimate["host_scheduling_bytes"], 2 * 100 + (1 << 30))
        self.assertEqual(estimate["device_scheduling_bytes"], 2 * 100 + (1 << 30))
        self.assertEqual(estimate["per_context_scheduling_bytes"], 100)
        self.assertFalse(estimate["whole_physical_peak_qualified"])
        plan["pins"]["backend"] = "cpu"
        estimate = proof_scheduling_estimate(plan, p)
        self.assertEqual(estimate["host_scheduling_bytes"], 2 * (100 + 12) + (1 << 30))
        self.assertEqual(estimate["device_scheduling_bytes"], 0)
        self.assertEqual(estimate["per_context_scheduling_bytes"], 112)

    def test_source_backed_admission_fails_closed_on_contract_or_mapping_gap(self):
        for fault in ("protocol", "source", "file", "owners", "pack", "preflight"):
            plan = self.source_backed_plan()
            if fault == "protocol": plan["source_backed_admission"]["protocol"] = "future/v2"
            if fault == "source": plan["pins"]["source"][next(iter(capacity.ADMISSION_SOURCES))] = "changed"
            if fault == "file": plan["pins"]["storage_inventory"]["source_file_logical_bytes"] = 101
            if fault == "owners": plan["live_contexts"] = 0
            if fault == "pack":
                plan["pins"]["backend"] = "cpu"
                plan["source_backed_admission"]["private_weights"]["cpu_packed_upper_bound_bytes"] = -1
            if fault == "preflight":
                plan["source_backed_admission"]["private_weights"]["cpu_packed_upper_bound_bytes"] = 21
            with self.subTest(fault=fault), self.assertRaises(RuntimeError):
                proof_scheduling_estimate(plan, self.proof())

    def pack_fixture(self):
        graph = {"inputs": {"input": {"dtype": "float32", "shape": [1, "S", 3]}},
                 "nodes": [{"id": f"linear{index}", "opType": "Linear",
                            "inputs": {"input": "input", "weight": "shared"},
                            "outputs": {"out": {"tensor": f"output{index}", "dtype": "float32", "shape": [1, "S", 33]}},
                            "params": {"weight_layout": "dout_din"}} for index in range(2)]}
        return graph, {"shared": {"dtype": "F32", "shape": [33, 3]}}

    def test_private_pack_counts_shared_consumers_and_panel_tail(self):
        graph, weights = self.pack_fixture()
        result = capacity.cpu_pack_upper_bound(graph, weights)
        self.assertEqual(result["cpu_packed_upper_bound_bytes"], 2 * 4 * 3 * 64)
        self.assertEqual(len(result["per_node"]), 2)
        for nr in (8, 16, 32):
            exact_one = 4 * 3 * ((33 + nr - 1) // nr) * nr
            self.assertGreaterEqual(result["per_node"][0]["upper_bound_bytes"], exact_one)

    def test_private_pack_rejects_uncovered_operators_dtype_features_or_layout(self):
        for fault in ("conv", "f16", "symbolic", "layout", "shape"):
            graph, weights = self.pack_fixture()
            if fault == "conv": graph["nodes"][0]["opType"] = "Conv2D"
            if fault == "f16": weights["shared"]["dtype"] = "F16"
            if fault == "symbolic": graph["inputs"]["input"]["shape"][-1] = "D"
            if fault == "layout": graph["nodes"][0]["params"]["weight_layout"] = "din_dout"
            if fault == "shape": weights["shared"]["shape"] = [3, 33]
            with self.subTest(fault=fault), self.assertRaises(RuntimeError):
                capacity.cpu_pack_upper_bound(graph, weights)

    def test_compile_attestation_failure_releases_all_returned_owners(self):
        released = []
        client = SimpleNamespace(
            create_runtime=lambda request: SimpleNamespace(runtime_id=1),
            load_model=lambda request: SimpleNamespace(model_id=2),
            compile_model=lambda request: SimpleNamespace(compiled_model_id=3, report=None),
            release_runtime=lambda reference: released.append(("runtime", reference.runtime_id)),
            release_model=lambda reference: released.append(("model", reference.model_id)),
            release_compiled_model=lambda reference: released.append(("compiled", reference.compiled_model_id)))
        @contextmanager
        def host(*args, **kwargs):
            try: yield None
            finally: released.append(("host", None))
        pb = SimpleNamespace(BuildProfile=SimpleNamespace(BUILD_PROFILE_FULL=1, BUILD_PROFILE_INFERENCE=2),
            BackendPolicyMode=SimpleNamespace(BACKEND_POLICY_MODE_REQUIRE=1),
            OperatorFallback=SimpleNamespace(OPERATOR_FALLBACK_FORBID=1))
        for name in ("Empty", "CreateRuntimeRequest", "RuntimeRef", "LoadModelRequest", "ModelRef",
                     "CompileModelRequest", "BackendPolicy", "CompiledModelRef"):
            setattr(pb, name, SimpleNamespace)
        vx = SimpleNamespace(pb=pb, open_library=host, VxInferenceServiceClient=lambda host: client,
            VxProfilingServiceClient=lambda host: None,
            VxPlatformServiceClient=lambda host: SimpleNamespace(get_platform_info=lambda request: SimpleNamespace(build_profile=1)))
        plan = {"pins": {"source_root": str(Path(__file__).resolve().parents[2]), "backend": "cpu", "profile": "full",
                "inputs": {name: {"path": "unused"} for name in ("library", "bridge", "graph", "weights")}},
                "minimum_free_disk_bytes": 0, "scheduling_estimate": {}, "cpu_threads": 2}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(capacity.importlib, "import_module", return_value=vx), \
                patch.dict(sys.modules, {"volvoxai._diagnostics": SimpleNamespace(_proto_json=lambda value: {})}), \
                patch.object(capacity, "admission", return_value={"admitted": True}), \
                patch.object(capacity.baseline, "route", side_effect=RuntimeError("attestation rejected")), \
                patch.object(capacity.baseline, "proc_status", return_value={"VmHWM": 1}):
            args = SimpleNamespace(mode="timing", admission_proof=None, tmpdir=Path(directory))
            receipt = {}
            with self.assertRaisesRegex(RuntimeError, "attestation rejected"):
                capacity.execute(args, plan, receipt)
        self.assertEqual(released, [("compiled", 3), ("model", 2), ("runtime", 1), ("host", None)])
        self.assertEqual(receipt["process_lifetime_final"], {"VmHWM": 1})

    def test_observer_failure_still_closes_nvml_and_retains_final_status(self):
        actions = []
        sampler = SimpleNamespace(samples=[{"monotonic_ns": 1, "VmHWM": 9}],
                                  error=RuntimeError("poll failed"), stop=lambda: actions.append("stop"))
        def rejected_sample():
            raise RuntimeError("device disappeared")
        nvml = SimpleNamespace(sample=rejected_sample, close=lambda: actions.append("close"))
        receipt = {}
        with patch.object(capacity.baseline, "proc_status", return_value={"VmHWM": 11}):
            with self.assertRaisesRegex(RuntimeError, "poll failed.*device disappeared"):
                capacity.finish_observers(sampler, nvml, receipt)
        self.assertEqual(actions, ["stop", "close"])
        self.assertEqual(receipt["observed_memory"]["VmHWM"], 9)
        self.assertEqual(receipt["process_lifetime_final"], {"VmHWM": 11})
        self.assertEqual(len(receipt["observer_cleanup_errors"]), 2)


class MeasurementRunnerTests(unittest.TestCase):
    IDLE = [{"cpu_busy_percent": 0, "gpu_devices": [{"utilization_percent": 0}]}]

    def test_cli_modes_dispatch_without_silently_mixing_scopes(self):
        common = ["--root", ".", "--bundle", ".", "--out", "unused"]
        qualified = ["--selection", "qualified", "--reference", "reference.json",
                     "--profile", "full", "--backend", "webgpu"]
        for options, target in (([], "run_all"), (qualified, "run_qualified")):
            with patch.object(sys, "argv", ["browser", *common, *options]), \
                    patch.object(browser, target, return_value=7) as run:
                self.assertEqual(browser.main(), 7)
                self.assertEqual(run.call_count, 1)
        for options in (["--profile", "full"], [*qualified, "--stage", "memory"]):
            with patch.object(sys, "argv", ["browser", *common, *options]), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    browser.main()
                self.assertEqual(error.exception.code, 2)
        with patch.object(sys, "argv", ["attention", "--series", "--plan", "plan.json",
                "--runtime-inventory", "runtime.json", "--out", "unused"]), \
                patch.object(attention, "run_series", return_value=7) as run:
            self.assertEqual(attention.main(), 7)
            self.assertEqual(run.call_count, 1)
        with patch.object(sys, "argv", ["attention", "--series", "--out", "unused"]), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                attention.main()
            self.assertEqual(error.exception.code, 2)

    def test_shared_idle_sampler_keeps_three_cpu_and_device_edges(self):
        ticks = ([0] * 8, [10, 0, 0, 80, 10, 0, 0, 0]) * 3
        with patch.object(telemetry, "cpu_ticks", side_effect=ticks), \
                patch.object(telemetry.time, "sleep") as sleep, \
                patch.object(telemetry.subprocess, "check_output", return_value="GPU-1, Device, 4, 32\n") as query:
            rows = telemetry.edge_samples()
        self.assertEqual(len(rows), 3)
        self.assertEqual(sleep.call_args_list, [unittest.mock.call(1)] * 3)
        self.assertEqual(query.call_count, 3)
        self.assertTrue(all(row["cpu_busy_percent"] == 10 for row in rows))
        self.assertTrue(all(row["gpu_devices"][0]["device_used_mib"] == 32 for row in rows))

    def browser_case(self, root, outcomes, edges):
        helper = root / "tools/llm_baseline/measure_native_baseline.py"
        helper.parent.mkdir(parents=True)
        helper.write_text("test-only telemetry identity\n")
        operand = root / "input"
        operand.write_bytes(b"pinned input")
        pins = {"input": browser.identity(operand)}
        reference = root / "reference.json"
        reference.write_text(json.dumps({"pins": pins}))
        bundle = root / "bundle"
        bundle.mkdir()
        args = SimpleNamespace(root=root, out=root / "result", bundle=bundle, reference=reference,
            profile="full", backend="webgpu", existing=[], attempt_cap=4, qualified_workers=2)
        commands = []

        def launch(command, **kwargs):
            commands.append(command)
            outcome = outcomes[len(commands) - 1]
            value = {"pins": pins, "status": "PASS", "samples": [],
                "cleanup": {"process_tree_drained": True},
                "system_swap_gate": {"before": 0, "after": 0},
                "browser_result": {"status": "PASS", "profile": "full", "backend": "webgpu",
                    "mode": "timing", "diagnostic_only": False, "qualification_claim": True,
                    "oracle_loaded": False, "snapshots": [], "device_buffers": [],
                    "cases": [{}] * 24, "workflow_timings": [outcome] * 24}}
            code = 0
            if outcome == "swap":
                value.update(status="FAIL", error="system swap activity invalidates timing measurement")
                value["system_swap_gate"]["after"] = 1
                code = 1
            elif outcome == "numeric":
                value["browser_result"]["status"] = "FAIL"
                code = 1
            elif outcome == "identity":
                operand.write_bytes(b"changed")
            Path(command[command.index("--out") + 1]).write_text(json.dumps(value))
            return SimpleNamespace(wait=lambda **kw: code, poll=lambda: code)

        with patch.object(browser, "edge_samples", side_effect=edges), \
                patch.object(browser.subprocess, "Popen", side_effect=launch), \
                patch.object(browser.signal, "signal"), redirect_stdout(io.StringIO()):
            status = browser.run_qualified(args)
        return status, json.loads((args.out / "completion.json").read_text()), commands

    def test_qualified_browser_retries_only_environment_and_keeps_chronological_workers(self):
        busy = [{"cpu_busy_percent": 6, "gpu_devices": []}]
        with tempfile.TemporaryDirectory(prefix="volvoxai-remaining-validation-test-") as directory:
            status, report, commands = self.browser_case(Path(directory), ["swap", 900, 1],
                                                        [busy] + [self.IDLE] * 5)
            self.assertEqual(status, 0)
            self.assertEqual([row["status"] for row in report["workers"]], ["FAIL", "FAIL", "PASS", "PASS"])
            retained = [json.loads(Path(pin["path"]).read_text()) for pin in report["qualified"]]
            self.assertEqual([value["browser_result"]["workflow_timings"][0] for value in retained], [900, 1])
            self.assertEqual(len(commands), 3)
            for command in commands:
                self.assertEqual(command[command.index("--observe") + 1], "false")
                self.assertEqual(command[command.index("--rounds") + 1], "8")

    def test_qualified_browser_numeric_and_identity_failures_are_terminal(self):
        for fault in ("numeric", "identity"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory(prefix="volvoxai-remaining-validation-test-") as directory:
                status, report, commands = self.browser_case(Path(directory), [fault], [self.IDLE] * 2)
                self.assertEqual(status, 1)
                self.assertEqual(len(commands), 1)
                self.assertEqual(report["qualified"], [])
                self.assertTrue(report["child_drained"])

    def test_attention_series_preserves_paired_order_and_removes_private_scratch(self):
        for valid in (True, False):
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                plan = root / "plan.json"
                plan.write_text(json.dumps({"timing_scope": "completed queue calls"}))
                runtime = root / "runtime"
                runtime.write_bytes(b"pinned runtime")
                inventory = root / "runtime.json"
                inventory.write_text(json.dumps([{**attention.identity(runtime), "path": "runtime"}]))
                args = SimpleNamespace(plan=plan, out=root / "result", runtime_inventory=inventory, deadline_seconds=60)
                order = []

                def launch(command, **kwargs):
                    self.assertEqual(Path(command[1]), Path(attention.__file__))
                    self.assertNotIn("--series", command)
                    order.append((command[command.index("--variant") + 1], command[command.index("--mode") + 1]))
                    scratch = Path(command[command.index("--tmpdir") + 1])
                    (scratch / "private-snapshot").write_bytes(b"owned")
                    Path(command[command.index("--out") + 1]).write_text(json.dumps({
                        "status": "PASS", "measurement_valid": valid, "system_swap_delta": {"pswpin": 0, "pswpout": 0}}))
                    return SimpleNamespace(wait=lambda **kw: 0, poll=lambda: 0)

                with patch.object(attention, "edge_samples", return_value=self.IDLE), \
                        patch.object(attention, "swap", return_value={"pswpin": 0, "pswpout": 0}), \
                        patch.object(attention.subprocess, "Popen", side_effect=launch), \
                        patch.object(attention.signal, "signal"), redirect_stdout(io.StringIO()):
                    status = attention.run_series(args)
                expected = [(v, "timing") for v in ("before", "after", "after", "before", "before", "after")]
                expected += [(v, "memory") for v in ("before", "after")]
                self.assertEqual(order, expected if valid else expected[:1])
                self.assertEqual(status, 0 if valid else 1)
                self.assertEqual(list(args.out.glob("temporary-*")), [])


if __name__ == "__main__":
    unittest.main()
