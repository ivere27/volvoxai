"""Check production CUDA block arithmetic/barriers on CPU; no GPU qualification."""
from pathlib import Path
import array
import ctypes as C
import importlib.util
import math
import shutil
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def attention_probe():
    path = Path(__file__).resolve().parents[2] / "tools/llm_baseline/probe_cuda_attention.py"
    spec = importlib.util.spec_from_file_location("attention_accuracy_probe", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class ProbeDriver:
    """Execute the probe's argument/lifetime checks with a fixed CPU oracle."""
    def __init__(self, probe, oracle, changed_launch=None):
        self.probe, self.oracle, self.changed_launch = probe, oracle, changed_launch
        self.memory, self.calls, self.launches = {}, [], 0

    def call(self, name, *args):
        self.calls.append(name)
        if name in ("DeviceGet", "DriverGetVersion"):
            C.cast(args[0], C.POINTER(C.c_int))[0] = 1
        elif name == "DeviceGetName":
            args[0].value = b"CPU argument fixture"
        elif name in ("CtxCreate", "StreamCreate", "ModuleLoadData", "ModuleGetFunction", "EventCreate"):
            C.cast(args[0], C.POINTER(C.c_void_p))[0] = len(self.calls)
        elif name == "MemAlloc":
            address = 0x100000 + len(self.calls) * 0x10000
            self.memory[address] = bytearray(args[1])
            C.cast(args[0], C.POINTER(C.c_uint64))[0] = address
        elif name == "MemFree":
            del self.memory[args[0].value]
        elif name == "MemcpyHtoD":
            self.memory[args[0].value][:] = C.string_at(args[1], args[2])
        elif name == "MemcpyDtoH":
            C.memmove(args[0], bytes(self.memory[args[1].value]), args[2])
        elif name == "LaunchKernel":
            self.launches += 1
            params = args[9]
            record = C.c_uint64.from_address(params[len(params) - 1]).value
            output = (struct.unpack_from("<Q", self.memory[record], 4 * 8)[0] if record
                      else C.c_uint64.from_address(params[4]).value)
            values = array.array("f", self.oracle)
            if self.launches == self.changed_launch:
                bits = struct.unpack("<I", values[:1].tobytes())[0]
                values[0] = struct.unpack("<f", struct.pack("<I", bits + 1))[0]
            guarded = array.array("f", [self.probe.SENTINEL] * self.probe.GUARD)
            guarded.extend(values)
            guarded.extend([self.probe.SENTINEL] * self.probe.GUARD)
            self.memory[output - self.probe.GUARD * 4][:] = guarded.tobytes()
        elif name == "EventElapsedTime":
            C.cast(args[0], C.POINTER(C.c_float))[0] = .01


class CudaCrossAttentionAccuracyTests(unittest.TestCase):
    def test_f64_masks_mha_gqa_self_row_tails_and_cooperative_barriers(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler unavailable")
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="volvox-cuda-cross-accuracy-") as directory:
            binary = Path(directory) / "accuracy"
            built = subprocess.run([compiler, "-std=c++17", "-O3", "-ffp-contract=off",
                "-Wall", "-Wextra", "-Werror",
                str(root / "native/tests/test_cuda_cross_attention_accuracy.cpp"),
                "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertEqual(checked.stdout.count("PASS"), 58 * 2 + 12, checked.stdout)
            self.assertEqual(checked.stdout.count("BITS "), 58 * 2, checked.stdout)
            self.assertEqual(checked.stdout.count("NOLOAD "), 12, checked.stdout)

    def test_probe_poison_cases_preserve_existing_seeds_and_masked_oracle(self):
        probe = attention_probe()
        cases = probe.geometries()
        self.assertEqual(len(cases), 85)
        for index, case in enumerate(cases[:43]):
            self.assertEqual(case["seed"], 710913 + index * 7919)
            self.assertNotIn("poison_masked_values", case)
        for index, original in ((43, 0), (44, 4)):
            expected = dict(cases[original], reverse=True)
            expected["label"] += "-reversed"
            self.assertEqual(cases[index], expected)
        self.assertTrue(cases[47]["cancellation"])
        self.assertEqual((cases[47]["width"], cases[48]["width"]), (64, 64))
        self.assertEqual(cases[47]["seed"], 710913 + 47 * 7919)
        q, k, v, mask, _ = probe.fixture(cases[47])
        oracle = probe.reference(cases[47], q, k, v, mask)
        self.assertTrue(all(math.isfinite(x) for x in oracle))
        for index in (45, 46, 48):
            case = cases[index]
            self.assertTrue(case["poison_masked_values"])
            self.assertEqual(case["seed"], 710913 + index * 7919)
            q, k, v, mask, _ = probe.fixture(case)
            columns = case["kv_heads"] * case["width"]
            for batch in range(case["batch"]):
                for key in range(case["keys"]):
                    expected_poison = not probe.allowed(case, mask, batch, 0, key)
                    values = v[(batch * case["keys"] + key) * columns:
                               (batch * case["keys"] + key + 1) * columns]
                    self.assertTrue(all(math.isnan(x) == expected_poison for x in values))
            oracle = probe.reference(case, q, k, v, mask)
            self.assertTrue(all(math.isfinite(x) for x in oracle))
            if case["all_masked"]:
                self.assertEqual(array.array("d", oracle).tobytes(), bytes(8 * len(oracle)))

    def test_probe_cache_boundaries_use_production_launch_and_masked_oracle(self):
        probe = attention_probe()
        cases = probe.geometries()[49:78]
        boundaries = cases[:24]
        self.assertEqual({g["keys"] for g in boundaries},
                         {47, 48, 49, 511, 512, 1536, 2048, 2049})
        self.assertEqual({g["width"] for g in boundaries}, {32, 64, 128})
        for case in boundaries:
            expected_tiles = (case["width"] + 31) // 32 if case["keys"] <= 48 else 1
            self.assertEqual(probe.launch(case, "parallel")["grid"][2], expected_tiles)
        for case in cases[-2:]:
            q, k, v, mask, _ = probe.fixture(case)
            oracle = probe.reference(case, q, k, v, mask)
            self.assertTrue(all(math.isfinite(value) for value in oracle))
            if case["all_masked"]:
                self.assertEqual(array.array("d", oracle).tobytes(), bytes(8 * len(oracle)))

    def test_probe_long_cancellation_and_multiquery_cases(self):
        probe = attention_probe()
        cases = probe.geometries()[78:]
        self.assertEqual([case["keys"] for case in cases[:5]], [49, 512, 1536, 2048, 2049])
        self.assertTrue(all(case["cancellation"] for case in cases[:5]))
        self.assertEqual([(case["queries"], case["keys"], case["heads"]) for case in cases[5:]],
                         [(8, 512, 32), (8, 1536, 32)])
        case = cases[0]
        q, k, v, mask, _ = probe.fixture(case)
        self.assertTrue(all(math.isfinite(value) for value in probe.reference(case, q, k, v, mask)))

    def test_probe_argument_and_repeat_hash_refusals_precede_further_timing(self):
        probe = attention_probe()
        geometry = probe.geometry("CPU-probe-controls", "gqa", 1, 1, 3, 2, 1, 2, 2)
        geometry["seed"] = 710913
        q, k, v, mask, _ = probe.fixture(geometry)
        oracle = probe.reference(geometry, q, k, v, mask)
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="volvox-cuda-probe-controls-") as directory:
            ptx, library = Path(directory) / "test.ptx", Path(directory) / "test.so"
            ptx.write_text("CPU fixture only\n"); library.write_bytes(b"CPU fixture only\n")
            args = SimpleNamespace(source_root=root, ptx=ptx, library=library, mode="parallel")
            for changed_launch, expected_error, event_count in (
                    (None, None, 8), (2, "direct/stored output changed", 0),
                    (4, "warm/measured output changed", 4)):
                with self.subTest(changed_launch=changed_launch):
                    driver = ProbeDriver(probe, oracle, changed_launch)
                    with patch.object(probe, "geometries", return_value=[geometry]), \
                            patch.object(probe, "Driver", return_value=driver):
                        receipt = probe.prepare(args)
                        if expected_error:
                            with self.assertRaisesRegex(AssertionError, expected_error):
                                probe.run(args, receipt)
                        else:
                            probe.run(args, receipt)
                            self.assertEqual(receipt["status"], "PASS")
                            config = receipt["configs"][0]
                            self.assertEqual([item["argument_storage"] for item in
                                config["argument_numerical_observations"]], ["direct", "stored"])
                            self.assertEqual(len(config["observations"]), 4)
                    self.assertEqual(driver.calls.count("EventRecord"), event_count)
                    self.assertEqual(driver.memory, {})
                    self.assertEqual(driver.calls[-1], "CtxDestroy")


if __name__ == "__main__":
    unittest.main()
