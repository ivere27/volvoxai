"""Trace summaries and vendor-tool annotations over the native full library.

NVTX is compiled into native builds with a GPU backend; without an injected
tool its calls return immediately, but the trace still counts what it emitted.
Vulkan/OpenGL labels are checked when that device is usable on this machine.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))

import numpy as np  # noqa: E402
from safetensors.numpy import save_file  # noqa: E402

import volvoxai as vx  # noqa: E402


def library_available() -> bool:
    try:
        vx.find_library()
    except vx.VolvoxAIError:
        return False
    return True


def write_package(directory: Path) -> Path:
    graph = {
        "format": "volvox-graph/v1", "dimensions": {}, "inputs": {"x": {"dtype": "float32", "shape": [2, 3]}},
        "nodes": [
            {"id": "fc", "opType": "Linear", "inputs": {"input": "x", "weight": "fc.w", "bias": "fc.b"},
             "outputs": {"out": {"tensor": "h", "dtype": "float32", "shape": [2, 4]}},
             "params": {"weight_layout": "dout_din"}},
            {"id": "act", "opType": "GELU", "inputs": {"input": "h"},
             "outputs": {"out": {"tensor": "y", "dtype": "float32", "shape": [2, 4]}}, "params": {}},
        ],
        "outputs": ["y"],
    }
    package = directory / "package"
    package.mkdir()
    (package / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    save_file({"fc.w": np.sin(np.arange(12, dtype=np.float32)).reshape(4, 3),
               "fc.b": np.array([.1, -.2, .3, -.4], dtype=np.float32)}, str(package / "model.safetensors"))
    return package


def statuses(info) -> dict[str, tuple[int, int]]:
    return {row.mechanism: (row.status, row.ranges) for row in info.external}


@unittest.skipUnless(library_available(), "libvolvoxai.so is not built")
class ProfilingExternalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = Path(tempfile.mkdtemp(prefix="volvoxai-profiling-external-"))
        cls.package = write_package(cls.directory)
        cls.x = np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.float32)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.directory, ignore_errors=True)

    def test_cpu_summary_and_nvtx_ranges(self) -> None:
        with vx.InferenceSession(self.package) as session:
            with session.trace(detail="nodes", execution_plans=True, external_annotations=True) as trace:
                for _ in range(3):
                    session.run(self.x)
            available = vx.pb.ObservationStatus.OBSERVATION_STATUS_AVAILABLE
            nvtx = statuses(trace.info)["nvtx"]
            self.assertEqual(nvtx[0], available)
            self.assertGreaterEqual(nvtx[1], 3 * 3)  # 3 calls: one call range and two node ranges each
            summary = trace.summary("node")
            linear = next(row for row in summary.rows if row.name == "Linear")
            self.assertEqual(linear.count, 3)
            self.assertEqual(linear.cost_per_call.multiply_accumulates, 24)
            self.assertEqual(list(linear.source_node_ids), ["fc"])
            table = trace.table("operator")
            self.assertIn("Linear", table)
            self.assertIn("GELU", table)

    def device_labels(self, backend: str, mechanism: str) -> None:
        try:
            session = vx.InferenceSession(self.package, backend=backend)
        except vx.VolvoxAIError as error:
            self.skipTest(f"{backend} is not usable here: {error}")
        with session:
            with session.trace(detail="nodes", external_annotations=True) as trace:
                session.run(self.x)
            status, ranges = statuses(trace.info)[mechanism]
            if status == vx.pb.ObservationStatus.OBSERVATION_STATUS_UNSUPPORTED:
                self.skipTest(f"{mechanism} is not offered by this driver")
            self.assertEqual(status, vx.pb.ObservationStatus.OBSERVATION_STATUS_AVAILABLE)
            self.assertEqual(ranges, 2)  # One label per executed node.

    def test_vulkan_debug_utils_labels(self) -> None:
        self.device_labels("vulkan", "vk_debug_utils")

    def test_opengl_khr_debug_groups(self) -> None:
        self.device_labels("opengl", "khr_debug")


if __name__ == "__main__":
    unittest.main()
