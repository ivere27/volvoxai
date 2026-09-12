"""C and Python must author the same template from the same graph.

`native/src/training/ptq_authoring.c` reimplements the rewrite that
`tools/exporter/typed_ptq.py` has been doing, so that PTQ stops being
Python-only: once the C is the implementation, every language that can reach
`proto/volvoxai.proto` can quantize a model, and the exporter becomes one more
caller rather than the only one.

Two implementations of one rewrite are worth having only while they agree.
This compares them on the fixture the golden test pins — same graph, same
weights, same configuration — and diffs the authored templates node by node.

It compares structure, not payloads. Authoring names the packed weights and
affines; it does not compute them, because their values depend on calibration
that has not run. Payload parity is the golden test's job, and it is a
separate question from this one.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import ptq_fixture as fixture  # noqa: E402
from test_ptq_golden import SOURCE_NAME, author  # noqa: E402

CONFIGURATIONS = {
    "symmetric-i8": ("i8", "symmetric"),
    "asymmetric-u8": ("u8", "asymmetric"),
}


def authoring_binary() -> Path | None:
    for candidate in (
        REPOSITORY_ROOT / "native" / "build" / "native" / "test_ptq_authoring",
        REPOSITORY_ROOT / "build" / "cmake" / "native" / "test_ptq_authoring",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def write_fixture(directory: Path) -> tuple[Path, Path]:
    """The same graph and weights the Python side authors from."""

    from safetensors.numpy import save_file

    directory.mkdir(parents=True, exist_ok=True)
    graph_path = directory / "graph.json"
    weights_path = directory / "model.safetensors"
    graph_path.write_text(json.dumps(fixture.document(), indent=1))
    save_file(
        {name: np.ascontiguousarray(value)
         for name, value in fixture.tensors().items()},
        str(weights_path),
    )
    return graph_path, weights_path


def author_in_c(binary: Path, directory: Path, config_name: str) -> dict:
    graph_path, weights_path = write_fixture(directory)
    template_path = directory / "template.json"
    activation, scheme = CONFIGURATIONS[config_name]
    result = subprocess.run(
        [
            str(binary), str(graph_path), str(weights_path), str(template_path),
            "--activation", activation, "--scheme", scheme,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"C authoring failed for {config_name}: {result.stderr.strip()}")
    return {
        "template": json.loads(template_path.read_text()),
        "plan": result.stdout.splitlines(),
    }


def node_index(document: dict) -> dict[str, dict]:
    return {node["id"]: node for node in document["nodes"]}


@unittest.skipUnless(authoring_binary(), "test_ptq_authoring is not built")
class AuthoringParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binary = authoring_binary()
        cls.directory = Path(tempfile.mkdtemp(prefix="volvoxai-ptq-parity-"))
        cls.c_authored = {
            name: author_in_c(cls.binary, cls.directory / name, name)
            for name in CONFIGURATIONS
        }
        cls.python_authored = {
            name: author(name)[0] for name in CONFIGURATIONS
        }

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.directory, ignore_errors=True)

    def test_same_nodes_in_the_same_order(self) -> None:
        """Node identity and order are the template's structure. A reader
        matches the template against the graph it came from by walking both,
        so a reordering is a different template even with identical content."""

        for name in CONFIGURATIONS:
            with self.subTest(config=name):
                c_ids = [n["id"] for n in self.c_authored[name]["template"]["nodes"]]
                py_ids = [n["id"] for n in self.python_authored[name]["nodes"]]
                self.assertEqual(c_ids, py_ids)

    def test_same_operator_for_every_node(self) -> None:
        for name in CONFIGURATIONS:
            with self.subTest(config=name):
                c_nodes = node_index(self.c_authored[name]["template"])
                py_nodes = node_index(self.python_authored[name])
                for node_id, expected in py_nodes.items():
                    self.assertEqual(
                        c_nodes[node_id]["opType"], expected["opType"],
                        f"{node_id} authored as a different operator")

    def test_same_operands_for_every_node(self) -> None:
        """The tensor names are the contract between authoring and everything
        downstream. A packed weight the writer cannot find by name is a
        template that only one implementation can consume."""

        for name in CONFIGURATIONS:
            with self.subTest(config=name):
                c_nodes = node_index(self.c_authored[name]["template"])
                py_nodes = node_index(self.python_authored[name])
                for node_id, expected in py_nodes.items():
                    self.assertEqual(
                        c_nodes[node_id]["inputs"], expected["inputs"],
                        f"{node_id} reads different tensors")

    def test_same_parameters_for_every_node(self) -> None:
        """A quantized node's parameters are canonical: every one present,
        defaults materialized, floats narrowed to the F32 they are stored as.
        Two implementations that disagree about a default disagree about what
        the kernel computes, and nothing else in this file would notice —
        the graph shape and every tensor name would still match."""

        for name in CONFIGURATIONS:
            with self.subTest(config=name):
                c_nodes = node_index(self.c_authored[name]["template"])
                py_nodes = node_index(self.python_authored[name])
                for node_id, expected in py_nodes.items():
                    self.assertEqual(
                        c_nodes[node_id].get("params", {}),
                        expected.get("params", {}),
                        f"{node_id} authored with different parameters")

    def test_same_result_tensor_and_storage(self) -> None:
        for name in CONFIGURATIONS:
            with self.subTest(config=name):
                c_nodes = node_index(self.c_authored[name]["template"])
                py_nodes = node_index(self.python_authored[name])
                for node_id, expected in py_nodes.items():
                    produced = expected["outputs"]["out"]
                    actual = c_nodes[node_id]["outputs"]["out"]
                    self.assertEqual(actual["tensor"], produced["tensor"],
                                     f"{node_id} produces a different tensor")
                    self.assertEqual(actual["dtype"], produced["dtype"],
                                     f"{node_id} produces a different storage")

    def test_same_affine_table(self) -> None:
        """Which tensors have an affine, whether it is per-tensor or
        per-channel, and which tensors hold it. Getting the axis wrong is the
        mistake that produces plausible output and wrong numbers."""

        for name in CONFIGURATIONS:
            with self.subTest(config=name):
                c_table = self.c_authored[name]["template"]["quantization"]
                py_table = self.python_authored[name]["quantization"]
                self.assertEqual(c_table["format"], py_table["format"])
                self.assertEqual(
                    sorted(c_table["tensors"]), sorted(py_table["tensors"]))
                for tensor, expected in py_table["tensors"].items():
                    self.assertEqual(c_table["tensors"][tensor], expected,
                                     f"{tensor} has a different affine")

    def test_graph_boundary_is_unchanged(self) -> None:
        """Quantization is an interior change. If the declared inputs or
        outputs moved, the package is not a drop-in replacement for the float
        one and every caller has to be edited."""

        for name in CONFIGURATIONS:
            with self.subTest(config=name):
                c_template = self.c_authored[name]["template"]
                py_template = self.python_authored[name]
                self.assertEqual(c_template["inputs"], py_template["inputs"])
                self.assertEqual(c_template["outputs"], py_template["outputs"])
                self.assertEqual(c_template["format"], py_template["format"])

    def test_plan_names_the_same_layers(self) -> None:
        """The plan is what CreatePtqPlan takes next, so a layer C omits is a
        weight nothing packs."""

        for name in CONFIGURATIONS:
            with self.subTest(config=name):
                lines = self.c_authored[name]["plan"]
                layers = [line for line in lines if line.startswith("layer ")]
                observers = [line for line in lines
                             if line.startswith("observer ")]
                py_template = self.python_authored[name]
                quantized = sum(
                    1 for node in py_template["nodes"]
                    if node["opType"] == "QLinear")
                affines = py_template["quantization"]["tensors"]
                per_tensor = sum(
                    1 for entry in affines.values()
                    if entry["scheme"] == "per_tensor")
                self.assertEqual(len(layers), quantized)
                self.assertEqual(len(observers), per_tensor)

    def test_c_reports_what_it_did(self) -> None:
        for name in CONFIGURATIONS:
            with self.subTest(config=name):
                lines = self.c_authored[name]["plan"]
                counted = dict(
                    line.split(" ", 1) for line in lines
                    if line.startswith(("quantized_nodes", "retained_float")))
                self.assertEqual(int(counted["quantized_nodes"]), 6)
                self.assertEqual(int(counted["retained_float_nodes"]), 0)


if __name__ == "__main__":
    unittest.main()


def author_python(document: dict, tensors: dict, config_name: str) -> dict:
    """Today's Python authoring over an arbitrary document."""

    from tools.exporter.optimizer.target import TargetEnvironment
    from tools.exporter.optimizer.typed_pipeline import author_runtime_ptq_package
    from tools.exporter.runtime_ir import import_runtime_package
    from tools.exporter.typed_ptq import (
        calibration_profile_from_ranges,
        required_ptq_observations,
    )
    from test_ptq_golden import configuration

    config = configuration(config_name)
    graph = import_runtime_package(
        document, dict(tensors), source_name=SOURCE_NAME)
    demanded = sorted(required_ptq_observations(graph, None, config=config))
    profile = calibration_profile_from_ranges(
        graph, fixture.ranges(demanded),
        sample_count=fixture.SAMPLE_COUNT,
        sample_digest=fixture.SAMPLE_DIGEST,
        config=config)
    authored, _, _ = author_runtime_ptq_package(
        document, tensors, profile, shape_profile={},
        source_name=SOURCE_NAME, config=config,
        target_environment=TargetEnvironment(
            backend_profile="portable", compile_backend="wasm",
            tune_backend="native-cpu"))
    return authored


def author_in_c_document(binary: Path, directory: Path, document: dict,
                         tensors: dict) -> dict:
    """C authoring over an arbitrary document, through the file entry point."""

    from safetensors.numpy import save_file

    directory.mkdir(parents=True, exist_ok=True)
    graph_path = directory / "graph.json"
    weights_path = directory / "model.safetensors"
    template_path = directory / "template.json"
    graph_path.write_text(json.dumps(document, indent=1))
    save_file({name: np.ascontiguousarray(value)
               for name, value in tensors.items()}, str(weights_path))
    result = subprocess.run(
        [str(binary), str(graph_path), str(weights_path), str(template_path)],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"C authoring failed: {result.stderr.strip()}")
    return json.loads(template_path.read_text())


class DocumentParityMixin:
    """Diffs one document's C and Python templates node by node."""

    def assert_templates_agree(self, c_template: dict,
                               python_template: dict) -> None:
        c_nodes = node_index(c_template)
        py_nodes = node_index(python_template)
        self.assertEqual([n["id"] for n in c_template["nodes"]],
                         [n["id"] for n in python_template["nodes"]])
        for node_id, expected in py_nodes.items():
            actual = c_nodes[node_id]
            self.assertEqual(actual["opType"], expected["opType"], node_id)
            self.assertEqual(actual["inputs"], expected["inputs"], node_id)
            self.assertEqual(actual.get("params", {}),
                             expected.get("params", {}), node_id)
            self.assertEqual(actual["outputs"]["out"],
                             expected["outputs"]["out"], node_id)
        self.assertEqual(c_template["quantization"],
                         python_template["quantization"])
        self.assertEqual(c_template["inputs"], python_template["inputs"])
        self.assertEqual(c_template["outputs"], python_template["outputs"])


@unittest.skipUnless(authoring_binary(), "test_ptq_authoring is not built")
class ConvParityTest(unittest.TestCase, DocumentParityMixin):
    """Conv2D packs a weight, like Linear, but carries geometry.

    An implicit stride or padding would make the template describe one
    convolution and the kernel perform another — an output of the wrong size
    rather than one that is wrong by a little — so every geometry parameter is
    stated, and both implementations must state the same ones.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = Path(tempfile.mkdtemp(prefix="volvoxai-ptq-conv-"))
        document = fixture.conv_document()
        tensors = fixture.conv_tensors()
        cls.c_template = author_in_c_document(
            authoring_binary(), cls.directory, document, tensors)
        cls.python_template = author_python(document, tensors, "symmetric-i8")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.directory, ignore_errors=True)

    def test_templates_agree(self) -> None:
        self.assert_templates_agree(self.c_template, self.python_template)

    def test_geometry_is_stated_not_defaulted(self) -> None:
        conv = node_index(self.c_template)["node_0"]
        self.assertEqual(conv["opType"], "QConv2D")
        for name in ("stride", "dilation", "pads", "groups", "data_layout",
                     "weight_layout", "relu"):
            self.assertIn(name, conv["params"])
        # pads expands from the two-element padding the source declared.
        self.assertEqual(conv["params"]["pads"], [1, 1, 1, 1])
        self.assertEqual(conv["params"]["weight_layout"], "OHWI")

    def test_the_packed_weight_is_per_channel(self) -> None:
        affines = self.c_template["quantization"]["tensors"]
        per_axis = [entry for entry in affines.values()
                    if entry["scheme"] == "per_axis"]
        self.assertEqual(len(per_axis), 1)
        self.assertEqual(per_axis[0]["axis"], 0)


@unittest.skipUnless(authoring_binary(), "test_ptq_authoring is not built")
class DinDoutParityTest(unittest.TestCase, DocumentParityMixin):
    """The other dense weight layout.

    The golden fixture is dout_din, so nothing there would notice an
    implementation that read the declaration and then used the wrong extent.
    This one is din_dout with unequal widths, where that mistake produces a
    bias-length disagreement rather than a plausible template.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = Path(tempfile.mkdtemp(prefix="volvoxai-ptq-layout-"))
        document = fixture.din_dout_document()
        tensors = fixture.din_dout_tensors()
        cls.c_template = author_in_c_document(
            authoring_binary(), cls.directory, document, tensors)
        cls.python_template = author_python(document, tensors, "symmetric-i8")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.directory, ignore_errors=True)

    def test_templates_agree(self) -> None:
        self.assert_templates_agree(self.c_template, self.python_template)

    def test_the_packed_weight_is_per_channel_on_axis_zero(self) -> None:
        """Whichever way the source stored it, the packed form is
        [dout, din]."""

        affines = self.c_template["quantization"]["tensors"]
        per_axis = [entry for entry in affines.values()
                    if entry["scheme"] == "per_axis"]
        self.assertEqual(len(per_axis), 1)
        self.assertEqual(per_axis[0]["axis"], 0)


@unittest.skipUnless(authoring_binary(), "test_ptq_authoring is not built")
class CrossSdpaParityTest(unittest.TestCase, DocumentParityMixin):
    """Attention where nothing is packed.

    Q, K and V are three independently calibrated activations, so QSDPA is a
    byte node. The keep mask is copied through untouched: it selects rather
    than scales, and quantizing it would change which keys participate.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = Path(tempfile.mkdtemp(prefix="volvoxai-ptq-sdpa-"))
        cls.variants = {}
        for name, options in (
            ("plain", {}),
            ("masked", {"mask": True}),
            ("scaled", {"scale": True}),
        ):
            document = fixture.cross_sdpa_document(**options)
            tensors = fixture.cross_sdpa_tensors()
            cls.variants[name] = (
                author_in_c_document(authoring_binary(),
                                     cls.directory / name, document, tensors),
                author_python(document, tensors, "symmetric-i8"),
            )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.directory, ignore_errors=True)

    def test_templates_agree(self) -> None:
        for name, (c_template, python_template) in self.variants.items():
            with self.subTest(variant=name):
                self.assert_templates_agree(c_template, python_template)

    def test_scale_appears_only_when_the_source_declared_one(self) -> None:
        """An absent scale means 1/sqrt(head_dim). Materializing that default
        would turn a derived value into a declared one, and freeze it."""

        plain = node_index(self.variants["plain"][0])["node_0"]
        scaled = node_index(self.variants["scaled"][0])["node_0"]
        self.assertNotIn("scale", plain["params"])
        self.assertIn("scale", scaled["params"])

    def test_the_keep_mask_is_not_quantized(self) -> None:
        masked = node_index(self.variants["masked"][0])["node_0"]
        self.assertEqual(masked["inputs"]["mask"], "keep")
        affines = self.variants["masked"][0]["quantization"]["tensors"]
        self.assertNotIn("keep", affines)

    def test_causal_stays_a_boolean(self) -> None:
        """A number here would read as a count. The kernel branches on it."""

        for name, (c_template, _) in self.variants.items():
            with self.subTest(variant=name):
                causal = node_index(c_template)["node_0"]["params"]["causal"]
                self.assertIsInstance(causal, bool)


@unittest.skipUnless(authoring_binary(), "test_ptq_authoring is not built")
class BatchMatMulParityTest(unittest.TestCase):
    """An operator the golden fixture does not reach.

    BatchMatMul is the case where nothing is packed: both operands are
    dynamic, so both are calibrated and the product is computed in the byte
    domain. An implementation that treated it as a dense node would try to
    pack an activation against a calibrated range, which is silently wrong
    rather than loudly broken — so it is checked on its own graph.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from safetensors.numpy import save_file

        cls.binary = authoring_binary()
        cls.directory = Path(tempfile.mkdtemp(prefix="volvoxai-ptq-bmm-"))
        cls.directory.mkdir(parents=True, exist_ok=True)
        document = fixture.batch_matmul_document()
        tensors = fixture.batch_matmul_tensors()
        graph_path = cls.directory / "graph.json"
        weights_path = cls.directory / "model.safetensors"
        graph_path.write_text(json.dumps(document, indent=1))
        save_file({name: np.ascontiguousarray(value)
                   for name, value in tensors.items()}, str(weights_path))
        template_path = cls.directory / "template.json"
        result = subprocess.run(
            [str(cls.binary), str(graph_path), str(weights_path),
             str(template_path)],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise AssertionError(f"C authoring failed: {result.stderr.strip()}")
        cls.c_template = json.loads(template_path.read_text())
        cls.python_template = author_python(document, tensors, "symmetric-i8")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.directory, ignore_errors=True)

    def test_same_nodes_operands_and_parameters(self) -> None:
        c_nodes = node_index(self.c_template)
        py_nodes = node_index(self.python_template)
        self.assertEqual([n["id"] for n in self.c_template["nodes"]],
                         [n["id"] for n in self.python_template["nodes"]])
        for node_id, expected in py_nodes.items():
            actual = c_nodes[node_id]
            self.assertEqual(actual["opType"], expected["opType"], node_id)
            self.assertEqual(actual["inputs"], expected["inputs"], node_id)
            self.assertEqual(actual.get("params", {}),
                             expected.get("params", {}), node_id)
            self.assertEqual(actual["outputs"]["out"],
                             expected["outputs"]["out"], node_id)

    def test_same_affine_table(self) -> None:
        self.assertEqual(self.c_template["quantization"],
                         self.python_template["quantization"])

    def test_both_operands_are_calibrated(self) -> None:
        """Nothing is packed here, so every affine is per-tensor. A per-axis
        entry would mean one operand had been mistaken for a weight."""

        schemes = {entry["scheme"]
                   for entry in self.c_template["quantization"]["tensors"].values()}
        self.assertEqual(schemes, {"per_tensor"})
