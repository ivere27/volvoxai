"""A byte-level reference for what PTQ authoring produces today.

The authoring stage — the part that decides which nodes become QLinear, where
QuantizeLinear and DequantizeLinear boundaries go, and what the packed weights
and affines contain — is 3,579 lines of numerics in
``tools/exporter/typed_ptq.py``.  It is being reimplemented in C so that every
language reaching ``proto/volvoxai.proto`` can run it, not only Python.

Numerics of that size cannot be moved without a reference.  A test that only
checks "the output has QLinear nodes" passes for an implementation that gets
every scale wrong.  This one pins the exact bytes: the authored graph, every
packed tensor, and the affine parameters that decide accuracy.

So the digests below are not decoration.  They are the contract the C port has
to meet, and a diff in either of them means the two implementations disagree
about arithmetic.

Changing a digest is a deliberate act.  When authoring genuinely changes,
regenerate with::

    python3 python/tests/test_ptq_golden.py --print

and say in the commit message what moved and why.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import ptq_fixture as fixture  # noqa: E402

# `source_name` is part of the graph fingerprint, and the fingerprint is what
# binds a calibration profile to the graph it was measured on. Both sides must
# spell it the same or the profile is refused as belonging to another revision.
SOURCE_NAME = "graph.json"

# --- the contract ----------------------------------------------------------
#
# Two configurations, because they take different paths through the numerics.
# The default is symmetric int8, where every zero point is zero and a port that
# never computes one at all still passes. Asymmetric uint8 is the configuration
# that makes activation zero points load-bearing, and it is measurably faster
# in WebAssembly, so it is the one an accuracy regression would show up in
# first.
GOLDEN = {
    "symmetric-i8": {
        "graph": "21e57aff555f4471650b2267dc0252ab9a030f62b5174982104cfd50cd38a0d5",
        "tensors": "b23fac627f611f6ad6be9008d210c702eed7c3942777f1b259111d1770dc55ba",
        "operators": {
            "DequantizeLinear": 1, "QAdd": 1, "QGELU": 1, "QLayerNorm": 1,
            "QLinear": 3, "QuantizeLinear": 1,
        },
        "affines": {
            "__ptq__.0046188248b7fd4f8a8b.dequantize": (0.023665079846978188, 0),
            "__ptq__.4c074ba6b69e65fed84e.quantize": (0.00860297679901123, 0),
        },
    },
    "asymmetric-u8": {
        "graph": "dc6696c55bceba33275a3564ed73018c9308d9ad5c1c68a74772d49e8e41bffe",
        "tensors": "cae6792524a91a3f0010b05edb1ec141fb7a7cfcc15a425b4ac3e677b6768d0f",
        "operators": {
            "DequantizeLinear": 1, "QAdd": 1, "QGELU": 1, "QLayerNorm": 1,
            "QLinear": 3, "QuantizeLinear": 1,
        },
        # Non-zero zero points, which is the whole reason this configuration is
        # here: an implementation that only ever emits symmetric affines
        # reproduces the entry above and fails this one.
        "affines": {
            "__ptq__.0046188248b7fd4f8a8b.dequantize": (0.017050297930836678, 176),
            "__ptq__.4c074ba6b69e65fed84e.quantize": (0.008425560779869556, 130),
        },
    },
}


def configuration(name: str):
    """The PTQConfig one golden entry was measured with."""

    from tools.exporter.typed_ptq import PTQConfig

    if name == "symmetric-i8":
        return PTQConfig()
    if name == "asymmetric-u8":
        return PTQConfig(activation_dtype="uint8", activation_scheme="asymmetric")
    raise KeyError(name)


def author(config_name: str = "symmetric-i8") -> tuple[dict, dict[str, np.ndarray]]:
    """Run today's Python authoring over the fixture."""

    from tools.exporter.optimizer.target import TargetEnvironment
    from tools.exporter.optimizer.typed_pipeline import author_runtime_ptq_package
    from tools.exporter.runtime_ir import import_runtime_package
    from tools.exporter.typed_ptq import (
        calibration_profile_from_ranges,
        required_ptq_observations,
    )

    document = fixture.document()
    tensors = fixture.tensors()
    config = configuration(config_name)

    graph = import_runtime_package(
        document, dict(tensors), source_name=SOURCE_NAME)
    demanded = sorted(required_ptq_observations(graph, None, config=config))
    profile = calibration_profile_from_ranges(
        graph,
        fixture.ranges(demanded),
        sample_count=fixture.SAMPLE_COUNT,
        sample_digest=fixture.SAMPLE_DIGEST,
        config=config,
    )

    authored_document, authored_tensors, _ = author_runtime_ptq_package(
        document,
        tensors,
        profile,
        shape_profile={},
        source_name=SOURCE_NAME,
        config=config,
        target_environment=TargetEnvironment(
            backend_profile="portable",
            compile_backend="wasm",
            tune_backend="native-cpu",
        ),
    )
    return authored_document, authored_tensors


def graph_digest(document: dict) -> str:
    """Structure only — key order and whitespace are not the contract."""

    blob = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def tensor_digest(tensors: dict[str, np.ndarray]) -> str:
    """Name, dtype, shape and payload of every tensor, in name order.

    Payload alone would not catch a weight that kept its values but changed
    dtype or layout, and those are exactly the mistakes a port makes.
    """

    digest = hashlib.sha256()
    for name in sorted(tensors):
        value = np.ascontiguousarray(tensors[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(value.shape).encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def operator_counts(document: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in document["nodes"]:
        counts[node["opType"]] = counts.get(node["opType"], 0) + 1
    return dict(sorted(counts.items()))


def affines(document: dict, tensors: dict[str, np.ndarray]) -> dict:
    """Every quantization boundary's scale and zero point.

    These decide accuracy, and a port can reproduce the graph structure
    exactly while getting them wrong — so they are checked by value as well as
    inside the tensor digest, where a mismatch would say only "some byte
    differs".
    """

    found: dict[str, tuple[float, int]] = {}
    for node in document["nodes"]:
        if node["opType"] not in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        inputs = node["inputs"]
        scale_name = inputs.get("scale")
        zero_name = inputs.get("zero_point")
        if scale_name is None or scale_name not in tensors:
            continue
        scale = np.asarray(tensors[scale_name])
        if scale.size != 1:
            continue  # per-channel; the tensor digest covers it
        zero = 0
        if zero_name is not None and zero_name in tensors:
            zero_value = np.asarray(tensors[zero_name])
            if zero_value.size == 1:
                zero = int(zero_value.reshape(-1)[0])
        found[node["id"]] = (float(scale.reshape(-1)[0]), zero)
    return dict(sorted(found.items()))


class PtqGoldenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.authored = {name: author(name) for name in GOLDEN}

    def test_graph_structure_is_unchanged(self) -> None:
        for name, expected in GOLDEN.items():
            with self.subTest(config=name):
                document, _ = self.authored[name]
                self.assertEqual(graph_digest(document), expected["graph"])

    def test_packed_payloads_are_unchanged(self) -> None:
        for name, expected in GOLDEN.items():
            with self.subTest(config=name):
                _, tensors = self.authored[name]
                self.assertEqual(tensor_digest(tensors), expected["tensors"])

    def test_operator_mix_is_unchanged(self) -> None:
        """A readable first failure. When the digests move, this says whether
        the shape of the answer changed or only its numbers."""

        for name, expected in GOLDEN.items():
            with self.subTest(config=name):
                document, _ = self.authored[name]
                self.assertEqual(operator_counts(document), expected["operators"])

    def test_affine_parameters_are_unchanged(self) -> None:
        for name, expected in GOLDEN.items():
            with self.subTest(config=name):
                document, tensors = self.authored[name]
                self.assertEqual(affines(document, tensors), expected["affines"])

    def test_authoring_is_deterministic(self) -> None:
        """Two runs must agree, or the golden pins a coin flip and the C port
        has nothing to match."""

        for name in GOLDEN:
            with self.subTest(config=name):
                document, tensors = self.authored[name]
                again_document, again_tensors = author(name)
                self.assertEqual(
                    graph_digest(again_document), graph_digest(document))
                self.assertEqual(
                    tensor_digest(again_tensors), tensor_digest(tensors))

    def test_quantized_the_dense_nodes(self) -> None:
        """The fixture exists to be quantized. If authoring silently declined
        every node, the digests above would still be stable — and meaningless."""

        for name in GOLDEN:
            with self.subTest(config=name):
                counts = operator_counts(self.authored[name][0])
                self.assertGreater(counts.get("QLinear", 0), 0)
                self.assertNotIn("Linear", counts)


def _print_golden() -> None:
    print("GOLDEN = {")
    for name in ("symmetric-i8", "asymmetric-u8"):
        document, tensors = author(name)
        print(f'    "{name}": {{')
        print(f'        "graph": "{graph_digest(document)}",')
        print(f'        "tensors": "{tensor_digest(tensors)}",')
        print(f'        "operators": {operator_counts(document)!r},')
        print('        "affines": {')
        for node_id, (scale, zero) in affines(document, tensors).items():
            print(f'            "{node_id}": ({scale!r}, {zero}),')
        print("        },")
        print("    },")
    print("}")


if __name__ == "__main__":
    if "--print" in sys.argv:
        _print_golden()
    else:
        unittest.main()
