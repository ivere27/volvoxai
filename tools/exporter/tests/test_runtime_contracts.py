from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.exporter.errors import ExporterError
from tools.exporter.ir import GraphIR as TypedGraphIR
from tools.exporter.ir import IRDialect, OpNode, TensorValue
from tools.exporter.optimizer.safetensors_io import (
    read_safetensors,
    write_safetensors,
)
from tools.exporter.runtime_ir import import_runtime_package, load_runtime_document
from tools.exporter.runtime_names import (
    OBJECT_PROTOTYPE_OWN_NAMES,
    is_runtime_graph_name,
)
from tools.exporter.runtime_tensors import runtime_tensor_allocation


def _identity_document() -> dict:
    return {
        "format": "volvox-graph/v1",
        "dimensions": {},
        "inputs": {"x": {"shape": [1, 4], "dtype": "float32"}},
        "outputs": ["y"],
        "nodes": [{
            "id": "identity",
            "opType": "Identity",
            "inputs": {"input": "x"},
            "outputs": {
                "out": {"tensor": "y", "shape": [1, 4], "dtype": "float32"},
            },
            "params": {},
        }],
    }


def _typed_identity(name: str, dialect: IRDialect) -> TypedGraphIR:
    graph = TypedGraphIR("test", "names", dialect=dialect)
    graph.add_tensor(TensorValue(
        name, (1,), "float32", "float32", public_input=True,
    ))
    graph.add_tensor(TensorValue(
        "y", (1,), "float32", "float32", public_output=True,
    ))
    graph.inputs.append(name)
    graph.add_node(OpNode.from_maps(
        "identity", "Identity", {"input": name}, {"out": "y"},
    ))
    graph.outputs.append("y")
    return graph


def _import_persisted(document: dict, tensors: dict) -> TypedGraphIR:
    """Exercise the same strict JSON-load and typed-import boundary as tools."""

    with tempfile.TemporaryDirectory(
        prefix="volvox-runtime-contract-",
    ) as directory:
        path = Path(directory) / "graph.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        loaded = load_runtime_document(path)
        return import_runtime_package(loaded, tensors, source_name=str(path))


class RuntimeNameContractTests(unittest.TestCase):
    def test_matches_ecmascript_graph_valid_name_exactly(self):
        rejected = {
            "",
            "\t\n\u00a0\u2007\u2028\u2029\ufeff",
            "__metadata__",
            *OBJECT_PROTOTYPE_OWN_NAMES,
        }
        for name in rejected:
            with self.subTest(name=name):
                self.assertFalse(is_runtime_graph_name(name))

        # U+0085 is stripped by Python str.strip(), but not ECMAScript
        # String.trim(); surrounding ordinary whitespace also remains legal.
        for name in ("tensor", " tensor ", "\u0085"):
            with self.subTest(name=name):
                self.assertTrue(is_runtime_graph_name(name))

    def test_source_ir_preserves_names_that_runtime_ir_rejects(self):
        source = _typed_identity("constructor", IRDialect.SOURCE)
        source.verify(IRDialect.SOURCE)

        runtime = _typed_identity("constructor", IRDialect.RUNTIME)
        with self.assertRaises(ExporterError) as caught:
            runtime.verify(IRDialect.RUNTIME)
        self.assertEqual(caught.exception.diagnostic.code, "VXIR043")

    def test_typed_import_rejects_reserved_names_in_every_graph_namespace(self):
        mutations = []
        for reserved in (*sorted(OBJECT_PROTOTYPE_OWN_NAMES), "__metadata__", "\ufeff"):
            def input_name(document, name=reserved):
                descriptor = document["inputs"].pop("x")
                document["inputs"][name] = descriptor
                document["nodes"][0]["inputs"]["input"] = name

            mutations.append((f"tensor:{reserved}", input_name))

        mutations.extend((
            (
                "input-port",
                lambda document: document["nodes"][0].update(
                    inputs={"__proto__": "x"}
                ),
            ),
            (
                "output-port",
                lambda document: document["nodes"][0].update(
                    outputs={
                        "constructor": {
                            "tensor": "y",
                            "shape": [1, 4],
                            "dtype": "float32",
                        },
                    },
                ),
            ),
            (
                "opType",
                lambda document: document["nodes"][0].update(
                    opType="hasOwnProperty"
                ),
            ),
        ))

        for label, mutate in mutations:
            with self.subTest(label=label):
                document = _identity_document()
                mutate(document)
                with self.assertRaises(ExporterError) as caught:
                    import_runtime_package(document, {})
                self.assertEqual(caught.exception.diagnostic.code, "VXRTIR026")

    def test_typed_import_rejects_a_reserved_safetensors_name(self):
        with self.assertRaises(ExporterError) as caught:
            import_runtime_package(
                _identity_document(),
                {"__proto__": np.asarray([1.0], dtype=np.float32)},
            )
        self.assertEqual(caught.exception.diagnostic.code, "VXRTIR026")

    def test_strict_persisted_boundary_uses_the_same_name_rule(self):
        document = _identity_document()
        document["nodes"][0]["inputs"] = {"constructor": "x"}
        with self.assertRaises(ExporterError) as caught:
            _import_persisted(document, {})
        self.assertEqual(caught.exception.diagnostic.code, "VXRTIR026")
        self.assertIn("invalid or reserved", caught.exception.diagnostic.message)

        with self.assertRaises(ExporterError) as caught:
            _import_persisted(_identity_document(), {
                "__proto__": np.asarray([1], dtype=np.int8),
            })
        self.assertEqual(caught.exception.diagnostic.code, "VXRTIR026")
        self.assertIn("safetensors tensor name", caught.exception.diagnostic.message)

    def test_safetensors_io_uses_the_exact_runtime_name_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            write_safetensors(
                path, {"\u0085": np.asarray([1], dtype=np.int8)},
            )
            self.assertEqual(set(read_safetensors(path)), {"\u0085"})
            for reserved in ("constructor", "__proto__", "__metadata__"):
                with self.subTest(reserved=reserved):
                    with self.assertRaisesRegex(ValueError, "Graph.validName"):
                        write_safetensors(
                            path,
                            {reserved: np.asarray([1], dtype=np.int8)},
                        )


class RuntimeTensorAllocationTests(unittest.TestCase):
    def test_checks_dimension_product_and_final_byte_size(self):
        self.assertEqual(runtime_tensor_allocation([], "float32"), (1, 4))
        self.assertEqual(runtime_tensor_allocation([3, 5], "int8"), (15, 15))
        self.assertEqual(runtime_tensor_allocation([2], "float16"), (2, 8))
        with self.assertRaisesRegex(ValueError, "element count"):
            runtime_tensor_allocation([1 << 27, 1 << 27], "int8")
        with self.assertRaisesRegex(ValueError, "byte size"):
            runtime_tensor_allocation([1 << 51], "float32")

    def test_typed_import_rejects_unsafe_input_and_output_allocations(self):
        for location in ("input", "output"):
            with self.subTest(location=location):
                document = _identity_document()
                if location == "input":
                    document["inputs"]["x"]["shape"] = [1 << 27, 1 << 27]
                else:
                    document["nodes"][0]["outputs"]["out"]["shape"] = [
                        1 << 27, 1 << 27,
                    ]
                with self.assertRaises(ExporterError) as caught:
                    import_runtime_package(document, {})
                self.assertEqual(caught.exception.diagnostic.code, "VXRTIR003")

    def test_typed_import_rejects_unsafe_safetensors_runtime_allocations(self):
        cases = {
            "element-product": np.lib.stride_tricks.as_strided(
                np.zeros((1,), dtype=np.int8),
                shape=(1 << 27, 1 << 27),
                strides=(0, 0),
            ),
            # The physical F16 span would be 2**52 bytes and JSON-safe. The
            # loader expands it to F32, whose 2**53-byte Tensor is not safe.
            "expanded-f16-bytes": np.lib.stride_tricks.as_strided(
                np.zeros((1,), dtype=np.float16),
                shape=(1 << 51,),
                strides=(0,),
            ),
        }
        for label, huge_view in cases.items():
            with self.subTest(label=label):
                document = {
                    "format": "volvox-graph/v1",
                    "dimensions": {},
                    "inputs": {},
                    "outputs": ["huge"],
                    "nodes": [],
                }
                with self.assertRaises(ExporterError) as caught:
                    import_runtime_package(document, {"huge": huge_view})
                self.assertEqual(caught.exception.diagnostic.code, "VXRTIR027")

    def test_f16_safetensors_initializer_storage_remains_legal(self):
        document = {
            "format": "volvox-graph/v1",
            "dimensions": {},
            "inputs": {
                "x": {"shape": [1, 1, 1, 1], "dtype": "float32"},
            },
            "outputs": ["y"],
            "nodes": [{
                "id": "conv",
                "opType": "Conv2D",
                "inputs": {"input": "x", "weight": "half_weight"},
                "outputs": {
                    "out": {
                        "tensor": "y",
                        "shape": [1, 1, 1, 1],
                        "dtype": "float32",
                    },
                },
                "params": {"weight_layout": "HWIO"},
            }],
        }
        weights = {
            "half_weight": np.zeros((1, 1, 1, 1), dtype=np.float16),
        }
        graph = import_runtime_package(document, weights)
        self.assertEqual(graph.tensors["half_weight"].dtype, "float16")
        self.assertEqual(graph.tensors["half_weight"].source_dtype, "float16")

    def test_strict_persisted_boundary_rejects_unsafe_input_and_output_shapes(self):
        for location in ("input", "output"):
            with self.subTest(location=location):
                document = _identity_document()
                if location == "input":
                    document["inputs"]["x"]["shape"] = [1 << 27, 1 << 27]
                else:
                    document["nodes"][0]["outputs"]["out"]["shape"] = [
                        1 << 27, 1 << 27,
                    ]
                with self.assertRaises(ExporterError) as caught:
                    _import_persisted(document, {})
                self.assertEqual(caught.exception.diagnostic.code, "VXRTIR003")
                self.assertIn("invalid bounded shape", caught.exception.diagnostic.message)

if __name__ == "__main__":
    unittest.main()
