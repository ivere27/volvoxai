from __future__ import annotations

import unittest

from tools.exporter.ir import (
    AffineQuantization,
    GraphIR,
    IRDialect,
    TensorValue,
)
from tools.exporter.typed_broadcast import (
    build_descriptor_preserving_byte_expand,
    concrete_broadcast_shape,
)


class TypedBroadcastTests(unittest.TestCase):
    def test_proves_only_concrete_right_aligned_rank_eight_broadcasts(self):
        self.assertEqual(
            concrete_broadcast_shape((2, 1, 3), (1, 4, 1)),
            (2, 4, 3),
        )
        self.assertEqual(
            concrete_broadcast_shape((64,), (1, 402, 64)),
            (1, 402, 64),
        )
        for left, right in (
            ((2, 3), (4, 3)),
            ((0, 3), (1, 3)),
            ((None, 3), (1, 3)),
            ((), (1,)),
            ((1,) * 9, (1,) * 9),
        ):
            with self.subTest(left=left, right=right):
                self.assertIsNone(concrete_broadcast_shape(left, right))

    def test_plans_deterministic_descriptor_preserving_byte_expand(self):
        graph = GraphIR(
            source_format="test",
            source_name="broadcast",
            dialect=IRDialect.RUNTIME,
        )
        affine = AffineQuantization(
            scheme="per_tensor", scale="scale", zero_point="zero",
        )
        graph.add_tensor(TensorValue(
            name="x", shape=(1, 1, 64), dtype="uint8",
            source_dtype="uint8", quantization=affine, public_input=True,
        ))
        graph.inputs.append("x")

        plan = build_descriptor_preserving_byte_expand(
            graph, "x", (1, 402, 64), name_stem="q.b",
        )
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.node.op_type, "Expand")
        self.assertEqual(plan.node.input_map(), {"input": "x"})
        self.assertEqual(plan.tensor.shape, (1, 402, 64))
        self.assertEqual(plan.tensor.dtype, "uint8")
        self.assertIs(plan.tensor.quantization, affine)
        self.assertFalse(plan.tensor.initializer)

        occupied = {plan.tensor.name, plan.node.name}
        second = build_descriptor_preserving_byte_expand(
            graph, "x", (1, 402, 64), name_stem="q.b", occupied=occupied,
        )
        self.assertIsNotNone(second)
        assert second is not None
        self.assertNotEqual(second.tensor.name, plan.tensor.name)
        self.assertNotEqual(second.node.name, plan.node.name)

        self.assertIsNone(build_descriptor_preserving_byte_expand(
            graph, "x", (1, 1, 64), name_stem="identity",
        ))
        self.assertIsNone(build_descriptor_preserving_byte_expand(
            graph, "x", (1, 402, 63), name_stem="bad",
        ))


if __name__ == "__main__":
    unittest.main()
