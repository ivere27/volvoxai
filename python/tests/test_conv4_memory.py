"""Public CPU memory admission must reserve packed Conv2/3/4's bounded scratch.

The workspace size comes from observed allocations, not kernel tile constants.
The same admitted context executes the minimum, maximum and smaller B/H/W
shapes, with an independent integer oracle and unchanged scratch capacity.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
from safetensors.numpy import load, save

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
import volvoxai as vx

pb = vx.pb


def fixture(bits=4):
    graph = {
        "format": "volvox-graph/v1",
        "dimensions": {"B": {"min": 1, "max": 2},
                       "H": {"min": 1, "max": 5},
                       "W": {"min": 1, "max": 7}},
        "inputs": {"image": {"shape": ["B", "H", "W", 3], "dtype": "int8"}},
        "nodes": [{
            "id": "conv4", "opType": "QConv2D",
            "inputs": {"input": "image", "weight": "weight"},
            "outputs": {"out": {"tensor": "features",
                "shape": ["B", "H", "W", 2], "dtype": "int8"}},
            "params": {"weight_bits": bits, "kernel_size": [1, 1],
                "data_layout": "NHWC", "weight_layout": "OHWI",
                "groups": 1, "stride": [1, 1], "dilation": [1, 1],
                "pads": [0, 0, 0, 0]},
        }],
        "outputs": ["features"],
        "quantization": {"format": "volvox-affine-safetensors/v1", "tensors": {
            "image": {"scheme": "per_tensor", "scale_tensor": "xs",
                      "zero_point_tensor": "xz"},
            "weight": {"scheme": "per_axis", "axis": 0,
                       "scale_tensor": "ws", "zero_point_tensor": "wz"},
            "features": {"scheme": "per_tensor", "scale_tensor": "ys",
                         "zero_point_tensor": "yz"},
        }},
        "execution": {"stateless": {}, "entrypoints": [{
            "name": "evaluate", "effect": 1, "allowed_ranges": [1],
            "inputs": ["image"], "outputs": ["features"],
        }]},
    }
    # Logical output rows [1, 0, -1] and [0, 1, 1], with canonical zero
    # padding in the unused high nibble. No bias or F32 fallback is needed.
    arrays = {
        "weight": np.array([[0x89, 0x87], [0x98, 0x89]], np.uint8),
        "xs": np.array([1], np.float32), "xz": np.array([0], np.int8),
        "ws": np.array([1, 1], np.float32), "wz": np.array([8, 8], np.uint8),
        "ys": np.array([1], np.float32), "yz": np.array([0], np.int8),
    }
    codes = np.array([[1, 0, -1], [0, 1, 1]], np.int8)
    offset = 1 << (bits - 1)
    packed = np.zeros((2, (3 * bits + 7) // 8), np.uint8)
    for column in range(3):
        for bit in range(bits):
            position = column * bits + bit
            packed[:, position // 8] |= (((codes[:, column].astype(np.int32) + offset) >> bit) & 1).astype(np.uint8) << (position % 8)
    if bits != 3:
        for column in range(3, packed.shape[1] * 8 // bits):
            packed[:, column * bits // 8] |= offset << (column * bits % 8)
    arrays["weight"] = packed
    arrays["wz"] = np.array([offset, offset], np.uint8)
    return pb.ModelPackage(graph_document=json.dumps(graph).encode(),
                           weight_shards=[save(arrays)])


class Conv4MemoryTest(unittest.TestCase):
    def test_workspace_budget_and_bounded_dynamic_shapes(self):
        for bits in (2, 3, 4):
            with self.subTest(weight_bits=bits):
                self.check_workspace(bits)

    def test_lowbit_noncanonical_tail_is_rejected_before_execution(self):
        with vx.open_library() as host:
            client = vx.VxInferenceServiceClient(host)
            runtime = client.create_runtime(pb.CreateRuntimeRequest(cpu_threads=1))
            try:
                for bits in (2,3):
                    with self.subTest(bits=bits):
                        package = fixture(bits)
                        arrays = load(package.weight_shards[0])
                        arrays["weight"][0,-1] ^= 1 << (3 * bits % 8)
                        package.weight_shards[0] = save(arrays)
                        # Load retains immutable package bytes. Physical storage
                        # validation belongs to compilation, before any context
                        # or Execute call can consume the malformed tail.
                        model = client.load_model(pb.LoadModelRequest(
                            runtime_id=runtime.runtime_id, package=package))
                        try:
                            with self.assertRaises(vx.VolvoxAIError) as refused:
                                client.compile_model(pb.CompileModelRequest(model_id=model.model_id,
                                    policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                                        backends=["cpu"], operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
                            self.assertEqual(refused.exception.response.compiled_model_id, 0)
                        finally:
                            client.release_model(pb.ModelRef(model_id=model.model_id))
            finally:
                client.release_runtime(pb.RuntimeRef(runtime_id=runtime.runtime_id))

    def check_workspace(self, bits):
        host = vx.open_library()
        self.addCleanup(host.close)
        client = vx.VxInferenceServiceClient(host)
        profiling = vx.VxProfilingServiceClient(host)
        runtime = client.create_runtime(pb.CreateRuntimeRequest(cpu_threads=1))
        self.addCleanup(client.release_runtime, pb.RuntimeRef(runtime_id=runtime.runtime_id))
        model = client.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                                                     package=fixture(bits)))
        self.addCleanup(client.release_model, pb.ModelRef(model_id=model.model_id))
        compiled = client.compile_model(pb.CompileModelRequest(model_id=model.model_id,
            policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                backends=["cpu"], operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
        self.addCleanup(client.release_compiled_model,
            pb.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
        self.assertEqual(compiled.report.backend, "cpu")

        # Compile publishes the complete shape-domain bound. The explicit
        # application budget belongs to CreateExecutionContext, not Compile.
        bounds = {item.budget_domain_id: item for item in compiled.memory_bounds.bounds}
        budget = 64 << 20
        self.assertLess(bounds["provider-resident"].maximum_bytes.bytes, budget)
        context = client.create_execution_context(pb.CreateExecutionContextRequest(
            compiled_model_id=compiled.compiled_model_id, graph=pb.GraphExecution(),
            limits=pb.ExecutionLimits(max_memory_bytes=budget)))
        self.addCleanup(client.release_execution_context,
            pb.ExecutionContextRef(context_id=context.context_id))

        def scratch_capacities():
            memory = profiling.get_resource_snapshot(
                pb.GetResourceSnapshotRequest(context_id=context.context_id)).snapshot.memory
            self.assertFalse(memory.truncated)
            # Context scope also includes reachable shared model resources;
            # count only scratch actually owned by this execution context.
            capacities = sorted(item.capacity_bytes for item in memory.allocations
                if item.owner.kind == pb.MemoryOwnerKind.MEMORY_OWNER_KIND_EXECUTION_CONTEXT
                and item.owner.owner_id == str(context.context_id)
                and item.role == pb.MemoryResourceRole.MEMORY_RESOURCE_ROLE_SCRATCH)
            self.assertTrue(capacities, "Conv4 must reserve context-owned scratch before execution")
            self.assertTrue(all(value > 0 for value in capacities))
            return capacities

        reserved = scratch_capacities()
        scratch_bytes = sum(reserved)
        # This tiny tensor domain isolates workspace from activation capacity.
        self.assertLess(bounds["maximum-tensor"].maximum_bytes.bytes, scratch_bytes)
        self.assertGreaterEqual(bounds["provider-resident"].maximum_bytes.bytes, scratch_bytes)
        with self.assertRaises(vx.VolvoxAIError) as refused:
            client.create_execution_context(pb.CreateExecutionContextRequest(
                compiled_model_id=compiled.compiled_model_id, graph=pb.GraphExecution(),
                limits=pb.ExecutionLimits(max_memory_bytes=scratch_bytes - 1)))
        self.assertEqual(refused.exception.status, pb.NativeStatus.NATIVE_STATUS_OUT_OF_MEMORY)
        self.assertEqual(refused.exception.response.context_id, 0)

        for shape in ((1, 1, 1, 3), (2, 5, 7, 3), (1, 2, 3, 3), (2, 5, 7, 3)):
            with self.subTest(shape=shape):
                values = (np.arange(np.prod(shape)) % 7 - 3).astype(np.int8).reshape(shape)
                packet = client.execute(pb.ExecuteRequest(context_id=context.context_id,
                    scope=pb.ExecutionScope(entrypoint="evaluate", full=pb.Empty()),
                    inputs=pb.InputBindings(values=[pb.Tensor(name="image",
                        dtype=pb.DataType.DATA_TYPE_I8, shape=list(shape), inline=values.tobytes())])))
                self.assertEqual(packet.report.backend, "cpu")
                self.assertEqual(len(packet.outputs), 1)
                output = packet.outputs[0]
                self.assertEqual(output.dtype, pb.DataType.DATA_TYPE_I8)
                self.assertEqual(output.shape, list(shape[:-1]) + [2])
                actual = np.frombuffer(output.inline, np.int8).reshape(output.shape)
                wide = values.astype(np.int32)
                expected = np.stack((wide[..., 0] - wide[..., 2],
                                     wide[..., 1] + wide[..., 2]), axis=-1).astype(np.int8)
                np.testing.assert_array_equal(actual, expected)
                # Allocation IDs are snapshot-local; capacities are the public
                # evidence that changing geometry did not grow live scratch.
                self.assertEqual(scratch_capacities(), reserved)


if __name__ == "__main__":
    unittest.main()
