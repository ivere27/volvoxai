"""Exercise an installed wheel with no checkout or system-library dependency.

The Docker builder copies this file outside the repository and runs it with
isolated Python interpreters. A GPU host can set VOLVOXAI_WHEEL_TEST_BACKEND
to cuda, vulkan or opengl to require that physical execution route.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
import volvoxai as vx


pb = vx.pb
BUNDLED = (Path(vx.__file__).parent / "BUILD-INFO.json").is_file()
BACKEND = os.environ.get("VOLVOXAI_WHEEL_TEST_BACKEND", "cpu")


def cli(*arguments, success=True):
    result = subprocess.run([sys.executable, "-I", "-m", "volvoxai", *map(str, arguments)],
                            text=True, capture_output=True, timeout=90)
    if success and result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    if not success and not result.returncode:
        raise AssertionError("command unexpectedly succeeded")
    return result


@unittest.skipUnless(BUNDLED, "requires a built, installed wheel")
class InstalledWheelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        cls.temporary = tempfile.TemporaryDirectory(prefix="volvoxai-installed-")
        cls.directory = Path(cls.temporary.name)
        cls.weight = np.array([[.5, -.3], [.25, .2], [-.1, .4]], dtype=np.float32)
        cls.bias = np.array([.05, -.1], dtype=np.float32)
        cls.values = np.array([[1, -.7, 2]], dtype=np.float32)
        graph = helper.make_graph([
            helper.make_node("MatMul", ["input", "weight"], ["projected"]),
            helper.make_node("Add", ["projected", "bias"], ["output"]),
        ], "wheel-test", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
            initializer=[numpy_helper.from_array(cls.weight, "weight"),
                         numpy_helper.from_array(cls.bias, "bias")])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = 9
        onnx.checker.check_model(model)
        onnx.save(model, cls.directory / "model.onnx")
        cls.fp32 = cls.directory / "fp32"
        cli("export", "--model", cls.directory / "model.onnx", "--out",
            cls.fp32 / "model.safetensors", "--output-name", "output")
        cls.input_name, = json.loads((cls.fp32 / "graph.json").read_text())["inputs"]
        calibration = cls.directory / "calibration"
        calibration.mkdir()
        for index, values in enumerate(([-1, .2, .3], [0, 1, -.5], [1, -.7, 2])):
            np.savez(calibration / f"{index:03}.npz",
                     **{cls.input_name: np.array([values], dtype=np.float32)})
        cls.ptq = cls.directory / "ptq"
        cls.ptq_arguments = [
            "ptq", "--graph", cls.fp32 / "graph.json", "--weights",
            cls.fp32 / "model.safetensors", "--calibration", calibration,
        ]
        cli(*cls.ptq_arguments, "--out", cls.ptq)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def model(self, host, directory):
        inference = vx.VxInferenceServiceClient(host)
        runtime = inference.create_runtime(pb.CreateRuntimeRequest(cpu_threads=1))
        model = inference.load_model(pb.LoadModelRequest(
            runtime_id=runtime.runtime_id, graph_path=str(directory / "graph.json"),
            weight_paths=[str(directory / "model.safetensors")]))
        return inference, model

    def execute(self, directory):
        with vx.open_library() as host:
            inference, model = self.model(host, directory)
            compiled = inference.compile_model(pb.CompileModelRequest(
                model_id=model.model_id,
                policy=pb.BackendPolicy(
                    mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                    backends=[BACKEND],
                    operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
            self.assertEqual(compiled.report.backend, BACKEND)
            self.assertTrue(compiled.report.route.attested)
            context = inference.create_execution_context(pb.CreateExecutionContextRequest(
                compiled_model_id=compiled.compiled_model_id, graph=pb.GraphExecution()))
            result = inference.execute(pb.ExecuteRequest(context_id=context.context_id,
                scope=pb.ExecutionScope(entrypoint="evaluate", full=pb.Empty()),
                inputs=pb.InputBindings(values=[pb.Tensor(name=self.input_name, shape=[1, 3],
                    dtype=pb.DataType.DATA_TYPE_F32, inline=self.values.tobytes())])))
            self.assertEqual(result.report.backend, BACKEND)
            output = next(tensor for tensor in result.outputs if tensor.name == "output")
            return np.frombuffer(output.inline, dtype=np.float32).reshape(output.shape).copy()

    def test_fp32_and_ptq(self):
        expected = self.values @ self.weight + self.bias
        with self.subTest(backend=BACKEND):
            np.testing.assert_allclose(self.execute(self.fp32), expected,
                                       rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(self.execute(self.ptq), expected,
                                       rtol=0, atol=.035)
        from safetensors.numpy import load_file
        packed = load_file(self.ptq / "model.safetensors")
        self.assertTrue(any(value.dtype == np.int8 for value in packed.values()))

    def test_numpy_session_fp32_and_ptq(self):
        expected = self.values @ self.weight + self.bias
        for directory, tolerance in ((self.fp32, 1e-6), (self.ptq, .035)):
            with self.subTest(model=directory.name, backend=BACKEND):
                with vx.InferenceSession(directory, backend=BACKEND) as session:
                    self.assertEqual(session.backend, BACKEND)
                    self.assertTrue(session.report.route.attested)
                    output = session.run(self.values)["output"]
                    repeated = session.run({self.input_name: self.values})["output"]
                np.testing.assert_allclose(output, expected, rtol=1e-5, atol=tolerance)
                np.testing.assert_array_equal(output, repeated)

    def test_complete_generated_api_and_bundled_libraries(self):
        all_services = {"VxPlatformService", "VxProfilingService", "VxDebugService", "VxTextService", "VxPlanningService",
                        "VxInferenceService", "VxSchedulerService", "VxTrainingService",
                        "VxQuantizationService", "VxBufferService"}
        with vx.open_library() as host:
            self.assertEqual(vx.find_library().parent, Path(vx.__file__).parent)
            client = vx.VxPlatformServiceClient(host)
            info = client.get_platform_info(pb.Empty())
            self.assertEqual(info.library_version, vx.__version__)
            self.assertEqual(set(info.compiled_backends), {"cpu", "cuda", "vulkan", "opengl"})
            description = client.describe_api(pb.DescribeApiRequest())
            self.assertEqual({item.service.rsplit(".", 1)[-1]
                              for item in description.methods}, all_services)
            for method in description.methods:
                service = method.service.rsplit(".", 1)[-1]
                snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2",
                               re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", method.name)).lower()
                for suffix in ("Client", "AsyncClient"):
                    self.assertTrue(callable(getattr(getattr(vx, service + suffix), snake)))
                self.assertTrue(hasattr(pb, method.request_type.rsplit(".", 1)[-1]))
                self.assertTrue(hasattr(pb, method.response_type.rsplit(".", 1)[-1]))

    def test_text_planning_and_scheduling(self):
        with vx.open_library() as host:
            text = vx.VxTextServiceClient(host)
            tokenizer = text.create_tokenizer(pb.CreateTokenizerRequest(
                vocabulary_json=b'{"a":0,"b":1}'))
            encoded = text.encode_text(pb.EncodeTextRequest(
                tokenizer_id=tokenizer.tokenizer_id, text="abba"))
            self.assertEqual(list(encoded.tokens), [0, 1, 1, 0])
            decoded = text.decode_tokens(pb.DecodeTokensRequest(
                tokenizer_id=tokenizer.tokenizer_id, tokens=list(encoded.tokens)))
            self.assertEqual(decoded.text, "abba")
            inference, model = self.model(host, self.fp32)
            planning = vx.VxPlanningServiceClient(host)
            planning.create_graph_plan(pb.CreateGraphPlanRequest(model_id=model.model_id))
            scheduler = vx.VxSchedulerServiceClient(host)
            queue = scheduler.create_batch_queue(pb.CreateBatchQueueRequest(max_slots=2))
            work = scheduler.submit_batch_work(pb.SubmitBatchWorkRequest(
                queue_id=queue.queue_id, payload=b"wheel", stateless=pb.StatelessBatchWork(rows=1)))
            reference = pb.BatchWorkRef(queue_id=queue.queue_id, work_id=work.work_id)
            queued = scheduler.get_batch_work(reference)
            self.assertEqual(queued.state, pb.BatchWorkState.BATCH_WORK_STATE_QUEUED)
            scheduler.cancel_batch_work(reference)
            scheduler.release_batch_queue(pb.BatchQueueRef(queue_id=queue.queue_id))

    def test_async_clients(self):
        async def exercise():
            async with vx.AsyncModuleHost(vx.open_library()) as host:
                client = vx.VxPlatformServiceAsyncClient(host)
                info = await client.get_platform_info(pb.Empty())
                self.assertEqual(info.library_version, vx.__version__)
                runtime = await vx.VxInferenceServiceAsyncClient(host).create_runtime(
                    pb.CreateRuntimeRequest(cpu_threads=1))
        asyncio.run(exercise())

    def test_ptq_refuses_bad_calibration_and_preserves_existing_output(self):
        before = (self.ptq / "model.safetensors").read_bytes()
        cli(*self.ptq_arguments, "--out", self.ptq, success=False)
        self.assertEqual((self.ptq / "model.safetensors").read_bytes(), before)
        bad = self.directory / "bad-calibration"
        bad.mkdir()
        np.savez(bad / "000.npz", **{self.input_name: self.values.astype(np.float64)})
        destination = self.directory / "invalid"
        cli("ptq", "--graph", self.fp32 / "graph.json", "--weights",
            self.fp32 / "model.safetensors", "--calibration", bad, "--out", destination,
            success=False)
        self.assertFalse((destination / "graph.json").exists())

    def test_cli_entry_and_packaged_native_runner(self):
        self.assertEqual(cli("--version").stdout.strip(), vx.__version__)
        executable = Path(sys.executable).parent / "volvoxai"
        result = subprocess.run([executable, "--version"], text=True, capture_output=True, check=True)
        self.assertEqual(result.stdout.strip(), vx.__version__)
        from volvoxai.exporter.native_execution import _native_runner_path
        runner = _native_runner_path()
        self.assertEqual(runner.parent, Path(vx.__file__).parent / "bin")
        subprocess.run([runner, "--version"], check=True, capture_output=True)
        input_path = self.directory / "input.f32"
        output_path = self.directory / "output.f32"
        self.values.tofile(input_path)
        subprocess.run([runner, "run", self.fp32, "--cpu", "--threads", "1",
                        "--input", f"{self.input_name}={input_path}",
                        "--output", f"output={output_path}"], check=True, capture_output=True)
        np.testing.assert_allclose(np.fromfile(output_path, dtype=np.float32).reshape(1, 2),
                                   self.values @ self.weight + self.bias, rtol=1e-5, atol=1e-6)

    def test_direct_hf_uses_packaged_builder_and_original_bf16(self):
        from volvoxai.exporter.hf_decoder import DecoderConfig
        config = {"model_type": "llama", "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 4, "vocab_size": 8, "max_position_embeddings": 4,
            "rms_norm_eps": 1e-5, "hidden_act": "silu", "tie_word_embeddings": True}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "config.json").write_text(json.dumps(config))
            header, payload = {}, bytearray()
            for name, shape in DecoderConfig.from_hf(config).weight_shapes().items():
                begin = len(payload)
                payload.extend(bytes(int(np.prod(shape)) * 2))
                header[name] = {"dtype": "BF16", "shape": list(shape),
                                "data_offsets": [begin, len(payload)]}
            encoded = json.dumps(header).encode()
            encoded += b" " * (-len(encoded) % 8)
            weight = directory / "model.safetensors"
            weight.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)
            with vx.InferenceSession.from_pretrained(directory, max_context=4, backend=BACKEND) as session:
                self.assertEqual(session.weight_paths, (weight,))
                self.assertFalse((directory / "graph.json").exists())
                weight.unlink()
                result = session.run({"token_ids": np.array([[1, 2]], np.int32),
                    "position_ids": np.array([0, 1], np.int32),
                    "attention_mask": np.ones((1, 2), np.int32)})
                np.testing.assert_array_equal(result["logits"], np.zeros((1, 2, 8), np.float32))
            self.assertIn("volvoxai.exporter.hf_decoder", sys.modules)
            self.assertNotIn("tools.exporter.hf_decoder", sys.modules)


if __name__ == "__main__":
    if not BUNDLED:
        raise SystemExit("Install the wheel before running its qualification tests")
    unittest.main(verbosity=2)
