"""PTQ, start to finish, through the generated Python schema projection.

This test constructs protobuf messages and reads protobuf responses without
reaching into the private native engine lifecycle.

The sequence is the service's three steps:

    AuthorPtqTemplate   FP32 graph  -> quantized template + the plan for it
    CreatePtqPlan       template    -> a plan bound to a Model revision
    Calibrate / Write   plan        -> observed ranges -> a written package

It needs the full-profile library: an inference build registers no
quantization handlers or public quantization symbols.
"""

from __future__ import annotations

import json
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import ptq_fixture as fixture  # noqa: E402
import volvoxai  # noqa: E402


def full_library_available() -> bool:
    try:
        volvoxai.find_library("full")
    except volvoxai.VolvoxAIError:
        return False
    return True


def write_fixture(directory: Path) -> tuple[Path, Path]:
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


@unittest.skipUnless(full_library_available(), "libvolvoxai-full.so is not built")
class PtqThroughTheSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = Path(tempfile.mkdtemp(prefix="volvoxai-ptq-e2e-"))
        cls.host = volvoxai.open_library("full")
        cls.quantization = volvoxai.VxQuantizationServiceClient(cls.host)
        cls.inference = volvoxai.VxInferenceServiceClient(cls.host)
        cls.graph_path, cls.weights_path = write_fixture(cls.directory / "fp32")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.host.close()
        shutil.rmtree(cls.directory, ignore_errors=True)

    def author(self, name: str, **config) -> tuple[Path, object]:
        template_path = self.directory / name / "template.json"
        template_path.parent.mkdir(parents=True, exist_ok=True)
        request = volvoxai.pb.AuthorPtqTemplateRequest(
            source_graph_path=str(self.graph_path),
            weight_paths=[str(self.weights_path)],
            template_graph_path=str(template_path),
        )
        if config:
            request.config = volvoxai.pb.PtqAuthoringConfig(**config)
        info = self.quantization.author_ptq_template(request)
        volvoxai.check(info.report, "AuthorPtqTemplate")
        return template_path, info

    def test_authoring_answers_over_the_module_abi(self) -> None:
        """The step that used to be Python-only, called as an RPC."""

        template_path, info = self.author("plain")
        self.assertTrue(template_path.is_file())
        self.assertEqual(info.quantized_nodes, 6)
        self.assertEqual(info.retained_float_nodes, 0)
        self.assertEqual(len(info.layers), 3)
        self.assertEqual(
            sorted(info.required_observations),
            ["activated", "contracted", "expanded", "hidden", "normed",
             "output", "residual"])

    def test_authored_template_is_a_quantized_graph(self) -> None:
        template_path, _ = self.author("shape")
        template = json.loads(template_path.read_text())
        operators = {node["opType"] for node in template["nodes"]}
        self.assertIn("QLinear", operators)
        self.assertIn("QuantizeLinear", operators)
        self.assertIn("DequantizeLinear", operators)
        self.assertNotIn("Linear", operators)
        # The boundary is unchanged, which is what makes the package a
        # drop-in replacement for the float one.
        self.assertEqual(template["inputs"], fixture.document()["inputs"])
        self.assertEqual(template["outputs"], fixture.document()["outputs"])

    def test_configuration_changes_the_answer(self) -> None:
        """A configuration a caller sends has to reach the authoring. If it
        were dropped the two templates would be identical and every asymmetric
        request would silently get symmetric output."""

        symmetric, _ = self.author("sym")
        asymmetric, info = self.author(
            "asym",
            activation_dtype=volvoxai.pb.DataType.DATA_TYPE_U8,
            activation_scheme=volvoxai.pb.PtqScheme.PTQ_SCHEME_ASYMMETRIC,
        )
        self.assertNotEqual(symmetric.read_text(), asymmetric.read_text())
        storages = {
            node["outputs"]["out"]["dtype"]
            for node in json.loads(asymmetric.read_text())["nodes"]
            if node["opType"] != "DequantizeLinear"
        }
        self.assertEqual(storages, {"uint8"})
        self.assertTrue(all(
            observer.dtype == volvoxai.pb.DataType.DATA_TYPE_U8
            for observer in info.observers))

    def test_retaining_an_operator_in_float(self) -> None:
        """Naming an operator is a decision, not a failure. The nodes around
        it stay quantized and the boundaries move to match."""

        _, info = self.author("retain", float_operators=["GELU"])
        self.assertEqual(info.quantized_nodes, 5)
        self.assertEqual(info.retained_float_nodes, 1)

    def test_refuses_an_operator_it_cannot_author(self) -> None:
        """Refusing beats quietly retaining: a graph authored differently here
        than by the exporter shows up as an accuracy gap much later."""

        source = self.directory / "conv" / "graph.json"
        source.parent.mkdir(parents=True, exist_ok=True)
        document = fixture.document()
        document["nodes"][1]["opType"] = "Conv2D"
        source.write_text(json.dumps(document, indent=1))

        info = self.quantization.author_ptq_template(
            volvoxai.pb.AuthorPtqTemplateRequest(
                source_graph_path=str(source),
                weight_paths=[str(self.weights_path)],
                template_graph_path=str(self.directory / "conv" / "t.json"),
            ))
        self.assertNotEqual(
            info.report.status, volvoxai.pb.NativeStatus.NATIVE_STATUS_OK)
        self.assertIn("Conv2D", info.report.message)

    def run_pipeline(self, through: str) -> None:
        """Author, plan, calibrate, write — every step an RPC, no Python
        numerics anywhere in the path. Stops after `through`."""

        template_path, authored = self.author("pipeline")

        runtime = self.inference.create_runtime(
            volvoxai.pb.CreateRuntimeRequest())
        volvoxai.check(runtime.report, "CreateRuntime")
        try:
            model = self.inference.load_model(volvoxai.pb.LoadModelRequest(
                runtime_id=runtime.runtime_id,
                graph_path=str(self.graph_path),
                weight_paths=[str(self.weights_path)],
            ))
            volvoxai.check(model.report, "LoadModel")

            plan = self.quantization.create_ptq_plan(
                volvoxai.pb.CreatePtqPlanRequest(
                    model_id=model.model_id,
                    template_graph_path=str(template_path),
                    profile_names=["default"],
                    observers=list(authored.observers),
                    layers=list(authored.layers),
                ))
            volvoxai.check(plan.report, "CreatePtqPlan")
            self.assertGreater(plan.ptq_plan_id, 0)
            self.assertEqual([spec.name for spec in plan.inputs], ["hidden"])

            # One deterministic calibration batch. The values matter only in
            # that every observed tensor sees a finite range.
            values = np.linspace(-2.0, 2.0, 1 * 4 * 8, dtype=np.float32)
            calibrated = self.quantization.calibrate_ptq_plan(
                volvoxai.pb.CalibratePtqPlanRequest(
                    ptq_plan_id=plan.ptq_plan_id,
                    profile_name="default",
                    sample_name="batch-0",
                    sample_count=1,
                    inputs=[volvoxai.pb.Tensor(
                        name="hidden",
                        shape=[1, 4, 8],
                        dtype=volvoxai.pb.DataType.DATA_TYPE_F32,
                        inline=values.tobytes(),
                    )],
                ))
            volvoxai.check(calibrated.report, "CalibratePtqPlan")
            self.assertEqual(calibrated.calibration_batches, 1)
            self.assertTrue(calibrated.coverage_complete)
            if through == "calibrate":
                self.quantization.release_ptq_plan(
                    volvoxai.pb.PtqPlanRef(ptq_plan_id=plan.ptq_plan_id))
                self.inference.release_model(
                    volvoxai.pb.ModelRef(model_id=model.model_id))
                return

            written = self.directory / "package"
            written.mkdir(parents=True, exist_ok=True)
            package = self.quantization.write_ptq_package(
                volvoxai.pb.WritePtqPackageRequest(
                    ptq_plan_id=plan.ptq_plan_id,
                    output_graph_path=str(written / "graph.json"),
                    output_weights_path=str(written / "model.safetensors"),
                ))
            volvoxai.check(package.report, "WritePtqPackage")

            self.assertTrue((written / "graph.json").is_file())
            self.assertTrue((written / "model.safetensors").is_file())
            self.assertGreater((written / "model.safetensors").stat().st_size, 0)

            self.quantization.release_ptq_plan(
                volvoxai.pb.PtqPlanRef(ptq_plan_id=plan.ptq_plan_id))
            self.inference.release_model(
                volvoxai.pb.ModelRef(model_id=model.model_id))
        finally:
            self.inference.release_runtime(
                volvoxai.pb.RuntimeRef(runtime_id=runtime.runtime_id))

    def test_authoring_and_calibration_run_as_rpcs(self) -> None:
        """Author, create a plan bound to a Model revision, and calibrate it —
        all through the schema, with no Python numerics in the path."""

        self.run_pipeline(through="calibrate")

    def test_the_whole_pipeline_writes_a_package(self) -> None:
        """Author, plan, calibrate, write — every step an RPC."""

        self.run_pipeline(through="write")

    def test_the_written_package_runs(self) -> None:
        """A package that loads and executes is the only proof that authoring,
        calibration and packing agree with each other. Each could be
        self-consistent and still produce a graph the engine refuses."""

        package = self.directory / "package"
        if not (package / "graph.json").is_file():
            self.run_pipeline(through="write")

        runtime = self.inference.create_runtime(volvoxai.pb.CreateRuntimeRequest())
        volvoxai.check(runtime.report, "CreateRuntime")
        model = compiled = context = result = None
        try:
            model = self.inference.load_model(volvoxai.pb.LoadModelRequest(
                runtime_id=runtime.runtime_id,
                graph_path=str(package / "graph.json"),
                weight_paths=[str(package / "model.safetensors")],
            ))
            volvoxai.check(model.report, "LoadModel")
            compiled = self.inference.compile_model(
                volvoxai.pb.CompileModelRequest(model_id=model.model_id))
            volvoxai.check(compiled.report, "CompileModel")
            context = self.inference.create_execution_context(
                volvoxai.pb.CreateExecutionContextRequest(
                    compiled_model_id=compiled.compiled_model_id))
            volvoxai.check(context.report, "CreateExecutionContext")
            self.assertEqual([spec.name for spec in context.inputs], ["hidden"])

            values = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
            result = self.inference.execute(volvoxai.pb.ExecuteRequest(
                context_id=context.context_id,
                inputs=[volvoxai.pb.Tensor(
                    name="hidden",
                    shape=[1, 4, 8],
                    dtype=volvoxai.pb.DataType.DATA_TYPE_F32,
                    inline=values.tobytes(),
                )],
            ))
            volvoxai.check(result.report, "Execute")
            info = self.inference.get_result(
                volvoxai.pb.ResultRef(result_id=result.result_id))
            volvoxai.check(info.report, "GetResult")
            output = next(item for item in info.outputs if item.name == "output")
            self.assertEqual(tuple(output.shape), (1, 4, 8))
            read = self.inference.read_output(volvoxai.pb.ReadOutputRequest(
                result_id=result.result_id, name="output"))
            volvoxai.check(read.report, "ReadOutput")
            produced = np.frombuffer(read.tensor.inline, dtype=np.float32)
            self.assertTrue(np.all(np.isfinite(produced)))
        finally:
            if result is not None:
                self.inference.release_result(
                    volvoxai.pb.ResultRef(result_id=result.result_id))
            if context is not None:
                self.inference.release_execution_context(
                    volvoxai.pb.ExecutionContextRef(context_id=context.context_id))
            if compiled is not None:
                self.inference.release_compiled_model(
                    volvoxai.pb.CompiledModelRef(
                        compiled_model_id=compiled.compiled_model_id))
            if model is not None:
                self.inference.release_model(
                    volvoxai.pb.ModelRef(model_id=model.model_id))
            self.inference.release_runtime(
                volvoxai.pb.RuntimeRef(runtime_id=runtime.runtime_id))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(full_library_available(), "libvolvoxai-full.so is not built")
class InMemoryAuthoringTest(unittest.TestCase):
    """Authoring without naming a file.

    A browser has no path to give and no path to receive. The same is true of
    any host that holds a model in memory — a server that fetched it, a test
    that generated it. Bytes in, bytes out, and the authoring underneath is
    the identical code, which is the only reason the WebAssembly build can
    serve this service at all rather than growing a second implementation.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = Path(tempfile.mkdtemp(prefix="volvoxai-ptq-mem-"))
        cls.host = volvoxai.open_library("full")
        cls.quantization = volvoxai.VxQuantizationServiceClient(
            cls.host)
        cls.graph_path, cls.weights_path = write_fixture(cls.directory)
        cls.graph_bytes = cls.graph_path.read_bytes()
        cls.weight_bytes = cls.weights_path.read_bytes()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.host.close()
        shutil.rmtree(cls.directory, ignore_errors=True)

    def author_from_memory(self) -> object:
        info = self.quantization.author_ptq_template(
            volvoxai.pb.AuthorPtqTemplateRequest(
                source_graph=self.graph_bytes,
                weight_shards=[self.weight_bytes],
            ))
        volvoxai.check(info.report, "AuthorPtqTemplate")
        return info

    def test_returns_the_template_as_bytes(self) -> None:
        info = self.author_from_memory()
        self.assertTrue(info.template_graph)
        template = json.loads(info.template_graph)
        self.assertEqual(template["format"], "volvox-graph/v1")
        self.assertIn("quantization", template)
        self.assertEqual(info.quantized_nodes, 6)

    def test_matches_what_the_file_path_produces(self) -> None:
        """Two entry points, one implementation. If these diverged, a browser
        and a server would quantize the same model differently."""

        from_memory = json.loads(self.author_from_memory().template_graph)

        written = self.directory / "from-path.json"
        info = self.quantization.author_ptq_template(
            volvoxai.pb.AuthorPtqTemplateRequest(
                source_graph_path=str(self.graph_path),
                weight_paths=[str(self.weights_path)],
                template_graph_path=str(written),
            ))
        volvoxai.check(info.report, "AuthorPtqTemplate")
        from_path = json.loads(written.read_text())

        self.assertEqual(from_memory, from_path)
        # A caller who asked for a file gets the file, not both.
        self.assertFalse(info.template_graph)

    def test_a_malformed_graph_is_refused_by_message(self) -> None:
        info = self.quantization.author_ptq_template(
            volvoxai.pb.AuthorPtqTemplateRequest(
                source_graph=b"{not json",
                weight_shards=[self.weight_bytes],
            ))
        self.assertNotEqual(
            info.report.status, volvoxai.pb.NativeStatus.NATIVE_STATUS_OK)
        self.assertIn("JSON", info.report.message)

    def test_a_malformed_weight_shard_is_refused(self) -> None:
        info = self.quantization.author_ptq_template(
            volvoxai.pb.AuthorPtqTemplateRequest(
                source_graph=self.graph_bytes,
                weight_shards=[b"not a safetensors file"],
            ))
        self.assertNotEqual(
            info.report.status, volvoxai.pb.NativeStatus.NATIVE_STATUS_OK)
        self.assertIn("safetensors", info.report.message)
