"""The generated Python proto projection against a shipped native library."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))

import volvoxai  # noqa: E402


def library_available() -> bool:
    try:
        volvoxai.find_library()
    except volvoxai.VolvoxAIError:
        return False
    return True


class OperationReportTest(unittest.TestCase):
    def test_error_identity_comes_from_the_proto_enums(self) -> None:
        pb = volvoxai.pb
        for message in ("fixture missing", "out of memory", ""):
            with self.subTest(message=message):
                report = pb.OperationReport(
                    status=pb.NativeStatus.NATIVE_STATUS_NOT_FOUND,
                    code=pb.OperationCode.OPERATION_CODE_NOT_FOUND,
                    stage=pb.OperationStage.OPERATION_STAGE_MODEL_LOAD,
                    message=message,
                )
                with self.assertRaises(volvoxai.VolvoxAIError) as caught:
                    host = Mock()
                    host.unary.return_value = pb.ModelHandle(report=report).to_bytes()
                    volvoxai.VxInferenceServiceClient(host).load_model(pb.LoadModelRequest())
                self.assertEqual(caught.exception.status, pb.NativeStatus.NATIVE_STATUS_NOT_FOUND)
                self.assertEqual(caught.exception.code, pb.OperationCode.OPERATION_CODE_NOT_FOUND)
                self.assertEqual(caught.exception.stage, pb.OperationStage.OPERATION_STAGE_MODEL_LOAD)
                self.assertIn(message or "OPERATION_CODE_NOT_FOUND", str(caught.exception))
                self.assertIs(caught.exception.report, caught.exception.response.report)


@unittest.skipUnless(library_available(), "libvolvoxai.so is not built")
class GeneratedClientTest(unittest.TestCase):
    def test_invalid_calls_raise_at_the_call_site(self) -> None:
        pb = volvoxai.pb
        with volvoxai.open_library() as host:
            inference = volvoxai.VxInferenceServiceClient(host)
            runtime = inference.create_runtime(pb.CreateRuntimeRequest(cpu_threads=1))
            with self.assertRaises(volvoxai.VolvoxAIError) as caught:
                inference.load_model(pb.LoadModelRequest(
                    runtime_id=runtime.runtime_id,
                    package=pb.ModelPackage(graph_document=b"not json"),
                ))
            self.assertNotEqual(caught.exception.status, pb.NativeStatus.NATIVE_STATUS_OK)
            self.assertEqual(caught.exception.response.model_id, 0)
            self.assertIs(caught.exception.report, caught.exception.response.report)
            # A refused operation does not poison the owner or its runtime.
            inference.release_runtime(pb.RuntimeRef(runtime_id=runtime.runtime_id))
            with self.assertRaises(volvoxai.VolvoxAIError):
                inference.get_model_info(pb.ModelRef(model_id=0))

    def test_platform_reports_the_full_build(self) -> None:
        host = volvoxai.open_library()
        try:
            platform = volvoxai.VxPlatformServiceClient(host)
            info = platform.get_platform_info(volvoxai.pb.Empty())
            self.assertEqual(info.api_version, 1)
            self.assertEqual(volvoxai.pb.BuildProfile(info.profile),
                             volvoxai.pb.BuildProfile.BUILD_PROFILE_FULL)
            self.assertEqual(
                info.transport,
                volvoxai.pb.TransportProfile.TRANSPORT_PROFILE_IN_PROCESS,
            )
        finally:
            host.close()

    def test_generated_inference_client_owns_the_lifecycle(self) -> None:
        model_dir = REPOSITORY_ROOT / "models" / "tinystories_1m"
        if not model_dir.is_dir():
            self.skipTest("tinystories model fixture is absent")
        host = volvoxai.open_library()
        inference = volvoxai.VxInferenceServiceClient(host)
        handles: list[tuple[object, object]] = []
        try:
            runtime = inference.create_runtime(volvoxai.pb.CreateRuntimeRequest())
            model = inference.load_model(
                volvoxai.pb.LoadModelRequest(
                    runtime_id=runtime.runtime_id,
                    graph_path=str(model_dir / "graph.json"),
                    weight_paths=[str(model_dir / "model.safetensors")],
                )
            )
            compiled = inference.compile_model(
                volvoxai.pb.CompileModelRequest(model_id=model.model_id)
            )
            context = inference.create_execution_context(
                volvoxai.pb.CreateExecutionContextRequest(
                    compiled_model_id=compiled.compiled_model_id
                )
            )
            self.assertEqual({item.name for item in context.inputs}, {"tokens", "positions"})

            handles = [
                (inference.release_execution_context,
                 volvoxai.pb.ExecutionContextRef(context_id=context.context_id)),
                (inference.release_compiled_model,
                 volvoxai.pb.CompiledModelRef(
                     compiled_model_id=compiled.compiled_model_id)),
                (inference.release_model,
                 volvoxai.pb.ModelRef(model_id=model.model_id)),
                (inference.release_runtime,
                 volvoxai.pb.RuntimeRef(runtime_id=runtime.runtime_id)),
            ]
        finally:
            for release, reference in handles:
                release(reference)
            host.close()


if __name__ == "__main__":
    unittest.main()
