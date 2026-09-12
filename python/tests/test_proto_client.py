"""The generated Python proto projection against a shipped native library."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))

import volvoxai  # noqa: E402


def library_available(profile: str) -> bool:
    try:
        volvoxai.find_library(profile)
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
                    volvoxai.check(report, "LoadModel")
                self.assertEqual(caught.exception.status, pb.NativeStatus.NATIVE_STATUS_NOT_FOUND)
                self.assertEqual(caught.exception.code, pb.OperationCode.OPERATION_CODE_NOT_FOUND)
                self.assertEqual(caught.exception.stage, pb.OperationStage.OPERATION_STAGE_MODEL_LOAD)
                self.assertIn(message or "OPERATION_CODE_NOT_FOUND", str(caught.exception))


@unittest.skipUnless(library_available("inference"), "libvolvoxai.so is not built")
class GeneratedClientTest(unittest.TestCase):
    def test_platform_and_profile(self) -> None:
        expected = {
            "inference": volvoxai.pb.BuildProfile.BUILD_PROFILE_INFERENCE,
            "full": volvoxai.pb.BuildProfile.BUILD_PROFILE_FULL,
        }
        for profile, build in expected.items():
            if not library_available(profile):
                continue
            with self.subTest(profile=profile):
                host = volvoxai.open_library(profile)
                try:
                    platform = volvoxai.VxPlatformServiceClient(host)
                    info = platform.get_platform_info(volvoxai.pb.Empty())
                    self.assertEqual(info.api_version, 1)
                    self.assertEqual(volvoxai.pb.BuildProfile(info.profile), build)
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
        host = volvoxai.open_library("inference")
        inference = volvoxai.VxInferenceServiceClient(host)
        handles: list[tuple[object, object]] = []
        try:
            runtime = inference.create_runtime(volvoxai.pb.CreateRuntimeRequest())
            volvoxai.check(runtime.report, "CreateRuntime")
            model = inference.load_model(
                volvoxai.pb.LoadModelRequest(
                    runtime_id=runtime.runtime_id,
                    graph_path=str(model_dir / "graph.json"),
                    weight_paths=[str(model_dir / "model.safetensors")],
                )
            )
            volvoxai.check(model.report, "LoadModel")
            compiled = inference.compile_model(
                volvoxai.pb.CompileModelRequest(model_id=model.model_id)
            )
            volvoxai.check(compiled.report, "CompileModel")
            context = inference.create_execution_context(
                volvoxai.pb.CreateExecutionContextRequest(
                    compiled_model_id=compiled.compiled_model_id
                )
            )
            volvoxai.check(context.report, "CreateExecutionContext")
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
                volvoxai.check(release(reference), release.__name__)
            host.close()


if __name__ == "__main__":
    unittest.main()
