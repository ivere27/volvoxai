"""Public Python calls raise at the failure boundary and retain typed evidence."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
import sys
from typing import get_type_hints
import unittest
from unittest.mock import patch

if not sys.flags.isolated:
    source_python = Path(__file__).resolve().parent.parent
    if (source_python / "volvoxai/__init__.py").is_file():
        sys.path.insert(0, str(source_python))

import volvoxai as vx
import volvoxai_client as generated
from synurang import DecodeError

pb = vx.pb


class BytesHost:
    def __init__(self, response=None, *, error=None):
        self.payload = response.to_bytes() if response is not None else b""
        self.error = error
        self.calls = []

    def unary(self, method, request, *, timeout=None):
        self.calls.append((method, request, timeout))
        if self.error is not None:
            raise self.error
        return self.payload


class AsyncBytesHost(BytesHost):
    async def unary(self, method, request, *, timeout=None):
        return super().unary(method, request, timeout=timeout)


class ClientErrorTest(unittest.TestCase):
    def test_all_generated_services_keep_their_signatures_and_raise(self):
        async def exercise():
            counts = {False: 0, True: 0}
            for name, original in vars(generated).items():
                if not name.startswith("Vx") or not name.endswith("Client"):
                    continue
                public = getattr(vx, name)
                for method_name, method in vars(original).items():
                    if method_name.startswith("_"):
                        continue
                    with self.subTest(service=name, method=method_name):
                        annotations = get_type_hints(method)
                        request = annotations["request"]()
                        response_type = annotations["return"]
                        asynchronous = inspect.iscoroutinefunction(method)
                        counts[asynchronous] += 1
                        self.assertEqual(inspect.signature(getattr(public, method_name)),
                                         inspect.signature(method))
                        self.assertEqual(get_type_hints(getattr(public, method_name)), annotations)
                        self.assertEqual(inspect.iscoroutinefunction(getattr(public, method_name)),
                                         asynchronous)
                        report = pb.OperationReport(
                            status=pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT,
                            code=pb.OperationCode.OPERATION_CODE_INVALID_ARGUMENT,
                            message="public call refused",
                        )
                        response = response_type()
                        if isinstance(response, pb.OperationReport):
                            response = report
                        elif "report" in response_type.__fields_by_name__:
                            response.report = report
                        else:
                            report = None
                        host = (AsyncBytesHost if asynchronous else BytesHost)(response)
                        call = getattr(public(host), method_name)
                        if report is not None:
                            with self.assertRaises(vx.VolvoxAIError) as caught:
                                result = call(request, timeout=1.25)
                                if asynchronous:
                                    await result
                            error = caught.exception
                            self.assertEqual(error.response.to_dict(), response.to_dict())
                            self.assertEqual(error.report.to_dict(), report.to_dict())
                            self.assertEqual(error.operation, method.__qualname__)
                        else:
                            result = call(request, timeout=1.25)
                            if asynchronous:
                                result = await result
                            self.assertEqual(result.to_dict(), response.to_dict())
                        self.assertEqual(len(host.calls), 1)
                        self.assertEqual(host.calls[0][1:], (request.to_bytes(), 1.25))
            self.assertEqual(counts, {False: 107, True: 107})
        asyncio.run(exercise())

    def test_success_decodes_once_and_keeps_backend_evidence(self):
        response = pb.CompiledModelHandle(compiled_model_id=41,
            report=pb.OperationReport(backend="cpu", route=pb.RouteEvidence(attested=True)))
        host = BytesHost(response)
        with patch.object(pb.CompiledModelHandle, "from_bytes",
                          wraps=pb.CompiledModelHandle.from_bytes) as decode:
            result = vx.VxInferenceServiceClient(host).compile_model(pb.CompileModelRequest())
        self.assertEqual(result.compiled_model_id, 41)
        self.assertEqual(result.report.backend, "cpu")
        self.assertTrue(result.report.route.attested)
        self.assertEqual(decode.call_count, 1)
        self.assertEqual(len(host.calls), 1)

    def test_every_non_ok_status_is_an_exception(self):
        for status in pb.NativeStatus:
            if status == pb.NativeStatus.NATIVE_STATUS_OK:
                continue
            with self.subTest(status=status):
                host = BytesHost(pb.OperationReport(status=status))
                with self.assertRaises(vx.VolvoxAIError) as caught:
                    vx.VxInferenceServiceClient(host).release_model(pb.ModelRef())
                self.assertEqual(caught.exception.status, status)
                self.assertIs(caught.exception.response, caught.exception.report)

    def test_missing_required_report_is_a_protocol_failure(self):
        with self.assertRaises(vx.VolvoxAIError) as caught:
            vx.VxInferenceServiceClient(BytesHost(pb.ModelHandle())).load_model(pb.LoadModelRequest())
        self.assertEqual(caught.exception.status, pb.NativeStatus.NATIVE_STATUS_INTERNAL)
        self.assertEqual(caught.exception.code, pb.OperationCode.OPERATION_CODE_INTERNAL)
        self.assertIsNone(caught.exception.report)
        self.assertIsInstance(caught.exception.response, pb.ModelHandle)

    def test_future_enum_values_are_preserved(self):
        host = BytesHost(pb.OperationReport(status=-500, code=500, stage=500))
        with self.assertRaises(vx.VolvoxAIError) as caught:
            vx.VxInferenceServiceClient(host).release_model(pb.ModelRef())
        self.assertEqual((caught.exception.status, caught.exception.code, caught.exception.stage),
                         (-500, 500, 500))

    def test_pending_results_and_nested_evidence_are_successful_responses(self):
        pending = pb.ResultInfo(state=pb.ResultState.RESULT_STATE_PENDING,
                                report=pb.OperationReport())
        result = vx.VxInferenceServiceClient(BytesHost(pending)).get_result(pb.ResultRef())
        self.assertEqual(result.state, pb.ResultState.RESULT_STATE_PENDING)
        # A returned plan/revision is evidence, not another operation response.
        package = pb.PtqPackageInfo(report=pb.OperationReport(),
                                   plan=pb.PtqPlanInfo(revision=pb.RevisionInfo()))
        result = vx.VxQuantizationServiceClient(BytesHost(package)).write_ptq_package(
            pb.WritePtqPackageRequest())
        self.assertIsNone(result.plan.report)
        self.assertIsNone(result.plan.revision.report)
        # DescribeStatus returns status data without an operation report.
        description = pb.StatusDescription(status=pb.NativeStatus.NATIVE_STATUS_BUSY)
        result = vx.VxPlatformServiceClient(BytesHost(description)).describe_status(
            pb.DescribeStatusRequest())
        self.assertEqual(result.status, pb.NativeStatus.NATIVE_STATUS_BUSY)

    def test_transport_and_decode_errors_are_not_reclassified(self):
        error = vx.FfiError("deadline exceeded", grpc_code=4)
        with self.assertRaises(vx.FfiError) as caught:
            vx.VxInferenceServiceClient(BytesHost(error=error)).create_runtime(pb.CreateRuntimeRequest())
        self.assertIs(caught.exception, error)
        host = BytesHost()
        host.payload = b"\x80"
        with self.assertRaises(DecodeError):
            vx.VxInferenceServiceClient(host).create_runtime(pb.CreateRuntimeRequest())

    def test_async_cancellation_and_transport_errors_are_preserved(self):
        async def exercise():
            for error in (asyncio.CancelledError(), vx.FfiError("deadline", grpc_code=4)):
                with self.subTest(error=type(error).__name__):
                    with self.assertRaises(type(error)) as caught:
                        await vx.VxInferenceServiceAsyncClient(AsyncBytesHost(error=error)).create_runtime(
                            pb.CreateRuntimeRequest())
                    self.assertIs(caught.exception, error)
        asyncio.run(exercise())

    def test_raw_generated_client_is_not_modified(self):
        response = pb.ModelHandle(report=pb.OperationReport(
            status=pb.NativeStatus.NATIVE_STATUS_NOT_FOUND))
        result = generated.VxInferenceServiceClient(BytesHost(response)).load_model(pb.LoadModelRequest())
        self.assertEqual(result.report.status, pb.NativeStatus.NATIVE_STATUS_NOT_FOUND)
        self.assertFalse(hasattr(vx, "check"))

    def test_async_native_failure(self):
        try:
            vx.find_library()
        except vx.VolvoxAIError:
            self.skipTest("the native library is not built")

        async def exercise():
            async with vx.AsyncModuleHost(vx.open_library()) as host:
                inference = vx.VxInferenceServiceAsyncClient(host)
                runtime = await inference.create_runtime(pb.CreateRuntimeRequest(cpu_threads=1))
                with self.assertRaises(vx.VolvoxAIError) as caught:
                    await inference.load_model(pb.LoadModelRequest(
                        runtime_id=runtime.runtime_id,
                        package=pb.ModelPackage(graph_document=b"not json")))
                self.assertIs(caught.exception.report, caught.exception.response.report)
                await inference.release_runtime(pb.RuntimeRef(runtime_id=runtime.runtime_id))
            with self.assertRaises(vx.PluginClosedError):
                await inference.create_runtime(pb.CreateRuntimeRequest())
        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main(verbosity=2)
