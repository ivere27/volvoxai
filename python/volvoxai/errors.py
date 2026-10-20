"""Python exceptions for the engine's generated OperationReport contract."""

from __future__ import annotations

import volvoxai_pb2 as pb
from google.protobuf.message import Message


class VolvoxAIError(RuntimeError):
    """A VolvoxAI operation failed.

    Service clients raise this automatically for a failed operation. ``report``
    and ``response`` retain all of the engine's evidence, including diagnostics
    and any returned handles. ``operation`` identifies the Python method.

    Status, stage and code come from the proto enums; unknown future values are
    retained as integers. Setup failures have no report. Synurang call failures
    remain ``FfiError`` exceptions, and asyncio cancellation is propagated.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        stage: int | None = None,
        code: int | None = None,
        backend: str = "",
        report: pb.OperationReport | None = None,
        response: Message | None = None,
        operation: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.stage = stage
        self.code = code
        self.backend = backend
        self.report = report
        self.response = response
        self.operation = operation
        self.node = report.offending_node.index if report is not None and report.HasField("offending_node") else None


def _enum_name(enum, value):
    try:
        return enum.Name(value)
    except ValueError:
        return str(value)


def _raise_for_report(
    response: Message, report: pb.OperationReport | None, operation: str,
) -> None:
    """Check only the response's schema-selected operation report."""

    if report is None:
        raise VolvoxAIError(
            f"{operation}: response carried no OperationReport",
            status=pb.NativeStatus.NATIVE_STATUS_INTERNAL,
            code=pb.OperationCode.OPERATION_CODE_INTERNAL,
            stage=pb.OperationStage.OPERATION_STAGE_NONE,
            response=response,
            operation=operation,
        )
    if report.status == pb.NativeStatus.NATIVE_STATUS_OK:
        return
    status = report.status
    code = report.code
    stage = report.stage
    detail_code = code if code != pb.OperationCode.OPERATION_CODE_NONE else status
    detail = report.message or _enum_name(
        pb.OperationCode if code != pb.OperationCode.OPERATION_CODE_NONE else pb.NativeStatus, detail_code)
    where = _enum_name(pb.OperationStage, stage) if stage else ""
    location = f" at {where}" if where else ""
    raise VolvoxAIError(
        f"{operation} failed{location}: {detail}",
        status=status,
        stage=stage,
        code=code,
        backend=report.backend,
        report=report,
        response=response,
        operation=operation,
    )
