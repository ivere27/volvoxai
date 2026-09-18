"""Python exceptions for the engine's generated OperationReport contract."""

from __future__ import annotations

import volvoxai_lite as pb
from synurang import ProtoMessage


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
        status: pb.NativeStatus | int | None = None,
        stage: pb.OperationStage | int | None = None,
        code: pb.OperationCode | int | None = None,
        backend: str = "",
        report: pb.OperationReport | None = None,
        response: ProtoMessage | None = None,
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
        self.node = report.offending_node.index if report and report.offending_node else None


def _enum_value(enum, value):
    try:
        return enum(value)
    except ValueError:
        return value


def _raise_for_report(
    response: ProtoMessage, report: pb.OperationReport | None, operation: str,
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
    status = _enum_value(pb.NativeStatus, report.status)
    code = _enum_value(pb.OperationCode, report.code)
    stage = _enum_value(pb.OperationStage, report.stage)
    detail_code = code if code != pb.OperationCode.OPERATION_CODE_NONE else status
    detail = report.message or getattr(detail_code, "name", str(detail_code))
    where = getattr(stage, "name", str(stage)) if stage else ""
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
