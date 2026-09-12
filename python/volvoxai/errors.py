"""Convert a decoded OperationReport into a typed VolvoxAI domain error.

Generated clients preserve Synurang module-call transport errors. This helper
checks the separate, richer engine report carried by an operation response.
"""

from __future__ import annotations

import volvoxai_lite as pb


class VolvoxAIError(RuntimeError):
    """A VolvoxAI operation failed.

    ``status``, ``stage`` and ``code`` are populated when the failure came back
    as an ``OperationReport``; they are ``None`` for transport and setup
    failures, which have no report to carry them.
    """

    def __init__(
        self,
        message: str,
        *,
        status: pb.NativeStatus | None = None,
        stage: pb.OperationStage | None = None,
        code: pb.OperationCode | None = None,
        backend: str = "",
    ) -> None:
        super().__init__(message)
        self.status = status
        self.stage = stage
        self.code = code
        self.backend = backend


def check(report: pb.OperationReport | None, operation: str) -> pb.OperationReport:
    """Raise unless ``report`` says the operation succeeded.

    A missing report is itself a failure: every handler sets one, so its
    absence means the response did not come from a handler that ran.
    """

    if report is None:
        raise VolvoxAIError(f"{operation}: response carried no report")
    status = pb.NativeStatus(report.status)
    if status == pb.NativeStatus.NATIVE_STATUS_OK:
        return report
    code = pb.OperationCode(report.code)
    detail = report.message or (
        code.name if code != pb.OperationCode.OPERATION_CODE_NONE else status.name
    )
    where = pb.OperationStage(report.stage).name if report.stage else ""
    location = f" at {where}" if where else ""
    raise VolvoxAIError(
        f"{operation} failed{location}: {detail}",
        status=status,
        stage=pb.OperationStage(report.stage),
        code=code,
        backend=report.backend,
    )
