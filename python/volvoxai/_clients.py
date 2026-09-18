"""Apply Python exception semantics to the generated service methods.

The generator owns every signature, codec and dispatch path. The adapter checks
the already decoded response once, with no additional RPC or protobuf decoding.
Generated classes and the vendored Synurang runtime are never modified.
"""

from __future__ import annotations

from functools import wraps
from inspect import iscoroutinefunction
from typing import TypeVar, get_type_hints

import volvoxai_client as generated
import volvoxai_lite as pb

from .errors import _raise_for_report


_Client = TypeVar("_Client")


def _operation(method, *, direct: bool):
    def validate(response):
        report = response if direct else response.report
        _raise_for_report(response, report, method.__qualname__)
        return response

    if iscoroutinefunction(method):
        @wraps(method)
        async def asynchronous(self, request, *, timeout=None):
            return validate(await method(self, request, timeout=timeout))
        wrapped = asynchronous
    else:
        @wraps(method)
        def synchronous(self, request, *, timeout=None):
            return validate(method(self, request, timeout=timeout))
        wrapped = synchronous
    # Resolve the generator's postponed annotations in its own namespace so
    # get_type_hints() also works on the public adapter.
    wrapped.__annotations__ = get_type_hints(method)
    return wrapped


def _service(client: type[_Client]) -> type[_Client]:
    for name, method in vars(client.__bases__[0]).items():
        if name.startswith("_"):
            continue
        response = get_type_hints(method)["return"]
        direct = response is pb.OperationReport
        report = response.__fields_by_name__.get("report")
        if direct or (report and report.type_name == pb.OperationReport.__proto_name__):
            setattr(client, name, _operation(method, direct=direct))
    return client


@_service
class VxPlatformServiceClient(generated.VxPlatformServiceClient):
    """Platform and API discovery; failed operations raise VolvoxAIError."""


@_service
class VxPlatformServiceAsyncClient(generated.VxPlatformServiceAsyncClient):
    """Async platform and API discovery."""


@_service
class VxTextServiceClient(generated.VxTextServiceClient):
    """Text processing; failed operations raise VolvoxAIError."""


@_service
class VxTextServiceAsyncClient(generated.VxTextServiceAsyncClient):
    """Async text processing."""


@_service
class VxPlanningServiceClient(generated.VxPlanningServiceClient):
    """Graph planning; failed operations raise VolvoxAIError."""


@_service
class VxPlanningServiceAsyncClient(generated.VxPlanningServiceAsyncClient):
    """Async graph planning."""


@_service
class VxInferenceServiceClient(generated.VxInferenceServiceClient):
    """Model execution; failed operations raise VolvoxAIError."""


@_service
class VxInferenceServiceAsyncClient(generated.VxInferenceServiceAsyncClient):
    """Async model execution."""


@_service
class VxBufferServiceClient(generated.VxBufferServiceClient):
    """Storage, access leases, copies and standard DLPack ownership."""


@_service
class VxBufferServiceAsyncClient(generated.VxBufferServiceAsyncClient):
    """Async storage operations."""


@_service
class VxSchedulerServiceClient(generated.VxSchedulerServiceClient):
    """Execution scheduling; failed operations raise VolvoxAIError."""


@_service
class VxSchedulerServiceAsyncClient(generated.VxSchedulerServiceAsyncClient):
    """Async execution scheduling."""


@_service
class VxTrainingServiceClient(generated.VxTrainingServiceClient):
    """Training and adapters; failed operations raise VolvoxAIError."""


@_service
class VxTrainingServiceAsyncClient(generated.VxTrainingServiceAsyncClient):
    """Async training and adapters."""


@_service
class VxQuantizationServiceClient(generated.VxQuantizationServiceClient):
    """PTQ and weight conversion; failed operations raise VolvoxAIError."""


@_service
class VxQuantizationServiceAsyncClient(generated.VxQuantizationServiceAsyncClient):
    """Async PTQ and weight conversion."""
