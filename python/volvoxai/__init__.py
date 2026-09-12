"""Source-build Python clients for VolvoxAI's generated protobuf API.

Create an owner with ``open_library()`` and pass it directly to a generated
``Vx*ServiceClient``. Close the owner to release its module instance. Async
clients accept ``AsyncModuleHost(open_library())`` from the same pinned runtime.
The fixed release inventory does not include this Python package or shared
libraries; ``make build_native_libraries`` builds the development libraries.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# The generated modules import each other by flat name (`import volvoxai_lite`),
# which is what makes them equally usable from a plain script, a vendored copy,
# and an installed wheel. Put their directory on the path here so importing
# this package is all a caller has to do.
_GENERATED = _Path(__file__).resolve().parents[2] / "runtime" / "generated" / "python"
if _GENERATED.is_dir() and str(_GENERATED) not in _sys.path:
    _sys.path.insert(0, str(_GENERATED))

import volvoxai_lite as pb  # noqa: E402
from volvoxai_client import (  # noqa: E402
    VxPlatformServiceClient,
    VxPlatformServiceAsyncClient,
    VxTextServiceClient,
    VxTextServiceAsyncClient,
    VxPlanningServiceClient,
    VxPlanningServiceAsyncClient,
    VxInferenceServiceClient,
    VxInferenceServiceAsyncClient,
    VxSchedulerServiceClient,
    VxSchedulerServiceAsyncClient,
    VxTrainingServiceClient,
    VxTrainingServiceAsyncClient,
    VxQuantizationServiceClient,
    VxQuantizationServiceAsyncClient,
)
from synurang import AsyncModuleHost, ModuleHost  # noqa: E402

from ._library import find_library, library_filename, open_library  # noqa: E402
from .errors import VolvoxAIError, check  # noqa: E402

__all__ = [
    "AsyncModuleHost",
    "ModuleHost",
    "VolvoxAIError",
    "VxPlatformServiceClient",
    "VxPlatformServiceAsyncClient",
    "VxTextServiceClient",
    "VxTextServiceAsyncClient",
    "VxPlanningServiceClient",
    "VxPlanningServiceAsyncClient",
    "VxInferenceServiceClient",
    "VxInferenceServiceAsyncClient",
    "VxSchedulerServiceClient",
    "VxSchedulerServiceAsyncClient",
    "VxTrainingServiceClient",
    "VxTrainingServiceAsyncClient",
    "VxQuantizationServiceClient",
    "VxQuantizationServiceAsyncClient",
    "check",
    "find_library",
    "library_filename",
    "open_library",
    "pb",
]
