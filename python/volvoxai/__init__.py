"""NumPy workflows and Python clients for VolvoxAI's generated protobuf API.

Use ``InferenceSession(model).run(inputs)`` for NumPy inference or its async
counterpart with ``async with``. ``quantize`` and ``TrainingSession`` select
the full library for calibration and training workflows. For the
complete API, create an owner with ``open_library()`` and pass it to a generated
``Vx*ServiceClient``. Close the owner to release its module instance. Async
clients accept ``AsyncModuleHost(open_library())`` from the same pinned runtime.
Service methods return successful responses and raise ``VolvoxAIError`` for
failed operation reports. Call transport failures remain ``FfiError`` exceptions.
Linux wheels bundle the native libraries and their loader. Repository users
can build development libraries with ``make build_native_libraries``.
"""

from __future__ import annotations

import sys as _sys
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version
from pathlib import Path as _Path
from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    from ._model import ModelPackage
    from ._quantization import QuantizationResult, quantize
    from ._training import AdamW, CrossEntropyLoss, SGD, TrainingMetric, TrainingResult, TrainingSession

try:
    __version__ = _version("volvoxai")
except _PackageNotFoundError:
    import json as _json
    __version__ = _json.loads(
        (_Path(__file__).resolve().parents[2] / "package.json").read_text()
    )["version"]

# The generated modules import each other by flat name (`import volvoxai_lite`),
# which is what makes them equally usable from a plain script, a vendored copy,
# and an installed wheel. Put their directory on the path here so importing
# this package is all a caller has to do.
_GENERATED = _Path(__file__).resolve().parents[2] / "runtime" / "generated" / "python"
if _GENERATED.is_dir() and str(_GENERATED) not in _sys.path:
    _sys.path.insert(0, str(_GENERATED))

import volvoxai_lite as pb  # noqa: E402
from ._clients import (  # noqa: E402
    VxPlatformServiceClient,
    VxProfilingServiceClient,
    VxProfilingServiceAsyncClient,
    VxPlatformServiceAsyncClient,
    VxTextServiceClient,
    VxTextServiceAsyncClient,
    VxPlanningServiceClient,
    VxPlanningServiceAsyncClient,
    VxBufferServiceClient,
    VxBufferServiceAsyncClient,
    VxInferenceServiceClient,
    VxInferenceServiceAsyncClient,
    VxSchedulerServiceClient,
    VxSchedulerServiceAsyncClient,
    VxTrainingServiceClient,
    VxTrainingServiceAsyncClient,
    VxQuantizationServiceClient,
    VxQuantizationServiceAsyncClient,
)
from synurang import AsyncModuleHost, FfiError, ModuleHost, PluginClosedError  # noqa: E402

from ._library import find_library, library_filename, open_library  # noqa: E402
from .errors import VolvoxAIError  # noqa: E402
from ._session import InferenceSession  # noqa: E402
from ._async_session import AsyncInferenceSession  # noqa: E402
from ._metadata import Dimension, TensorSpec  # noqa: E402
from ._runtime import Runtime
from ._tensor import Tensor, TensorOutputs  # noqa: E402


def __getattr__(name):
    # Plain inference imports neither training/quantization nor exporter frontends.
    import importlib
    modules = {"quantize": "_quantization", "QuantizationResult": "_quantization",
               "ModelPackage": "_model", "TrainingSession": "_training",
               "CrossEntropyLoss": "_training", "AdamW": "_training", "SGD": "_training",
               "TrainingResult": "_training", "TrainingMetric": "_training"}
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{modules[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "AsyncInferenceSession",
    "Dimension",
    "TensorSpec",
    "Tensor",
    "TensorOutputs",
    "quantize",
    "QuantizationResult",
    "ModelPackage",
    "TrainingSession",
    "CrossEntropyLoss",
    "AdamW",
    "SGD",
    "TrainingResult",
    "TrainingMetric",
    "AsyncModuleHost",
    "FfiError",
    "InferenceSession",
    "ModuleHost",
    "PluginClosedError",
    "VolvoxAIError",
    "VxPlatformServiceClient",
    "VxProfilingServiceClient",
    "VxProfilingServiceAsyncClient",
    "VxPlatformServiceAsyncClient",
    "VxTextServiceClient",
    "VxTextServiceAsyncClient",
    "VxPlanningServiceClient",
    "VxPlanningServiceAsyncClient",
    "VxBufferServiceClient",
    "VxBufferServiceAsyncClient",
    "Runtime",
    "VxInferenceServiceClient",
    "VxInferenceServiceAsyncClient",
    "VxSchedulerServiceClient",
    "VxSchedulerServiceAsyncClient",
    "VxTrainingServiceClient",
    "VxTrainingServiceAsyncClient",
    "VxQuantizationServiceClient",
    "VxQuantizationServiceAsyncClient",
    "find_library",
    "library_filename",
    "open_library",
    "pb",
]
