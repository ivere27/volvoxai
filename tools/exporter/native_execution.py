"""Thin fixed-graph execution harness backed only by the native C runtime.

This module owns container validation and temporary package I/O, not numerical
semantics. It publishes strict ``volvox-graph/v1`` graphs, invokes the fixed
native inference CLI, and copies its typed raw outputs into immutable arrays.

Intermediate capture is structural: private graph clones expose node outputs
as ordinary public outputs. No operator is evaluated or approximated here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, NoReturn

import numpy as np

from .errors import Diagnostic, ExporterError
from .ir import GraphIR, IRDialect
from .optimizer.safetensors_io import write_safetensors
from .runtime_ir import export_runtime_package


_RUNNER_ENV = "VOLVOXAI_NATIVE_RUNNER"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_PACKAGED_RUNNER = Path(__file__).resolve().parents[1] / "bin" / "volvoxai"
_FIXED_RUNNER = (
    _PACKAGED_RUNNER if _PACKAGED_RUNNER.is_file()
    else _REPOSITORY_ROOT / "native" / "volvoxai"
)
_CLI_BINDING_LIMIT = 32
_NUMPY_DTYPES = {
    "float32": np.dtype(np.float32),
    "int32": np.dtype(np.int32),
    "int8": np.dtype(np.int8),
    "uint8": np.dtype(np.uint8),
}
_RAW_SUFFIXES = {
    "float32": ".f32",
    "int32": ".i32",
    "int8": ".i8",
    "uint8": ".u8",
}


def _fail(code: str, message: str, *, tensor: str | None = None) -> NoReturn:
    raise ExporterError(Diagnostic(
        code=code,
        message=message,
        stage="native-execute",
        source_node=tensor,
    ))


def _readonly_copy(value: Any) -> np.ndarray:
    copied = np.array(value, copy=True, order="C")
    copied.setflags(write=False)
    return copied


def _native_runner_path() -> Path:
    configured = os.environ.get(_RUNNER_ENV)
    candidate = Path(configured) if configured else _FIXED_RUNNER
    if configured and not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        _fail(
            "VXCNATIVE001",
            f"native C runner is unavailable at {candidate}; build the fixed "
            f"'volvoxai' CMake target or set {_RUNNER_ENV} to its exact path",
        )
    return candidate


def _capture_order(graph: GraphIR) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for node in graph.nodes:
        for port in node.outputs:
            if port.value is not None and port.value not in seen:
                names.append(port.value)
                seen.add(port.value)
    for name in graph.outputs:
        if name not in seen:
            names.append(name)
            seen.add(name)
    return tuple(names)


@dataclass(frozen=True)
class BoundNativeInput:
    """One exact public input owned by a native execution binding."""

    name: str
    data: np.ndarray


@dataclass(frozen=True)
class NativeInputBinding:
    """Complete fixed input set accepted by one native runner instance."""

    inputs: tuple[BoundNativeInput, ...]
    _runner_identity: object

    def input(self, name: str) -> BoundNativeInput | None:
        for value in self.inputs:
            if value.name == name:
                return value
        return None


@dataclass(frozen=True)
class NativeExecution:
    """Immutable C-runtime outputs and structurally captured intermediates."""

    outputs: Mapping[str, np.ndarray]
    intermediates: Mapping[str, np.ndarray]
    tensors: Mapping[str, np.ndarray]
    binding: NativeInputBinding

    def tensor(self, name: str) -> np.ndarray:
        return self.tensors[name]


class NativeGraphRunner:
    """Publish and execute one concrete RuntimeIR graph through native C."""

    def __init__(self, graph: GraphIR, initializers: Mapping[str, Any]):
        if not isinstance(graph, GraphIR):
            raise TypeError("graph must be a GraphIR")
        graph.verify(IRDialect.RUNTIME)
        graph.verify_concrete_bound()
        if len(graph.inputs) > _CLI_BINDING_LIMIT:
            _fail(
                "VXCNATIVE002",
                f"native CLI accepts at most {_CLI_BINDING_LIMIT} public inputs",
            )
        self.graph = graph
        self._runner = _native_runner_path()
        self._binding_identity = object()
        self._initializers = self._load_initializers(initializers)

    def bind(self, inputs: Mapping[str, Any]) -> NativeInputBinding:
        if not isinstance(inputs, Mapping):
            _fail("VXCNATIVE002", "native inputs must be a tensor mapping")
        if any(not isinstance(name, str) for name in inputs):
            _fail("VXCNATIVE002", "public input names must be strings")
        expected = set(self.graph.inputs)
        actual = set(inputs)
        if actual != expected:
            _fail(
                "VXCNATIVE003",
                "native inputs must exactly match the graph interface "
                f"(missing={sorted(expected - actual)!r}, "
                f"unexpected={sorted(actual - expected)!r})",
            )
        return NativeInputBinding(
            inputs=tuple(
                BoundNativeInput(
                    name=name,
                    data=self._validate_array(name, inputs[name], role="input"),
                )
                for name in self.graph.inputs
            ),
            _runner_identity=self._binding_identity,
        )

    def run(self, binding: NativeInputBinding) -> NativeExecution:
        if not isinstance(binding, NativeInputBinding):
            _fail(
                "VXCNATIVE002",
                "native execution requires a NativeInputBinding; call bind first",
            )
        if binding._runner_identity is not self._binding_identity:
            _fail("VXCNATIVE003", "native input binding is stale or fabricated")
        if tuple(value.name for value in binding.inputs) != tuple(self.graph.inputs):
            _fail("VXCNATIVE003", "native input binding is stale or fabricated")
        for value in binding.inputs:
            self._validate_array(value.name, value.data, role="bound input")
        return self._run_bound(binding)

    def _load_initializers(
        self,
        initializers: Mapping[str, Any],
    ) -> dict[str, np.ndarray]:
        if not isinstance(initializers, Mapping):
            _fail("VXCNATIVE004", "initializers must be a tensor mapping")
        expected = {
            name for name, tensor in self.graph.tensors.items()
            if tensor.initializer
        }
        actual = set(initializers)
        if actual != expected:
            _fail(
                "VXCNATIVE004",
                "initializer inventory differs from RuntimeIR "
                f"(missing={sorted(expected - actual)!r}, "
                f"unexpected={sorted(actual - expected)!r})",
            )
        return {
            name: self._validate_array(
                name, initializers[name], role="initializer",
            )
            for name in self.graph.tensors
            if name in expected
        }

    def _validate_array(self, name: str, value: Any, *, role: str) -> np.ndarray:
        descriptor = self.graph.tensors[name]
        dtype = _NUMPY_DTYPES.get(descriptor.dtype)
        if dtype is None:
            _fail(
                "VXCNATIVE005",
                f"tensor {name!r} has unsupported dtype {descriptor.dtype!r}",
                tensor=name,
            )
        array = np.asarray(value)
        expected_shape = tuple(int(extent) for extent in descriptor.shape)
        if array.shape != expected_shape or array.dtype != dtype:
            _fail(
                "VXCNATIVE005",
                f"{role} {name!r} is shape={array.shape}, dtype={array.dtype}; "
                f"expected shape={expected_shape}, dtype={dtype}",
                tensor=name,
            )
        return _readonly_copy(array)

    @staticmethod
    def _capture_graph(graph: GraphIR, outputs: tuple[str, ...]) -> GraphIR:
        captured = copy.deepcopy(graph)
        captured.outputs[:] = list(outputs)
        selected = set(outputs)
        for name, tensor in captured.tensors.items():
            tensor.public_output = name in selected
        captured.verify(IRDialect.RUNTIME)
        captured.verify_concrete_bound()
        return captured

    def _run_bound(self, binding: NativeInputBinding) -> NativeExecution:
        capture_names = _capture_order(self.graph)
        captures: dict[str, np.ndarray] = {}
        with tempfile.TemporaryDirectory(
            prefix="volvoxai-native-execution-",
        ) as temporary:
            root = Path(temporary)
            input_paths: dict[str, Path] = {}
            for position, value in enumerate(binding.inputs):
                suffix = _RAW_SUFFIXES[self.graph.tensors[value.name].dtype]
                path = root / f"input-{position}{suffix}"
                path.write_bytes(value.data.tobytes(order="C"))
                input_paths[value.name] = path

            for start in range(0, len(capture_names), _CLI_BINDING_LIMIT):
                chunk = capture_names[start:start + _CLI_BINDING_LIMIT]
                published = self._capture_graph(self.graph, chunk)
                document, packaged = export_runtime_package(
                    published,
                    self._initializers,
                )
                (root / "graph.json").write_text(
                    json.dumps(
                        document,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    encoding="utf-8",
                )
                write_safetensors(root / "model.safetensors", packaged)

                command = [
                    os.fspath(self._runner),
                    "run",
                    os.fspath(root),
                    "--cpu",
                    "--threads",
                    "1",
                ]
                for name in self.graph.inputs:
                    command.extend(("--input", f"{name}={input_paths[name]}"))
                output_paths: dict[str, Path] = {}
                for position, name in enumerate(chunk):
                    suffix = _RAW_SUFFIXES[self.graph.tensors[name].dtype]
                    path = root / f"output-{start + position}{suffix}"
                    output_paths[name] = path
                    command.extend(("--output", f"{name}={path}"))
                completed = subprocess.run(
                    command,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if completed.returncode != 0:
                    detail = completed.stderr.strip() or completed.stdout.strip()
                    _fail(
                        "VXCNATIVE006",
                        "native C execution failed with status "
                        f"{completed.returncode}: {detail}",
                    )
                for name in chunk:
                    descriptor = self.graph.tensors[name]
                    dtype = _NUMPY_DTYPES[descriptor.dtype]
                    payload = output_paths[name].read_bytes()
                    shape = tuple(int(extent) for extent in descriptor.shape)
                    expected_elements = 1
                    for extent in shape:
                        expected_elements *= extent
                    expected_bytes = expected_elements * dtype.itemsize
                    if len(payload) != expected_bytes:
                        _fail(
                            "VXCNATIVE007",
                            f"native output {name!r} has {len(payload)} bytes; "
                            f"expected {expected_bytes}",
                            tensor=name,
                        )
                    captures[name] = _readonly_copy(
                        np.frombuffer(payload, dtype=dtype).reshape(shape),
                    )

        tensors = {
            name: _readonly_copy(value)
            for name, value in self._initializers.items()
        }
        tensors.update({
            value.name: _readonly_copy(value.data) for value in binding.inputs
        })
        tensors.update(captures)
        return NativeExecution(
            outputs=MappingProxyType({
                name: captures[name] for name in self.graph.outputs
            }),
            intermediates=MappingProxyType(dict(captures)),
            tensors=MappingProxyType(tensors),
            binding=binding,
        )


def execute_native_graph(
    graph: GraphIR,
    initializers: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> NativeExecution:
    """Execute one concrete RuntimeIR graph through the native C runtime."""

    runner = NativeGraphRunner(graph, initializers)
    return runner.run(runner.bind(inputs))


__all__ = [
    "BoundNativeInput",
    "NativeExecution",
    "NativeGraphRunner",
    "NativeInputBinding",
    "execute_native_graph",
]
