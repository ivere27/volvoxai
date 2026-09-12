"""Stable named-tensor differential diagnostics for typed RuntimeIR.

C-native executions expose structurally captured node outputs as mappings.
This module only compares those arrays and reports the first divergent stable
tensor name; it does not execute or approximate graph operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from .ir import GraphIR, IRDialect
from .native_execution import NativeExecution


@dataclass(frozen=True)
class TensorDivergence:
    tensor_name: str
    reason: str
    expected_shape: Optional[tuple[int, ...]]
    actual_shape: Optional[tuple[int, ...]]
    expected_dtype: Optional[str]
    actual_dtype: Optional[str]
    max_abs_error: Optional[float]
    max_relative_error: Optional[float]
    mismatch_count: Optional[int]
    first_mismatch_index: Optional[tuple[int, ...]]

    def summary(self) -> str:
        max_error = (
            "n/a" if self.max_abs_error is None else f"{self.max_abs_error:.9g}"
        )
        return (
            f"tensor {self.tensor_name!r}: {self.reason}; "
            f"expected shape={self.expected_shape}, dtype={self.expected_dtype}; "
            f"actual shape={self.actual_shape}, dtype={self.actual_dtype}; "
            f"max_abs_error={max_error}"
        )


@dataclass(frozen=True)
class DifferentialComparison:
    compared_tensors: tuple[str, ...]
    first_divergence: Optional[TensorDivergence] = None

    @property
    def equivalent(self) -> bool:
        return self.first_divergence is None

    def summary(self) -> str:
        if self.first_divergence is None:
            return f"equivalent ({len(self.compared_tensors)} tensors compared)"
        return self.first_divergence.summary()

    def raise_for_divergence(self) -> None:
        if self.first_divergence is not None:
            raise AssertionError(self.first_divergence.summary())


def runtime_capture_order(graph: GraphIR) -> tuple[str, ...]:
    """Return stable node-output order for differential capture comparison."""

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


def compare_executions(
    graph: GraphIR,
    expected: NativeExecution | Mapping[str, Any],
    actual: NativeExecution | Mapping[str, Any],
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    equal_nan: bool = True,
) -> DifferentialComparison:
    """Compare captures in execution order and stop at the first divergence."""

    graph.verify(IRDialect.RUNTIME)
    expected_values = (
        expected.intermediates
        if isinstance(expected, NativeExecution)
        else expected
    )
    actual_values = (
        actual.intermediates if isinstance(actual, NativeExecution) else actual
    )
    return compare_tensor_maps(
        expected_values,
        actual_values,
        tensor_order=runtime_capture_order(graph),
        atol=atol,
        rtol=rtol,
        equal_nan=equal_nan,
    )


def compare_tensor_maps(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    tensor_order: Optional[Iterable[str]] = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    equal_nan: bool = True,
) -> DifferentialComparison:
    """Compare named tensors and describe the first mismatch completely."""

    if atol < 0 or rtol < 0 or not np.isfinite(atol) or not np.isfinite(rtol):
        raise ValueError("differential tolerances must be finite and non-negative")
    order = tuple(tensor_order) if tensor_order is not None else tuple(expected)
    compared: list[str] = []
    for name in order:
        compared.append(name)
        if name not in expected:
            actual_array = np.asarray(actual[name]) if name in actual else None
            return DifferentialComparison(tuple(compared), TensorDivergence(
                tensor_name=name,
                reason="missing expected tensor",
                expected_shape=None,
                actual_shape=(tuple(actual_array.shape)
                              if actual_array is not None else None),
                expected_dtype=None,
                actual_dtype=(str(actual_array.dtype)
                              if actual_array is not None else None),
                max_abs_error=None,
                max_relative_error=None,
                mismatch_count=None,
                first_mismatch_index=None,
            ))
        expected_array = np.asarray(expected[name])
        if name not in actual:
            return DifferentialComparison(tuple(compared), TensorDivergence(
                tensor_name=name,
                reason="missing actual tensor",
                expected_shape=tuple(expected_array.shape),
                actual_shape=None,
                expected_dtype=str(expected_array.dtype),
                actual_dtype=None,
                max_abs_error=None,
                max_relative_error=None,
                mismatch_count=None,
                first_mismatch_index=None,
            ))
        actual_array = np.asarray(actual[name])
        shape_mismatch = expected_array.shape != actual_array.shape
        dtype_mismatch = expected_array.dtype != actual_array.dtype
        if shape_mismatch:
            return DifferentialComparison(tuple(compared), TensorDivergence(
                tensor_name=name,
                reason="shape mismatch",
                expected_shape=tuple(expected_array.shape),
                actual_shape=tuple(actual_array.shape),
                expected_dtype=str(expected_array.dtype),
                actual_dtype=str(actual_array.dtype),
                max_abs_error=None,
                max_relative_error=None,
                mismatch_count=None,
                first_mismatch_index=None,
            ))

        close, max_abs, max_relative = _numeric_comparison(
            expected_array,
            actual_array,
            atol=atol,
            rtol=rtol,
            equal_nan=equal_nan,
        )
        mismatch = np.logical_not(close)
        mismatch_count = int(np.count_nonzero(mismatch))
        if dtype_mismatch or mismatch_count:
            first_index = None
            if mismatch_count:
                flat_index = int(np.flatnonzero(mismatch.reshape(-1))[0])
                first_index = tuple(int(index) for index in np.unravel_index(
                    flat_index, expected_array.shape,
                ))
            return DifferentialComparison(tuple(compared), TensorDivergence(
                tensor_name=name,
                reason="dtype mismatch" if dtype_mismatch else "value mismatch",
                expected_shape=tuple(expected_array.shape),
                actual_shape=tuple(actual_array.shape),
                expected_dtype=str(expected_array.dtype),
                actual_dtype=str(actual_array.dtype),
                max_abs_error=max_abs,
                max_relative_error=max_relative,
                mismatch_count=mismatch_count,
                first_mismatch_index=first_index,
            ))
    return DifferentialComparison(tuple(compared))


def _numeric_comparison(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    atol: float,
    rtol: float,
    equal_nan: bool,
) -> tuple[np.ndarray, float, float]:
    if expected.dtype.kind not in "biuf" or actual.dtype.kind not in "biuf":
        close = np.equal(expected, actual)
        error = 0.0 if np.all(close) else float("inf")
        return close, error, error
    if expected.dtype.kind in "biu" and actual.dtype.kind in "biu":
        close = np.equal(expected, actual)
        absolute = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
        max_abs = float(np.max(absolute)) if absolute.size else 0.0
        denominator = np.abs(expected.astype(np.float64))
        relative = np.divide(
            absolute,
            denominator,
            out=np.where(absolute == 0.0, 0.0, np.inf),
            where=denominator != 0.0,
        )
        return close, max_abs, float(np.max(relative)) if relative.size else 0.0
    expected64 = expected.astype(np.float64)
    actual64 = actual.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        close = np.isclose(
            expected64,
            actual64,
            atol=atol,
            rtol=rtol,
            equal_nan=equal_nan,
        )
        finite = np.isfinite(expected64) & np.isfinite(actual64)
        absolute = np.where(finite, np.abs(expected64 - actual64), 0.0)
        nonfinite_mismatch = (~finite) & (~close)
        if np.any(nonfinite_mismatch):
            max_abs = float("inf")
            max_relative = float("inf")
        else:
            max_abs = float(np.max(absolute)) if absolute.size else 0.0
            denominator = np.abs(expected64)
            relative = np.divide(
                absolute,
                denominator,
                out=np.where(absolute == 0.0, 0.0, np.inf),
                where=denominator != 0.0,
            )
            max_relative = float(np.max(relative)) if relative.size else 0.0
    return close, max_abs, max_relative


__all__ = [
    "DifferentialComparison",
    "TensorDivergence",
    "compare_executions",
    "compare_tensor_maps",
    "runtime_capture_order",
]
