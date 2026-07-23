"""Current JavaScript runtime tensor allocation limits.

``volvox-graph/v1`` shapes are consumed by :class:`ts/core/Tensor.Tensor`.
Every runnable Python boundary must therefore prove the same exact integer
arithmetic before it publishes a descriptor that JavaScript cannot construct.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


JS_NUMBER_MAX_SAFE_INTEGER = (1 << 53) - 1

# Runtime graph values use the four canonical execution dtypes. F16 is legal
# only as immutable safetensors storage, but allocation accounting is still
# four bytes per element because GraphLoader expands it to an F32 array before
# constructing the JavaScript weight Tensor. Physical F16 file-span checks
# remain the responsibility of the safetensors parser.
RUNTIME_DTYPE_BYTES = {
    "float32": 4,
    "float16": 4,
    "int32": 4,
    "int8": 1,
    "uint8": 1,
}


def runtime_tensor_allocation(
    shape: Any,
    dtype: Any,
) -> tuple[int, int]:
    """Return ``(elements, bytes)`` or reject a non-constructible Tensor.

    Scalar shape ``[]`` is legal and contains one element. Dimensions, their
    product, and the final byte count must all be positive JSON-safe integers.
    The explicit division guards document the JavaScript multiplication
    boundary without relying on Python's unbounded integer arithmetic.
    """

    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes, bytearray)):
        raise ValueError("shape must be an array")
    bytes_per_element = RUNTIME_DTYPE_BYTES.get(dtype)
    if bytes_per_element is None:
        raise ValueError(f"unsupported runtime tensor dtype {dtype!r}")

    elements = 1
    for index, dimension in enumerate(shape):
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            or dimension > JS_NUMBER_MAX_SAFE_INTEGER
        ):
            raise ValueError(
                f"shape dimension {index} must be a positive JSON-safe integer"
            )
        if elements > JS_NUMBER_MAX_SAFE_INTEGER // dimension:
            raise ValueError("shape element count exceeds JavaScript's safe integer range")
        elements *= dimension

    if elements > JS_NUMBER_MAX_SAFE_INTEGER // bytes_per_element:
        raise ValueError("tensor byte size exceeds JavaScript's safe integer range")
    return elements, elements * bytes_per_element


def safetensors_storage_dtype(dtype: Any) -> str | None:
    """Return a supported current safetensors storage dtype spelling."""

    name = str(dtype)
    return name if name in RUNTIME_DTYPE_BYTES else None
