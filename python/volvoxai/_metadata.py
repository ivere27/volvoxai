"""Immutable Python views of the tensor contracts declared by the C engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import volvoxai_lite as pb


NUMPY_DTYPES = {
    pb.DataType.DATA_TYPE_BOOL: np.dtype("bool"),
    pb.DataType.DATA_TYPE_U8: np.dtype("uint8"),
    pb.DataType.DATA_TYPE_I8: np.dtype("int8"),
    pb.DataType.DATA_TYPE_I16: np.dtype("<i2"),
    pb.DataType.DATA_TYPE_U16: np.dtype("<u2"),
    pb.DataType.DATA_TYPE_F16: np.dtype("<f2"),
    pb.DataType.DATA_TYPE_I32: np.dtype("<i4"),
    pb.DataType.DATA_TYPE_U32: np.dtype("<u4"),
    pb.DataType.DATA_TYPE_F32: np.dtype("<f4"),
    pb.DataType.DATA_TYPE_C64: np.dtype("<c8"),
    pb.DataType.DATA_TYPE_F64: np.dtype("<f8"),
    pb.DataType.DATA_TYPE_I64: np.dtype("<i8"),
    pb.DataType.DATA_TYPE_U64: np.dtype("<u8"),
}
_STORAGE_NAMES = {
    pb.DataType.DATA_TYPE_UNSPECIFIED: "unspecified",
    pb.DataType.DATA_TYPE_F4: "float4",
    pb.DataType.DATA_TYPE_F6_E2M3: "float6_e2m3",
    pb.DataType.DATA_TYPE_F6_E3M2: "float6_e3m2",
    pb.DataType.DATA_TYPE_F8_E5M2: "float8_e5m2",
    pb.DataType.DATA_TYPE_F8_E4M3: "float8_e4m3",
    pb.DataType.DATA_TYPE_F8_E8M0: "float8_e8m0",
    pb.DataType.DATA_TYPE_F8_E4M3FNUZ: "float8_e4m3fnuz",
    pb.DataType.DATA_TYPE_F8_E5M2FNUZ: "float8_e5m2fnuz",
    pb.DataType.DATA_TYPE_BF16: "bfloat16",
    **{key: dtype.name for key, dtype in NUMPY_DTYPES.items()},
}


@dataclass(frozen=True)
class Dimension:
    min: int
    max: int
    multiple_of: int


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int | str, ...]
    dtype: str
    constraints: Mapping[str, Dimension]

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "constraints", MappingProxyType(dict(self.constraints)))

    @classmethod
    def _from_proto(cls, spec: pb.TensorSpec) -> TensorSpec:
        shape = []
        constraints = {}
        for axis in spec.dimensions:
            if axis.kind == pb.DimensionKind.DIMENSION_KIND_FIXED:
                shape.append(axis.min)
            elif axis.kind == pb.DimensionKind.DIMENSION_KIND_SYMBOLIC:
                shape.append(axis.symbol)
                constraints[axis.symbol] = Dimension(axis.min, axis.max, axis.multiple_of)
            else:
                raise ValueError(f"Invalid dimension contract for {spec.name!r}.")
        return cls(spec.name, tuple(shape), _STORAGE_NAMES.get(spec.dtype,
                   f"unknown({spec.dtype})"), constraints)


class _SessionMetadata:
    @property
    def inputs(self) -> tuple[TensorSpec, ...]:
        """Ordered, immutable model input contracts."""
        return self._inputs

    @property
    def outputs(self) -> tuple[TensorSpec, ...]:
        """Ordered, immutable model output contracts."""
        return self._outputs
