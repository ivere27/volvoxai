//! Conversions between `volvoxai.v1.DataType` codes, their human-readable
//! names, and the native engine's compact tensor type ids, plus packed
//! byte-size computation for a dtype/shape pair.

use std::ffi::c_int;

use crate::ffi::FfiError;
use crate::pb::DataType;
use crate::{err, INVALID_ARGUMENT, T_F16, T_F32, T_I32, T_I8, T_U8};

// Storage bit width of a volvoxai.v1.DataType value.
pub(crate) fn dtype_bits(dt: i32) -> usize {
    match DataType::try_from(dt).ok() {
        Some(DataType::Bool) => 8,
        Some(DataType::F4) => 4,
        Some(DataType::F6E2m3 | DataType::F6E3m2) => 6,
        Some(
            DataType::U8
            | DataType::I8
            | DataType::F8E5m2
            | DataType::F8E4m3
            | DataType::F8E8m0
            | DataType::F8E4m3fnuz
            | DataType::F8E5m2fnuz,
        ) => 8,
        Some(DataType::I16 | DataType::U16 | DataType::F16 | DataType::Bf16) => 16,
        Some(DataType::I32 | DataType::U32 | DataType::F32) => 32,
        Some(DataType::C64 | DataType::F64 | DataType::I64 | DataType::U64) => 64,
        Some(DataType::Unspecified) | None => 0,
    }
}

pub(crate) fn dtype_bytes(dt: i32) -> usize {
    let bits = dtype_bits(dt);
    if bits > 0 && bits % 8 == 0 {
        bits / 8
    } else {
        0
    }
}

pub(crate) fn dtype_name(dt: i32) -> &'static str {
    DataType::try_from(dt)
        .ok()
        .and_then(|dtype| dtype.as_str_name().strip_prefix("DATA_TYPE_"))
        .unwrap_or("UNKNOWN")
}

pub(crate) fn dtype_from_name(name: &str) -> i32 {
    let proto_name = format!("DATA_TYPE_{name}");
    DataType::from_str_name(&proto_name)
        .unwrap_or(DataType::Unspecified) as i32
}

pub(crate) fn proto_to_native_dtype(dtype: i32) -> Option<c_int> {
    match DataType::try_from(dtype).ok()? {
        DataType::F32 => Some(T_F32),
        DataType::I8 => Some(T_I8),
        DataType::U8 => Some(T_U8),
        DataType::I32 => Some(T_I32),
        DataType::F16 => Some(T_F16),
        _ => None,
    }
}

pub(crate) fn native_to_proto_dtype(dtype: c_int) -> i32 {
    match dtype {
        T_F32 => DataType::F32 as i32,
        T_I8 => DataType::I8 as i32,
        T_U8 => DataType::U8 as i32,
        T_I32 => DataType::I32 as i32,
        T_F16 => DataType::F16 as i32,
        _ => DataType::Unspecified as i32,
    }
}

pub(crate) fn tensor_nbytes(dtype: i32, shape: &[i64]) -> Result<usize, FfiError> {
    let bits = dtype_bits(dtype);
    if bits == 0 {
        return Err(err(
            format!("unsupported dtype {}", dtype_name(dtype)),
            INVALID_ARGUMENT,
        ));
    }
    let mut elements = 1usize;
    for &dim in shape {
        if dim < 0 {
            return Err(err("negative tensor dimension", INVALID_ARGUMENT));
        }
        elements = elements
            .checked_mul(dim as usize)
            .ok_or_else(|| err("tensor is too large", INVALID_ARGUMENT))?;
    }
    let total_bits = elements
        .checked_mul(bits)
        .ok_or_else(|| err("tensor is too large", INVALID_ARGUMENT))?;
    if total_bits % 8 != 0 {
        return Err(err(
            format!(
                "tensor {} has non-byte-aligned packed size",
                dtype_name(dtype)
            ),
            INVALID_ARGUMENT,
        ));
    }
    Ok(total_bits / 8)
}
