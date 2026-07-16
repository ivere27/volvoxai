//! VolvoxAI Synurang runtime — a single `libvolvoxai` cdylib that wraps the C
//! inference engine. Rust here is only glue: it implements the generated
//! `VolvoxAiServicePlugin` trait and calls the engine over `extern "C"`. The
//! native engine sources are cc-compiled into this same library by build.rs.
//!
//! Modules:
//!   - `abi`        `extern "C"` declarations for the cc-compiled engine.
//!   - `dtype`      DataType code/name/native-id conversions and packed sizes.
//!   - `checkpoint` atomic staged file/directory commits and training-checkpoint I/O.
//!   - `pb` / `ffi` generated protobuf messages and the Synurang plugin ABI.
//!
//! The rest of this file is the plugin state (`Inner`/`Plugin`) and the
//! `VolvoxAiServicePlugin` trait implementation that dispatches each RPC.
//!
//! The graph engine is a global singleton (native/src/runtime/engine.c), so graph execution
//! is serialized and only one base model is loaded at a time. Immutable adapter
//! staging/updates and atomic activation use a separate registry lock, allowing
//! hot updates while an inference request pins its adapter version.

use std::ffi::{c_char, c_int, c_long, CStr, CString};
use std::os::raw::c_void;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex, RwLock,
};
use std::{
    fs,
    io::{Read, Write},
};

use serde_json::{json, Value};

// prost-generated protobuf messages (package volvoxai.v1).
pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/volvoxai.v1.rs"));
}

// Synurang-generated plugin trait + dispatcher + Synurang_* FFI exports. Loaded
// as a module file (not include!) so its leading inner attributes are legal;
// `make proto_rust` injects `use super::pb::*;` + `use std::boxed::Box;` at the
// top so its unqualified type names resolve here.
#[path = "gen/volvoxai_ffi_plugin.rs"]
mod ffi;

use ffi::{register_volvox_ai_service_plugin, FfiError, PluginStreamSender, VolvoxAiServicePlugin};
use pb::*;
mod abi;
use abi::*;
mod dtype;
use dtype::*;
mod checkpoint;
use checkpoint::*;

const MAX_SEQ: usize = 4096;
const TENSOR_ACCESS_READABLE: u32 = TensorAccessFlag::Readable as u32;
const TENSOR_ACCESS_WRITABLE: u32 = TensorAccessFlag::Writable as u32;
const DATA_TYPE_F16: i32 = DataType::F16 as i32;
const DATA_TYPE_I32: i32 = DataType::I32 as i32;
const DATA_TYPE_F32: i32 = DataType::F32 as i32;
const T_F32: c_int = 0;
const T_I8: c_int = 1;
const T_U8: c_int = 2;
const T_I32: c_int = 3;
const T_F16: c_int = 4;
const NATIVE_ADAPTER_UPDATE_ASSIGN: c_int = 0;
const NATIVE_ADAPTER_UPDATE_ADD: c_int = 1;
const NATIVE_TENSOR_UPDATE_ASSIGN: c_int = 0;
const NATIVE_TENSOR_UPDATE_ADD: c_int = 1;
const NATIVE_TENSOR_UPDATE_SGD: c_int = 2;
const NATIVE_TENSOR_UPDATE_ADAMW: c_int = 3;

// gRPC status codes carried in FfiError.grpc_code.
const INVALID_ARGUMENT: i32 = 3;
const NOT_FOUND: i32 = 5;
const PERMISSION_DENIED: i32 = 7;
const FAILED_PRECONDITION: i32 = 9;
const UNIMPLEMENTED: i32 = 12;
const INTERNAL: i32 = 13;
const UNAVAILABLE: i32 = 14;

fn err(msg: impl Into<String>, code: i32) -> FfiError {
    FfiError::new(msg, code, code)
}

fn adapter_update_mode_to_native(mode: i32) -> Option<c_int> {
    match AdapterUpdateMode::try_from(mode).ok()? {
        AdapterUpdateMode::AdapterUpdateAssign => Some(NATIVE_ADAPTER_UPDATE_ASSIGN),
        AdapterUpdateMode::AdapterUpdateAdd => Some(NATIVE_ADAPTER_UPDATE_ADD),
    }
}

fn tensor_update_mode_to_native(mode: TensorUpdateMode) -> c_int {
    match mode {
        TensorUpdateMode::TensorUpdateAssign => NATIVE_TENSOR_UPDATE_ASSIGN,
        TensorUpdateMode::TensorUpdateAdd => NATIVE_TENSOR_UPDATE_ADD,
        TensorUpdateMode::TensorUpdateSgd => NATIVE_TENSOR_UPDATE_SGD,
        TensorUpdateMode::TensorUpdateAdamw => NATIVE_TENSOR_UPDATE_ADAMW,
    }
}

fn string_entries_from_map(
    entries: std::collections::HashMap<String, String>,
) -> Vec<StringEntry> {
    let mut entries: Vec<_> = entries
        .into_iter()
        .map(|(key, value)| StringEntry { key, value })
        .collect();
    entries.sort_by(|left, right| left.key.cmp(&right.key));
    entries
}

fn string_entries_to_map(
    entries: &[StringEntry],
) -> std::collections::HashMap<String, String> {
    entries
        .iter()
        .map(|entry| (entry.key.clone(), entry.value.clone()))
        .collect()
}

pub(crate) fn string_entry_value<'a>(entries: &'a [StringEntry], key: &str) -> Option<&'a str> {
    entries
        .iter()
        .rev()
        .find(|entry| entry.key == key)
        .map(|entry| entry.value.as_str())
}

fn string_entry_upsert(entries: &mut Vec<StringEntry>, key: impl Into<String>, value: impl Into<String>) {
    let key = key.into();
    entries.retain(|entry| entry.key != key);
    entries.push(StringEntry {
        key,
        value: value.into(),
    });
}

fn tensor_shape_entries_from_map(
    entries: std::collections::HashMap<String, TensorShape>,
) -> Vec<TensorShapeEntry> {
    let mut entries: Vec<_> = entries
        .into_iter()
        .map(|(key, value)| TensorShapeEntry {
            key,
            value: Some(value),
        })
        .collect();
    entries.sort_by(|left, right| left.key.cmp(&right.key));
    entries
}

fn tensor_shape_entries_to_map(
    entries: &[TensorShapeEntry],
) -> std::collections::HashMap<String, TensorShape> {
    entries
        .iter()
        .map(|entry| {
            (
                entry.key.clone(),
                entry.value.clone().unwrap_or_default(),
            )
        })
        .collect()
}

fn value_shape(v: &Value) -> Vec<i64> {
    v.as_array()
        .map(|a| a.iter().filter_map(|d| d.as_i64()).collect())
        .unwrap_or_default()
}

fn metadata_map(v: Option<&Value>) -> std::collections::HashMap<String, String> {
    let mut out = std::collections::HashMap::new();
    if let Some(obj) = v.and_then(Value::as_object) {
        for (k, value) in obj {
            if let Some(s) = value.as_str() {
                out.insert(k.clone(), s.to_string());
            }
        }
    }
    out
}

pub(crate) fn strict_metadata_map(
    value: Option<&Value>,
    context: &str,
) -> Result<std::collections::HashMap<String, String>, FfiError> {
    let Some(value) = value else {
        return Ok(Default::default());
    };
    let object = value.as_object().ok_or_else(|| {
        err(
            format!("{context} metadata must be an object"),
            INVALID_ARGUMENT,
        )
    })?;
    object
        .iter()
        .map(|(key, value)| {
            value
                .as_str()
                .map(|value| (key.clone(), value.to_string()))
                .ok_or_else(|| {
                    err(
                        format!("{context} metadata value {key} must be a string"),
                        INVALID_ARGUMENT,
                    )
                })
        })
        .collect()
}

fn tensor_spec_from_value(v: &Value) -> TensorSpec {
    let dtype = dtype_from_name(v.get("dtype").and_then(Value::as_str).unwrap_or(""));
    TensorSpec {
        name: v
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        shape: value_shape(v.get("shape").unwrap_or(&Value::Null)),
        dtype,
        access_flags: TENSOR_ACCESS_READABLE,
        size_bytes: v.get("size_bytes").and_then(Value::as_i64).unwrap_or(0),
    }
}

fn tensor_shape_from_value(v: &Value) -> TensorShape {
    TensorShape {
        dims: value_shape(v),
    }
}

fn canonical_op_name(op: i32) -> Result<Option<&'static str>, FfiError> {
    let op = OpType::try_from(op).map_err(|_| {
        err(
            format!("unknown OpType numeric value {op}"),
            INVALID_ARGUMENT,
        )
    })?;
    let name = match op {
        OpType::OpUnspecified => return Ok(None),
        OpType::OpMatmul => "MatMul",
        OpType::OpLinear => "Linear",
        OpType::OpGemm => "Gemm",
        OpType::OpQlinear => "QLinear",
        OpType::OpQmatmul => "QMatMul",
        OpType::OpQgemm => "QGemm",
        OpType::OpConv2d => "Conv2D",
        OpType::OpConv1d => "Conv1D",
        OpType::OpConvTranspose2d => "ConvTranspose2D",
        OpType::OpQconv2d => "QConv2D",
        OpType::OpLayerNorm => "LayerNorm",
        OpType::OpRmsNorm => "RMSNorm",
        OpType::OpBatchNorm2d => "BatchNorm2D",
        OpType::OpGroupNorm => "GroupNorm",
        OpType::OpQlayerNorm => "QLayerNorm",
        OpType::OpQgroupNorm => "QGroupNorm",
        OpType::OpEmbedding => "Embedding",
        OpType::OpSdpa => "SDPA",
        OpType::OpCrossSdpa => "CrossSDPA",
        OpType::OpCrossAttention => "CrossAttention",
        OpType::OpMoeRouter => "MoERouter",
        OpType::OpMoeLinear => "MoELinear",
        OpType::OpQembedding => "QEmbedding",
        OpType::OpQmaskedMean => "QMaskedMean",
        OpType::OpQsdpa => "QSDPA",
        OpType::OpMaxPool2d => "MaxPool2D",
        OpType::OpAveragePool => "AveragePool",
        OpType::OpAveragePool2d => "AveragePool2D",
        OpType::OpGlobalAveragePool => "GlobalAveragePool",
        OpType::OpResize => "Resize",
        OpType::OpResizeNearest2d => "ResizeNearest2D",
        OpType::OpUpsampleNearest2d => "UpsampleNearest2D",
        OpType::OpUpsample2x => "Upsample2x",
        OpType::OpInterpLinear1d => "InterpLinear1D",
        OpType::OpInterp1d => "Interp1D",
        OpType::OpRelu => "ReLU",
        OpType::OpLeakyRelu => "LeakyReLU",
        OpType::OpPrelu => "PReLU",
        OpType::OpGelu => "GELU",
        OpType::OpSilu => "SiLU",
        OpType::OpSwish => "Swish",
        OpType::OpSigmoid => "Sigmoid",
        OpType::OpHardSwish => "HardSwish",
        OpType::OpHardSigmoid => "HardSigmoid",
        OpType::OpTanh => "Tanh",
        OpType::OpClip => "Clip",
        OpType::OpQgelu => "QGELU",
        OpType::OpQsilu => "QSiLU",
        OpType::OpSin => "Sin",
        OpType::OpCos => "Cos",
        OpType::OpAdd => "Add",
        OpType::OpMul => "Mul",
        OpType::OpSub => "Sub",
        OpType::OpDiv => "Div",
        OpType::OpQadd => "QAdd",
        OpType::OpSoftmax => "Softmax",
        OpType::OpLogSoftmax => "LogSoftmax",
        OpType::OpReduceSum => "ReduceSum",
        OpType::OpReduceMean => "ReduceMean",
        OpType::OpArgmax => "ArgMax",
        OpType::OpQargmax => "QArgMax",
        OpType::OpTranspose => "Transpose",
        OpType::OpConcat => "Concat",
        OpType::OpConcat2 => "Concat2",
        OpType::OpSplit => "Split",
        OpType::OpSlice => "Slice",
        OpType::OpPad => "Pad",
        OpType::OpExpand => "Expand",
        OpType::OpBroadcast => "Broadcast",
        OpType::OpReshape => "Reshape",
        OpType::OpFlatten => "Flatten",
        OpType::OpSqueeze => "Squeeze",
        OpType::OpUnsqueeze => "Unsqueeze",
        OpType::OpGather => "Gather",
        OpType::OpGatherElements => "GatherElements",
        OpType::OpWhere => "Where",
        OpType::OpMask => "Mask",
        OpType::OpCast => "Cast",
        OpType::OpDequantizeLinear => "DequantizeLinear",
        OpType::OpQuantizeLinear => "QuantizeLinear",
        OpType::OpRequantizeLinear => "RequantizeLinear",
        OpType::OpNonMaxSuppression => "NonMaxSuppression",
        OpType::OpSpatialSoftargmaxY => "SpatialSoftargmaxY",
        OpType::OpProfileX => "ProfileX",
        OpType::OpProfileY => "ProfileY",
        OpType::OpMeanHeight => "MeanHeight",
        OpType::OpDropout => "Dropout",
        OpType::OpIdentity => "Identity",
        OpType::OpShape => "Shape",
        OpType::OpSize => "Size",
        OpType::OpTopk => "TopK",
        OpType::OpRope => "RoPE",
        OpType::OpRotaryEmbedding => "RotaryEmbedding",
        OpType::OpSsmScan => "SSMScan",
        OpType::OpSelectiveScan => "SelectiveScan",
    };
    Ok(Some(name))
}

fn op_type_from_name(name: &str) -> i32 {
    for op in OpType::OpUnspecified as i32..=OpType::OpSelectiveScan as i32 {
        if canonical_op_name(op).ok().flatten() == Some(name) {
            return op;
        }
    }
    OpType::OpUnspecified as i32
}

fn resolve_op_name(
    op: Option<i32>,
    raw_name: Option<&str>,
    context: &str,
    required: bool,
) -> Result<Option<String>, FfiError> {
    let canonical = op.map(canonical_op_name).transpose()?.flatten();
    let raw_name = raw_name.filter(|name| !name.is_empty());
    if let Some(name) = raw_name {
        validate_native_text(name, context, NATIVE_MAX_OP_NAME)?;
    }
    if let (Some(canonical), Some(raw)) = (canonical, raw_name) {
        if canonical != raw {
            return Err(err(
                format!("{context} conflicts: enum resolves to {canonical}, raw op_name is {raw}"),
                INVALID_ARGUMENT,
            ));
        }
    }
    let resolved = canonical.or(raw_name).map(str::to_string);
    if required && resolved.is_none() {
        return Err(err(format!("{context} is required"), INVALID_ARGUMENT));
    }
    Ok(resolved)
}

fn graph_from_inspection(root: &Value) -> GraphInfo {
    let nodes = root
        .get("nodes")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(|node| {
                    let mut inputs = std::collections::HashMap::new();
                    if let Some(obj) = node.get("inputs").and_then(Value::as_object) {
                        for (k, v) in obj {
                            inputs.insert(k.clone(), v.as_str().unwrap_or("").to_string());
                        }
                    }
                    let mut outputs = std::collections::HashMap::new();
                    if let Some(obj) = node.get("outputs").and_then(Value::as_object) {
                        for (k, v) in obj {
                            outputs.insert(k.clone(), v.as_str().unwrap_or("").to_string());
                        }
                    }
                    let mut output_shapes = std::collections::HashMap::new();
                    if let Some(obj) = node.get("output_shapes").and_then(Value::as_object) {
                        for (k, v) in obj {
                            output_shapes.insert(k.clone(), tensor_shape_from_value(v));
                        }
                    }
                    let op_name = node
                        .get("op")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    GraphNode {
                        index: node.get("index").and_then(Value::as_i64).unwrap_or(0) as i32,
                        id: node
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        op: op_type_from_name(&op_name),
                        op_name,
                        inputs: string_entries_from_map(inputs),
                        outputs: string_entries_from_map(outputs),
                        output_shapes: tensor_shape_entries_from_map(output_shapes),
                        params_json: node.get("params").map(Value::to_string),
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    GraphInfo { nodes }
}

fn safetensors_info_from_value(v: &Value) -> SafetensorsInfo {
    let tensors = v
        .get("tensors")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(|t| SafetensorsTensorInfo {
                    name: t
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    shape: value_shape(t.get("shape").unwrap_or(&Value::Null)),
                    dtype: dtype_from_name(t.get("dtype").and_then(Value::as_str).unwrap_or("")),
                    data_start: t.get("data_start").and_then(Value::as_i64).unwrap_or(0),
                    data_end: t.get("data_end").and_then(Value::as_i64).unwrap_or(0),
                    access_flags: t
                        .get("access_flags")
                        .and_then(Value::as_u64)
                        .unwrap_or(TENSOR_ACCESS_READABLE as u64)
                        as u32,
                })
                .collect()
        })
        .unwrap_or_default();
    SafetensorsInfo {
        path: v
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        size_bytes: v.get("size_bytes").and_then(Value::as_i64).unwrap_or(0),
        tensors,
        metadata: string_entries_from_map(metadata_map(v.get("metadata"))),
    }
}

unsafe fn native_inspection(
    include_graph: bool,
    include_tensors: bool,
    include_weight_files: bool,
    include_params: bool,
) -> Result<Value, FfiError> {
    let ptr = volvoxai_engine_inspect_model_json(
        include_graph as c_int,
        include_tensors as c_int,
        include_weight_files as c_int,
        include_params as c_int,
    );
    if ptr.is_null() {
        return Err(err("native model inspection failed", INTERNAL));
    }
    let text = CStr::from_ptr(ptr).to_string_lossy().into_owned();
    free(ptr as *mut c_void);
    serde_json::from_str(&text)
        .map_err(|e| err(format!("bad native inspection JSON: {e}"), INTERNAL))
}

fn validate_safetensors_coverage(
    tensors: &[SafetensorsTensorInfo],
    data_size: u64,
    context: &str,
) -> Result<(), FfiError> {
    let mut ranges: Vec<_> = tensors
        .iter()
        .map(|tensor| (tensor.data_start, tensor.data_end, tensor.name.as_str()))
        .collect();
    ranges.sort_unstable_by_key(|(start, end, _)| (*start, *end));

    let mut cursor = 0u64;
    for (start, end, name) in ranges {
        let start = u64::try_from(start).map_err(|_| {
            err(
                format!("{context} tensor {name} has a negative offset"),
                INVALID_ARGUMENT,
            )
        })?;
        let end = u64::try_from(end).map_err(|_| {
            err(
                format!("{context} tensor {name} has a negative offset"),
                INVALID_ARGUMENT,
            )
        })?;
        if start != cursor {
            let issue = if start < cursor { "overlap" } else { "gap" };
            return Err(err(
                format!("{context} tensor payload has a {issue} before {name}"),
                INVALID_ARGUMENT,
            ));
        }
        cursor = end;
    }
    if cursor != data_size {
        return Err(err(
            format!("{context} tensor payload coverage ends at {cursor}, expected {data_size}"),
            INVALID_ARGUMENT,
        ));
    }
    Ok(())
}

pub(crate) fn read_safetensors_info(path: &str, include_metadata: bool) -> Result<SafetensorsInfo, FfiError> {
    let mut file = fs::File::open(path)
        .map_err(|e| err(format!("read safetensors failed: {e}"), NOT_FOUND))?;
    let file_size = file
        .metadata()
        .map_err(|e| err(format!("stat safetensors failed: {e}"), NOT_FOUND))?
        .len();
    if file_size < 8 {
        return Err(err("invalid safetensors file", INVALID_ARGUMENT));
    }
    let mut length_bytes = [0u8; 8];
    file.read_exact(&mut length_bytes).map_err(|e| {
        err(
            format!("read safetensors length failed: {e}"),
            INVALID_ARGUMENT,
        )
    })?;
    let header_len_u64 = u64::from_le_bytes(length_bytes);
    if header_len_u64 > file_size - 8 {
        return Err(err("invalid safetensors header length", INVALID_ARGUMENT));
    }
    let header_len = usize::try_from(header_len_u64)
        .map_err(|_| err("safetensors header is too large", INVALID_ARGUMENT))?;
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes).map_err(|e| {
        err(
            format!("read safetensors header failed: {e}"),
            INVALID_ARGUMENT,
        )
    })?;
    let header: Value = serde_json::from_slice(&header_bytes)
        .map_err(|e| err(format!("bad safetensors header: {e}"), INVALID_ARGUMENT))?;
    let data_size = file_size - 8 - header_len_u64;
    let mut tensors = Vec::new();
    let object = header
        .as_object()
        .ok_or_else(|| err("safetensors header must be an object", INVALID_ARGUMENT))?;
    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }
        let offsets = value
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| err("bad safetensors data_offsets", INVALID_ARGUMENT))?;
        if offsets.len() != 2 {
            return Err(err(
                "safetensors data_offsets must contain exactly two values",
                INVALID_ARGUMENT,
            ));
        }
        let start = offsets
            .first()
            .and_then(Value::as_u64)
            .ok_or_else(|| err("bad safetensors data start", INVALID_ARGUMENT))?;
        let end = offsets
            .get(1)
            .and_then(Value::as_u64)
            .ok_or_else(|| err("bad safetensors data end", INVALID_ARGUMENT))?;
        if start > end || end > data_size || end > i64::MAX as u64 {
            return Err(err(
                "safetensors data offsets are out of range",
                INVALID_ARGUMENT,
            ));
        }
        let shape = value_shape(value.get("shape").unwrap_or(&Value::Null));
        let dtype = dtype_from_name(value.get("dtype").and_then(Value::as_str).unwrap_or(""));
        if end - start != tensor_nbytes(dtype, &shape)? as u64 {
            return Err(err(
                format!("safetensors tensor {name} byte size does not match shape/dtype"),
                INVALID_ARGUMENT,
            ));
        }
        tensors.push(SafetensorsTensorInfo {
            name: name.clone(),
            shape,
            dtype,
            data_start: start as i64,
            data_end: end as i64,
            access_flags: TENSOR_ACCESS_READABLE,
        });
    }
    validate_safetensors_coverage(&tensors, data_size, "safetensors")?;
    Ok(SafetensorsInfo {
        path: path.to_string(),
        size_bytes: i64::try_from(file_size).unwrap_or(i64::MAX),
        tensors,
        metadata: if include_metadata {
            string_entries_from_map(strict_metadata_map(
                header.get("__metadata__"),
                "safetensors",
            )?)
        } else {
            Default::default()
        },
    })
}

fn read_adapter_checkpoint(
    path: &str,
) -> Result<(SafetensorsInfo, AdapterManifest, Vec<Tensor>), FfiError> {
    let data = fs::read(path)
        .map_err(|e| err(format!("read adapter checkpoint failed: {e}"), NOT_FOUND))?;
    parse_adapter_checkpoint_bytes(path, &data)
}

fn parse_adapter_checkpoint_bytes(
    path: &str,
    data: &[u8],
) -> Result<(SafetensorsInfo, AdapterManifest, Vec<Tensor>), FfiError> {
    if data.len() < 8 {
        return Err(err("invalid adapter safetensors file", INVALID_ARGUMENT));
    }
    let header_len = usize::try_from(u64::from_le_bytes(data[..8].try_into().unwrap()))
        .map_err(|_| err("adapter safetensors header is too large", INVALID_ARGUMENT))?;
    let header_end = 8usize.checked_add(header_len).ok_or_else(|| {
        err(
            "invalid adapter safetensors header length",
            INVALID_ARGUMENT,
        )
    })?;
    if header_end > data.len() {
        return Err(err(
            "invalid adapter safetensors header length",
            INVALID_ARGUMENT,
        ));
    }
    let header: Value = serde_json::from_slice(&data[8..header_end]).map_err(|e| {
        err(
            format!("bad adapter safetensors header: {e}"),
            INVALID_ARGUMENT,
        )
    })?;
    let data_size = data.len() - header_end;
    let mut infos = Vec::new();
    let object = header.as_object().ok_or_else(|| {
        err(
            "adapter safetensors header must be an object",
            INVALID_ARGUMENT,
        )
    })?;
    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }
        let offsets = value
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| err("bad adapter tensor data_offsets", INVALID_ARGUMENT))?;
        if offsets.len() != 2 {
            return Err(err(
                "adapter tensor data_offsets must contain exactly two values",
                INVALID_ARGUMENT,
            ));
        }
        let start = offsets
            .first()
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| err("bad adapter tensor data start", INVALID_ARGUMENT))?;
        let end = offsets
            .get(1)
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| err("bad adapter tensor data end", INVALID_ARGUMENT))?;
        if start > end || end > data_size || end > i64::MAX as usize {
            return Err(err(
                "adapter tensor data offsets are out of range",
                INVALID_ARGUMENT,
            ));
        }
        let shape = value_shape(value.get("shape").unwrap_or(&Value::Null));
        let dtype = dtype_from_name(value.get("dtype").and_then(Value::as_str).unwrap_or(""));
        if end - start != tensor_nbytes(dtype, &shape)? {
            return Err(err(
                format!("adapter tensor {name} byte size does not match shape/dtype"),
                INVALID_ARGUMENT,
            ));
        }
        infos.push(SafetensorsTensorInfo {
            name: name.clone(),
            shape,
            dtype,
            data_start: start as i64,
            data_end: end as i64,
            access_flags: TENSOR_ACCESS_READABLE,
        });
    }
    validate_safetensors_coverage(&infos, data_size as u64, "adapter safetensors")?;
    let info = SafetensorsInfo {
        path: path.to_string(),
        size_bytes: i64::try_from(data.len()).unwrap_or(i64::MAX),
        tensors: infos,
        metadata: string_entries_from_map(strict_metadata_map(
            header.get("__metadata__"),
            "adapter safetensors",
        )?),
    };
    let encoded = string_entry_value(&info.metadata, ADAPTER_MANIFEST_METADATA_KEY)
        .ok_or_else(|| {
            err(
                format!("adapter checkpoint is missing {ADAPTER_MANIFEST_METADATA_KEY} metadata"),
                INVALID_ARGUMENT,
            )
        })?;
    let mut manifest = manifest_from_json(encoded, &info)?;
    apply_adapter_scale_default(&mut manifest);
    validate_adapter_manifest(&manifest, None)?;

    let info_by_name: std::collections::HashMap<&str, &SafetensorsTensorInfo> = info
        .tensors
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect();
    let referenced_names: std::collections::HashSet<&str> = manifest
        .targets
        .iter()
        .flat_map(|target| &target.tensors)
        .map(|binding| binding.tensor_name.as_str())
        .collect();
    if referenced_names.len() != info.tensors.len() {
        return Err(err(
            "adapter checkpoint contains tensors not bound by its LoRA manifest",
            INVALID_ARGUMENT,
        ));
    }
    let mut seen = std::collections::HashSet::new();
    let mut tensors = Vec::new();
    for binding in manifest.targets.iter().flat_map(|target| &target.tensors) {
        if !seen.insert(binding.tensor_name.as_str()) {
            continue;
        }
        let tensor = info_by_name
            .get(binding.tensor_name.as_str())
            .ok_or_else(|| {
                err(
                    format!("adapter tensor {} is missing", binding.tensor_name),
                    INVALID_ARGUMENT,
                )
            })?;
        let start = header_end
            .checked_add(tensor.data_start as usize)
            .ok_or_else(|| err("adapter tensor offset overflow", INVALID_ARGUMENT))?;
        let end = header_end
            .checked_add(tensor.data_end as usize)
            .ok_or_else(|| err("adapter tensor offset overflow", INVALID_ARGUMENT))?;
        tensors.push(Tensor {
            name: tensor.name.clone(),
            shape: tensor.shape.clone(),
            dtype: tensor.dtype,
            data: data[start..end].to_vec(),
            quant: None,
            access_flags: TENSOR_ACCESS_READABLE,
            initializer: None,
        });
    }
    validate_adapter_manifest(&manifest, Some(&tensors))?;
    Ok((info, manifest, tensors))
}

fn read_extra_adapter_checkpoint(
    path: &str,
) -> Result<Option<(SafetensorsInfo, AdapterManifest, Vec<Tensor>)>, FfiError> {
    let mut file = fs::File::open(path)
        .map_err(|e| err(format!("read extra weights failed: {e}"), NOT_FOUND))?;
    let file_size = file
        .metadata()
        .map_err(|e| err(format!("stat extra weights failed: {e}"), NOT_FOUND))?
        .len();
    if file_size < 8 {
        return Err(err("invalid extra safetensors file", INVALID_ARGUMENT));
    }
    let mut length_bytes = [0u8; 8];
    file.read_exact(&mut length_bytes).map_err(|e| {
        err(
            format!("read extra weights length failed: {e}"),
            INVALID_ARGUMENT,
        )
    })?;
    let header_len_u64 = u64::from_le_bytes(length_bytes);
    if header_len_u64 > file_size - 8 {
        return Err(err(
            "invalid extra safetensors header length",
            INVALID_ARGUMENT,
        ));
    }
    let header_len = usize::try_from(header_len_u64)
        .map_err(|_| err("extra safetensors header is too large", INVALID_ARGUMENT))?;
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes).map_err(|e| {
        err(
            format!("read extra weights header failed: {e}"),
            INVALID_ARGUMENT,
        )
    })?;
    let header: Value = serde_json::from_slice(&header_bytes).map_err(|e| {
        err(
            format!("bad extra safetensors header: {e}"),
            INVALID_ARGUMENT,
        )
    })?;
    if !header.is_object() {
        return Err(err(
            "extra safetensors header must be an object",
            INVALID_ARGUMENT,
        ));
    }
    let metadata = strict_metadata_map(header.get("__metadata__"), "extra safetensors")?;
    if !metadata.contains_key(ADAPTER_MANIFEST_METADATA_KEY) {
        // The native model loader owns base-shard tensor parsing. This open
        // still proves the path/header is available before replacing a model.
        return Ok(None);
    }
    let capacity = usize::try_from(file_size)
        .map_err(|_| err("adapter checkpoint is too large", INVALID_ARGUMENT))?;
    let mut data = Vec::with_capacity(capacity);
    data.extend_from_slice(&length_bytes);
    data.extend_from_slice(&header_bytes);
    file.read_to_end(&mut data).map_err(|e| {
        err(
            format!("read adapter checkpoint failed: {e}"),
            INVALID_ARGUMENT,
        )
    })?;
    if data.len() != capacity {
        return Err(err(
            "adapter checkpoint changed while being read",
            FAILED_PRECONDITION,
        ));
    }
    parse_adapter_checkpoint_bytes(path, &data).map(Some)
}

fn write_safetensors(
    path: &str,
    tensors: &[Tensor],
    metadata: &std::collections::HashMap<String, String>,
    overwrite: bool,
) -> Result<SafetensorsInfo, FfiError> {
    let _persistence = PERSISTENCE_COMMIT.lock().unwrap();
    if !overwrite && fs::metadata(path).is_ok() {
        return Err(err(
            format!("refusing to overwrite {path}"),
            INVALID_ARGUMENT,
        ));
    }
    let mut header = serde_json::Map::new();
    if !metadata.is_empty() {
        header.insert("__metadata__".to_string(), json!(metadata));
    }
    let mut offset = 0usize;
    let mut names = std::collections::HashSet::new();
    for tensor in tensors {
        if tensor.initializer.is_some() {
            return Err(err(
                format!("tensor {} initializer was not materialized", tensor.name),
                INVALID_ARGUMENT,
            ));
        }
        if tensor.name.is_empty() || !names.insert(tensor.name.as_str()) {
            return Err(err(
                "safetensors tensor names must be non-empty and unique",
                INVALID_ARGUMENT,
            ));
        }
        let nbytes = tensor_nbytes(tensor.dtype, &tensor.shape)?;
        if tensor.data.len() != nbytes {
            return Err(err(
                format!(
                    "tensor {} byte-size mismatch: got {}, want {}",
                    tensor.name,
                    tensor.data.len(),
                    nbytes
                ),
                INVALID_ARGUMENT,
            ));
        }
        let end = offset
            .checked_add(nbytes)
            .ok_or_else(|| err("safetensors payload is too large", INVALID_ARGUMENT))?;
        header.insert(
            tensor.name.clone(),
            json!({
                "dtype": dtype_name(tensor.dtype),
                "shape": tensor.shape,
                "data_offsets": [offset, end],
            }),
        );
        offset = end;
    }
    let mut header_bytes = serde_json::to_vec(&Value::Object(header)).map_err(|e| {
        err(
            format!("serialize safetensors header failed: {e}"),
            INTERNAL,
        )
    })?;
    let aligned = (header_bytes.len() + 7) & !7;
    header_bytes.resize(aligned, b' ');
    let mut out = Vec::with_capacity(8 + aligned + offset);
    out.extend_from_slice(&(aligned as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    for tensor in tensors {
        out.extend_from_slice(&tensor.data);
    }
    let target = PathBuf::from(path);
    let stage = unique_sibling_path(&target, "stage")?;
    let write_result = (|| -> Result<(), FfiError> {
        let mut file = fs::File::create(&stage)
            .map_err(|e| err(format!("create safetensors stage failed: {e}"), INTERNAL))?;
        file.write_all(&out)
            .map_err(|e| err(format!("write safetensors stage failed: {e}"), INTERNAL))?;
        Ok(())
    })();
    if let Err(error) = write_result {
        let _ = fs::remove_file(&stage);
        return Err(error);
    }
    commit_staged_files(vec![(stage, target)])?;
    read_safetensors_info(path, true)
}

static PERSIST_SEQUENCE: AtomicU64 = AtomicU64::new(1);
static PERSISTENCE_COMMIT: Mutex<()> = Mutex::new(());

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x03ff) as u32;
    let out = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut frac = frac;
            let mut exp = -14i32;
            while (frac & 0x0400) == 0 {
                frac <<= 1;
                exp -= 1;
            }
            frac &= 0x03ff;
            sign | (((exp + 127) as u32) << 23) | (frac << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        sign | (((exp - 15 + 127) as u32) << 23) | (frac << 13)
    };
    f32::from_bits(out)
}

fn read_le<const N: usize>(data: &[u8], i: usize) -> [u8; N] {
    let mut out = [0u8; N];
    out.copy_from_slice(&data[i * N..(i + 1) * N]);
    out
}

unsafe fn graph_input_destination(
    name: &str,
) -> Result<(CString, usize, Vec<i64>, c_int, usize), FfiError> {
    if name.is_empty() {
        return Err(err("graph input name is required", INVALID_ARGUMENT));
    }
    let cname = CString::new(name).map_err(|_| err("bad input name", INVALID_ARGUMENT))?;
    if volvoxai_engine_is_graph_input(cname.as_ptr()) != 1 {
        return Err(err(
            format!("tensor {name} is not a declared graph input"),
            INVALID_ARGUMENT,
        ));
    }
    let mut numel: c_long = 0;
    let mut native_shape = [0 as c_int; 8];
    let mut native_ndim = 0;
    let mut native_dtype = 0;
    let mut element_size = 0usize;
    if volvoxai_engine_tensor_info_ex(
        cname.as_ptr(),
        &mut numel,
        native_shape.as_mut_ptr(),
        &mut native_ndim,
        &mut native_dtype,
        &mut element_size,
    ) != 0
        || numel < 0
        || native_ndim < 0
        || native_ndim as usize > native_shape.len()
    {
        return Err(err(
            format!("graph input {name} has invalid native metadata"),
            INTERNAL,
        ));
    }
    let shape = native_shape[..native_ndim as usize]
        .iter()
        .map(|dim| *dim as i64)
        .collect();
    Ok((cname, numel as usize, shape, native_dtype, element_size))
}

unsafe fn copy_tensor_to_input(t: &Tensor) -> Result<(), FfiError> {
    let (cname, numel, expected_shape, native_dtype, native_width) =
        graph_input_destination(&t.name)?;
    if t.shape != expected_shape {
        return Err(err(
            format!(
                "input {} shape {:?} does not match model shape {:?}",
                t.name, t.shape, expected_shape
            ),
            INVALID_ARGUMENT,
        ));
    }
    let width = dtype_bytes(t.dtype);
    if width == 0 {
        return Err(err(
            format!(
                "input {} has unsupported dtype {}",
                t.name,
                dtype_name(t.dtype)
            ),
            INVALID_ARGUMENT,
        ));
    }
    let want = numel
        .checked_mul(width)
        .ok_or_else(|| err(format!("input {} is too large", t.name), INVALID_ARGUMENT))?;
    if t.data.len() != want {
        return Err(err(
            format!(
                "input {} byte-size mismatch: got {}, want {} for {} elements of {}",
                t.name,
                t.data.len(),
                want,
                numel,
                dtype_name(t.dtype)
            ),
            INVALID_ARGUMENT,
        ));
    }

    let requested_native_dtype = proto_to_native_dtype(t.dtype);
    if requested_native_dtype == Some(native_dtype) && width == native_width {
        if volvoxai_engine_set_input_raw(
            cname.as_ptr(),
            native_dtype,
            t.data.as_ptr() as *const c_void,
            t.data.len(),
        ) != 0
        {
            return Err(err(
                format!("copy exact input {} failed", t.name),
                FAILED_PRECONDITION,
            ));
        }
        return Ok(());
    }
    if native_dtype != T_F32 || native_width != std::mem::size_of::<f32>() {
        return Err(err(
            format!(
                "input {} dtype {} cannot be converted into native dtype {}",
                t.name,
                dtype_name(t.dtype),
                native_to_proto_dtype(native_dtype)
            ),
            INVALID_ARGUMENT,
        ));
    }

    let mut destination_numel = 0;
    let dst = volvoxai_engine_input_ptr(cname.as_ptr(), &mut destination_numel);
    if dst.is_null() || destination_numel < 0 || destination_numel as usize != numel {
        return Err(err(
            format!("graph input {} became unavailable", t.name),
            FAILED_PRECONDITION,
        ));
    }

    let dst = std::slice::from_raw_parts_mut(dst, numel);
    match DataType::try_from(t.dtype) {
        Ok(DataType::F32) => {
            std::ptr::copy_nonoverlapping(t.data.as_ptr(), dst.as_mut_ptr() as *mut u8, want)
        }
        Ok(DataType::F16) => {
            for (i, out) in dst.iter_mut().enumerate() {
                *out = f16_to_f32(u16::from_le_bytes(read_le::<2>(&t.data, i)));
            }
        }
        Ok(DataType::Bf16) => {
            for (i, out) in dst.iter_mut().enumerate() {
                *out = f32::from_bits((u16::from_le_bytes(read_le::<2>(&t.data, i)) as u32) << 16);
            }
        }
        Ok(DataType::F64) => {
            for (i, out) in dst.iter_mut().enumerate() {
                *out = f64::from_le_bytes(read_le::<8>(&t.data, i)) as f32;
            }
        }
        Ok(DataType::I8) => {
            for (out, &x) in dst.iter_mut().zip(&t.data) {
                *out = (x as i8) as f32;
            }
        }
        Ok(DataType::I16) => {
            for (i, out) in dst.iter_mut().enumerate() {
                *out = i16::from_le_bytes(read_le::<2>(&t.data, i)) as f32;
            }
        }
        Ok(DataType::I32) => {
            for (i, out) in dst.iter_mut().enumerate() {
                *out = i32::from_le_bytes(read_le::<4>(&t.data, i)) as f32;
            }
        }
        Ok(DataType::I64) => {
            for (i, out) in dst.iter_mut().enumerate() {
                *out = i64::from_le_bytes(read_le::<8>(&t.data, i)) as f32;
            }
        }
        Ok(DataType::U8 | DataType::Bool) => {
            for (out, &x) in dst.iter_mut().zip(&t.data) {
                *out = x as f32;
            }
        }
        Ok(DataType::U16) => {
            for (i, out) in dst.iter_mut().enumerate() {
                *out = u16::from_le_bytes(read_le::<2>(&t.data, i)) as f32;
            }
        }
        Ok(DataType::U32) => {
            for (i, out) in dst.iter_mut().enumerate() {
                *out = u32::from_le_bytes(read_le::<4>(&t.data, i)) as f32;
            }
        }
        Ok(DataType::U64) => {
            for (i, out) in dst.iter_mut().enumerate() {
                *out = u64::from_le_bytes(read_le::<8>(&t.data, i)) as f32;
            }
        }
        _ => {
            return Err(err(
                format!(
                    "input {} has unsupported dtype {}",
                    t.name,
                    dtype_name(t.dtype)
                ),
                INVALID_ARGUMENT,
            ))
        }
    }
    Ok(())
}

fn argmax(v: &[f32]) -> i32 {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best as i32
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum RuntimeBackend {
    #[default]
    Cpu,
    Vulkan,
    OpenGl,
    Metal,
    Nnapi,
}

impl RuntimeBackend {
    fn proto(self) -> i32 {
        match self {
            Self::Cpu => Backend::Cpu as i32,
            Self::Vulkan => Backend::Vulkan as i32,
            Self::OpenGl => Backend::Opengl as i32,
            Self::Metal => Backend::Metal as i32,
            Self::Nnapi => Backend::Nnapi as i32,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Vulkan => "Vulkan",
            Self::OpenGl => "OpenGL",
            Self::Metal => "Metal",
            Self::Nnapi => "NNAPI",
        }
    }

    fn native(self) -> c_int {
        match self {
            Self::Cpu => 0,
            Self::Vulkan => 1,
            Self::OpenGl => 2,
            Self::Metal => 3,
            Self::Nnapi => 4,
        }
    }

    fn from_native(value: c_int) -> Result<Self, FfiError> {
        match value {
            0 => Ok(Self::Cpu),
            1 => Ok(Self::Vulkan),
            2 => Ok(Self::OpenGl),
            3 => Ok(Self::Metal),
            4 => Ok(Self::Nnapi),
            _ => Err(err(
                format!("native engine reported unknown backend value {value}"),
                INTERNAL,
            )),
        }
    }
}

fn parse_backend(value: i32) -> Result<RuntimeBackend, FfiError> {
    match Backend::try_from(value) {
        Ok(Backend::Unspecified | Backend::Cpu) => Ok(RuntimeBackend::Cpu),
        Ok(Backend::Vulkan) => Ok(RuntimeBackend::Vulkan),
        Ok(Backend::Opengl) => Ok(RuntimeBackend::OpenGl),
        Ok(Backend::Metal) => Ok(RuntimeBackend::Metal),
        Ok(Backend::Nnapi) => Ok(RuntimeBackend::Nnapi),
        Err(_) => Err(err(
            format!("unknown backend enum value {value}"),
            INVALID_ARGUMENT,
        )),
    }
}

fn model_backend(exec: &Option<ExecOptions>) -> Result<RuntimeBackend, FfiError> {
    exec.as_ref()
        .and_then(|options| options.backend)
        .map(parse_backend)
        .transpose()
        .map(|backend| backend.unwrap_or(RuntimeBackend::Cpu))
}

fn model_engine_options(
    backend: RuntimeBackend,
    exec: &Option<ExecOptions>,
) -> Result<NativeEngineOptions, FfiError> {
    let cpu_threads = exec
        .as_ref()
        .and_then(|options| options.num_threads)
        .unwrap_or(0);
    if cpu_threads < 0 {
        return Err(err("num_threads must be non-negative", INVALID_ARGUMENT));
    }
    Ok(NativeEngineOptions {
        backend: backend.native(),
        debug: c_int::from(
            exec.as_ref()
                .and_then(|options| options.debug)
                .unwrap_or(false),
        ),
        cpu_threads,
    })
}

fn current_engine_options() -> Result<NativeEngineOptions, FfiError> {
    let mut options = NativeEngineOptions::default();
    if unsafe { volvoxai_engine_get_options(&mut options) } != 0 {
        return Err(err("failed to read native engine options", INTERNAL));
    }
    Ok(options)
}

fn validate_engine_backend_request(
    backend: RuntimeBackend,
    exec: &Option<ExecOptions>,
) -> Result<(), FfiError> {
    #[cfg(not(target_os = "macos"))]
    if backend == RuntimeBackend::Metal {
        return Err(err(
            "Metal is only compiled into macOS runtime builds",
            UNIMPLEMENTED,
        ));
    }
    let _ = model_engine_options(backend, exec)?;
    Ok(())
}

fn configure_engine_backend(
    backend: RuntimeBackend,
    exec: &Option<ExecOptions>,
) -> Result<(), FfiError> {
    validate_engine_backend_request(backend, exec)?;
    let options = model_engine_options(backend, exec)?;
    if unsafe { volvoxai_engine_configure(&options) } != 0 {
        return Err(err(
            format!(
                "{} was explicitly requested but could not initialize",
                backend.name()
            ),
            UNAVAILABLE,
        ));
    }

    let capability_error = unsafe {
        match backend {
            RuntimeBackend::Vulkan if vk_training_available() != 1 => Some(
                "Vulkan was explicitly requested but could not initialize a compute-capable device",
            ),
            RuntimeBackend::OpenGl => {
                let available = opengl_training_available() == 1;
                opengl_release_current();
                (!available).then_some(
                    "OpenGL was explicitly requested but no OpenGL 4.3+/OpenGL ES 3.1+ compute context is available",
                )
            }
            #[cfg(target_os = "macos")]
            RuntimeBackend::Metal if metal_training_available() != 1 => Some(
                "Metal was explicitly requested but could not initialize a compute-capable device",
            ),
            _ => None,
        }
    };
    if let Some(message) = capability_error {
        unsafe { volvoxai_engine_shutdown() };
        return Err(err(message, UNAVAILABLE));
    }
    Ok(())
}

#[derive(Debug)]
struct BackendCallGuard {
    open_gl_current: bool,
    previous_debug: c_int,
}

impl BackendCallGuard {
    fn shutdown_engine(mut self) {
        unsafe { volvoxai_engine_shutdown() };
        // Shutdown resets public options to CPU defaults. Keep Drop from
        // restoring the model-scoped debug setting after that reset.
        self.previous_debug = 0;
    }
}

impl Drop for BackendCallGuard {
    fn drop(&mut self) {
        if self.open_gl_current {
            unsafe { opengl_release_current() };
        }
        unsafe {
            let _ = volvoxai_engine_set_debug(self.previous_debug);
        }
    }
}

struct BackendLoadGuard {
    armed: bool,
    backend_call: Option<BackendCallGuard>,
}

impl BackendLoadGuard {
    fn new(backend_call: BackendCallGuard) -> Self {
        Self {
            armed: true,
            backend_call: Some(backend_call),
        }
    }

    fn commit(mut self) {
        self.armed = false;
    }
}

impl Drop for BackendLoadGuard {
    fn drop(&mut self) {
        if self.armed {
            unsafe { volvoxai_engine_shutdown() };
            if let Some(backend_call) = self.backend_call.as_mut() {
                backend_call.previous_debug = 0;
            }
        }
    }
}

struct NativeTrainingBackendGuard {
    required: c_int,
}

impl NativeTrainingBackendGuard {
    fn new(backend: RuntimeBackend) -> Result<Self, FfiError> {
        let required = match backend {
            RuntimeBackend::Cpu => 0,
            RuntimeBackend::Vulkan => 1,
            RuntimeBackend::OpenGl => 2,
            RuntimeBackend::Metal => 3,
            RuntimeBackend::Nnapi => {
                return Err(err("NNAPI has no native training backend", UNIMPLEMENTED))
            }
        };
        if unsafe { volvoxai_engine_require_training_backend(required) } != 0 {
            return Err(err("failed to configure the native training backend policy", INTERNAL));
        }
        Ok(Self { required })
    }
}

impl Drop for NativeTrainingBackendGuard {
    fn drop(&mut self) {
        let _ = self.required;
        unsafe { let _ = volvoxai_engine_require_training_backend(0); }
    }
}

fn begin_backend_call(
    backend: RuntimeBackend,
    exec: &Option<ExecOptions>,
    training: bool,
) -> Result<BackendCallGuard, FfiError> {
    if let Some(requested) = exec.as_ref().and_then(|options| options.backend) {
        let requested = if requested == Backend::Unspecified as i32 {
            backend
        } else {
            parse_backend(requested)?
        };
        if requested != backend {
            return Err(err(
                format!(
                    "this model uses {}; per-call backend {} is not allowed",
                    backend.name(), requested.name()
                ),
                FAILED_PRECONDITION,
            ));
        }
    }
    let options = current_engine_options()?;
    if RuntimeBackend::from_native(options.backend)? != backend {
        let actual_name = unsafe {
            let name = volvoxai_engine_backend_name();
            if name.is_null() {
                "unknown".to_string()
            } else {
                CStr::from_ptr(name).to_string_lossy().into_owned()
            }
        };
        return Err(err(
            format!(
                "model expects {} but native engine is configured for {actual_name}",
                backend.name()
            ),
            INTERNAL,
        ));
    }
    if let Some(threads) = exec.as_ref().and_then(|options| options.num_threads) {
        if threads < 0 {
            return Err(err("num_threads must be non-negative", INVALID_ARGUMENT));
        }
        if threads != options.cpu_threads {
            return Err(err(
                format!(
                    "this model uses {} CPU threads; per-call num_threads {threads} is not allowed",
                    options.cpu_threads
                ),
                FAILED_PRECONDITION,
            ));
        }
    }
    unsafe {
        let previous_debug = volvoxai_engine_debug();
        if let Some(debug) = exec.as_ref().and_then(|options| options.debug) {
            if volvoxai_engine_set_debug(c_int::from(debug)) != 0 {
                return Err(err("failed to configure native debug mode", INTERNAL));
            }
        }
        let open_gl_current = backend == RuntimeBackend::OpenGl;
        if open_gl_current && opengl_make_current() != 0 {
            let _ = volvoxai_engine_set_debug(previous_debug);
            return Err(err(
                "OpenGL context could not be made current on this service worker thread",
                UNAVAILABLE,
            ));
        }
        let guard = BackendCallGuard {
            open_gl_current,
            previous_debug,
        };
        if training {
            match backend {
                RuntimeBackend::Vulkan if vk_training_available() != 1 => {
                    return Err(err("Vulkan training backend is not initialized", UNAVAILABLE))
                }
                RuntimeBackend::OpenGl if opengl_training_available() != 1 => {
                    return Err(err("OpenGL training backend is not initialized", UNAVAILABLE));
                }
                RuntimeBackend::Metal => {
                    #[cfg(target_os = "macos")]
                    if metal_training_available() != 1 {
                        return Err(err("Metal training backend is not initialized", UNAVAILABLE));
                    }
                    #[cfg(not(target_os = "macos"))]
                    {
                        return Err(err(
                            "Metal training is only available in macOS runtime builds",
                            UNIMPLEMENTED,
                        ));
                    }
                }
                RuntimeBackend::Nnapi => {
                    return Err(err(
                        format!("{} training is unavailable in this runtime", backend.name()),
                        UNIMPLEMENTED,
                    ));
                }
                _ => {}
            }
        }
        Ok(guard)
    }
}

struct ExecutionRowGuard {
    previous: c_int,
}

impl ExecutionRowGuard {
    fn new(row: c_int) -> Result<Self, FfiError> {
        if row < -1 {
            return Err(err("execution row must be -1 or non-negative", INVALID_ARGUMENT));
        }
        let previous = unsafe { volvoxai_engine_execution_row() };
        if unsafe { volvoxai_engine_set_execution_row(row) } != 0 {
            return Err(err(
                format!("execution row {row} is outside the graph output domain"),
                INVALID_ARGUMENT,
            ));
        }
        Ok(Self { previous })
    }
}

impl Drop for ExecutionRowGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = volvoxai_engine_set_execution_row(self.previous);
        }
    }
}

unsafe fn native_graph_interface_names(inputs: bool) -> Result<Vec<String>, FfiError> {
    let count = if inputs {
        volvoxai_engine_graph_input_count()
    } else {
        volvoxai_engine_graph_output_count()
    };
    if count < 0 {
        return Err(err("native graph interface count is invalid", INTERNAL));
    }
    let mut names = Vec::with_capacity(count as usize);
    for index in 0..count {
        let name = if inputs {
            volvoxai_engine_graph_input_name(index)
        } else {
            volvoxai_engine_graph_output_name(index)
        };
        if name.is_null() {
            return Err(err(
                format!("native graph interface name {index} is unavailable"),
                INTERNAL,
            ));
        }
        names.push(
            CStr::from_ptr(name)
                .to_str()
                .map_err(|_| err("native graph interface name is not UTF-8", INTERNAL))?
                .to_string(),
        );
    }
    Ok(names)
}

unsafe fn first_graph_input_name() -> Result<String, FfiError> {
    native_graph_interface_names(true)?
        .into_iter()
        .next()
        .ok_or_else(|| err("model has no declared graph input", FAILED_PRECONDITION))
}

unsafe fn first_graph_output_name() -> Result<String, FfiError> {
    native_graph_interface_names(false)?
        .into_iter()
        .next()
        .ok_or_else(|| err("model has no declared graph output", FAILED_PRECONDITION))
}

// Copy a materialized F32 graph tensor into a proto Tensor.
unsafe fn read_tensor(name: &str) -> Result<Tensor, FfiError> {
    let cname = CString::new(name).map_err(|_| err("bad tensor name", INVALID_ARGUMENT))?;
    let mut numel: c_long = 0;
    let mut shape = [0 as c_int; 8];
    let mut ndim: c_int = 0;
    let mut dtype = 0;
    let mut elem_size = 0usize;
    if volvoxai_engine_tensor_info_ex(
        cname.as_ptr(),
        &mut numel,
        shape.as_mut_ptr(),
        &mut ndim,
        &mut dtype,
        &mut elem_size,
    ) != 0
    {
        return Err(err(format!("no such tensor: {name}"), NOT_FOUND));
    }
    if numel < 0 || ndim < 0 {
        return Err(err(
            format!("tensor has invalid metadata: {name}"),
            INTERNAL,
        ));
    }
    let nbytes = (numel as usize)
        .checked_mul(elem_size)
        .ok_or_else(|| err(format!("tensor is too large: {name}"), INTERNAL))?;
    let mut data = vec![0u8; nbytes];
    if volvoxai_engine_copy_tensor_raw(cname.as_ptr(), data.as_mut_ptr() as *mut c_void, nbytes)
        != 0
    {
        return Err(err(format!("copy tensor failed: {name}"), INTERNAL));
    }
    Ok(Tensor {
        name: name.to_string(),
        shape: shape[..ndim as usize].iter().map(|&d| d as i64).collect(),
        dtype: native_to_proto_dtype(dtype),
        data,
        quant: None,
        access_flags: TENSOR_ACCESS_READABLE,
        initializer: None,
    })
}

unsafe fn read_tensor_row_f32(name: &str, row: c_int) -> Result<Tensor, FfiError> {
    let cname = CString::new(name).map_err(|_| err("bad tensor name", INVALID_ARGUMENT))?;
    let mut count: c_int = 0;
    let values = volvoxai_engine_tensor_row_f32(cname.as_ptr(), row, &mut count);
    if values.is_null() || count <= 0 {
        return Err(err(
            format!("tensor {name} has no readable F32 row {row}"),
            FAILED_PRECONDITION,
        ));
    }
    let data = std::slice::from_raw_parts(values as *const u8, count as usize * 4).to_vec();
    Ok(Tensor {
        name: name.to_string(),
        shape: vec![count as i64],
        dtype: DATA_TYPE_F32,
        data,
        quant: None,
        access_flags: TENSOR_ACCESS_READABLE,
        initializer: None,
    })
}

unsafe fn read_tensor_for_execution_row(
    name: &str,
    row: Option<c_int>,
) -> Result<Tensor, FfiError> {
    match row {
        Some(row) => read_tensor_row_f32(name, row),
        None => read_tensor(name),
    }
}

unsafe fn read_tensor_f32_values(name: &str) -> Result<Vec<f32>, FfiError> {
    let cname = CString::new(name).map_err(|_| err("bad tensor name", INVALID_ARGUMENT))?;
    let mut numel: c_long = 0;
    if volvoxai_engine_tensor_info_ex(
        cname.as_ptr(),
        &mut numel,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    ) != 0
    {
        return Err(err(format!("no such tensor: {name}"), NOT_FOUND));
    }
    if numel < 0 {
        return Err(err(format!("tensor has invalid size: {name}"), INTERNAL));
    }
    let mut values = vec![0.0f32; numel as usize];
    if volvoxai_engine_copy_tensor_f32(cname.as_ptr(), values.as_mut_ptr(), numel) != 0 {
        return Err(err(format!("copy tensor as F32 failed: {name}"), INTERNAL));
    }
    Ok(values)
}

unsafe fn tensor_spec_native(name: &str, access_flags: u32) -> Result<TensorSpec, FfiError> {
    let cname = CString::new(name).map_err(|_| err("bad tensor name", INVALID_ARGUMENT))?;
    let mut numel: c_long = 0;
    let mut shape = [0 as c_int; 8];
    let mut ndim: c_int = 0;
    let mut native_dtype: c_int = 0;
    let mut elem_size: usize = 0;
    if volvoxai_engine_tensor_info_ex(
        cname.as_ptr(),
        &mut numel,
        shape.as_mut_ptr(),
        &mut ndim,
        &mut native_dtype,
        &mut elem_size,
    ) != 0
    {
        return Err(err(format!("no such tensor: {name}"), NOT_FOUND));
    }
    if numel < 0 || ndim < 0 {
        return Err(err(format!("tensor {name} has invalid metadata"), INTERNAL));
    }
    let size_bytes = (numel as usize)
        .checked_mul(elem_size)
        .ok_or_else(|| err(format!("tensor {name} is too large"), INTERNAL))?;
    Ok(TensorSpec {
        name: name.to_string(),
        shape: shape[..ndim as usize].iter().map(|&d| d as i64).collect(),
        dtype: native_to_proto_dtype(native_dtype),
        access_flags,
        size_bytes: size_bytes as i64,
    })
}

const NATIVE_MAX_TENSORS: usize = 1024;
const NATIVE_MAX_NODES: usize = 1024;
const NATIVE_MAX_NODE_REFS: usize = 12;
const NATIVE_MAX_RANK: usize = 8;
const NATIVE_MAX_TENSOR_NAME: usize = 127;
const NATIVE_MAX_REF_KEY: usize = 23;
const NATIVE_MAX_OP_NAME: usize = 39;

struct CreatedModelBacking {
    dir: PathBuf,
    paths: ModelPaths,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
}

fn validate_native_text(value: &str, context: &str, max_bytes: usize) -> Result<(), FfiError> {
    if value.is_empty() {
        return Err(err(
            format!("{context} must not be empty"),
            INVALID_ARGUMENT,
        ));
    }
    if value.as_bytes().contains(&0) {
        return Err(err(format!("{context} contains NUL"), INVALID_ARGUMENT));
    }
    if value.len() > max_bytes {
        return Err(err(
            format!("{context} exceeds the native {max_bytes}-byte limit"),
            INVALID_ARGUMENT,
        ));
    }
    Ok(())
}

fn validate_created_shape(shape: &[i64], context: &str) -> Result<usize, FfiError> {
    if shape.len() > NATIVE_MAX_RANK {
        return Err(err(
            format!("{context} rank exceeds the native {NATIVE_MAX_RANK}-dimension limit"),
            INVALID_ARGUMENT,
        ));
    }
    shape.iter().try_fold(1usize, |elements, &dimension| {
        if dimension <= 0 || dimension > c_int::MAX as i64 {
            return Err(err(
                format!("{context} dimensions must be in 1..={}", c_int::MAX),
                INVALID_ARGUMENT,
            ));
        }
        elements
            .checked_mul(dimension as usize)
            .ok_or_else(|| err(format!("{context} is too large"), INVALID_ARGUMENT))
    })
}

struct InitializerRng {
    state: u32,
}

impl InitializerRng {
    fn new(seed: u64) -> Self {
        Self { state: seed as u32 }
    }

    fn uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x6d2b_79f5);
        let mut value = self.state;
        value = (value ^ (value >> 15)).wrapping_mul(value | 1);
        value ^= value.wrapping_add((value ^ (value >> 7)).wrapping_mul(value | 61));
        ((value ^ (value >> 14)) as f64) / 4_294_967_296.0
    }
}

fn append_normal_values(
    values: &mut Vec<f32>,
    elements: usize,
    rng: &mut InitializerRng,
    mean: f64,
    stddev: f64,
) -> Result<(), FfiError> {
    while values.len() < elements {
        let u1 = rng.uniform().max(f64::EPSILON);
        let u2 = rng.uniform();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        for standard in [radius * angle.cos(), radius * angle.sin()] {
            if values.len() == elements {
                break;
            }
            let value = (mean + stddev * standard) as f32;
            if !value.is_finite() {
                return Err(err("tensor initializer overflowed F32", INVALID_ARGUMENT));
            }
            values.push(value);
        }
    }
    Ok(())
}

fn materialize_tensor_initializer(tensor: &Tensor, context: &str) -> Result<Tensor, FfiError> {
    let Some(initializer) = tensor.initializer.as_ref() else {
        return Ok(tensor.clone());
    };
    if !tensor.data.is_empty() {
        return Err(err(
            format!("{context} {} cannot provide both data and an initializer", tensor.name),
            INVALID_ARGUMENT,
        ));
    }
    if tensor.quant.is_some() {
        return Err(err(
            format!("{context} {} initializer cannot be quantized", tensor.name),
            INVALID_ARGUMENT,
        ));
    }
    if tensor.dtype != DATA_TYPE_F32 {
        return Err(err(
            format!("{context} {} initializers currently require F32", tensor.name),
            INVALID_ARGUMENT,
        ));
    }
    let elements = validate_created_shape(&tensor.shape, &format!("{context} {}", tensor.name))?;
    let mut rng = InitializerRng::new(initializer.seed.unwrap_or(0));
    let mut values = Vec::with_capacity(elements);
    match initializer.kind {
        1 => values.resize(elements, 0.0f32),
        2 => values.resize(elements, 1.0f32),
        3 => {
            let mean = initializer.mean.unwrap_or(0.0) as f64;
            let stddev = initializer.stddev.unwrap_or(0.02) as f64;
            if !mean.is_finite() || !stddev.is_finite() || stddev < 0.0 {
                return Err(err(
                    format!("{context} {} NORMAL requires finite mean and non-negative stddev", tensor.name),
                    INVALID_ARGUMENT,
                ));
            }
            append_normal_values(&mut values, elements, &mut rng, mean, stddev)?;
        }
        4 | 5 => {
            let (fan_in, fan_out) = if tensor.shape.len() == 1 {
                let fan = tensor.shape[0] as f64;
                (fan, fan)
            } else {
                let receptive_field = tensor.shape[..tensor.shape.len() - 2]
                    .iter()
                    .try_fold(1f64, |product, dimension| {
                        let product = product * *dimension as f64;
                        product.is_finite().then_some(product).ok_or_else(|| {
                            err(
                                format!("{context} {} Xavier fan is too large", tensor.name),
                                INVALID_ARGUMENT,
                            )
                        })
                    })?;
                (
                    tensor.shape[tensor.shape.len() - 2] as f64 * receptive_field,
                    tensor.shape[tensor.shape.len() - 1] as f64 * receptive_field,
                )
            };
            let gain = initializer.gain.unwrap_or(1.0) as f64;
            if !gain.is_finite() || gain < 0.0 {
                return Err(err(
                    format!("{context} {} Xavier gain must be finite and non-negative", tensor.name),
                    INVALID_ARGUMENT,
                ));
            }
            if initializer.kind == 4 {
                let bound = gain * (6.0 / (fan_in + fan_out)).sqrt();
                for _ in 0..elements {
                    values.push(((2.0 * rng.uniform() - 1.0) * bound) as f32);
                }
            } else {
                let stddev = gain * (2.0 / (fan_in + fan_out)).sqrt();
                append_normal_values(&mut values, elements, &mut rng, 0.0, stddev)?;
            }
        }
        _ => {
            return Err(err(
                format!("{context} {} has an unspecified initializer", tensor.name),
                INVALID_ARGUMENT,
            ))
        }
    }
    let mut initialized = tensor.clone();
    initialized.data = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    initialized.initializer = None;
    Ok(initialized)
}

fn created_input_dtype_name(dtype: i32) -> Option<&'static str> {
    match dtype {
        DATA_TYPE_I32 => Some("int32"),
        DATA_TYPE_F32 => Some("float32"),
        _ => None,
    }
}

fn private_model_directory() -> Result<PathBuf, FfiError> {
    use std::os::unix::fs::DirBuilderExt;

    let root = std::env::temp_dir();
    for _ in 0..32 {
        let sequence = PERSIST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let candidate = root.join(format!(".volvoxai-model-{}-{sequence}", std::process::id()));
        let mut builder = fs::DirBuilder::new();
        builder.mode(0o700);
        match builder.create(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(err(
                    format!("create private model backing failed: {error}"),
                    INTERNAL,
                ))
            }
        }
    }
    Err(err("could not allocate private model backing", INTERNAL))
}

fn created_model_definition(
    request: &CreateModelRequest,
) -> Result<(Value, Vec<TensorSpec>, Vec<TensorSpec>), FfiError> {
    if request.inputs.is_empty() {
        return Err(err(
            "CreateModel requires at least one graph input",
            INVALID_ARGUMENT,
        ));
    }
    let graph_nodes = request
        .graph
        .as_ref()
        .map(|graph| graph.nodes.as_slice())
        .unwrap_or_default();
    if graph_nodes.len() > NATIVE_MAX_NODES {
        return Err(err(
            format!("CreateModel graph exceeds the native {NATIVE_MAX_NODES}-node limit"),
            INVALID_ARGUMENT,
        ));
    }

    let mut known_tensors = std::collections::HashSet::new();
    let mut input_json = serde_json::Map::new();
    let mut normalized_inputs = Vec::with_capacity(request.inputs.len());
    for input in &request.inputs {
        validate_native_text(
            &input.name,
            "CreateModel input name",
            NATIVE_MAX_TENSOR_NAME,
        )?;
        if !known_tensors.insert(input.name.clone()) {
            return Err(err(
                format!("duplicate CreateModel tensor name {}", input.name),
                INVALID_ARGUMENT,
            ));
        }
        validate_created_shape(&input.shape, &format!("input {}", input.name))?;
        let dtype_name = created_input_dtype_name(input.dtype).ok_or_else(|| {
            err(
                format!(
                    "CreateModel input {} uses unsupported dtype {}",
                    input.name,
                    dtype_name(input.dtype)
                ),
                INVALID_ARGUMENT,
            )
        })?;
        let size_bytes = tensor_nbytes(input.dtype, &input.shape)?;
        input_json.insert(
            input.name.clone(),
            json!({ "shape": input.shape, "dtype": dtype_name }),
        );
        normalized_inputs.push(TensorSpec {
            name: input.name.clone(),
            shape: input.shape.clone(),
            dtype: input.dtype,
            access_flags: TENSOR_ACCESS_READABLE | TENSOR_ACCESS_WRITABLE,
            size_bytes: size_bytes as i64,
        });
    }

    let mut weight_names = std::collections::HashSet::new();
    for tensor in &request.tensors {
        validate_native_text(
            &tensor.name,
            "CreateModel tensor name",
            NATIVE_MAX_TENSOR_NAME,
        )?;
        if !weight_names.insert(tensor.name.clone()) || !known_tensors.insert(tensor.name.clone()) {
            return Err(err(
                format!("duplicate CreateModel tensor name {}", tensor.name),
                INVALID_ARGUMENT,
            ));
        }
        validate_created_shape(&tensor.shape, &format!("tensor {}", tensor.name))?;
        if proto_to_native_dtype(tensor.dtype).is_none() {
            return Err(err(
                format!(
                    "CreateModel tensor {} uses unsupported native dtype {}",
                    tensor.name,
                    dtype_name(tensor.dtype)
                ),
                INVALID_ARGUMENT,
            ));
        }
        if tensor.quant.is_some() {
            return Err(err(
                format!(
                    "CreateModel tensor {} cannot carry a separate quantization descriptor",
                    tensor.name
                ),
                INVALID_ARGUMENT,
            ));
        }
        let expected = tensor_nbytes(tensor.dtype, &tensor.shape)?;
        if tensor.data.len() != expected {
            return Err(err(
                format!(
                    "CreateModel tensor {} byte-size mismatch: got {}, want {}",
                    tensor.name,
                    tensor.data.len(),
                    expected
                ),
                INVALID_ARGUMENT,
            ));
        }
    }
    if known_tensors.len() > NATIVE_MAX_TENSORS {
        return Err(err(
            format!("CreateModel definition exceeds the native {NATIVE_MAX_TENSORS}-tensor limit"),
            INVALID_ARGUMENT,
        ));
    }

    let mut nodes_json = Vec::with_capacity(graph_nodes.len());
    let mut graph_outputs = std::collections::HashMap::new();
    for (node_index, node) in graph_nodes.iter().enumerate() {
        let node_inputs = string_entries_to_map(&node.inputs);
        let node_outputs = string_entries_to_map(&node.outputs);
        let node_output_shapes = tensor_shape_entries_to_map(&node.output_shapes);
        let op_name = resolve_op_name(
            Some(node.op),
            Some(&node.op_name),
            &format!("CreateModel node {node_index} op"),
            true,
        )?
        .unwrap();
        if node_inputs.len() > NATIVE_MAX_NODE_REFS
            || node_outputs.is_empty()
            || node_outputs.len() > NATIVE_MAX_NODE_REFS
        {
            return Err(err(
                format!(
                    "CreateModel node {node_index} requires 1..={NATIVE_MAX_NODE_REFS} outputs and at most {NATIVE_MAX_NODE_REFS} inputs"
                ),
                INVALID_ARGUMENT,
            ));
        }
        let mut inputs = serde_json::Map::new();
        for (key, tensor_name) in &node_inputs {
            validate_native_text(
                key,
                &format!("CreateModel node {node_index} input key"),
                NATIVE_MAX_REF_KEY,
            )?;
            validate_native_text(
                tensor_name,
                &format!("CreateModel node {node_index} input tensor"),
                NATIVE_MAX_TENSOR_NAME,
            )?;
            if !known_tensors.contains(tensor_name) {
                return Err(err(
                    format!(
                        "CreateModel node {node_index} references unavailable tensor {tensor_name}; nodes must be topologically ordered"
                    ),
                    INVALID_ARGUMENT,
                ));
            }
            inputs.insert(key.clone(), json!(tensor_name));
        }

        if node_output_shapes.len() != node_outputs.len()
            || node_outputs
                .keys()
                .any(|key| !node_output_shapes.contains_key(key))
        {
            return Err(err(
                format!("CreateModel node {node_index} requires one output shape for every output"),
                INVALID_ARGUMENT,
            ));
        }
        let mut outputs = serde_json::Map::new();
        let mut output_shapes = serde_json::Map::new();
        let mut pending_outputs = Vec::with_capacity(node_outputs.len());
        for (key, tensor_name) in &node_outputs {
            validate_native_text(
                key,
                &format!("CreateModel node {node_index} output key"),
                NATIVE_MAX_REF_KEY,
            )?;
            validate_native_text(
                tensor_name,
                &format!("CreateModel node {node_index} output tensor"),
                NATIVE_MAX_TENSOR_NAME,
            )?;
            if known_tensors.contains(tensor_name)
                || pending_outputs
                    .iter()
                    .any(|(name, _): &(String, TensorSpec)| name == tensor_name)
            {
                return Err(err(
                    format!("duplicate CreateModel tensor name {tensor_name}"),
                    INVALID_ARGUMENT,
                ));
            }
            let shape = &node_output_shapes[key].dims;
            let elements = validate_created_shape(
                shape,
                &format!("CreateModel node {node_index} output {tensor_name}"),
            )?;
            outputs.insert(key.clone(), json!(tensor_name));
            output_shapes.insert(key.clone(), json!(shape));
            pending_outputs.push((
                tensor_name.clone(),
                TensorSpec {
                    name: tensor_name.clone(),
                    shape: shape.clone(),
                    dtype: DATA_TYPE_F32,
                    access_flags: TENSOR_ACCESS_READABLE,
                    size_bytes: elements
                        .checked_mul(std::mem::size_of::<f32>())
                        .ok_or_else(|| {
                            err(
                                format!("CreateModel output {tensor_name} is too large"),
                                INVALID_ARGUMENT,
                            )
                        })? as i64,
                },
            ));
        }

        let params = node
            .params_json
            .as_deref()
            .map(|text| {
                let value: Value = serde_json::from_str(text).map_err(|error| {
                    err(
                        format!("CreateModel node {node_index} has invalid params_json: {error}"),
                        INVALID_ARGUMENT,
                    )
                })?;
                if !value.is_object() {
                    return Err(err(
                        format!("CreateModel node {node_index} params_json must be an object"),
                        INVALID_ARGUMENT,
                    ));
                }
                Ok(value)
            })
            .transpose()?;
        let mut node_json = serde_json::Map::new();
        node_json.insert("opType".into(), json!(op_name));
        node_json.insert("inputs".into(), Value::Object(inputs));
        node_json.insert("outputs".into(), Value::Object(outputs));
        node_json.insert("outputs_shape".into(), Value::Object(output_shapes));
        if let Some(params) = params {
            node_json.insert("params".into(), params);
        }
        nodes_json.push(Value::Object(node_json));
        for (name, spec) in pending_outputs {
            known_tensors.insert(name.clone());
            graph_outputs.insert(name, spec);
        }
        if known_tensors.len() > NATIVE_MAX_TENSORS {
            return Err(err(
                format!("CreateModel graph exceeds the native {NATIVE_MAX_TENSORS}-tensor limit"),
                INVALID_ARGUMENT,
            ));
        }
    }

    if request.output_names.is_empty() && !graph_nodes.is_empty() {
        return Err(err(
            "CreateModel requires at least one output_name",
            INVALID_ARGUMENT,
        ));
    }
    let mut seen_outputs = std::collections::HashSet::new();
    let mut normalized_outputs = Vec::with_capacity(request.output_names.len());
    let mut root_outputs = serde_json::Map::new();
    for name in &request.output_names {
        if !seen_outputs.insert(name.as_str()) {
            return Err(err(
                format!("duplicate CreateModel output_name {name}"),
                INVALID_ARGUMENT,
            ));
        }
        let spec = graph_outputs.get(name).ok_or_else(|| {
            err(
                format!("CreateModel output_name {name} is not produced by the graph"),
                INVALID_ARGUMENT,
            )
        })?;
        root_outputs.insert(name.clone(), json!(name));
        normalized_outputs.push(spec.clone());
    }

    Ok((
        json!({
            "format": "volvox.api.v1",
            "inputs": Value::Object(input_json),
            "outputs": Value::Object(root_outputs),
            "nodes": nodes_json,
        }),
        normalized_inputs,
        normalized_outputs,
    ))
}

fn materialize_created_model(
    request: &CreateModelRequest,
) -> Result<CreatedModelBacking, FfiError> {
    let mut materialized = request.clone();
    materialized.tensors = request
        .tensors
        .iter()
        .map(|tensor| materialize_tensor_initializer(tensor, "CreateModel tensor"))
        .collect::<Result<_, _>>()?;
    let (config, inputs, outputs) = created_model_definition(&materialized)?;
    let dir = private_model_directory()?;
    let result = (|| {
        let config_path = dir.join("config.json");
        let weights_path = dir.join("model.safetensors");
        let config_path_string = config_path
            .to_str()
            .ok_or_else(|| err("private config path is not UTF-8", INTERNAL))?
            .to_string();
        let weights_path_string = weights_path
            .to_str()
            .ok_or_else(|| err("private weights path is not UTF-8", INTERNAL))?
            .to_string();
        let mut metadata = string_entries_to_map(&materialized.metadata);
        metadata.insert("volvox.model_origin".to_string(), "api".to_string());
        metadata.insert(
            "volvox.model_format".to_string(),
            "volvox.api.v1".to_string(),
        );
        write_safetensors(&weights_path_string, &materialized.tensors, &metadata, false)?;
        let bytes = serde_json::to_vec_pretty(&config).map_err(|error| {
            err(
                format!("serialize API model config failed: {error}"),
                INTERNAL,
            )
        })?;
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&config_path)
            .map_err(|error| err(format!("create API model config failed: {error}"), INTERNAL))?;
        file.write_all(&bytes)
            .map_err(|error| err(format!("write API model config failed: {error}"), INTERNAL))?;
        file.sync_all()
            .map_err(|error| err(format!("fsync API model config failed: {error}"), INTERNAL))?;
        sync_parent(&config_path)?;
        let tokenizer_path = if materialized.tokenizer.is_empty() {
            String::new()
        } else {
            let tokenizer_path = dir.join("tokenizer.bin");
            let mut tokenizer = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tokenizer_path)
                .map_err(|error| err(format!("create API tokenizer failed: {error}"), INTERNAL))?;
            tokenizer
                .write_all(&materialized.tokenizer)
                .map_err(|error| err(format!("write API tokenizer failed: {error}"), INTERNAL))?;
            tokenizer
                .sync_all()
                .map_err(|error| err(format!("fsync API tokenizer failed: {error}"), INTERNAL))?;
            sync_parent(&tokenizer_path)?;
            tokenizer_path
                .to_str()
                .ok_or_else(|| err("private tokenizer path is not UTF-8", INTERNAL))?
                .to_string()
        };
        Ok(CreatedModelBacking {
            dir: dir.clone(),
            paths: ModelPaths {
                config_path: config_path_string,
                weights_path: weights_path_string,
                tokenizer_path,
                extra_weights_paths: Vec::new(),
            },
            inputs,
            outputs,
        })
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&dir);
    }
    result
}

fn reconcile_owned_model_dirs(previous: Option<&Path>, current: Option<&Path>, keep_current: bool) {
    if let Some(path) = previous {
        if !keep_current || current != Some(path) {
            let _ = fs::remove_dir_all(path);
        }
    }
    if !keep_current {
        if let Some(path) = current {
            if previous != Some(path) {
                let _ = fs::remove_dir_all(path);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------
const ADAPTER_MANIFEST_FORMAT: &str = "volvox.adapter.v1";
const ADAPTER_MANIFEST_METADATA_KEY: &str = "volvox_adapter_manifest";

#[derive(Clone)]
struct AdapterRecord {
    native_name: String,
    info: AdapterInfo,
}

struct Inner {
    model_id: Option<String>,
    model_backend: RuntimeBackend,
    info: Option<ModelInfo>,
    config_path: String,
    tokenizer_path: String,
    model_metadata: std::collections::HashMap<String, String>,
    owned_model_dir: Option<PathBuf>,
    tok: *mut c_void,
    counter: u64,
    training_step: i64,
    training_optimizer: Option<OptimizerOptions>,
    training_update_mode: Option<i32>,
    training_accumulation_microbatches: i32,
    training_accumulation_steps: i32,
    training_accumulation_optimizer_step: Option<i64>,
    training_accumulation_optimizer: Option<OptimizerOptions>,
    training_accumulation_update_mode: Option<i32>,
    allow_tensor_updates: bool,
    allow_graph_patches: bool,
    adapter_counter: u64,
    adapter_generation: u64,
    adapters: Vec<AdapterRecord>,
    active_adapters: AdapterSelection,
    merged_adapter: Option<AdapterVersionRef>,
    pre_merge_active: Option<AdapterSelection>,
}
// tok is a raw pointer, only ever touched under the mutex.
unsafe impl Send for Inner {}

struct Plugin {
    inner: Mutex<Inner>,
    // Serializes whole model replacement/unload/checkpoint-restore transactions;
    // model_lifecycle protects the individual native critical sections inside.
    model_transaction: Mutex<()>,
    // Graph execution remains serialized because the native graph arena and
    // KV cache are process-global. Adapter registry operations use a separate
    // lock so an atomic activation never waits for a long Run/Generate call.
    engine_exec: Mutex<()>,
    adapter_admin: Mutex<()>,
    model_lifecycle: RwLock<()>,
}

#[derive(Debug)]
struct PreparedCrossEntropyLoss {
    name: String,
    c_name: CString,
    c_logits: CString,
    targets: Vec<c_int>,
    ignore_index: c_int,
    last_token: c_int,
    weight: f32,
    normalizer: f32,
}

impl PreparedCrossEntropyLoss {
    fn native(&self) -> NativeCrossEntropyLoss {
        NativeCrossEntropyLoss {
            name: self.c_name.as_ptr(),
            logits_name: self.c_logits.as_ptr(),
            targets: self.targets.as_ptr(),
            target_count: self.targets.len() as c_int,
            ignore_index: self.ignore_index,
            row_index: self.last_token,
            weight: self.weight,
            normalizer: self.normalizer,
        }
    }
}

fn prepare_cross_entropy_losses(
    request: &TrainStepRequest,
    accumulation_steps: i32,
) -> Result<Vec<PreparedCrossEntropyLoss>, FfiError> {
    let descriptors = if request.losses.is_empty() {
        if request.logits_tensor.is_empty() {
            return Err(err("TrainStep requires logits_tensor or losses", INVALID_ARGUMENT));
        }
        if request.target_ids.is_empty() {
            return Err(err("TrainStep requires target_ids or losses", INVALID_ARGUMENT));
        }
        vec![CrossEntropyLoss {
            name: "loss".to_string(),
            logits_tensor: request.logits_tensor.clone(),
            target_ids: request.target_ids.clone(),
            weight: Some(1.0),
            ignore_id: request.ignore_id,
            last_token: request.last_token,
            normalizer: None,
        }]
    } else {
        if !request.logits_tensor.is_empty()
            || !request.target_ids.is_empty()
            || request.last_token.is_some()
            || request.ignore_id.is_some()
        {
            return Err(err(
                "TrainStep accepts losses or the legacy single-loss fields, not both",
                INVALID_ARGUMENT,
            ));
        }
        request.losses.clone()
    };
    if descriptors.len() > c_int::MAX as usize {
        return Err(err("TrainStep has too many losses", INVALID_ARGUMENT));
    }

    let mut names = std::collections::HashSet::new();
    let mut prepared = Vec::with_capacity(descriptors.len());
    for descriptor in descriptors {
        if descriptor.name.is_empty() || !names.insert(descriptor.name.clone()) {
            return Err(err(
                "TrainStep loss names must be unique non-empty strings",
                INVALID_ARGUMENT,
            ));
        }
        if descriptor.logits_tensor.is_empty() {
            return Err(err(
                format!(
                    "TrainStep loss '{}' requires logits_tensor",
                    descriptor.name
                ),
                INVALID_ARGUMENT,
            ));
        }
        if descriptor.target_ids.is_empty() || descriptor.target_ids.len() > c_int::MAX as usize {
            return Err(err(
                format!(
                    "TrainStep loss '{}' requires a supported number of target_ids",
                    descriptor.name
                ),
                INVALID_ARGUMENT,
            ));
        }
        let ignore_index = descriptor.ignore_id.unwrap_or(c_int::MIN);
        let targets = descriptor
            .target_ids
            .iter()
            .map(|&id| {
                if id == ignore_index as i64 {
                    Ok(ignore_index)
                } else if id < 0 || id > c_int::MAX as i64 {
                    Err(err(
                        format!(
                            "TrainStep loss '{}' target id is out of int32 range",
                            descriptor.name
                        ),
                        INVALID_ARGUMENT,
                    ))
                } else {
                    Ok(id as c_int)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        if descriptor.last_token.is_some_and(|position| position < 0) {
            return Err(err(
                format!(
                    "TrainStep loss '{}' last_token must be non-negative",
                    descriptor.name
                ),
                INVALID_ARGUMENT,
            ));
        }
        let weight = descriptor.weight.unwrap_or(1.0);
        if !weight.is_finite() || weight < 0.0 {
            return Err(err(
                format!(
                    "TrainStep loss '{}' weight must be finite and non-negative",
                    descriptor.name
                ),
                INVALID_ARGUMENT,
            ));
        }
        if descriptor
            .normalizer
            .is_some_and(|normalizer| !normalizer.is_finite() || normalizer <= 0.0)
        {
            return Err(err(
                format!(
                    "TrainStep loss '{}' normalizer must be finite and positive",
                    descriptor.name
                ),
                INVALID_ARGUMENT,
            ));
        }
        if accumulation_steps > 1 && descriptor.normalizer.is_none() {
            return Err(err(
                format!(
                    "TrainStep loss '{}' requires normalizer when gradient accumulation is enabled",
                    descriptor.name
                ),
                INVALID_ARGUMENT,
            ));
        }
        prepared.push(PreparedCrossEntropyLoss {
            c_name: CString::new(descriptor.name.as_str())
                .map_err(|_| err("bad TrainStep loss name", INVALID_ARGUMENT))?,
            c_logits: CString::new(descriptor.logits_tensor.as_str())
                .map_err(|_| err("bad TrainStep logits tensor name", INVALID_ARGUMENT))?,
            name: descriptor.name,
            targets,
            ignore_index,
            last_token: descriptor.last_token.unwrap_or(-1),
            weight,
            normalizer: descriptor.normalizer.unwrap_or(0.0),
        });
    }
    Ok(prepared)
}

impl Plugin {
    fn new() -> Self {
        Plugin {
            inner: Mutex::new(Inner {
                model_id: None,
                model_backend: RuntimeBackend::Cpu,
                info: None,
                config_path: String::new(),
                tokenizer_path: String::new(),
                model_metadata: std::collections::HashMap::new(),
                owned_model_dir: None,
                tok: std::ptr::null_mut(),
                counter: 0,
                training_step: 0,
                training_optimizer: None,
                training_update_mode: None,
                training_accumulation_microbatches: 0,
                training_accumulation_steps: 1,
                training_accumulation_optimizer_step: None,
                training_accumulation_optimizer: None,
                training_accumulation_update_mode: None,
                allow_tensor_updates: false,
                allow_graph_patches: false,
                adapter_counter: 0,
                adapter_generation: 0,
                adapters: Vec::new(),
                active_adapters: AdapterSelection::default(),
                merged_adapter: None,
                pre_merge_active: None,
            }),
            model_transaction: Mutex::new(()),
            engine_exec: Mutex::new(()),
            adapter_admin: Mutex::new(()),
            model_lifecycle: RwLock::new(()),
        }
    }

    fn load_model_paths(
        &self,
        paths: ModelPaths,
        exec: Option<ExecOptions>,
        edit: ModelEditOptions,
        declared_inputs: Vec<TensorSpec>,
        declared_outputs: Vec<TensorSpec>,
        owned_model_dir: Option<PathBuf>,
        restored_training_step: i64,
        optimizer_state_path: Option<String>,
        restored_optimizer: Option<OptimizerOptions>,
        restored_update_mode: Option<i32>,
        model_metadata: std::collections::HashMap<String, String>,
    ) -> Result<LoadModelResponse, FfiError> {
        let allow_tensor_updates = edit.allow_tensor_updates || edit.open_weights_writable;
        let requested_backend = model_backend(&exec)?;
        validate_engine_backend_request(requested_backend, &exec)?;
        let cfg = CString::new(paths.config_path.as_str())
            .map_err(|_| err("bad config path", INVALID_ARGUMENT))?;
        let mut declared_inputs = declared_inputs;
        let mut declared_outputs = declared_outputs;
        let mut weight_strings = Vec::new();
        let mut adapter_sources = Vec::new();
        if !paths.weights_path.is_empty() {
            read_safetensors_info(&paths.weights_path, false)?;
            weight_strings.push(
                CString::new(paths.weights_path.as_str())
                    .map_err(|_| err("bad weights path", INVALID_ARGUMENT))?,
            );
        }
        for path in &paths.extra_weights_paths {
            match read_extra_adapter_checkpoint(path)? {
                Some((_file, manifest, tensors)) => {
                    CString::new(path.as_str())
                        .map_err(|_| err("bad adapter checkpoint path", INVALID_ARGUMENT))?;
                    adapter_sources.push((path.clone(), manifest, tensors));
                }
                None => {
                    weight_strings.push(
                        CString::new(path.as_str())
                            .map_err(|_| err("bad extra weights path", INVALID_ARGUMENT))?,
                    );
                }
            }
        }
        let tokenizer_path = if paths.tokenizer_path.is_empty() {
            None
        } else {
            fs::metadata(&paths.tokenizer_path)
                .map_err(|e| err(format!("tokenizer path is unavailable: {e}"), NOT_FOUND))?;
            Some(
                CString::new(paths.tokenizer_path.as_str())
                    .map_err(|_| err("bad tokenizer path", INVALID_ARGUMENT))?,
            )
        };
        let weight_ptrs: Vec<*const c_char> =
            weight_strings.iter().map(|path| path.as_ptr()).collect();
        let weight_ptrs_data = if weight_ptrs.is_empty() {
            std::ptr::null()
        } else {
            weight_ptrs.as_ptr()
        };

        let _lifecycle = self.model_lifecycle.write().unwrap();
        let _exec = self.engine_exec.lock().unwrap();
        let _admin = self.adapter_admin.lock().unwrap();
        let mut inner = self.inner.lock().unwrap();
        let previous_backend_call = begin_backend_call(inner.model_backend, &None, false)?;
        let previous_owned = inner.owned_model_dir.take();
        let reuses_previous = previous_owned.as_ref().is_some_and(|directory| {
            Path::new(&paths.config_path).starts_with(directory)
                || Path::new(&paths.weights_path).starts_with(directory)
                || paths
                    .extra_weights_paths
                    .iter()
                    .any(|path| Path::new(path).starts_with(directory))
        });
        let current_owned = owned_model_dir.or_else(|| {
            if reuses_previous {
                previous_owned.clone()
            } else {
                None
            }
        });
        unsafe {
            if !inner.tok.is_null() {
                volvoxai_tokenizer_free(inner.tok);
                inner.tok = std::ptr::null_mut();
            }
            previous_backend_call.shutdown_engine();
            inner.model_id = None;
            inner.model_backend = RuntimeBackend::Cpu;
            inner.info = None;
            inner.config_path.clear();
            inner.tokenizer_path.clear();
            inner.model_metadata.clear();
            inner.owned_model_dir = None;
            inner.training_step = 0;
            inner.training_optimizer = None;
            inner.training_update_mode = None;
            inner.training_accumulation_microbatches = 0;
            inner.training_accumulation_steps = 1;
            inner.training_accumulation_optimizer_step = None;
            inner.training_accumulation_optimizer = None;
            inner.training_accumulation_update_mode = None;
            inner.allow_tensor_updates = false;
            inner.allow_graph_patches = false;
            inner.adapter_counter = 0;
            inner.adapter_generation += 1;
            inner.adapters.clear();
            inner.active_adapters = AdapterSelection::default();
            inner.merged_adapter = None;
            inner.pre_merge_active = None;
            if let Err(error) = configure_engine_backend(requested_backend, &exec) {
                reconcile_owned_model_dirs(
                    previous_owned.as_deref(),
                    current_owned.as_deref(),
                    false,
                );
                return Err(error);
            }
            let backend_call = match begin_backend_call(requested_backend, &None, false) {
                Ok(guard) => guard,
                Err(error) => {
                    reconcile_owned_model_dirs(
                        previous_owned.as_deref(),
                        current_owned.as_deref(),
                        false,
                    );
                    return Err(error);
                }
            };
            let backend_load = BackendLoadGuard::new(backend_call);
            if volvoxai_engine_init_with_weight_files(
                cfg.as_ptr(),
                weight_ptrs_data,
                weight_ptrs.len() as c_int,
            ) != 0
            {
                volvoxai_engine_shutdown();
                reconcile_owned_model_dirs(
                    previous_owned.as_deref(),
                    current_owned.as_deref(),
                    false,
                );
                return Err(err("volvoxai_engine_init failed", INTERNAL));
            }
            if let Some(path) = optimizer_state_path.as_deref() {
                let path = CString::new(path)
                    .map_err(|_| err("bad optimizer checkpoint path", INVALID_ARGUMENT))?;
                let mut native_step: c_long = 0;
                if volvoxai_engine_load_optimizer_state(path.as_ptr(), &mut native_step) != 0
                    || native_step != restored_training_step
                {
                    volvoxai_engine_shutdown();
                    reconcile_owned_model_dirs(
                        previous_owned.as_deref(),
                        current_owned.as_deref(),
                        false,
                    );
                    return Err(err(
                        "optimizer checkpoint is invalid or its training step does not match the manifest",
                        INVALID_ARGUMENT,
                    ));
                }
            }
            let mut model_num_ops = 0;
            let inferred_declarations = (|| {
                if declared_inputs.is_empty() {
                    declared_inputs = native_graph_interface_names(true)?
                        .into_iter()
                        .map(|name| tensor_spec_native(&name, TENSOR_ACCESS_READABLE))
                        .collect::<Result<_, _>>()?;
                }
                if declared_outputs.is_empty() {
                    declared_outputs = native_graph_interface_names(false)?
                        .into_iter()
                        .map(|name| tensor_spec_native(&name, TENSOR_ACCESS_READABLE))
                        .collect::<Result<_, _>>()?;
                }
                let inspection = native_inspection(false, false, false, false)?;
                let count = inspection
                    .get("num_ops")
                    .and_then(Value::as_i64)
                    .ok_or_else(|| err("native inspection omitted num_ops", INTERNAL))?;
                model_num_ops = i32::try_from(count)
                    .map_err(|_| err("native num_ops is out of int32 range", INTERNAL))?;
                Ok::<(), FfiError>(())
            })();
            if let Err(error) = inferred_declarations {
                volvoxai_engine_shutdown();
                reconcile_owned_model_dirs(
                    previous_owned.as_deref(),
                    current_owned.as_deref(),
                    false,
                );
                return Err(error);
            }
            let mut loaded_adapters = Vec::with_capacity(adapter_sources.len());
            for (path, manifest, tensors) in adapter_sources {
                if let Err(error) = validate_adapter_targets_against_model(&manifest) {
                    volvoxai_engine_shutdown();
                    reconcile_owned_model_dirs(
                        previous_owned.as_deref(),
                        current_owned.as_deref(),
                        false,
                    );
                    return Err(error);
                }
                inner.adapter_counter += 1;
                let adapter_id = manifest.adapter_id.clone();
                if adapter_id.is_empty() {
                    volvoxai_engine_shutdown();
                    reconcile_owned_model_dirs(
                        previous_owned.as_deref(),
                        current_owned.as_deref(),
                        false,
                    );
                    return Err(err(
                        "adapter checkpoint has an empty adapter_id",
                        INVALID_ARGUMENT,
                    ));
                }
                let version_id = format!("{adapter_id}@v{}", inner.adapter_counter);
                if let Err(error) =
                    native_stage_adapter(&manifest, &version_id, &version_id, &tensors)
                {
                    volvoxai_engine_shutdown();
                    reconcile_owned_model_dirs(
                        previous_owned.as_deref(),
                        current_owned.as_deref(),
                        false,
                    );
                    return Err(error);
                }
                let manifest = match normalize_adapter_manifest(manifest) {
                    Ok(manifest) => manifest,
                    Err(error) => {
                        volvoxai_engine_shutdown();
                        reconcile_owned_model_dirs(
                            previous_owned.as_deref(),
                            current_owned.as_deref(),
                            false,
                        );
                        return Err(error);
                    }
                };
                let parent_version_id = string_entry_value(
                    &manifest.metadata,
                    "volvox.parent_version_id",
                )
                    .map(str::to_string)
                    .unwrap_or_default();
                let optimizer_step = string_entry_value(
                    &manifest.metadata,
                    "volvox.optimizer_step",
                )
                    .and_then(|step| step.parse::<i64>().ok())
                    .unwrap_or(0);
                loaded_adapters.push(AdapterRecord {
                    native_name: version_id.clone(),
                    info: AdapterInfo {
                        adapter: Some(AdapterVersionRef {
                            adapter_id,
                            version_id,
                        }),
                        manifest: Some(manifest.clone()),
                        parent_version_id,
                        source_path: path,
                        size_bytes: adapter_manifest_resident_bytes(&manifest),
                        active: false,
                        merged: false,
                        optimizer_step,
                    },
                });
            }
            if let Some(path) = tokenizer_path {
                inner.tok = volvoxai_tokenizer_init(path.as_ptr(), std::ptr::null());
            }
            inner.counter += 1;
            let id = format!("model-{}", inner.counter);
            let info = ModelInfo {
                inputs: declared_inputs,
                outputs: declared_outputs,
                num_ops: model_num_ops,
                supports_kv_cache: !inner.tok.is_null(),
                has_tokenizer: !inner.tok.is_null(),
                op_counts: vec![],
                writable_weights: allow_tensor_updates,
                patchable_graph: edit.allow_graph_patches,
            };
            inner.model_id = Some(id.clone());
            inner.model_backend = requested_backend;
            inner.info = Some(info.clone());
            inner.config_path = paths.config_path.clone();
            inner.tokenizer_path = paths.tokenizer_path.clone();
            inner.model_metadata = model_metadata;
            inner.training_step = restored_training_step;
            inner.training_optimizer = restored_optimizer;
            inner.training_update_mode = restored_update_mode;
            inner.training_accumulation_microbatches = 0;
            inner.training_accumulation_steps = 1;
            inner.training_accumulation_optimizer_step = None;
            inner.training_accumulation_optimizer = None;
            inner.training_accumulation_update_mode = None;
            inner.allow_tensor_updates = allow_tensor_updates;
            inner.allow_graph_patches = edit.allow_graph_patches;
            inner.adapters = loaded_adapters;
            inner.active_adapters = AdapterSelection::default();
            inner.merged_adapter = None;
            inner.pre_merge_active = None;
            reconcile_owned_model_dirs(previous_owned.as_deref(), current_owned.as_deref(), true);
            inner.owned_model_dir = current_owned;
            backend_load.commit();
            Ok(LoadModelResponse {
                model_id: id,
                info: Some(info),
            })
        }
    }

    fn stage_adapter_version(
        &self,
        model_id: &str,
        manifest_override: Option<AdapterManifest>,
        parent: Option<&AdapterVersionRef>,
        tensors: &[Tensor],
        source_path: Option<&str>,
        activate: bool,
        optimizer_step: i64,
    ) -> Result<AdapterInfo, FfiError> {
        let _lifecycle = self.model_lifecycle.read().unwrap();
        let (mut manifest, adapter_id, version_id, native_name, parent_version_id) = {
            let mut inner = self.inner.lock().unwrap();
            require(&inner, model_id)?;
            let parent_info = parent
                .map(|reference| {
                    adapter_record(&inner, reference).map(|record| record.info.clone())
                })
                .transpose()?;
            let mut manifest = manifest_override
                .or_else(|| parent_info.as_ref().and_then(|info| info.manifest.clone()))
                .ok_or_else(|| err("adapter manifest is required", INVALID_ARGUMENT))?;
            if manifest.format.is_empty() {
                manifest.format = ADAPTER_MANIFEST_FORMAT.to_string();
            }
            let inherited_id = parent_info
                .as_ref()
                .and_then(|info| info.adapter.as_ref())
                .map(|reference| reference.adapter_id.as_str());
            if let Some(id) = inherited_id {
                if !manifest.adapter_id.is_empty() && manifest.adapter_id != id {
                    return Err(err(
                        "UpdateAdapter cannot change the adapter_id",
                        INVALID_ARGUMENT,
                    ));
                }
                manifest.adapter_id = id.to_string();
            }
            inner.adapter_counter += 1;
            let sequence = inner.adapter_counter;
            let adapter_id = if manifest.adapter_id.is_empty() {
                format!("adapter-{sequence}")
            } else {
                manifest.adapter_id.clone()
            };
            manifest.adapter_id = adapter_id.clone();
            let version_id = format!("{adapter_id}@v{sequence}");
            let native_name = version_id.clone();
            let parent_version_id = parent
                .map(|reference| reference.version_id.clone())
                .unwrap_or_default();
            (
                manifest,
                adapter_id,
                version_id,
                native_name,
                parent_version_id,
            )
        };

        apply_adapter_scale_default(&mut manifest);
        validate_adapter_manifest(&manifest, Some(tensors))?;
        unsafe { validate_adapter_targets_against_model(&manifest)? };
        unsafe { native_stage_adapter(&manifest, &native_name, &version_id, tensors)? };
        let _admin = self.adapter_admin.lock().unwrap();
        unsafe {
            if activate {
                let c_name = CString::new(native_name.as_str()).unwrap();
                if volvoxai_engine_adapter_activate(c_name.as_ptr()) != 0 {
                    let _ = volvoxai_engine_adapter_remove(c_name.as_ptr());
                    return Err(err("native adapter activation failed", FAILED_PRECONDITION));
                }
            }
        }

        let manifest = normalize_adapter_manifest(manifest)?;
        let parent_version_id = if parent_version_id.is_empty() {
            string_entry_value(&manifest.metadata, "volvox.parent_version_id")
                .map(str::to_string)
                .unwrap_or_default()
        } else {
            parent_version_id
        };
        let optimizer_step = if optimizer_step == 0 {
            string_entry_value(&manifest.metadata, "volvox.optimizer_step")
                .and_then(|step| step.parse::<i64>().ok())
                .unwrap_or(0)
        } else {
            optimizer_step
        };
        let resident_bytes = adapter_manifest_resident_bytes(&manifest);
        let info = AdapterInfo {
            adapter: Some(AdapterVersionRef {
                adapter_id,
                version_id,
            }),
            manifest: Some(manifest),
            parent_version_id,
            source_path: source_path.unwrap_or_default().to_string(),
            size_bytes: resident_bytes,
            active: activate,
            merged: false,
            optimizer_step,
        };
        let mut inner = self.inner.lock().unwrap();
        // adapter_admin excludes LoadModel/UnloadModel, so model identity is
        // stable across the native registry call without holding state mutex.
        require(&inner, model_id)?;
        inner.adapters.push(AdapterRecord {
            native_name,
            info: info.clone(),
        });
        if activate {
            inner.active_adapters = AdapterSelection {
                routes: vec![AdapterRoute {
                    adapter: info.adapter.clone(),
                    scale: None,
                }],
            };
        }
        inner.adapter_generation += 1;
        Ok(adapter_info_with_state(
            &inner,
            inner.adapters.last().unwrap(),
        ))
    }

    fn update_adapter_version(
        &self,
        model_id: &str,
        parent: &AdapterVersionRef,
        updates: &[AdapterTensorUpdate],
        activate: bool,
        optimizer_step: i64,
    ) -> Result<AdapterInfo, FfiError> {
        if updates.is_empty() {
            return Err(err(
                "UpdateAdapter requires at least one tensor update",
                INVALID_ARGUMENT,
            ));
        }
        if optimizer_step < 0 {
            return Err(err(
                "UpdateAdapter optimizer_step must be non-negative",
                INVALID_ARGUMENT,
            ));
        }
        let _lifecycle = self.model_lifecycle.read().unwrap();
        let (parent_record, adapter_id, version_id, native_name) = {
            let mut inner = self.inner.lock().unwrap();
            require(&inner, model_id)?;
            if activate {
                require_unmerged(&inner, "UpdateAdapter activation")?;
            }
            let parent_record = adapter_record(&inner, parent)?.clone();
            inner.adapter_counter += 1;
            let adapter_id = parent.adapter_id.clone();
            let version_id = format!("{adapter_id}@v{}", inner.adapter_counter);
            let native_name = version_id.clone();
            (parent_record, adapter_id, version_id, native_name)
        };
        let mut manifest = parent_record
            .info
            .manifest
            .clone()
            .ok_or_else(|| err("parent adapter manifest is unavailable", INTERNAL))?;
        let specs: std::collections::HashMap<&str, &TensorSpec> = manifest
            .targets
            .iter()
            .flat_map(|target| &target.tensors)
            .filter_map(|binding| {
                binding
                    .spec
                    .as_ref()
                    .map(|spec| (binding.tensor_name.as_str(), spec))
            })
            .collect();
        let mut seen = std::collections::HashSet::new();
        let mut tensors = Vec::with_capacity(updates.len());
        let mut modes = Vec::with_capacity(updates.len());
        for update in updates {
            let tensor = update
                .tensor
                .as_ref()
                .ok_or_else(|| err("adapter update is missing its tensor", INVALID_ARGUMENT))?;
            if !seen.insert(tensor.name.as_str()) {
                return Err(err(
                    format!("adapter tensor {} is updated more than once", tensor.name),
                    INVALID_ARGUMENT,
                ));
            }
            let spec = specs.get(tensor.name.as_str()).ok_or_else(|| {
                err(
                    format!("parent adapter has no tensor {}", tensor.name),
                    NOT_FOUND,
                )
            })?;
            if tensor.shape != spec.shape {
                return Err(err(
                    format!(
                        "adapter update {} shape {:?} does not match {:?}",
                        tensor.name, tensor.shape, spec.shape
                    ),
                    INVALID_ARGUMENT,
                ));
            }
            if tensor.dtype != DATA_TYPE_F32 && tensor.dtype != DATA_TYPE_F16 {
                return Err(err(
                    format!("adapter update {} must be F32/F16", tensor.name),
                    INVALID_ARGUMENT,
                ));
            }
            if tensor.data.len() != tensor_nbytes(tensor.dtype, &tensor.shape)? {
                return Err(err(
                    format!("adapter update {} payload size is invalid", tensor.name),
                    INVALID_ARGUMENT,
                ));
            }
            let native_mode = adapter_update_mode_to_native(update.mode).ok_or_else(|| {
                err(
                    format!("adapter update {} has an invalid mode", tensor.name),
                    INVALID_ARGUMENT,
                )
            })?;
            tensors.push(tensor);
            modes.push(native_mode);
        }

        let source = CString::new(parent_record.native_name.as_str()).unwrap();
        let c_adapter_id = CString::new(adapter_id.as_str())
            .map_err(|_| err("adapter id contains NUL", INVALID_ARGUMENT))?;
        let c_version = CString::new(native_name.as_str()).unwrap();
        let names: Vec<CString> = tensors
            .iter()
            .map(|tensor| CString::new(tensor.name.as_str()))
            .collect::<Result<_, _>>()
            .map_err(|_| err("adapter tensor name contains NUL", INVALID_ARGUMENT))?;
        let name_ptrs: Vec<*const c_char> = names.iter().map(|name| name.as_ptr()).collect();
        let data_ptrs: Vec<*const c_void> = tensors
            .iter()
            .map(|tensor| tensor.data.as_ptr() as *const c_void)
            .collect();
        let dtypes: Vec<c_int> = tensors
            .iter()
            .map(|tensor| proto_to_native_dtype(tensor.dtype).unwrap())
            .collect();
        let sizes: Vec<usize> = tensors.iter().map(|tensor| tensor.data.len()).collect();
        let metadata_json = CString::new(
            json!({ "volvox.optimizer_step": optimizer_step.to_string() }).to_string(),
        )
        .unwrap();
        unsafe {
            if volvoxai_engine_adapter_clone_update_with_metadata(
                source.as_ptr(),
                c_adapter_id.as_ptr(),
                c_version.as_ptr(),
                name_ptrs.as_ptr(),
                data_ptrs.as_ptr(),
                dtypes.as_ptr(),
                sizes.as_ptr(),
                modes.as_ptr(),
                tensors.len() as c_int,
                metadata_json.as_ptr(),
            ) != 0
            {
                return Err(err(
                    "native adapter copy-on-write update failed",
                    INVALID_ARGUMENT,
                ));
            }
        }
        let _admin = self.adapter_admin.lock().unwrap();
        unsafe {
            if activate && volvoxai_engine_adapter_activate(c_version.as_ptr()) != 0 {
                let _ = volvoxai_engine_adapter_remove(c_version.as_ptr());
                return Err(err("native adapter activation failed", FAILED_PRECONDITION));
            }
        }

        string_entry_upsert(
            &mut manifest.metadata,
            "volvox.parent_version_id".to_string(),
            parent.version_id.clone(),
        );
        string_entry_upsert(
            &mut manifest.metadata,
            "volvox.optimizer_step".to_string(),
            optimizer_step.to_string(),
        );
        let mut info = parent_record.info;
        info.manifest = Some(manifest);
        info.adapter = Some(AdapterVersionRef {
            adapter_id,
            version_id,
        });
        info.parent_version_id = parent.version_id.clone();
        info.source_path.clear();
        info.active = activate;
        info.merged = false;
        info.optimizer_step = optimizer_step;
        let mut inner = self.inner.lock().unwrap();
        require(&inner, model_id)?;
        inner.adapters.push(AdapterRecord {
            native_name,
            info: info.clone(),
        });
        if activate {
            inner.active_adapters = AdapterSelection {
                routes: vec![AdapterRoute {
                    adapter: info.adapter.clone(),
                    scale: None,
                }],
            };
        }
        inner.adapter_generation += 1;
        Ok(adapter_info_with_state(
            &inner,
            inner.adapters.last().unwrap(),
        ))
    }

    fn train_step_impl(&self, request: TrainStepRequest) -> Result<TrainStepResponse, FfiError> {
        let _persistence = request
            .persist_safetensors
            .then(|| PERSISTENCE_COMMIT.lock().unwrap());
        let _exec = self.engine_exec.lock().unwrap();
        let _admin = self.adapter_admin.lock().unwrap();
        let mut inner = self.inner.lock().unwrap();
        require(&inner, &request.model_id)?;
        require_unmerged(&inner, "TrainStep")?;
        let _backend_call = begin_backend_call(inner.model_backend, &None, true)?;
        let native_backend_policy = NativeTrainingBackendGuard::new(inner.model_backend)?;
        let _execution_row = ExecutionRowGuard::new(-1)?;
        if !inner.allow_tensor_updates {
            return Err(err(
                "TrainStep requires CreateModel or LoadModelRequest.edit.allow_tensor_updates/open_weights_writable",
                PERMISSION_DENIED,
            ));
        }
        if request.trainable_tensors.is_empty() {
            return Err(err(
                "TrainStep requires trainable_tensors",
                INVALID_ARGUMENT,
            ));
        }

        let accumulation = request.accumulation.clone().unwrap_or_default();
        if accumulation.steps < 0 {
            return Err(err(
                "TrainStep accumulation.steps must be non-negative",
                INVALID_ARGUMENT,
            ));
        }
        let accumulation_steps = accumulation.steps.max(1);
        let pending_before = if accumulation.reset {
            0
        } else {
            inner.training_accumulation_microbatches
        };
        if pending_before > 0 && accumulation_steps != inner.training_accumulation_steps {
            return Err(err(
                format!(
                    "TrainStep accumulation.steps must remain {} until the pending window is applied or reset",
                    inner.training_accumulation_steps
                ),
                FAILED_PRECONDITION,
            ));
        }
        let losses = prepare_cross_entropy_losses(&request, accumulation_steps)?;

        let update_mode_value = request
            .update_mode
            .or_else(|| {
                (pending_before > 0)
                    .then_some(inner.training_accumulation_update_mode)
                    .flatten()
            })
            .or(inner.training_update_mode)
            .unwrap_or(TensorUpdateMode::TensorUpdateAdamw as i32);
        let update_mode = TensorUpdateMode::try_from(update_mode_value).map_err(|_| {
            err(
                format!("TrainStep has invalid update_mode {update_mode_value}"),
                INVALID_ARGUMENT,
            )
        })?;
        if !matches!(
            update_mode,
            TensorUpdateMode::TensorUpdateSgd | TensorUpdateMode::TensorUpdateAdamw
        ) {
            return Err(err(
                "TrainStep supports SGD or AdamW update_mode",
                INVALID_ARGUMENT,
            ));
        }
        let native_update_mode = tensor_update_mode_to_native(update_mode);
        let mut seen_trainables = std::collections::HashSet::new();
        let mut trainable_names = Vec::with_capacity(request.trainable_tensors.len());
        let mut updated_tensors = Vec::with_capacity(request.trainable_tensors.len());
        for name in &request.trainable_tensors {
            if !seen_trainables.insert(name.as_str()) {
                return Err(err(
                    format!("TrainStep trainable tensor {name} is duplicated"),
                    INVALID_ARGUMENT,
                ));
            }
            let c_name = CString::new(name.as_str())
                .map_err(|_| err("bad trainable tensor name", INVALID_ARGUMENT))?;
            let spec = unsafe {
                tensor_spec_native(name, TENSOR_ACCESS_READABLE | TENSOR_ACCESS_WRITABLE)?
            };
            if spec.dtype != DATA_TYPE_F32 {
                return Err(err(
                    format!(
                        "TrainStep trainable tensor {name} must be F32, got {}",
                        dtype_name(spec.dtype)
                    ),
                    INVALID_ARGUMENT,
                ));
            }
            if unsafe { volvoxai_engine_is_model_weight(c_name.as_ptr()) } != 1 {
                return Err(err(
                    format!(
                        "TrainStep trainable tensor {name} must be a safetensors-backed model weight"
                    ),
                    INVALID_ARGUMENT,
                ));
            }
            trainable_names.push(c_name);
            updated_tensors.push(spec);
        }
        let optimizer = request
            .optimizer
            .clone()
            .or_else(|| {
                (pending_before > 0)
                    .then(|| inner.training_accumulation_optimizer.clone())
                    .flatten()
            })
            .or_else(|| inner.training_optimizer.clone());
        let fallback_step = if pending_before > 0 {
            inner
                .training_accumulation_optimizer_step
                .ok_or_else(|| err("pending optimizer step is unavailable", INTERNAL))?
        } else {
            inner
                .training_step
                .checked_add(1)
                .ok_or_else(|| err("TrainStep optimizer step overflow", FAILED_PRECONDITION))?
        };
        let (lr, beta1, beta2, eps, weight_decay, max_grad_norm, step) =
            optimizer_values(None, optimizer.as_ref(), fallback_step)?;
        let explicit_step = request.optimizer.as_ref().and_then(|options| options.step);
        if explicit_step.is_some_and(|step| step <= inner.training_step) {
            return Err(err(
                format!(
                    "TrainStep explicit optimizer step must be greater than {}",
                    inner.training_step
                ),
                FAILED_PRECONDITION,
            ));
        }
        if pending_before > 0
            && inner
                .training_accumulation_optimizer_step
                .is_some_and(|pending_step| pending_step != step)
        {
            return Err(err(
                "TrainStep optimizer step cannot change inside an accumulation window",
                FAILED_PRECONDITION,
            ));
        }
        let next_microbatch = pending_before
            .checked_add(1)
            .ok_or_else(|| err("TrainStep accumulation counter overflow", FAILED_PRECONDITION))?;
        let predicted_apply = accumulation.flush || next_microbatch >= accumulation_steps;
        if request.persist_safetensors && !predicted_apply {
            return Err(err(
                "TrainStep cannot persist weights before the accumulation window applies; persist on the final or flushed microbatch",
                FAILED_PRECONDITION,
            ));
        }
        let persist = request.persist_safetensors;
        let flush_paths = request.weight_paths.clone();

        unsafe {
            for t in &request.inputs {
                copy_tensor_to_input(t)?;
            }
            let trainable_ptrs: Vec<*const c_char> =
                trainable_names.iter().map(|s| s.as_ptr()).collect();
            let native_losses: Vec<NativeCrossEntropyLoss> =
                losses.iter().map(PreparedCrossEntropyLoss::native).collect();
            let mut loss = 0.0f32;
            let mut native_metrics = vec![NativeCrossEntropyMetric::default(); losses.len()];
            let mut accumulated_microbatches: c_int = 0;
            let mut update_applied: c_int = 0;
            if accumulation.reset {
                inner.training_accumulation_microbatches = 0;
                inner.training_accumulation_steps = 1;
                inner.training_accumulation_optimizer_step = None;
                inner.training_accumulation_optimizer = None;
                inner.training_accumulation_update_mode = None;
            }
            if volvoxai_engine_train_step_multi(
                native_losses.as_ptr(),
                native_losses.len() as c_int,
                trainable_ptrs.as_ptr(),
                trainable_ptrs.len() as c_int,
                native_update_mode,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                max_grad_norm,
                step as c_long,
                accumulation_steps,
                c_int::from(accumulation.flush),
                c_int::from(accumulation.reset),
                &mut loss,
                native_metrics.as_mut_ptr(),
                &mut accumulated_microbatches,
                &mut update_applied,
            ) != 0
            {
                return Err(err(
                    "TrainStep failed; check loss descriptors, accumulation consistency, trainable tensor names, and requested-backend backward support",
                    INVALID_ARGUMENT,
                ));
            }
            let used_training_backend = volvoxai_engine_last_training_backend();
            let had_contribution = losses.iter().zip(native_metrics.iter()).any(
                |(descriptor, metric)| descriptor.weight > 0.0 && metric.examples > 0,
            );
            if native_backend_policy.required != 0 && had_contribution &&
                used_training_backend != native_backend_policy.required {
                return Err(err(
                    "native TrainStep completed without the explicitly requested GPU backend",
                    INTERNAL,
                ));
            }
            let metric_examples: i64 = native_metrics
                .iter()
                .map(|metric| i64::from(metric.examples))
                .sum();
            let metrics_valid = native_metrics.iter().all(|metric| {
                metric.loss.is_finite()
                    && metric.correct >= 0
                    && metric.examples >= 0
                    && metric.correct <= metric.examples
                    && metric.normalizer.is_finite()
                    && metric.normalizer >= 0.0
            });
            if !loss.is_finite()
                || !metrics_valid
                || accumulated_microbatches < 0
                || accumulated_microbatches > accumulation_steps
                || (accumulated_microbatches == 0
                    && (!predicted_apply || metric_examples != 0))
                || (update_applied != 0 && update_applied != 1)
            {
                return Err(err(
                    "TrainStep native accumulation result is invalid",
                    INTERNAL,
                ));
            }
            let update_applied = update_applied == 1;
            let canonical_optimizer = OptimizerOptions {
                learning_rate: lr,
                beta1: Some(beta1),
                beta2: Some(beta2),
                epsilon: Some(eps),
                weight_decay: Some(weight_decay),
                max_grad_norm: Some(max_grad_norm),
                step: None,
            };
            // The native call has committed weights and optimizer moments. A
            // later file-persistence error cannot roll that live update back,
            // so keep the runtime step synchronized with native state.
            let committed_step = if update_applied {
                inner.training_step = step;
                inner.training_optimizer = Some(canonical_optimizer.clone());
                inner.training_update_mode = Some(update_mode as i32);
                inner.training_accumulation_microbatches = 0;
                inner.training_accumulation_steps = 1;
                inner.training_accumulation_optimizer_step = None;
                inner.training_accumulation_optimizer = None;
                inner.training_accumulation_update_mode = None;
                step
            } else if predicted_apply {
                // An all-ignore or zero-weight window closes without invoking
                // the optimizer, preserving the previous immediate-step
                // behavior and avoiding a permanently pending empty window.
                inner.training_accumulation_microbatches = 0;
                inner.training_accumulation_steps = 1;
                inner.training_accumulation_optimizer_step = None;
                inner.training_accumulation_optimizer = None;
                inner.training_accumulation_update_mode = None;
                updated_tensors.clear();
                inner.training_step
            } else {
                inner.training_accumulation_microbatches = accumulated_microbatches;
                inner.training_accumulation_steps = accumulation_steps;
                inner.training_accumulation_optimizer_step = Some(step);
                inner.training_accumulation_optimizer = Some(canonical_optimizer);
                inner.training_accumulation_update_mode = Some(update_mode as i32);
                updated_tensors.clear();
                inner.training_step
            };

            let weights = if persist {
                let inspect = native_inspection(false, false, true, false)?;
                let weight_count = inspect
                    .get("weights")
                    .and_then(Value::as_array)
                    .map(|a| a.len())
                    .unwrap_or(0);
                for i in 0..weight_count {
                    let override_path = flush_paths.get(i).cloned().unwrap_or_default();
                    let c_path = if override_path.is_empty() {
                        None
                    } else {
                        Some(
                            CString::new(override_path.as_str())
                                .map_err(|_| err("bad flush path", INVALID_ARGUMENT))?,
                        )
                    };
                    let ptr = c_path
                        .as_ref()
                        .map(|s| s.as_ptr())
                        .unwrap_or(std::ptr::null());
                    if volvoxai_engine_save_weight_file(i as c_int, ptr) != 0 {
                        return Err(err(format!("flush weight file {i} failed"), INTERNAL));
                    }
                }
                let post = native_inspection(false, false, true, false)?;
                post.get("weights")
                    .and_then(Value::as_array)
                    .map(|items| items.iter().map(safetensors_info_from_value).collect())
                    .unwrap_or_default()
            } else {
                Vec::new()
            };

            let loss_metrics: Vec<CrossEntropyLossMetric> = losses
                .iter()
                .zip(native_metrics)
                .map(|(descriptor, metric)| CrossEntropyLossMetric {
                    name: descriptor.name.clone(),
                    loss: metric.loss,
                    correct: metric.correct,
                    examples: metric.examples,
                    normalizer: metric.normalizer,
                })
                .collect();
            let correct = loss_metrics.iter().try_fold(0i32, |sum, metric| {
                sum.checked_add(metric.correct)
                    .ok_or_else(|| err("TrainStep correct metric overflow", INTERNAL))
            })?;
            let examples = loss_metrics.iter().try_fold(0i32, |sum, metric| {
                sum.checked_add(metric.examples)
                    .ok_or_else(|| err("TrainStep example metric overflow", INTERNAL))
            })?;

            Ok(TrainStepResponse {
                loss,
                correct,
                examples,
                updated_tensors,
                weight_files: weights,
                step: committed_step,
                losses: loss_metrics,
                accumulated_microbatches,
                accumulation_steps,
                update_applied,
            })
        }
    }
}

struct AdapterRouteGuard;

impl Drop for AdapterRouteGuard {
    fn drop(&mut self) {
        unsafe { volvoxai_engine_adapter_route_end() }
    }
}

fn adapter_kind_name(kind: i32) -> Option<&'static str> {
    match AdapterKind::try_from(kind).ok()? {
        AdapterKind::AdapterLora => Some("lora"),
        AdapterKind::Unspecified => None,
    }
}

fn adapter_kind_from_value(value: Option<&Value>) -> i32 {
    match value.and_then(Value::as_str) {
        Some("lora") => AdapterKind::AdapterLora as i32,
        _ => AdapterKind::Unspecified as i32,
    }
}

fn adapter_layout_name(layout: i32) -> Option<&'static str> {
    match AdapterMatrixLayout::try_from(layout).ok()? {
        AdapterMatrixLayout::Peft => Some("peft"),
        AdapterMatrixLayout::Canonical => Some("din_r_r_dout"),
        AdapterMatrixLayout::Unspecified => None,
    }
}

fn adapter_layout_from_value(value: Option<&Value>) -> i32 {
    match value.and_then(Value::as_str).unwrap_or("") {
        "peft" => AdapterMatrixLayout::Peft as i32,
        "din_r_r_dout" => AdapterMatrixLayout::Canonical as i32,
        _ => AdapterMatrixLayout::Unspecified as i32,
    }
}

fn adapter_role_json_key(role: i32) -> Result<&'static str, FfiError> {
    match AdapterTensorRole::try_from(role) {
        Ok(AdapterTensorRole::AdapterTensorA) => Ok("a"),
        Ok(AdapterTensorRole::AdapterTensorB) => Ok("b"),
        Ok(AdapterTensorRole::AdapterTensorUnspecified) | Err(_) => Err(err(
            format!("adapter tensor role {role} is not supported by {ADAPTER_MANIFEST_FORMAT}"),
            INVALID_ARGUMENT,
        )),
    }
}

fn adapter_role_from_key(key: &str) -> i32 {
    match key {
        "a" => AdapterTensorRole::AdapterTensorA as i32,
        "b" => AdapterTensorRole::AdapterTensorB as i32,
        _ => AdapterTensorRole::AdapterTensorUnspecified as i32,
    }
}

fn validate_adapter_manifest(
    manifest: &AdapterManifest,
    tensors: Option<&[Tensor]>,
) -> Result<(), FfiError> {
    if manifest.format != ADAPTER_MANIFEST_FORMAT {
        return Err(err(
            format!("adapter manifest format must be {ADAPTER_MANIFEST_FORMAT}"),
            INVALID_ARGUMENT,
        ));
    }
    if string_entry_value(&manifest.metadata, "volvox.optimizer_step")
        .map_or(false, |step| {
            step.parse::<i64>().map_or(true, |step| step < 0)
        })
    {
        return Err(err(
            "volvox.optimizer_step metadata must be a non-negative int64 string",
            INVALID_ARGUMENT,
        ));
    }
    if AdapterKind::try_from(manifest.kind) != Ok(AdapterKind::AdapterLora) {
        return Err(err(
            "only the ADAPTER_LORA kind is supported",
            INVALID_ARGUMENT,
        ));
    }
    if manifest.targets.is_empty() {
        return Err(err(
            "adapter manifest requires at least one target",
            INVALID_ARGUMENT,
        ));
    }

    let payload_by_name: std::collections::HashMap<&str, &Tensor> = tensors
        .unwrap_or_default()
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect();
    if tensors.map_or(false, |items| payload_by_name.len() != items.len()) {
        return Err(err("adapter tensor names must be unique", INVALID_ARGUMENT));
    }
    let mut bound_names = std::collections::HashSet::new();
    for target in &manifest.targets {
        if target.base_tensor.is_empty()
            || target.rank <= 0
            || !target.alpha.is_finite()
            || target.alpha <= 0.0
        {
            return Err(err(
                format!(
                    "adapter target {} has invalid weight/rank/alpha",
                    target.target_id
                ),
                INVALID_ARGUMENT,
            ));
        }
        if adapter_layout_name(target.adapter_layout).is_none() {
            return Err(err(
                format!(
                    "adapter target {} requires an explicit supported layout",
                    target.target_id
                ),
                INVALID_ARGUMENT,
            ));
        }
        if !matches!(
            AdapterMatrixLayout::try_from(target.adapter_layout),
            Ok(AdapterMatrixLayout::Peft | AdapterMatrixLayout::Canonical)
        ) {
            return Err(err(
                format!(
                    "adapter target {} layout is reserved until Conv-LoRA support is implemented",
                    target.target_id
                ),
                INVALID_ARGUMENT,
            ));
        }
        if let Some(scale) = target.scale {
            if !scale.is_finite() {
                return Err(err("adapter scale must be finite", INVALID_ARGUMENT));
            }
        }
        let mut roles = std::collections::HashSet::new();
        for binding in &target.tensors {
            let role = adapter_role_json_key(binding.role)?;
            if !roles.insert(role.to_string()) || binding.tensor_name.is_empty() {
                return Err(err(
                    format!(
                        "adapter target {} has duplicate/empty tensor bindings",
                        target.target_id
                    ),
                    INVALID_ARGUMENT,
                ));
            }
            if !bound_names.insert(binding.tensor_name.clone()) {
                return Err(err(
                    format!(
                        "adapter tensor {} is bound more than once",
                        binding.tensor_name
                    ),
                    INVALID_ARGUMENT,
                ));
            }
            if let Some(items) = tensors {
                let tensor = payload_by_name
                    .get(binding.tensor_name.as_str())
                    .ok_or_else(|| {
                        err(
                            format!("adapter tensor payload {} is missing", binding.tensor_name),
                            INVALID_ARGUMENT,
                        )
                    })?;
                let spec = binding.spec.as_ref().ok_or_else(|| {
                    err(
                        format!(
                            "adapter tensor {} requires an explicit spec",
                            binding.tensor_name
                        ),
                        INVALID_ARGUMENT,
                    )
                })?;
                if spec.name != binding.tensor_name
                    || spec.shape != tensor.shape
                    || spec.dtype != tensor.dtype
                {
                    return Err(err(
                        format!(
                            "adapter tensor {} does not match its binding spec",
                            binding.tensor_name
                        ),
                        INVALID_ARGUMENT,
                    ));
                }
                if tensor.dtype != DATA_TYPE_F32 && tensor.dtype != DATA_TYPE_F16 {
                    return Err(err(
                        format!("adapter tensor {} must be F32 or F16", tensor.name),
                        INVALID_ARGUMENT,
                    ));
                }
                let want = tensor_nbytes(tensor.dtype, &tensor.shape)?;
                if tensor.data.len() != want || spec.size_bytes != want as i64 {
                    return Err(err(
                        format!(
                            "adapter tensor {} byte size does not match its spec",
                            tensor.name
                        ),
                        INVALID_ARGUMENT,
                    ));
                }
                let _ = items;
            }
        }
        if !roles.contains("a") || !roles.contains("b") {
            return Err(err(
                format!(
                    "adapter target {} requires A and B tensors",
                    target.target_id
                ),
                INVALID_ARGUMENT,
            ));
        }
    }
    if let Some(items) = tensors {
        if bound_names.len() != items.len() {
            return Err(err(
                "adapter payload contains unbound tensors",
                INVALID_ARGUMENT,
            ));
        }
    }
    Ok(())
}

fn adapter_manifest_json(
    manifest: &AdapterManifest,
    native_name: &str,
    version_id: &str,
) -> Result<String, FfiError> {
    let kind = adapter_kind_name(manifest.kind)
        .ok_or_else(|| err("adapter kind is required", INVALID_ARGUMENT))?;
    let mut targets = Vec::with_capacity(manifest.targets.len());
    for target in &manifest.targets {
        let default_scale = target.alpha / target.rank as f32;
        let mut value = serde_json::Map::new();
        value.insert("id".into(), json!(target.target_id));
        value.insert("op".into(), json!(target.op_id));
        value.insert("weight".into(), json!(target.base_tensor));
        value.insert("kind".into(), json!(kind));
        value.insert(
            "layout".into(),
            json!(adapter_layout_name(target.adapter_layout).unwrap_or("")),
        );
        value.insert("rank".into(), json!(target.rank));
        value.insert("alpha".into(), json!(target.alpha));
        value.insert("scale".into(), json!(target.scale.unwrap_or(default_scale)));
        let mut specs = Vec::with_capacity(target.tensors.len());
        for binding in &target.tensors {
            let key = adapter_role_json_key(binding.role)?;
            value.insert(key.to_string(), json!(binding.tensor_name));
            let spec = binding.spec.clone().unwrap_or_default();
            specs.push(json!({
                "role": key,
                "name": binding.tensor_name,
                "shape": spec.shape,
                "dtype": dtype_name(spec.dtype),
            }));
        }
        value.insert("tensors".into(), Value::Array(specs));
        targets.push(Value::Object(value));
    }
    serde_json::to_string(&json!({
        "format": ADAPTER_MANIFEST_FORMAT,
        "name": native_name,
        "adapter_id": manifest.adapter_id,
        "version_id": version_id,
        "kind": kind,
        "targets": targets,
        "metadata": string_entries_to_map(&manifest.metadata),
    }))
    .map_err(|e| err(format!("serialize adapter manifest failed: {e}"), INTERNAL))
}

fn validate_lora_base_dtype(dtype: c_int, tensor_name: &str) -> Result<(), FfiError> {
    if dtype != T_F32 && dtype != T_F16 {
        return Err(err(
            format!("LoRA base tensor {tensor_name} must use F32/F16 storage"),
            INVALID_ARGUMENT,
        ));
    }
    Ok(())
}

unsafe fn validate_adapter_targets_against_model(
    manifest: &AdapterManifest,
) -> Result<(), FfiError> {
    let mut weights = std::collections::HashSet::new();
    for target in &manifest.targets {
        if !matches!(
            AdapterMatrixLayout::try_from(target.adapter_layout),
            Ok(AdapterMatrixLayout::Peft | AdapterMatrixLayout::Canonical)
        ) {
            return Err(err(
                format!(
                    "adapter target {} uses a layout not implemented by the linear runtime",
                    target.target_id
                ),
                INVALID_ARGUMENT,
            ));
        }
        if !weights.insert(target.base_tensor.as_str()) {
            return Err(err(
                format!(
                    "adapter manifest targets base tensor {} more than once",
                    target.base_tensor
                ),
                INVALID_ARGUMENT,
            ));
        }
        let name = CString::new(target.base_tensor.as_str())
            .map_err(|_| err("adapter base tensor name contains NUL", INVALID_ARGUMENT))?;
        let mut shape = [0 as c_int; 8];
        let mut ndim = 0;
        let mut dtype = 0;
        if volvoxai_engine_tensor_info_ex(
            name.as_ptr(),
            std::ptr::null_mut(),
            shape.as_mut_ptr(),
            &mut ndim,
            &mut dtype,
            std::ptr::null_mut(),
        ) != 0
        {
            return Err(err(
                format!("adapter base tensor {} does not exist", target.base_tensor),
                INVALID_ARGUMENT,
            ));
        }
        if ndim != 2 || shape[0] <= 0 || shape[1] <= 0 {
            return Err(err(
                format!(
                    "adapter base tensor {} must be a non-empty rank-2 linear weight",
                    target.base_tensor
                ),
                INVALID_ARGUMENT,
            ));
        }
        validate_lora_base_dtype(dtype, &target.base_tensor)?;

        let mut d_in = 0;
        let mut d_out = 0;
        let mut base_out_in = 0;
        if volvoxai_engine_linear_weight_layout(
            name.as_ptr(),
            &mut d_in,
            &mut d_out,
            &mut base_out_in,
        ) != 0
            || d_in <= 0
            || d_out <= 0
        {
            return Err(err(
                format!(
                    "adapter base tensor {} is not used by a supported linear node",
                    target.base_tensor
                ),
                INVALID_ARGUMENT,
            ));
        }
        let d_in = d_in as i64;
        let d_out = d_out as i64;
        for binding in &target.tensors {
            let spec = binding.spec.as_ref().ok_or_else(|| {
                err(
                    format!("adapter tensor {} is missing its spec", binding.tensor_name),
                    INVALID_ARGUMENT,
                )
            })?;
            if spec.dtype != DATA_TYPE_F32 && spec.dtype != DATA_TYPE_F16 {
                return Err(err(
                    format!("adapter tensor {} must be F32/F16", binding.tensor_name),
                    INVALID_ARGUMENT,
                ));
            }
            let role = AdapterTensorRole::try_from(binding.role).ok();
            let layout = AdapterMatrixLayout::try_from(target.adapter_layout).ok();
            let expected_shape = match (role, layout) {
                (Some(AdapterTensorRole::AdapterTensorA), Some(AdapterMatrixLayout::Peft)) => {
                    vec![target.rank as i64, d_in]
                }
                (
                    Some(AdapterTensorRole::AdapterTensorA),
                    Some(AdapterMatrixLayout::Canonical),
                ) => vec![d_in, target.rank as i64],
                (Some(AdapterTensorRole::AdapterTensorB), Some(AdapterMatrixLayout::Peft)) => {
                    vec![d_out, target.rank as i64]
                }
                (
                    Some(AdapterTensorRole::AdapterTensorB),
                    Some(AdapterMatrixLayout::Canonical),
                ) => vec![target.rank as i64, d_out],
                _ => {
                    return Err(err(
                        format!("unsupported LoRA tensor role {}", binding.role),
                        INVALID_ARGUMENT,
                    ));
                }
            };
            if spec.shape != expected_shape {
                return Err(err(
                    format!(
                        "adapter tensor {} shape {:?} does not match {:?}",
                        binding.tensor_name, spec.shape, expected_shape
                    ),
                    INVALID_ARGUMENT,
                ));
            }
        }
    }
    Ok(())
}

fn apply_adapter_scale_default(manifest: &mut AdapterManifest) {
    for target in &mut manifest.targets {
        if target.scale.is_none() && target.rank > 0 && target.alpha.is_finite() {
            target.scale = Some(target.alpha / target.rank as f32);
        }
    }
}

fn normalize_adapter_manifest(mut manifest: AdapterManifest) -> Result<AdapterManifest, FfiError> {
    apply_adapter_scale_default(&mut manifest);
    for target in &mut manifest.targets {
        let source_layout = AdapterMatrixLayout::try_from(target.adapter_layout).map_err(|_| {
            err(
                format!("adapter target {} has an invalid layout", target.target_id),
                INVALID_ARGUMENT,
            )
        })?;
        target.adapter_layout = AdapterMatrixLayout::Canonical as i32;
        for binding in &mut target.tensors {
            let spec = binding.spec.as_mut().ok_or_else(|| {
                err(
                    format!("adapter tensor {} is missing its spec", binding.tensor_name),
                    INVALID_ARGUMENT,
                )
            })?;
            let mut shape = spec.shape.clone();
            if source_layout == AdapterMatrixLayout::Peft {
                shape.swap(0, 1);
            }
            let elements = shape
                .iter()
                .try_fold(1i64, |n, dim| n.checked_mul(*dim))
                .ok_or_else(|| err("normalized adapter tensor is too large", INVALID_ARGUMENT))?;
            spec.name = binding.tensor_name.clone();
            spec.shape = shape;
            spec.dtype = DATA_TYPE_F32;
            spec.access_flags = TENSOR_ACCESS_READABLE;
            spec.size_bytes = elements
                .checked_mul(4)
                .ok_or_else(|| err("normalized adapter tensor is too large", INVALID_ARGUMENT))?;
        }
    }
    Ok(manifest)
}

fn adapter_manifest_resident_bytes(manifest: &AdapterManifest) -> i64 {
    let mut names = std::collections::HashSet::new();
    manifest
        .targets
        .iter()
        .flat_map(|target| &target.tensors)
        .filter(|binding| names.insert(binding.tensor_name.as_str()))
        .filter_map(|binding| binding.spec.as_ref().map(|spec| spec.size_bytes))
        .sum()
}

fn adapter_manifests_equal(left: &AdapterManifest, right: &AdapterManifest) -> bool {
    if string_entries_to_map(&left.metadata) != string_entries_to_map(&right.metadata) {
        return false;
    }
    let mut left = left.clone();
    let mut right = right.clone();
    left.metadata.clear();
    right.metadata.clear();
    left == right
}

unsafe fn native_stage_adapter(
    manifest: &AdapterManifest,
    native_name: &str,
    version_id: &str,
    tensors: &[Tensor],
) -> Result<(), FfiError> {
    let manifest_json = adapter_manifest_json(manifest, native_name, version_id)?;
    let c_manifest = CString::new(manifest_json)
        .map_err(|_| err("adapter manifest contains NUL", INVALID_ARGUMENT))?;
    let names: Vec<CString> = tensors
        .iter()
        .map(|tensor| CString::new(tensor.name.as_str()))
        .collect::<Result<_, _>>()
        .map_err(|_| err("adapter tensor name contains NUL", INVALID_ARGUMENT))?;
    let name_ptrs: Vec<*const c_char> = names.iter().map(|name| name.as_ptr()).collect();
    let data_ptrs: Vec<*const c_void> = tensors
        .iter()
        .map(|tensor| tensor.data.as_ptr() as *const c_void)
        .collect();
    let dtypes: Vec<c_int> = tensors
        .iter()
        .map(|tensor| {
            proto_to_native_dtype(tensor.dtype).ok_or_else(|| {
                err(
                    format!("unsupported adapter dtype {}", dtype_name(tensor.dtype)),
                    INVALID_ARGUMENT,
                )
            })
        })
        .collect::<Result<_, _>>()?;
    let sizes: Vec<usize> = tensors.iter().map(|tensor| tensor.data.len()).collect();
    if volvoxai_engine_adapter_stage_json(
        c_manifest.as_ptr(),
        name_ptrs.as_ptr(),
        data_ptrs.as_ptr(),
        dtypes.as_ptr(),
        sizes.as_ptr(),
        tensors.len() as c_int,
    ) != 0
    {
        return Err(err("native adapter staging failed", INVALID_ARGUMENT));
    }
    Ok(())
}

fn validate_checkpoint_tensor_descriptors(
    target: &Value,
    tensor_info: &std::collections::HashMap<&str, &SafetensorsTensorInfo>,
) -> Result<(), FfiError> {
    let Some(descriptors) = target.get("tensors") else {
        return Ok(());
    };
    let descriptors = descriptors.as_array().ok_or_else(|| {
        err(
            "LoRA target tensors descriptor must be an array",
            INVALID_ARGUMENT,
        )
    })?;
    if descriptors.len() != 2 {
        return Err(err(
            "LoRA target tensors descriptor must contain exactly A and B",
            INVALID_ARGUMENT,
        ));
    }

    let mut roles = std::collections::HashSet::new();
    for descriptor in descriptors {
        let object = descriptor
            .as_object()
            .ok_or_else(|| err("LoRA tensor descriptor must be an object", INVALID_ARGUMENT))?;
        if object
            .keys()
            .any(|key| !matches!(key.as_str(), "role" | "name" | "shape" | "dtype"))
        {
            return Err(err(
                "LoRA tensor descriptor contains an unsupported field",
                INVALID_ARGUMENT,
            ));
        }
        let role = descriptor
            .get("role")
            .and_then(Value::as_str)
            .filter(|role| *role == "a" || *role == "b")
            .ok_or_else(|| {
                err(
                    "LoRA tensor descriptor role must be a or b",
                    INVALID_ARGUMENT,
                )
            })?;
        if !roles.insert(role) {
            return Err(err(
                format!("LoRA tensor descriptor role {role} is duplicated"),
                INVALID_ARGUMENT,
            ));
        }
        let expected_name = target.get(role).and_then(Value::as_str).ok_or_else(|| {
            err(
                format!("LoRA target is missing tensor {role}"),
                INVALID_ARGUMENT,
            )
        })?;
        let name = descriptor
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| err("LoRA tensor descriptor name is required", INVALID_ARGUMENT))?;
        if name != expected_name {
            return Err(err(
                format!("LoRA tensor descriptor {role} name does not match its binding"),
                INVALID_ARGUMENT,
            ));
        }
        let info = tensor_info.get(expected_name).ok_or_else(|| {
            err(
                format!("adapter manifest references missing tensor {expected_name}"),
                INVALID_ARGUMENT,
            )
        })?;
        let shape = descriptor
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| err("LoRA tensor descriptor shape is required", INVALID_ARGUMENT))?;
        if shape.len() != 2
            || shape
                .iter()
                .map(Value::as_i64)
                .collect::<Option<Vec<_>>>()
                .as_deref()
                != Some(info.shape.as_slice())
        {
            return Err(err(
                format!("LoRA tensor descriptor {role} shape does not match its payload"),
                INVALID_ARGUMENT,
            ));
        }
        let dtype = descriptor
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| err("LoRA tensor descriptor dtype is required", INVALID_ARGUMENT))?;
        if dtype != dtype_name(info.dtype) {
            return Err(err(
                format!("LoRA tensor descriptor {role} dtype does not match its payload"),
                INVALID_ARGUMENT,
            ));
        }
    }
    if !roles.contains("a") || !roles.contains("b") {
        return Err(err(
            "LoRA target tensors descriptor must contain one A and one B",
            INVALID_ARGUMENT,
        ));
    }
    Ok(())
}

fn manifest_from_json(text: &str, file: &SafetensorsInfo) -> Result<AdapterManifest, FfiError> {
    let value: Value = serde_json::from_str(text).map_err(|e| {
        err(
            format!("invalid adapter manifest JSON: {e}"),
            INVALID_ARGUMENT,
        )
    })?;
    let format = value.get("format").and_then(Value::as_str).unwrap_or("");
    if format != ADAPTER_MANIFEST_FORMAT {
        return Err(err(
            format!("adapter manifest format must be {ADAPTER_MANIFEST_FORMAT}"),
            INVALID_ARGUMENT,
        ));
    }
    let kind = adapter_kind_from_value(value.get("kind"));
    if kind != 1 {
        return Err(err(
            "adapter checkpoint kind must be lora",
            INVALID_ARGUMENT,
        ));
    }
    let tensor_info: std::collections::HashMap<&str, &SafetensorsTensorInfo> =
        file.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    let targets = value
        .get("targets")
        .and_then(Value::as_array)
        .ok_or_else(|| err("adapter manifest targets are missing", INVALID_ARGUMENT))?
        .iter()
        .map(|target| {
            let object = target
                .as_object()
                .ok_or_else(|| err("adapter target must be an object", INVALID_ARGUMENT))?;
            for key in object.keys() {
                if !matches!(
                    key.as_str(),
                    "id" | "op"
                        | "weight"
                        | "kind"
                        | "layout"
                        | "rank"
                        | "alpha"
                        | "scale"
                        | "a"
                        | "b"
                        | "tensors"
                ) {
                    return Err(err(
                        format!("unsupported LoRA target field {key}"),
                        INVALID_ARGUMENT,
                    ));
                }
            }
            if let Some(target_kind) = target.get("kind") {
                let target_kind = adapter_kind_from_value(Some(target_kind));
                if target_kind != kind {
                    return Err(err(
                        "adapter target kind must match the manifest kind",
                        INVALID_ARGUMENT,
                    ));
                }
            }
            validate_checkpoint_tensor_descriptors(target, &tensor_info)?;
            let mut tensors = Vec::new();
            for key in ["a", "b"] {
                let Some(name) = target.get(key).and_then(Value::as_str) else {
                    continue;
                };
                let info = tensor_info.get(name).ok_or_else(|| {
                    err(
                        format!("adapter manifest references missing tensor {name}"),
                        INVALID_ARGUMENT,
                    )
                })?;
                let role = adapter_role_from_key(key);
                tensors.push(AdapterTensorBinding {
                    role,
                    tensor_name: name.to_string(),
                    spec: Some(TensorSpec {
                        name: name.to_string(),
                        shape: info.shape.clone(),
                        dtype: info.dtype,
                        access_flags: TENSOR_ACCESS_READABLE,
                        size_bytes: info.data_end - info.data_start,
                    }),
                });
            }
            Ok(AdapterTarget {
                target_id: target
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                op_id: target
                    .get("op")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                base_tensor: target
                    .get("weight")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                adapter_layout: adapter_layout_from_value(target.get("layout")),
                rank: target.get("rank").and_then(Value::as_i64).unwrap_or(0) as i32,
                alpha: target.get("alpha").and_then(Value::as_f64).unwrap_or(0.0) as f32,
                scale: target
                    .get("scale")
                    .and_then(Value::as_f64)
                    .map(|x| x as f32),
                tensors,
            })
        })
        .collect::<Result<Vec<_>, FfiError>>()?;
    Ok(AdapterManifest {
        format: format.to_string(),
        adapter_id: value
            .get("adapter_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        kind,
        targets,
        metadata: string_entries_from_map(strict_metadata_map(
            value.get("metadata"),
            "adapter manifest",
        )?),
    })
}

fn adapter_state(inner: &Inner) -> AdapterState {
    AdapterState {
        active: Some(inner.active_adapters.clone()),
        merged: inner.merged_adapter.clone(),
        generation: inner.adapter_generation,
    }
}

fn adapter_ref_eq(a: &AdapterVersionRef, b: &AdapterVersionRef) -> bool {
    a.adapter_id == b.adapter_id && a.version_id == b.version_id
}

fn adapter_record<'a>(
    inner: &'a Inner,
    reference: &AdapterVersionRef,
) -> Result<&'a AdapterRecord, FfiError> {
    inner
        .adapters
        .iter()
        .find(|record| {
            record
                .info
                .adapter
                .as_ref()
                .map_or(false, |candidate| adapter_ref_eq(candidate, reference))
        })
        .ok_or_else(|| {
            err(
                format!(
                    "unknown adapter version {}@{}",
                    reference.adapter_id, reference.version_id
                ),
                NOT_FOUND,
            )
        })
}

fn adapter_info_with_state(inner: &Inner, record: &AdapterRecord) -> AdapterInfo {
    let mut info = record.info.clone();
    if let Some(reference) = &info.adapter {
        info.active = inner.active_adapters.routes.iter().any(|route| {
            route
                .adapter
                .as_ref()
                .map_or(false, |candidate| adapter_ref_eq(candidate, reference))
        });
        info.merged = inner
            .merged_adapter
            .as_ref()
            .map_or(false, |candidate| adapter_ref_eq(candidate, reference));
    }
    info
}

fn resolve_adapter_selection(
    inner: &Inner,
    selection: &AdapterSelection,
) -> Result<Vec<(String, f32)>, FfiError> {
    selection
        .routes
        .iter()
        .map(|route| {
            let scale = route.scale.unwrap_or(1.0);
            if !scale.is_finite() {
                return Err(err("adapter route scale must be finite", INVALID_ARGUMENT));
            }
            let Some(reference) = route.adapter.as_ref() else {
                if scale != 1.0 {
                    return Err(err(
                        "a base-only adapter route must omit scale or use 1",
                        INVALID_ARGUMENT,
                    ));
                }
                return Ok((String::new(), 1.0));
            };
            Ok((adapter_record(inner, reference)?.native_name.clone(), scale))
        })
        .collect()
}

fn validate_route_batch_dims(
    selection: &AdapterSelection,
    batch_dims: &[usize],
) -> Result<(), FfiError> {
    let routes = selection.routes.len();
    if routes <= 1 {
        return Ok(());
    }
    if batch_dims.is_empty()
        || !batch_dims.iter().any(|batch| *batch == routes)
        || batch_dims
            .iter()
            .any(|batch| *batch != 1 && *batch != routes)
    {
        return Err(err(
            format!(
                "per-batch adapter route count {routes} does not match input batch dimensions {:?}",
                batch_dims
            ),
            INVALID_ARGUMENT,
        ));
    }
    Ok(())
}

unsafe fn validate_run_inputs_and_routes(
    selection: Option<&AdapterSelection>,
    inputs: &[Tensor],
) -> Result<(), FfiError> {
    let mut batch_dims = Vec::with_capacity(inputs.len());
    for tensor in inputs {
        let (_, _, actual_shape, _, _) = graph_input_destination(&tensor.name)?;
        if tensor.shape != actual_shape {
            return Err(err(
                format!(
                    "input {} shape {:?} does not match model shape {:?}",
                    tensor.name, tensor.shape, actual_shape
                ),
                INVALID_ARGUMENT,
            ));
        }
        let batch = if actual_shape.len() >= 2 {
            actual_shape[0]
        } else {
            1
        };
        if batch <= 0 {
            return Err(err(
                "input batch dimension must be positive",
                INVALID_ARGUMENT,
            ));
        }
        batch_dims.push(batch as usize);
    }
    selection.map_or(Ok(()), |selection| {
        validate_route_batch_dims(selection, &batch_dims)
    })
}

unsafe fn begin_adapter_route(
    resolved: Option<&[(String, f32)]>,
) -> Result<AdapterRouteGuard, FfiError> {
    let rc = match resolved {
        None => volvoxai_engine_adapter_route_begin(std::ptr::null()),
        Some(routes) => {
            let names: Vec<CString> = routes
                .iter()
                .map(|(name, _)| CString::new(name.as_str()))
                .collect::<Result<_, _>>()
                .map_err(|_| err("adapter name contains NUL", INVALID_ARGUMENT))?;
            let name_ptrs: Vec<*const c_char> = names.iter().map(|name| name.as_ptr()).collect();
            let scales: Vec<f32> = routes.iter().map(|(_, scale)| *scale).collect();
            volvoxai_engine_adapter_route_begin_many(
                name_ptrs.as_ptr(),
                scales.as_ptr(),
                routes.len() as c_int,
            )
        }
    };
    if rc != 0 {
        return Err(err("failed to pin adapter route", FAILED_PRECONDITION));
    }
    Ok(AdapterRouteGuard)
}

fn require(inner: &Inner, id: &str) -> Result<(), FfiError> {
    match &inner.model_id {
        Some(m) if m == id => Ok(()),
        _ => Err(err(
            format!("unknown or unloaded model_id: {id}"),
            NOT_FOUND,
        )),
    }
}

fn require_unmerged(inner: &Inner, operation: &str) -> Result<(), FfiError> {
    if inner.merged_adapter.is_some() {
        return Err(err(
            format!("{operation} is not allowed while an adapter is merged; unmerge first"),
            FAILED_PRECONDITION,
        ));
    }
    Ok(())
}

fn require_no_pending_accumulation(inner: &Inner, operation: &str) -> Result<(), FfiError> {
    if inner.training_accumulation_microbatches > 0 {
        return Err(err(
            format!(
                "{operation} is not allowed with {} pending accumulated microbatch(es); flush or reset the training window first",
                inner.training_accumulation_microbatches
            ),
            FAILED_PRECONDITION,
        ));
    }
    Ok(())
}

fn optimizer_values(
    local: Option<&OptimizerOptions>,
    base: Option<&OptimizerOptions>,
    fallback_step: i64,
) -> Result<(f32, f32, f32, f32, f32, f32, i64), FfiError> {
    let validate = |options: &OptimizerOptions, context: &str| -> Result<(), FfiError> {
        let valid = options.learning_rate.is_finite()
            && options.learning_rate >= 0.0
            && options.beta1.map_or(true, |value| value.is_finite() && (0.0..1.0).contains(&value))
            && options.beta2.map_or(true, |value| value.is_finite() && (0.0..1.0).contains(&value))
            && options.epsilon.map_or(true, |value| value.is_finite() && value > 0.0)
            && options.weight_decay.map_or(true, |value| value.is_finite() && value >= 0.0)
            && options.max_grad_norm.map_or(true, |value| value.is_finite() && value >= 0.0)
            && options.step.map_or(true, |value| value > 0);
        if !valid {
            return Err(err(
                format!(
                    "{context} requires finite learning_rate>=0, 0<=beta1/beta2<1, epsilon>0, weight_decay/max_grad_norm>=0, and a positive step"
                ),
                INVALID_ARGUMENT,
            ));
        }
        Ok(())
    };
    if let Some(options) = base {
        validate(options, "optimizer")?;
    }
    if let Some(options) = local {
        validate(options, "local optimizer")?;
    }
    if fallback_step <= 0 {
        return Err(err("optimizer fallback step must be positive", INVALID_ARGUMENT));
    }
    let learning_rate = local
        .map(|o| o.learning_rate)
        .or_else(|| base.map(|o| o.learning_rate))
        .unwrap_or(1.0e-3);
    let beta1 = local
        .and_then(|o| o.beta1)
        .or_else(|| base.and_then(|o| o.beta1))
        .unwrap_or(0.9);
    let beta2 = local
        .and_then(|o| o.beta2)
        .or_else(|| base.and_then(|o| o.beta2))
        .unwrap_or(0.999);
    let epsilon = local
        .and_then(|o| o.epsilon)
        .or_else(|| base.and_then(|o| o.epsilon))
        .unwrap_or(1.0e-8);
    let weight_decay = local
        .and_then(|o| o.weight_decay)
        .or_else(|| base.and_then(|o| o.weight_decay))
        .unwrap_or(0.0);
    let max_grad_norm = local
        .and_then(|o| o.max_grad_norm)
        .or_else(|| base.and_then(|o| o.max_grad_norm))
        .unwrap_or(0.0);
    let step = local
        .and_then(|o| o.step)
        .or_else(|| base.and_then(|o| o.step))
        .unwrap_or(fallback_step);
    if c_long::try_from(step).is_err() {
        return Err(err("optimizer step exceeds native c_long range", INVALID_ARGUMENT));
    }
    Ok((
        learning_rate,
        beta1,
        beta2,
        epsilon,
        weight_decay,
        max_grad_norm,
        step,
    ))
}

fn f32_payload(name: &str, data: &[u8]) -> Result<Vec<f32>, FfiError> {
    if data.len() % 4 != 0 {
        return Err(err(
            format!("tensor {name} F32 payload has non-multiple-of-4 byte length"),
            INVALID_ARGUMENT,
        ));
    }
    let mut out = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}

fn i32_input_tensor(name: String, shape: Vec<i64>, values: &[i32]) -> Tensor {
    Tensor {
        name,
        shape,
        dtype: DATA_TYPE_I32,
        data: values.iter().flat_map(|value| value.to_le_bytes()).collect(),
        quant: None,
        access_flags: TENSOR_ACCESS_READABLE,
        initializer: None,
    }
}

struct PreparedGraphPatch {
    node_index: i32,
    changed_index: i32,
    mode: NodePatchMode,
    patch: Value,
    changed_outputs: Vec<String>,
}

fn graph_nodes_for_simulation(root: &Value) -> Vec<Option<GraphNode>> {
    let graph = graph_from_inspection(root);
    let disabled: Vec<bool> = root
        .get("nodes")
        .and_then(Value::as_array)
        .map(|nodes| {
            nodes
                .iter()
                .map(|node| {
                    node.get("disabled")
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
                })
                .collect()
        })
        .unwrap_or_default();
    graph
        .nodes
        .into_iter()
        .enumerate()
        .map(|(index, node)| {
            if disabled.get(index).copied().unwrap_or(false) {
                None
            } else {
                Some(node)
            }
        })
        .collect()
}

fn tensor_specs_by_name(root: &Value) -> std::collections::HashMap<String, TensorSpec> {
    root.get("tensors")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(tensor_spec_from_value)
                .map(|spec| (spec.name.clone(), spec))
                .collect()
        })
        .unwrap_or_default()
}

fn validate_declared_output_names(
    names: &[String],
    produced: &std::collections::HashSet<String>,
) -> Result<(), FfiError> {
    let mut seen = std::collections::HashSet::new();
    for name in names {
        validate_native_text(name, "PatchGraph declared output", NATIVE_MAX_TENSOR_NAME)?;
        if !seen.insert(name.as_str()) {
            return Err(err(
                format!("PatchGraph declared output {name} is duplicated"),
                INVALID_ARGUMENT,
            ));
        }
        if !produced.contains(name) {
            return Err(err(
                format!("PatchGraph declared output {name} is not produced by the resulting graph"),
                INVALID_ARGUMENT,
            ));
        }
    }
    Ok(())
}

fn parse_node_patch_mode(value: Option<i32>) -> Result<NodePatchMode, FfiError> {
    match value {
        None => Ok(NodePatchMode::Merge),
        Some(value) if value == NodePatchMode::Unspecified as i32 => Ok(NodePatchMode::Merge),
        Some(value) => NodePatchMode::try_from(value).map_err(|_| {
            err(
                format!("unknown graph node patch mode: {value}"),
                INVALID_ARGUMENT,
            )
        }),
    }
}

fn prepare_graph_patches(
    root: &Value,
    patches: Vec<GraphNodePatch>,
    declared_output_names: &[String],
) -> Result<(Vec<PreparedGraphPatch>, bool), FfiError> {
    let mut nodes = graph_nodes_for_simulation(root);
    let mut tensor_specs = tensor_specs_by_name(root);
    let mut prepared = Vec::with_capacity(patches.len());
    let mut structural = false;

    for (patch_number, patch) in patches.into_iter().enumerate() {
        let patch_inputs = string_entries_to_map(&patch.inputs);
        let patch_outputs = string_entries_to_map(&patch.outputs);
        let patch_output_shapes = tensor_shape_entries_to_map(&patch.output_shapes);
        let node_index = patch.node_index.ok_or_else(|| {
            err(
                format!("PatchGraph patch {patch_number} requires node_index"),
                INVALID_ARGUMENT,
            )
        })?;
        let mode = parse_node_patch_mode(patch.mode)?;
        if node_index < -1 || (node_index == -1 && mode != NodePatchMode::InsertAfter) {
            return Err(err(
                "node_index=-1 is reserved for INSERT_AFTER append/first-node insertion",
                INVALID_ARGUMENT,
            ));
        }
        let insert = matches!(
            mode,
            NodePatchMode::InsertBefore | NodePatchMode::InsertAfter
        );
        let replace = mode == NodePatchMode::Replace;
        let target = usize::try_from(node_index).ok();
        if node_index != -1 {
            let index = target.unwrap();
            if index >= nodes.len() {
                return Err(err(
                    format!("PatchGraph node_index {node_index} is out of range"),
                    INVALID_ARGUMENT,
                ));
            }
            if !insert && nodes[index].is_none() {
                return Err(err(
                    format!("PatchGraph node_index {node_index} is deleted"),
                    FAILED_PRECONDITION,
                ));
            }
        }

        let op_name = resolve_op_name(
            patch.op,
            patch.op_name.as_deref(),
            &format!("PatchGraph patch {patch_number} op"),
            insert || replace,
        )?;
        if (insert || replace) && patch_outputs.is_empty() {
            return Err(err(
                format!("PatchGraph patch {patch_number} requires at least one output"),
                INVALID_ARGUMENT,
            ));
        }
        if mode == NodePatchMode::Delete
            && (op_name.is_some()
                || !patch_inputs.is_empty()
                || !patch_outputs.is_empty()
                || !patch_output_shapes.is_empty()
                || patch.params_json.is_some())
        {
            return Err(err(
                format!("PatchGraph DELETE patch {patch_number} must not include a node payload"),
                INVALID_ARGUMENT,
            ));
        }
        if patch_inputs.len() > NATIVE_MAX_NODE_REFS
            || patch_outputs.len() > NATIVE_MAX_NODE_REFS
        {
            return Err(err(
                format!(
                    "PatchGraph patch {patch_number} exceeds the native {NATIVE_MAX_NODE_REFS}-reference limit"
                ),
                INVALID_ARGUMENT,
            ));
        }
        let mut inputs_json = serde_json::Map::new();
        for (key, tensor_name) in &patch_inputs {
            validate_native_text(
                key,
                &format!("PatchGraph patch {patch_number} input key"),
                NATIVE_MAX_REF_KEY,
            )?;
            validate_native_text(
                tensor_name,
                &format!("PatchGraph patch {patch_number} input tensor"),
                NATIVE_MAX_TENSOR_NAME,
            )?;
            if !tensor_specs.contains_key(tensor_name) {
                return Err(err(
                    format!(
                        "PatchGraph patch {patch_number} references unavailable tensor {tensor_name}"
                    ),
                    INVALID_ARGUMENT,
                ));
            }
            inputs_json.insert(key.clone(), json!(tensor_name));
        }
        if patch_output_shapes
            .keys()
            .any(|key| !patch_outputs.contains_key(key))
        {
            return Err(err(
                format!(
                    "PatchGraph patch {patch_number} has an output shape without a matching output"
                ),
                INVALID_ARGUMENT,
            ));
        }
        let mut outputs_json = serde_json::Map::new();
        let mut shapes_json = serde_json::Map::new();
        let mut changed_outputs = Vec::new();
        if matches!(mode, NodePatchMode::Delete | NodePatchMode::Replace) {
            let previous_outputs = string_entries_to_map(
                &nodes[target.unwrap()].as_ref().unwrap().outputs,
            );
            changed_outputs.extend(previous_outputs.into_values());
        } else if mode == NodePatchMode::Merge {
            let previous_outputs = string_entries_to_map(
                &nodes[target.unwrap()].as_ref().unwrap().outputs,
            );
            changed_outputs.extend(
                patch_outputs
                    .keys()
                    .filter_map(|key| previous_outputs.get(key).cloned()),
            );
        }
        for (key, tensor_name) in &patch_outputs {
            validate_native_text(
                key,
                &format!("PatchGraph patch {patch_number} output key"),
                NATIVE_MAX_REF_KEY,
            )?;
            validate_native_text(
                tensor_name,
                &format!("PatchGraph patch {patch_number} output tensor"),
                NATIVE_MAX_TENSOR_NAME,
            )?;
            let shape = patch_output_shapes
                .get(key)
                .map(|shape| shape.dims.as_slice());
            if !tensor_specs.contains_key(tensor_name) && shape.is_none() {
                return Err(err(
                    format!(
                        "PatchGraph new output tensor {tensor_name} requires output_shapes[{key}]"
                    ),
                    INVALID_ARGUMENT,
                ));
            }
            if let Some(shape) = shape {
                let elements = validate_created_shape(
                    shape,
                    &format!("PatchGraph output tensor {tensor_name}"),
                )?;
                if let Some(existing) = tensor_specs.get(tensor_name) {
                    if existing.shape != shape {
                        return Err(err(
                            format!(
                                "PatchGraph cannot reshape existing tensor {tensor_name} from {:?} to {:?}",
                                existing.shape, shape
                            ),
                            INVALID_ARGUMENT,
                        ));
                    }
                } else {
                    tensor_specs.insert(
                        tensor_name.clone(),
                        TensorSpec {
                            name: tensor_name.clone(),
                            shape: shape.to_vec(),
                            dtype: DATA_TYPE_F32,
                            access_flags: TENSOR_ACCESS_READABLE,
                            size_bytes: elements
                                .checked_mul(std::mem::size_of::<f32>())
                                .ok_or_else(|| {
                                    err(
                                        format!("PatchGraph output {tensor_name} is too large"),
                                        INVALID_ARGUMENT,
                                    )
                                })? as i64,
                        },
                    );
                }
                shapes_json.insert(key.clone(), json!(shape));
            }
            outputs_json.insert(key.clone(), json!(tensor_name));
            changed_outputs.push(tensor_name.clone());
        }
        let params = patch
            .params_json
            .as_deref()
            .map(|text| {
                let value: Value = serde_json::from_str(text).map_err(|error| {
                    err(
                        format!("PatchGraph patch {patch_number} has invalid params_json: {error}"),
                        INVALID_ARGUMENT,
                    )
                })?;
                if !value.is_object() {
                    return Err(err(
                        format!("PatchGraph patch {patch_number} params_json must be an object"),
                        INVALID_ARGUMENT,
                    ));
                }
                Ok(value)
            })
            .transpose()?;

        let mut patch_json = serde_json::Map::new();
        if let Some(name) = &op_name {
            patch_json.insert("op".to_string(), json!(name));
        }
        if !inputs_json.is_empty() {
            patch_json.insert("inputs".to_string(), Value::Object(inputs_json));
        }
        if !outputs_json.is_empty() {
            patch_json.insert("outputs".to_string(), Value::Object(outputs_json));
        }
        if !shapes_json.is_empty() {
            patch_json.insert("output_shapes".to_string(), Value::Object(shapes_json));
        }
        if let Some(params) = &patch.params_json {
            patch_json.insert("params_json".to_string(), json!(params));
        }

        let replacement_node = GraphNode {
            index: 0,
            id: String::new(),
            op: op_name
                .as_deref()
                .map(op_type_from_name)
                .unwrap_or_default(),
            op_name: op_name.clone().unwrap_or_default(),
            inputs: string_entries_from_map(patch_inputs.clone()),
            outputs: string_entries_from_map(patch_outputs.clone()),
            output_shapes: tensor_shape_entries_from_map(patch_output_shapes.clone()),
            params_json: params.as_ref().map(Value::to_string),
        };
        let patch_is_structural = mode != NodePatchMode::Merge
            || op_name.is_some()
            || !patch_inputs.is_empty()
            || !patch_outputs.is_empty()
            || !patch_output_shapes.is_empty();
        let changed_index = match mode {
            NodePatchMode::Delete => {
                nodes.remove(target.unwrap());
                target.unwrap() as i32
            }
            NodePatchMode::InsertBefore | NodePatchMode::InsertAfter => {
                let position = if node_index == -1 {
                    nodes.len()
                } else if mode == NodePatchMode::InsertBefore {
                    target.unwrap()
                } else {
                    target.unwrap() + 1
                };
                nodes.insert(position, Some(replacement_node));
                position as i32
            }
            NodePatchMode::Replace => {
                nodes[target.unwrap()] = Some(replacement_node);
                target.unwrap() as i32
            }
            NodePatchMode::Merge => {
                let node = nodes[target.unwrap()].as_mut().unwrap();
                if let Some(name) = op_name.as_ref() {
                    node.op = op_type_from_name(name);
                    node.op_name = name.clone();
                }
                let mut inputs = string_entries_to_map(&node.inputs);
                inputs.extend(patch_inputs);
                node.inputs = string_entries_from_map(inputs);
                let mut outputs = string_entries_to_map(&node.outputs);
                outputs.extend(patch_outputs);
                node.outputs = string_entries_from_map(outputs);
                let mut output_shapes = tensor_shape_entries_to_map(&node.output_shapes);
                output_shapes.extend(patch_output_shapes);
                node.output_shapes = tensor_shape_entries_from_map(output_shapes);
                if let Some(params) = params {
                    node.params_json = Some(params.to_string());
                }
                if node.inputs.len() > NATIVE_MAX_NODE_REFS
                    || node.outputs.len() > NATIVE_MAX_NODE_REFS
                {
                    return Err(err(
                        format!(
                            "PatchGraph merge at node {node_index} exceeds native reference limits"
                        ),
                        INVALID_ARGUMENT,
                    ));
                }
                target.unwrap() as i32
            }
            NodePatchMode::Unspecified => unreachable!(),
        };
        if nodes.len() > NATIVE_MAX_NODES {
            return Err(err(
                format!("PatchGraph exceeds the native {NATIVE_MAX_NODES}-node limit"),
                INVALID_ARGUMENT,
            ));
        }
        structural |= patch_is_structural;
        prepared.push(PreparedGraphPatch {
            node_index,
            changed_index,
            mode,
            patch: Value::Object(patch_json),
            changed_outputs,
        });
    }

    let mut produced = std::collections::HashSet::new();
    for name in nodes
        .iter()
        .flatten()
        .flat_map(|node| node.outputs.iter().map(|entry| &entry.value))
    {
        if !produced.insert(name.clone()) {
            return Err(err(
                format!("PatchGraph resulting tensor {name} has multiple producers"),
                INVALID_ARGUMENT,
            ));
        }
    }
    validate_declared_output_names(declared_output_names, &produced)?;
    Ok((prepared, structural))
}

fn active_produced_names(root: &Value) -> std::collections::HashSet<String> {
    root.get("nodes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|node| {
            !node
                .get("disabled")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        })
        .filter_map(|node| node.get("outputs").and_then(Value::as_object))
        .flat_map(|outputs| outputs.values())
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect()
}

fn output_specs_from_inspection(
    root: &Value,
    names: &[String],
) -> Result<Vec<TensorSpec>, FfiError> {
    let produced = active_produced_names(root);
    validate_declared_output_names(names, &produced).map_err(|error| {
        err(
            format!(
                "native patch result disagrees with preflight: {}",
                error.message
            ),
            INTERNAL,
        )
    })?;
    let specs = tensor_specs_by_name(root);
    names
        .iter()
        .map(|name| {
            specs.get(name).cloned().ok_or_else(|| {
                err(
                    format!("native patch output tensor {name} has no metadata"),
                    INTERNAL,
                )
            })
        })
        .collect()
}

impl VolvoxAiServicePlugin for Plugin {
    fn ping(&self, _request: Empty) -> Result<PingResponse, FfiError> {
        let active = self.inner.lock().unwrap().model_backend;
        let mut available_backends = vec![RuntimeBackend::Cpu.proto()];
        if active != RuntimeBackend::Cpu {
            available_backends.push(active.proto());
        }
        Ok(PingResponse {
            version: "volvoxai-runtime/0.1".into(),
            available_backends,
        })
    }

    fn get_capabilities(&self, _request: Empty) -> Result<Capabilities, FfiError> {
        // TODO: emit the docs/operation_list.md matrix per backend/op.
        let active = self.inner.lock().unwrap().model_backend;
        let mut backends = vec![BackendCapability {
            backend: RuntimeBackend::Cpu.proto(),
            available: true,
            ops: vec![],
        }];
        if active != RuntimeBackend::Cpu {
            backends.push(BackendCapability {
                backend: active.proto(),
                available: true,
                ops: vec![],
            });
        }
        Ok(Capabilities {
            version: "volvoxai-runtime/0.1".into(),
            backends,
            adapters: Some(AdapterCapabilities {
                kinds: vec![AdapterKind::AdapterLora as i32],
                immutable_versions: true,
                hot_swap: true,
                per_batch_routing: true,
                adapter_only_persistence: true,
                merge_unmerge: true,
                gpu_grouped_matmul: false,
            }),
        })
    }

    fn load_model(&self, request: LoadModelRequest) -> Result<LoadModelResponse, FfiError> {
        let _transaction = self.model_transaction.lock().unwrap();
        let paths = match request.source {
            Some(load_model_request::Source::Paths(paths)) => paths,
            Some(load_model_request::Source::Inline(_)) => {
                return Err(err(
                    "inline model bytes not supported yet; use paths",
                    UNIMPLEMENTED,
                ))
            }
            None => return Err(err("missing model source", INVALID_ARGUMENT)),
        };
        self.load_model_paths(
            paths,
            request.exec,
            request.edit.unwrap_or_default(),
            Vec::new(),
            Vec::new(),
            None,
            0,
            None,
            None,
            None,
            std::collections::HashMap::new(),
        )
    }

    fn create_model(&self, request: CreateModelRequest) -> Result<LoadModelResponse, FfiError> {
        let _transaction = self.model_transaction.lock().unwrap();
        let backing = materialize_created_model(&request)?;
        let directory = backing.dir.clone();
        let result = self.load_model_paths(
            backing.paths,
            request.exec,
            ModelEditOptions {
                open_weights_writable: true,
                allow_tensor_updates: true,
                allow_graph_patches: true,
            },
            backing.inputs,
            backing.outputs,
            Some(directory.clone()),
            0,
            None,
            None,
            None,
            string_entries_to_map(&request.metadata),
        );
        if result.is_err() {
            let _ = fs::remove_dir_all(directory);
        }
        result
    }
    fn unload_model(&self, request: ModelRef) -> Result<Empty, FfiError> {
        let _transaction = self.model_transaction.lock().unwrap();
        let _lifecycle = self.model_lifecycle.write().unwrap();
        let _exec = self.engine_exec.lock().unwrap();
        let _admin = self.adapter_admin.lock().unwrap();
        let mut inner = self.inner.lock().unwrap();
        require(&inner, &request.model_id)?;
        let backend_call = begin_backend_call(inner.model_backend, &None, false)?;
        let owned_model_dir = inner.owned_model_dir.take();
        unsafe {
            if !inner.tok.is_null() {
                volvoxai_tokenizer_free(inner.tok);
                inner.tok = std::ptr::null_mut();
            }
        }
        backend_call.shutdown_engine();
        inner.model_id = None;
        inner.model_backend = RuntimeBackend::Cpu;
        inner.info = None;
        inner.config_path.clear();
        inner.tokenizer_path.clear();
        inner.model_metadata.clear();
        inner.owned_model_dir = None;
        inner.training_step = 0;
        inner.training_optimizer = None;
        inner.training_update_mode = None;
        inner.training_accumulation_microbatches = 0;
        inner.training_accumulation_steps = 1;
        inner.training_accumulation_optimizer_step = None;
        inner.training_accumulation_optimizer = None;
        inner.training_accumulation_update_mode = None;
        inner.allow_tensor_updates = false;
        inner.allow_graph_patches = false;
        inner.adapter_counter = 0;
        inner.adapter_generation += 1;
        inner.adapters.clear();
        inner.active_adapters = AdapterSelection::default();
        inner.merged_adapter = None;
        inner.pre_merge_active = None;
        drop(inner);
        if let Some(directory) = owned_model_dir {
            let _ = fs::remove_dir_all(directory);
        }
        Ok(Empty {})
    }

    fn get_model_info(&self, request: ModelRef) -> Result<ModelInfo, FfiError> {
        let inner = self.inner.lock().unwrap();
        require(&inner, &request.model_id)?;
        Ok(inner.info.clone().unwrap_or_default())
    }

    fn inspect_model(&self, request: InspectModelRequest) -> Result<ModelInspection, FfiError> {
        let _exec = self.engine_exec.lock().unwrap();
        let inner = self.inner.lock().unwrap();
        require(&inner, &request.model_id)?;
        let _backend_call = begin_backend_call(inner.model_backend, &None, false)?;
        unsafe {
            let root = native_inspection(
                request.include_graph,
                request.include_tensors,
                request.include_weight_files,
                request.include_params_json,
            )?;
            let mut info = inner.info.clone().unwrap_or_default();
            if let Some(n) = root.get("num_ops").and_then(Value::as_i64) {
                info.num_ops = n as i32;
            }
            let tensors = root
                .get("tensors")
                .and_then(Value::as_array)
                .map(|items| items.iter().map(tensor_spec_from_value).collect())
                .unwrap_or_default();
            let graph = if request.include_graph {
                Some(graph_from_inspection(&root))
            } else {
                None
            };
            let weights = root
                .get("weights")
                .and_then(Value::as_array)
                .map(|items| items.iter().map(safetensors_info_from_value).collect())
                .unwrap_or_default();
            Ok(ModelInspection {
                info: Some(info),
                tensors,
                graph,
                weight_files: weights,
            })
        }
    }

    fn save_model_weights(
        &self,
        request: SaveModelWeightsRequest,
    ) -> Result<SaveModelWeightsResponse, FfiError> {
        let _persistence = PERSISTENCE_COMMIT.lock().unwrap();
        let _exec = self.engine_exec.lock().unwrap();
        let _admin = self.adapter_admin.lock().unwrap();
        let inner = self.inner.lock().unwrap();
        require(&inner, &request.model_id)?;
        require_unmerged(&inner, "SaveModelWeights")?;
        require_no_pending_accumulation(&inner, "SaveModelWeights")?;
        let _backend_call = begin_backend_call(inner.model_backend, &None, false)?;
        if !inner.allow_tensor_updates {
            return Err(err(
                "SaveModelWeights requires LoadModelRequest.edit.allow_tensor_updates or open_weights_writable",
                PERMISSION_DENIED,
            ));
        }
        unsafe {
            let inspect = native_inspection(false, false, true, false)?;
            let loaded_weights = inspect
                .get("weights")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            if request.weight_paths.len() > loaded_weights.len() {
                return Err(err(
                    "SaveModelWeights supplied more paths than loaded weight shards",
                    INVALID_ARGUMENT,
                ));
            }
            let paths: Vec<String> = loaded_weights
                .iter()
                .enumerate()
                .map(|(index, weight)| {
                    request
                        .weight_paths
                        .get(index)
                        .filter(|path| !path.is_empty())
                        .cloned()
                        .or_else(|| {
                            weight
                                .get("path")
                                .and_then(Value::as_str)
                                .map(str::to_string)
                        })
                        .filter(|path| !path.is_empty())
                        .ok_or_else(|| {
                            err(
                                format!("weight shard {index} has no persistence path"),
                                INVALID_ARGUMENT,
                            )
                        })
                })
                .collect::<Result<_, _>>()?;

            if request.atomic {
                let mut staged = Vec::with_capacity(paths.len());
                for (index, path) in paths.iter().enumerate() {
                    let target = PathBuf::from(path);
                    let stage = match unique_sibling_path(&target, "stage") {
                        Ok(stage) => stage,
                        Err(error) => {
                            for (prior, _) in &staged {
                                let _ = fs::remove_file(prior);
                            }
                            return Err(error);
                        }
                    };
                    let c_stage = CString::new(stage.to_string_lossy().as_bytes()).unwrap();
                    if volvoxai_engine_save_weight_file(index as c_int, c_stage.as_ptr()) != 0 {
                        let _ = fs::remove_file(&stage);
                        for (prior, _) in &staged {
                            let _ = fs::remove_file(prior);
                        }
                        return Err(err(format!("flush weight file {index} failed"), INTERNAL));
                    }
                    staged.push((stage, target));
                }
                commit_staged_files(staged)?;
            } else {
                for (index, path) in paths.iter().enumerate() {
                    let c_path = CString::new(path.as_str())
                        .map_err(|_| err("bad flush path", INVALID_ARGUMENT))?;
                    if volvoxai_engine_save_weight_file(index as c_int, c_path.as_ptr()) != 0 {
                        return Err(err(format!("flush weight file {index} failed"), INTERNAL));
                    }
                    sync_file(Path::new(path))?;
                    sync_parent(Path::new(path))?;
                }
            }
            let weights = paths
                .iter()
                .map(|path| read_safetensors_info(path, true))
                .collect::<Result<_, _>>()?;
            Ok(SaveModelWeightsResponse {
                weight_files: weights,
                weight_paths: paths,
            })
        }
    }

    fn add_model_tensor(&self, request: AddModelTensorRequest) -> Result<TensorSpec, FfiError> {
        let _lifecycle = self.model_lifecycle.write().unwrap();
        let _exec = self.engine_exec.lock().unwrap();
        let _admin = self.adapter_admin.lock().unwrap();
        let inner = self.inner.lock().unwrap();
        require(&inner, &request.model_id)?;
        require_unmerged(&inner, "AddModelTensor")?;
        require_no_pending_accumulation(&inner, "AddModelTensor")?;
        let _backend_call = begin_backend_call(inner.model_backend, &None, false)?;
        if !inner.adapters.is_empty() {
            return Err(err(
                "AddModelTensor is not allowed while adapter versions are loaded; remove them first",
                FAILED_PRECONDITION,
            ));
        }
        if !inner.allow_tensor_updates {
            return Err(err(
                "AddModelTensor requires CreateModel or writable model edit options",
                PERMISSION_DENIED,
            ));
        }
        let tensor = request
            .tensor
            .as_ref()
            .ok_or_else(|| err("AddModelTensor requires tensor", INVALID_ARGUMENT))?;
        let tensor = materialize_tensor_initializer(tensor, "AddModelTensor")?;
        validate_native_text(&tensor.name, "AddModelTensor name", NATIVE_MAX_TENSOR_NAME)?;
        validate_created_shape(&tensor.shape, &format!("AddModelTensor {}", tensor.name))?;
        if tensor.quant.is_some() {
            return Err(err(
                "AddModelTensor does not accept a quantization descriptor",
                INVALID_ARGUMENT,
            ));
        }
        let native_dtype = proto_to_native_dtype(tensor.dtype).ok_or_else(|| {
            err(
                format!("AddModelTensor does not support dtype {}", dtype_name(tensor.dtype)),
                INVALID_ARGUMENT,
            )
        })?;
        let expected = tensor_nbytes(tensor.dtype, &tensor.shape)?;
        if tensor.data.len() != expected {
            return Err(err(
                format!(
                    "AddModelTensor {} byte-size mismatch: got {}, want {}",
                    tensor.name,
                    tensor.data.len(),
                    expected
                ),
                INVALID_ARGUMENT,
            ));
        }
        let shape: Vec<c_int> = tensor.shape.iter().map(|dimension| *dimension as c_int).collect();
        let name = CString::new(tensor.name.as_str())
            .map_err(|_| err("bad AddModelTensor name", INVALID_ARGUMENT))?;
        if unsafe {
            volvoxai_engine_add_model_tensor_raw(
                name.as_ptr(),
                shape.as_ptr(),
                shape.len() as c_int,
                native_dtype,
                tensor.data.as_ptr() as *const c_void,
                tensor.data.len(),
            )
        } != 0
        {
            return Err(err(
                format!("failed to add model tensor {}", tensor.name),
                INVALID_ARGUMENT,
            ));
        }
        unsafe {
            tensor_spec_native(
                &tensor.name,
                TENSOR_ACCESS_READABLE | TENSOR_ACCESS_WRITABLE,
            )
        }
    }

    fn remove_model_tensor(&self, request: RemoveModelTensorRequest) -> Result<Empty, FfiError> {
        let _lifecycle = self.model_lifecycle.write().unwrap();
        let _exec = self.engine_exec.lock().unwrap();
        let _admin = self.adapter_admin.lock().unwrap();
        let inner = self.inner.lock().unwrap();
        require(&inner, &request.model_id)?;
        require_unmerged(&inner, "RemoveModelTensor")?;
        require_no_pending_accumulation(&inner, "RemoveModelTensor")?;
        let _backend_call = begin_backend_call(inner.model_backend, &None, false)?;
        if !inner.adapters.is_empty() {
            return Err(err(
                "RemoveModelTensor is not allowed while adapter versions are loaded; remove them first",
                FAILED_PRECONDITION,
            ));
        }
        if !inner.allow_tensor_updates {
            return Err(err(
                "RemoveModelTensor requires CreateModel or writable model edit options",
                PERMISSION_DENIED,
            ));
        }
        validate_native_text(
            &request.name,
            "RemoveModelTensor name",
            NATIVE_MAX_TENSOR_NAME,
        )?;
        let name = CString::new(request.name.as_str())
            .map_err(|_| err("bad RemoveModelTensor name", INVALID_ARGUMENT))?;
        if unsafe { volvoxai_engine_remove_model_tensor(name.as_ptr()) } != 0 {
            return Err(err(
                format!(
                    "failed to remove model tensor {}; it may be missing or referenced by the graph",
                    request.name
                ),
                FAILED_PRECONDITION,
            ));
        }
        Ok(Empty {})
    }

    fn save_training_checkpoint(
        &self,
        request: SaveTrainingCheckpointRequest,
    ) -> Result<TrainingCheckpointInfo, FfiError> {
        if request.directory.is_empty() {
            return Err(err(
                "SaveTrainingCheckpoint requires directory",
                INVALID_ARGUMENT,
            ));
        }
        let _persistence = PERSISTENCE_COMMIT.lock().unwrap();
        let _lifecycle = self.model_lifecycle.write().unwrap();
        let _exec = self.engine_exec.lock().unwrap();
        let _admin = self.adapter_admin.lock().unwrap();
        let inner = self.inner.lock().unwrap();
        require(&inner, &request.model_id)?;
        require_unmerged(&inner, "SaveTrainingCheckpoint")?;
        let _backend_call = begin_backend_call(inner.model_backend, &None, false)?;
        if inner.training_accumulation_microbatches > 0 {
            return Err(err(
                format!(
                    "SaveTrainingCheckpoint cannot serialize {} pending accumulated microbatch(es); flush or reset the window first",
                    inner.training_accumulation_microbatches
                ),
                FAILED_PRECONDITION,
            ));
        }
        if !inner.adapters.is_empty() {
            return Err(err(
                "SaveTrainingCheckpoint does not include staged adapters; remove them first",
                FAILED_PRECONDITION,
            ));
        }
        let target = PathBuf::from(&request.directory);
        let parent = target
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)
            .map_err(|error| err(format!("create checkpoint parent failed: {error}"), INTERNAL))?;
        let stage = unique_sibling_path(&target, "stage")?;
        fs::create_dir(&stage)
            .map_err(|error| err(format!("create checkpoint stage failed: {error}"), INTERNAL))?;

        let result = (|| -> Result<TrainingCheckpointInfo, FfiError> {
            let inspection = unsafe { native_inspection(false, false, true, false)? };
            let weight_count = inspection
                .get("weights")
                .and_then(Value::as_array)
                .map(|weights| weights.len())
                .unwrap_or(0);
            if weight_count == 0 {
                return Err(err("loaded model has no weight shard", FAILED_PRECONDITION));
            }

            let config_name = "config.json";
            let config_path = stage.join(config_name);
            let config_c = CString::new(config_path.to_string_lossy().as_bytes()).unwrap();
            if unsafe { volvoxai_engine_save_config(config_c.as_ptr()) } != 0 {
                return Err(err("save checkpoint graph config failed", INTERNAL));
            }
            sync_file(&config_path)?;

            let weight_names: Vec<String> = (0..weight_count)
                .map(|index| {
                    if weight_count == 1 {
                        "model.safetensors".to_string()
                    } else {
                        format!("model-{:05}-of-{:05}.safetensors", index + 1, weight_count)
                    }
                })
                .collect();
            for (index, name) in weight_names.iter().enumerate() {
                let path = stage.join(name);
                let path_c = CString::new(path.to_string_lossy().as_bytes()).unwrap();
                if unsafe { volvoxai_engine_save_weight_file(index as c_int, path_c.as_ptr()) } != 0 {
                    return Err(err(format!("save checkpoint weight shard {index} failed"), INTERNAL));
                }
                sync_file(&path)?;
            }

            let optimizer_name = "optimizer.safetensors";
            let optimizer_path = stage.join(optimizer_name);
            let optimizer_c = CString::new(optimizer_path.to_string_lossy().as_bytes()).unwrap();
            if unsafe {
                volvoxai_engine_save_optimizer_state(optimizer_c.as_ptr(), inner.training_step)
            } != 0
            {
                return Err(err("save checkpoint optimizer state failed", INTERNAL));
            }
            sync_file(&optimizer_path)?;

            let tokenizer_bytes = match request.tokenizer.as_ref() {
                Some(bytes) => Some(bytes.clone()),
                None if !inner.tokenizer_path.is_empty() => Some(
                    fs::read(&inner.tokenizer_path)
                        .map_err(|error| err(format!("read model tokenizer failed: {error}"), NOT_FOUND))?,
                ),
                None => None,
            };
            let tokenizer_name = if let Some(bytes) = tokenizer_bytes {
                if bytes.is_empty() {
                    return Err(err("checkpoint tokenizer bytes cannot be empty", INVALID_ARGUMENT));
                }
                write_synced_bytes(&stage.join("tokenizer.bin"), &bytes, "checkpoint tokenizer")?;
                Some("tokenizer.bin")
            } else {
                None
            };
            let mut metadata = inner.model_metadata.clone();
            metadata.extend(string_entries_to_map(&request.metadata));
            let training_optimizer = inner.training_optimizer.as_ref().map(|options| {
                json!({
                    "learning_rate": options.learning_rate,
                    "beta1": options.beta1,
                    "beta2": options.beta2,
                    "epsilon": options.epsilon,
                    "weight_decay": options.weight_decay,
                    "max_grad_norm": options.max_grad_norm,
                })
            });
            let manifest = json!({
                "format": TRAINING_CHECKPOINT_FORMAT,
                "training_step": inner.training_step,
                "config": config_name,
                "weights": weight_names,
                "optimizer": optimizer_name,
                "training_optimizer": training_optimizer,
                "training_update_mode": inner.training_update_mode,
                "tokenizer": tokenizer_name,
                "metadata": metadata,
            });
            let manifest_bytes = serde_json::to_vec_pretty(&manifest)
                .map_err(|error| err(format!("serialize checkpoint manifest failed: {error}"), INTERNAL))?;
            write_synced_bytes(&stage.join("manifest.json"), &manifest_bytes, "checkpoint manifest")?;
            fs::File::open(&stage)
                .and_then(|directory| directory.sync_all())
                .map_err(|error| err(format!("fsync checkpoint directory failed: {error}"), INTERNAL))?;
            commit_staged_directory(&stage, &target, request.overwrite)?;

            let weight_paths: Vec<PathBuf> = weight_names.iter().map(|name| target.join(name)).collect();
            let weight_files = weight_paths
                .iter()
                .map(|path| read_safetensors_info(&path.to_string_lossy(), true))
                .collect::<Result<Vec<_>, _>>()?;
            let optimizer_path = target.join(optimizer_name);
            let paths = ModelPaths {
                config_path: target.join(config_name).to_string_lossy().into_owned(),
                weights_path: weight_paths[0].to_string_lossy().into_owned(),
                tokenizer_path: tokenizer_name
                    .map(|name| target.join(name).to_string_lossy().into_owned())
                    .unwrap_or_default(),
                extra_weights_paths: weight_paths
                    .iter()
                    .skip(1)
                    .map(|path| path.to_string_lossy().into_owned())
                    .collect(),
            };
            Ok(TrainingCheckpointInfo {
                directory: target.to_string_lossy().into_owned(),
                format: TRAINING_CHECKPOINT_FORMAT.to_string(),
                training_step: inner.training_step,
                paths: Some(paths),
                weight_files,
                optimizer_file: Some(read_safetensors_info(
                    &optimizer_path.to_string_lossy(),
                    true,
                )?),
                metadata: string_entries_from_map(metadata),
            })
        })();
        if result.is_err() {
            let _ = fs::remove_dir_all(&stage);
        }
        result
    }

    fn load_training_checkpoint(
        &self,
        request: LoadTrainingCheckpointRequest,
    ) -> Result<LoadTrainingCheckpointResponse, FfiError> {
        let _transaction = self.model_transaction.lock().unwrap();
        if request.directory.is_empty() {
            return Err(err(
                "LoadTrainingCheckpoint requires directory",
                INVALID_ARGUMENT,
            ));
        }
        let directory = PathBuf::from(&request.directory);
        let parsed = parse_training_checkpoint(&directory)?;
        let weight_files = parsed
            .weights
            .iter()
            .map(|path| read_safetensors_info(&path.to_string_lossy(), true))
            .collect::<Result<Vec<_>, _>>()?;
        // All deterministic checkpoint errors are rejected before the global
        // native engine is replaced. This also validates moment finiteness.
        let optimizer_file = validate_training_optimizer_checkpoint(&parsed, &weight_files)?;
        let paths = ModelPaths {
            config_path: parsed.config.to_string_lossy().into_owned(),
            weights_path: parsed.weights[0].to_string_lossy().into_owned(),
            tokenizer_path: parsed
                .tokenizer
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned())
                .unwrap_or_default(),
            extra_weights_paths: parsed
                .weights
                .iter()
                .skip(1)
                .map(|path| path.to_string_lossy().into_owned())
                .collect(),
        };
        // The native engine is a singleton, so checkpoint loading cannot build
        // a second graph before swapping. Save a complete rollback package for
        // the current handle; deterministic errors were already preflighted,
        // and an unexpected init/allocation failure restores this package.
        let previous = {
            let inner = self.inner.lock().unwrap();
            inner.model_id.as_ref().map(|model_id| {
                (
                    model_id.clone(),
                    inner.allow_tensor_updates,
                    inner.allow_graph_patches,
                    inner.model_backend,
                )
            })
        };
        let rollback = if let Some((model_id, _, _, _)) = previous.as_ref() {
            let rollback_dir = private_model_directory()?;
            let save = self.save_training_checkpoint(SaveTrainingCheckpointRequest {
                model_id: model_id.clone(),
                directory: rollback_dir.to_string_lossy().into_owned(),
                overwrite: true,
                tokenizer: None,
                metadata: Vec::new(),
            });
            if let Err(error) = save {
                let _ = fs::remove_dir_all(&rollback_dir);
                return Err(error);
            }
            Some(rollback_dir)
        } else {
            None
        };
        let load_result = self.load_model_paths(
            paths.clone(),
            request.exec,
            ModelEditOptions {
                open_weights_writable: true,
                allow_tensor_updates: true,
                allow_graph_patches: true,
            },
            Vec::new(),
            Vec::new(),
            None,
            parsed.training_step,
            Some(parsed.optimizer.to_string_lossy().into_owned()),
            parsed.training_optimizer.clone(),
            parsed.training_update_mode,
            parsed.metadata.clone(),
        );
        let model = match load_result {
            Ok(model) => {
                if let Some(rollback) = rollback.as_ref() {
                    let _ = fs::remove_dir_all(rollback);
                }
                model
            }
            Err(load_error) => {
                let Some(rollback_dir) = rollback.as_ref() else {
                    return Err(load_error);
                };
                let original_is_still_active = {
                    let inner = self.inner.lock().unwrap();
                    previous.as_ref().map_or(false, |(model_id, _, _, backend)| {
                        inner.model_id.as_ref() == Some(model_id) &&
                            inner.model_backend == *backend
                    })
                };
                if original_is_still_active {
                    let _ = fs::remove_dir_all(rollback_dir);
                    return Err(load_error);
                }
                let rollback_checkpoint = parse_training_checkpoint(rollback_dir)?;
                let rollback_paths = ModelPaths {
                    config_path: rollback_checkpoint.config.to_string_lossy().into_owned(),
                    weights_path: rollback_checkpoint.weights[0].to_string_lossy().into_owned(),
                    tokenizer_path: rollback_checkpoint
                        .tokenizer
                        .as_ref()
                        .map(|path| path.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    extra_weights_paths: rollback_checkpoint
                        .weights
                        .iter()
                        .skip(1)
                        .map(|path| path.to_string_lossy().into_owned())
                        .collect(),
                };
                let (_, allow_tensor_updates, allow_graph_patches, backend) = previous.as_ref().unwrap();
                let restore = self.load_model_paths(
                    rollback_paths,
                    Some(ExecOptions {
                        backend: Some(backend.proto()),
                        ..Default::default()
                    }),
                    ModelEditOptions {
                        open_weights_writable: *allow_tensor_updates,
                        allow_tensor_updates: *allow_tensor_updates,
                        allow_graph_patches: *allow_graph_patches,
                    },
                    Vec::new(),
                    Vec::new(),
                    Some(rollback_dir.clone()),
                    rollback_checkpoint.training_step,
                    Some(rollback_checkpoint.optimizer.to_string_lossy().into_owned()),
                    rollback_checkpoint.training_optimizer,
                    rollback_checkpoint.training_update_mode,
                    rollback_checkpoint.metadata,
                );
                match restore {
                    Ok(_) => {
                        // Preserve the pre-call handle even though the internal
                        // restore passed through the normal loader.
                        self.inner.lock().unwrap().model_id =
                            Some(previous.as_ref().unwrap().0.clone());
                        return Err(load_error);
                    }
                    Err(restore_error) => {
                        let _ = fs::remove_dir_all(rollback_dir);
                        return Err(err(
                            format!(
                                "checkpoint load failed ({load_error}); rollback also failed ({restore_error})"
                            ),
                            INTERNAL,
                        ));
                    }
                }
            }
        };
        Ok(LoadTrainingCheckpointResponse {
            model: Some(model),
            checkpoint: Some(TrainingCheckpointInfo {
                directory: directory.to_string_lossy().into_owned(),
                format: TRAINING_CHECKPOINT_FORMAT.to_string(),
                training_step: parsed.training_step,
                paths: Some(paths),
                weight_files,
                optimizer_file: Some(optimizer_file),
                metadata: string_entries_from_map(parsed.metadata),
            }),
        })
    }

    fn get_graph(&self, request: ModelRef) -> Result<GraphInfo, FfiError> {
        let _exec = self.engine_exec.lock().unwrap();
        let inner = self.inner.lock().unwrap();
        require(&inner, &request.model_id)?;
        let _backend_call = begin_backend_call(inner.model_backend, &None, false)?;
        unsafe {
            let root = native_inspection(true, false, false, true)?;
            Ok(graph_from_inspection(&root))
        }
    }

    fn inspect_safetensors(
        &self,
        request: InspectSafetensorsRequest,
    ) -> Result<SafetensorsInfo, FfiError> {
        read_safetensors_info(&request.path, request.include_metadata)
    }

    fn create_safetensors(
        &self,
        request: CreateSafetensorsRequest,
    ) -> Result<SafetensorsInfo, FfiError> {
        write_safetensors(
            &request.path,
            &request.tensors,
            &string_entries_to_map(&request.metadata),
            request.overwrite,
        )
    }

    fn stage_adapter(&self, request: StageAdapterRequest) -> Result<AdapterInfo, FfiError> {
        self.stage_adapter_version(
            &request.model_id,
            request.manifest,
            None,
            &request.tensors,
            None,
            request.activate,
            0,
        )
    }

    fn load_adapter(&self, request: LoadAdapterRequest) -> Result<AdapterInfo, FfiError> {
        if request.path.is_empty() {
            return Err(err("LoadAdapter requires path", INVALID_ARGUMENT));
        }
        let (_file, manifest, tensors) = read_adapter_checkpoint(&request.path)?;
        if let Some(expected) = request.manifest {
            if !adapter_manifests_equal(&expected, &manifest) {
                return Err(err(
                    "LoadAdapter expected manifest does not match checkpoint metadata",
                    INVALID_ARGUMENT,
                ));
            }
        }
        if let Some(expected_id) = request.adapter_id.as_deref() {
            if expected_id != manifest.adapter_id {
                return Err(err(
                    "LoadAdapter adapter_id does not match checkpoint metadata",
                    INVALID_ARGUMENT,
                ));
            }
        }
        self.stage_adapter_version(
            &request.model_id,
            Some(manifest),
            None,
            &tensors,
            Some(&request.path),
            request.activate,
            0,
        )
    }

    fn list_adapters(
        &self,
        request: ListAdaptersRequest,
    ) -> Result<ListAdaptersResponse, FfiError> {
        let _admin = self.adapter_admin.lock().unwrap();
        {
            let inner = self.inner.lock().unwrap();
            require(&inner, &request.model_id)?;
        }
        unsafe {
            let ptr = volvoxai_engine_adapter_list_json();
            if ptr.is_null() {
                return Err(err("native adapter registry inspection failed", INTERNAL));
            }
            let text = CStr::from_ptr(ptr).to_string_lossy().into_owned();
            free(ptr as *mut c_void);
            serde_json::from_str::<Value>(&text)
                .map_err(|e| err(format!("bad native adapter registry JSON: {e}"), INTERNAL))?;
        }
        let inner = self.inner.lock().unwrap();
        let adapters = inner
            .adapters
            .iter()
            .filter(|record| {
                request.adapter_id.as_ref().map_or(true, |id| {
                    record
                        .info
                        .adapter
                        .as_ref()
                        .map_or(false, |reference| reference.adapter_id == *id)
                })
            })
            .map(|record| adapter_info_with_state(&inner, record))
            .collect();
        Ok(ListAdaptersResponse {
            adapters,
            state: Some(adapter_state(&inner)),
        })
    }

    fn activate_adapter(&self, request: ActivateAdapterRequest) -> Result<AdapterState, FfiError> {
        let selection = request
            .selection
            .ok_or_else(|| err("ActivateAdapter requires selection", INVALID_ARGUMENT))?;
        if selection.routes.len() > 1 {
            return Err(err(
                "the model default can contain only one broadcast adapter; use per-call routing for per-batch adapters",
                INVALID_ARGUMENT,
            ));
        }
        let _admin = self.adapter_admin.lock().unwrap();
        let resolved = {
            let inner = self.inner.lock().unwrap();
            require(&inner, &request.model_id)?;
            require_unmerged(&inner, "ActivateAdapter")?;
            resolve_adapter_selection(&inner, &selection)?
        };
        if resolved.iter().any(|(_, scale)| *scale != 1.0) {
            return Err(err(
                "persistent activation does not support a request-local scale; put scale in the manifest",
                INVALID_ARGUMENT,
            ));
        }
        let native_name = resolved
            .first()
            .map(|(name, _)| name.as_str())
            .unwrap_or("");
        let c_name = CString::new(native_name).unwrap();
        unsafe {
            if volvoxai_engine_adapter_activate(c_name.as_ptr()) != 0 {
                return Err(err("native adapter activation failed", FAILED_PRECONDITION));
            }
        }
        let mut inner = self.inner.lock().unwrap();
        require(&inner, &request.model_id)?;
        inner.active_adapters = selection;
        inner.adapter_generation += 1;
        Ok(adapter_state(&inner))
    }

    fn update_adapter(&self, request: UpdateAdapterRequest) -> Result<AdapterInfo, FfiError> {
        let parent = request
            .parent
            .as_ref()
            .ok_or_else(|| err("UpdateAdapter requires parent", INVALID_ARGUMENT))?;
        self.update_adapter_version(
            &request.model_id,
            parent,
            &request.updates,
            request.activate,
            request.optimizer_step,
        )
    }

    fn remove_adapter(&self, request: RemoveAdapterRequest) -> Result<Empty, FfiError> {
        let reference = request
            .adapter
            .as_ref()
            .ok_or_else(|| err("RemoveAdapter requires adapter", INVALID_ARGUMENT))?;
        let _admin = self.adapter_admin.lock().unwrap();
        let native_name = {
            let inner = self.inner.lock().unwrap();
            require(&inner, &request.model_id)?;
            if inner.active_adapters.routes.iter().any(|route| {
                route
                    .adapter
                    .as_ref()
                    .map_or(false, |candidate| adapter_ref_eq(candidate, reference))
            }) {
                return Err(err(
                    "cannot remove the active adapter version",
                    FAILED_PRECONDITION,
                ));
            }
            if inner
                .merged_adapter
                .as_ref()
                .map_or(false, |candidate| adapter_ref_eq(candidate, reference))
            {
                return Err(err(
                    "cannot remove the merged adapter version",
                    FAILED_PRECONDITION,
                ));
            }
            if inner.pre_merge_active.as_ref().map_or(false, |selection| {
                selection.routes.iter().any(|route| {
                    route
                        .adapter
                        .as_ref()
                        .map_or(false, |candidate| adapter_ref_eq(candidate, reference))
                })
            }) {
                return Err(err(
                    "cannot remove an adapter version saved for unmerge",
                    FAILED_PRECONDITION,
                ));
            }
            adapter_record(&inner, reference)?.native_name.clone()
        };
        let c_name = CString::new(native_name.as_str()).unwrap();
        unsafe {
            if volvoxai_engine_adapter_remove(c_name.as_ptr()) != 0 {
                return Err(err(
                    "native adapter removal failed; the version may still be pinned by a request",
                    FAILED_PRECONDITION,
                ));
            }
        }
        let mut inner = self.inner.lock().unwrap();
        inner
            .adapters
            .retain(|record| record.native_name != native_name);
        inner.adapter_generation += 1;
        Ok(Empty {})
    }

    fn merge_adapter(&self, request: MergeAdapterRequest) -> Result<AdapterState, FfiError> {
        let reference = request
            .adapter
            .as_ref()
            .ok_or_else(|| err("MergeAdapter requires adapter", INVALID_ARGUMENT))?;
        let _lifecycle = self.model_lifecycle.write().unwrap();
        let _exec = self.engine_exec.lock().unwrap();
        let _admin = self.adapter_admin.lock().unwrap();
        let (native_name, backend) = {
            let inner = self.inner.lock().unwrap();
            require(&inner, &request.model_id)?;
            if inner.merged_adapter.is_some() {
                return Err(err("an adapter is already merged", FAILED_PRECONDITION));
            }
            (
                adapter_record(&inner, reference)?.native_name.clone(),
                inner.model_backend,
            )
        };
        let _backend_call = begin_backend_call(backend, &None, false)?;
        let c_name = CString::new(native_name).unwrap();
        unsafe {
            if volvoxai_engine_adapter_merge(c_name.as_ptr()) != 0 {
                return Err(err("native adapter merge failed", FAILED_PRECONDITION));
            }
        }
        let mut inner = self.inner.lock().unwrap();
        inner.pre_merge_active = Some(inner.active_adapters.clone());
        inner.merged_adapter = Some(reference.clone());
        inner.active_adapters = AdapterSelection::default();
        inner.adapter_generation += 1;
        Ok(adapter_state(&inner))
    }

    fn unmerge_adapter(&self, request: UnmergeAdapterRequest) -> Result<AdapterState, FfiError> {
        let _lifecycle = self.model_lifecycle.write().unwrap();
        let _exec = self.engine_exec.lock().unwrap();
        let _admin = self.adapter_admin.lock().unwrap();
        let backend = {
            let inner = self.inner.lock().unwrap();
            require(&inner, &request.model_id)?;
            let merged = inner
                .merged_adapter
                .as_ref()
                .ok_or_else(|| err("no adapter is merged", FAILED_PRECONDITION))?;
            if let Some(guard) = &request.adapter {
                if !adapter_ref_eq(guard, merged) {
                    return Err(err(
                        "merged adapter guard does not match",
                        FAILED_PRECONDITION,
                    ));
                }
            }
            inner.model_backend
        };
        let _backend_call = begin_backend_call(backend, &None, false)?;
        unsafe {
            if volvoxai_engine_adapter_unmerge() != 0 {
                return Err(err("native adapter unmerge failed", FAILED_PRECONDITION));
            }
        }
        let mut inner = self.inner.lock().unwrap();
        inner.merged_adapter = None;
        inner.active_adapters = inner.pre_merge_active.take().unwrap_or_default();
        inner.adapter_generation += 1;
        Ok(adapter_state(&inner))
    }

    fn save_adapter(&self, request: SaveAdapterRequest) -> Result<SaveAdapterResponse, FfiError> {
        let reference = request
            .adapter
            .as_ref()
            .ok_or_else(|| err("SaveAdapter requires adapter", INVALID_ARGUMENT))?;
        if request.path.is_empty() {
            return Err(err("SaveAdapter requires path", INVALID_ARGUMENT));
        }
        let _persistence = PERSISTENCE_COMMIT.lock().unwrap();
        let _lifecycle = self.model_lifecycle.read().unwrap();
        let _admin = self.adapter_admin.lock().unwrap();
        let (native_name, info) = {
            let inner = self.inner.lock().unwrap();
            require(&inner, &request.model_id)?;
            let record = adapter_record(&inner, reference)?;
            (
                record.native_name.clone(),
                adapter_info_with_state(&inner, record),
            )
        };
        drop(_admin);
        let target = PathBuf::from(&request.path);
        let output_path = if request.atomic {
            unique_sibling_path(&target, "stage")?
        } else {
            target.clone()
        };
        let c_name = CString::new(native_name.as_str()).unwrap();
        let c_path = CString::new(output_path.to_string_lossy().as_bytes())
            .map_err(|_| err("adapter output path contains NUL", INVALID_ARGUMENT))?;
        unsafe {
            if volvoxai_engine_adapter_save(c_name.as_ptr(), c_path.as_ptr()) != 0 {
                if request.atomic {
                    let _ = fs::remove_file(&output_path);
                }
                return Err(err("native adapter save failed", INTERNAL));
            }
        }
        if request.atomic {
            commit_staged_files(vec![(output_path, target)])?;
        } else {
            sync_file(&output_path)?;
            sync_parent(&output_path)?;
        }
        let weight_file = read_safetensors_info(&request.path, true)?;
        if string_entry_value(&weight_file.metadata, ADAPTER_MANIFEST_METADATA_KEY).is_none() {
            return Err(err(
                format!("native adapter checkpoint omitted {ADAPTER_MANIFEST_METADATA_KEY}"),
                INTERNAL,
            ));
        }
        let mut persisted = info;
        persisted.source_path = request.path.clone();
        let _admin = self.adapter_admin.lock().unwrap();
        let mut inner = self.inner.lock().unwrap();
        if let Some(record) = inner
            .adapters
            .iter_mut()
            .find(|record| record.native_name == native_name)
        {
            record.info.source_path = request.path.clone();
        }
        Ok(SaveAdapterResponse {
            adapter: Some(persisted),
            weight_file: Some(weight_file),
            path: request.path,
        })
    }

    fn list_models(&self, _request: Empty) -> Result<ListModelsResponse, FfiError> {
        let inner = self.inner.lock().unwrap();
        let mut r = ListModelsResponse::default();
        if let Some(id) = &inner.model_id {
            r.model_ids.push(id.clone());
            if let Some(info) = &inner.info {
                r.models.push(info.clone());
            }
        }
        Ok(r)
    }

    fn run(&self, request: RunRequest) -> Result<RunResponse, FfiError> {
        let _exec = self.engine_exec.lock().unwrap();
        let (_route, backend) = {
            let _admin = self.adapter_admin.lock().unwrap();
            let (resolved, backend) = {
                let inner = self.inner.lock().unwrap();
                require(&inner, &request.model_id)?;
                unsafe {
                    validate_run_inputs_and_routes(request.adapters.as_ref(), &request.inputs)?
                };
                let resolved = request
                    .adapters
                    .as_ref()
                    .map(|selection| resolve_adapter_selection(&inner, selection))
                    .transpose()?;
                (resolved, inner.model_backend)
            };
            (unsafe { begin_adapter_route(resolved.as_deref())? }, backend)
        };
        let _backend_call = begin_backend_call(backend, &request.exec, false)?;
        if request.last_token.is_some_and(|row| row < -1) {
            return Err(err(
                "Run last_token must be -1 or non-negative",
                INVALID_ARGUMENT,
            ));
        }
        let output_row = request.last_token.filter(|row| *row >= 0);
        let _execution_row = ExecutionRowGuard::new(output_row.unwrap_or(-1))?;
        unsafe {
            for t in &request.inputs {
                copy_tensor_to_input(t)?;
            }
            if volvoxai_engine_forward() != 0 {
                return Err(err("volvoxai_engine_forward failed", INTERNAL));
            }
            let output_names = if request.output_names.is_empty() {
                native_graph_interface_names(false)?
            } else {
                request.output_names
            };
            let outputs = output_names
                .iter()
                .map(|name| read_tensor_for_execution_row(name, output_row))
                .collect::<Result<_, _>>()?;
            Ok(RunResponse {
                outputs,
                timing: None,
            })
        }
    }

    // Streaming greedy (argmax) decode; mirrors the generate wrapper in
    // examples/native_task_cli/main.c. Sampling (temperature/top_p/top_k) and
    // VLM image policy are TODO.
    fn generate(
        &self,
        request: GenerateRequest,
        stream: &dyn PluginStreamSender<GenerateEvent>,
    ) -> Result<(), FfiError> {
        if request
            .adapters
            .as_ref()
            .map_or(false, |selection| selection.routes.len() > 1)
        {
            return Err(err(
                "Generate accepts only a base route or one broadcast adapter",
                INVALID_ARGUMENT,
            ));
        }
        let _exec = self.engine_exec.lock().unwrap();
        let _admin = self.adapter_admin.lock().unwrap();
        let (tok, resolved, backend) = {
            let inner = self.inner.lock().unwrap();
            require(&inner, &request.model_id)?;
            let resolved = request
                .adapters
                .as_ref()
                .map(|selection| resolve_adapter_selection(&inner, selection))
                .transpose()?;
            (inner.tok, resolved, inner.model_backend)
        };
        let _route = unsafe { begin_adapter_route(resolved.as_deref())? };
        drop(_admin);
        if tok.is_null() {
            return Err(err(
                "model has no tokenizer; Generate needs one",
                UNIMPLEMENTED,
            ));
        }
        if request.image.is_some() {
            return Err(err("VLM/chat image input is a TODO", UNIMPLEMENTED));
        }
        let _backend_call = begin_backend_call(backend, &request.exec, false)?;
        unsafe {
            let cfg = request.r#gen.unwrap_or_default();
            let max_new = cfg.max_new_tokens.unwrap_or(50);
            let eos = cfg.eos_token.unwrap_or(-1);
            let pad = cfg.pad_token.unwrap_or(50256);

            let prompt = CString::new(request.prompt.as_str())
                .map_err(|_| err("bad prompt", INVALID_ARGUMENT))?;
            let tokens = CString::new("tokens").unwrap();
            let token_input_name = if volvoxai_engine_is_graph_input(tokens.as_ptr()) == 1 {
                "tokens".to_string()
            } else {
                first_graph_input_name()?
            };
            let logits_name = first_graph_output_name()?;
            let logits_name = CString::new(logits_name.as_str())
                .map_err(|_| err("bad graph output name", INTERNAL))?;
            let (token_name, token_numel, _, token_dtype, token_width) =
                graph_input_destination(&token_input_name)?;
            if token_numel == 0 {
                return Err(err("model has no token input", INVALID_ARGUMENT));
            }
            if !((token_dtype == T_I32 || token_dtype == T_F32)
                && token_width == std::mem::size_of::<c_int>())
            {
                return Err(err(
                    "model token input must use I32 or legacy F32 storage",
                    FAILED_PRECONDITION,
                ));
            }
            let cap = token_numel.min(MAX_SEQ);
            let mut ids = vec![0 as c_int; MAX_SEQ];
            let n = volvoxai_tokenizer_encode(tok, prompt.as_ptr(), ids.as_mut_ptr(), cap as c_int);
            if n <= 0 {
                return Err(err("prompt encoded to zero tokens", INVALID_ARGUMENT));
            }
            let mut token_values = vec![pad as c_int; token_numel];
            token_values[..n as usize].copy_from_slice(&ids[..n as usize]);
            let set_token_input = |values: &[c_int]| -> Result<(), FfiError> {
                let (data, nbytes, converted);
                if token_dtype == T_I32 {
                    data = values.as_ptr() as *const c_void;
                    nbytes = std::mem::size_of_val(values);
                    converted = None;
                } else {
                    let floats = values.iter().map(|value| *value as f32).collect::<Vec<_>>();
                    nbytes = std::mem::size_of_val(floats.as_slice());
                    data = floats.as_ptr() as *const c_void;
                    converted = Some(floats);
                }
                let status = volvoxai_engine_set_input_raw(
                    token_name.as_ptr(),
                    token_dtype,
                    data,
                    nbytes,
                );
                drop(converted);
                if status != 0 {
                    return Err(err(
                        "model token input became unavailable",
                        FAILED_PRECONDITION,
                    ));
                }
                Ok(())
            };
            set_token_input(&token_values)?;
            let _execution_row = ExecutionRowGuard::new(-1)?;
            if volvoxai_engine_forward_prefix(n) != 0 {
                return Err(err("volvoxai_engine_forward_prefix failed", INTERNAL));
            }

            let mut full = String::new();
            let mut reason = 2; // FINISH_LENGTH
            let mut pos = n - 1;
            let mut generated = 0;
            for _ in 0..max_new {
                if pos as usize >= cap - 1 {
                    break;
                }
                if stream.is_cancelled() {
                    reason = 3; // FINISH_STOP
                    break;
                }
                let mut count: c_int = 0;
                let lg = volvoxai_engine_tensor_row_f32(logits_name.as_ptr(), pos, &mut count);
                if lg.is_null() || count <= 0 {
                    return Err(err("no graph output row after generation step", INTERNAL));
                }
                let logits = std::slice::from_raw_parts(lg, count as usize);
                let next = argmax(logits);
                let piece = volvoxai_tokenizer_decode(tok, next);
                let text = if piece.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(piece).to_string_lossy().into_owned()
                };
                full.push_str(&text);

                let ev = GenerateEvent {
                    event: Some(generate_event::Event::Token(Token {
                        id: next,
                        text,
                        position: pos + 1,
                    })),
                };
                generated += 1;
                if !stream.send(ev) {
                    reason = 3;
                    break;
                }
                if next == eos {
                    reason = 1; // FINISH_EOS
                    break;
                }
                // Feed the sampled token and advance one position.
                pos += 1;
                token_values[pos as usize] = next;
                set_token_input(&token_values)?;
                if volvoxai_engine_forward_row(pos) != 0 {
                    return Err(err("volvoxai_engine_forward_row failed", INTERNAL));
                }
            }

            let done = GenerateDone {
                text: full,
                num_tokens: generated,
                reason,
                timing: None,
            };
            stream.send(GenerateEvent {
                event: Some(generate_event::Event::Done(done)),
            });
            Ok(())
        }
    }

    fn get_tensor(&self, request: GetTensorRequest) -> Result<Tensor, FfiError> {
        let _exec = self.engine_exec.lock().unwrap();
        let inner = self.inner.lock().unwrap();
        require(&inner, &request.model_id)?;
        let _backend_call = begin_backend_call(inner.model_backend, &None, false)?;
        unsafe { read_tensor(&request.name) }
    }

    fn set_tensor(&self, request: SetTensorRequest) -> Result<TensorSpec, FfiError> {
        let persist = request.persist_safetensors
            || request
                .safetensors_path
                .as_ref()
                .map(|path| !path.is_empty())
                .unwrap_or(false);
        let _persistence = persist.then(|| PERSISTENCE_COMMIT.lock().unwrap());
        let _exec = self.engine_exec.lock().unwrap();
        let _admin = self.adapter_admin.lock().unwrap();
        let inner = self.inner.lock().unwrap();
        require(&inner, &request.model_id)?;
        require_unmerged(&inner, "SetTensor")?;
        require_no_pending_accumulation(&inner, "SetTensor")?;
        let _backend_call = begin_backend_call(inner.model_backend, &None, false)?;
        if !inner.allow_tensor_updates {
            return Err(err(
                "SetTensor requires LoadModelRequest.edit.allow_tensor_updates or open_weights_writable",
                PERMISSION_DENIED,
            ));
        }
        let tensor = request
            .tensor
            .ok_or_else(|| err("missing tensor", INVALID_ARGUMENT))?;

        unsafe {
            let cname = CString::new(tensor.name.as_str())
                .map_err(|_| err("bad tensor name", INVALID_ARGUMENT))?;
            let mut numel: c_long = 0;
            let mut shape = [0 as c_int; 8];
            let mut ndim: c_int = 0;
            let mut native_dtype: c_int = 0;
            let mut elem_size: usize = 0;
            if volvoxai_engine_tensor_info_ex(
                cname.as_ptr(),
                &mut numel,
                shape.as_mut_ptr(),
                &mut ndim,
                &mut native_dtype,
                &mut elem_size,
            ) != 0
            {
                return Err(err(format!("no such tensor: {}", tensor.name), NOT_FOUND));
            }
            if numel < 0 {
                return Err(err(
                    format!("tensor {} has invalid size", tensor.name),
                    INTERNAL,
                ));
            }
            let actual_shape: Vec<i64> = shape[..ndim as usize].iter().map(|&d| d as i64).collect();
            if !tensor.shape.is_empty() && tensor.shape != actual_shape {
                return Err(err(
                    format!(
                        "tensor {} shape mismatch: got {:?}, want {:?}",
                        tensor.name, tensor.shape, actual_shape
                    ),
                    INVALID_ARGUMENT,
                ));
            }
            let requested_native_dtype = proto_to_native_dtype(tensor.dtype).ok_or_else(|| {
                err(
                    format!(
                        "SetTensor does not support dtype {}",
                        dtype_name(tensor.dtype)
                    ),
                    INVALID_ARGUMENT,
                )
            })?;
            if requested_native_dtype != native_dtype {
                return Err(err(
                    format!(
                        "tensor {} dtype mismatch: got {}, want {}",
                        tensor.name,
                        dtype_name(tensor.dtype),
                        dtype_name(native_to_proto_dtype(native_dtype))
                    ),
                    INVALID_ARGUMENT,
                ));
            }

            let want = (numel as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| err(format!("tensor {} is too large", tensor.name), INTERNAL))?;
            if tensor.data.len() != want {
                return Err(err(
                    format!(
                        "tensor {} byte-size mismatch: got {}, want {}",
                        tensor.name,
                        tensor.data.len(),
                        want
                    ),
                    INVALID_ARGUMENT,
                ));
            }
            if volvoxai_engine_set_tensor_raw(
                cname.as_ptr(),
                native_dtype,
                tensor.data.as_ptr() as *const c_void,
                tensor.data.len(),
            ) != 0
            {
                return Err(err(
                    format!("failed to update tensor {}", tensor.name),
                    INTERNAL,
                ));
            }
            if persist {
                let c_path = request
                    .safetensors_path
                    .as_ref()
                    .filter(|path| !path.is_empty())
                    .map(|path| CString::new(path.as_str()))
                    .transpose()
                    .map_err(|_| err("bad safetensors output path", INVALID_ARGUMENT))?;
                let ptr = c_path
                    .as_ref()
                    .map(|s| s.as_ptr())
                    .unwrap_or(std::ptr::null());
                let file_index = volvoxai_engine_tensor_weight_file_index(cname.as_ptr());
                if file_index < 0 {
                    return Err(err(
                        format!("tensor {} is not backed by safetensors", tensor.name),
                        NOT_FOUND,
                    ));
                }
                if volvoxai_engine_save_weight_file(file_index, ptr) != 0 {
                    return Err(err("persisting safetensors update failed", INTERNAL));
                }
            }
            Ok(TensorSpec {
                name: tensor.name,
                shape: actual_shape,
                dtype: native_to_proto_dtype(native_dtype),
                access_flags: TENSOR_ACCESS_READABLE | TENSOR_ACCESS_WRITABLE,
                size_bytes: want as i64,
            })
        }
    }

    fn apply_tensor_updates(
        &self,
        request: ApplyTensorUpdatesRequest,
    ) -> Result<ApplyTensorUpdatesResponse, FfiError> {
        let _persistence = request
            .persist_safetensors
            .then(|| PERSISTENCE_COMMIT.lock().unwrap());
        let _exec = self.engine_exec.lock().unwrap();
        let _admin = self.adapter_admin.lock().unwrap();
        let mut inner = self.inner.lock().unwrap();
        require(&inner, &request.model_id)?;
        require_unmerged(&inner, "ApplyTensorUpdates")?;
        require_no_pending_accumulation(&inner, "ApplyTensorUpdates")?;
        let _backend_call = begin_backend_call(inner.model_backend, &None, false)?;
        if !inner.allow_tensor_updates {
            return Err(err(
                "ApplyTensorUpdates requires LoadModelRequest.edit.allow_tensor_updates or open_weights_writable",
                PERMISSION_DENIED,
            ));
        }
        if request.updates.is_empty() {
            return Err(err(
                "ApplyTensorUpdates requires at least one update",
                INVALID_ARGUMENT,
            ));
        }
        let base_optimizer = request.optimizer.clone();
        let persist = request.persist_safetensors;
        let flush_paths = request.weight_paths.clone();
        let fallback_step = inner
            .training_step
            .checked_add(1)
            .ok_or_else(|| err("ApplyTensorUpdates optimizer step overflow", FAILED_PRECONDITION))?;
        let (_, _, _, _, _, _, base_step) =
            optimizer_values(None, base_optimizer.as_ref(), fallback_step)?;

        struct PreparedTensorUpdate {
            tensor: Tensor,
            spec: TensorSpec,
            cname: CString,
            native_dtype: c_int,
            payload: Option<Vec<f32>>,
            optimizer: (f32, f32, f32, f32, f32, f32, i64),
            original: Vec<u8>,
            mode: TensorUpdateMode,
        }

        let mut seen = std::collections::HashSet::new();
        let mut prepared = Vec::with_capacity(request.updates.len());
        let mut optimizer_batch_step = None;
        let mut optimizer_batch_config: Option<(TensorUpdateMode, [u32; 6])> = None;
        let mut committed_step = inner.training_step;
        unsafe {
            // Validate and materialize the complete batch before the first
            // native mutation. This prevents a bad tail item from committing a
            // valid prefix.
            for update in &request.updates {
                let mode = TensorUpdateMode::try_from(update.mode).map_err(|_| {
                    err(
                        format!("unsupported tensor update mode {}", update.mode),
                        INVALID_ARGUMENT,
                    )
                })?;
                let tensor = update
                    .tensor
                    .as_ref()
                    .ok_or_else(|| err("missing update tensor", INVALID_ARGUMENT))?;
                if !seen.insert(tensor.name.as_str()) {
                    return Err(err(
                        format!("tensor update target {} is duplicated", tensor.name),
                        INVALID_ARGUMENT,
                    ));
                }
                if tensor.quant.is_some() || tensor.initializer.is_some() {
                    return Err(err(
                        format!("tensor update {} cannot carry quantization/initializer metadata", tensor.name),
                        INVALID_ARGUMENT,
                    ));
                }
                let spec = tensor_spec_native(
                    &tensor.name,
                    TENSOR_ACCESS_READABLE | TENSOR_ACCESS_WRITABLE,
                )?;
                if !tensor.shape.is_empty() && tensor.shape != spec.shape {
                    return Err(err(
                        format!(
                            "tensor {} shape mismatch: got {:?}, want {:?}",
                            tensor.name, tensor.shape, spec.shape
                        ),
                        INVALID_ARGUMENT,
                    ));
                }
                let cname = CString::new(tensor.name.as_str())
                    .map_err(|_| err("bad tensor name", INVALID_ARGUMENT))?;
                if volvoxai_engine_is_graph_input(cname.as_ptr()) == 1 {
                    return Err(err(
                        format!(
                            "tensor update {} cannot mutate a graph input; use the input API",
                            tensor.name
                        ),
                        INVALID_ARGUMENT,
                    ));
                }
                let native_dtype = proto_to_native_dtype(tensor.dtype).ok_or_else(|| {
                    err(
                        format!("tensor update does not support dtype {}", dtype_name(tensor.dtype)),
                        INVALID_ARGUMENT,
                    )
                })?;
                if tensor.dtype != spec.dtype || tensor.data.len() != spec.size_bytes as usize {
                    return Err(err(
                        format!(
                            "tensor {} payload does not match target dtype/size",
                            tensor.name
                        ),
                        INVALID_ARGUMENT,
                    ));
                }
                let params = optimizer_values(
                    update.optimizer.as_ref(),
                    base_optimizer.as_ref(),
                    base_step,
                )?;
                if matches!(
                    mode,
                    TensorUpdateMode::TensorUpdateSgd | TensorUpdateMode::TensorUpdateAdamw
                ) {
                    if volvoxai_engine_is_model_weight(cname.as_ptr()) != 1 {
                        return Err(err(
                            format!(
                                "tensor {} optimizer target must be a safetensors-backed model weight",
                                tensor.name
                            ),
                            INVALID_ARGUMENT,
                        ));
                    }
                    if params.6 <= inner.training_step {
                        return Err(err(
                            format!(
                                "tensor {} optimizer step must be greater than {}",
                                tensor.name, inner.training_step
                            ),
                            FAILED_PRECONDITION,
                        ));
                    }
                    if optimizer_batch_step.is_some_and(|step| step != params.6) {
                        return Err(err(
                            "ApplyTensorUpdates optimizer updates must use one coherent step",
                            INVALID_ARGUMENT,
                        ));
                    }
                    optimizer_batch_step = Some(params.6);
                    let config = (
                        mode,
                        [
                            params.0.to_bits(),
                            params.1.to_bits(),
                            params.2.to_bits(),
                            params.3.to_bits(),
                            params.4.to_bits(),
                            params.5.to_bits(),
                        ],
                    );
                    if optimizer_batch_config.is_some_and(|existing| existing != config) {
                        return Err(err(
                            "ApplyTensorUpdates optimizer updates must use one mode/hyperparameter set",
                            INVALID_ARGUMENT,
                        ));
                    }
                    optimizer_batch_config = Some(config);
                    committed_step = params.6;
                }
                let payload = if tensor.dtype == DATA_TYPE_F32 {
                    let values = f32_payload(&tensor.name, &tensor.data)?;
                    if values.iter().any(|value| !value.is_finite()) {
                        return Err(err(
                            format!("tensor update {} contains a non-finite value", tensor.name),
                            INVALID_ARGUMENT,
                        ));
                    }
                    Some(values)
                } else {
                    None
                };
                match mode {
                    TensorUpdateMode::TensorUpdateAssign => {
                        // Any native dtype accepted by SetTensor is valid.
                    }
                    TensorUpdateMode::TensorUpdateAdd
                    | TensorUpdateMode::TensorUpdateSgd
                    | TensorUpdateMode::TensorUpdateAdamw => {
                        if tensor.dtype != DATA_TYPE_F32 || spec.dtype != DATA_TYPE_F32 {
                            return Err(err(
                                format!(
                                    "update mode {} requires F32 tensor {}, got payload {} target {}",
                                    update.mode,
                                    tensor.name,
                                    dtype_name(tensor.dtype),
                                    dtype_name(spec.dtype)
                                ),
                                INVALID_ARGUMENT,
                            ));
                        }
                    }
                }
                let original = read_tensor(&tensor.name)?.data;
                prepared.push(PreparedTensorUpdate {
                    tensor: tensor.clone(),
                    spec,
                    cname,
                    native_dtype,
                    payload,
                    optimizer: params,
                    original,
                    mode,
                });
            }

            let snapshot_base = std::env::temp_dir().join(format!(
                ".volvox-optimizer-{}",
                PERSIST_SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            let optimizer_snapshot = unique_sibling_path(&snapshot_base, "snapshot")?;
            let optimizer_snapshot_c =
                CString::new(optimizer_snapshot.to_string_lossy().as_bytes()).unwrap();
            if volvoxai_engine_save_optimizer_state(
                optimizer_snapshot_c.as_ptr(),
                inner.training_step,
            ) != 0
            {
                return Err(err("snapshot optimizer state failed", INTERNAL));
            }
            let rollback = |prepared: &[PreparedTensorUpdate]| -> Result<(), FfiError> {
                for update in prepared {
                    if volvoxai_engine_set_tensor_raw(
                        update.cname.as_ptr(),
                        update.native_dtype,
                        update.original.as_ptr() as *const c_void,
                        update.original.len(),
                    ) != 0
                    {
                        return Err(err("rollback tensor update failed", INTERNAL));
                    }
                }
                let mut restored_step: c_long = 0;
                if volvoxai_engine_load_optimizer_state(
                    optimizer_snapshot_c.as_ptr(),
                    &mut restored_step,
                ) != 0
                    || restored_step != inner.training_step
                {
                    return Err(err("rollback optimizer state failed", INTERNAL));
                }
                Ok(())
            };

            for update in &prepared {
                let rc = if update.mode == TensorUpdateMode::TensorUpdateAssign {
                    volvoxai_engine_set_tensor_raw(
                        update.cname.as_ptr(),
                        update.native_dtype,
                        update.tensor.data.as_ptr() as *const c_void,
                        update.tensor.data.len(),
                    )
                } else {
                    let payload = update.payload.as_ref().unwrap();
                    let (lr, beta1, beta2, eps, weight_decay, max_grad_norm, step) =
                        update.optimizer;
                    volvoxai_engine_apply_tensor_update_f32(
                        update.cname.as_ptr(),
                        payload.as_ptr(),
                        payload.len() as c_long,
                        tensor_update_mode_to_native(update.mode),
                        lr,
                        beta1,
                        beta2,
                        eps,
                        weight_decay,
                        max_grad_norm,
                        step as c_long,
                    )
                };
                if rc != 0 {
                    let rollback_result = rollback(&prepared);
                    let _ = fs::remove_file(&optimizer_snapshot);
                    rollback_result?;
                    return Err(err(
                        format!("failed to apply tensor update {}", update.tensor.name),
                        INTERNAL,
                    ));
                }
            }

            let persistence_result: Result<Vec<SafetensorsInfo>, FfiError> = if persist {
                (|| -> Result<Vec<SafetensorsInfo>, FfiError> {
                let inspect = native_inspection(false, false, true, false)?;
                let loaded_weights = inspect
                    .get("weights")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                if flush_paths.len() > loaded_weights.len() {
                    Err(err(
                        "ApplyTensorUpdates supplied more paths than loaded weight shards",
                        INVALID_ARGUMENT,
                    ))
                } else {
                    let targets = loaded_weights
                        .iter()
                        .enumerate()
                        .map(|(index, weight)| {
                            flush_paths
                                .get(index)
                                .filter(|path| !path.is_empty())
                                .cloned()
                                .or_else(|| weight.get("path").and_then(Value::as_str).map(str::to_string))
                                .filter(|path| !path.is_empty())
                                .ok_or_else(|| err("weight shard has no persistence path", INVALID_ARGUMENT))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let mut staged = Vec::with_capacity(targets.len());
                    for (index, target) in targets.iter().enumerate() {
                        let target = PathBuf::from(target);
                        let stage = unique_sibling_path(&target, "stage")?;
                        let stage_c = CString::new(stage.to_string_lossy().as_bytes()).unwrap();
                        if volvoxai_engine_save_weight_file(index as c_int, stage_c.as_ptr()) != 0 {
                            for (path, _) in &staged {
                                let _ = fs::remove_file(path);
                            }
                            return Err(err(format!("flush weight file {index} failed"), INTERNAL));
                        }
                        staged.push((stage, target));
                    }
                    commit_staged_files(staged)?;
                    targets
                        .iter()
                        .map(|path| read_safetensors_info(path, true))
                        .collect()
                }
                })()
            } else {
                Ok(Vec::new())
            };
            let weights = match persistence_result {
                Ok(weights) => weights,
                Err(error) => {
                    let rollback_result = rollback(&prepared);
                    let _ = fs::remove_file(&optimizer_snapshot);
                    rollback_result?;
                    return Err(error);
                }
            };
            let _ = fs::remove_file(&optimizer_snapshot);
            inner.training_step = committed_step;
            if let Some((mode, bits)) = optimizer_batch_config {
                inner.training_optimizer = Some(OptimizerOptions {
                    learning_rate: f32::from_bits(bits[0]),
                    beta1: Some(f32::from_bits(bits[1])),
                    beta2: Some(f32::from_bits(bits[2])),
                    epsilon: Some(f32::from_bits(bits[3])),
                    weight_decay: Some(f32::from_bits(bits[4])),
                    max_grad_norm: Some(f32::from_bits(bits[5])),
                    step: None,
                });
                inner.training_update_mode = Some(mode as i32);
            }

            Ok(ApplyTensorUpdatesResponse {
                updated_tensors: prepared.into_iter().map(|update| update.spec).collect(),
                weight_files: weights,
                step: committed_step,
            })
        }
    }

    fn train_step(&self, request: TrainStepRequest) -> Result<TrainStepResponse, FfiError> {
        self.train_step_impl(request)
    }

    fn seq2_seq_train_step(
        &self,
        request: Seq2SeqTrainStepRequest,
    ) -> Result<TrainStepResponse, FfiError> {
        let ignore_id = request.ignore_id.unwrap_or(c_int::MIN);
        if request.batch_size <= 0 || request.source_length <= 0 || request.target_length <= 0 {
            return Err(err(
                "Seq2SeqTrainStep batch_size and sequence lengths must be positive",
                INVALID_ARGUMENT,
            ));
        }
        if request.encoder_tokens_input.is_empty() || request.decoder_tokens_input.is_empty() {
            return Err(err(
                "Seq2SeqTrainStep requires encoder_tokens_input and decoder_tokens_input",
                INVALID_ARGUMENT,
            ));
        }
        if request.encoder_tokens_input == request.decoder_tokens_input {
            return Err(err(
                "Seq2SeqTrainStep encoder and decoder token inputs must be distinct",
                INVALID_ARGUMENT,
            ));
        }
        if request.bos_id < 0 || request.pad_id < 0 || request.bos_id == request.pad_id {
            return Err(err(
                "Seq2SeqTrainStep requires distinct non-negative bos_id and pad_id",
                INVALID_ARGUMENT,
            ));
        }
        let batch = request.batch_size as usize;
        let source_length = request.source_length as usize;
        let target_length = request.target_length as usize;
        let source_count = batch
            .checked_mul(source_length)
            .ok_or_else(|| err("Seq2SeqTrainStep source shape is too large", INVALID_ARGUMENT))?;
        let target_count = batch
            .checked_mul(target_length)
            .ok_or_else(|| err("Seq2SeqTrainStep target shape is too large", INVALID_ARGUMENT))?;
        if request.source_ids.len() != source_count || request.target_ids.len() != target_count {
            return Err(err(
                format!(
                    "Seq2SeqTrainStep rectangular payload mismatch: source got {}, want {}; target got {}, want {}",
                    request.source_ids.len(),
                    source_count,
                    request.target_ids.len(),
                    target_count
                ),
                INVALID_ARGUMENT,
            ));
        }
        if request.source_ids.iter().any(|id| *id < 0) {
            return Err(err(
                "Seq2SeqTrainStep source token ids must be non-negative",
                INVALID_ARGUMENT,
            ));
        }
        if request
            .target_ids
            .iter()
            .any(|id| *id < 0 && *id != ignore_id)
        {
            return Err(err(
                "Seq2SeqTrainStep target ids must be non-negative or ignore_id",
                INVALID_ARGUMENT,
            ));
        }

        let mut decoder_ids = vec![request.pad_id; target_count];
        let mut labels = Vec::with_capacity(target_count);
        let mut encoder_mask = Vec::with_capacity(source_count);
        let mut decoder_mask = vec![0i32; target_count];
        let mut encoder_positions = Vec::with_capacity(source_count);
        let mut decoder_positions = Vec::with_capacity(target_count);
        for id in &request.source_ids {
            encoder_mask.push(i32::from(*id != request.pad_id));
        }
        for row in 0..batch {
            encoder_positions.extend((0..source_length).map(|position| position as i32));
            decoder_positions.extend((0..target_length).map(|position| position as i32));
            let base = row * target_length;
            decoder_ids[base] = request.bos_id;
            decoder_mask[base] = 1;
            for column in 0..target_length {
                let id = request.target_ids[base + column];
                labels.push(if id == request.pad_id { ignore_id } else { id } as i64);
                if column + 1 < target_length {
                    let shifted = if id == request.pad_id || id == ignore_id {
                        request.pad_id
                    } else {
                        id
                    };
                    decoder_ids[base + column + 1] = shifted;
                    decoder_mask[base + column + 1] = i32::from(shifted != request.pad_id);
                }
            }
        }
        let source_shape = vec![request.batch_size as i64, request.source_length as i64];
        let target_shape = vec![request.batch_size as i64, request.target_length as i64];
        let mut inputs = vec![
            i32_input_tensor(
                request.encoder_tokens_input,
                source_shape.clone(),
                &request.source_ids,
            ),
            i32_input_tensor(
                request.decoder_tokens_input,
                target_shape.clone(),
                &decoder_ids,
            ),
        ];
        if let Some(name) = request.encoder_mask_input {
            if name.is_empty() {
                return Err(err(
                    "Seq2SeqTrainStep encoder_mask_input cannot be empty when present",
                    INVALID_ARGUMENT,
                ));
            }
            inputs.push(i32_input_tensor(name, source_shape.clone(), &encoder_mask));
        }
        if let Some(name) = request.decoder_mask_input {
            if name.is_empty() {
                return Err(err(
                    "Seq2SeqTrainStep decoder_mask_input cannot be empty when present",
                    INVALID_ARGUMENT,
                ));
            }
            inputs.push(i32_input_tensor(name, target_shape.clone(), &decoder_mask));
        }
        if let Some(name) = request.encoder_positions_input {
            if name.is_empty() {
                return Err(err(
                    "Seq2SeqTrainStep encoder_positions_input cannot be empty when present",
                    INVALID_ARGUMENT,
                ));
            }
            inputs.push(i32_input_tensor(name, source_shape.clone(), &encoder_positions));
        }
        if let Some(name) = request.decoder_positions_input {
            if name.is_empty() {
                return Err(err(
                    "Seq2SeqTrainStep decoder_positions_input cannot be empty when present",
                    INVALID_ARGUMENT,
                ));
            }
            inputs.push(i32_input_tensor(name, target_shape.clone(), &decoder_positions));
        }
        let mut names = std::collections::HashSet::new();
        if inputs.iter().any(|input| !names.insert(input.name.as_str())) {
            return Err(err(
                "Seq2SeqTrainStep input names must be unique",
                INVALID_ARGUMENT,
            ));
        }
        self.train_step_impl(TrainStepRequest {
            model_id: request.model_id,
            inputs,
            logits_tensor: request.logits_tensor,
            target_ids: labels,
            trainable_tensors: request.trainable_tensors,
            optimizer: request.optimizer,
            update_mode: request.update_mode,
            persist_safetensors: request.persist_safetensors,
            weight_paths: request.weight_paths,
            last_token: None,
            ignore_id: Some(ignore_id),
            losses: Vec::new(),
            accumulation: None,
        })
    }

    fn train_lo_ra_step(
        &self,
        request: LoRaTrainStepRequest,
    ) -> Result<LoRaTrainStepResponse, FfiError> {
        let response = self.train_step_impl(TrainStepRequest {
            model_id: request.model_id,
            inputs: request.inputs,
            logits_tensor: request.logits_tensor,
            target_ids: request.target_ids,
            trainable_tensors: request.trainable_tensors,
            optimizer: request.optimizer,
            update_mode: request.update_mode,
            persist_safetensors: request.persist_safetensors,
            weight_paths: request.weight_paths,
            last_token: request.last_token,
            ignore_id: request.ignore_id,
            losses: Vec::new(),
            accumulation: None,
        })?;
        Ok(LoRaTrainStepResponse {
            loss: response.loss,
            correct: response.correct,
            examples: response.examples,
            updated_tensors: response.updated_tensors,
            weight_files: response.weight_files,
            step: response.step,
        })
    }
    fn patch_graph(&self, request: PatchGraphRequest) -> Result<PatchGraphResponse, FfiError> {
        let persist_config = request.persist_config;
        let _persistence = persist_config.then(|| PERSISTENCE_COMMIT.lock().unwrap());
        let _lifecycle = self.model_lifecycle.write().unwrap();
        let _exec = self.engine_exec.lock().unwrap();
        let _admin = self.adapter_admin.lock().unwrap();
        let mut inner = self.inner.lock().unwrap();
        require(&inner, &request.model_id)?;
        require_unmerged(&inner, "PatchGraph")?;
        require_no_pending_accumulation(&inner, "PatchGraph")?;
        let _backend_call = begin_backend_call(inner.model_backend, &None, false)?;
        if !inner.adapters.is_empty() {
            return Err(err(
                "PatchGraph is not allowed while adapter versions are loaded; remove them first",
                FAILED_PRECONDITION,
            ));
        }
        if !inner.allow_graph_patches {
            return Err(err(
                "PatchGraph requires CreateModel or LoadModelRequest.edit.allow_graph_patches",
                PERMISSION_DENIED,
            ));
        }

        let previous_info = inner.info.clone().unwrap_or_default();
        let previous_output_names: Vec<String> = previous_info
            .outputs
            .iter()
            .map(|output| output.name.clone())
            .collect();
        let declared_update = request.declared_outputs.map(|update| update.names);
        let desired_output_names = declared_update
            .as_ref()
            .cloned()
            .unwrap_or_else(|| previous_output_names.clone());
        let pre_patch = unsafe { native_inspection(true, true, false, true)? };
        let (prepared, structural) =
            prepare_graph_patches(&pre_patch, request.patches, &desired_output_names)?;
        let reoptimize = request.reoptimize.unwrap_or(false);
        let rebuild = request.rebuild_plan.unwrap_or(false) || structural || reoptimize;
        if prepared.is_empty() && (rebuild || reoptimize) {
            return Err(err(
                "PatchGraph rebuild_plan/reoptimize requires at least one node patch",
                INVALID_ARGUMENT,
            ));
        }
        if persist_config {
            if inner.config_path.is_empty() {
                return Err(err(
                    "PatchGraph persist_config requires a loaded config path",
                    INVALID_ARGUMENT,
                ));
            }
            CString::new(inner.config_path.as_str())
                .map_err(|_| err("bad config path", INVALID_ARGUMENT))?;
        }

        let invoked_native = !prepared.is_empty() || declared_update.is_some();
        if invoked_native {
            let entries: Vec<Value> = prepared
                .iter()
                .map(|patch| {
                    json!({
                        "node_index": patch.node_index,
                        "mode": patch.mode as i32,
                        "patch": patch.patch.clone(),
                    })
                })
                .collect();
            let mut transaction = serde_json::Map::new();
            transaction.insert("patches".into(), Value::Array(entries));
            if let Some(names) = declared_update.as_ref() {
                transaction.insert("declared_outputs".into(), json!(names));
            }
            if persist_config {
                transaction.insert(
                    "persist_config_path".into(),
                    json!(inner.config_path.as_str()),
                );
            }
            let encoded = CString::new(Value::Object(transaction).to_string())
                .map_err(|_| err("bad graph patch transaction JSON", INVALID_ARGUMENT))?;
            let result = unsafe {
                volvoxai_engine_patch_graph_json(
                    encoded.as_ptr(),
                    rebuild as c_int,
                    reoptimize as c_int,
                )
            };
            if result != 0 {
                return Err(err(
                    "graph patch transaction failed; no patches were committed",
                    INVALID_ARGUMENT,
                ));
            }
        }

        let post_patch = if invoked_native {
            unsafe { native_inspection(true, true, false, true)? }
        } else {
            pre_patch
        };
        let refreshed_outputs = output_specs_from_inspection(&post_patch, &desired_output_names)?;
        let mut info = previous_info;
        if let Some(num_ops) = post_patch.get("num_ops").and_then(Value::as_i64) {
            info.num_ops = num_ops as i32;
        }
        info.outputs = refreshed_outputs;

        let mut changed_tensor_names: std::collections::HashSet<String> = prepared
            .iter()
            .flat_map(|patch| patch.changed_outputs.iter().cloned())
            .collect();
        if declared_update.is_some() {
            let old: std::collections::HashSet<_> = previous_output_names.iter().cloned().collect();
            let new: std::collections::HashSet<_> = desired_output_names.iter().cloned().collect();
            changed_tensor_names.extend(old.symmetric_difference(&new).cloned());
        }
        let mut changed_tensor_names: Vec<_> = changed_tensor_names.into_iter().collect();
        changed_tensor_names.sort();
        let changed_node_indices = prepared.iter().map(|patch| patch.changed_index).collect();

        // Native graph/config persistence committed transactionally, so runtime
        // metadata can now follow the authoritative state.
        inner.info = Some(info.clone());
        if persist_config && !invoked_native {
            let cfg = CString::new(inner.config_path.as_str())
                .map_err(|_| err("bad config path", INVALID_ARGUMENT))?;
            if unsafe { volvoxai_engine_save_config(cfg.as_ptr()) } != 0 {
                return Err(err("persisting patched config failed", INTERNAL));
            }
        }
        Ok(PatchGraphResponse {
            info: Some(info),
            changed_node_indices,
            changed_tensor_names,
        })
    }
    fn classify(&self, request: ClassifyRequest) -> Result<ClassifyResponse, FfiError> {
        let _exec = self.engine_exec.lock().unwrap();
        let inner = self.inner.lock().unwrap();
        require(&inner, &request.model_id)?;
        let _backend_call = begin_backend_call(inner.model_backend, &request.exec, false)?;
        unsafe {
            let img = request
                .image
                .ok_or_else(|| err("missing image", INVALID_ARGUMENT))?;
            let t = match img.source {
                Some(image::Source::Tensor(t)) => t,
                _ => {
                    return Err(err(
                        "encoded-image decode is a TODO; pass Image.tensor (NHWC)",
                        UNIMPLEMENTED,
                    ))
                }
            };
            let in_name = if img.input_name.is_empty() {
                first_graph_input_name()?
            } else {
                img.input_name.clone()
            };
            let mut input = t;
            input.name = in_name;
            let _execution_row = ExecutionRowGuard::new(-1)?;
            copy_tensor_to_input(&input)?;
            if volvoxai_engine_forward() != 0 {
                return Err(err("volvoxai_engine_forward failed", INTERNAL));
            }

            let logits_name = match request
                .logits_tensor
                .as_ref()
                .filter(|name| !name.is_empty())
            {
                Some(name) => name.clone(),
                None => first_graph_output_name()?,
            };
            let logits = read_tensor_f32_values(&logits_name)?;
            let sl = logits.as_slice();
            let ln = sl.len();
            let k = (request.top_k.unwrap_or(5).max(0) as usize).min(ln);
            let mut idx: Vec<usize> = (0..ln).collect();
            idx.sort_by(|&a, &b| {
                sl[b]
                    .partial_cmp(&sl[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let classes = idx
                .into_iter()
                .take(k)
                .map(|i| ClassScore {
                    index: i as i32,
                    score: sl[i],
                    label: request.labels.get(i).cloned().unwrap_or_default(),
                })
                .collect();
            Ok(ClassifyResponse {
                classes,
                timing: None,
            })
        }
    }

    fn detect(&self, _request: DetectRequest) -> Result<DetectResponse, FfiError> {
        Err(err(
            "Detect not yet ported from examples/native_task_cli/main.c",
            UNIMPLEMENTED,
        ))
    }

    fn recognize_ctc(&self, _request: CtcRequest) -> Result<CtcResponse, FfiError> {
        Err(err(
            "RecognizeCtc not yet ported from examples/native_task_cli/main.c",
            UNIMPLEMENTED,
        ))
    }
}

// Register the plugin when the shared library is loaded, and also expose an
// explicit init for hosts that call one. register uses OnceLock::set, so the
// second call is a harmless no-op.
#[ctor::ctor]
fn on_load() {
    register_volvox_ai_service_plugin(Plugin::new());
}

#[no_mangle]
pub extern "C" fn VolvoxAI_Init() {
    register_volvox_ai_service_plugin(Plugin::new());
}

#[cfg(test)]
mod adapter_contract_tests {
    use super::*;
    use prost::Message;

    #[derive(Clone, PartialEq, ::prost::Message)]
    struct LegacyAdapterMetadata {
        #[prost(map = "string, string", tag = "5")]
        metadata: std::collections::HashMap<String, String>,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    struct LegacyGraphNodeMaps {
        #[prost(map = "string, string", tag = "5")]
        inputs: std::collections::HashMap<String, String>,
        #[prost(map = "string, string", tag = "6")]
        outputs: std::collections::HashMap<String, String>,
        #[prost(map = "string, message", tag = "7")]
        output_shapes: std::collections::HashMap<String, TensorShape>,
    }

    #[test]
    fn repeated_entries_preserve_legacy_map_wire_format() {
        let legacy_metadata = LegacyAdapterMetadata {
            metadata: std::collections::HashMap::from([
                ("alpha".to_string(), "one".to_string()),
                ("beta".to_string(), "two".to_string()),
            ]),
        };
        let decoded = AdapterManifest::decode(legacy_metadata.encode_to_vec().as_slice()).unwrap();
        assert_eq!(
            string_entries_to_map(&decoded.metadata),
            legacy_metadata.metadata
        );

        let repeated_metadata = AdapterManifest {
            metadata: vec![
                StringEntry {
                    key: "alpha".to_string(),
                    value: "one".to_string(),
                },
                StringEntry {
                    key: "beta".to_string(),
                    value: "two".to_string(),
                },
            ],
            ..Default::default()
        };
        let decoded = LegacyAdapterMetadata::decode(
            repeated_metadata.encode_to_vec().as_slice(),
        )
        .unwrap();
        assert_eq!(decoded.metadata, legacy_metadata.metadata);

        let legacy_graph = LegacyGraphNodeMaps {
            inputs: std::collections::HashMap::from([(
                "x".to_string(),
                "input".to_string(),
            )]),
            outputs: std::collections::HashMap::from([(
                "y".to_string(),
                "output".to_string(),
            )]),
            output_shapes: std::collections::HashMap::from([(
                "y".to_string(),
                TensorShape { dims: vec![2, 3] },
            )]),
        };
        let decoded = GraphNode::decode(legacy_graph.encode_to_vec().as_slice()).unwrap();
        assert_eq!(string_entries_to_map(&decoded.inputs), legacy_graph.inputs);
        assert_eq!(string_entries_to_map(&decoded.outputs), legacy_graph.outputs);
        assert_eq!(
            tensor_shape_entries_to_map(&decoded.output_shapes),
            legacy_graph.output_shapes
        );

        let repeated_graph = GraphNode {
            inputs: string_entries_from_map(legacy_graph.inputs.clone()),
            outputs: string_entries_from_map(legacy_graph.outputs.clone()),
            output_shapes: tensor_shape_entries_from_map(legacy_graph.output_shapes.clone()),
            ..Default::default()
        };
        let decoded = LegacyGraphNodeMaps::decode(repeated_graph.encode_to_vec().as_slice())
            .unwrap();
        assert_eq!(decoded.inputs, legacy_graph.inputs);
        assert_eq!(decoded.outputs, legacy_graph.outputs);
        assert_eq!(decoded.output_shapes, legacy_graph.output_shapes);
    }

    #[test]
    fn repeated_entry_helpers_use_map_compatible_last_value_semantics() {
        let mut entries = vec![
            StringEntry {
                key: "key".to_string(),
                value: "old".to_string(),
            },
            StringEntry {
                key: "key".to_string(),
                value: "new".to_string(),
            },
        ];
        assert_eq!(string_entry_value(&entries, "key"), Some("new"));
        assert_eq!(string_entries_to_map(&entries)["key"], "new");
        string_entry_upsert(&mut entries, "key", "newest");
        assert_eq!(entries.len(), 1);
        assert_eq!(string_entry_value(&entries, "key"), Some("newest"));

        let left = AdapterManifest {
            metadata: vec![
                StringEntry {
                    key: "a".to_string(),
                    value: "1".to_string(),
                },
                StringEntry {
                    key: "b".to_string(),
                    value: "2".to_string(),
                },
            ],
            ..Default::default()
        };
        let mut right = left.clone();
        right.metadata.reverse();
        assert!(adapter_manifests_equal(&left, &right));
        right.metadata[0].value = "changed".to_string();
        assert!(!adapter_manifests_equal(&left, &right));
    }

    #[test]
    fn every_op_enum_name_roundtrips_and_conflicts_are_rejected() {
        let mut names = std::collections::HashSet::new();
        for value in OpType::OpUnspecified as i32..=OpType::OpSelectiveScan as i32 {
            let Ok(op) = OpType::try_from(value) else {
                continue;
            };
            let name = canonical_op_name(value).unwrap();
            if op == OpType::OpUnspecified {
                assert_eq!(name, None);
            } else {
                let name = name.expect("every concrete OpType needs a canonical native name");
                assert!(names.insert(name), "duplicate canonical op name {name}");
                assert_eq!(op_type_from_name(name), value, "failed roundtrip for {op:?}");
            }
        }
        assert_eq!(
            op_type_from_name("GroupNorm"),
            OpType::OpGroupNorm as i32
        );
        assert_eq!(
            op_type_from_name("QGroupNorm"),
            OpType::OpQgroupNorm as i32
        );
        assert_eq!(
            resolve_op_name(Some(OpType::OpMatmul as i32), None, "test op", true).unwrap(),
            Some("MatMul".to_string())
        );
        assert!(resolve_op_name(
            Some(OpType::OpMatmul as i32),
            Some("Add"),
            "test op",
            true
        )
        .is_err());
        assert_eq!(
            resolve_op_name(
                Some(OpType::OpUnspecified as i32),
                Some("CustomOp"),
                "test op",
                true
            )
            .unwrap(),
            Some("CustomOp".to_string())
        );
        assert!(canonical_op_name(999).is_err());
    }

    #[test]
    fn every_dtype_and_native_dtype_mapping_roundtrips() {
        for value in DataType::Unspecified as i32..=DataType::U64 as i32 {
            let dtype = DataType::try_from(value).unwrap();
            assert_eq!(dtype_from_name(dtype_name(value)), value, "{dtype:?}");
            assert_eq!(dtype_bits(value) == 0, dtype == DataType::Unspecified);
        }
        assert_eq!(dtype_name(i32::MAX), "UNKNOWN");
        assert_eq!(dtype_from_name("UNKNOWN"), DataType::Unspecified as i32);

        for (dtype, native) in [
            (DataType::F32, T_F32),
            (DataType::I8, T_I8),
            (DataType::U8, T_U8),
            (DataType::I32, T_I32),
            (DataType::F16, T_F16),
        ] {
            assert_eq!(proto_to_native_dtype(dtype as i32), Some(native));
            assert_eq!(native_to_proto_dtype(native), dtype as i32);
        }
        assert_eq!(proto_to_native_dtype(DataType::F64 as i32), None);
        assert_eq!(native_to_proto_dtype(i32::MAX), DataType::Unspecified as i32);
    }

    #[test]
    fn adapter_and_update_enum_mappings_are_exhaustive() {
        assert_eq!(
            adapter_kind_name(AdapterKind::AdapterLora as i32),
            Some("lora")
        );
        assert_eq!(
            adapter_kind_from_value(Some(&json!("lora"))),
            AdapterKind::AdapterLora as i32
        );
        assert_eq!(adapter_kind_name(AdapterKind::Unspecified as i32), None);

        for (layout, name) in [
            (AdapterMatrixLayout::Peft, "peft"),
            (AdapterMatrixLayout::Canonical, "din_r_r_dout"),
        ] {
            assert_eq!(adapter_layout_name(layout as i32), Some(name));
            assert_eq!(adapter_layout_from_value(Some(&json!(name))), layout as i32);
        }
        assert_eq!(
            adapter_layout_name(AdapterMatrixLayout::Unspecified as i32),
            None
        );

        for (role, key) in [
            (AdapterTensorRole::AdapterTensorA, "a"),
            (AdapterTensorRole::AdapterTensorB, "b"),
        ] {
            assert_eq!(adapter_role_json_key(role as i32).unwrap(), key);
            assert_eq!(adapter_role_from_key(key), role as i32);
        }
        assert!(adapter_role_json_key(AdapterTensorRole::AdapterTensorUnspecified as i32).is_err());

        for (mode, native) in [
            (AdapterUpdateMode::AdapterUpdateAssign, NATIVE_ADAPTER_UPDATE_ASSIGN),
            (AdapterUpdateMode::AdapterUpdateAdd, NATIVE_ADAPTER_UPDATE_ADD),
        ] {
            assert_eq!(adapter_update_mode_to_native(mode as i32), Some(native));
        }
        assert_eq!(adapter_update_mode_to_native(i32::MAX), None);

        for (mode, native) in [
            (TensorUpdateMode::TensorUpdateAssign, NATIVE_TENSOR_UPDATE_ASSIGN),
            (TensorUpdateMode::TensorUpdateAdd, NATIVE_TENSOR_UPDATE_ADD),
            (TensorUpdateMode::TensorUpdateSgd, NATIVE_TENSOR_UPDATE_SGD),
            (TensorUpdateMode::TensorUpdateAdamw, NATIVE_TENSOR_UPDATE_ADAMW),
        ] {
            assert_eq!(tensor_update_mode_to_native(mode), native);
        }

        assert_eq!(parse_node_patch_mode(None).unwrap(), NodePatchMode::Merge);
        assert_eq!(
            parse_node_patch_mode(Some(NodePatchMode::Unspecified as i32)).unwrap(),
            NodePatchMode::Merge
        );
        for mode in [
            NodePatchMode::Merge,
            NodePatchMode::Replace,
            NodePatchMode::InsertBefore,
            NodePatchMode::InsertAfter,
            NodePatchMode::Delete,
        ] {
            assert_eq!(parse_node_patch_mode(Some(mode as i32)).unwrap(), mode);
        }
        assert!(parse_node_patch_mode(Some(i32::MAX)).is_err());
    }

    fn binding(role: i32, name: &str) -> AdapterTensorBinding {
        AdapterTensorBinding {
            role,
            tensor_name: name.to_string(),
            spec: None,
        }
    }

    fn manifest(kind: i32, tensors: Vec<AdapterTensorBinding>) -> AdapterManifest {
        AdapterManifest {
            format: ADAPTER_MANIFEST_FORMAT.to_string(),
            adapter_id: "contract".to_string(),
            kind,
            targets: vec![AdapterTarget {
                target_id: "target".to_string(),
                op_id: String::new(),
                base_tensor: "weight".to_string(),
                adapter_layout: AdapterMatrixLayout::Canonical as i32,
                rank: 4,
                alpha: 8.0,
                scale: None,
                tensors,
            }],
            metadata: Default::default(),
        }
    }

    #[test]
    fn only_standard_lora_kind_is_accepted() {
        let valid = manifest(
            AdapterKind::AdapterLora as i32,
            vec![
                binding(AdapterTensorRole::AdapterTensorA as i32, "a"),
                binding(AdapterTensorRole::AdapterTensorB as i32, "b"),
            ],
        );
        validate_adapter_manifest(&valid, None).unwrap();

        let unsupported = manifest(
            2,
            vec![
                binding(AdapterTensorRole::AdapterTensorA as i32, "a"),
                binding(AdapterTensorRole::AdapterTensorB as i32, "b"),
            ],
        );
        assert!(validate_adapter_manifest(&unsupported, None).is_err());
    }

    #[test]
    fn zero_alpha_is_rejected_before_native_manifest_serialization() {
        let mut manifest = manifest(
            AdapterKind::AdapterLora as i32,
            vec![
                binding(AdapterTensorRole::AdapterTensorA as i32, "a"),
                binding(AdapterTensorRole::AdapterTensorB as i32, "b"),
            ],
        );
        manifest.targets[0].alpha = 0.0;
        apply_adapter_scale_default(&mut manifest);
        assert!(validate_adapter_manifest(&manifest, None).is_err());
    }

    #[test]
    fn run_routing_preserves_absent_vs_explicit_base() {
        let absent = RunRequest::default();
        let absent = RunRequest::decode(absent.encode_to_vec().as_slice()).unwrap();
        assert!(absent.adapters.is_none());

        let explicit_base = RunRequest {
            adapters: Some(AdapterSelection::default()),
            ..Default::default()
        };
        let explicit_base = RunRequest::decode(explicit_base.encode_to_vec().as_slice()).unwrap();
        assert!(explicit_base.adapters.is_some());
        assert!(explicit_base.adapters.unwrap().routes.is_empty());
    }

    #[test]
    fn capabilities_are_lora_only_and_do_not_claim_gpu_grouping() {
        let capabilities = Plugin::new().get_capabilities(Empty {}).unwrap();
        let adapters = capabilities.adapters.unwrap();
        assert_eq!(adapters.kinds, vec![AdapterKind::AdapterLora as i32]);
        assert!(!adapters.gpu_grouped_matmul);
    }

    #[test]
    fn backend_selection_is_model_scoped_and_rejects_unknown_values() {
        assert_eq!(model_backend(&None).unwrap(), RuntimeBackend::Cpu);
        for (proto, runtime) in [
            (Backend::Unspecified, RuntimeBackend::Cpu),
            (Backend::Cpu, RuntimeBackend::Cpu),
            (Backend::Vulkan, RuntimeBackend::Vulkan),
            (Backend::Opengl, RuntimeBackend::OpenGl),
            (Backend::Metal, RuntimeBackend::Metal),
            (Backend::Nnapi, RuntimeBackend::Nnapi),
        ] {
            assert_eq!(parse_backend(proto as i32).unwrap(), runtime);
        }
        for (native, runtime) in [
            (0, RuntimeBackend::Cpu),
            (1, RuntimeBackend::Vulkan),
            (2, RuntimeBackend::OpenGl),
            (3, RuntimeBackend::Metal),
            (4, RuntimeBackend::Nnapi),
        ] {
            assert_eq!(runtime.native(), native);
            assert_eq!(RuntimeBackend::from_native(native).unwrap(), runtime);
            assert_eq!(parse_backend(runtime.proto()).unwrap(), runtime);
        }
        assert_eq!(parse_backend(999).unwrap_err().grpc_code, INVALID_ARGUMENT);
        assert_eq!(
            RuntimeBackend::from_native(999).unwrap_err().grpc_code,
            INTERNAL
        );

        let error = begin_backend_call(
            RuntimeBackend::Vulkan,
            &Some(ExecOptions {
                backend: Some(RuntimeBackend::Cpu.proto()),
                ..Default::default()
            }),
            false,
        )
        .unwrap_err();
        assert_eq!(error.grpc_code, FAILED_PRECONDITION);

        let options = model_engine_options(
            RuntimeBackend::Vulkan,
            &Some(ExecOptions {
                backend: Some(RuntimeBackend::Vulkan.proto()),
                num_threads: Some(3),
                debug: Some(true),
                ..Default::default()
            }),
        )
        .unwrap();
        assert_eq!(options.backend, RuntimeBackend::Vulkan.native());
        assert_eq!(options.cpu_threads, 3);
        assert_eq!(options.debug, 1);
        assert_eq!(
            RuntimeBackend::from_native(options.backend).unwrap(),
            RuntimeBackend::Vulkan
        );
        assert!(model_engine_options(
            RuntimeBackend::Cpu,
            &Some(ExecOptions {
                num_threads: Some(-1),
                ..Default::default()
            }),
        )
        .is_err());
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn metal_is_rejected_by_non_macos_runtime_without_initializing_a_gpu() {
        let error = configure_engine_backend(RuntimeBackend::Metal, &None).unwrap_err();
        assert_eq!(error.grpc_code, UNIMPLEMENTED);
        assert!(error.message.contains("macOS runtime"));
    }

    #[test]
    fn lora_rejects_integer_base_weights() {
        validate_lora_base_dtype(T_F32, "weight").unwrap();
        validate_lora_base_dtype(T_F16, "weight").unwrap();
        assert!(validate_lora_base_dtype(T_I8, "weight").is_err());
        assert!(validate_lora_base_dtype(T_U8, "weight").is_err());
    }

    #[test]
    fn per_batch_route_count_matches_leading_input_dimension() {
        let reference = AdapterVersionRef {
            adapter_id: "mixed".to_string(),
            version_id: "mixed@v1".to_string(),
        };
        let selection = AdapterSelection {
            routes: vec![
                AdapterRoute::default(),
                AdapterRoute {
                    adapter: Some(reference.clone()),
                    scale: None,
                },
            ],
        };
        assert!(validate_route_batch_dims(&selection, &[2]).is_ok());
        assert!(validate_route_batch_dims(&selection, &[3]).is_err());
        assert!(validate_route_batch_dims(&selection, &[1, 2]).is_ok());
        assert!(validate_route_batch_dims(&selection, &[2, 3]).is_err());
        let plugin = Plugin::new();
        let mut inner = plugin.inner.lock().unwrap();
        inner.adapters.push(AdapterRecord {
            native_name: "mixed@v1".to_string(),
            info: AdapterInfo {
                adapter: Some(reference),
                ..Default::default()
            },
        });
        let resolved = resolve_adapter_selection(&inner, &selection).unwrap();
        assert_eq!(
            resolved,
            vec![(String::new(), 1.0), ("mixed@v1".to_string(), 1.0)]
        );
    }

    #[test]
    fn v1_rejects_unknown_layouts_roles_and_non_string_metadata() {
        let mut reserved_layout = manifest(
            AdapterKind::AdapterLora as i32,
            vec![
                binding(AdapterTensorRole::AdapterTensorA as i32, "a"),
                binding(AdapterTensorRole::AdapterTensorB as i32, "b"),
            ],
        );
        reserved_layout.targets[0].adapter_layout = AdapterMatrixLayout::Unspecified as i32;
        assert!(validate_adapter_manifest(&reserved_layout, None).is_err());

        let unsupported_role = manifest(
            AdapterKind::AdapterLora as i32,
            vec![
                binding(AdapterTensorRole::AdapterTensorA as i32, "a"),
                binding(AdapterTensorRole::AdapterTensorB as i32, "b"),
                binding(3, "x"),
            ],
        );
        assert!(validate_adapter_manifest(&unsupported_role, None).is_err());
        assert!(strict_metadata_map(Some(&json!({ "bad": 7 })), "test").is_err());
    }

    #[test]
    fn checkpoint_target_kind_cannot_override_manifest_kind() {
        let tensor = |name: &str, shape: Vec<i64>, start: i64, end: i64| SafetensorsTensorInfo {
            name: name.to_string(),
            shape,
            dtype: DATA_TYPE_F32,
            data_start: start,
            data_end: end,
            access_flags: TENSOR_ACCESS_READABLE,
        };
        let file = SafetensorsInfo {
            tensors: vec![
                tensor("a", vec![8, 4], 0, 128),
                tensor("b", vec![4, 12], 128, 320),
            ],
            ..Default::default()
        };
        let malformed = json!({
            "format": ADAPTER_MANIFEST_FORMAT,
            "adapter_id": "bad",
            "version_id": "bad@v1",
            "kind": "lora",
            "targets": [{
                "weight": "weight",
                "kind": "unsupported",
                "layout": "din_r_r_dout",
                "rank": 4,
                "alpha": 4,
                "a": "a",
                "b": "b"
            }]
        });
        assert!(manifest_from_json(&malformed.to_string(), &file).is_err());
    }

    #[test]
    fn checkpoint_tensor_descriptors_must_match_payload() {
        let tensor = |name: &str, shape: Vec<i64>, start: i64, end: i64| SafetensorsTensorInfo {
            name: name.to_string(),
            shape,
            dtype: DATA_TYPE_F32,
            data_start: start,
            data_end: end,
            access_flags: TENSOR_ACCESS_READABLE,
        };
        let file = SafetensorsInfo {
            tensors: vec![
                tensor("a", vec![8, 4], 0, 128),
                tensor("b", vec![4, 12], 128, 320),
            ],
            ..Default::default()
        };
        let malformed = json!({
            "format": ADAPTER_MANIFEST_FORMAT,
            "adapter_id": "bad-spec",
            "version_id": "bad-spec@v1",
            "kind": "lora",
            "targets": [{
                "weight": "weight",
                "kind": "lora",
                "layout": "din_r_r_dout",
                "rank": 4,
                "alpha": 4,
                "a": "a",
                "b": "b",
                "tensors": [
                    { "role": "a", "name": "a", "shape": [8, 4], "dtype": "F32" },
                    { "role": "b", "name": "b", "shape": [4, 11], "dtype": "F32" }
                ]
            }]
        });
        assert!(manifest_from_json(&malformed.to_string(), &file).is_err());
    }

    #[test]
    fn safetensors_coverage_rejects_payload_gaps_and_overlaps() {
        let tensor = |name: &str, start: i64, end: i64| SafetensorsTensorInfo {
            name: name.to_string(),
            shape: vec![1],
            dtype: DATA_TYPE_F32,
            data_start: start,
            data_end: end,
            access_flags: TENSOR_ACCESS_READABLE,
        };

        let valid = vec![tensor("a", 0, 4), tensor("b", 4, 8)];
        validate_safetensors_coverage(&valid, 8, "test").unwrap();

        let gap = vec![tensor("a", 0, 4), tensor("b", 5, 9)];
        let gap_error = validate_safetensors_coverage(&gap, 9, "test").unwrap_err();
        assert!(gap_error.message.contains("gap"));

        let overlap = vec![tensor("a", 0, 4), tensor("b", 3, 7)];
        let overlap_error = validate_safetensors_coverage(&overlap, 7, "test").unwrap_err();
        assert!(overlap_error.message.contains("overlap"));
    }

    #[test]
    fn peft_manifest_normalizes_to_canonical_f32_specs() {
        let mut peft = manifest(
            AdapterKind::AdapterLora as i32,
            vec![
                binding(AdapterTensorRole::AdapterTensorA as i32, "a"),
                binding(AdapterTensorRole::AdapterTensorB as i32, "b"),
            ],
        );
        peft.targets[0].adapter_layout = AdapterMatrixLayout::Peft as i32;
        peft.targets[0].tensors[0].spec = Some(TensorSpec {
            name: "a".to_string(),
            shape: vec![4, 8],
            dtype: DATA_TYPE_F16,
            access_flags: 0,
            size_bytes: 64,
        });
        peft.targets[0].tensors[1].spec = Some(TensorSpec {
            name: "b".to_string(),
            shape: vec![12, 4],
            dtype: DATA_TYPE_F16,
            access_flags: 0,
            size_bytes: 96,
        });
        let normalized = normalize_adapter_manifest(peft).unwrap();
        let target = &normalized.targets[0];
        assert_eq!(
            target.adapter_layout,
            AdapterMatrixLayout::Canonical as i32
        );
        assert_eq!(target.tensors[0].spec.as_ref().unwrap().shape, vec![8, 4]);
        assert_eq!(target.tensors[1].spec.as_ref().unwrap().shape, vec![4, 12]);
        assert!(target
            .tensors
            .iter()
            .all(|binding| binding.spec.as_ref().unwrap().dtype == DATA_TYPE_F32));
    }

    #[test]
    fn weighted_losses_are_prepared_for_the_native_abi() {
        let request = TrainStepRequest {
            losses: vec![
                CrossEntropyLoss {
                    name: "tokens".to_string(),
                    logits_tensor: "lm_logits".to_string(),
                    target_ids: vec![3, -100],
                    weight: Some(1.0),
                    ignore_id: Some(-100),
                    last_token: None,
                    normalizer: Some(12.0),
                },
                CrossEntropyLoss {
                    name: "router".to_string(),
                    logits_tensor: "router_logits".to_string(),
                    target_ids: vec![2],
                    weight: Some(0.1),
                    ignore_id: None,
                    last_token: Some(0),
                    normalizer: Some(4.0),
                },
            ],
            ..Default::default()
        };
        let prepared = prepare_cross_entropy_losses(&request, 4).unwrap();
        assert_eq!(prepared.len(), 2);
        assert_eq!(prepared[0].name, "tokens");
        assert_eq!(prepared[0].targets, vec![3, -100]);
        assert_eq!(prepared[0].ignore_index, -100);
        assert_eq!(prepared[0].normalizer, 12.0);
        assert_eq!(prepared[1].weight, 0.1);
        assert_eq!(prepared[1].last_token, 0);
    }

    #[test]
    fn accumulated_losses_require_explicit_full_window_normalizers() {
        let request = TrainStepRequest {
            losses: vec![CrossEntropyLoss {
                name: "tokens".to_string(),
                logits_tensor: "logits".to_string(),
                target_ids: vec![1],
                ..Default::default()
            }],
            ..Default::default()
        };
        let error = prepare_cross_entropy_losses(&request, 2).unwrap_err();
        assert!(error.message.contains("requires normalizer"));
        assert!(prepare_cross_entropy_losses(&request, 1).is_ok());
    }

    #[test]
    fn legacy_and_repeated_loss_fields_are_mutually_exclusive() {
        let request = TrainStepRequest {
            logits_tensor: "legacy".to_string(),
            target_ids: vec![0],
            losses: vec![CrossEntropyLoss {
                name: "new".to_string(),
                logits_tensor: "new_logits".to_string(),
                target_ids: vec![1],
                ..Default::default()
            }],
            ..Default::default()
        };
        let error = prepare_cross_entropy_losses(&request, 1).unwrap_err();
        assert!(error.message.contains("not both"));
    }

    #[test]
    fn multi_loss_accumulation_fields_survive_protobuf_roundtrip() {
        let response = TrainStepResponse {
            loss: 1.25,
            losses: vec![CrossEntropyLossMetric {
                name: "router".to_string(),
                loss: 0.25,
                correct: 2,
                examples: 3,
                normalizer: 8.0,
            }],
            accumulated_microbatches: 3,
            accumulation_steps: 4,
            update_applied: false,
            ..Default::default()
        };
        let decoded = TrainStepResponse::decode(response.encode_to_vec().as_slice()).unwrap();
        assert_eq!(decoded.losses[0].name, "router");
        assert_eq!(decoded.accumulated_microbatches, 3);
        assert_eq!(decoded.accumulation_steps, 4);
        assert!(!decoded.update_applied);
    }
}
