//! In-process Synurang FFI over VolvoxAI's public opaque full-profile lifecycle.

use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fs;
use std::path::Path;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex, MutexGuard,
};

// prost-generated messages for package `volvoxai.runtime`.
pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/volvoxai.runtime.rs"));
}

// Synurang-generated dispatcher and plugin trait.
#[path = "gen/volvoxai_ffi_plugin.rs"]
mod ffi;

mod abi;

use abi::*;
use ffi::{register_runtime_service_plugin, FfiError, RuntimeServicePlugin};
use pb::*;

const INVALID_ARGUMENT: i32 = 3;
const NOT_FOUND: i32 = 5;
const FAILED_PRECONDITION: i32 = 9;
const ABORTED: i32 = 10;
const INTERNAL: i32 = 13;
const UNAVAILABLE: i32 = 14;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn service_error(message: impl Into<String>, grpc_code: i32) -> FfiError {
    let native_status = match grpc_code {
        INVALID_ARGUMENT => VX_STATUS_INVALID_ARGUMENT,
        NOT_FOUND => VX_STATUS_NOT_FOUND,
        FAILED_PRECONDITION => VX_STATUS_HANDLE_DISPOSED,
        UNAVAILABLE => VX_STATUS_BACKEND_UNAVAILABLE,
        _ => VX_STATUS_INTERNAL,
    };
    FfiError::new(message, native_status, grpc_code)
}

fn required_text(value: &str, field: &str) -> Result<CString, FfiError> {
    if value.is_empty() {
        return Err(service_error(
            format!("{field} must not be empty"),
            INVALID_ARGUMENT,
        ));
    }
    CString::new(value).map_err(|_| {
        service_error(
            format!("{field} must not contain a NUL byte"),
            INVALID_ARGUMENT,
        )
    })
}

fn optional_text(value: &str, field: &str) -> Result<Option<CString>, FfiError> {
    if value.is_empty() {
        return Ok(None);
    }
    required_text(value, field).map(Some)
}

fn is_canonical_graph_path(graph_path: &str) -> bool {
    let basename = Path::new(graph_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    basename == "graph.json"
        || (basename.len() > ".graph.json".len() && basename.ends_with(".graph.json"))
}

fn validate_graph_document(graph_path: &str) -> Result<(), FfiError> {
    if !is_canonical_graph_path(graph_path) {
        return Err(service_error(
            "graph_path must name graph.json or a named *.graph.json document",
            INVALID_ARGUMENT,
        ));
    }
    let bytes = fs::read(graph_path).map_err(|error| {
        service_error(
            format!("graph_path is not readable: {error}"),
            INVALID_ARGUMENT,
        )
    })?;
    let document: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
        service_error(
            format!("graph_path is not valid JSON: {error}"),
            INVALID_ARGUMENT,
        )
    })?;
    if document.get("format").and_then(serde_json::Value::as_str) != Some("volvox-graph/v1") {
        return Err(service_error(
            "graph root format must be exactly volvox-graph/v1",
            INVALID_ARGUMENT,
        ));
    }
    Ok(())
}

fn native_text<const N: usize>(value: &[c_char; N]) -> String {
    let length = value.iter().position(|byte| *byte == 0).unwrap_or(N);
    let bytes = unsafe { std::slice::from_raw_parts(value.as_ptr().cast::<u8>(), length) };
    String::from_utf8_lossy(bytes).into_owned()
}

fn native_status_name(status: c_int) -> String {
    let pointer = unsafe { vx_status_string(status) };
    if pointer.is_null() {
        return format!("NATIVE_STATUS_{status}");
    }
    unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned()
}

fn report_native_status(status: c_int) -> i32 {
    NativeStatus::try_from(status).unwrap_or(NativeStatus::Internal) as i32
}

fn report_stage(stage: c_int) -> i32 {
    OperationStage::try_from(stage).unwrap_or(OperationStage::None) as i32
}

fn report_policy_mode(mode: c_int) -> i32 {
    match mode {
        VX_BACKEND_REQUIRE => BackendPolicyMode::Require as i32,
        _ => BackendPolicyMode::Prefer as i32,
    }
}

fn report_operator_fallback(value: c_int) -> i32 {
    match value {
        VX_OPERATOR_FALLBACK_FORBID => OperatorFallback::Forbid as i32,
        _ => OperatorFallback::Allow as i32,
    }
}

fn report_message(report: &VxReport) -> OperationReport {
    let reason = native_text(&report.reason);
    OperationReport {
        native_status: report_native_status(report.status),
        code: if reason.is_empty() {
            native_status_name(report.status)
        } else {
            reason
        },
        stage: report_stage(report.stage),
        message: native_text(&report.message),
        backend: native_text(&report.backend),
        device: native_text(&report.device),
        execution_id: report.execution_id,
        runtime_id: report.runtime_id,
        model_id: report.model_id,
        compiled_model_id: report.compiled_model_id,
        context_id: report.context_id,
        graph_id: report.graph_id,
        graph_revision: report.graph_revision,
        weight_id: report.weight_id,
        weight_revision: report.weight_revision,
        adapter_id: report.adapter_id,
        adapter_revision: report.adapter_revision,
        allocated_bytes: report.allocated_bytes,
        result_bytes: report.result_bytes,
        compile_time_ms: report.compile_time_ms,
        execution_time_ms: report.execution_time_ms,
        policy_mode: report_policy_mode(report.policy_mode),
        operator_fallback: report_operator_fallback(report.operator_fallback),
        candidate_count: report.candidate_count,
        tier_fallback_used: report.tier_fallback_used != 0,
        operator_fallback_used: report.operator_fallback_used != 0,
        route_attested: report.route_attested != 0,
        candidate_outcomes: native_text(&report.candidate_outcomes),
        route_evidence: native_text(&report.route_evidence),
        fallback_evidence: native_text(&report.fallback_evidence),
        offending_node: native_text(&report.offending_node),
        decode_state: native_text(&report.decode_state),
    }
}

fn grpc_code_for_native(status: c_int) -> i32 {
    match status {
        VX_STATUS_INVALID_ARGUMENT | VX_STATUS_INVALID_GRAPH => INVALID_ARGUMENT,
        VX_STATUS_NOT_FOUND => NOT_FOUND,
        VX_STATUS_HANDLE_DISPOSED | VX_STATUS_RESULT_DISPOSED => FAILED_PRECONDITION,
        VX_STATUS_REVISION_CONFLICT => ABORTED,
        VX_STATUS_BACKEND_UNAVAILABLE
        | VX_STATUS_BACKEND_UNSUPPORTED
        | VX_STATUS_BACKEND_REQUIRED
        | VX_STATUS_DEVICE_LOST => UNAVAILABLE,
        _ => INTERNAL,
    }
}

fn check_native(status: c_int, report: &VxReport) -> Result<OperationReport, FfiError> {
    if status == VX_STATUS_OK {
        return Ok(report_message(report));
    }
    let converted = report_message(report);
    let detail = if converted.message.is_empty() {
        native_status_name(status)
    } else if converted.code.is_empty() {
        converted.message
    } else {
        format!("{}: {}", converted.code, converted.message)
    };
    Err(FfiError::new(detail, status, grpc_code_for_native(status)))
}

fn native_dtype(dtype: i32) -> Result<c_int, FfiError> {
    match DataType::try_from(dtype).ok() {
        Some(DataType::F32 | DataType::I8 | DataType::U8 | DataType::I32) => Ok(dtype),
        _ => Err(service_error("unsupported tensor dtype", INVALID_ARGUMENT)),
    }
}

fn protobuf_dtype(dtype: c_int) -> Result<i32, FfiError> {
    match DataType::try_from(dtype).ok() {
        Some(DataType::F32 | DataType::I8 | DataType::U8 | DataType::I32) => Ok(dtype),
        _ => Err(service_error(
            "native returned an unsupported execution tensor dtype",
            INTERNAL,
        )),
    }
}

fn dtype_byte_width(dtype: c_int) -> usize {
    match dtype {
        VX_DTYPE_F32 | VX_DTYPE_I32 => 4,
        VX_DTYPE_I8 | VX_DTYPE_U8 => 1,
        _ => 0,
    }
}

fn protobuf_location(location: c_int) -> Result<i32, FfiError> {
    MemoryLocation::try_from(location)
        .map(|value| value as i32)
        .map_err(|_| service_error("native returned an invalid memory location", INTERNAL))
}

fn tensor_info_message(info: &VxTensorInfo) -> Result<TensorInfo, FfiError> {
    if info.rank as usize > VX_MAX_TENSOR_RANK {
        return Err(service_error(
            "native tensor rank exceeds the public limit",
            INTERNAL,
        ));
    }
    let name = if info.name.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(info.name) }
            .to_string_lossy()
            .into_owned()
    };
    Ok(TensorInfo {
        name,
        shape: info.shape[..info.rank as usize].to_vec(),
        dtype: protobuf_dtype(info.dtype)?,
        byte_size: u64::try_from(info.byte_size)
            .map_err(|_| service_error("tensor byte size is not representable", INTERNAL))?,
        location: protobuf_location(info.location)?,
    })
}

fn input_descriptors(
    count: usize,
    mut query: impl FnMut(usize, &mut VxTensorInfo, &mut VxReport) -> c_int,
) -> Result<Vec<TensorInfo>, FfiError> {
    let mut inputs = Vec::with_capacity(count);
    // Count functions have no report channel and return zero for both a live
    // zero-input owner and a closed owner. Probe index zero so native lifecycle
    // errors win over request-shape validation.
    for index in 0..count.max(1) {
        let mut info = VxTensorInfo::new();
        let mut report = VxReport::new();
        let status = query(index, &mut info, &mut report);
        if count == 0 && status == VX_STATUS_NOT_FOUND {
            return Ok(inputs);
        }
        check_native(status, &report)?;
        inputs.push(tensor_info_message(&info)?);
    }
    Ok(inputs)
}

fn revision_message(revision: &VxRevisionInfo, report: OperationReport) -> RevisionInfo {
    RevisionInfo {
        graph_id: revision.graph_id,
        graph_revision: revision.graph_revision,
        weight_id: revision.weight_id,
        weight_revision: revision.weight_revision,
        adapter_id: revision.adapter_id,
        adapter_revision: revision.adapter_revision,
        report: Some(report),
    }
}

fn checked_tensor_bytes(shape: &[i64], native_dtype: c_int) -> Result<usize, FfiError> {
    if shape.len() > VX_MAX_TENSOR_RANK {
        return Err(service_error("tensor rank exceeds 8", INVALID_ARGUMENT));
    }
    let mut elements = 1usize;
    for dimension in shape {
        let dimension = usize::try_from(*dimension).map_err(|_| {
            service_error("tensor dimensions must be non-negative", INVALID_ARGUMENT)
        })?;
        elements = elements
            .checked_mul(dimension)
            .ok_or_else(|| service_error("tensor is too large", INVALID_ARGUMENT))?;
    }
    elements
        .checked_mul(dtype_byte_width(native_dtype))
        .ok_or_else(|| service_error("tensor is too large", INVALID_ARGUMENT))
}

fn validate_host_input(
    input: &Tensor,
    descriptors: &[TensorInfo],
) -> Result<(CString, c_int), FfiError> {
    if MemoryLocation::try_from(input.location).ok() != Some(MemoryLocation::Host) {
        return Err(service_error(
            "input.location must be MEMORY_LOCATION_HOST",
            INVALID_ARGUMENT,
        ));
    }
    let name = required_text(&input.name, "input.name")?;
    let descriptor = descriptors
        .iter()
        .find(|descriptor| descriptor.name == input.name)
        .ok_or_else(|| {
            service_error(
                format!("unknown input tensor {}", input.name),
                INVALID_ARGUMENT,
            )
        })?;
    if input.shape != descriptor.shape {
        return Err(service_error(
            format!(
                "input {} shape {:?} does not match declared shape {:?}",
                input.name, input.shape, descriptor.shape
            ),
            INVALID_ARGUMENT,
        ));
    }
    if input.dtype != descriptor.dtype {
        return Err(service_error(
            format!(
                "input {} dtype {} does not match declared dtype {}",
                input.name, input.dtype, descriptor.dtype
            ),
            INVALID_ARGUMENT,
        ));
    }
    let dtype = native_dtype(input.dtype)?;
    let expected_bytes = checked_tensor_bytes(&descriptor.shape, dtype)?;
    let descriptor_bytes = usize::try_from(descriptor.byte_size)
        .map_err(|_| service_error("declared input byte size is not representable", INTERNAL))?;
    if expected_bytes != descriptor_bytes {
        return Err(service_error(
            "native input descriptor has an inconsistent byte size",
            INTERNAL,
        ));
    }
    if input.data.len() != expected_bytes {
        return Err(service_error(
            format!(
                "input {} has {} bytes, expected {}",
                input.name,
                input.data.len(),
                expected_bytes
            ),
            INVALID_ARGUMENT,
        ));
    }
    Ok((name, dtype))
}

fn native_optimizer_options(
    requested: Option<TrainerOptimizerOptions>,
) -> Result<VxOptimizerOptions, FfiError> {
    let requested = requested.unwrap_or_default();
    let mut optimizer = VxOptimizerOptions::new();
    optimizer.kind = match TrainingOptimizerKind::try_from(requested.kind).ok() {
        Some(TrainingOptimizerKind::Adamw) => VX_OPTIMIZER_ADAMW,
        Some(TrainingOptimizerKind::Sgd) => VX_OPTIMIZER_SGD,
        None => {
            return Err(service_error(
                "invalid training optimizer kind",
                INVALID_ARGUMENT,
            ))
        }
    };
    if let Some(value) = requested.learning_rate {
        optimizer.learning_rate = value;
    }
    if let Some(value) = requested.beta1 {
        optimizer.beta1 = value;
    }
    if let Some(value) = requested.beta2 {
        optimizer.beta2 = value;
    }
    if let Some(value) = requested.epsilon {
        optimizer.epsilon = value;
    }
    if let Some(value) = requested.weight_decay {
        optimizer.weight_decay = value;
    }
    if let Some(value) = requested.max_gradient_norm {
        optimizer.max_gradient_norm = value;
    }
    Ok(optimizer)
}

macro_rules! retained_handle {
    ($guard:ident, $native:ident, $release:ident) => {
        struct $guard(*mut $native);

        impl $guard {
            fn pointer(&self) -> *mut $native {
                self.0
            }
        }

        impl Drop for $guard {
            fn drop(&mut self) {
                unsafe { $release(self.0) };
            }
        }
    };
}

retained_handle!(RuntimeGuard, VxRuntime, vx_runtime_release);
retained_handle!(ModelGuard, VxModel, vx_model_release);
retained_handle!(
    CompiledModelGuard,
    VxCompiledModel,
    vx_compiled_model_release
);
retained_handle!(
    ContextGuard,
    VxExecutionContext,
    vx_execution_context_release
);
retained_handle!(ResultGuard, VxResult, vx_result_release);
retained_handle!(TrainerGuard, VxTrainer, vx_trainer_release);
retained_handle!(PtqPlanGuard, VxPTQPlan, vx_ptq_plan_release);

#[derive(Default)]
struct HandleRegistry {
    entries: Mutex<HashMap<String, usize>>,
}

impl HandleRegistry {
    fn insert(&self, id: String, pointer: *mut c_void) {
        debug_assert!(!pointer.is_null());
        lock(&self.entries).insert(id, pointer as usize);
    }

    fn remove(&self, id: &str, kind: &str) -> Result<usize, FfiError> {
        lock(&self.entries)
            .remove(id)
            .ok_or_else(|| service_error(format!("unknown {kind} handle {id}"), NOT_FOUND))
    }
}

struct Plugin {
    next_handle: AtomicU64,
    runtimes: HandleRegistry,
    models: HandleRegistry,
    compiled_models: HandleRegistry,
    contexts: HandleRegistry,
    results: HandleRegistry,
    trainers: HandleRegistry,
    ptq_plans: HandleRegistry,
}

impl Default for Plugin {
    fn default() -> Self {
        Self {
            next_handle: AtomicU64::new(1),
            runtimes: HandleRegistry::default(),
            models: HandleRegistry::default(),
            compiled_models: HandleRegistry::default(),
            contexts: HandleRegistry::default(),
            results: HandleRegistry::default(),
            trainers: HandleRegistry::default(),
            ptq_plans: HandleRegistry::default(),
        }
    }
}

impl Plugin {
    fn new_id(&self, kind: &str) -> String {
        let sequence = self.next_handle.fetch_add(1, Ordering::Relaxed);
        format!("{kind}-{sequence}")
    }

    fn runtime(&self, id: &str) -> Result<RuntimeGuard, FfiError> {
        let entries = lock(&self.runtimes.entries);
        let pointer = entries
            .get(id)
            .copied()
            .ok_or_else(|| service_error(format!("unknown runtime handle {id}"), NOT_FOUND))?
            as *mut VxRuntime;
        unsafe { vx_runtime_retain(pointer) };
        Ok(RuntimeGuard(pointer))
    }

    fn model(&self, id: &str) -> Result<ModelGuard, FfiError> {
        let entries = lock(&self.models.entries);
        let pointer = entries
            .get(id)
            .copied()
            .ok_or_else(|| service_error(format!("unknown model handle {id}"), NOT_FOUND))?
            as *mut VxModel;
        unsafe { vx_model_retain(pointer) };
        Ok(ModelGuard(pointer))
    }

    fn compiled_model(&self, id: &str) -> Result<CompiledModelGuard, FfiError> {
        let entries = lock(&self.compiled_models.entries);
        let pointer = entries.get(id).copied().ok_or_else(|| {
            service_error(format!("unknown compiled model handle {id}"), NOT_FOUND)
        })? as *mut VxCompiledModel;
        unsafe { vx_compiled_model_retain(pointer) };
        Ok(CompiledModelGuard(pointer))
    }

    fn context(&self, id: &str) -> Result<ContextGuard, FfiError> {
        let entries = lock(&self.contexts.entries);
        let pointer = entries
            .get(id)
            .copied()
            .ok_or_else(|| service_error(format!("unknown context handle {id}"), NOT_FOUND))?
            as *mut VxExecutionContext;
        unsafe { vx_execution_context_retain(pointer) };
        Ok(ContextGuard(pointer))
    }

    fn result(&self, id: &str) -> Result<ResultGuard, FfiError> {
        let entries = lock(&self.results.entries);
        let pointer = entries
            .get(id)
            .copied()
            .ok_or_else(|| service_error(format!("unknown result handle {id}"), NOT_FOUND))?
            as *mut VxResult;
        unsafe { vx_result_retain(pointer) };
        Ok(ResultGuard(pointer))
    }

    fn trainer(&self, id: &str) -> Result<TrainerGuard, FfiError> {
        let entries = lock(&self.trainers.entries);
        let pointer = entries
            .get(id)
            .copied()
            .ok_or_else(|| service_error(format!("unknown trainer handle {id}"), NOT_FOUND))?
            as *mut VxTrainer;
        unsafe { vx_trainer_retain(pointer) };
        Ok(TrainerGuard(pointer))
    }

    fn ptq_plan(&self, id: &str) -> Result<PtqPlanGuard, FfiError> {
        let entries = lock(&self.ptq_plans.entries);
        let pointer = entries
            .get(id)
            .copied()
            .ok_or_else(|| service_error(format!("unknown PTQ plan handle {id}"), NOT_FOUND))?
            as *mut VxPTQPlan;
        unsafe { vx_ptq_plan_retain(pointer) };
        Ok(PtqPlanGuard(pointer))
    }

    fn context_inputs(
        &self,
        context: *mut VxExecutionContext,
    ) -> Result<Vec<TensorInfo>, FfiError> {
        let count = unsafe { vx_execution_context_input_count(context) };
        input_descriptors(count, |index, info, report| unsafe {
            vx_execution_context_input_info(context, index, info, report)
        })
    }

    fn result_outputs(&self, result: *mut VxResult) -> Result<Vec<TensorInfo>, FfiError> {
        let count = unsafe { vx_result_output_count(result) };
        let mut outputs = Vec::with_capacity(count);
        for index in 0..count {
            let mut info = VxTensorInfo::new();
            let mut report = VxReport::new();
            let status = unsafe { vx_result_output_info(result, index, &mut info, &mut report) };
            check_native(status, &report)?;
            outputs.push(tensor_info_message(&info)?);
        }
        Ok(outputs)
    }

    fn trainer_inputs(&self, trainer: *mut VxTrainer) -> Result<Vec<TensorInfo>, FfiError> {
        let count = unsafe { vx_trainer_input_count(trainer) };
        input_descriptors(count, |index, info, report| unsafe {
            vx_trainer_input_info(trainer, index, info, report)
        })
    }

    fn ptq_inputs(&self, plan: *mut VxPTQPlan) -> Result<Vec<TensorInfo>, FfiError> {
        let count = unsafe { vx_ptq_plan_input_count(plan) };
        input_descriptors(count, |index, info, report| unsafe {
            vx_ptq_plan_input_info(plan, index, info, report)
        })
    }

    fn ptq_info(&self, plan: *mut VxPTQPlan) -> Result<PtqPlanInfo, FfiError> {
        let mut native = VxPTQPlanInfo::new();
        let mut report = VxReport::new();
        let status = unsafe { vx_ptq_plan_info(plan, &mut native, &mut report) };
        let report = check_native(status, &report)?;
        let mut tensors = Vec::with_capacity(native.tensor_count);
        for index in 0..native.tensor_count {
            let mut parameters = VxPTQTensorParameters::new();
            let mut tensor_report = VxReport::new();
            let status = unsafe {
                vx_ptq_plan_tensor_parameters(plan, index, &mut parameters, &mut tensor_report)
            };
            check_native(status, &tensor_report)?;
            let tensor_name = native_text(&parameters.tensor_name);
            if tensor_name.is_empty() {
                return Err(service_error(
                    "native PTQ tensor parameter has no name",
                    INTERNAL,
                ));
            }
            let scheme = PtqScheme::try_from(parameters.scheme)
                .map_err(|_| service_error("native returned an invalid PTQ scheme", INTERNAL))?
                as i32;
            tensors.push(PtqTensorParameters {
                tensor_name,
                dtype: protobuf_dtype(parameters.dtype)?,
                scheme,
                scale: parameters.scale,
                zero_point: parameters.zero_point,
                observed_min: parameters.observed_min,
                observed_max: parameters.observed_max,
                observed_values: parameters.observed_values,
            });
        }
        Ok(PtqPlanInfo {
            calibration_samples: native.calibration_samples,
            tensors,
            revision: Some(revision_message(&native.revision, report.clone())),
            report: Some(report),
        })
    }

    fn store_result(
        &self,
        result: *mut VxResult,
        report: OperationReport,
    ) -> Result<ExecutionResultHandle, FfiError> {
        if result.is_null() {
            return Err(service_error(
                "native execution returned no result handle",
                INTERNAL,
            ));
        }
        let execution_id = unsafe { vx_result_execution_id(result) };
        let result_id = self.new_id("result");
        self.results
            .insert(result_id.clone(), result.cast::<c_void>());
        Ok(ExecutionResultHandle {
            result_id,
            execution_id,
            report: Some(report),
        })
    }
}

impl RuntimeServicePlugin for Plugin {
    fn create_runtime(&self, request: CreateRuntimeRequest) -> Result<RuntimeHandle, FfiError> {
        if request.cpu_threads < 0 {
            return Err(service_error(
                "cpu_threads must be non-negative",
                INVALID_ARGUMENT,
            ));
        }
        let options = VxRuntimeOptions {
            struct_size: std::mem::size_of::<VxRuntimeOptions>(),
            debug: i32::from(request.debug),
            cpu_threads: request.cpu_threads,
        };
        let mut runtime = std::ptr::null_mut();
        let mut report = VxReport::new();
        let status = unsafe { vx_runtime_create(&options, &mut runtime, &mut report) };
        let report = check_native(status, &report)?;
        if runtime.is_null() {
            return Err(service_error(
                "native runtime creation returned no handle",
                INTERNAL,
            ));
        }
        let runtime_id = self.new_id("runtime");
        self.runtimes
            .insert(runtime_id.clone(), runtime.cast::<c_void>());
        Ok(RuntimeHandle {
            runtime_id,
            report: Some(report),
        })
    }

    fn close_runtime(&self, request: RuntimeRef) -> Result<OperationReport, FfiError> {
        let runtime = self.runtime(&request.runtime_id)?;
        let mut report = VxReport::new();
        let status = unsafe { vx_runtime_close(runtime.pointer(), &mut report) };
        check_native(status, &report)
    }

    fn release_runtime(&self, request: RuntimeRef) -> Result<Empty, FfiError> {
        let pointer = self.runtimes.remove(&request.runtime_id, "runtime")? as *mut VxRuntime;
        unsafe { vx_runtime_release(pointer) };
        Ok(Empty {})
    }

    fn load_model(&self, request: LoadModelRequest) -> Result<ModelHandle, FfiError> {
        let runtime = self.runtime(&request.runtime_id)?;
        let graph_path = required_text(&request.graph_path, "graph_path")?;
        validate_graph_document(&request.graph_path)?;
        let weight_paths: Vec<CString> = request
            .weight_paths
            .iter()
            .enumerate()
            .map(|(index, path)| required_text(path, &format!("weight_paths[{index}]")))
            .collect::<Result<_, _>>()?;
        let weight_pointers: Vec<*const c_char> =
            weight_paths.iter().map(|path| path.as_ptr()).collect();
        let source = VxModelSource {
            struct_size: std::mem::size_of::<VxModelSource>(),
            graph_path: graph_path.as_ptr(),
            weight_paths: if weight_pointers.is_empty() {
                std::ptr::null()
            } else {
                weight_pointers.as_ptr()
            },
            weight_path_count: weight_pointers.len(),
        };
        let mut model = std::ptr::null_mut();
        let mut report = VxReport::new();
        let status =
            unsafe { vx_runtime_load_model(runtime.pointer(), &source, &mut model, &mut report) };
        let report = check_native(status, &report)?;
        if model.is_null() {
            return Err(service_error(
                "native model loading returned no handle",
                INTERNAL,
            ));
        }
        let model_id = self.new_id("model");
        self.models.insert(model_id.clone(), model.cast::<c_void>());
        Ok(ModelHandle {
            model_id,
            report: Some(report),
        })
    }

    fn get_model_revision(&self, request: ModelRef) -> Result<RevisionInfo, FfiError> {
        let model = self.model(&request.model_id)?;
        let mut revision = VxRevisionInfo::new();
        let mut report = VxReport::new();
        let status = unsafe { vx_model_revision_info(model.pointer(), &mut revision, &mut report) };
        let report = check_native(status, &report)?;
        Ok(revision_message(&revision, report))
    }

    fn publish_adapter(&self, request: PublishAdapterRequest) -> Result<AdapterRevision, FfiError> {
        let model = self.model(&request.model_id)?;
        let adapter_name = required_text(&request.adapter_name, "adapter_name")?;
        let package_path = optional_text(&request.package_path, "package_path")?;
        let version_name = optional_text(&request.version_name, "version_name")?;
        let source = VxAdapterSource {
            struct_size: std::mem::size_of::<VxAdapterSource>(),
            adapter_name: adapter_name.as_ptr(),
            package_path: package_path
                .as_ref()
                .map_or(std::ptr::null(), |value| value.as_ptr()),
            version_name: version_name
                .as_ref()
                .map_or(std::ptr::null(), |value| value.as_ptr()),
        };
        let mut published = VxAdapterRevision::new();
        let mut report = VxReport::new();
        let status = unsafe {
            vx_model_publish_adapter(model.pointer(), &source, &mut published, &mut report)
        };
        let report = check_native(status, &report)?;
        Ok(AdapterRevision {
            adapter_id: published.adapter_id,
            adapter_revision: published.adapter_revision,
            report: Some(report),
        })
    }

    fn release_model(&self, request: ModelRef) -> Result<Empty, FfiError> {
        let pointer = self.models.remove(&request.model_id, "model")? as *mut VxModel;
        unsafe { vx_model_release(pointer) };
        Ok(Empty {})
    }

    fn create_trainer(&self, request: CreateTrainerRequest) -> Result<TrainerHandle, FfiError> {
        let model = self.model(&request.model_id)?;
        let backend = optional_text(&request.backend, "backend")?;
        let options = VxTrainerOptions {
            struct_size: std::mem::size_of::<VxTrainerOptions>(),
            backend: backend
                .as_ref()
                .map_or(std::ptr::null(), |value| value.as_ptr()),
            rng_seed: request.rng_seed,
        };
        let mut trainer = std::ptr::null_mut();
        let mut report = VxReport::new();
        let status = unsafe {
            vx_model_create_trainer(model.pointer(), &options, &mut trainer, &mut report)
        };
        let report = check_native(status, &report)?;
        if trainer.is_null() {
            return Err(service_error(
                "native Trainer creation returned no handle",
                INTERNAL,
            ));
        }
        let inputs = match self.trainer_inputs(trainer) {
            Ok(inputs) => inputs,
            Err(error) => {
                unsafe {
                    let mut close_report = VxReport::new();
                    vx_trainer_close(trainer, &mut close_report);
                    vx_trainer_release(trainer);
                }
                return Err(error);
            }
        };
        let trainer_id = self.new_id("trainer");
        self.trainers
            .insert(trainer_id.clone(), trainer.cast::<c_void>());
        Ok(TrainerHandle {
            trainer_id,
            inputs,
            report: Some(report),
        })
    }

    fn close_trainer(&self, request: TrainerRef) -> Result<OperationReport, FfiError> {
        let trainer = self.trainer(&request.trainer_id)?;
        let mut report = VxReport::new();
        let status = unsafe { vx_trainer_close(trainer.pointer(), &mut report) };
        check_native(status, &report)
    }

    fn release_trainer(&self, request: TrainerRef) -> Result<Empty, FfiError> {
        let pointer = self.trainers.remove(&request.trainer_id, "trainer")? as *mut VxTrainer;
        unsafe { vx_trainer_release(pointer) };
        Ok(Empty {})
    }

    fn set_trainer_input(
        &self,
        request: SetTrainerInputRequest,
    ) -> Result<OperationReport, FfiError> {
        let trainer = self.trainer(&request.trainer_id)?;
        let input = request
            .input
            .ok_or_else(|| service_error("input tensor is required", INVALID_ARGUMENT))?;
        let descriptors = self.trainer_inputs(trainer.pointer())?;
        let (name, dtype) = validate_host_input(&input, &descriptors)?;
        let mut report = VxReport::new();
        let status = unsafe {
            vx_trainer_set_input(
                trainer.pointer(),
                name.as_ptr(),
                dtype,
                input.data.as_ptr().cast::<c_void>(),
                input.data.len(),
                &mut report,
            )
        };
        check_native(status, &report)
    }

    fn train_step(&self, request: TrainStepRequest) -> Result<TrainStepResult, FfiError> {
        let trainer = self.trainer(&request.trainer_id)?;
        let loss_names = request
            .losses
            .iter()
            .enumerate()
            .map(|(index, loss)| required_text(&loss.name, &format!("losses[{index}].name")))
            .collect::<Result<Vec<_>, _>>()?;
        let logits_names = request
            .losses
            .iter()
            .enumerate()
            .map(|(index, loss)| {
                required_text(&loss.logits_name, &format!("losses[{index}].logits_name"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let losses = request
            .losses
            .iter()
            .enumerate()
            .map(|(index, loss)| VxCrossEntropyLoss {
                struct_size: std::mem::size_of::<VxCrossEntropyLoss>(),
                name: loss_names[index].as_ptr(),
                logits_name: logits_names[index].as_ptr(),
                targets: loss.targets.as_ptr(),
                target_count: loss.targets.len(),
                ignore_index: loss.ignore_index.unwrap_or(-1),
                row_index: loss.row_index.unwrap_or(-1),
                weight: loss.weight.unwrap_or(1.0),
                normalizer: loss.normalizer,
            })
            .collect::<Vec<_>>();
        let trainable_names = request
            .trainable_names
            .iter()
            .enumerate()
            .map(|(index, name)| required_text(name, &format!("trainable_names[{index}]")))
            .collect::<Result<Vec<_>, _>>()?;
        let trainable_pointers = trainable_names
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();
        let optimizer = native_optimizer_options(request.optimizer)?;
        let options = VxTrainStepOptions {
            struct_size: std::mem::size_of::<VxTrainStepOptions>(),
            losses: losses.as_ptr(),
            loss_count: losses.len(),
            trainable_names: trainable_pointers.as_ptr(),
            trainable_count: trainable_pointers.len(),
            optimizer,
            accumulation_steps: request.accumulation_steps.unwrap_or(1),
            flush_accumulation: i32::from(request.flush_accumulation),
            reset_accumulation: i32::from(request.reset_accumulation),
        };
        let mut native_result = VxTrainStepResult::new();
        let mut report = VxReport::new();
        let status = unsafe {
            vx_trainer_train_step(trainer.pointer(), &options, &mut native_result, &mut report)
        };
        let report = check_native(status, &report)?;
        if native_result.metric_count > VX_MAX_TRAINING_LOSSES {
            return Err(service_error(
                "native Trainer returned too many loss metrics",
                INTERNAL,
            ));
        }
        let metrics = native_result.metrics[..native_result.metric_count]
            .iter()
            .map(|metric| TrainingMetric {
                name: native_text(&metric.name),
                loss: metric.loss,
                correct: metric.correct,
                examples: metric.examples,
                normalizer: metric.normalizer,
            })
            .collect();
        Ok(TrainStepResult {
            microbatch_id: native_result.microbatch_id,
            optimizer_step: native_result.optimizer_step,
            accumulated_microbatches: native_result.accumulated_microbatches,
            update_applied: native_result.update_applied != 0,
            loss: native_result.loss,
            metrics,
            backend: native_text(&native_result.backend),
            report: Some(report),
        })
    }

    fn commit_trainer(&self, request: TrainerRef) -> Result<RevisionInfo, FfiError> {
        let trainer = self.trainer(&request.trainer_id)?;
        let mut revision = VxRevisionInfo::new();
        let mut report = VxReport::new();
        let status = unsafe { vx_trainer_commit(trainer.pointer(), &mut revision, &mut report) };
        let report = check_native(status, &report)?;
        Ok(revision_message(&revision, report))
    }

    fn rollback_trainer(&self, request: TrainerRef) -> Result<OperationReport, FfiError> {
        let trainer = self.trainer(&request.trainer_id)?;
        let mut report = VxReport::new();
        let status = unsafe { vx_trainer_rollback(trainer.pointer(), &mut report) };
        check_native(status, &report)
    }

    fn export_trainer_weights(
        &self,
        request: ExportTrainerWeightsRequest,
    ) -> Result<OperationReport, FfiError> {
        let trainer = self.trainer(&request.trainer_id)?;
        let paths = request
            .output_paths
            .iter()
            .enumerate()
            .map(|(index, path)| required_text(path, &format!("output_paths[{index}]")))
            .collect::<Result<Vec<_>, _>>()?;
        let path_pointers = paths.iter().map(|path| path.as_ptr()).collect::<Vec<_>>();
        let mut report = VxReport::new();
        let status = unsafe {
            vx_trainer_export_weights(
                trainer.pointer(),
                path_pointers.as_ptr(),
                path_pointers.len(),
                &mut report,
            )
        };
        check_native(status, &report)
    }

    fn create_ptq_plan(&self, request: CreatePtqPlanRequest) -> Result<PtqPlanHandle, FfiError> {
        struct LayerNames {
            input: CString,
            output: CString,
            source_weight: CString,
            packed_weight: CString,
            source_bias: Option<CString>,
            packed_bias: Option<CString>,
        }

        let model = self.model(&request.model_id)?;
        validate_graph_document(&request.template_graph_path)?;
        let template_graph_path =
            required_text(&request.template_graph_path, "template_graph_path")?;
        if request.observers.is_empty() {
            return Err(service_error(
                "at least one PTQ observer is required",
                INVALID_ARGUMENT,
            ));
        }
        if request.layers.is_empty() {
            return Err(service_error(
                "at least one PTQ layer is required",
                INVALID_ARGUMENT,
            ));
        }
        let observer_names = request
            .observers
            .iter()
            .enumerate()
            .map(|(index, observer)| {
                required_text(
                    &observer.tensor_name,
                    &format!("observers[{index}].tensor_name"),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let native_observers = request
            .observers
            .iter()
            .enumerate()
            .map(|(index, observer)| {
                let dtype = native_dtype(observer.dtype)?;
                if dtype != VX_DTYPE_I8 && dtype != VX_DTYPE_U8 {
                    return Err(service_error(
                        format!("observers[{index}].dtype must be I8 or U8"),
                        INVALID_ARGUMENT,
                    ));
                }
                let scheme = match PtqScheme::try_from(observer.scheme).ok() {
                    Some(PtqScheme::Symmetric) => VX_PTQ_SCHEME_SYMMETRIC,
                    Some(PtqScheme::Asymmetric) => VX_PTQ_SCHEME_ASYMMETRIC,
                    None => {
                        return Err(service_error(
                            format!("observers[{index}].scheme is invalid"),
                            INVALID_ARGUMENT,
                        ))
                    }
                };
                Ok(VxPTQObserverSpec {
                    struct_size: std::mem::size_of::<VxPTQObserverSpec>(),
                    tensor_name: observer_names[index].as_ptr(),
                    dtype,
                    scheme,
                })
            })
            .collect::<Result<Vec<_>, FfiError>>()?;

        let layer_names = request
            .layers
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                Ok(LayerNames {
                    input: required_text(
                        &layer.input_tensor_name,
                        &format!("layers[{index}].input_tensor_name"),
                    )?,
                    output: required_text(
                        &layer.output_tensor_name,
                        &format!("layers[{index}].output_tensor_name"),
                    )?,
                    source_weight: required_text(
                        &layer.source_weight_name,
                        &format!("layers[{index}].source_weight_name"),
                    )?,
                    packed_weight: required_text(
                        &layer.packed_weight_name,
                        &format!("layers[{index}].packed_weight_name"),
                    )?,
                    source_bias: optional_text(
                        &layer.source_bias_name,
                        &format!("layers[{index}].source_bias_name"),
                    )?,
                    packed_bias: optional_text(
                        &layer.packed_bias_name,
                        &format!("layers[{index}].packed_bias_name"),
                    )?,
                })
            })
            .collect::<Result<Vec<_>, FfiError>>()?;
        let native_layers = request
            .layers
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                let mode = match PtqMode::try_from(layer.mode).ok() {
                    Some(PtqMode::W8a8) => VX_PTQ_MODE_W8A8,
                    None => {
                        return Err(service_error(
                            format!("layers[{index}].mode is invalid"),
                            INVALID_ARGUMENT,
                        ))
                    }
                };
                let kind = match PtqLayerKind::try_from(layer.kind).ok() {
                    Some(PtqLayerKind::Qlinear) => VX_PTQ_LAYER_QLINEAR,
                    Some(PtqLayerKind::Qconv2d) => VX_PTQ_LAYER_QCONV2D,
                    None => {
                        return Err(service_error(
                            format!("layers[{index}].kind is invalid"),
                            INVALID_ARGUMENT,
                        ))
                    }
                };
                let names = &layer_names[index];
                Ok(VxPTQLayerSpec {
                    struct_size: std::mem::size_of::<VxPTQLayerSpec>(),
                    mode,
                    kind,
                    node_index: layer.node_index,
                    weight_axis: layer.weight_axis,
                    input_tensor_name: names.input.as_ptr(),
                    output_tensor_name: names.output.as_ptr(),
                    source_weight_name: names.source_weight.as_ptr(),
                    packed_weight_name: names.packed_weight.as_ptr(),
                    source_bias_name: names
                        .source_bias
                        .as_ref()
                        .map_or(std::ptr::null(), |value| value.as_ptr()),
                    packed_bias_name: names
                        .packed_bias
                        .as_ref()
                        .map_or(std::ptr::null(), |value| value.as_ptr()),
                })
            })
            .collect::<Result<Vec<_>, FfiError>>()?;
        let options = VxPTQPlanOptions {
            struct_size: std::mem::size_of::<VxPTQPlanOptions>(),
            template_graph_path: template_graph_path.as_ptr(),
            observers: native_observers.as_ptr(),
            observer_count: native_observers.len(),
            layers: native_layers.as_ptr(),
            layer_count: native_layers.len(),
        };
        let mut plan = std::ptr::null_mut();
        let mut report = VxReport::new();
        let status =
            unsafe { vx_model_create_ptq_plan(model.pointer(), &options, &mut plan, &mut report) };
        let report = check_native(status, &report)?;
        if plan.is_null() {
            return Err(service_error(
                "native PTQ creation returned no handle",
                INTERNAL,
            ));
        }
        let inputs = match self.ptq_inputs(plan) {
            Ok(inputs) => inputs,
            Err(error) => {
                unsafe {
                    let mut close_report = VxReport::new();
                    vx_ptq_plan_close(plan, &mut close_report);
                    vx_ptq_plan_release(plan);
                }
                return Err(error);
            }
        };
        let ptq_plan_id = self.new_id("ptq-plan");
        self.ptq_plans
            .insert(ptq_plan_id.clone(), plan.cast::<c_void>());
        Ok(PtqPlanHandle {
            ptq_plan_id,
            inputs,
            report: Some(report),
        })
    }

    fn close_ptq_plan(&self, request: PtqPlanRef) -> Result<OperationReport, FfiError> {
        let plan = self.ptq_plan(&request.ptq_plan_id)?;
        let mut report = VxReport::new();
        let status = unsafe { vx_ptq_plan_close(plan.pointer(), &mut report) };
        check_native(status, &report)
    }

    fn release_ptq_plan(&self, request: PtqPlanRef) -> Result<Empty, FfiError> {
        let pointer = self.ptq_plans.remove(&request.ptq_plan_id, "PTQ plan")? as *mut VxPTQPlan;
        unsafe { vx_ptq_plan_release(pointer) };
        Ok(Empty {})
    }

    fn calibrate_ptq_plan(
        &self,
        request: CalibratePtqPlanRequest,
    ) -> Result<PtqCalibrationInfo, FfiError> {
        let plan = self.ptq_plan(&request.ptq_plan_id)?;
        let sample_name = required_text(&request.sample_name, "sample_name")?;
        let descriptors = self.ptq_inputs(plan.pointer())?;
        if request.inputs.len() != descriptors.len() {
            return Err(service_error(
                format!(
                    "calibration sample has {} inputs, expected {}",
                    request.inputs.len(),
                    descriptors.len()
                ),
                INVALID_ARGUMENT,
            ));
        }
        let mut seen = HashSet::with_capacity(request.inputs.len());
        let input_names = request
            .inputs
            .iter()
            .map(|input| {
                if !seen.insert(input.name.as_str()) {
                    return Err(service_error(
                        format!("duplicate calibration input {}", input.name),
                        INVALID_ARGUMENT,
                    ));
                }
                validate_host_input(input, &descriptors).map(|(name, _)| name)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let native_inputs = request
            .inputs
            .iter()
            .enumerate()
            .map(|(index, input)| VxPTQInput {
                struct_size: std::mem::size_of::<VxPTQInput>(),
                name: input_names[index].as_ptr(),
                dtype: input.dtype,
                data: input.data.as_ptr().cast::<c_void>(),
                byte_size: input.data.len(),
            })
            .collect::<Vec<_>>();
        let mut calibration_samples = 0;
        let mut report = VxReport::new();
        let status = unsafe {
            vx_ptq_plan_calibrate(
                plan.pointer(),
                sample_name.as_ptr(),
                native_inputs.as_ptr(),
                native_inputs.len(),
                &mut calibration_samples,
                &mut report,
            )
        };
        let report = check_native(status, &report)?;
        Ok(PtqCalibrationInfo {
            calibration_samples,
            report: Some(report),
        })
    }

    fn inspect_ptq_plan(&self, request: PtqPlanRef) -> Result<PtqPlanInfo, FfiError> {
        let plan = self.ptq_plan(&request.ptq_plan_id)?;
        self.ptq_info(plan.pointer())
    }

    fn write_ptq_package(
        &self,
        request: WritePtqPackageRequest,
    ) -> Result<PtqPackageInfo, FfiError> {
        let plan = self.ptq_plan(&request.ptq_plan_id)?;
        let output_graph_path = required_text(&request.output_graph_path, "output_graph_path")?;
        let output_weights_path =
            required_text(&request.output_weights_path, "output_weights_path")?;
        let options = VxPTQPackageOptions {
            struct_size: std::mem::size_of::<VxPTQPackageOptions>(),
            output_graph_path: output_graph_path.as_ptr(),
            output_weights_path: output_weights_path.as_ptr(),
        };
        let mut report = VxReport::new();
        let status = unsafe { vx_ptq_plan_write_package(plan.pointer(), &options, &mut report) };
        let report = check_native(status, &report)?;
        let info = self.ptq_info(plan.pointer())?;
        Ok(PtqPackageInfo {
            graph_path: request.output_graph_path,
            safetensors_path: request.output_weights_path,
            plan: Some(info),
            report: Some(report),
        })
    }

    fn compile_model(&self, request: CompileModelRequest) -> Result<CompiledModelHandle, FfiError> {
        let model = self.model(&request.model_id)?;
        let policy = request.policy.unwrap_or_default();
        let mode = match BackendPolicyMode::try_from(policy.mode).ok() {
            Some(BackendPolicyMode::Require) => VX_BACKEND_REQUIRE,
            Some(BackendPolicyMode::Prefer) => VX_BACKEND_PREFER,
            None => {
                return Err(service_error(
                    "invalid backend policy mode",
                    INVALID_ARGUMENT,
                ))
            }
        };
        let operator_fallback = match OperatorFallback::try_from(policy.operator_fallback).ok() {
            Some(OperatorFallback::Forbid) => VX_OPERATOR_FALLBACK_FORBID,
            Some(OperatorFallback::Allow) => VX_OPERATOR_FALLBACK_ALLOW,
            None => {
                return Err(service_error(
                    "invalid operator fallback policy",
                    INVALID_ARGUMENT,
                ))
            }
        };
        if mode == VX_BACKEND_REQUIRE && policy.backends.len() != 1 {
            return Err(service_error(
                "require policy needs exactly one backend",
                INVALID_ARGUMENT,
            ));
        }
        let backend_names: Vec<CString> = policy
            .backends
            .iter()
            .enumerate()
            .map(|(index, backend)| required_text(backend, &format!("policy.backends[{index}]")))
            .collect::<Result<_, _>>()?;
        let backend_pointers: Vec<*const c_char> = backend_names
            .iter()
            .map(|backend| backend.as_ptr())
            .collect();
        let native_policy = VxBackendPolicy {
            struct_size: std::mem::size_of::<VxBackendPolicy>(),
            mode,
            operator_fallback,
            backends: if backend_pointers.is_empty() {
                std::ptr::null()
            } else {
                backend_pointers.as_ptr()
            },
            backend_count: backend_pointers.len(),
        };
        let mut compiled = std::ptr::null_mut();
        let mut report = VxReport::new();
        let status = unsafe {
            vx_model_compile(model.pointer(), &native_policy, &mut compiled, &mut report)
        };
        let report = check_native(status, &report)?;
        if compiled.is_null() {
            return Err(service_error(
                "native model compilation returned no handle",
                INTERNAL,
            ));
        }
        let compiled_model_id = self.new_id("compiled");
        self.compiled_models
            .insert(compiled_model_id.clone(), compiled.cast::<c_void>());
        Ok(CompiledModelHandle {
            compiled_model_id,
            report: Some(report),
        })
    }

    fn get_compiled_model_report(
        &self,
        request: CompiledModelRef,
    ) -> Result<OperationReport, FfiError> {
        let compiled = self.compiled_model(&request.compiled_model_id)?;
        let mut report = VxReport::new();
        let status = unsafe { vx_compiled_model_report(compiled.pointer(), &mut report) };
        check_native(status, &report)
    }

    fn release_compiled_model(&self, request: CompiledModelRef) -> Result<Empty, FfiError> {
        let pointer = self
            .compiled_models
            .remove(&request.compiled_model_id, "compiled model")?
            as *mut VxCompiledModel;
        unsafe { vx_compiled_model_release(pointer) };
        Ok(Empty {})
    }

    fn create_execution_context(
        &self,
        request: CreateExecutionContextRequest,
    ) -> Result<ExecutionContextHandle, FfiError> {
        let compiled = self.compiled_model(&request.compiled_model_id)?;
        let options = VxContextOptions {
            struct_size: std::mem::size_of::<VxContextOptions>(),
            decode_row_mode: match DecodeRowMode::try_from(request.decode_row_mode).ok() {
                Some(DecodeRowMode::Disabled) => VX_DECODE_ROW_DISABLED,
                Some(DecodeRowMode::Auto) => VX_DECODE_ROW_AUTO,
                Some(DecodeRowMode::Required) => VX_DECODE_ROW_REQUIRED,
                None => return Err(service_error("invalid decode row mode", INVALID_ARGUMENT)),
            },
            require_incremental: i32::from(request.require_incremental),
        };
        let mut context = std::ptr::null_mut();
        let mut report = VxReport::new();
        let status = unsafe {
            vx_compiled_model_create_context(
                compiled.pointer(),
                &options,
                &mut context,
                &mut report,
            )
        };
        let report = check_native(status, &report)?;
        if context.is_null() {
            return Err(service_error(
                "native context creation returned no handle",
                INTERNAL,
            ));
        }
        let inputs = match self.context_inputs(context) {
            Ok(inputs) => inputs,
            Err(error) => {
                unsafe {
                    let mut close_report = VxReport::new();
                    vx_execution_context_close(context, &mut close_report);
                    vx_execution_context_release(context);
                }
                return Err(error);
            }
        };
        let context_id = self.new_id("context");
        self.contexts
            .insert(context_id.clone(), context.cast::<c_void>());
        Ok(ExecutionContextHandle {
            context_id,
            inputs,
            report: Some(report),
        })
    }

    fn close_execution_context(
        &self,
        request: ExecutionContextRef,
    ) -> Result<OperationReport, FfiError> {
        let context = self.context(&request.context_id)?;
        let mut report = VxReport::new();
        let status = unsafe { vx_execution_context_close(context.pointer(), &mut report) };
        check_native(status, &report)
    }

    fn release_execution_context(&self, request: ExecutionContextRef) -> Result<Empty, FfiError> {
        let pointer =
            self.contexts.remove(&request.context_id, "context")? as *mut VxExecutionContext;
        unsafe { vx_execution_context_release(pointer) };
        Ok(Empty {})
    }

    fn set_input(&self, request: SetInputRequest) -> Result<OperationReport, FfiError> {
        let context = self.context(&request.context_id)?;
        let input = request
            .input
            .ok_or_else(|| service_error("input tensor is required", INVALID_ARGUMENT))?;
        let descriptors = self.context_inputs(context.pointer())?;
        let (name, dtype) = validate_host_input(&input, &descriptors)?;
        let mut report = VxReport::new();
        let status = unsafe {
            vx_execution_context_set_input(
                context.pointer(),
                name.as_ptr(),
                dtype,
                input.data.as_ptr().cast::<c_void>(),
                input.data.len(),
                &mut report,
            )
        };
        check_native(status, &report)
    }

    fn execute(&self, request: ExecuteRequest) -> Result<ExecutionResultHandle, FfiError> {
        let context = self.context(&request.context_id)?;
        let mut result = std::ptr::null_mut();
        let mut report = VxReport::new();
        let status =
            unsafe { vx_execution_context_execute(context.pointer(), &mut result, &mut report) };
        let report = check_native(status, &report)?;
        self.store_result(result, report)
    }

    fn decode_seed(&self, request: DecodeSeedRequest) -> Result<ExecutionResultHandle, FfiError> {
        let context = self.context(&request.context_id)?;
        let mut result = std::ptr::null_mut();
        let mut report = VxReport::new();
        let status = unsafe {
            vx_execution_context_decode_seed(context.pointer(), &mut result, &mut report)
        };
        let report = check_native(status, &report)?;
        self.store_result(result, report)
    }

    fn decode_step(&self, request: DecodeStepRequest) -> Result<ExecutionResultHandle, FfiError> {
        let position = request.position.unwrap_or(-1);
        if position < -1 {
            return Err(service_error(
                "decode position must be -1 or non-negative",
                INVALID_ARGUMENT,
            ));
        }
        let context = self.context(&request.context_id)?;
        let mut result = std::ptr::null_mut();
        let mut report = VxReport::new();
        let status = unsafe {
            vx_execution_context_decode_step(context.pointer(), position, &mut result, &mut report)
        };
        let report = check_native(status, &report)?;
        self.store_result(result, report)
    }

    fn reset_decode(&self, request: ExecutionContextRef) -> Result<OperationReport, FfiError> {
        let context = self.context(&request.context_id)?;
        let mut report = VxReport::new();
        let status = unsafe { vx_execution_context_decode_reset(context.pointer(), &mut report) };
        check_native(status, &report)
    }

    fn select_adapter(&self, request: SelectAdapterRequest) -> Result<OperationReport, FfiError> {
        let context = self.context(&request.context_id)?;
        let revision = VxAdapterRevision {
            struct_size: std::mem::size_of::<VxAdapterRevision>(),
            adapter_id: request.adapter_id,
            adapter_revision: request.adapter_revision,
        };
        let mut report = VxReport::new();
        let status = unsafe {
            vx_execution_context_select_adapter(context.pointer(), &revision, &mut report)
        };
        check_native(status, &report)
    }

    fn rebind_adapter(&self, request: ExecutionContextRef) -> Result<OperationReport, FfiError> {
        let context = self.context(&request.context_id)?;
        let mut report = VxReport::new();
        let status = unsafe { vx_execution_context_rebind_adapter(context.pointer(), &mut report) };
        check_native(status, &report)
    }

    fn get_result(&self, request: ResultRef) -> Result<ResultInfo, FfiError> {
        let result = self.result(&request.result_id)?;
        Ok(ResultInfo {
            result_id: request.result_id,
            execution_id: unsafe { vx_result_execution_id(result.pointer()) },
            outputs: self.result_outputs(result.pointer())?,
        })
    }

    fn read_output(&self, request: ReadOutputRequest) -> Result<Tensor, FfiError> {
        let result = self.result(&request.result_id)?;
        let name = required_text(&request.name, "name")?;
        let outputs = self.result_outputs(result.pointer())?;
        let metadata = outputs
            .into_iter()
            .find(|output| output.name == request.name)
            .ok_or_else(|| service_error(format!("unknown output {}", request.name), NOT_FOUND))?;

        let mut required = 0usize;
        let mut report = VxReport::new();
        let status = unsafe {
            vx_result_read(
                result.pointer(),
                name.as_ptr(),
                std::ptr::null_mut(),
                0,
                &mut required,
                &mut report,
            )
        };
        check_native(status, &report)?;
        if u64::try_from(required).ok() != Some(metadata.byte_size) {
            return Err(service_error(
                "output size changed while reading a stable result",
                INTERNAL,
            ));
        }
        let mut data = Vec::new();
        data.try_reserve_exact(required)
            .map_err(|_| service_error("output allocation failed", INTERNAL))?;
        data.resize(required, 0);
        let mut copied = 0usize;
        report = VxReport::new();
        let status = unsafe {
            vx_result_read(
                result.pointer(),
                name.as_ptr(),
                data.as_mut_ptr().cast::<c_void>(),
                data.len(),
                &mut copied,
                &mut report,
            )
        };
        check_native(status, &report)?;
        if copied != data.len() {
            return Err(service_error(
                "native output read returned an inconsistent byte count",
                INTERNAL,
            ));
        }
        Ok(Tensor {
            name: metadata.name,
            shape: metadata.shape,
            dtype: metadata.dtype,
            data,
            location: MemoryLocation::Host as i32,
        })
    }

    fn release_result(&self, request: ResultRef) -> Result<Empty, FfiError> {
        let pointer = self.results.remove(&request.result_id, "result")? as *mut VxResult;
        unsafe { vx_result_release(pointer) };
        Ok(Empty {})
    }
}

#[ctor::ctor]
fn register_plugin() {
    register_runtime_service_plugin(Plugin::default());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_size_validation_is_checked() {
        assert_eq!(checked_tensor_bytes(&[2, 3], VX_DTYPE_F32).unwrap(), 24);
        assert!(checked_tensor_bytes(&[-1], VX_DTYPE_F32).is_err());
        assert!(checked_tensor_bytes(&[i64::MAX, 3], VX_DTYPE_F32).is_err());
        assert!(native_dtype(DataType::Unspecified as i32).is_err());
        assert!(native_dtype(DataType::Bool as i32).is_err());
        assert!(DataType::try_from(23).is_err());
    }

    #[test]
    fn graph_paths_use_only_canonical_package_names() {
        assert!(is_canonical_graph_path("model/graph.json"));
        assert!(is_canonical_graph_path("model/router.graph.json"));
        assert!(!is_canonical_graph_path("model/config.json"));
        assert!(!is_canonical_graph_path("model/router-graph.json"));
    }

    #[test]
    fn protobuf_enums_are_the_native_abi_contract() {
        assert_eq!(
            [
                VX_STATUS_OK,
                VX_STATUS_INVALID_ARGUMENT,
                VX_STATUS_OUT_OF_MEMORY,
                VX_STATUS_HANDLE_DISPOSED,
                VX_STATUS_IO_ERROR,
                VX_STATUS_INVALID_GRAPH,
                VX_STATUS_BACKEND_UNAVAILABLE,
                VX_STATUS_BACKEND_UNSUPPORTED,
                VX_STATUS_BACKEND_REQUIRED,
                VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN,
                VX_STATUS_EXECUTION_FAILED,
                VX_STATUS_RESULT_DISPOSED,
                VX_STATUS_DEVICE_LOST,
                VX_STATUS_ABI_UNSUPPORTED,
                VX_STATUS_NOT_FOUND,
                VX_STATUS_BUFFER_TOO_SMALL,
                VX_STATUS_INTERNAL,
                VX_STATUS_REVISION_CONFLICT,
            ],
            [
                NativeStatus::Ok as i32,
                NativeStatus::InvalidArgument as i32,
                NativeStatus::OutOfMemory as i32,
                NativeStatus::HandleDisposed as i32,
                NativeStatus::IoError as i32,
                NativeStatus::InvalidGraph as i32,
                NativeStatus::BackendUnavailable as i32,
                NativeStatus::BackendUnsupported as i32,
                NativeStatus::BackendRequired as i32,
                NativeStatus::OperatorFallbackForbidden as i32,
                NativeStatus::ExecutionFailed as i32,
                NativeStatus::ResultDisposed as i32,
                NativeStatus::DeviceLost as i32,
                NativeStatus::AbiUnsupported as i32,
                NativeStatus::NotFound as i32,
                NativeStatus::BufferTooSmall as i32,
                NativeStatus::Internal as i32,
                NativeStatus::RevisionConflict as i32,
            ]
        );
        assert_eq!(
            [
                VX_STAGE_NONE,
                VX_STAGE_RUNTIME_CREATE,
                VX_STAGE_MODEL_LOAD,
                VX_STAGE_COMPILE,
                VX_STAGE_CONTEXT_CREATE,
                VX_STAGE_INPUT,
                VX_STAGE_EXECUTE,
                VX_STAGE_READBACK,
                VX_STAGE_CLOSE,
                VX_STAGE_DECODE,
                VX_STAGE_ADAPTER,
                VX_STAGE_TRAINER_CREATE,
                VX_STAGE_TRAINER_INPUT,
                VX_STAGE_TRAINER_STEP,
                VX_STAGE_TRAINER_COMMIT,
                VX_STAGE_TRAINER_ROLLBACK,
                VX_STAGE_TRAINER_EXPORT,
                VX_STAGE_PTQ_CREATE,
                VX_STAGE_PTQ_CALIBRATE,
                VX_STAGE_PTQ_INSPECT,
                VX_STAGE_PTQ_WRITE,
                VX_STAGE_PTQ_CLOSE,
            ],
            [
                OperationStage::None as i32,
                OperationStage::RuntimeCreate as i32,
                OperationStage::ModelLoad as i32,
                OperationStage::Compile as i32,
                OperationStage::ContextCreate as i32,
                OperationStage::Input as i32,
                OperationStage::Execute as i32,
                OperationStage::Readback as i32,
                OperationStage::Close as i32,
                OperationStage::Decode as i32,
                OperationStage::Adapter as i32,
                OperationStage::TrainerCreate as i32,
                OperationStage::TrainerInput as i32,
                OperationStage::TrainerStep as i32,
                OperationStage::TrainerCommit as i32,
                OperationStage::TrainerRollback as i32,
                OperationStage::TrainerExport as i32,
                OperationStage::PtqCreate as i32,
                OperationStage::PtqCalibrate as i32,
                OperationStage::PtqInspect as i32,
                OperationStage::PtqWrite as i32,
                OperationStage::PtqClose as i32,
            ]
        );
        assert_eq!(
            [VX_DTYPE_F32, VX_DTYPE_I8, VX_DTYPE_U8, VX_DTYPE_I32],
            [
                DataType::F32 as i32,
                DataType::I8 as i32,
                DataType::U8 as i32,
                DataType::I32 as i32,
            ]
        );
        assert_eq!(
            [VX_MEMORY_HOST, VX_MEMORY_DEVICE],
            [MemoryLocation::Host as i32, MemoryLocation::Device as i32]
        );
        assert_eq!(
            [VX_BACKEND_PREFER, VX_BACKEND_REQUIRE],
            [
                BackendPolicyMode::Prefer as i32,
                BackendPolicyMode::Require as i32,
            ]
        );
        assert_eq!(
            [VX_OPERATOR_FALLBACK_ALLOW, VX_OPERATOR_FALLBACK_FORBID],
            [
                OperatorFallback::Allow as i32,
                OperatorFallback::Forbid as i32,
            ]
        );
        assert_eq!(
            [
                VX_DECODE_ROW_DISABLED,
                VX_DECODE_ROW_AUTO,
                VX_DECODE_ROW_REQUIRED,
            ],
            [
                DecodeRowMode::Disabled as i32,
                DecodeRowMode::Auto as i32,
                DecodeRowMode::Required as i32,
            ]
        );
        assert_eq!(
            [VX_OPTIMIZER_ADAMW, VX_OPTIMIZER_SGD],
            [
                TrainingOptimizerKind::Adamw as i32,
                TrainingOptimizerKind::Sgd as i32,
            ]
        );
    }

    #[test]
    fn omitted_optimizer_uses_the_protobuf_zero_default() {
        assert_eq!(
            native_optimizer_options(None).unwrap().kind,
            TrainingOptimizerKind::Adamw as i32
        );
        assert_eq!(
            native_optimizer_options(Some(TrainerOptimizerOptions {
                kind: TrainingOptimizerKind::Sgd as i32,
                ..Default::default()
            }))
            .unwrap()
            .kind,
            TrainingOptimizerKind::Sgd as i32
        );
    }

    #[test]
    fn native_report_conversion_preserves_route_evidence() {
        let mut report = VxReport::new();
        report.policy_mode = VX_BACKEND_REQUIRE;
        report.operator_fallback = VX_OPERATOR_FALLBACK_FORBID;
        report.route_attested = 1;
        for (destination, value) in [
            (&mut report.backend[..], b"cpu".as_slice()),
            (&mut report.route_evidence[..], b"all:cpu".as_slice()),
        ] {
            for (slot, byte) in destination.iter_mut().zip(value) {
                *slot = *byte as c_char;
            }
        }
        let converted = report_message(&report);
        assert_eq!(converted.backend, "cpu");
        assert_eq!(converted.route_evidence, "all:cpu");
        assert!(converted.route_attested);
        assert_eq!(converted.policy_mode, BackendPolicyMode::Require as i32);
        assert_eq!(converted.operator_fallback, OperatorFallback::Forbid as i32);
    }
}
