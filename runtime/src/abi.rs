//! Exact native full-profile ABI declarations consumed by the FFI plugin.

#![allow(dead_code)]

use std::ffi::{c_char, c_int, c_void};

use crate::pb::{
    BackendPolicyMode, DataType, DecodeRowMode, MemoryLocation, NativeStatus, OperationStage,
    OperatorFallback, PtqLayerKind, PtqMode, PtqScheme, TrainingOptimizerKind,
};

pub(crate) const VX_STATUS_OK: c_int = NativeStatus::Ok as c_int;
pub(crate) const VX_STATUS_INVALID_ARGUMENT: c_int = NativeStatus::InvalidArgument as c_int;
pub(crate) const VX_STATUS_OUT_OF_MEMORY: c_int = NativeStatus::OutOfMemory as c_int;
pub(crate) const VX_STATUS_HANDLE_DISPOSED: c_int = NativeStatus::HandleDisposed as c_int;
pub(crate) const VX_STATUS_IO_ERROR: c_int = NativeStatus::IoError as c_int;
pub(crate) const VX_STATUS_INVALID_GRAPH: c_int = NativeStatus::InvalidGraph as c_int;
pub(crate) const VX_STATUS_BACKEND_UNAVAILABLE: c_int = NativeStatus::BackendUnavailable as c_int;
pub(crate) const VX_STATUS_BACKEND_UNSUPPORTED: c_int = NativeStatus::BackendUnsupported as c_int;
pub(crate) const VX_STATUS_BACKEND_REQUIRED: c_int = NativeStatus::BackendRequired as c_int;
pub(crate) const VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN: c_int =
    NativeStatus::OperatorFallbackForbidden as c_int;
pub(crate) const VX_STATUS_EXECUTION_FAILED: c_int = NativeStatus::ExecutionFailed as c_int;
pub(crate) const VX_STATUS_RESULT_DISPOSED: c_int = NativeStatus::ResultDisposed as c_int;
pub(crate) const VX_STATUS_DEVICE_LOST: c_int = NativeStatus::DeviceLost as c_int;
pub(crate) const VX_STATUS_ABI_UNSUPPORTED: c_int = NativeStatus::AbiUnsupported as c_int;
pub(crate) const VX_STATUS_NOT_FOUND: c_int = NativeStatus::NotFound as c_int;
pub(crate) const VX_STATUS_BUFFER_TOO_SMALL: c_int = NativeStatus::BufferTooSmall as c_int;
pub(crate) const VX_STATUS_INTERNAL: c_int = NativeStatus::Internal as c_int;
pub(crate) const VX_STATUS_REVISION_CONFLICT: c_int = NativeStatus::RevisionConflict as c_int;

pub(crate) const VX_STAGE_NONE: c_int = OperationStage::None as c_int;
pub(crate) const VX_STAGE_RUNTIME_CREATE: c_int = OperationStage::RuntimeCreate as c_int;
pub(crate) const VX_STAGE_MODEL_LOAD: c_int = OperationStage::ModelLoad as c_int;
pub(crate) const VX_STAGE_COMPILE: c_int = OperationStage::Compile as c_int;
pub(crate) const VX_STAGE_CONTEXT_CREATE: c_int = OperationStage::ContextCreate as c_int;
pub(crate) const VX_STAGE_INPUT: c_int = OperationStage::Input as c_int;
pub(crate) const VX_STAGE_EXECUTE: c_int = OperationStage::Execute as c_int;
pub(crate) const VX_STAGE_READBACK: c_int = OperationStage::Readback as c_int;
pub(crate) const VX_STAGE_CLOSE: c_int = OperationStage::Close as c_int;
pub(crate) const VX_STAGE_DECODE: c_int = OperationStage::Decode as c_int;
pub(crate) const VX_STAGE_ADAPTER: c_int = OperationStage::Adapter as c_int;
pub(crate) const VX_STAGE_TRAINER_CREATE: c_int = OperationStage::TrainerCreate as c_int;
pub(crate) const VX_STAGE_TRAINER_INPUT: c_int = OperationStage::TrainerInput as c_int;
pub(crate) const VX_STAGE_TRAINER_STEP: c_int = OperationStage::TrainerStep as c_int;
pub(crate) const VX_STAGE_TRAINER_COMMIT: c_int = OperationStage::TrainerCommit as c_int;
pub(crate) const VX_STAGE_TRAINER_ROLLBACK: c_int = OperationStage::TrainerRollback as c_int;
pub(crate) const VX_STAGE_TRAINER_EXPORT: c_int = OperationStage::TrainerExport as c_int;
pub(crate) const VX_STAGE_PTQ_CREATE: c_int = OperationStage::PtqCreate as c_int;
pub(crate) const VX_STAGE_PTQ_CALIBRATE: c_int = OperationStage::PtqCalibrate as c_int;
pub(crate) const VX_STAGE_PTQ_INSPECT: c_int = OperationStage::PtqInspect as c_int;
pub(crate) const VX_STAGE_PTQ_WRITE: c_int = OperationStage::PtqWrite as c_int;
pub(crate) const VX_STAGE_PTQ_CLOSE: c_int = OperationStage::PtqClose as c_int;

pub(crate) const VX_DTYPE_F32: c_int = DataType::F32 as c_int;
pub(crate) const VX_DTYPE_I8: c_int = DataType::I8 as c_int;
pub(crate) const VX_DTYPE_U8: c_int = DataType::U8 as c_int;
pub(crate) const VX_DTYPE_I32: c_int = DataType::I32 as c_int;

pub(crate) const VX_BACKEND_PREFER: c_int = BackendPolicyMode::Prefer as c_int;
pub(crate) const VX_BACKEND_REQUIRE: c_int = BackendPolicyMode::Require as c_int;
pub(crate) const VX_OPERATOR_FALLBACK_ALLOW: c_int = OperatorFallback::Allow as c_int;
pub(crate) const VX_OPERATOR_FALLBACK_FORBID: c_int = OperatorFallback::Forbid as c_int;

pub(crate) const VX_MEMORY_HOST: c_int = MemoryLocation::Host as c_int;
pub(crate) const VX_MEMORY_DEVICE: c_int = MemoryLocation::Device as c_int;
pub(crate) const VX_DECODE_ROW_DISABLED: c_int = DecodeRowMode::Disabled as c_int;
pub(crate) const VX_DECODE_ROW_AUTO: c_int = DecodeRowMode::Auto as c_int;
pub(crate) const VX_DECODE_ROW_REQUIRED: c_int = DecodeRowMode::Required as c_int;
pub(crate) const VX_MAX_TENSOR_RANK: usize = 8;
pub(crate) const VX_DIMENSION_FIXED: c_int = 1;
pub(crate) const VX_DIMENSION_SYMBOLIC: c_int = 2;
pub(crate) const VX_MAX_TRAINING_LOSSES: usize = 8;
pub(crate) const VX_TRAINING_NAME_CAPACITY: usize = 128;
pub(crate) const VX_PTQ_NAME_CAPACITY: usize = 128;

pub(crate) const VX_OPTIMIZER_SGD: c_int = TrainingOptimizerKind::Sgd as c_int;
pub(crate) const VX_OPTIMIZER_ADAMW: c_int = TrainingOptimizerKind::Adamw as c_int;
pub(crate) const VX_PTQ_MODE_W8A8: c_int = PtqMode::W8a8 as c_int;
pub(crate) const VX_PTQ_SCHEME_SYMMETRIC: c_int = PtqScheme::Symmetric as c_int;
pub(crate) const VX_PTQ_SCHEME_ASYMMETRIC: c_int = PtqScheme::Asymmetric as c_int;
pub(crate) const VX_PTQ_LAYER_QLINEAR: c_int = PtqLayerKind::Qlinear as c_int;
pub(crate) const VX_PTQ_LAYER_QCONV2D: c_int = PtqLayerKind::Qconv2d as c_int;

const BACKEND_CAPACITY: usize = 64;
const DEVICE_CAPACITY: usize = 128;
const REASON_CAPACITY: usize = 64;
const MESSAGE_CAPACITY: usize = 256;
const CANDIDATES_CAPACITY: usize = 2048;
const ROUTE_CAPACITY: usize = 512;
const FALLBACK_CAPACITY: usize = 256;
const NODE_CAPACITY: usize = 128;
const DECODE_CAPACITY: usize = 128;

macro_rules! opaque_handle {
    ($name:ident) => {
        #[repr(C)]
        pub(crate) struct $name {
            _private: [u8; 0],
        }
    };
}

opaque_handle!(VxRuntime);
opaque_handle!(VxModel);
opaque_handle!(VxCompiledModel);
opaque_handle!(VxExecutionContext);
opaque_handle!(VxResult);
opaque_handle!(VxTrainer);
opaque_handle!(VxPTQPlan);

#[repr(C)]
pub(crate) struct VxReport {
    pub struct_size: usize,
    pub status: c_int,
    pub stage: c_int,
    pub execution_id: u64,
    pub runtime_id: u64,
    pub model_id: u64,
    pub compiled_model_id: u64,
    pub context_id: u64,
    pub graph_id: u64,
    pub graph_revision: u64,
    pub weight_id: u64,
    pub weight_revision: u64,
    pub adapter_id: u64,
    pub adapter_revision: u64,
    pub allocated_bytes: u64,
    pub result_bytes: u64,
    pub compile_time_ms: f64,
    pub execution_time_ms: f64,
    pub policy_mode: c_int,
    pub operator_fallback: c_int,
    pub candidate_count: u32,
    pub tier_fallback_used: c_int,
    pub operator_fallback_used: c_int,
    pub route_attested: c_int,
    pub backend: [c_char; BACKEND_CAPACITY],
    pub device: [c_char; DEVICE_CAPACITY],
    pub reason: [c_char; REASON_CAPACITY],
    pub message: [c_char; MESSAGE_CAPACITY],
    pub candidate_outcomes: [c_char; CANDIDATES_CAPACITY],
    pub route_evidence: [c_char; ROUTE_CAPACITY],
    pub fallback_evidence: [c_char; FALLBACK_CAPACITY],
    pub offending_node: [c_char; NODE_CAPACITY],
    pub decode_state: [c_char; DECODE_CAPACITY],
}

impl VxReport {
    pub(crate) fn new() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>(),
            status: VX_STATUS_OK,
            stage: VX_STAGE_NONE,
            execution_id: 0,
            runtime_id: 0,
            model_id: 0,
            compiled_model_id: 0,
            context_id: 0,
            graph_id: 0,
            graph_revision: 0,
            weight_id: 0,
            weight_revision: 0,
            adapter_id: 0,
            adapter_revision: 0,
            allocated_bytes: 0,
            result_bytes: 0,
            compile_time_ms: 0.0,
            execution_time_ms: 0.0,
            policy_mode: VX_BACKEND_PREFER,
            operator_fallback: VX_OPERATOR_FALLBACK_ALLOW,
            candidate_count: 0,
            tier_fallback_used: 0,
            operator_fallback_used: 0,
            route_attested: 0,
            backend: [0; BACKEND_CAPACITY],
            device: [0; DEVICE_CAPACITY],
            reason: [0; REASON_CAPACITY],
            message: [0; MESSAGE_CAPACITY],
            candidate_outcomes: [0; CANDIDATES_CAPACITY],
            route_evidence: [0; ROUTE_CAPACITY],
            fallback_evidence: [0; FALLBACK_CAPACITY],
            offending_node: [0; NODE_CAPACITY],
            decode_state: [0; DECODE_CAPACITY],
        }
    }
}

#[repr(C)]
pub(crate) struct VxRuntimeOptions {
    pub struct_size: usize,
    pub debug: c_int,
    pub cpu_threads: c_int,
}

#[repr(C)]
pub(crate) struct VxModelSource {
    pub struct_size: usize,
    pub graph_path: *const c_char,
    pub weight_paths: *const *const c_char,
    pub weight_path_count: usize,
    pub bank_residency: *const c_void,
    pub bank_residency_count: usize,
}

#[repr(C)]
pub(crate) struct VxBackendPolicy {
    pub struct_size: usize,
    pub mode: c_int,
    pub operator_fallback: c_int,
    pub backends: *const *const c_char,
    pub backend_count: usize,
}

#[repr(C)]
pub(crate) struct VxContextOptions {
    pub struct_size: usize,
    pub decode_row_mode: c_int,
    pub require_incremental: c_int,
}

#[repr(C)]
pub(crate) struct VxRevisionInfo {
    pub struct_size: usize,
    pub graph_id: u64,
    pub graph_revision: u64,
    pub weight_id: u64,
    pub weight_revision: u64,
    pub adapter_id: u64,
    pub adapter_revision: u64,
}

impl VxRevisionInfo {
    pub(crate) fn new() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>(),
            graph_id: 0,
            graph_revision: 0,
            weight_id: 0,
            weight_revision: 0,
            adapter_id: 0,
            adapter_revision: 0,
        }
    }
}

#[repr(C)]
pub(crate) struct VxAdapterSource {
    pub struct_size: usize,
    pub adapter_name: *const c_char,
    pub package_path: *const c_char,
    pub version_name: *const c_char,
}

#[repr(C)]
pub(crate) struct VxAdapterRevision {
    pub struct_size: usize,
    pub adapter_id: u64,
    pub adapter_revision: u64,
}

impl VxAdapterRevision {
    pub(crate) fn new() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>(),
            adapter_id: 0,
            adapter_revision: 0,
        }
    }
}

#[repr(C)]
pub(crate) struct VxTrainerOptions {
    pub struct_size: usize,
    pub backend: *const c_char,
    pub rng_seed: u64,
}

#[repr(C)]
pub(crate) struct VxCrossEntropyLoss {
    pub struct_size: usize,
    pub name: *const c_char,
    pub logits_name: *const c_char,
    pub targets: *const i32,
    pub target_count: usize,
    pub ignore_index: i32,
    pub row_index: i32,
    pub weight: f32,
    pub normalizer: f32,
}

#[repr(C)]
pub(crate) struct VxOptimizerOptions {
    pub struct_size: usize,
    pub kind: c_int,
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub max_gradient_norm: f32,
}

impl VxOptimizerOptions {
    pub(crate) fn new() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>(),
            kind: VX_OPTIMIZER_ADAMW,
            learning_rate: 1.0e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1.0e-8,
            weight_decay: 0.0,
            max_gradient_norm: 0.0,
        }
    }
}

#[repr(C)]
pub(crate) struct VxTrainStepOptions {
    pub struct_size: usize,
    pub inputs: *const VxTensorBinding,
    pub input_count: usize,
    pub losses: *const VxCrossEntropyLoss,
    pub loss_count: usize,
    pub trainable_names: *const *const c_char,
    pub trainable_count: usize,
    pub optimizer: VxOptimizerOptions,
    pub accumulation_steps: u32,
    pub flush_accumulation: c_int,
    pub reset_accumulation: c_int,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct VxTrainingMetric {
    pub name: [c_char; VX_TRAINING_NAME_CAPACITY],
    pub loss: f32,
    pub correct: c_int,
    pub examples: c_int,
    pub normalizer: f32,
}

impl VxTrainingMetric {
    const fn new() -> Self {
        Self {
            name: [0; VX_TRAINING_NAME_CAPACITY],
            loss: 0.0,
            correct: 0,
            examples: 0,
            normalizer: 0.0,
        }
    }
}

#[repr(C)]
pub(crate) struct VxTrainStepResult {
    pub struct_size: usize,
    pub microbatch_id: u64,
    pub optimizer_step: u64,
    pub accumulated_microbatches: u32,
    pub update_applied: c_int,
    pub loss: f32,
    pub metric_count: usize,
    pub metrics: [VxTrainingMetric; VX_MAX_TRAINING_LOSSES],
    pub backend: [c_char; BACKEND_CAPACITY],
}

impl VxTrainStepResult {
    pub(crate) fn new() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>(),
            microbatch_id: 0,
            optimizer_step: 0,
            accumulated_microbatches: 0,
            update_applied: 0,
            loss: 0.0,
            metric_count: 0,
            metrics: [VxTrainingMetric::new(); VX_MAX_TRAINING_LOSSES],
            backend: [0; BACKEND_CAPACITY],
        }
    }
}

#[repr(C)]
pub(crate) struct VxPTQObserverSpec {
    pub struct_size: usize,
    pub tensor_name: *const c_char,
    pub dtype: c_int,
    pub scheme: c_int,
}

#[repr(C)]
pub(crate) struct VxPTQLayerSpec {
    pub struct_size: usize,
    pub mode: c_int,
    pub kind: c_int,
    pub node_index: i32,
    pub weight_axis: i32,
    pub input_tensor_name: *const c_char,
    pub output_tensor_name: *const c_char,
    pub source_weight_name: *const c_char,
    pub packed_weight_name: *const c_char,
    pub source_bias_name: *const c_char,
    pub packed_bias_name: *const c_char,
}

#[repr(C)]
pub(crate) struct VxPTQPlanOptions {
    pub struct_size: usize,
    pub template_graph_path: *const c_char,
    pub profile_names: *const *const c_char,
    pub profile_count: usize,
    pub observers: *const VxPTQObserverSpec,
    pub observer_count: usize,
    pub layers: *const VxPTQLayerSpec,
    pub layer_count: usize,
}

#[repr(C)]
pub(crate) struct VxPTQCalibrationBatch {
    pub struct_size: usize,
    pub profile_name: *const c_char,
    pub sample_name: *const c_char,
    pub sample_count: u64,
    pub inputs: *const VxTensorBinding,
    pub input_count: usize,
}

#[repr(C)]
pub(crate) struct VxPTQPlanInfo {
    pub struct_size: usize,
    pub calibration_batches: u64,
    pub calibration_samples: u64,
    pub tensor_count: usize,
    pub profile_count: usize,
    pub covered_profile_count: usize,
    pub coverage_complete: c_int,
    pub revision: VxRevisionInfo,
}

impl VxPTQPlanInfo {
    pub(crate) fn new() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>(),
            calibration_batches: 0,
            calibration_samples: 0,
            tensor_count: 0,
            profile_count: 0,
            covered_profile_count: 0,
            coverage_complete: 0,
            revision: VxRevisionInfo::new(),
        }
    }
}

#[repr(C)]
pub(crate) struct VxPTQTensorParameters {
    pub struct_size: usize,
    pub tensor_name: [c_char; VX_PTQ_NAME_CAPACITY],
    pub dtype: c_int,
    pub scheme: c_int,
    pub scale: f32,
    pub zero_point: i32,
    pub observed_min: f32,
    pub observed_max: f32,
    pub observed_values: u64,
}

impl VxPTQTensorParameters {
    pub(crate) fn new() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>(),
            tensor_name: [0; VX_PTQ_NAME_CAPACITY],
            dtype: VX_DTYPE_I8,
            scheme: VX_PTQ_SCHEME_SYMMETRIC,
            scale: 0.0,
            zero_point: 0,
            observed_min: 0.0,
            observed_max: 0.0,
            observed_values: 0,
        }
    }
}

#[repr(C)]
pub(crate) struct VxPTQPackageOptions {
    pub struct_size: usize,
    pub output_graph_path: *const c_char,
    pub output_weights_path: *const c_char,
}

#[repr(C)]
pub(crate) struct VxTensorInfo {
    pub struct_size: usize,
    pub name: *const c_char,
    pub dtype: c_int,
    pub rank: u32,
    pub shape: [i64; VX_MAX_TENSOR_RANK],
    pub byte_size: usize,
    pub location: c_int,
}

impl VxTensorInfo {
    pub(crate) fn new() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>(),
            name: std::ptr::null(),
            dtype: VX_DTYPE_F32,
            rank: 0,
            shape: [0; VX_MAX_TENSOR_RANK],
            byte_size: 0,
            location: VX_MEMORY_HOST,
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct VxDimensionConstraint {
    pub struct_size: usize,
    pub kind: c_int,
    pub symbol: *const c_char,
    pub min: i64,
    pub max: i64,
    pub multiple_of: i64,
}

impl VxDimensionConstraint {
    fn new() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>(),
            kind: VX_DIMENSION_FIXED,
            symbol: std::ptr::null(),
            min: 1,
            max: 1,
            multiple_of: 1,
        }
    }
}

#[repr(C)]
pub(crate) struct VxTensorSpec {
    pub struct_size: usize,
    pub name: *const c_char,
    pub dtype: c_int,
    pub rank: u32,
    pub dimensions: [VxDimensionConstraint; VX_MAX_TENSOR_RANK],
    pub location: c_int,
}

impl VxTensorSpec {
    pub(crate) fn new() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>(),
            name: std::ptr::null(),
            dtype: VX_DTYPE_F32,
            rank: 0,
            dimensions: [VxDimensionConstraint::new(); VX_MAX_TENSOR_RANK],
            location: VX_MEMORY_HOST,
        }
    }
}

#[repr(C)]
pub(crate) struct VxTensorBinding {
    pub struct_size: usize,
    pub name: *const c_char,
    pub dtype: c_int,
    pub rank: u32,
    pub shape: [i64; VX_MAX_TENSOR_RANK],
    pub data: *const c_void,
    pub byte_size: usize,
    pub location: c_int,
}

extern "C" {
    pub(crate) fn vx_status_string(status: c_int) -> *const c_char;

    pub(crate) fn vx_runtime_create(
        options: *const VxRuntimeOptions,
        out_runtime: *mut *mut VxRuntime,
        report: *mut VxReport,
    ) -> c_int;
    pub(crate) fn vx_runtime_retain(runtime: *mut VxRuntime);
    pub(crate) fn vx_runtime_release(runtime: *mut VxRuntime);
    pub(crate) fn vx_runtime_close(runtime: *mut VxRuntime, report: *mut VxReport) -> c_int;
    pub(crate) fn vx_runtime_load_model(
        runtime: *mut VxRuntime,
        source: *const VxModelSource,
        out_model: *mut *mut VxModel,
        report: *mut VxReport,
    ) -> c_int;
    pub(crate) fn vx_model_retain(model: *mut VxModel);
    pub(crate) fn vx_model_release(model: *mut VxModel);
    pub(crate) fn vx_model_revision_info(
        model: *mut VxModel,
        info: *mut VxRevisionInfo,
        report: *mut VxReport,
    ) -> c_int;
    pub(crate) fn vx_model_publish_adapter(
        model: *mut VxModel,
        source: *const VxAdapterSource,
        published: *mut VxAdapterRevision,
        report: *mut VxReport,
    ) -> c_int;

    pub(crate) fn vx_model_create_trainer(
        model: *mut VxModel,
        options: *const VxTrainerOptions,
        out_trainer: *mut *mut VxTrainer,
        report: *mut VxReport,
    ) -> c_int;
    pub(crate) fn vx_trainer_retain(trainer: *mut VxTrainer);
    pub(crate) fn vx_trainer_release(trainer: *mut VxTrainer);
    pub(crate) fn vx_trainer_close(trainer: *mut VxTrainer, report: *mut VxReport) -> c_int;
    pub(crate) fn vx_trainer_input_count(trainer: *mut VxTrainer) -> usize;
    pub(crate) fn vx_trainer_input_spec(
        trainer: *mut VxTrainer,
        index: usize,
        spec: *mut VxTensorSpec,
        report: *mut VxReport,
    ) -> c_int;
    pub(crate) fn vx_trainer_train_step(
        trainer: *mut VxTrainer,
        options: *const VxTrainStepOptions,
        result: *mut VxTrainStepResult,
        report: *mut VxReport,
    ) -> c_int;
    pub(crate) fn vx_trainer_commit(
        trainer: *mut VxTrainer,
        published: *mut VxRevisionInfo,
        report: *mut VxReport,
    ) -> c_int;
    pub(crate) fn vx_trainer_rollback(trainer: *mut VxTrainer, report: *mut VxReport) -> c_int;
    pub(crate) fn vx_trainer_export_weights(
        trainer: *mut VxTrainer,
        output_paths: *const *const c_char,
        output_path_count: usize,
        report: *mut VxReport,
    ) -> c_int;

    pub(crate) fn vx_model_create_ptq_plan(
        model: *mut VxModel,
        options: *const VxPTQPlanOptions,
        out_plan: *mut *mut VxPTQPlan,
        report: *mut VxReport,
    ) -> c_int;
    pub(crate) fn vx_ptq_plan_retain(plan: *mut VxPTQPlan);
    pub(crate) fn vx_ptq_plan_release(plan: *mut VxPTQPlan);
    pub(crate) fn vx_ptq_plan_close(plan: *mut VxPTQPlan, report: *mut VxReport) -> c_int;
    pub(crate) fn vx_ptq_plan_input_count(plan: *mut VxPTQPlan) -> usize;
    pub(crate) fn vx_ptq_plan_input_spec(
        plan: *mut VxPTQPlan,
        index: usize,
        spec: *mut VxTensorSpec,
        report: *mut VxReport,
    ) -> c_int;
    pub(crate) fn vx_ptq_plan_calibrate(
        plan: *mut VxPTQPlan,
        batch: *const VxPTQCalibrationBatch,
        info: *mut VxPTQPlanInfo,
        report: *mut VxReport,
    ) -> c_int;
    pub(crate) fn vx_ptq_plan_info(
        plan: *mut VxPTQPlan,
        info: *mut VxPTQPlanInfo,
        report: *mut VxReport,
    ) -> c_int;
    pub(crate) fn vx_ptq_plan_coverage_json(
        plan: *mut VxPTQPlan,
        output: *mut c_char,
        output_capacity: usize,
        required_size: *mut usize,
        report: *mut VxReport,
    ) -> c_int;
    pub(crate) fn vx_ptq_plan_tensor_parameters(
        plan: *mut VxPTQPlan,
        index: usize,
        parameters: *mut VxPTQTensorParameters,
        report: *mut VxReport,
    ) -> c_int;
    pub(crate) fn vx_ptq_plan_write_package(
        plan: *mut VxPTQPlan,
        options: *const VxPTQPackageOptions,
        report: *mut VxReport,
    ) -> c_int;

    pub(crate) fn vx_model_compile(
        model: *mut VxModel,
        policy: *const VxBackendPolicy,
        out_compiled: *mut *mut VxCompiledModel,
        report: *mut VxReport,
    ) -> c_int;
    pub(crate) fn vx_compiled_model_retain(compiled: *mut VxCompiledModel);
    pub(crate) fn vx_compiled_model_release(compiled: *mut VxCompiledModel);
    pub(crate) fn vx_compiled_model_report(
        compiled: *const VxCompiledModel,
        report: *mut VxReport,
    ) -> c_int;
    pub(crate) fn vx_compiled_model_create_context(
        compiled: *mut VxCompiledModel,
        options: *const VxContextOptions,
        out_context: *mut *mut VxExecutionContext,
        report: *mut VxReport,
    ) -> c_int;

    pub(crate) fn vx_execution_context_retain(context: *mut VxExecutionContext);
    pub(crate) fn vx_execution_context_release(context: *mut VxExecutionContext);
    pub(crate) fn vx_execution_context_close(
        context: *mut VxExecutionContext,
        report: *mut VxReport,
    ) -> c_int;
    pub(crate) fn vx_execution_context_input_count(context: *mut VxExecutionContext) -> usize;
    pub(crate) fn vx_execution_context_input_spec(
        context: *mut VxExecutionContext,
        index: usize,
        spec: *mut VxTensorSpec,
        report: *mut VxReport,
    ) -> c_int;
    pub(crate) fn vx_execution_context_execute(
        context: *mut VxExecutionContext,
        inputs: *const VxTensorBinding,
        input_count: usize,
        out_result: *mut *mut VxResult,
        report: *mut VxReport,
    ) -> c_int;
    pub(crate) fn vx_execution_context_decode_seed(
        context: *mut VxExecutionContext,
        inputs: *const VxTensorBinding,
        input_count: usize,
        out_result: *mut *mut VxResult,
        report: *mut VxReport,
    ) -> c_int;
    pub(crate) fn vx_execution_context_decode_step(
        context: *mut VxExecutionContext,
        position: c_int,
        inputs: *const VxTensorBinding,
        input_count: usize,
        out_result: *mut *mut VxResult,
        report: *mut VxReport,
    ) -> c_int;
    pub(crate) fn vx_execution_context_decode_reset(
        context: *mut VxExecutionContext,
        report: *mut VxReport,
    ) -> c_int;
    pub(crate) fn vx_execution_context_select_adapter(
        context: *mut VxExecutionContext,
        revision: *const VxAdapterRevision,
        report: *mut VxReport,
    ) -> c_int;
    pub(crate) fn vx_execution_context_rebind_adapter(
        context: *mut VxExecutionContext,
        report: *mut VxReport,
    ) -> c_int;

    pub(crate) fn vx_result_retain(result: *mut VxResult);
    pub(crate) fn vx_result_release(result: *mut VxResult);
    pub(crate) fn vx_result_execution_id(result: *const VxResult) -> u64;
    pub(crate) fn vx_result_output_count(result: *const VxResult) -> usize;
    pub(crate) fn vx_result_output_info(
        result: *const VxResult,
        index: usize,
        info: *mut VxTensorInfo,
        report: *mut VxReport,
    ) -> c_int;
    pub(crate) fn vx_result_read(
        result: *const VxResult,
        name: *const c_char,
        destination: *mut c_void,
        capacity: usize,
        required: *mut usize,
        report: *mut VxReport,
    ) -> c_int;
}
