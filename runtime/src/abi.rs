//! FFI surface for the cc-compiled C inference engine
//! (`native/include/volvoxai.h` and `volvoxai_tokenizer.h`). These declarations
//! are the only boundary between the Rust plugin and the native engine; every
//! other module reaches the engine through the symbols re-exported here.

use std::ffi::{c_char, c_int, c_long};
use std::os::raw::c_void;

#[repr(C)]
pub(crate) struct NativeCrossEntropyLoss {
    pub name: *const c_char,
    pub logits_name: *const c_char,
    pub targets: *const c_int,
    pub target_count: c_int,
    pub ignore_index: c_int,
    pub row_index: c_int,
    pub weight: f32,
    pub normalizer: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct NativeCrossEntropyMetric {
    pub loss: f32,
    pub correct: c_int,
    pub examples: c_int,
    pub normalizer: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct NativeEngineOptions {
    pub backend: c_int,
    pub debug: c_int,
    pub cpu_threads: c_int,
}

extern "C" {
    pub(crate) fn volvoxai_engine_configure(options: *const NativeEngineOptions) -> c_int;
    pub(crate) fn volvoxai_engine_get_options(options: *mut NativeEngineOptions) -> c_int;
    pub(crate) fn volvoxai_engine_backend_name() -> *const c_char;
    pub(crate) fn volvoxai_engine_set_debug(enabled: c_int) -> c_int;
    pub(crate) fn volvoxai_engine_debug() -> c_int;
    pub(crate) fn volvoxai_engine_set_execution_row(row: c_int) -> c_int;
    pub(crate) fn volvoxai_engine_execution_row() -> c_int;
    pub(crate) fn volvoxai_engine_graph_input_count() -> c_int;
    pub(crate) fn volvoxai_engine_graph_input_name(index: c_int) -> *const c_char;
    pub(crate) fn volvoxai_engine_graph_output_count() -> c_int;
    pub(crate) fn volvoxai_engine_graph_output_name(index: c_int) -> *const c_char;
    pub(crate) fn volvoxai_engine_init_with_weight_files(
        config: *const c_char,
        weights: *const *const c_char,
        weight_count: c_int,
    ) -> c_int;
    pub(crate) fn volvoxai_engine_shutdown();
    pub(crate) fn volvoxai_engine_forward() -> c_int;
    pub(crate) fn volvoxai_engine_forward_prefix(row_count: c_int) -> c_int;
    pub(crate) fn volvoxai_engine_forward_row(row: c_int) -> c_int;
    pub(crate) fn volvoxai_engine_tensor_row_f32(
        name: *const c_char,
        row: c_int,
        count: *mut c_int,
    ) -> *const f32;
    pub(crate) fn volvoxai_engine_is_graph_input(name: *const c_char) -> c_int;
    pub(crate) fn volvoxai_engine_is_model_weight(name: *const c_char) -> c_int;
    pub(crate) fn volvoxai_engine_input_ptr(name: *const c_char, numel: *mut c_long) -> *mut f32;
    pub(crate) fn volvoxai_engine_set_input_raw(
        name: *const c_char,
        dtype: c_int,
        data: *const c_void,
        nbytes: usize,
    ) -> c_int;
    pub(crate) fn volvoxai_engine_tensor_info_ex(
        name: *const c_char,
        numel: *mut c_long,
        shape: *mut c_int,
        ndim: *mut c_int,
        dtype: *mut c_int,
        elem_size: *mut usize,
    ) -> c_int;
    pub(crate) fn volvoxai_engine_copy_tensor_raw(
        name: *const c_char,
        out: *mut c_void,
        nbytes: usize,
    ) -> c_int;
    pub(crate) fn volvoxai_engine_copy_tensor_f32(
        name: *const c_char,
        out: *mut f32,
        numel: c_long,
    ) -> c_int;
    pub(crate) fn volvoxai_engine_linear_weight_layout(
        weight_name: *const c_char,
        d_in: *mut c_int,
        d_out: *mut c_int,
        out_in: *mut c_int,
    ) -> c_int;
    pub(crate) fn volvoxai_engine_set_tensor_raw(
        name: *const c_char,
        dtype: c_int,
        data: *const c_void,
        nbytes: usize,
    ) -> c_int;
    pub(crate) fn volvoxai_engine_add_model_tensor_raw(
        name: *const c_char,
        shape: *const c_int,
        ndim: c_int,
        dtype: c_int,
        data: *const c_void,
        nbytes: usize,
    ) -> c_int;
    pub(crate) fn volvoxai_engine_remove_model_tensor(name: *const c_char) -> c_int;
    pub(crate) fn volvoxai_engine_apply_tensor_update_f32(
        name: *const c_char,
        update: *const f32,
        numel: c_long,
        mode: c_int,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        max_grad_norm: f32,
        step: c_long,
    ) -> c_int;
    pub(crate) fn volvoxai_engine_train_step_multi(
        losses: *const NativeCrossEntropyLoss,
        loss_count: c_int,
        trainable_names: *const *const c_char,
        trainable_count: c_int,
        update_mode: c_int,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        max_grad_norm: f32,
        step: c_long,
        accumulation_steps: c_int,
        flush_accumulation: c_int,
        reset_accumulation: c_int,
        out_loss: *mut f32,
        out_metrics: *mut NativeCrossEntropyMetric,
        out_accumulated_microbatches: *mut c_int,
        out_update_applied: *mut c_int,
    ) -> c_int;
    pub(crate) fn volvoxai_engine_require_training_backend(backend: c_int) -> c_int;
    pub(crate) fn volvoxai_engine_last_training_backend() -> c_int;
    pub(crate) fn volvoxai_engine_tensor_weight_file_index(name: *const c_char) -> c_int;
    pub(crate) fn volvoxai_engine_save_weight_file(index: c_int, path: *const c_char) -> c_int;
    pub(crate) fn volvoxai_engine_save_optimizer_state(
        path: *const c_char,
        training_step: c_long,
    ) -> c_int;
    pub(crate) fn volvoxai_engine_load_optimizer_state(
        path: *const c_char,
        training_step: *mut c_long,
    ) -> c_int;
    pub(crate) fn volvoxai_engine_adapter_stage_json(
        manifest_json: *const c_char,
        tensor_names: *const *const c_char,
        tensor_data: *const *const c_void,
        tensor_dtypes: *const c_int,
        tensor_nbytes: *const usize,
        tensor_count: c_int,
    ) -> c_int;
    pub(crate) fn volvoxai_engine_adapter_clone_update_with_metadata(
        source_version: *const c_char,
        new_adapter_id: *const c_char,
        new_version_id: *const c_char,
        tensor_names: *const *const c_char,
        tensor_data: *const *const c_void,
        tensor_dtypes: *const c_int,
        tensor_nbytes: *const usize,
        update_modes: *const c_int,
        tensor_count: c_int,
        metadata_json: *const c_char,
    ) -> c_int;
    pub(crate) fn volvoxai_engine_adapter_list_json() -> *mut c_char;
    pub(crate) fn volvoxai_engine_adapter_activate(name: *const c_char) -> c_int;
    pub(crate) fn volvoxai_engine_adapter_remove(name: *const c_char) -> c_int;
    pub(crate) fn volvoxai_engine_adapter_merge(name: *const c_char) -> c_int;
    pub(crate) fn volvoxai_engine_adapter_unmerge() -> c_int;
    pub(crate) fn volvoxai_engine_adapter_save(name: *const c_char, path: *const c_char) -> c_int;
    pub(crate) fn volvoxai_engine_adapter_route_begin(name: *const c_char) -> c_int;
    pub(crate) fn volvoxai_engine_adapter_route_begin_many(
        names: *const *const c_char,
        scales: *const f32,
        count: c_int,
    ) -> c_int;
    pub(crate) fn volvoxai_engine_adapter_route_end();
    pub(crate) fn volvoxai_engine_inspect_model_json(
        include_graph: c_int,
        include_tensors: c_int,
        include_weight_files: c_int,
        include_params: c_int,
    ) -> *mut c_char;
    pub(crate) fn volvoxai_engine_patch_graph_json(
        patches_json: *const c_char,
        rebuild_plan: c_int,
        reoptimize: c_int,
    ) -> c_int;
    pub(crate) fn volvoxai_engine_save_config(path: *const c_char) -> c_int;
    pub(crate) fn free(ptr: *mut c_void);

    pub(crate) fn volvoxai_tokenizer_init(vocab: *const c_char, merges: *const c_char) -> *mut c_void;
    pub(crate) fn volvoxai_tokenizer_free(t: *mut c_void);
    pub(crate) fn volvoxai_tokenizer_encode(
        t: *mut c_void,
        text: *const c_char,
        tokens: *mut c_int,
        max: c_int,
    ) -> c_int;
    pub(crate) fn volvoxai_tokenizer_decode(t: *mut c_void, id: c_int) -> *const c_char;

    pub(crate) fn vk_training_available() -> c_int;
    pub(crate) fn opengl_make_current() -> c_int;
    pub(crate) fn opengl_release_current();
    pub(crate) fn opengl_training_available() -> c_int;
    #[cfg(target_os = "macos")]
    pub(crate) fn metal_training_available() -> c_int;
}
