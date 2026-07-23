#ifndef VOLVOX_ENGINE_INTERNAL_H
#define VOLVOX_ENGINE_INTERNAL_H

#include "backend_config.h"
#include "runtime_state.h"
#include <stddef.h>
#include <stdint.h>

typedef struct VxBackend VxBackend;

#ifndef VOLVOXAI_ENABLE_TRAINING
#define VOLVOXAI_ENABLE_TRAINING 0
#endif

/* Core graph storage resolves through the calling context's explicit scope. */
#define g_t (vx_engine_state_current()->tensors)
#define g_nt (vx_engine_state_current()->tensor_count)
#define g_n (vx_engine_state_current()->nodes)
#define g_nn (vx_engine_state_current()->node_count)
#define g_qlinear_meta (vx_engine_state_current()->qlinear_metadata)
#define g_qconv_meta (vx_engine_state_current()->qconv_metadata)
#define g_qembedding_meta (vx_engine_state_current()->qembedding_metadata)
#define g_blob (vx_engine_state_current()->weight_blob)
#define g_weight_files (vx_engine_state_current()->weight_files)
#define g_weight_paths (vx_engine_state_current()->weight_paths)
#define g_weight_file_count (vx_engine_state_current()->weight_file_count)
#define g_graph_root (vx_engine_state_current()->graph_root)
#define g_first_input (vx_engine_state_current()->first_input)
#define g_loaded (vx_engine_state_current()->loaded)
#define g_weight_caches_dirty (vx_engine_state_current()->weight_caches_dirty)
#define g_active_row (vx_engine_state_current()->active_row)
#define g_prefix_rows (vx_engine_state_current()->prefix_rows)
#define g_prefix_row_capacity \
    (vx_engine_state_current()->prefix_row_capacity)
#define g_kcache (vx_engine_state_current()->kcache)
#define g_vcache (vx_engine_state_current()->vcache)
#define g_qwcache (vx_engine_state_current()->qwcache)
#define g_f16wcache (vx_engine_state_current()->f16wcache)
#define g_f16bcache (vx_engine_state_current()->f16bcache)
#define g_conv_wcache (vx_engine_state_current()->conv_wcache)
#define g_q8wcache (vx_engine_state_current()->q8wcache)
#define g_q8wcache_bytes (vx_engine_state_current()->q8wcache_bytes)
#define g_concat_sigmoid_fuse (vx_engine_state_current()->concat_sigmoid_fuse)
#define g_graph_opt_stats (vx_engine_state_current()->graph_opt_stats)
#define g_node_fusion (vx_engine_state_current()->node_fusion)
#define g_tensor_name_index (vx_engine_state_current()->tensor_name_index)
#define g_tensor_name_index_count (vx_engine_state_current()->tensor_name_index_count)
#define g_retired_f16_storage (vx_engine_state_current()->retired_f16_storage)
#define g_retired_f16_storage_count \
    (vx_engine_state_current()->retired_f16_storage_count)
#define g_arena_bufs (vx_engine_state_current()->arena_buffers)
#define g_arena_nbufs (vx_engine_state_current()->arena_buffer_count)
#define g_arena_tensor_indices (vx_engine_state_current()->arena_tensor_indices)
#define g_arena_tensor_count (vx_engine_state_current()->arena_tensor_count)
#define g_removed_graph_outputs (vx_engine_state_current()->removed_graph_outputs)
#define g_removed_graph_output_count \
    (vx_engine_state_current()->removed_graph_output_count)
#define g_graph_patch_reused_old_tensors \
    (vx_engine_state_current()->graph_patch_reused_old_tensors)
#define g_graph_patch_reused_indices \
    (vx_engine_state_current()->graph_patch_reused_indices)
#define g_graph_patch_reused_count \
    (vx_engine_state_current()->graph_patch_reused_count)
#define g_dirty (vx_engine_state_current()->incremental_dirty)
#define g_cache_valid (vx_engine_state_current()->incremental_cache_valid)
#define g_arena_detached (vx_engine_state_current()->incremental_arena_detached)
#define g_hybrid_row_active \
    (vx_engine_state_current()->incremental_hybrid_row_active)
#define g_hybrid_row_nodes \
    (vx_engine_state_current()->incremental_hybrid_row_nodes)
#define g_hybrid_prepared_row \
    (vx_engine_state_current()->incremental_hybrid_prepared_row)
#define g_incremental_plan (vx_engine_state_current()->incremental_plan)
#define g_active_decode_session \
    (vx_engine_state_current()->active_decode_session)
#if VOLVOXAI_ENABLE_TRAINING
#define g_opt_states (vx_engine_state_current()->optimizer_states)
#define g_training_accumulation \
    (vx_engine_state_current()->training_accumulation)
#define g_dynamic_autograd_context \
    (vx_engine_state_current()->dynamic_autograd_context)
#define g_dynamic_autograd_forward_capture \
    (vx_engine_state_current()->dynamic_autograd_forward_capture)
#define g_dynamic_autograd_forward_backend \
    (vx_engine_state_current()->dynamic_autograd_forward_backend)
#define g_native_training_mode \
    (vx_engine_state_current()->native_training_mode)
#define g_native_training_counter \
    (vx_engine_state_current()->native_training_counter)
#define g_required_training_backend \
    (vx_engine_state_current()->required_training_backend)
#define g_last_training_backend \
    (vx_engine_state_current()->last_training_backend)
#define g_gpu_training_dummy \
    (vx_engine_state_current()->gpu_training_dummy)
#endif
#define g_prof (vx_engine_state_current()->profiler_entries)
#define g_nprof (vx_engine_state_current()->profiler_entry_count)
#define g_vx_runtime_node (vx_engine_state_current()->runtime_node)
#define g_vx_cpu_dispatch (vx_engine_state_current()->runtime_cpu_dispatch)
#define g_vx_cuda_replay_eligible \
    (vx_engine_state_current()->runtime_cuda_replay_eligible)
#define g_vx_cuda_route_tracking \
    (vx_engine_state_current()->runtime_cuda_route_tracking)
#define g_vx_cuda_replay_generation \
    (vx_engine_state_current()->runtime_cuda_replay_generation)
#define g_vx_forward_ok (vx_engine_state_current()->runtime_forward_ok)
#define g_vx_backend_registry (vx_engine_state_current()->backend_registry)

#define g_use_vulkan (vx_engine_state_current()->use_vulkan)
#define g_use_nnapi (vx_engine_state_current()->use_nnapi)
#define g_use_opengl (vx_engine_state_current()->use_opengl)
#define g_use_metal (vx_engine_state_current()->use_metal)
#define g_use_cuda (vx_engine_state_current()->use_cuda)
#define g_debug (vx_engine_state_current()->debug)
#define g_execution_row (vx_engine_state_current()->execution_row)

double volvoxai_engine_now_ms(void);
T* t_find(const char* name);
/* The tensor table is a dense array, while steady-state graph execution needs
 * name resolution to stay independent of the model's tensor count.  Table
 * mutators keep this private hash index synchronized through these hooks. */
void volvoxai_engine_tensor_name_index_invalidate(void);
void volvoxai_engine_tensor_name_index_add(int tensor_index);
void volvoxai_engine_tensor_name_index_rebuild(void);
int volvoxai_engine_tensor_is_removed_output(const char* name);
void materialize_tensor_f32(T* t);
void vx_runtime_backend_reset(void);
void vx_runtime_backend_reset_transients(void);
void vx_runtime_backend_teardown(void);
void vx_runtime_backend_begin_forward(int ordinary_static_replay_eligible,
                                      uint64_t model_generation);
int vx_runtime_backend_prepare_forward(void);
int vx_runtime_backend_end_forward(int forward_ok);
int vx_runtime_backend_cuda_replay_eligible(void);
void vx_runtime_backend_mark_host(const void* host, size_t bytes, int is_weight);
int vx_runtime_backend_sync_host(const void* host, size_t bytes, int is_weight);
int vx_runtime_backend_has_graph(void);
int vx_runtime_backend_stage(void);
int vx_runtime_full_graph_execution_eligible(void);
void vx_decode_session_invalidate_model_locked(void);
int vx_decode_session_active_locked(void);
void vx_decode_session_invalidate_cache_locked(void);
int vx_incremental_hybrid_row_active_locked(void);
void volvoxai_engine_clear_qlinear_metadata(void);
void volvoxai_engine_clear_qconv_metadata(void);
void volvoxai_engine_clear_qembedding_metadata(void);
int vk_sync_host_tensor(T* t);
void vk_mark_owned_tensors_host_dirty(void);
char* read_file(const char* path, long* out_size);
int volvoxai_engine_load_weight_files(const char* const* paths, int count);
int build_graph(const char* graph_path);
void volvoxai_engine_free_arena(void);
int volvoxai_engine_prepare_tensor_table_mutation(void);
void volvoxai_engine_finish_tensor_table_mutation(void);
int volvoxai_engine_refresh_weight_caches(void);
void volvoxai_engine_model_lock(void);
void volvoxai_engine_model_unlock(void);
uint64_t volvoxai_engine_model_generation_locked(void);
void volvoxai_engine_model_generation_advance_locked(void);
int volvoxai_engine_adapter_effect_active_locked(void);
int volvoxai_engine_model_route_lease_active(void);
void volvoxai_engine_metadata_lock(void);
void volvoxai_engine_metadata_unlock(void);
#if VOLVOXAI_ENABLE_TRAINING
int volvoxai_engine_training_accumulation_pending(void);
void volvoxai_engine_reset_training_accumulation(void);
/* Called only while the model mutex is held.  A live eager tape owns the
   otherwise-empty runtime projection boundary, so model/backend lifecycle
   mutations must wait until that context is destroyed. */
int vx_dynamic_autograd_active_locked(void);
#else
static inline int volvoxai_engine_training_accumulation_pending(void) { return 0; }
static inline void volvoxai_engine_reset_training_accumulation(void) {}
static inline int vx_dynamic_autograd_active_locked(void) { return 0; }
#endif
int volvoxai_engine_linear_weight_layout(const char* weight_name, int* d_in, int* d_out, int* out_in);
int volvoxai_engine_forward(void);
int volvoxai_engine_forward_locked(void);
int volvoxai_engine_forward_incremental_locked(void);
int volvoxai_engine_forward_incremental_row_locked(int row);
int volvoxai_engine_tensor_is_model_weight_locked(const char* name);
int volvoxai_engine_sync_model_weights_locked(void);
int volvoxai_engine_tensor_is_quantization_parameter_locked(const char* name);
int volvoxai_engine_add_model_tensor_raw_locked(const char* name,
                                                const int* shape, int ndim,
                                                int dtype, const void* data,
                                                size_t nbytes);
int volvoxai_engine_remove_model_tensor_locked(const char* name);
#if VOLVOXAI_ENABLE_TRAINING
int volvoxai_engine_apply_tensor_update_f32(const char* name, const float* update, long numel,
                                   int mode, float learning_rate, float beta1, float beta2,
                                   float epsilon, float weight_decay, float max_grad_norm,
                                   long step);
int volvoxai_engine_prepare_tensor_update_f32(const char* name, long numel, int mode);
int volvoxai_engine_optimizer_state_buffers_f32(
    const char* name, long numel, float** first_moment,
    float** second_moment);
int volvoxai_engine_finalize_device_tensor_update_f32(const char* name,
                                                       long numel);
int volvoxai_engine_apply_tensor_update_f32_locked(const char* name, const float* update, long numel,
                                           int mode, float learning_rate, float beta1, float beta2,
                                           float epsilon, float weight_decay, float max_grad_norm,
                                           long step);
#endif
int prepack_conv_weights(void);
int run_node(Node* n, int idx, int is_last);
int run_node_cpu_direct(Node* n, int idx, int is_last);
/* Side-effect-free preflight for canonical operators whose CPU implementation
 * can refresh exactly one [1,S,...] row after a device seed. */
int vx_runtime_node_incremental_row_compatible(Node* n, int idx, int row);
void prof_reset(void);
void prof_add_entry(const char* op, double ms);
void prof_report(void);

#endif
