#ifndef VOLVOX_ADAPTER_RUNTIME_INTERNAL_H
#define VOLVOX_ADAPTER_RUNTIME_INTERNAL_H

#include "adapter_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) && !defined(_WIN32)
#define VX_ADAPTER_INTERNAL __attribute__((visibility("hidden")))
#else
#define VX_ADAPTER_INTERNAL
#endif

/* Registry lifecycle. Engine entry points provide the public locking contract. */
VX_ADAPTER_INTERNAL int vx_adapter_stage(const VxAdapterVersionSpec* spec);
VX_ADAPTER_INTERNAL int vx_adapter_clone_update(const char* source_version, const char* new_adapter_id,
                                                const char* new_version_id, const VxAdapterTensorUpdate* updates,
                                                int update_count);
VX_ADAPTER_INTERNAL int vx_adapter_clone_update_with_metadata(const char* source_version, const char* new_adapter_id,
                                                              const char* new_version_id, const VxAdapterTensorUpdate* updates,
                                                              int update_count, const char* metadata_json);
VX_ADAPTER_INTERNAL int vx_adapter_activate(const char* version_id);
VX_ADAPTER_INTERNAL int vx_adapter_remove(const char* version_id);
VX_ADAPTER_INTERNAL void vx_adapter_reset(void);

VX_ADAPTER_INTERNAL int vx_adapter_load_safetensors(const char* path, const char* version_override,
                                                    char* out_version_id, size_t out_version_id_size);
VX_ADAPTER_INTERNAL int vx_adapter_save_safetensors(const char* version_id, const char* path);
VX_ADAPTER_INTERNAL char* vx_adapter_list_json(void);

VX_ADAPTER_INTERNAL int vx_adapter_request_begin(const char* version_id);
VX_ADAPTER_INTERNAL int vx_adapter_request_begin_many(const char* const* version_ids, const float* scales, int count);
VX_ADAPTER_INTERNAL void vx_adapter_request_end(void);
VX_ADAPTER_INTERNAL int vx_adapter_run_begin(void);
VX_ADAPTER_INTERNAL void vx_adapter_run_end(void);
VX_ADAPTER_INTERNAL int vx_adapter_run_has_target(const char* weight_name);
VX_ADAPTER_INTERNAL unsigned long vx_adapter_debug_run_registry_lock_count(void);
VX_ADAPTER_INTERNAL void vx_adapter_debug_reset_run_registry_lock_count(void);

/* A request may route one adapter per batch member. Linear backends query the
 * resulting contiguous row segments while the run holds version references;
 * A/B pointers remain valid until vx_adapter_run_end(). row_start is relative
 * to the caller's selected row window. */
typedef struct {
    const float* a;
    const float* b;
    int row_start;
    int row_count;
    int rank;
    float scale;
} VxAdapterLinearSegment;
VX_ADAPTER_INTERNAL int vx_adapter_run_linear_segments(
    const char* weight_name, int batch_size, long row_offset, int rows,
    long total_rows, int d_in, int d_out,
    VxAdapterLinearSegment* segments, int capacity);

VX_ADAPTER_INTERNAL int vx_adapter_apply_linear(const char* weight_name, const float* input,
                                                const void* base_weight, VxDataType base_dtype, const float* bias,
                                                float* output, int rows, int d_in, int d_out,
                                                int base_out_in, int batch_size, long row_offset, long total_rows);

VX_ADAPTER_INTERNAL int vx_adapter_materialize_weight(const char* version_id, const char* weight_name,
                                                      const void* base_weight, VxDataType base_dtype,
                                                      int d_in, int d_out, int base_out_in, float* out_weight);
VX_ADAPTER_INTERNAL int vx_adapter_target_count(const char* version_id);
VX_ADAPTER_INTERNAL int vx_adapter_target_info(const char* version_id, int index, VxAdapterTargetInfo* out);
VX_ADAPTER_INTERNAL int vx_adapter_get_active(char* out, size_t out_size);

#undef VX_ADAPTER_INTERNAL

#ifdef __cplusplus
}
#endif

#endif
