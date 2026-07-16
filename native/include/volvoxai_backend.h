#ifndef VOLVOXAI_BACKEND_H
#define VOLVOXAI_BACKEND_H

/*
 * VolvoxAI backend SDK -- stable C ABI for an out-of-tree device backend.
 *
 * Registration only makes a backend discoverable.  It does not change the
 * active engine policy.  Select a registered backend by its exact name with
 * volvoxai_engine_configure_backend() before loading a model.  The selected
 * backend may decline individual nodes; the built-in CPU implementation is
 * always the final correctness fallback.
 *
 * Node and tensor handles are borrowed, opaque graph views.  Handles and
 * metadata pointers returned by their accessors are valid only for the
 * supports() or run() callback that received the node.  A backend may retain
 * only the pointer value returned by vx_tensor_data()/vx_tensor_cdata() as an
 * opaque device-storage key until reset() or teardown(); it may dereference
 * that pointer only while the corresponding host storage is current.
 *
 * Every backend callback runs under the engine's model lock and must not call
 * volvoxai_engine_* APIs.  supports() and run() inspect their borrowed graph
 * views only through the vx_* accessors below; lifecycle callbacks may call
 * the vendor's own driver API.  One nested callback is intentional: calling
 * vx_tensor_sync_host() from run() may synchronously invoke this backend's
 * sync_host(), which must not wait on a lock already held by run().
 *
 * ABI v1 offers only complete inference nodes.  The runtime bypasses public
 * node callbacks during training, for adapter-modified Linear nodes, for
 * legacy QTensor nodes, for privately fused producers, and for prefix/row-
 * window execution.  Those cases continue through built-in backends and the
 * CPU correctness fallback.  Because v1 cannot coordinate the runtime's
 * private attention K/V rows, selecting a public backend also disables native
 * incremental-row/decode-row negotiation; complete-node dependency
 * incremental execution remains available.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* supports()/run() results.  A declined run() must not modify its outputs. */
enum {
    VX_DECLINED = 0,
    VX_HANDLED = 1,
    VX_ERROR = -1
};

/* init() results.  An explicitly selected backend must report READY. */
enum {
    VX_INIT_READY = 0,
    VX_INIT_UNAVAILABLE = 1,
    VX_INIT_ERROR = -1
};

/* Stable storage codes; these match VolvoxAIDataType in volvoxai.h. */
typedef int32_t VxDtype;
enum {
    VX_DTYPE_F32 = 0,
    VX_DTYPE_I8 = 1,
    VX_DTYPE_U8 = 2,
    VX_DTYPE_I32 = 3,
    VX_DTYPE_F16 = 4
};

typedef int32_t VxQuantizationKind;
enum {
    VX_QUANT_NONE = 0,
    VX_QUANT_PER_TENSOR = 1,
    VX_QUANT_PER_AXIS = 2
};

typedef struct VxTensor VxTensor;
typedef struct VxNode VxNode;

/* Tensor metadata and borrowed host storage.  Data is contiguous row-major. */
const char* vx_tensor_name(const VxTensor* tensor);
VxDtype vx_tensor_dtype(const VxTensor* tensor);
int32_t vx_tensor_ndim(const VxTensor* tensor);
const int32_t* vx_tensor_shape(const VxTensor* tensor);
int64_t vx_tensor_numel(const VxTensor* tensor);
size_t vx_tensor_element_size(const VxTensor* tensor);
int vx_tensor_is_graph_input(const VxTensor* tensor);
int vx_tensor_is_weight(const VxTensor* tensor);
void* vx_tensor_data(VxTensor* tensor);
const void* vx_tensor_cdata(const VxTensor* tensor);
/* Make device-resident contents current in host storage.  This is normally
 * needed only when a device-resident backend deliberately reads on the host. */
int vx_tensor_sync_host(const VxTensor* tensor);

/* Quantization accessors return 1 when the requested descriptor exists and 0
 * otherwise.  Per-axis values are read one channel at a time so the ABI does
 * not expose the runtime's JSON or internal metadata allocation. */
VxQuantizationKind vx_tensor_quantization_kind(const VxTensor* tensor);
int vx_tensor_quant(const VxTensor* tensor, float* scale, int32_t* zero_point);
int vx_tensor_quant_axis(const VxTensor* tensor, int32_t* axis, int32_t* count);
int vx_tensor_quant_axis_value(const VxTensor* tensor, int32_t index,
                               float* scale, int32_t* zero_point);

/* Ordered operands plus their semantic blueprint keys ("input", "weight",
 * "bias", ...).  Key lookup is preferred when an operator has aliases or
 * optional operands. */
const char* vx_node_op(const VxNode* node);
int32_t vx_node_input_count(const VxNode* node);
const char* vx_node_input_key(const VxNode* node, int32_t index);
const VxTensor* vx_node_input(const VxNode* node, int32_t index);
const VxTensor* vx_node_input_by_key(const VxNode* node, const char* key);
int32_t vx_node_output_count(const VxNode* node);
const char* vx_node_output_key(const VxNode* node, int32_t index);
VxTensor* vx_node_output(const VxNode* node, int32_t index);
VxTensor* vx_node_output_by_key(const VxNode* node, const char* key);

/* Typed attribute reads from the node's params object.  A missing or mistyped
 * scalar returns the fallback.  Array reads return zero when absent, the exact
 * element count on success, and -1 for a malformed member or short buffer;
 * pass NULL,0 to validate and query the required count without copying. */
int32_t vx_node_attr_int(const VxNode* node, const char* key, int32_t fallback);
float vx_node_attr_float(const VxNode* node, const char* key, float fallback);
int vx_node_attr_bool(const VxNode* node, const char* key, int fallback);
const char* vx_node_attr_string(const VxNode* node, const char* key,
                                const char* fallback);
int32_t vx_node_attr_ints(const VxNode* node, const char* key,
                          int32_t* out, int32_t capacity);
int32_t vx_node_attr_floats(const VxNode* node, const char* key,
                            float* out, int32_t capacity);

#define VX_BACKEND_ABI_V1 UINT32_C(1)
#define VX_BACKEND_FLAG_DEVICE_RESIDENT_OUTPUTS UINT32_C(1)

typedef struct VxBackendV1 {
    /* Set to sizeof(VxBackendV1) and VX_BACKEND_ABI_V1.  struct_size lets a
     * future ABI append fields without making an older runtime read them. */
    uint32_t struct_size;
    uint32_t abi_version;
    const char* name;
    void* user_data;
    uint32_t flags;

    /* Required callbacks. teardown() runs once after a non-ready init()
     * attempt, or after a ready backend is replaced or shut down. */
    int (*init)(void* user_data);
    int (*supports)(void* user_data, const VxNode* node);
    int (*run)(void* user_data, const VxNode* node);
    void (*teardown)(void* user_data);

    /* Optional lifecycle callbacks.  begin/end and mark/sync must be supplied
     * as pairs.  end_forward() and sync_host() return zero on success.
     *
     * A host-output backend reads/writes the borrowed host pointers.  A
     * device-resident backend sets the flag above, keys its storage by those
     * stable pointers, and implements sync_host() to copy its latest value into
     * the mutable host allocation.  It may call vx_tensor_sync_host() when a
     * particular run deliberately needs a host-current input. */
    void (*reset)(void* user_data);
    void (*begin_forward)(void* user_data);
    int (*end_forward)(void* user_data);
    void (*mark_host)(void* user_data, const void* host, size_t bytes,
                      int is_weight);
    int (*sync_host)(void* user_data, void* host, size_t bytes, int is_weight);
} VxBackendV1;

/* The descriptor and name are copied.  user_data, including anything reached
 * through it, must remain valid while this backend can be selected.  Register
 * before a model is loaded.  Names are case-sensitive and must not use a
 * built-in name.  Returns zero on success. */
int volvoxai_register_backend(const VxBackendV1* backend);

/* Select a registered backend by exact name before model load.  Selection
 * initializes it immediately and fails if the backend is unavailable; failure
 * preserves the prior engine policy.  Built-in names (cpu, vulkan, opengl,
 * metal, nnapi) are accepted as a convenience. */
int volvoxai_engine_configure_backend(const char* name);

#ifdef __cplusplus
}
#endif

#endif /* VOLVOXAI_BACKEND_H */
