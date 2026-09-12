/* Rewriting an FP32 graph into the quantized template PTQ consumes.
 *
 * This is the step that decides. Everything else in VxQuantizationService
 * measures or materializes against decisions already made here: which nodes
 * become quantized compute, where the affine boundaries sit, and what the
 * introduced tensors are called.
 *
 * It lived in Python — `tools/exporter/typed_ptq.py` — which meant PTQ was
 * Python-only even though the rest of the pipeline had been C for a long time.
 * Moving it here is what lets any language that can reach
 * `proto/volvoxai.proto` quantize a model: Python over the Synurang plugin
 * ABI, TypeScript over the WebAssembly host, and the six other targets the
 * generator emits.
 *
 * The output is a template, not a package. It names the packed weights and
 * affine parameters but does not compute them, because their values depend on
 * calibration that has not happened yet. CreatePtqPlan takes the template and
 * the plan this returns; Calibrate observes; WritePtqPackage fills in the
 * payloads.
 */
#ifndef VOLVOXAI_PTQ_AUTHORING_H
#define VOLVOXAI_PTQ_AUTHORING_H

#include <stddef.h>
#include <stdint.h>

#include "ptq_names.h"
#include "volvoxai_full_enums.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VX_PTQ_AUTHORING_NAME_CAPACITY 128u
#define VX_PTQ_AUTHORING_MESSAGE_CAPACITY 256u

/* Storage for a quantized value.
 *
 * The schema's DataType is the vocabulary, and quantization uses two of it.
 * Naming those two here — rather than declaring a parallel enum — is what
 * keeps a caller from having to translate between two spellings of int8, and
 * what keeps this from drifting when the schema gains a storage kind. */
typedef VxDataType VxPtqStorage;
#define VX_PTQ_STORAGE_I8 VX_DTYPE_I8
#define VX_PTQ_STORAGE_U8 VX_DTYPE_U8

typedef struct VxPtqAuthoringConfig {
    size_t struct_size;
    VxPtqStorage activation_storage;
    VxPTQScheme activation_scheme;
    VxPtqStorage weight_storage;
    int reduce_range;
    /* Operator kinds and node ids to leave in F32. Borrowed for the duration
     * of the call. */
    const char* const* float_operators;
    size_t float_operator_count;
    const char* const* float_nodes;
    size_t float_node_count;
    /* Nodes to consider. Empty means every node that qualifies. */
    const char* const* selected_nodes;
    size_t selected_node_count;
} VxPtqAuthoringConfig;

#define VX_PTQ_AUTHORING_CONFIG_INIT                                       \
    {                                                                      \
        sizeof(VxPtqAuthoringConfig), VX_PTQ_STORAGE_I8,                   \
            VX_PTQ_SCHEME_SYMMETRIC, VX_PTQ_STORAGE_I8, 0, NULL, 0u, NULL, \
            0u, NULL, 0u                                                   \
    }

/* One tensor calibration must observe, and how it will be represented. */
typedef struct VxPtqAuthoredObserver {
    char tensor_name[VX_PTQ_AUTHORING_NAME_CAPACITY];
    /* The byte tensor the template renamed it to, which is where its affine
     * lives. The writer fills in that affine's payload. */
    char quantized_tensor_name[VX_PTQ_AUTHORING_NAME_CAPACITY];
    VxPtqStorage storage;
    VxPTQScheme scheme;
} VxPtqAuthoredObserver;

typedef VxPTQLayerKind VxPtqAuthoredLayerKind;
#define VX_PTQ_AUTHORED_LAYER_QLINEAR VX_PTQ_LAYER_QLINEAR
#define VX_PTQ_AUTHORED_LAYER_QCONV2D VX_PTQ_LAYER_QCONV2D

/* One node that became quantized compute, and the tensors it needs packed. */
typedef struct VxPtqAuthoredLayer {
    VxPtqAuthoredLayerKind kind;
    /* Position in the source graph; the id locates the same node in the
     * template, where the affine boundaries have shifted every position. */
    int32_t node_index;
    char node_id[VX_PTQ_AUTHORING_NAME_CAPACITY];
    /* The weight axis that carries the output channels, which is the axis
     * per-channel scales are indexed by. */
    int32_t weight_axis;
    char input_tensor_name[VX_PTQ_AUTHORING_NAME_CAPACITY];
    char output_tensor_name[VX_PTQ_AUTHORING_NAME_CAPACITY];
    char source_weight_name[VX_PTQ_AUTHORING_NAME_CAPACITY];
    char packed_weight_name[VX_PTQ_AUTHORING_NAME_CAPACITY];
    char source_bias_name[VX_PTQ_AUTHORING_NAME_CAPACITY];
    char packed_bias_name[VX_PTQ_AUTHORING_NAME_CAPACITY];
} VxPtqAuthoredLayer;

/* Everything the next call needs, plus what it did.
 *
 * Owned by this module; release with vx_ptq_authored_free. `message` explains
 * a failure and is empty on success. */
typedef struct VxPtqAuthored {
    VxPtqAuthoredObserver* observers;
    size_t observer_count;
    VxPtqAuthoredLayer* layers;
    size_t layer_count;
    uint64_t quantized_nodes;
    uint64_t retained_float_nodes;
    /* Schema-owned status for the failure in message. OK on success. */
    VxStatus status;
    char message[VX_PTQ_AUTHORING_MESSAGE_CAPACITY];
} VxPtqAuthored;

/* One weight shard, already in memory. Borrowed for the duration of the
 * call: authoring reads shapes and dtypes and copies nothing. */
typedef struct VxPtqWeightBlob {
    const void* bytes;
    size_t byte_count;
} VxPtqWeightBlob;

/* Authors a template from bytes, and returns bytes.
 *
 * This is the whole of authoring. It touches no filesystem, which is what
 * lets the same implementation serve a native caller, a plugin host, and a
 * WebAssembly build with no `fopen` at all.
 *
 * `*template_json_out` receives a NUL-terminated document the caller frees
 * with vx_ptq_authored_release_template. Returns 0, or -1 with `out->message`
 * explaining why.
 *
 * Several shards may be passed. A weight the graph refers to but no shard
 * provides is a failure, not a node to skip: a graph that cannot be loaded is
 * not a graph that should be quantized. */
int vx_ptq_author_graph(const char* source_graph_json,
                        size_t source_graph_length,
                        const VxPtqWeightBlob* weights,
                        size_t weight_count,
                        const VxPtqAuthoringConfig* config,
                        char** template_json_out,
                        size_t* template_length_out,
                        VxPtqAuthored* out);

void vx_ptq_authored_release_template(char* template_json);

/* Writes an already-authored document with the native file convention. The
 * authored plan stays owned by `out`; only its typed status/message change on
 * an invalid destination or I/O failure. */
int vx_ptq_write_template_file(const char* template_graph_path,
                               const char* template_json,
                               size_t template_length,
                               VxPtqAuthored* out);

/* The same thing against paths, for a caller that has files.
 *
 * Reads `source_graph_path` and its weight shards, writes the template to
 * `template_graph_path`, and fills `out`. Nothing is written to the weights. */
int vx_ptq_author_template(const char* source_graph_path,
                           const char* const* weight_paths,
                           size_t weight_path_count,
                           const char* template_graph_path,
                           const VxPtqAuthoringConfig* config,
                           VxPtqAuthored* out);

/* Reads the same path-backed inputs but returns the authored document instead
 * of writing a file. The caller releases `*template_json_out` with
 * vx_ptq_authored_release_template. */
int vx_ptq_author_template_to_memory(
    const char* source_graph_path,
    const char* const* weight_paths,
    size_t weight_path_count,
    const VxPtqAuthoringConfig* config,
    char** template_json_out,
    size_t* template_length_out,
    VxPtqAuthored* out);

void vx_ptq_authored_init(VxPtqAuthored* out);
void vx_ptq_authored_free(VxPtqAuthored* out);

#ifdef __cplusplus
}
#endif

#endif /* VOLVOXAI_PTQ_AUTHORING_H */
