/* PTQ authoring, reachable from JavaScript.
 *
 * The engine's answer to "can the browser quantize a model" is the same C that
 * answers it for Python: `vx_ptq_author_graph` works on bytes, so the only
 * thing missing was a way to hand bytes across the WebAssembly boundary and
 * get bytes back.
 *
 * That boundary has no structs and no strings — only integers and a shared
 * linear memory — so the shape here is: the host allocates, writes its graph
 * and weight shards, calls one function, and reads the result out of the
 * pointers this reports. The full-profile host in `ts/full.ts` drives it, and
 * turns what it reads into the same PtqTemplateInfo the native handler returns.
 *
 * There is no second implementation in TypeScript, and that is the point. A
 * template authored in a browser is byte-identical to one authored on a
 * server because it is authored by the same code.
 */
#include "ptq_authoring.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define VX_WASM_EXPORT(name) \
    __attribute__((export_name(name), visibility("default"), used))

#define VX_PTQ_WASM_ABI_VERSION 1u

/* One call's result, read back field by field. Held rather than returned
 * because a WebAssembly function returns one number and this is several. */
typedef struct VxPtqWasmResult {
    char* template_json;
    size_t template_length;
    VxPtqAuthored authored;
} VxPtqWasmResult;

static VxPtqWasmResult vx_ptq_wasm_result;

VX_WASM_EXPORT("vx_ptq_wasm_abi_version")
uint32_t vx_ptq_wasm_abi_version(void) {
    return VX_PTQ_WASM_ABI_VERSION;
}

VX_WASM_EXPORT("vx_ptq_wasm_alloc")
void* vx_ptq_wasm_alloc(int32_t size) {
    return size > 0 ? malloc((size_t)size) : NULL;
}

VX_WASM_EXPORT("vx_ptq_wasm_free")
void vx_ptq_wasm_free(void* pointer) {
    free(pointer);
}

/* Releases the previous call's result. Called before each author so a host
 * that forgets cannot accumulate, and exported so one that remembers can free
 * promptly. */
VX_WASM_EXPORT("vx_ptq_wasm_release")
void vx_ptq_wasm_release(void) {
    vx_ptq_authored_release_template(vx_ptq_wasm_result.template_json);
    vx_ptq_authored_free(&vx_ptq_wasm_result.authored);
    memset(&vx_ptq_wasm_result, 0, sizeof(vx_ptq_wasm_result));
}

/* Authors one template.
 *
 * `shard_pointers` and `shard_lengths` are parallel arrays of `shard_count`
 * entries, which is how a variable number of buffers crosses a boundary that
 * has no arrays. Returns 1 on success and 0 on failure; either way the
 * outcome is readable through the accessors below, because a failure that
 * cannot say why is not much better than a crash.
 *
 * The configuration arrives as scalars and pointer tables rather than a
 * struct: a struct layout shared between C and JavaScript is a second contract
 * to keep in step. Repeated operator/node names use the same pointer-table
 * shape as the weight shards above.
 */
VX_WASM_EXPORT("vx_ptq_wasm_author")
int32_t vx_ptq_wasm_author(
        const char* graph_json,
        int32_t graph_length,
        const uint32_t* shard_pointers,
        const uint32_t* shard_lengths,
        int32_t shard_count,
        int32_t activation_storage,
        int32_t activation_scheme,
        int32_t weight_storage,
        const char* const* float_operators,
        int32_t float_operator_count,
        const char* const* float_nodes,
        int32_t float_node_count,
        const char* const* selected_nodes,
        int32_t selected_node_count,
        int32_t reduce_range) {
    VxPtqWeightBlob* blobs = NULL;
    VxPtqAuthoringConfig config = VX_PTQ_AUTHORING_CONFIG_INIT;
    int authored_failed;
    int32_t index;

    vx_ptq_wasm_release();
    if (!graph_json || graph_length <= 0 || shard_count < 0 ||
        float_operator_count < 0 || float_node_count < 0 ||
        selected_node_count < 0 ||
        (shard_count && (!shard_pointers || !shard_lengths)) ||
        (float_operator_count && !float_operators) ||
        (float_node_count && !float_nodes) ||
        (selected_node_count && !selected_nodes)) {
        vx_ptq_wasm_result.authored.status = VX_STATUS_INVALID_ARGUMENT;
        return 0;
    }
    if (shard_count) {
        if ((size_t)shard_count > SIZE_MAX / sizeof(*blobs)) {
            strcpy(vx_ptq_wasm_result.authored.message,
                   "weight shard table is too large");
            vx_ptq_wasm_result.authored.status = VX_STATUS_OUT_OF_MEMORY;
            return 0;
        }
        blobs = (VxPtqWeightBlob*)malloc(sizeof(*blobs) * (size_t)shard_count);
        if (!blobs) {
            strcpy(vx_ptq_wasm_result.authored.message,
                   "out of memory allocating weight shards");
            vx_ptq_wasm_result.authored.status = VX_STATUS_OUT_OF_MEMORY;
            return 0;
        }
    }
    for (index = 0; index < shard_count; index++) {
        blobs[index].bytes = (const void*)(uintptr_t)shard_pointers[index];
        blobs[index].byte_count = (size_t)shard_lengths[index];
    }

    config.activation_storage = activation_storage == VX_PTQ_STORAGE_U8
                                    ? VX_PTQ_STORAGE_U8
                                    : VX_PTQ_STORAGE_I8;
    config.activation_scheme = activation_scheme == VX_PTQ_SCHEME_ASYMMETRIC
                                   ? VX_PTQ_SCHEME_ASYMMETRIC
                                   : VX_PTQ_SCHEME_SYMMETRIC;
    config.weight_storage = weight_storage == VX_PTQ_STORAGE_I8
                                ? VX_PTQ_STORAGE_I8
                                : (VxPtqStorage)weight_storage;
    config.float_operators = float_operators;
    config.float_operator_count = (size_t)float_operator_count;
    config.float_nodes = float_nodes;
    config.float_node_count = (size_t)float_node_count;
    config.selected_nodes = selected_nodes;
    config.selected_node_count = (size_t)selected_node_count;
    config.reduce_range = reduce_range ? 1 : 0;

    authored_failed = vx_ptq_author_graph(
        graph_json, (size_t)graph_length, blobs, (size_t)shard_count, &config,
        &vx_ptq_wasm_result.template_json,
        &vx_ptq_wasm_result.template_length, &vx_ptq_wasm_result.authored) != 0;
    free(blobs);
    if (authored_failed) {
        return 0;
    }
    return 1;
}

VX_WASM_EXPORT("vx_ptq_wasm_template")
const char* vx_ptq_wasm_template(void) {
    return vx_ptq_wasm_result.template_json;
}

VX_WASM_EXPORT("vx_ptq_wasm_template_length")
int32_t vx_ptq_wasm_template_length(void) {
    return (int32_t)vx_ptq_wasm_result.template_length;
}

/* Why the last call failed, as a NUL-terminated message. Empty after a
 * success. */
VX_WASM_EXPORT("vx_ptq_wasm_message")
const char* vx_ptq_wasm_message(void) {
    return vx_ptq_wasm_result.authored.message;
}

VX_WASM_EXPORT("vx_ptq_wasm_status")
int32_t vx_ptq_wasm_status(void) {
    return (int32_t)vx_ptq_wasm_result.authored.status;
}

VX_WASM_EXPORT("vx_ptq_wasm_quantized_nodes")
int32_t vx_ptq_wasm_quantized_nodes(void) {
    return (int32_t)vx_ptq_wasm_result.authored.quantized_nodes;
}

VX_WASM_EXPORT("vx_ptq_wasm_retained_float_nodes")
int32_t vx_ptq_wasm_retained_float_nodes(void) {
    return (int32_t)vx_ptq_wasm_result.authored.retained_float_nodes;
}

/* The derived plan, read one field at a time.
 *
 * A host builds PtqObserverSpec and PtqLayerSpec messages from these, which
 * is what lets the WebAssembly path hand its result to CreatePtqPlan exactly
 * as the native path does. Out-of-range indices report empty rather than
 * trapping: a host looping to a count it read separately should not be able
 * to fault the module. */
VX_WASM_EXPORT("vx_ptq_wasm_observer_count")
int32_t vx_ptq_wasm_observer_count(void) {
    return (int32_t)vx_ptq_wasm_result.authored.observer_count;
}

static const VxPtqAuthoredObserver* vx_ptq_wasm_observer(int32_t index) {
    if (index < 0 ||
        (size_t)index >= vx_ptq_wasm_result.authored.observer_count) {
        return NULL;
    }
    return &vx_ptq_wasm_result.authored.observers[index];
}

VX_WASM_EXPORT("vx_ptq_wasm_observer_tensor")
const char* vx_ptq_wasm_observer_tensor(int32_t index) {
    const VxPtqAuthoredObserver* observer = vx_ptq_wasm_observer(index);
    return observer ? observer->tensor_name : "";
}

VX_WASM_EXPORT("vx_ptq_wasm_observer_quantized")
const char* vx_ptq_wasm_observer_quantized(int32_t index) {
    const VxPtqAuthoredObserver* observer = vx_ptq_wasm_observer(index);
    return observer ? observer->quantized_tensor_name : "";
}

VX_WASM_EXPORT("vx_ptq_wasm_observer_storage")
int32_t vx_ptq_wasm_observer_storage(int32_t index) {
    const VxPtqAuthoredObserver* observer = vx_ptq_wasm_observer(index);
    return observer ? (int32_t)observer->storage : 0;
}

VX_WASM_EXPORT("vx_ptq_wasm_observer_scheme")
int32_t vx_ptq_wasm_observer_scheme(int32_t index) {
    const VxPtqAuthoredObserver* observer = vx_ptq_wasm_observer(index);
    return observer ? (int32_t)observer->scheme : 0;
}

VX_WASM_EXPORT("vx_ptq_wasm_layer_count")
int32_t vx_ptq_wasm_layer_count(void) {
    return (int32_t)vx_ptq_wasm_result.authored.layer_count;
}

static const VxPtqAuthoredLayer* vx_ptq_wasm_layer(int32_t index) {
    if (index < 0 || (size_t)index >= vx_ptq_wasm_result.authored.layer_count) {
        return NULL;
    }
    return &vx_ptq_wasm_result.authored.layers[index];
}

VX_WASM_EXPORT("vx_ptq_wasm_layer_kind")
int32_t vx_ptq_wasm_layer_kind(int32_t index) {
    const VxPtqAuthoredLayer* layer = vx_ptq_wasm_layer(index);
    return layer ? (int32_t)layer->kind : 0;
}

VX_WASM_EXPORT("vx_ptq_wasm_layer_node_index")
int32_t vx_ptq_wasm_layer_node_index(int32_t index) {
    const VxPtqAuthoredLayer* layer = vx_ptq_wasm_layer(index);
    return layer ? layer->node_index : -1;
}

VX_WASM_EXPORT("vx_ptq_wasm_layer_weight_axis")
int32_t vx_ptq_wasm_layer_weight_axis(int32_t index) {
    const VxPtqAuthoredLayer* layer = vx_ptq_wasm_layer(index);
    return layer ? layer->weight_axis : 0;
}

/* Which string, chosen by `field`, so one export covers six names instead of
 * six exports covering one each. */
enum {
    VX_PTQ_WASM_LAYER_NODE_ID = 0,
    VX_PTQ_WASM_LAYER_INPUT = 1,
    VX_PTQ_WASM_LAYER_OUTPUT = 2,
    VX_PTQ_WASM_LAYER_SOURCE_WEIGHT = 3,
    VX_PTQ_WASM_LAYER_PACKED_WEIGHT = 4,
    VX_PTQ_WASM_LAYER_SOURCE_BIAS = 5,
    VX_PTQ_WASM_LAYER_PACKED_BIAS = 6
};

VX_WASM_EXPORT("vx_ptq_wasm_layer_name")
const char* vx_ptq_wasm_layer_name(int32_t index, int32_t field) {
    const VxPtqAuthoredLayer* layer = vx_ptq_wasm_layer(index);
    if (!layer) return "";
    switch (field) {
    case VX_PTQ_WASM_LAYER_NODE_ID: return layer->node_id;
    case VX_PTQ_WASM_LAYER_INPUT: return layer->input_tensor_name;
    case VX_PTQ_WASM_LAYER_OUTPUT: return layer->output_tensor_name;
    case VX_PTQ_WASM_LAYER_SOURCE_WEIGHT: return layer->source_weight_name;
    case VX_PTQ_WASM_LAYER_PACKED_WEIGHT: return layer->packed_weight_name;
    case VX_PTQ_WASM_LAYER_SOURCE_BIAS: return layer->source_bias_name;
    case VX_PTQ_WASM_LAYER_PACKED_BIAS: return layer->packed_bias_name;
    default: return "";
    }
}
