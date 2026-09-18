/* VxQuantizationService — full-profile post-training quantization authoring.
 *
 * A plan retains its Model, pins the exact current revision, snapshots the
 * caller-authored graph template at creation, and owns a private CPU engine
 * used only for calibration.
 *
 * This file exists only in the full profile. The inference profile never
 * registers the service, so its RPCs report "not implemented".
 */
#include "vx_api_convert.h"
#include "vx_api_handles.h"
#include "ptq_authoring.h"
#include "vx_training_lifecycle.h"
#include "volvoxai_ffi.h"
#include "vx_api.h"
#include "vx_api_authoring.h"

#include <string.h>

static void vx_api_retain_ptq_plan(void* pointer) {
    vx_ptq_plan_retain((VxPTQPlan*)pointer);
}

static void vx_api_release_ptq_plan_handle(void* pointer) {
    vx_ptq_plan_release((VxPTQPlan*)pointer);
}

static int vx_api_ptq_bytes_has_nul(const SynurangLiteBytes* value) {
    return value && value->len &&
           (!value->data || memchr(value->data, '\0', value->len) != NULL);
}

static int vx_api_ptq_strings_have_nul(const SynurangLiteBytes* strings,
                                       size_t count) {
    size_t index;
    if (count && !strings) return 1;
    for (index = 0; index < count; index++) {
        if (vx_api_ptq_bytes_has_nul(&strings[index])) return 1;
    }
    return 0;
}

/* AuthorPtqTemplate — FP32 graph in, quantized template and its plan out.
 *
 * This is the only RPC in the service that takes no handle. Authoring consumes
 * caller-owned bytes or one-shot file paths and has nothing to retain; binding
 * it to a loaded Model would mean a caller had to load a model it does not
 * intend to run. The plan comes back in the shape CreatePtqPlan takes it, so
 * the usual sequence is to hand the response straight on.
 */
static int vx_api_author_ptq_template(
        const VolvoxaiV1AuthorPtqTemplateRequest* request,
        VolvoxaiV1PtqTemplateInfo* response,
        void* user_data) {
    (void)user_data;
#if defined(__wasm__)
    if (request->field_template_graph_path.len ||
        (!request->field_source_graph.len &&
         (request->field_source_graph_path.len || request->field_weight_paths.len))) {
        return vx_api_report_fail(response->_allocator, &response->field_report,
                                  VX_STATUS_TRANSPORT_UNSUPPORTED, VX_STAGE_PTQ_AUTHOR,
                                  VX_CODE_TRANSPORT_UNSUPPORTED,
                                  "WASM PTQ authoring takes source_graph/weight_shards and returns template_graph bytes")
                   ? 0 : -1;
    }
#endif
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(user_data);
    VxPtqAuthoringConfig config = VX_PTQ_AUTHORING_CONFIG_INIT;
    VxPtqAuthored authored;
    const VolvoxaiV1PtqAuthoringConfig* requested = request->field_config;
    const char* source_path;
    const char* template_path;
    const char* const* weight_paths;
    VolvoxaiV1PtqObserverSpec* observers;
    VolvoxaiV1PtqLayerSpec* layers;
    SynurangLiteBytes* required;
    char* template_json = NULL;
    size_t template_length = 0u;
    int authored_failed;
    size_t index;
    int result = -1;

    vx_ptq_authored_init(&authored);
    source_path = NULL;
    weight_paths = NULL;
    if (vx_api_ptq_bytes_has_nul(&request->field_template_graph_path) ||
        (!request->field_source_graph.len &&
         (vx_api_ptq_bytes_has_nul(&request->field_source_graph_path) ||
          vx_api_ptq_strings_have_nul(request->field_weight_paths.data,
                                      request->field_weight_paths.len)))) {
        return vx_api_report_fail(
                   allocator, &response->field_report,
                   VX_STATUS_INVALID_ARGUMENT, VX_STAGE_PTQ_AUTHOR,
                   VX_CODE_INVALID_ARGUMENT, "PTQ paths must be NUL-free strings")
                   ? 0 : -1;
    }
    template_path =
        vx_api_scratch_cstr(&scratch, &request->field_template_graph_path);
    if (!request->field_source_graph.len) {
        source_path =
            vx_api_scratch_cstr(&scratch, &request->field_source_graph_path);
        weight_paths = vx_api_scratch_cstr_array(
            &scratch, request->field_weight_paths.data,
            sizeof(SynurangLiteBytes), request->field_weight_paths.len);
    }
    if (vx_api_scratch_failed(&scratch)) {
        vx_api_scratch_release(&scratch);
        return -1;
    }

    if (requested) {
        /* UNSPECIFIED means "the default", not "an error": a caller that sends
         * an empty config gets symmetric int8, which is the conservative
         * choice and the one the exporter has always used. */
        if (vx_api_ptq_strings_have_nul(requested->field_float_operators.data,
                                        requested->field_float_operators.len) ||
            vx_api_ptq_strings_have_nul(requested->field_float_nodes.data,
                                        requested->field_float_nodes.len) ||
            vx_api_ptq_strings_have_nul(requested->field_selected_nodes.data,
                                        requested->field_selected_nodes.len)) {
            vx_api_scratch_release(&scratch);
            return vx_api_report_fail(
                       allocator, &response->field_report,
                       VX_STATUS_INVALID_ARGUMENT, VX_STAGE_PTQ_AUTHOR,
                       VX_CODE_INVALID_ARGUMENT,
                       "PTQ operator and node names must be NUL-free strings")
                       ? 0 : -1;
        }
        if (requested->field_activation_dtype == VOLVOXAI_V1_DATA_TYPE_U8) {
            config.activation_storage = VX_PTQ_STORAGE_U8;
        } else if (requested->field_activation_dtype != VOLVOXAI_V1_DATA_TYPE_I8 &&
                   requested->field_activation_dtype !=
                       VOLVOXAI_V1_DATA_TYPE_UNSPECIFIED) {
            vx_api_scratch_release(&scratch);
            return vx_api_report_fail(allocator, &response->field_report,
                                      VX_STATUS_INVALID_ARGUMENT,
                                      VX_STAGE_PTQ_AUTHOR, VX_CODE_INVALID_ARGUMENT,
                                      "activations quantize to I8 or U8")
                       ? 0 : -1;
        }
        if (requested->field_weight_dtype != VOLVOXAI_V1_DATA_TYPE_I8 &&
            requested->field_weight_dtype != VOLVOXAI_V1_DATA_TYPE_UNSPECIFIED) {
            vx_api_scratch_release(&scratch);
            return vx_api_report_fail(allocator, &response->field_report,
                                      VX_STATUS_INVALID_ARGUMENT,
                                      VX_STAGE_PTQ_AUTHOR, VX_CODE_INVALID_ARGUMENT,
                                      "weights quantize to I8")
                       ? 0 : -1;
        }
        if (requested->field_activation_scheme !=
                VOLVOXAI_V1_PTQ_SCHEME_SYMMETRIC &&
            requested->field_activation_scheme !=
                VOLVOXAI_V1_PTQ_SCHEME_ASYMMETRIC) {
            vx_api_scratch_release(&scratch);
            return vx_api_report_fail(allocator, &response->field_report,
                                      VX_STATUS_INVALID_ARGUMENT,
                                      VX_STAGE_PTQ_AUTHOR, VX_CODE_INVALID_ARGUMENT,
                                      "activation_scheme must be SYMMETRIC or ASYMMETRIC")
                       ? 0 : -1;
        }
        config.activation_scheme =
            requested->field_activation_scheme == VOLVOXAI_V1_PTQ_SCHEME_ASYMMETRIC
                ? VX_PTQ_SCHEME_ASYMMETRIC
                : VX_PTQ_SCHEME_SYMMETRIC;
        config.reduce_range = requested->field_reduce_range ? 1 : 0;
        config.float_operators =
            vx_api_scratch_cstr_array(&scratch,
                                      requested->field_float_operators.data,
                                      sizeof(SynurangLiteBytes),
                                      requested->field_float_operators.len);
        config.float_operator_count = requested->field_float_operators.len;
        config.float_nodes =
            vx_api_scratch_cstr_array(&scratch,
                                      requested->field_float_nodes.data,
                                      sizeof(SynurangLiteBytes),
                                      requested->field_float_nodes.len);
        config.float_node_count = requested->field_float_nodes.len;
        config.selected_nodes =
            vx_api_scratch_cstr_array(&scratch,
                                      requested->field_selected_nodes.data,
                                      sizeof(SynurangLiteBytes),
                                      requested->field_selected_nodes.len);
        config.selected_node_count = requested->field_selected_nodes.len;
        if (vx_api_scratch_failed(&scratch)) {
            vx_api_scratch_release(&scratch);
            return -1;
        }
    }

    /* Bytes where the caller sent them, files otherwise. Input form and
     * output form are independent: either input can return response bytes or
     * write template_graph_path. Author to memory first so all four cases use
     * the same template bytes and plan. */
    if (request->field_source_graph.len) {
        VxPtqWeightBlob* blobs = NULL;
        size_t shard_count = request->field_weight_shards.len;
        if (shard_count) {
            blobs = (VxPtqWeightBlob*)vx_api_scratch_alloc(
                &scratch, sizeof(*blobs) * shard_count);
            if (!blobs) {
                vx_api_scratch_release(&scratch);
                return -1;
            }
            for (index = 0; index < shard_count; index++) {
                blobs[index].bytes = request->field_weight_shards.data[index].data;
                blobs[index].byte_count =
                    request->field_weight_shards.data[index].len;
            }
        }
        authored_failed = vx_ptq_author_graph(
            (const char*)request->field_source_graph.data,
            request->field_source_graph.len, blobs, shard_count, &config,
            &template_json, &template_length, &authored) != 0;
    } else {
        authored_failed = vx_ptq_author_template_to_memory(
            source_path, weight_paths, request->field_weight_paths.len,
            &config, &template_json, &template_length, &authored) != 0;
    }
    if (!authored_failed && template_path) {
        authored_failed = vx_ptq_write_template_file(
            template_path, template_json, template_length, &authored) != 0;
        if (!authored_failed) {
            vx_ptq_authored_release_template(template_json);
            template_json = NULL;
            template_length = 0u;
        }
    }
    if (authored_failed) {
        VxStatus failure_status = authored.status == VX_STATUS_OK
            ? VX_STATUS_INVALID_GRAPH : authored.status;
        VxOperationCode failure_code = failure_status == VX_STATUS_OUT_OF_MEMORY
            ? VX_CODE_OUT_OF_MEMORY
            : failure_status == VX_STATUS_IO_ERROR ? VX_CODE_IO_ERROR
            : failure_status == VX_STATUS_INVALID_ARGUMENT ? VX_CODE_INVALID_ARGUMENT
            : VX_CODE_INVALID_GRAPH;
        int reported = vx_api_report_fail(
            allocator, &response->field_report, failure_status,
            VX_STAGE_PTQ_AUTHOR, failure_code,
            authored.message[0] ? authored.message : "authoring failed");
        vx_ptq_authored_release_template(template_json);
        vx_ptq_authored_free(&authored);
        vx_api_scratch_release(&scratch);
        return reported ? 0 : -1;
    }
    /* With no output path, the template comes back in the response regardless
     * of whether the source arrived as bytes or as a native path. */
    if (template_json &&
        synurang_lite_bytes_assign(allocator, &response->field_template_graph,
                                   template_json, template_length) !=
            SYNURANG_LITE_OK) {
        goto done;
    }

    response->field_quantized_nodes = authored.quantized_nodes;
    response->field_retained_float_nodes = authored.retained_float_nodes;

    if (authored.observer_count) {
        observers = (VolvoxaiV1PtqObserverSpec*)allocator->allocate(
            allocator->context, sizeof(*observers) * authored.observer_count);
        required = (SynurangLiteBytes*)allocator->allocate(
            allocator->context, sizeof(*required) * authored.observer_count);
        if (!observers || !required) goto done;
        memset(required, 0, sizeof(*required) * authored.observer_count);
        response->field_observers.data = observers;
        response->field_observers.len = authored.observer_count;
        response->field_observers.cap = authored.observer_count;
        response->field_required_observations.data = required;
        response->field_required_observations.len = authored.observer_count;
        response->field_required_observations.cap = authored.observer_count;
        for (index = 0; index < authored.observer_count; index++) {
            const VxPtqAuthoredObserver* source = &authored.observers[index];
            size_t length = strlen(source->tensor_name);
            volvoxai_v1_ptq_observer_spec_init_with_allocator(&observers[index],
                                                              allocator);
            if (synurang_lite_bytes_assign(allocator,
                                           &observers[index].field_tensor_name,
                                           source->tensor_name, length) !=
                    SYNURANG_LITE_OK ||
                synurang_lite_bytes_assign(allocator, &required[index],
                                           source->tensor_name, length) !=
                    SYNURANG_LITE_OK ||
                synurang_lite_bytes_assign(
                    allocator, &observers[index].field_quantized_tensor_name,
                    source->quantized_tensor_name,
                    strlen(source->quantized_tensor_name)) !=
                    SYNURANG_LITE_OK) {
                goto done;
            }
            observers[index].field_dtype =
                source->storage == VX_PTQ_STORAGE_U8 ? VOLVOXAI_V1_DATA_TYPE_U8
                                                     : VOLVOXAI_V1_DATA_TYPE_I8;
            observers[index].field_scheme =
                source->scheme == VX_PTQ_SCHEME_ASYMMETRIC
                    ? VOLVOXAI_V1_PTQ_SCHEME_ASYMMETRIC
                    : VOLVOXAI_V1_PTQ_SCHEME_SYMMETRIC;
        }
    }

    if (authored.layer_count) {
        layers = (VolvoxaiV1PtqLayerSpec*)allocator->allocate(
            allocator->context, sizeof(*layers) * authored.layer_count);
        if (!layers) goto done;
        response->field_layers.data = layers;
        response->field_layers.len = authored.layer_count;
        response->field_layers.cap = authored.layer_count;
        for (index = 0; index < authored.layer_count; index++) {
            const VxPtqAuthoredLayer* source = &authored.layers[index];
            struct { SynurangLiteBytes* slot; const char* text; } strings[7];
            size_t which;
            volvoxai_v1_ptq_layer_spec_init_with_allocator(&layers[index],
                                                           allocator);
            layers[index].field_mode = VOLVOXAI_V1_PTQ_MODE_W8A8;
            layers[index].field_kind =
                source->kind == VX_PTQ_AUTHORED_LAYER_QCONV2D
                    ? VOLVOXAI_V1_PTQ_LAYER_KIND_QCONV2D
                    : VOLVOXAI_V1_PTQ_LAYER_KIND_QLINEAR;
            layers[index].field_node_index = source->node_index;
            layers[index].field_weight_axis = source->weight_axis;
            strings[0].slot = &layers[index].field_input_tensor_name;
            strings[0].text = source->input_tensor_name;
            strings[1].slot = &layers[index].field_output_tensor_name;
            strings[1].text = source->output_tensor_name;
            strings[2].slot = &layers[index].field_source_weight_name;
            strings[2].text = source->source_weight_name;
            strings[3].slot = &layers[index].field_packed_weight_name;
            strings[3].text = source->packed_weight_name;
            strings[4].slot = &layers[index].field_source_bias_name;
            strings[4].text = source->source_bias_name;
            strings[5].slot = &layers[index].field_packed_bias_name;
            strings[5].text = source->packed_bias_name;
            strings[6].slot = &layers[index].field_node_id;
            strings[6].text = source->node_id;
            for (which = 0; which < 7u; which++) {
                if (!strings[which].text[0]) continue;
                if (synurang_lite_bytes_assign(allocator, strings[which].slot,
                                               strings[which].text,
                                               strlen(strings[which].text)) !=
                    SYNURANG_LITE_OK) {
                    goto done;
                }
            }
        }
    }

    result = vx_api_report_ok(allocator, &response->field_report,
                              VX_STAGE_PTQ_AUTHOR) ? 0 : -1;
done:
    vx_ptq_authored_release_template(template_json);
    vx_ptq_authored_free(&authored);
    vx_api_scratch_release(&scratch);
    return result;
}

/* Fills the typed plan info, including the coverage record that replaced the
 * previous JSON string projection. */
static int vx_api_plan_info(const SynurangLiteAllocator* allocator,
                            VolvoxaiV1PtqPlanInfo* response,
                            VxPTQPlan* plan,
                            const VxPTQPlanInfo* info) {
    VolvoxaiV1PtqTensorParameters* tensors;
    VolvoxaiV1PtqProfileCoverage* profiles;
    VolvoxaiV1PtqCoverage* coverage;
    size_t index;

    response->field_calibration_batches = info->calibration_batches;
    response->field_calibration_samples = info->calibration_samples;

    if (info->tensor_count) {
        tensors = (VolvoxaiV1PtqTensorParameters*)allocator->allocate(
            allocator->context, sizeof(*tensors) * info->tensor_count);
        if (!tensors) return 0;
        response->field_tensors.data = tensors;
        response->field_tensors.len = info->tensor_count;
        response->field_tensors.cap = info->tensor_count;
        for (index = 0; index < info->tensor_count; index++) {
            VxPTQTensorParameters source = VX_PTQ_TENSOR_PARAMETERS_INIT;
            VxReport ignored = VX_REPORT_INIT;
            volvoxai_v1_ptq_tensor_parameters_init_with_allocator(&tensors[index],
                                                                  allocator);
            if (vx_ptq_plan_tensor_parameters(plan, index, &source, &ignored) !=
                VX_STATUS_OK) {
                continue;
            }
            if (source.tensor_name[0] &&
                synurang_lite_bytes_assign(allocator, &tensors[index].field_tensor_name,
                                           source.tensor_name,
                                           strlen(source.tensor_name)) !=
                    SYNURANG_LITE_OK) {
                return 0;
            }
            tensors[index].field_dtype = (VolvoxaiV1DataType)source.dtype;
            tensors[index].field_scheme = (VolvoxaiV1PtqScheme)source.scheme;
            tensors[index].field_scale = source.scale;
            tensors[index].field_zero_point = source.zero_point;
            tensors[index].field_observed_min = source.observed_min;
            tensors[index].field_observed_max = source.observed_max;
            tensors[index].field_observed_values = source.observed_values;
        }
    }

    coverage = (VolvoxaiV1PtqCoverage*)allocator->allocate(allocator->context,
                                                           sizeof(*coverage));
    if (!coverage) return 0;
    volvoxai_v1_ptq_coverage_init_with_allocator(coverage, allocator);
    response->field_coverage = coverage;
    coverage->field_complete = info->coverage_complete != 0;
    coverage->field_total_batches = info->calibration_batches;
    coverage->field_total_samples = info->calibration_samples;
    if (synurang_lite_bytes_assign(allocator, &coverage->field_format,
                                   "volvox.ptq-coverage/v1",
                                   strlen("volvox.ptq-coverage/v1")) !=
        SYNURANG_LITE_OK) {
        return 0;
    }

    if (info->profile_count) {
        profiles = (VolvoxaiV1PtqProfileCoverage*)allocator->allocate(
            allocator->context, sizeof(*profiles) * info->profile_count);
        if (!profiles) return 0;
        coverage->field_profiles.data = profiles;
        coverage->field_profiles.len = info->profile_count;
        coverage->field_profiles.cap = info->profile_count;
        for (index = 0; index < info->profile_count; index++) {
            VxPTQProfileCoverage source = VX_PTQ_PROFILE_COVERAGE_INIT;
            VxReport ignored = VX_REPORT_INIT;
            volvoxai_v1_ptq_profile_coverage_init_with_allocator(&profiles[index],
                                                                 allocator);
            if (vx_ptq_plan_profile_coverage(plan, index, &source, &ignored) !=
                VX_STATUS_OK) {
                continue;
            }
            if (source.profile_name[0] &&
                synurang_lite_bytes_assign(allocator, &profiles[index].field_name,
                                           source.profile_name,
                                           strlen(source.profile_name)) !=
                    SYNURANG_LITE_OK) {
                return 0;
            }
            profiles[index].field_batches = source.calibration_batches;
            profiles[index].field_samples = source.calibration_samples;
            /* The engine reports only counts for signatures and symbols, not
             * the values themselves, so those repeated fields stay empty
             * rather than being filled with invented entries. */
        }
    }

    if (info->revision.struct_size == sizeof(info->revision)) {
        VolvoxaiV1RevisionInfo* revision = (VolvoxaiV1RevisionInfo*)allocator->allocate(
            allocator->context, sizeof(*revision));
        if (!revision) return 0;
        volvoxai_v1_revision_info_init_with_allocator(revision, allocator);
        revision->field_graph_id = info->revision.graph_id;
        revision->field_graph_revision = info->revision.graph_revision;
        revision->field_weight_id = info->revision.weight_id;
        revision->field_weight_revision = info->revision.weight_revision;
        revision->field_adapter_id = info->revision.adapter_id;
        revision->field_adapter_revision = info->revision.adapter_revision;
        response->field_revision = revision;
    }
    return 1;
}

static int vx_api_create_ptq_plan(const VolvoxaiV1CreatePtqPlanRequest* request,
                                  VolvoxaiV1PtqPlanHandle* response,
                                  void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(user_data);
    VxPTQPlanOptions options = VX_PTQ_PLAN_OPTIONS_INIT;
    VxPTQObserverSpec* observers = NULL;
    VxPTQLayerSpec* layers = NULL;
    VxReport report = VX_REPORT_INIT;
    VxPTQPlan* plan = NULL;
    VxStatus status;
    size_t index;
    size_t count;
    VolvoxaiV1TensorSpec* specs;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxModel* model;
    int result = 0;

    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_MODEL,
                               request->field_model_id, &lease)) {
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_HANDLE_DISPOSED, VX_STAGE_PTQ_CREATE,
                                  VX_CODE_HANDLE_DISPOSED, "unknown or released model")
                   ? 0 : -1;
    }
    model = (VxModel*)lease.pointer;
    options.template_graph_path =
        vx_api_scratch_cstr(&scratch, &request->field_template_graph_path);
    options.template_graph = request->field_template_graph.data;
    options.template_graph_size = request->field_template_graph.len;
    options.profile_names = vx_api_scratch_cstr_array(&scratch,
                                                      request->field_profile_names.data,
                                                      sizeof(SynurangLiteBytes),
                                                      request->field_profile_names.len);
    options.profile_count = request->field_profile_names.len;

    if (request->field_observers.len) {
        observers = (VxPTQObserverSpec*)vx_api_scratch_alloc(
            &scratch, sizeof(*observers) * request->field_observers.len);
        if (observers) {
            for (index = 0; index < request->field_observers.len; index++) {
                const VolvoxaiV1PtqObserverSpec* source =
                    &request->field_observers.data[index];
                observers[index] = (VxPTQObserverSpec)VX_PTQ_OBSERVER_SPEC_INIT;
                observers[index].tensor_name =
                    vx_api_scratch_cstr(&scratch, &source->field_tensor_name);
                observers[index].dtype = (VxDataType)source->field_dtype;
                observers[index].scheme = (VxPTQScheme)source->field_scheme;
                observers[index].quantized_tensor_name =
                    vx_api_scratch_cstr(&scratch,
                                        &source->field_quantized_tensor_name);
            }
            options.observers = observers;
            options.observer_count = request->field_observers.len;
        }
    }

    if (request->field_layers.len) {
        layers = (VxPTQLayerSpec*)vx_api_scratch_alloc(
            &scratch, sizeof(*layers) * request->field_layers.len);
        if (layers) {
            for (index = 0; index < request->field_layers.len; index++) {
                const VolvoxaiV1PtqLayerSpec* source = &request->field_layers.data[index];
                layers[index] = (VxPTQLayerSpec)VX_PTQ_LAYER_SPEC_INIT;
                layers[index].mode = (VxPTQMode)source->field_mode;
                layers[index].kind = (VxPTQLayerKind)source->field_kind;
                layers[index].node_index = source->field_node_index;
                layers[index].weight_axis = source->field_weight_axis;
                layers[index].input_tensor_name =
                    vx_api_scratch_cstr(&scratch, &source->field_input_tensor_name);
                layers[index].output_tensor_name =
                    vx_api_scratch_cstr(&scratch, &source->field_output_tensor_name);
                layers[index].source_weight_name =
                    vx_api_scratch_cstr(&scratch, &source->field_source_weight_name);
                layers[index].packed_weight_name =
                    vx_api_scratch_cstr(&scratch, &source->field_packed_weight_name);
                layers[index].source_bias_name =
                    vx_api_scratch_cstr(&scratch, &source->field_source_bias_name);
                layers[index].packed_bias_name =
                    vx_api_scratch_cstr(&scratch, &source->field_packed_bias_name);
                layers[index].node_id =
                    vx_api_scratch_cstr(&scratch, &source->field_node_id);
            }
            options.layers = layers;
            options.layer_count = request->field_layers.len;
        }
    }

    if (vx_api_scratch_failed(&scratch)) {
        result = -1;
        goto done;
    }

    status = vx_model_create_ptq_plan(model, &options, &plan, &report);
    if (!vx_api_report_attach(allocator, &response->field_report, &report)) {
        result = -1;
        goto done;
    }
    if (status != VX_STATUS_OK) goto done;

    if (!vx_api_report_ok(allocator, &response->field_report,
                          VX_STAGE_PTQ_INSPECT)) {
        result = -1;
        goto done;
    }
    count = vx_ptq_plan_input_count(plan);
    if (count != 0u) {
        specs = (VolvoxaiV1TensorSpec*)allocator->allocate(allocator->context,
                                                           sizeof(*specs) * count);
        if (!specs) {
            result = -1;
            goto done;
        }
        response->field_inputs.data = specs;
        response->field_inputs.len = count;
        response->field_inputs.cap = count;
        for (index = 0; index < count; index++) {
            VxTensorSpec native = VX_TENSOR_SPEC_INIT;
            VxReport ignored = VX_REPORT_INIT;
            volvoxai_v1_tensor_spec_init_with_allocator(&specs[index], allocator);
            if (vx_ptq_plan_input_spec(plan, index, &native, &ignored) !=
                VX_STATUS_OK) {
                continue;
            }
            if (!vx_api_tensor_spec_from_native(allocator, &specs[index], &native)) {
                result = -1;
                goto done;
            }
        }
    }

    response->field_ptq_plan_id = vx_api_handle_insert_with_lineage(user_data,
        VX_API_HANDLE_PTQ_PLAN, plan,
        vx_api_retain_ptq_plan, vx_api_release_ptq_plan_handle, &lease.lineage);
    if (!response->field_ptq_plan_id) {
        result = vx_api_report_fail(allocator, &response->field_report,
                                    VX_STATUS_OUT_OF_MEMORY, VX_STAGE_PTQ_CREATE,
                                    VX_CODE_OUT_OF_MEMORY, "handle registry allocation failed")
                     ? 0 : -1;
    } else {
        plan = NULL; /* ownership transferred to the registry */
    }
done:
    vx_api_scratch_release(&scratch);
    if (plan) vx_ptq_plan_release(plan);
    vx_api_report_set_public_lineage(response->field_report, lease.lineage.runtime_id,
        lease.lineage.model_id, lease.lineage.compiled_model_id, lease.lineage.context_id);
    vx_api_handle_lease_release(&lease);
    return result;
}

static int vx_api_release_ptq_plan(const VolvoxaiV1PtqPlanRef* request,
                                   VolvoxaiV1OperationReport* response,
                                   void* user_data) {
    (void)user_data;
    (void)vx_api_handle_remove(user_data, VX_API_HANDLE_PTQ_PLAN,
                               request->field_ptq_plan_id);
    response->field_stage = VOLVOXAI_V1_OPERATION_STAGE_CLOSE;
    return 0;
}

static int vx_api_calibrate_ptq_plan(const VolvoxaiV1CalibratePtqPlanRequest* request,
                                     VolvoxaiV1PtqCalibrationInfo* response,
                                     void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(user_data);
    VxPTQCalibrationBatch batch = VX_PTQ_CALIBRATION_BATCH_INIT;
    VxPTQPlanInfo info = VX_PTQ_PLAN_INFO_INIT;
    VxTensorBinding* bindings = NULL;
    VxReport report = VX_REPORT_INIT;
    VxStatus status;
    size_t index;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxPTQPlan* plan;
    int result = 0;

    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_PTQ_PLAN,
                               request->field_ptq_plan_id, &lease)) {
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_HANDLE_DISPOSED, VX_STAGE_PTQ_CALIBRATE,
                                  VX_CODE_HANDLE_DISPOSED, "unknown or released plan")
                   ? 0 : -1;
    }
    plan = (VxPTQPlan*)lease.pointer;
    batch.profile_name = vx_api_scratch_cstr(&scratch, &request->field_profile_name);
    batch.sample_name = vx_api_scratch_cstr(&scratch, &request->field_sample_name);
    batch.sample_count = request->field_sample_count;

    if (request->field_inputs.len) {
        bindings = (VxTensorBinding*)vx_api_scratch_alloc(
            &scratch, sizeof(*bindings) * request->field_inputs.len);
        if (!bindings) {
            result = -1;
            goto done;
        }
        for (index = 0; index < request->field_inputs.len; index++) {
            status = vx_api_binding_from_tensor(&scratch, &bindings[index],
                                                &request->field_inputs.data[index]);
            if (status != VX_STATUS_OK) {
                result = vx_api_report_binding_fail(
                    allocator, &response->field_report, status,
                    VX_STAGE_PTQ_CALIBRATE,
                    "calibration input could not be bound", &request->field_inputs.data[index], index)
                             ? 0 : -1;
                goto done;
            }
        }
        batch.inputs = bindings;
        batch.input_count = request->field_inputs.len;
    }

    status = vx_ptq_plan_calibrate(plan, &batch, &info, &report);
    if (status == VX_STATUS_OK) {
        response->field_calibration_batches = info.calibration_batches;
        response->field_calibration_samples = info.calibration_samples;
        response->field_coverage_complete = info.coverage_complete != 0;
    }
    result = vx_api_report_attach(allocator, &response->field_report, &report) ? 0 : -1;
done:
    vx_api_scratch_release(&scratch);
    vx_api_report_set_public_lineage(response->field_report, lease.lineage.runtime_id,
        lease.lineage.model_id, lease.lineage.compiled_model_id, lease.lineage.context_id);
    vx_api_handle_lease_release(&lease);
    return result;
}

static int vx_api_inspect_ptq_plan(const VolvoxaiV1PtqPlanRef* request,
                                   VolvoxaiV1PtqPlanInfo* response,
                                   void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxPTQPlanInfo info = VX_PTQ_PLAN_INFO_INIT;
    VxReport report = VX_REPORT_INIT;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxPTQPlan* plan;
    int result = 0;

    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_PTQ_PLAN,
                               request->field_ptq_plan_id, &lease)) {
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_HANDLE_DISPOSED, VX_STAGE_PTQ_INSPECT,
                                  VX_CODE_HANDLE_DISPOSED, "unknown or released plan")
                   ? 0 : -1;
    }
    plan = (VxPTQPlan*)lease.pointer;
    if (vx_ptq_plan_info(plan, &info, &report) == VX_STATUS_OK) {
        if (!vx_api_plan_info(allocator, response, plan, &info)) {
            result = -1;
            goto done;
        }
    }
    result = vx_api_report_attach(allocator, &response->field_report, &report) ? 0 : -1;
done:
    vx_api_report_set_public_lineage(response->field_report, lease.lineage.runtime_id,
        lease.lineage.model_id, lease.lineage.compiled_model_id, lease.lineage.context_id);
    vx_api_handle_lease_release(&lease);
    return result;
}

static int vx_api_write_ptq_package(const VolvoxaiV1WritePtqPackageRequest* request,
                                    VolvoxaiV1PtqPackageInfo* response,
                                    void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(user_data);
    VxPTQPackageOptions options = VX_PTQ_PACKAGE_OPTIONS_INIT;
    unsigned char* graph_bytes = NULL;
    unsigned char* weights_bytes = NULL;
    size_t graph_size = 0;
    size_t weights_size = 0;
    VxPTQPlanInfo info = VX_PTQ_PLAN_INFO_INIT;
    VxReport report = VX_REPORT_INIT;
    VxStatus status;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxPTQPlan* plan;
    int result = 0;

#if defined(__wasm__)
    if (request->field_output_graph_path.len || request->field_output_weights_path.len)
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_TRANSPORT_UNSUPPORTED, VX_STAGE_PTQ_WRITE,
                                  VX_CODE_TRANSPORT_UNSUPPORTED, "WASM returns package bytes; omit output paths") ? 0 : -1;
#endif
    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_PTQ_PLAN,
                               request->field_ptq_plan_id, &lease)) {
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_HANDLE_DISPOSED, VX_STAGE_PTQ_WRITE,
                                  VX_CODE_HANDLE_DISPOSED, "unknown or released plan")
                   ? 0 : -1;
    }
    plan = (VxPTQPlan*)lease.pointer;
    options.output_graph_path =
        vx_api_scratch_cstr(&scratch, &request->field_output_graph_path);
    options.output_weights_path =
        vx_api_scratch_cstr(&scratch, &request->field_output_weights_path);
    if (!request->field_output_graph_path.len && !request->field_output_weights_path.len) {
        options.graph_bytes = &graph_bytes;
        options.graph_size = &graph_size;
        options.weights_bytes = &weights_bytes;
        options.weights_size = &weights_size;
    }
    if (vx_api_scratch_failed(&scratch)) {
        result = -1;
        goto done;
    }

    status = vx_ptq_plan_write_package(plan, &options, &report);
    if (status == VX_STATUS_OK) {
        if ((graph_bytes && synurang_lite_bytes_assign(
                 allocator, &response->field_graph, graph_bytes, graph_size) != SYNURANG_LITE_OK) ||
            (weights_bytes && synurang_lite_bytes_assign(
                 allocator, &response->field_weights, weights_bytes, weights_size) != SYNURANG_LITE_OK)) {
            result = -1;
            goto done;
        }
        if (options.output_graph_path &&
            synurang_lite_bytes_assign(allocator, &response->field_graph_path,
                                       options.output_graph_path,
                                       strlen(options.output_graph_path)) !=
                SYNURANG_LITE_OK) {
            result = -1;
            goto done;
        }
        if (options.output_weights_path &&
            synurang_lite_bytes_assign(allocator, &response->field_safetensors_path,
                                       options.output_weights_path,
                                       strlen(options.output_weights_path)) !=
                SYNURANG_LITE_OK) {
            result = -1;
            goto done;
        }
        if (vx_ptq_plan_info(plan, &info, &report) == VX_STATUS_OK) {
            VolvoxaiV1PtqPlanInfo* plan_info = (VolvoxaiV1PtqPlanInfo*)
                allocator->allocate(allocator->context, sizeof(*plan_info));
            if (!plan_info) {
                result = -1;
                goto done;
            }
            volvoxai_v1_ptq_plan_info_init_with_allocator(plan_info, allocator);
            response->field_plan = plan_info;
            if (!vx_api_plan_info(allocator, plan_info, plan, &info)) {
                result = -1;
                goto done;
            }
        }
    }
    result = vx_api_report_attach(allocator, &response->field_report, &report) ? 0 : -1;
done:
    free(graph_bytes);
    free(weights_bytes);
    vx_api_scratch_release(&scratch);
    vx_api_report_set_public_lineage(response->field_report, lease.lineage.runtime_id,
        lease.lineage.model_id, lease.lineage.compiled_model_id, lease.lineage.context_id);
    vx_api_handle_lease_release(&lease);
    return result;
}

#include "vx_api_weight_conversion.inc"

VX_API_UNARY(vx_api_quantize_weight, VolvoxaiV1QuantizeWeightRequest, VolvoxaiV1QuantizedWeight,
    volvoxai_v1_quantized_weight, vx_quantization_quantize_weight_respond)
VX_API_UNARY(vx_api_export_quantized_trainer_weights, VolvoxaiV1ExportQuantizedTrainerWeightsRequest, VolvoxaiV1QuantizedTrainerWeights,
    volvoxai_v1_quantized_trainer_weights, vx_quantization_export_quantized_trainer_weights_respond)
VX_API_UNARY(vx_api_dequantize_weight, VolvoxaiV1DequantizeWeightRequest, VolvoxaiV1InitializedTensor,
    volvoxai_v1_initialized_tensor, vx_quantization_dequantize_weight_respond)
VX_API_UNARY(vx_api_author_ptq_template, VolvoxaiV1AuthorPtqTemplateRequest, VolvoxaiV1PtqTemplateInfo,
    volvoxai_v1_ptq_template_info, vx_quantization_author_ptq_template_respond)
VX_API_UNARY(vx_api_create_ptq_plan, VolvoxaiV1CreatePtqPlanRequest, VolvoxaiV1PtqPlanHandle,
    volvoxai_v1_ptq_plan_handle, vx_quantization_create_ptq_plan_respond)
VX_API_UNARY(vx_api_release_ptq_plan, VolvoxaiV1PtqPlanRef, VolvoxaiV1OperationReport,
    volvoxai_v1_operation_report, vx_quantization_release_ptq_plan_respond)
VX_API_UNARY(vx_api_calibrate_ptq_plan, VolvoxaiV1CalibratePtqPlanRequest, VolvoxaiV1PtqCalibrationInfo,
    volvoxai_v1_ptq_calibration_info, vx_quantization_calibrate_ptq_plan_respond)
VX_API_UNARY(vx_api_inspect_ptq_plan, VolvoxaiV1PtqPlanRef, VolvoxaiV1PtqPlanInfo,
    volvoxai_v1_ptq_plan_info, vx_quantization_inspect_ptq_plan_respond)
VX_API_UNARY(vx_api_write_ptq_package, VolvoxaiV1WritePtqPackageRequest, VolvoxaiV1PtqPackageInfo,
    volvoxai_v1_ptq_package_info, vx_quantization_write_ptq_package_respond)

int vx_api_install_quantization_handlers(SynurangInstance* instance, VxApiRegistry* registry) {
    VxQuantizationServiceHandlers handlers;
    memset(&handlers, 0, sizeof(handlers));
    handlers.quantize_weight.message = vx_api_quantize_weight_call;
    handlers.export_quantized_trainer_weights.message = vx_api_export_quantized_trainer_weights_call;
    handlers.dequantize_weight.message = vx_api_dequantize_weight_call;
    handlers.author_ptq_template.message = vx_api_author_ptq_template_call;
    handlers.create_ptq_plan.message = vx_api_create_ptq_plan_call;
    handlers.release_ptq_plan.message = vx_api_release_ptq_plan_call;
    handlers.calibrate_ptq_plan.message = vx_api_calibrate_ptq_plan_call;
    handlers.inspect_ptq_plan.message = vx_api_inspect_ptq_plan_call;
    handlers.write_ptq_package.message = vx_api_write_ptq_package_call;
    return vx_quantization_register(instance, &handlers, registry);
}
