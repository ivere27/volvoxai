/* One full-profile quantization plan lifecycle. Policies select numerical
 * behavior; model family and bit-pair shorthand never select an API path. */
#include "vx_api_convert.h"
#include "vx_api_handles.h"
#include "ptq_authoring.h"
#include "packed_linear.h"
#include "vx_training_lifecycle.h"
#include "volvoxai_ffi.h"
#include "vx_api.h"
#include "vx_api_authoring.h"
#include "generated/operator_vocabulary.h"
#include "cJSON.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static void vx_api_retain_quantization_plan(void* p) {
    vx_quantization_plan_retain((VxQuantizationPlan*)p);
}
static void vx_api_release_quantization_plan_handle(void* p) {
    vx_quantization_plan_release((VxQuantizationPlan*)p);
}
static int vx_api_ptq_bytes_has_nul(const SynurangLiteBytes* value) {
    return value && value->len &&
        (!value->data || memchr(value->data, '\0', value->len) != NULL);
}
static int vx_api_ptq_strings_have_nul(const SynurangLiteBytes* values, size_t count) {
    if (count && !values) return 1;
    for (size_t i = 0; i < count; i++)
        if (vx_api_ptq_bytes_has_nul(&values[i])) return 1;
    return 0;
}
static int vx_api_quantization_names_valid(const SynurangLiteBytes* values, size_t count) {
    if (vx_api_ptq_strings_have_nul(values, count)) return 0;
    for (size_t i = 0; i < count; i++) if (!values[i].len) return 0;
    return 1;
}

/* Decode the schema's typed policy; resolution and capability checks happen
 * once against the immutable source graph, before a handle is published. */
static int vx_api_weight_policy(const VolvoxaiV1WeightQuantization* weight) {
    return weight && weight->field_format &&
        (weight->field_format->field_bits == 2 || weight->field_format->field_bits == 3 || weight->field_format->field_bits == 4 ||
         weight->field_format->field_bits == 8) &&
        (weight->field_algorithm == VOLVOXAI_V1_WEIGHT_ALGORITHM_RTN ||
         weight->field_algorithm == VOLVOXAI_V1_WEIGHT_ALGORITHM_MSE_CLIP ||
         weight->field_algorithm == VOLVOXAI_V1_WEIGHT_ALGORITHM_IMPORTANCE);
}

static int vx_api_quantization_config(const VolvoxaiV1QuantizationConfig* requested,
        VxApiScratch* scratch, VxPtqAuthoringConfig* config, int* packed) {
    if (!requested || !vx_api_weight_policy(requested->field_weights)) return 0;
    const VolvoxaiV1WeightQuantization* weight = requested->field_weights;
    const VolvoxaiV1StaticActivations* activation = requested->field_static_activations;
    config->weight_only = activation == NULL;
    *packed = config->weight_only;
    config->mode = weight->field_format->field_bits == 2 ? VX_PTQ_MODE_W2A8 :
                   weight->field_format->field_bits == 3 ? VX_PTQ_MODE_W3A8 :
                   weight->field_format->field_bits == 4 ? VX_PTQ_MODE_W4A8 : VX_PTQ_MODE_W8A8;
    config->weight_algorithm = (VxWeightAlgorithm)weight->field_algorithm;
    if (activation) {
        if ((activation->field_dtype != VOLVOXAI_V1_DATA_TYPE_I8 &&
             activation->field_dtype != VOLVOXAI_V1_DATA_TYPE_U8) ||
            (activation->field_scheme != VOLVOXAI_V1_AFFINE_SCHEME_SYMMETRIC &&
             activation->field_scheme != VOLVOXAI_V1_AFFINE_SCHEME_ASYMMETRIC)) return 0;
        config->activation_storage = (VxPtqStorage)activation->field_dtype;
        config->activation_scheme = (VxAffineScheme)activation->field_scheme;
        config->reduce_range = activation->field_reduce_weight_range;
    }
    uint32_t group = weight->field_format->field_group_size;
    config->group_size = group;
    if (activation ? (group != 0 ||
                       (weight->field_format->field_bits == 8
                        ? weight->field_algorithm != VOLVOXAI_V1_WEIGHT_ALGORITHM_RTN
                        : (config->reduce_range ||
                           (weight->field_algorithm != VOLVOXAI_V1_WEIGHT_ALGORITHM_RTN &&
                            weight->field_algorithm != VOLVOXAI_V1_WEIGHT_ALGORITHM_MSE_CLIP &&
                            weight->field_algorithm != VOLVOXAI_V1_WEIGHT_ALGORITHM_IMPORTANCE))))
                   : !vx_packed_group_size_supported((int)group)) return 0;
    if (requested->field_rules.len) {
        VxQuantizationRule* rules = vx_api_scratch_alloc(scratch,
            requested->field_rules.len * sizeof(*rules));
        if (!rules) return 0;
        config->rules = rules;
        config->rule_count = requested->field_rules.len;
        for (size_t i = 0; i < config->rule_count; i++) {
            const VolvoxaiV1QuantizationRule* source = &requested->field_rules.data[i];
            VxQuantizationRule* target = &rules[i];
            const SynurangLiteBytes* match = source->which_selector == 1 ? &source->field_node_id :
                source->which_selector == 2 ? &source->field_operator_name :
                source->which_selector == 3 ? &source->field_node_glob : NULL;
            if (!match || !match->len || vx_api_ptq_bytes_has_nul(match)) return 0;
            target->selector = source->which_selector;
            target->match = vx_api_scratch_cstr(scratch, match);
            if (!target->match) return 0;
            if (source->which_selector == 2) {
                target->operator_kind = vx_operator_kind_from_name(target->match);
                if (target->operator_kind == VX_OP_UNSPECIFIED) return 0;
            }
            if (source->which_action == 5 && source->field_keep_float) target->keep_float = 1;
            else if (source->which_action == 4 && vx_api_weight_policy(source->field_weights)) {
                target->bits = source->field_weights->field_format->field_bits;
                target->group_size = source->field_weights->field_format->field_group_size;
                target->algorithm = (VxWeightAlgorithm)source->field_weights->field_algorithm;
            } else return 0;
        }
    }
    return 1;
}

static int vx_api_coverage_details(const SynurangLiteAllocator* allocator,
        VolvoxaiV1CalibrationCoverage* coverage, VxPTQPlan* plan) {
    char* snapshot = vx_ptq_plan_coverage_snapshot(plan);
    cJSON* root = snapshot ? cJSON_Parse(snapshot) : NULL;
    free(snapshot);
    if (!root) return 0;
    int ok = 0;
    cJSON* fingerprint = cJSON_GetObjectItemCaseSensitive(root, "logicalFingerprint");
    cJSON* profiles = cJSON_GetObjectItemCaseSensitive(root, "profiles");
    if (!cJSON_IsString(fingerprint) || !cJSON_IsArray(profiles) ||
        (size_t)cJSON_GetArraySize(profiles) != coverage->field_profiles.len ||
        synurang_lite_bytes_assign(allocator, &coverage->field_logical_fingerprint,
            fingerprint->valuestring, strlen(fingerprint->valuestring))) goto done;
    for (size_t i = 0; i < coverage->field_profiles.len; i++) {
        VolvoxaiV1CalibrationProfileCoverage* target = &coverage->field_profiles.data[i];
        cJSON* profile = cJSON_GetArrayItem(profiles, (int)i);
        cJSON* signatures = cJSON_GetObjectItemCaseSensitive(profile, "signatures");
        cJSON* symbols = cJSON_GetObjectItemCaseSensitive(profile, "symbols");
        cJSON* activations = cJSON_GetObjectItemCaseSensitive(profile, "activationSamples");
        size_t n = (size_t)cJSON_GetArraySize(signatures);
        if (n) {
            target->field_signatures.data = allocator->allocate(allocator->context, n * sizeof(SynurangLiteBytes));
            if (!target->field_signatures.data) goto done;
            memset(target->field_signatures.data, 0, n * sizeof(SynurangLiteBytes));
            target->field_signatures.len = target->field_signatures.cap = n;
            for (size_t j = 0; j < n; j++) {
                cJSON* value = cJSON_GetArrayItem(signatures, (int)j);
                if (!cJSON_IsString(value) || synurang_lite_bytes_assign(allocator,
                    &target->field_signatures.data[j], value->valuestring, strlen(value->valuestring))) goto done;
            }
        }
        n = (size_t)cJSON_GetArraySize(symbols);
        if (n) {
            target->field_symbols.data = allocator->allocate(allocator->context, n * sizeof(*target->field_symbols.data));
            if (!target->field_symbols.data) goto done;
            target->field_symbols.len = target->field_symbols.cap = n;
            for (size_t j = 0; j < n; j++)
                volvoxai_v1_calibration_symbol_coverage_init_with_allocator(&target->field_symbols.data[j], allocator);
            size_t j = 0;
            for (cJSON* value = symbols->child; value; value = value->next, j++) {
                VolvoxaiV1CalibrationSymbolCoverage* symbol = &target->field_symbols.data[j];
                cJSON* minimum = cJSON_GetObjectItemCaseSensitive(value, "minimum");
                cJSON* maximum = cJSON_GetObjectItemCaseSensitive(value, "maximum");
                if (!value->string || !cJSON_IsNumber(minimum) || !cJSON_IsNumber(maximum) ||
                    synurang_lite_bytes_assign(allocator, &symbol->field_symbol, value->string, strlen(value->string))) goto done;
                symbol->field_minimum = (int64_t)minimum->valuedouble;
                symbol->field_maximum = (int64_t)maximum->valuedouble;
            }
        }
        n = (size_t)cJSON_GetArraySize(activations);
        if (n) {
            target->field_activations.data = allocator->allocate(allocator->context, n * sizeof(*target->field_activations.data));
            if (!target->field_activations.data) goto done;
            target->field_activations.len = target->field_activations.cap = n;
            for (size_t j = 0; j < n; j++)
                volvoxai_v1_calibration_activation_coverage_init_with_allocator(&target->field_activations.data[j], allocator);
            size_t j = 0;
            for (cJSON* value = activations->child; value; value = value->next, j++) {
                VolvoxaiV1CalibrationActivationCoverage* activation = &target->field_activations.data[j];
                if (!value->string || !cJSON_IsNumber(value) || synurang_lite_bytes_assign(allocator,
                    &activation->field_tensor_name, value->string, strlen(value->string))) goto done;
                activation->field_observed_values = (uint64_t)value->valuedouble;
            }
        }
    }
    ok = 1;
done:
    cJSON_Delete(root);
    return ok;
}

/* Fills the typed plan info, including the coverage record that replaced the
 * previous JSON string projection. */
static int vx_api_plan_info(const SynurangLiteAllocator* allocator,
                            VolvoxaiV1CalibrationState* response,
                            VxPTQPlan* plan,
                            const VxPTQPlanInfo* info) {
    VolvoxaiV1CalibratedAffine* tensors;
    VolvoxaiV1CalibrationProfileCoverage* profiles;
    VolvoxaiV1CalibrationCoverage* coverage;
    size_t index;

    response->field_batches = info->calibration_batches;
    response->field_samples = info->calibration_samples;

    if (info->tensor_count) {
        tensors = (VolvoxaiV1CalibratedAffine*)allocator->allocate(
            allocator->context, sizeof(*tensors) * info->tensor_count);
        if (!tensors) return 0;
        response->field_affines.data = tensors;
        response->field_affines.len = info->tensor_count;
        response->field_affines.cap = info->tensor_count;
        for (index = 0; index < info->tensor_count; index++)
            volvoxai_v1_calibrated_affine_init_with_allocator(&tensors[index], allocator);
        for (index = 0; index < info->tensor_count; index++) {
            VxPTQTensorParameters source = VX_PTQ_TENSOR_PARAMETERS_INIT;
            VxReport ignored = VX_REPORT_INIT;
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
            tensors[index].field_scheme = (VolvoxaiV1AffineScheme)source.scheme;
            tensors[index].field_scale = source.scale;
            tensors[index].field_zero_point = source.zero_point;
            tensors[index].field_observed_min = source.observed_min;
            tensors[index].field_observed_max = source.observed_max;
            tensors[index].field_observed_values = source.observed_values;
        }
    }

    coverage = (VolvoxaiV1CalibrationCoverage*)allocator->allocate(allocator->context,
                                                           sizeof(*coverage));
    if (!coverage) return 0;
    volvoxai_v1_calibration_coverage_init_with_allocator(coverage, allocator);
    response->field_coverage = coverage;
    coverage->field_complete = info->coverage_complete != 0;
    if (synurang_lite_bytes_assign(allocator, &coverage->field_format,
                                   "volvox.calibration-coverage/v2",
                                   strlen("volvox.calibration-coverage/v2")) !=
        SYNURANG_LITE_OK) {
        return 0;
    }

    if (info->profile_count) {
        profiles = (VolvoxaiV1CalibrationProfileCoverage*)allocator->allocate(
            allocator->context, sizeof(*profiles) * info->profile_count);
        if (!profiles) return 0;
        coverage->field_profiles.data = profiles;
        coverage->field_profiles.len = info->profile_count;
        coverage->field_profiles.cap = info->profile_count;
        for (index = 0; index < info->profile_count; index++)
            volvoxai_v1_calibration_profile_coverage_init_with_allocator(&profiles[index], allocator);
        for (index = 0; index < info->profile_count; index++) {
            VxPTQProfileCoverage source = VX_PTQ_PROFILE_COVERAGE_INIT;
            VxReport ignored = VX_REPORT_INIT;
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
        }
    }

    return vx_api_coverage_details(allocator, coverage, plan);
}

static int vx_api_quantization_info(VolvoxaiV1QuantizationPlanInfo* response,
        VxQuantizationPlan* plan) {
    const SynurangLiteAllocator* allocator = response->_allocator;
    response->field_requires_calibration = vx_quantization_plan_requires_calibration(plan);
    response->field_ready = !response->field_requires_calibration;
    const char* backend = vx_quantization_plan_calibration_backend(plan);
    if (backend && synurang_lite_bytes_assign(allocator,
        &response->field_calibration_backend, backend, strlen(backend)) != SYNURANG_LITE_OK)
        return 0;
    uint64_t ignored;
    vx_quantization_plan_counts(plan, &ignored, &response->field_float_nodes);
    VxPTQPlan* calibration = vx_quantization_plan_static(plan);
    if (calibration) {
        VxPTQPlanInfo info = VX_PTQ_PLAN_INFO_INIT;
        VxReport report = VX_REPORT_INIT;
        if (vx_ptq_plan_info(calibration, &info, &report) != VX_STATUS_OK) return 0;
        VolvoxaiV1CalibrationState* detail = allocator->allocate(allocator->context, sizeof(*detail));
        if (!detail) return 0;
        volvoxai_v1_calibration_state_init_with_allocator(detail, allocator);
        response->field_calibration = detail;
        if (!vx_api_plan_info(allocator, detail, calibration, &info)) return 0;
        if (detail->field_batches) detail->field_importance_tensors = vx_quantization_plan_importance_count(plan);
        response->field_ready = info.coverage_complete != 0;
    }
    size_t count = vx_quantization_plan_input_count(plan);
    if (count) {
        VolvoxaiV1TensorSpec* specs = allocator->allocate(allocator->context, count * sizeof(*specs));
        if (!specs) return 0;
        response->field_calibration_inputs.data = specs;
        response->field_calibration_inputs.len = response->field_calibration_inputs.cap = count;
        for (size_t i = 0; i < count; i++)
            volvoxai_v1_tensor_spec_init_with_allocator(&specs[i], allocator);
        for (size_t i = 0; i < count; i++) {
            VxTensorSpec spec = VX_TENSOR_SPEC_INIT;
            VxReport report = VX_REPORT_INIT;
            if (vx_quantization_plan_input_spec(plan, i, &spec, &report) != VX_STATUS_OK ||
                !vx_api_tensor_spec_from_native(allocator, &specs[i], &spec)) return 0;
        }
    }
    const VxQuantizationNode* nodes = vx_quantization_plan_nodes(plan, &count);
    if (!count) return 1;
    VolvoxaiV1QuantizationNodeInfo* resolved = allocator->allocate(allocator->context, count * sizeof(*resolved));
    VolvoxaiV1QuantizedNodeCount* counts = allocator->allocate(allocator->context, count * sizeof(*counts));
    if (!resolved || !counts) return 0;
    response->field_resolved_nodes.data = resolved;
    response->field_resolved_nodes.len = response->field_resolved_nodes.cap = count;
    response->field_quantized.data = counts;
    response->field_quantized.cap = count;
    for (size_t i = 0; i < count; i++) {
        volvoxai_v1_quantization_node_info_init_with_allocator(&resolved[i], allocator);
        volvoxai_v1_quantized_node_count_init_with_allocator(&counts[i], allocator);
    }
    for (size_t i = 0; i < count; i++) {
        const VxQuantizationNode* node = &nodes[i];
        if (synurang_lite_bytes_assign(allocator, &resolved[i].field_node_id,
                node->node_id, strlen(node->node_id))) return 0;
        if (node->keep_float) {
            resolved[i].which_action = 3;
            resolved[i].field_keep_float = allocator->allocate(allocator->context, sizeof(*resolved[i].field_keep_float));
            if (!resolved[i].field_keep_float) return 0;
            volvoxai_v1_empty_init_with_allocator(resolved[i].field_keep_float, allocator);
            continue;
        }
        resolved[i].which_action = 2;
        VolvoxaiV1WeightQuantization* policy = allocator->allocate(allocator->context, sizeof(*policy));
        if (!policy) return 0;
        volvoxai_v1_weight_quantization_init_with_allocator(policy, allocator);
        resolved[i].field_weights = policy;
        policy->field_algorithm = (VolvoxaiV1WeightAlgorithm)node->algorithm;
        policy->field_format = allocator->allocate(allocator->context, sizeof(*policy->field_format));
        if (!policy->field_format) return 0;
        volvoxai_v1_weight_format_init_with_allocator(policy->field_format, allocator);
        policy->field_format->field_bits = node->bits;
        policy->field_format->field_group_size = node->group_size;
        size_t index = 0;
        while (index < response->field_quantized.len &&
            (counts[index].field_format->field_bits != node->bits ||
             counts[index].field_format->field_group_size != node->group_size)) index++;
        if (index == response->field_quantized.len) {
            counts[index].field_format = allocator->allocate(allocator->context, sizeof(*counts[index].field_format));
            if (!counts[index].field_format) return 0;
            volvoxai_v1_weight_format_init_with_allocator(counts[index].field_format, allocator);
            counts[index].field_format->field_bits = node->bits;
            counts[index].field_format->field_group_size = node->group_size;
            response->field_quantized.len++;
        }
        counts[index].field_nodes++;
    }
    return 1;
}

static int vx_api_quantization_info_child(const SynurangLiteAllocator* allocator,
        VolvoxaiV1QuantizationPlanInfo** slot, VxQuantizationPlan* plan) {
    *slot = allocator->allocate(allocator->context, sizeof(**slot));
    if (!*slot) return 0;
    volvoxai_v1_quantization_plan_info_init_with_allocator(*slot, allocator);
    return vx_api_quantization_info(*slot, plan);
}

static int vx_api_create_quantization_plan(
        const VolvoxaiV1CreateQuantizationPlanRequest* request,
        VolvoxaiV1QuantizationPlanHandle* response, void* user_data) {
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(user_data);
    VxPtqAuthoringConfig config = VX_PTQ_AUTHORING_CONFIG_INIT;
    VxQuantizationPlanOptions options = VX_QUANTIZATION_PLAN_OPTIONS_INIT;
    VxQuantizationPlan* plan = NULL;
    VxReport report = VX_REPORT_INIT;
    int result = -1;
#if defined(__wasm__)
    if (request->which_source == 1)
        return vx_api_report_fail(allocator, &response->field_report,
            VX_STATUS_TRANSPORT_UNSUPPORTED, VX_STAGE_PTQ_CREATE,
            VX_CODE_TRANSPORT_UNSUPPORTED, "WASM quantization requires graph and weight bytes") ? 0 : -1;
#endif
    if (!vx_api_quantization_config(request->field_config, &scratch, &config, &options.packed_weights)) {
        result = vx_api_report_fail(allocator, &response->field_report,
            VX_STATUS_INVALID_ARGUMENT, VX_STAGE_PTQ_CREATE, VX_CODE_INVALID_PTQ_SPEC,
            "invalid quantization policy: static requires per-channel I8 RTN or Conv I2/I3/I4 RTN/MSE/IMPORTANCE; weight-only requires "
            "group-32/64/128/256 I2/I3/I4/I8; rules need typed nonempty selectors and actions") ? 0 : -1;
        goto done;
    }
    options.config = &config;
    const VolvoxaiV1CalibrationOptions* calibration = request->field_calibration;
    if (calibration) {
        if (calibration->field_cpu_threads > INT32_MAX ||
            vx_api_ptq_bytes_has_nul(&calibration->field_backend) ||
            !vx_api_quantization_names_valid(calibration->field_profile_names.data,
                calibration->field_profile_names.len)) goto invalid_source;
        options.cpu_threads = (int32_t)calibration->field_cpu_threads;
        if (calibration->field_backend.len) {
            options.calibration_backend = vx_api_scratch_cstr(&scratch, &calibration->field_backend);
            if (!options.calibration_backend) goto done;
        }
        options.profile_names = vx_api_scratch_cstr_array(&scratch,
            calibration->field_profile_names.data, sizeof(SynurangLiteBytes),
            calibration->field_profile_names.len);
        options.profile_count = calibration->field_profile_names.len;
    }
    if (request->which_source == 2 && request->field_bytes) {
        const VolvoxaiV1ModelBytes* bytes = request->field_bytes;
        if (bytes->field_weight_shards.len != 1) goto invalid_shards;
        if (!bytes->field_graph.len || !bytes->field_weight_shards.data[0].len) goto invalid_source;
        VxPtqWeightBlob* blob = vx_api_scratch_alloc(&scratch, sizeof(*blob));
        if (!blob) goto done;
        blob->bytes = bytes->field_weight_shards.data[0].data;
        blob->byte_count = bytes->field_weight_shards.data[0].len;
        options.source_graph = (const char*)bytes->field_graph.data;
        options.source_graph_size = bytes->field_graph.len;
        options.weights = blob;
        options.weight_count = 1;
    } else if (request->which_source == 1 && request->field_files) {
        const VolvoxaiV1ModelFiles* files = request->field_files;
        if (files->field_weight_paths.len != 1) goto invalid_shards;
        if (!files->field_graph_path.len || vx_api_ptq_bytes_has_nul(&files->field_graph_path) ||
            !vx_api_quantization_names_valid(files->field_weight_paths.data, files->field_weight_paths.len))
            goto invalid_source;
        options.source_graph_path = vx_api_scratch_cstr(&scratch, &files->field_graph_path);
        options.weight_paths = vx_api_scratch_cstr_array(&scratch, files->field_weight_paths.data,
            sizeof(SynurangLiteBytes), files->field_weight_paths.len);
        options.weight_path_count = 1;
    } else goto invalid_source;
    if (vx_api_scratch_failed(&scratch)) goto done;
    VxStatus status = vx_quantization_plan_create(&options, &plan, &report);
    if (!vx_api_report_attach(allocator, &response->field_report, &report)) goto done;
    result = 0;
    if (status != VX_STATUS_OK) goto done;
    if (!vx_api_quantization_info_child(allocator, &response->field_info, plan)) {
        result = -1;
        goto done;
    }
    response->field_quantization_plan_id = vx_api_handle_insert_with_lineage(user_data,
        VX_API_HANDLE_QUANTIZATION_PLAN, plan, vx_api_retain_quantization_plan,
        vx_api_release_quantization_plan_handle, NULL);
    if (response->field_quantization_plan_id) plan = NULL;
    else result = vx_api_report_fail(allocator, &response->field_report,
        VX_STATUS_OUT_OF_MEMORY, VX_STAGE_PTQ_CREATE, VX_CODE_OUT_OF_MEMORY,
        "quantization plan handle allocation failed") ? 0 : -1;
    goto done;
invalid_source:
    result = vx_api_report_fail(allocator, &response->field_report,
        VX_STATUS_INVALID_ARGUMENT, VX_STAGE_PTQ_CREATE, VX_CODE_INVALID_PTQ_SPEC,
        "invalid quantization source or calibration options") ? 0 : -1;
    goto done;
invalid_shards:
    result = vx_api_report_fail(allocator, &response->field_report,
        VX_STATUS_INVALID_ARGUMENT, VX_STAGE_PTQ_CREATE, VX_CODE_PTQ_SINGLE_SHARD_REQUIRED,
        "quantization requires exactly one source SafeTensors shard") ? 0 : -1;
done:
    vx_quantization_plan_release(plan);
    vx_api_scratch_release(&scratch);
    return result;
}

static int vx_api_release_quantization_plan(const VolvoxaiV1QuantizationPlanRef* request,
        VolvoxaiV1OperationReport* response, void* user_data) {
    (void)vx_api_handle_remove(user_data, VX_API_HANDLE_QUANTIZATION_PLAN,
                               request->field_quantization_plan_id);
    response->field_stage = VOLVOXAI_V1_OPERATION_STAGE_PTQ_CLOSE;
    return 0;
}

static int vx_api_calibrate_quantization_plan(
        const VolvoxaiV1CalibrateQuantizationPlanRequest* request,
        VolvoxaiV1CalibrationProgress* response, void* user_data) {
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(user_data);
    VxPTQCalibrationBatch batch = VX_PTQ_CALIBRATION_BATCH_INIT;
    VxPTQPlanInfo info = VX_PTQ_PLAN_INFO_INIT;
    VxReport report = VX_REPORT_INIT;
    int result = -1;
    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_QUANTIZATION_PLAN,
                               request->field_quantization_plan_id, &lease))
        return vx_api_report_fail(allocator, &response->field_report,
            VX_STATUS_HANDLE_DISPOSED, VX_STAGE_PTQ_CALIBRATE, VX_CODE_HANDLE_DISPOSED,
            "unknown or released quantization plan") ? 0 : -1;
    VxPTQPlan* plan = vx_quantization_plan_static(lease.pointer);
    if (!plan || vx_api_ptq_bytes_has_nul(&request->field_profile_name) ||
        vx_api_ptq_bytes_has_nul(&request->field_sample_name)) {
        result = vx_api_report_fail(allocator, &response->field_report,
            VX_STATUS_INVALID_ARGUMENT, VX_STAGE_PTQ_CALIBRATE, VX_CODE_INVALID_PTQ_SPEC,
            !plan ? "the resolved quantization policy does not require calibration"
                  : "calibration names must be NUL-free") ? 0 : -1;
        goto done;
    }
    batch.profile_name = request->field_profile_name.len
        ? vx_api_scratch_cstr(&scratch, &request->field_profile_name) : "default";
    batch.sample_name = vx_api_scratch_cstr(&scratch, &request->field_sample_name);
    batch.sample_count = request->field_sample_count;
    if (request->field_inputs.len) {
        VxTensorBinding* bindings = vx_api_scratch_alloc(&scratch,
            request->field_inputs.len * sizeof(*bindings));
        if (!bindings) goto done;
        for (size_t i = 0; i < request->field_inputs.len; i++) {
            VxStatus status = vx_api_binding_from_tensor(&scratch, &bindings[i],
                &request->field_inputs.data[i]);
            if (status != VX_STATUS_OK) {
                result = vx_api_report_binding_fail(allocator, &response->field_report,
                    status, VX_STAGE_PTQ_CALIBRATE, "calibration input could not be bound",
                    &request->field_inputs.data[i], i) ? 0 : -1;
                goto done;
            }
        }
        batch.inputs = bindings;
        batch.input_count = request->field_inputs.len;
    }
    if (vx_api_scratch_failed(&scratch)) goto done;
    /* Prepare the complete compact acknowledgement before any observation is
     * committed. Successful calibration performs no fallible response work. */
    if (!vx_api_report_ok(allocator, &response->field_report, VX_STAGE_PTQ_CALIBRATE)) goto done;
    if (vx_ptq_plan_calibrate(plan, &batch, &info, &report) == VX_STATUS_OK) {
        response->field_batches = info.calibration_batches;
        response->field_samples = info.calibration_samples;
        response->field_ready = info.coverage_complete != 0;
        result = 0;
    } else result = vx_api_report_from_native(response->field_report, &report) ? 0 : -1;
done:
    vx_api_scratch_release(&scratch);
    vx_api_handle_lease_release(&lease);
    return result;
}

static int vx_api_inspect_quantization_plan(const VolvoxaiV1QuantizationPlanRef* request,
        VolvoxaiV1QuantizationPlanResponse* response, void* user_data) {
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_QUANTIZATION_PLAN,
                               request->field_quantization_plan_id, &lease))
        return vx_api_report_fail(response->_allocator, &response->field_report,
            VX_STATUS_HANDLE_DISPOSED, VX_STAGE_PTQ_INSPECT, VX_CODE_HANDLE_DISPOSED,
            "unknown or released quantization plan") ? 0 : -1;
    int result = vx_api_quantization_info_child(response->_allocator, &response->field_info, lease.pointer) &&
        vx_api_report_ok(response->_allocator, &response->field_report, VX_STAGE_PTQ_INSPECT) ? 0 : -1;
    vx_api_handle_lease_release(&lease);
    return result;
}

static int vx_api_export_quantized_model(const VolvoxaiV1ExportQuantizedModelRequest* request,
        VolvoxaiV1QuantizedModelInfo* response, void* user_data) {
    const SynurangLiteAllocator* allocator = response->_allocator;
    const VolvoxaiV1ModelFiles* files = request->field_output_files;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(user_data);
    VxPTQPackageOptions options = VX_PTQ_PACKAGE_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    unsigned char *graph = NULL, *weights = NULL;
    size_t graph_size = 0, weights_size = 0;
    int result = -1;
#if defined(__wasm__)
    if (files)
        return vx_api_report_fail(allocator, &response->field_report,
            VX_STATUS_TRANSPORT_UNSUPPORTED, VX_STAGE_PTQ_WRITE, VX_CODE_TRANSPORT_UNSUPPORTED,
            "WASM returns package bytes; omit output_files") ? 0 : -1;
#endif
    if (files && (!files->field_graph_path.len || files->field_weight_paths.len != 1 ||
        vx_api_ptq_bytes_has_nul(&files->field_graph_path) ||
        !vx_api_quantization_names_valid(files->field_weight_paths.data, files->field_weight_paths.len)))
        return vx_api_report_fail(allocator, &response->field_report,
            VX_STATUS_INVALID_ARGUMENT, VX_STAGE_PTQ_WRITE, VX_CODE_INVALID_PTQ_OUTPUT_PATHS,
            "output_files requires graph.json and one sibling safetensors path") ? 0 : -1;
    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_QUANTIZATION_PLAN,
                               request->field_quantization_plan_id, &lease))
        return vx_api_report_fail(allocator, &response->field_report,
            VX_STATUS_HANDLE_DISPOSED, VX_STAGE_PTQ_WRITE, VX_CODE_HANDLE_DISPOSED,
            "unknown or released quantization plan") ? 0 : -1;
    if (files) {
        options.output_graph_path = vx_api_scratch_cstr(&scratch, &files->field_graph_path);
        options.output_weights_path = vx_api_scratch_cstr(&scratch, &files->field_weight_paths.data[0]);
    } else {
        options.graph_bytes = &graph; options.graph_size = &graph_size;
        options.weights_bytes = &weights; options.weights_size = &weights_size;
    }
    if (vx_api_scratch_failed(&scratch)) goto done;
    if (vx_quantization_plan_write_package(lease.pointer, &options, &report) == VX_STATUS_OK) {
        if (files) {
            response->which_artifact = 1;
            response->field_files = allocator->allocate(allocator->context, sizeof(*response->field_files));
            if (!response->field_files) goto done;
            volvoxai_v1_model_files_init_with_allocator(response->field_files, allocator);
            VolvoxaiV1ModelFiles* output = response->field_files;
            output->field_weight_paths.data = allocator->allocate(allocator->context, sizeof(SynurangLiteBytes));
            if (!output->field_weight_paths.data) goto done;
            memset(output->field_weight_paths.data, 0, sizeof(SynurangLiteBytes));
            output->field_weight_paths.len = output->field_weight_paths.cap = 1;
            if (synurang_lite_bytes_assign(allocator, &output->field_graph_path,
                    options.output_graph_path, strlen(options.output_graph_path)) ||
                synurang_lite_bytes_assign(allocator, &output->field_weight_paths.data[0],
                    options.output_weights_path, strlen(options.output_weights_path))) goto done;
        } else {
            response->which_artifact = 2;
            response->field_bytes = allocator->allocate(allocator->context, sizeof(*response->field_bytes));
            if (!response->field_bytes) goto done;
            volvoxai_v1_model_bytes_init_with_allocator(response->field_bytes, allocator);
            VolvoxaiV1ModelBytes* output = response->field_bytes;
            output->field_weight_shards.data = allocator->allocate(allocator->context, sizeof(SynurangLiteBytes));
            if (!output->field_weight_shards.data) goto done;
            memset(output->field_weight_shards.data, 0, sizeof(SynurangLiteBytes));
            output->field_weight_shards.len = output->field_weight_shards.cap = 1;
            if (synurang_lite_bytes_assign(allocator, &output->field_graph, graph, graph_size) ||
                synurang_lite_bytes_assign(allocator, &output->field_weight_shards.data[0], weights, weights_size)) goto done;
        }
        if (!vx_api_quantization_info_child(allocator, &response->field_plan, lease.pointer)) goto done;
    }
    result = vx_api_report_attach(allocator, &response->field_report, &report) ? 0 : -1;
done:
    free(graph); free(weights);
    vx_api_scratch_release(&scratch);
    vx_api_handle_lease_release(&lease);
    return result;
}

#include "vx_api_weight_conversion.inc"

VX_API_UNARY(vx_api_quantize_weight, VolvoxaiV1QuantizeWeightRequest, VolvoxaiV1QuantizeWeightResponse,
    volvoxai_v1_quantize_weight_response, vx_quantization_quantize_weight_respond)
VX_API_UNARY(vx_api_dequantize_weight, VolvoxaiV1DequantizeWeightRequest, VolvoxaiV1InitializedTensor,
    volvoxai_v1_initialized_tensor, vx_quantization_dequantize_weight_respond)
VX_API_UNARY(vx_api_create_quantization_plan, VolvoxaiV1CreateQuantizationPlanRequest, VolvoxaiV1QuantizationPlanHandle,
    volvoxai_v1_quantization_plan_handle, vx_quantization_create_quantization_plan_respond)
VX_API_UNARY(vx_api_release_quantization_plan, VolvoxaiV1QuantizationPlanRef, VolvoxaiV1OperationReport,
    volvoxai_v1_operation_report, vx_quantization_release_quantization_plan_respond)
VX_API_UNARY(vx_api_calibrate_quantization_plan, VolvoxaiV1CalibrateQuantizationPlanRequest, VolvoxaiV1CalibrationProgress,
    volvoxai_v1_calibration_progress, vx_quantization_calibrate_quantization_plan_respond)
VX_API_UNARY(vx_api_inspect_quantization_plan, VolvoxaiV1QuantizationPlanRef, VolvoxaiV1QuantizationPlanResponse,
    volvoxai_v1_quantization_plan_response, vx_quantization_inspect_quantization_plan_respond)
VX_API_UNARY(vx_api_export_quantized_model, VolvoxaiV1ExportQuantizedModelRequest, VolvoxaiV1QuantizedModelInfo,
    volvoxai_v1_quantized_model_info, vx_quantization_export_quantized_model_respond)

int vx_api_install_quantization_handlers(SynurangInstance* instance, VxApiRegistry* registry) {
    VxQuantizationServiceHandlers handlers;
    memset(&handlers, 0, sizeof(handlers));
    handlers.quantize_weight.message = vx_api_quantize_weight_call;
    handlers.dequantize_weight.message = vx_api_dequantize_weight_call;
    handlers.create_quantization_plan.message = vx_api_create_quantization_plan_call;
    handlers.release_quantization_plan.message = vx_api_release_quantization_plan_call;
    handlers.calibrate_quantization_plan.message = vx_api_calibrate_quantization_plan_call;
    handlers.inspect_quantization_plan.message = vx_api_inspect_quantization_plan_call;
    handlers.export_quantized_model.message = vx_api_export_quantized_model_call;
    return vx_quantization_register(instance, &handlers, registry);
}
