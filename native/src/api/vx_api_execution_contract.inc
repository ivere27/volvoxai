/* JSON is the package representation of the typed proto execution contract.
 * These projections preserve every field; runtime validation owns semantics. */
static int vx_contract_string_read(const SynurangLiteAllocator* allocator,
    SynurangLiteBytes* out, const cJSON* object, const char* name) {
    const cJSON* value = cJSON_GetObjectItemCaseSensitive(object, name);
    if (!value) return 1;
    return cJSON_IsString(value) && synurang_lite_bytes_assign(allocator, out,
        value->valuestring, strlen(value->valuestring)) == SYNURANG_LITE_OK;
}

static int vx_contract_enum_read(const cJSON* value, const char* const* names,
    size_t count) {
    if (cJSON_IsNumber(value)) return value->valueint;
    if (cJSON_IsString(value)) {
        for (size_t i = 0; i < count; ++i)
            if (!strcmp(value->valuestring, names[i])) return (int)i + 1;
    }
    return 0;
}

int vx_api_execution_contract_from_json(const SynurangLiteAllocator* allocator,
    VolvoxaiV1ModelExecutionContract** output, const cJSON* root) {
    if (!cJSON_IsObject(root)) return 0;
    VolvoxaiV1ModelExecutionContract* out = allocator->allocate(allocator->context, sizeof(*out));
    if (!out) return 0;
    volvoxai_v1_model_execution_contract_init_with_allocator(out, allocator);
    *output = out;
#define VX_CONTRACT_READ_STRING(json, object, name) \
    if (!vx_contract_string_read(allocator, &(object)->field_##name, json, #name)) return 0
#define VX_CONTRACT_READ_STRINGS(json, object, type, name) do { \
    const cJSON* array = cJSON_GetObjectItemCaseSensitive(json, #name); \
    if (array && !cJSON_IsArray(array)) return 0; \
    for (const cJSON* item = array ? array->child : NULL; item; item = item->next) { \
        SynurangLiteBytes* value = volvoxai_v1_##type##_add_##name(object); \
        if (!value || !cJSON_IsString(item) || synurang_lite_bytes_assign(allocator, \
            value, item->valuestring, strlen(item->valuestring)) != SYNURANG_LITE_OK) return 0; \
    } \
} while (0)
#define VX_CONTRACT_NEW(slot, type, init) do { \
    slot = allocator->allocate(allocator->context, sizeof(type)); \
    if (!slot) return 0; \
    volvoxai_v1_##init##_init_with_allocator(slot, allocator); \
} while (0)
#define VX_CONTRACT_READ_INT(json, object, name) do { \
    const cJSON* value = cJSON_GetObjectItemCaseSensitive(json, #name); \
    if (value && !cJSON_IsNumber(value)) return 0; \
    (object)->field_##name = value ? (uint32_t)value->valuedouble : 0; \
} while (0)
    const cJSON* entries = cJSON_GetObjectItemCaseSensitive(root, "entrypoints");
    if (!cJSON_IsArray(entries)) return 0;
    for (const cJSON* item = entries->child; item; item = item->next) {
        VolvoxaiV1ExecutionEntrypoint* entry = volvoxai_v1_model_execution_contract_add_entrypoints(out);
        if (!entry || !cJSON_IsObject(item)) return 0;
        VX_CONTRACT_READ_STRING(item, entry, name);
        static const char* const effects[] = {"EXECUTION_EFFECT_EVALUATE",
            "EXECUTION_EFFECT_INITIALIZE", "EXECUTION_EFFECT_TRANSITION"};
        entry->field_effect = vx_contract_enum_read(
            cJSON_GetObjectItemCaseSensitive(item, "effect"), effects, 3);
        if (!entry->field_effect) return 0;
        VX_CONTRACT_READ_STRINGS(item, entry, execution_entrypoint, inputs);
        VX_CONTRACT_READ_STRINGS(item, entry, execution_entrypoint, outputs);
        const cJSON* ranges = cJSON_GetObjectItemCaseSensitive(item, "allowed_ranges");
        if (!cJSON_IsArray(ranges)) return 0;
        for (const cJSON* range = ranges->child; range; range = range->next) {
            VolvoxaiV1ExecutionRangeKind* value = volvoxai_v1_execution_entrypoint_add_allowed_ranges(entry);
            static const char* const range_names[] = {"EXECUTION_RANGE_KIND_FULL",
                "EXECUTION_RANGE_KIND_PREFIX_ROWS", "EXECUTION_RANGE_KIND_SLOT_ROWS",
                "EXECUTION_RANGE_KIND_DEPENDENCIES"};
            if (!value) return 0;
            *value = (VolvoxaiV1ExecutionRangeKind)vx_contract_enum_read(range, range_names, 4);
            if (!*value) return 0;
        }
    }
    VX_CONTRACT_READ_STRINGS(root, out, model_execution_contract, invariant_inputs);
    const cJSON* state = cJSON_GetObjectItemCaseSensitive(root, "stateless");
    if (state) {
        out->which_state = 3;
        VX_CONTRACT_NEW(out->field_stateless, VolvoxaiV1Empty, empty);
    } else if ((state = cJSON_GetObjectItemCaseSensitive(root, "incremental"))) {
        out->which_state = 4;
        VX_CONTRACT_NEW(out->field_incremental, VolvoxaiV1IncrementalStateContract, incremental_state_contract);
        VX_CONTRACT_READ_STRING(state, out->field_incremental, active_lengths_input);
        VX_CONTRACT_READ_STRING(state, out->field_incremental, position_input);
        VX_CONTRACT_READ_INT(state, out->field_incremental, sequence_axis);
        VX_CONTRACT_READ_STRINGS(state, out->field_incremental, incremental_state_contract, update_inputs);
    } else if ((state = cJSON_GetObjectItemCaseSensitive(root, "tensor_feedback"))) {
        out->which_state = 5;
        VX_CONTRACT_NEW(out->field_tensor_feedback, VolvoxaiV1TensorFeedbackStateContract, tensor_feedback_state_contract);
        VX_CONTRACT_READ_STRING(state, out->field_tensor_feedback, active_lengths_input);
        VX_CONTRACT_READ_STRING(state, out->field_tensor_feedback, position_input);
        const cJSON* bindings = cJSON_GetObjectItemCaseSensitive(state, "bindings");
        if (bindings && !cJSON_IsArray(bindings)) return 0;
        for (const cJSON* item = bindings ? bindings->child : NULL; item; item = item->next) {
            VolvoxaiV1TensorFeedback* binding = volvoxai_v1_tensor_feedback_state_contract_add_bindings(out->field_tensor_feedback);
            if (!binding) return 0;
            VX_CONTRACT_READ_STRING(item, binding, input_name);
            VX_CONTRACT_READ_STRING(item, binding, output_name);
        }
    } else return 0;
    const cJSON* generation = cJSON_GetObjectItemCaseSensitive(root, "generation");
    if (generation) {
        VX_CONTRACT_NEW(out->field_generation, VolvoxaiV1GenerationContract, generation_contract);
        VX_CONTRACT_READ_STRING(generation, out->field_generation, initialize_entrypoint);
        VX_CONTRACT_READ_STRING(generation, out->field_generation, transition_entrypoint);
        VX_CONTRACT_READ_STRING(generation, out->field_generation, token_input);
        VX_CONTRACT_READ_STRING(generation, out->field_generation, token_output);
        VX_CONTRACT_READ_STRING(generation, out->field_generation, keep_input);
        VX_CONTRACT_READ_STRING(generation, out->field_generation, logits_output);
        VX_CONTRACT_READ_STRING(generation, out->field_generation, output_name);
        VX_CONTRACT_READ_INT(generation, out->field_generation, logits_axis);
        VX_CONTRACT_READ_INT(generation, out->field_generation, id_domain_size);
        VX_CONTRACT_READ_INT(generation, out->field_generation, sequence_axis);
        const cJSON* prefix = cJSON_GetObjectItemCaseSensitive(generation, "full_prefix");
        if (prefix) {
            VX_CONTRACT_NEW(out->field_generation->field_full_prefix,
                VolvoxaiV1FullPrefixGenerationContract, full_prefix_generation_contract);
            VolvoxaiV1FullPrefixGenerationContract* target = out->field_generation->field_full_prefix;
            VX_CONTRACT_READ_STRING(prefix, target, position_input);
            VX_CONTRACT_READ_STRING(prefix, target, sequence_input);
            VX_CONTRACT_READ_STRING(prefix, target, chunk_mask_input);
            VX_CONTRACT_READ_INT(prefix, target, chunk_size);
            VX_CONTRACT_READ_STRINGS(prefix, target, full_prefix_generation_contract, zero_append_inputs);
            VX_CONTRACT_READ_STRING(prefix, target, multimodal_position_input);
        }
        const cJSON* controls = cJSON_GetObjectItemCaseSensitive(generation, "incremental_controls");
        if (controls) {
            VX_CONTRACT_NEW(out->field_generation->field_incremental_controls,
                VolvoxaiV1IncrementalGenerationContract, incremental_generation_contract);
            VolvoxaiV1IncrementalGenerationContract* target = out->field_generation->field_incremental_controls;
            VX_CONTRACT_READ_STRING(controls, target, position_input);
            VX_CONTRACT_READ_STRING(controls, target, sequence_input);
            VX_CONTRACT_READ_STRING(controls, target, chunk_mask_input);
            VX_CONTRACT_READ_INT(controls, target, chunk_size);
            VX_CONTRACT_READ_STRINGS(controls, target, incremental_generation_contract, zero_append_inputs);
            VX_CONTRACT_READ_STRING(controls, target, multimodal_position_input);
        }
    }
#undef VX_CONTRACT_READ_STRING
#undef VX_CONTRACT_READ_STRINGS
#undef VX_CONTRACT_READ_INT
#undef VX_CONTRACT_NEW
    return 1;
}

static int vx_contract_string_write(cJSON* object, const char* name,
    const SynurangLiteBytes* bytes) {
    if (bytes->len && (!bytes->data || memchr(bytes->data, 0, bytes->len))) return 0;
    if (bytes->len == SIZE_MAX) return 0;
    char* text = malloc(bytes->len + 1);
    if (!text) return 0;
    if (bytes->len) memcpy(text, bytes->data, bytes->len);
    text[bytes->len] = 0;
    int ok = cJSON_AddStringToObject(object, name, text) != NULL;
    free(text);
    return ok;
}

cJSON* vx_api_execution_contract_json(const VolvoxaiV1ModelExecutionContract* source) {
    if (!source) return NULL;
    cJSON* root = cJSON_CreateObject();
    if (!root) return NULL;
#define VX_CONTRACT_WRITE_STRING(json, object, name) \
    if (!vx_contract_string_write(json, #name, &(object)->field_##name)) goto failed
#define VX_CONTRACT_WRITE_STRINGS(json, object, name) do { \
    cJSON* array = cJSON_AddArrayToObject(json, #name); \
    if (!array) goto failed; \
    for (size_t i = 0; i < (object)->field_##name.len; ++i) { \
        cJSON* member = cJSON_CreateObject(); \
        if (!member) goto failed; \
        if (!vx_contract_string_write(member, "value", &(object)->field_##name.data[i])) { \
            cJSON_Delete(member); goto failed; \
        } \
        cJSON* value = cJSON_DetachItemFromObjectCaseSensitive(member, "value"); \
        cJSON_Delete(member); \
        if (!cJSON_AddItemToArray(array, value)) { cJSON_Delete(value); goto failed; } \
    } \
} while (0)
#define VX_CONTRACT_WRITE_INT(json, object, name) \
    if (!cJSON_AddNumberToObject(json, #name, (double)(object)->field_##name)) goto failed
    cJSON* entries = cJSON_AddArrayToObject(root, "entrypoints");
    if (!entries) goto failed;
    for (size_t e = 0; e < source->field_entrypoints.len; ++e) {
        const VolvoxaiV1ExecutionEntrypoint* entry = &source->field_entrypoints.data[e];
        cJSON* item = cJSON_CreateObject();
        if (!item) goto failed;
        if (!cJSON_AddItemToArray(entries, item)) { cJSON_Delete(item); goto failed; }
        VX_CONTRACT_WRITE_STRING(item, entry, name);
        VX_CONTRACT_WRITE_INT(item, entry, effect);
        VX_CONTRACT_WRITE_STRINGS(item, entry, inputs);
        VX_CONTRACT_WRITE_STRINGS(item, entry, outputs);
        cJSON* ranges = cJSON_AddArrayToObject(item, "allowed_ranges");
        if (!ranges) goto failed;
        for (size_t r = 0; r < entry->field_allowed_ranges.len; ++r) {
            cJSON* value = cJSON_CreateNumber(entry->field_allowed_ranges.data[r]);
            if (!value || !cJSON_AddItemToArray(ranges, value)) { cJSON_Delete(value); goto failed; }
        }
    }
    VX_CONTRACT_WRITE_STRINGS(root, source, invariant_inputs);
    cJSON* state;
    if (source->which_state == 3 && source->field_stateless) {
        if (!cJSON_AddObjectToObject(root, "stateless")) goto failed;
    } else if (source->which_state == 4 && source->field_incremental) {
        state = cJSON_AddObjectToObject(root, "incremental");
        if (!state) goto failed;
        VX_CONTRACT_WRITE_STRING(state, source->field_incremental, active_lengths_input);
        VX_CONTRACT_WRITE_STRING(state, source->field_incremental, position_input);
        VX_CONTRACT_WRITE_INT(state, source->field_incremental, sequence_axis);
        VX_CONTRACT_WRITE_STRINGS(state, source->field_incremental, update_inputs);
    } else if (source->which_state == 5 && source->field_tensor_feedback) {
        state = cJSON_AddObjectToObject(root, "tensor_feedback");
        if (!state) goto failed;
        VX_CONTRACT_WRITE_STRING(state, source->field_tensor_feedback, active_lengths_input);
        VX_CONTRACT_WRITE_STRING(state, source->field_tensor_feedback, position_input);
        cJSON* bindings = cJSON_AddArrayToObject(state, "bindings");
        if (!bindings) goto failed;
        for (size_t b = 0; b < source->field_tensor_feedback->field_bindings.len; ++b) {
            const VolvoxaiV1TensorFeedback* binding = &source->field_tensor_feedback->field_bindings.data[b];
            cJSON* item = cJSON_CreateObject();
            if (!item) goto failed;
            if (!cJSON_AddItemToArray(bindings, item)) { cJSON_Delete(item); goto failed; }
            VX_CONTRACT_WRITE_STRING(item, binding, input_name);
            VX_CONTRACT_WRITE_STRING(item, binding, output_name);
        }
    } else goto failed;
    if (source->field_generation) {
        cJSON* generation = cJSON_AddObjectToObject(root, "generation");
        if (!generation) goto failed;
        VX_CONTRACT_WRITE_STRING(generation, source->field_generation, initialize_entrypoint);
        VX_CONTRACT_WRITE_STRING(generation, source->field_generation, transition_entrypoint);
        VX_CONTRACT_WRITE_STRING(generation, source->field_generation, token_input);
        VX_CONTRACT_WRITE_STRING(generation, source->field_generation, token_output);
        VX_CONTRACT_WRITE_STRING(generation, source->field_generation, keep_input);
        VX_CONTRACT_WRITE_STRING(generation, source->field_generation, logits_output);
        VX_CONTRACT_WRITE_STRING(generation, source->field_generation, output_name);
        VX_CONTRACT_WRITE_INT(generation, source->field_generation, logits_axis);
        VX_CONTRACT_WRITE_INT(generation, source->field_generation, id_domain_size);
        VX_CONTRACT_WRITE_INT(generation, source->field_generation, sequence_axis);
        if (source->field_generation->field_full_prefix) {
            const VolvoxaiV1FullPrefixGenerationContract* controls = source->field_generation->field_full_prefix;
            cJSON* object = cJSON_AddObjectToObject(generation, "full_prefix");
            if (!object) goto failed;
            VX_CONTRACT_WRITE_STRING(object, controls, position_input);
            if (controls->field_sequence_input.len) {
                VX_CONTRACT_WRITE_STRING(object, controls, sequence_input);
            }
            if (controls->field_chunk_mask_input.len) {
                VX_CONTRACT_WRITE_STRING(object, controls, chunk_mask_input);
            }
            if (controls->field_chunk_mask_input.len || controls->field_chunk_size) {
                VX_CONTRACT_WRITE_INT(object, controls, chunk_size);
            }
            VX_CONTRACT_WRITE_STRINGS(object, controls, zero_append_inputs);
            if (controls->field_multimodal_position_input.len) {
                VX_CONTRACT_WRITE_STRING(object, controls, multimodal_position_input);
            }
        }
        if (source->field_generation->field_incremental_controls) {
            const VolvoxaiV1IncrementalGenerationContract* controls = source->field_generation->field_incremental_controls;
            cJSON* object = cJSON_AddObjectToObject(generation, "incremental_controls");
            if (!object) goto failed;
            VX_CONTRACT_WRITE_STRING(object, controls, position_input);
            if (controls->field_sequence_input.len) {
                VX_CONTRACT_WRITE_STRING(object, controls, sequence_input);
            }
            if (controls->field_chunk_mask_input.len) {
                VX_CONTRACT_WRITE_STRING(object, controls, chunk_mask_input);
            }
            if (controls->field_chunk_mask_input.len || controls->field_chunk_size) {
                VX_CONTRACT_WRITE_INT(object, controls, chunk_size);
            }
            VX_CONTRACT_WRITE_STRINGS(object, controls, zero_append_inputs);
            if (controls->field_multimodal_position_input.len) {
                VX_CONTRACT_WRITE_STRING(object, controls, multimodal_position_input);
            }
        }
    }
#undef VX_CONTRACT_WRITE_STRING
#undef VX_CONTRACT_WRITE_STRINGS
#undef VX_CONTRACT_WRITE_INT
    return root;
failed:
    cJSON_Delete(root);
    return NULL;
}
