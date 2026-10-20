/* The package carries the proto ModelExecutionContract as a JSON projection.
 * Model owns this immutable document; all runtime roles come from it. */
static const cJSON* vx_execution_document(const VxModel* model) {
    return model ? cJSON_GetObjectItemCaseSensitive(model->logical_graph, "execution") : NULL;
}

const void* vx_model_execution_contract_document(const VxModel* model) {
    return vx_execution_document(model);
}

static const char* vx_execution_string(const cJSON* object, const char* key) {
    const cJSON* value = cJSON_GetObjectItemCaseSensitive(object, key);
    return cJSON_IsString(value) && value->valuestring && value->valuestring[0]
        ? value->valuestring : NULL;
}

static int vx_execution_enum(const cJSON* value, const char* prefix,
                             const char* const* names, size_t count) {
    if (cJSON_IsNumber(value) && isfinite(value->valuedouble) &&
        value->valuedouble >= 1 && value->valuedouble <= (double)count &&
        value->valuedouble == trunc(value->valuedouble)) return value->valueint;
    if (!cJSON_IsString(value) || !value->valuestring) return 0;
    size_t prefix_bytes = strlen(prefix);
    if (strncmp(value->valuestring, prefix, prefix_bytes)) return 0;
    for (size_t i = 0; i < count; i++)
        if (!strcmp(value->valuestring + prefix_bytes, names[i])) return (int)i + 1;
    return 0;
}

static int vx_execution_effect(const cJSON* entry) {
    static const char* const names[] = {"EVALUATE", "INITIALIZE", "TRANSITION"};
    return vx_execution_enum(cJSON_GetObjectItemCaseSensitive(entry, "effect"),
        "EXECUTION_EFFECT_", names, 3);
}

static int vx_execution_range(const cJSON* value) {
    static const char* const names[] = {"FULL", "PREFIX_ROWS", "SLOT_ROWS", "DEPENDENCIES"};
    return vx_execution_enum(value, "EXECUTION_RANGE_KIND_", names, 4);
}

static const cJSON* vx_execution_entry(const VxModel* model, const char* name) {
    const cJSON* entries = cJSON_GetObjectItemCaseSensitive(vx_execution_document(model), "entrypoints");
    for (const cJSON* entry = entries ? entries->child : NULL; entry; entry = entry->next) {
        const char* candidate = vx_execution_string(entry, "name");
        if (name && candidate && !strcmp(name, candidate)) return entry;
    }
    return NULL;
}

static int vx_execution_names_contains(const cJSON* names, const char* name) {
    for (const cJSON* item = names ? names->child : NULL; item; item = item->next)
        if (cJSON_IsString(item) && item->valuestring && name && !strcmp(item->valuestring, name)) return 1;
    return 0;
}

static int vx_execution_declared(const VxDeclaredTensor* tensors, size_t count, const char* name) {
    for (size_t i = 0; name && i < count; i++)
        if (!strcmp(tensors[i].name, name)) return 1;
    return 0;
}

static const VxDeclaredTensor* vx_execution_tensor(const VxDeclaredTensor* tensors,
    size_t count, const char* name) {
    for (size_t i = 0; name && i < count; i++) if (!strcmp(tensors[i].name, name)) return &tensors[i];
    return NULL;
}

static int vx_execution_same_shape(const VxDeclaredTensor* a, const VxDeclaredTensor* b) {
    if (!a || !b || a->dtype != b->dtype || a->rank != b->rank) return 0;
    for (uint32_t i = 0; i < a->rank; i++) {
        if (a->minimums[i] != b->minimums[i] || a->maximums[i] != b->maximums[i] ||
            a->multiples[i] != b->multiples[i] || a->kinds[i] != b->kinds[i] ||
            !!a->symbols[i] != !!b->symbols[i] ||
            (a->symbols[i] && strcmp(a->symbols[i], b->symbols[i]))) return 0;
    }
    return 1;
}

static int vx_execution_names_valid(const cJSON* names, const VxDeclaredTensor* tensors,
                                     size_t count, int required) {
    if (!names) return !required;
    if (!cJSON_IsArray(names)) return 0;
    for (const cJSON* item = names->child; item; item = item->next) {
        if (!cJSON_IsString(item) || !item->valuestring ||
            !vx_execution_declared(tensors, count, item->valuestring)) return 0;
        for (const cJSON* prior = names->child; prior != item; prior = prior->next)
            if (!strcmp(prior->valuestring, item->valuestring)) return 0;
    }
    return 1;
}

static const cJSON* vx_execution_state_contract(const VxModel* model, int* kind) {
    const cJSON* document = vx_execution_document(model);
    const cJSON* state = cJSON_GetObjectItemCaseSensitive(document, "stateless");
    if (state) { *kind = 1; return state; }
    state = cJSON_GetObjectItemCaseSensitive(document, "incremental");
    if (state) { *kind = 2; return state; }
    *kind = 3;
    return cJSON_GetObjectItemCaseSensitive(document, "tensor_feedback");
}

static int vx_full_prefix_axis(const VxDeclaredTensor* tensor, uint32_t axis,
        const VxDeclaredTensor* tokens) {
    return tensor && axis < tensor->rank && tokens && tokens->rank == 2 &&
        tensor->minimums[axis] == tokens->minimums[1] && tensor->maximums[axis] == tokens->maximums[1] &&
        tensor->multiples[axis] == tokens->multiples[1] && tensor->kinds[axis] == tokens->kinds[1] &&
        !!tensor->symbols[axis] == !!tokens->symbols[1] &&
        (!tensor->symbols[axis] || !strcmp(tensor->symbols[axis], tokens->symbols[1]));
}

/* The contract classifies every input, so generation never guesses which
 * integer tensor is a cursor, a mask, or immutable conditioning. */
static int vx_full_prefix_input_kind(const cJSON* generation, const char* name) {
    const cJSON* prefix = cJSON_GetObjectItemCaseSensitive(generation, "full_prefix");
    const char* fields[] = {"token_input", "keep_input", "position_input", "sequence_input",
        "chunk_mask_input", "multimodal_position_input"};
    int kind = 0;
    for (int i = 0; i < 6; ++i) {
        const char* value = vx_execution_string(i < 2 ? generation : prefix, fields[i]);
        if (value && !strcmp(value, name)) { if (kind) return -1; kind = i + 1; }
    }
    if (vx_execution_names_contains(cJSON_GetObjectItemCaseSensitive(prefix, "zero_append_inputs"), name)) {
        if (kind) return -1;
        kind = 7;
    }
    return kind;
}

static int vx_full_prefix_contract_valid(const VxModel* model, const cJSON* generation,
        const cJSON* initialize, const cJSON* transition) {
    static const char* const fields[] = {"position_input", "sequence_input", "chunk_mask_input",
        "chunk_size", "zero_append_inputs", "multimodal_position_input"};
    const cJSON* prefix = cJSON_GetObjectItemCaseSensitive(generation, "full_prefix");
    if (!vx_json_object_fields(prefix, fields, 6, 1, 6) ||
        !vx_execution_string(prefix, "position_input")) return 0;
    const char* bindings[] = {"position_input", "sequence_input", "chunk_mask_input", "multimodal_position_input"};
    for (int i = 0; i < 4; ++i) {
        const cJSON* item = cJSON_GetObjectItemCaseSensitive(prefix, bindings[i]);
        if (item && (!cJSON_IsString(item) || (item->valuestring[0] &&
            !vx_execution_declared(model->inputs, model->input_count, item->valuestring)))) return 0;
    }
    if (!vx_execution_names_valid(cJSON_GetObjectItemCaseSensitive(prefix, "zero_append_inputs"),
            model->inputs, model->input_count, 0)) return 0;
    const cJSON* chunk = cJSON_GetObjectItemCaseSensitive(prefix, "chunk_size");
    const int has_chunk = vx_execution_string(prefix, "chunk_mask_input") != NULL;
    if ((chunk && (!cJSON_IsNumber(chunk) || chunk->valuedouble != trunc(chunk->valuedouble) ||
            chunk->valuedouble < 0 || chunk->valuedouble > INT32_MAX)) ||
        (has_chunk && (!chunk || chunk->valueint <= 0)) || (!has_chunk && chunk && chunk->valueint)) return 0;
    const VxDeclaredTensor* tokens = vx_execution_tensor(model->inputs, model->input_count,
        vx_execution_string(generation, "token_input"));
    if (!tokens || tokens->rank != 2 || tokens->multiples[1] != 1 || tokens->minimums[1] < 1) return 0;
    const cJSON* invariants = cJSON_GetObjectItemCaseSensitive(vx_execution_document(model), "invariant_inputs");
    const cJSON* entries[] = {initialize, transition};
    for (int i = 0; i < 2; ++i) {
        if (vx_execution_effect(entries[i]) != VX_EXECUTION_EFFECT_EVALUATE) return 0;
        const cJSON* ranges = cJSON_GetObjectItemCaseSensitive(entries[i], "allowed_ranges");
        if (cJSON_GetArraySize(ranges) != 1 || vx_execution_range(ranges->child) != VX_EXECUTION_RANGE_FULL) return 0;
        const cJSON* outputs = cJSON_GetObjectItemCaseSensitive(entries[i], "outputs");
        if (!vx_execution_names_contains(outputs, vx_execution_string(generation, "token_output")) ||
            !vx_execution_names_contains(outputs, vx_execution_string(generation, "logits_output"))) return 0;
        for (size_t j = 0; j < model->input_count; ++j)
            if (!vx_execution_names_contains(cJSON_GetObjectItemCaseSensitive(entries[i], "inputs"), model->inputs[j].name)) return 0;
    }
    for (size_t i = 0; i < model->input_count; ++i) {
        const VxDeclaredTensor* tensor = &model->inputs[i];
        int kind = vx_full_prefix_input_kind(generation, tensor->name);
        if (kind < 0) return 0;
        if (!kind) {
            if (!vx_execution_names_contains(invariants, tensor->name)) return 0;
            for (uint32_t axis = 0; axis < tensor->rank; ++axis)
                if (tensor->minimums[axis] != tensor->maximums[axis]) return 0;
            continue;
        }
        if (tensor->dtype != VX_DTYPE_I32 || vx_execution_names_contains(invariants, tensor->name)) return 0;
        if (kind <= 2) { if (!vx_execution_same_shape(tokens, tensor)) return 0; }
        else if (kind == 3 || kind == 4) {
            if (tensor->rank != 1 || !vx_full_prefix_axis(tensor, 0, tokens)) return 0;
        } else if (kind == 5) {
            if (tensor->rank != 3 || tensor->minimums[0] != 1 || tensor->maximums[0] != 1 ||
                !vx_full_prefix_axis(tensor, 1, tokens) || !vx_full_prefix_axis(tensor, 2, tokens)) return 0;
        } else if (kind == 6) {
            if (tensor->rank != 2 || tensor->minimums[0] != 3 || tensor->maximums[0] != 3 ||
                !vx_full_prefix_axis(tensor, 1, tokens)) return 0;
        } else if (!((tensor->rank == 1 && vx_full_prefix_axis(tensor, 0, tokens)) ||
            (tensor->rank == 2 && tensor->minimums[0] == 1 && tensor->maximums[0] == 1 &&
             vx_full_prefix_axis(tensor, 1, tokens)))) return 0;
    }
    return 1;
}
/* Explicit fixed-capacity controls. Only the newly accepted token row (and
 * its mask row/column) changes; already computed activations remain valid. */
static int vx_incremental_control_kind(const cJSON* generation,const char* name) {
    const cJSON* controls=cJSON_GetObjectItemCaseSensitive(generation,"incremental_controls");
    const char* fields[]={"position_input","sequence_input","chunk_mask_input","multimodal_position_input"};
    int kind=0;
    for(int i=0;i<4;i++) {
        const char* value=vx_execution_string(controls,fields[i]);
        if(value && !strcmp(value,name)) {if(kind) return -1;kind=i+3;}
    }
    if(vx_execution_names_contains(cJSON_GetObjectItemCaseSensitive(controls,"zero_append_inputs"),name)) {
        if(kind) return -1;
        kind=7;
    }
    return kind;
}

static int vx_incremental_controls_contract_valid(const VxModel* model,const cJSON* generation) {
    static const char* const fields[]={"position_input","sequence_input","chunk_mask_input","chunk_size","zero_append_inputs","multimodal_position_input"};
    const cJSON* controls=cJSON_GetObjectItemCaseSensitive(generation,"incremental_controls");
    if(!controls) return 1;
    const char* token=vx_execution_string(generation,"token_input");
    const char* keep=vx_execution_string(generation,"keep_input");
    if(!token || !keep) return 0;
    const cJSON* document=vx_execution_document(model);
    const cJSON* incremental=cJSON_GetObjectItemCaseSensitive(document,"incremental");
    const cJSON* invariants=cJSON_GetObjectItemCaseSensitive(document,"invariant_inputs");
    const cJSON* updates=cJSON_GetObjectItemCaseSensitive(incremental,"update_inputs");
    const char* position=vx_execution_string(controls,"position_input");
    const char* state_position=vx_execution_string(incremental,"position_input");
    if(!vx_json_object_fields(controls,fields,6,1,6) || !position || !state_position || strcmp(position,state_position) ||
       cJSON_GetObjectItemCaseSensitive(generation,"full_prefix")) return 0;
    const VxDeclaredTensor* tokens=vx_execution_tensor(model->inputs,model->input_count,token);
    if(!tokens || tokens->rank!=2) return 0;
    if(!vx_execution_names_valid(cJSON_GetObjectItemCaseSensitive(controls,"zero_append_inputs"),model->inputs,model->input_count,0)) return 0;
    const cJSON* chunk=cJSON_GetObjectItemCaseSensitive(controls,"chunk_size");
    int has_chunk=vx_execution_string(controls,"chunk_mask_input")!=NULL;
    if((chunk && (!cJSON_IsNumber(chunk) || chunk->valuedouble!=trunc(chunk->valuedouble) || chunk->valuedouble<0 || chunk->valuedouble>INT32_MAX)) ||
       (has_chunk && (!chunk || chunk->valueint<=0)) || (!has_chunk && chunk && chunk->valueint)) return 0;
    for(int i=0;i<4;i++) {
        const cJSON* item=cJSON_GetObjectItemCaseSensitive(controls,fields[i==3?5:i]);
        if(item && (!cJSON_IsString(item) || (item->valuestring[0] && !vx_execution_declared(model->inputs,model->input_count,item->valuestring)))) return 0;
    }
    for(size_t i=0;i<model->input_count;i++) {
        const VxDeclaredTensor* tensor=&model->inputs[i];
        int kind=vx_incremental_control_kind(generation,tensor->name);
        if(kind<0) return 0;
        if(!kind) {
            if(!strcmp(tensor->name,token) || !strcmp(tensor->name,keep)) continue;
            if(!vx_execution_names_contains(invariants,tensor->name)) return 0;
            continue;
        }
        if(tensor->dtype!=VX_DTYPE_I32 || vx_execution_names_contains(invariants,tensor->name) || !vx_execution_names_contains(updates,tensor->name)) return 0;
        if(kind==3 || kind==4) {if(tensor->rank!=1 || !vx_full_prefix_axis(tensor,0,tokens)) return 0;}
        else if(kind==5) {
            if(tensor->rank!=3 || tensor->minimums[0]!=1 || tensor->maximums[0]!=1 ||
               !vx_full_prefix_axis(tensor,1,tokens) || !vx_full_prefix_axis(tensor,2,tokens)) return 0;
        } else if(kind==6) {
            if(tensor->rank!=2 || tensor->minimums[0]!=3 || tensor->maximums[0]!=3 || !vx_full_prefix_axis(tensor,1,tokens)) return 0;
        } else if(!((tensor->rank==1 && vx_full_prefix_axis(tensor,0,tokens)) ||
            (tensor->rank==2 && tensor->minimums[0]==1 && tensor->maximums[0]==1 && vx_full_prefix_axis(tensor,1,tokens)))) return 0;
        if(!strcmp(tensor->name,token) || !strcmp(tensor->name,keep)) return 0;
    }
    return 1;
}

static int vx_execution_contract_valid(const VxModel* model) {
    static const char* const fields[] = {"entrypoints", "invariant_inputs", "stateless",
        "incremental", "tensor_feedback", "generation"};
    static const char* const entry_fields[] = {"name", "effect", "allowed_ranges", "inputs", "outputs"};
    static const char* const incremental_fields[] = {"active_lengths_input", "position_input", "sequence_axis", "update_inputs"};
    static const char* const feedback_fields[] = {"bindings", "position_input", "active_lengths_input"};
    static const char* const binding_fields[] = {"input_name", "output_name"};
    static const char* const generation_fields[] = {"initialize_entrypoint", "transition_entrypoint",
        "token_input", "token_output", "keep_input", "logits_output", "logits_axis",
        "id_domain_size", "output_name", "sequence_axis", "full_prefix", "incremental_controls"};
    const cJSON* document = vx_execution_document(model);
    if (!document || !vx_json_object_fields(document, fields, 6, 2, 6)) return 0;
    int state_count = !!cJSON_GetObjectItemCaseSensitive(document, "stateless") +
        !!cJSON_GetObjectItemCaseSensitive(document, "incremental") +
        !!cJSON_GetObjectItemCaseSensitive(document, "tensor_feedback");
    if (state_count != 1) return 0;
    int state_kind = 0;
    const cJSON* state = vx_execution_state_contract(model, &state_kind);
    if (!cJSON_IsObject(state)) return 0;
    if (state_kind == 1 && state->child) return 0;
    if (state_kind == 2 && !vx_json_object_fields(state, incremental_fields, 4, 2, 4)) return 0;
    if (state_kind == 3 && !vx_json_object_fields(state, feedback_fields, 3, 1, 3)) return 0;
    if (!vx_execution_names_valid(cJSON_GetObjectItemCaseSensitive(document, "invariant_inputs"),
            model->inputs, model->input_count, 0)) return 0;
    const cJSON* entries = cJSON_GetObjectItemCaseSensitive(document, "entrypoints");
    if (!cJSON_IsArray(entries) || !entries->child) return 0;
    for (const cJSON* entry = entries->child; entry; entry = entry->next) {
        const char* name = vx_execution_string(entry, "name");
        int effect = vx_execution_effect(entry);
        if (!vx_json_object_fields(entry, entry_fields, 5, 5, 5) || !name || !effect ||
            (state_kind == 1 && effect != VX_EXECUTION_EFFECT_EVALUATE) ||
            vx_execution_entry(model, name) != entry ||
            !vx_execution_names_valid(cJSON_GetObjectItemCaseSensitive(entry, "inputs"),
                model->inputs, model->input_count, 1) ||
            !vx_execution_names_valid(cJSON_GetObjectItemCaseSensitive(entry, "outputs"),
                model->outputs, model->output_count, 1)) return 0;
        const cJSON* ranges = cJSON_GetObjectItemCaseSensitive(entry, "allowed_ranges");
        if (!cJSON_IsArray(ranges) || !ranges->child) return 0;
        unsigned seen = 0;
        for (const cJSON* item = ranges->child; item; item = item->next) {
            int range = vx_execution_range(item);
            if (!range || (seen & (1u << range))) return 0;
            seen |= 1u << range;
        }
    }
    const cJSON* length_binding = cJSON_GetObjectItemCaseSensitive(state, "active_lengths_input");
    const cJSON* position_binding = cJSON_GetObjectItemCaseSensitive(state, "position_input");
    /* Optional empty names are absent bindings, but present values must still
     * fit the typed contract returned by GetModelInfo. */
    if ((length_binding && !cJSON_IsString(length_binding)) ||
        (position_binding && !cJSON_IsString(position_binding))) return 0;
    const char* length_name = vx_execution_string(state, "active_lengths_input");
    const char* position_name = vx_execution_string(state, "position_input");
    if (state_kind == 3 && !length_name && !position_name) return 0;
    if (state_kind == 2 && (!length_name || !vx_execution_names_valid(
            cJSON_GetObjectItemCaseSensitive(state, "update_inputs"), model->inputs, model->input_count, 1))) return 0;
    if ((length_name && !vx_execution_declared(model->inputs, model->input_count, length_name)) ||
        (position_name && !vx_execution_declared(model->inputs, model->input_count, position_name))) return 0;
    const VxDeclaredTensor* lengths = vx_execution_tensor(model->inputs, model->input_count, length_name);
    const VxDeclaredTensor* position = vx_execution_tensor(model->inputs, model->input_count, position_name);
    if ((lengths && (lengths->dtype != VX_DTYPE_I32 || lengths->rank != 1)) ||
        (position && position->dtype != VX_DTYPE_I32)) return 0;
    if (state_kind == 2) {
        const cJSON* sequence_axis = cJSON_GetObjectItemCaseSensitive(state, "sequence_axis");
        if (!cJSON_IsNumber(sequence_axis) || sequence_axis->valuedouble != 1) return 0;
        const cJSON* updates = cJSON_GetObjectItemCaseSensitive(state, "update_inputs");
        const cJSON* invariants = cJSON_GetObjectItemCaseSensitive(document, "invariant_inputs");
        for (const cJSON* item = updates->child; item; item = item->next)
            if (vx_execution_names_contains(invariants, item->valuestring)) return 0;
    }
    if (state_kind == 3) {
        const cJSON* bindings = cJSON_GetObjectItemCaseSensitive(state, "bindings");
        if (!cJSON_IsArray(bindings) || !bindings->child) return 0;
        for (const cJSON* binding = bindings->child; binding; binding = binding->next) {
            if (!vx_json_object_fields(binding, binding_fields, 2, 2, 2) ||
                !vx_execution_declared(model->inputs, model->input_count, vx_execution_string(binding, "input_name")) ||
                !vx_execution_declared(model->outputs, model->output_count, vx_execution_string(binding, "output_name"))) return 0;
            const VxDeclaredTensor* input = vx_execution_tensor(model->inputs, model->input_count,
                vx_execution_string(binding, "input_name"));
            const VxDeclaredTensor* output = vx_execution_tensor(model->outputs, model->output_count,
                vx_execution_string(binding, "output_name"));
            /* A recurrent state may grow (past[P] -> present[P+1]). The next
             * invocation validates the actual source shape against the input
             * domain before rebinding; declaration identity would reject it. */
            if (input->dtype != output->dtype || input->rank != output->rank) return 0;
            for (const cJSON* prior = bindings->child; prior != binding; prior = prior->next)
                if (!strcmp(vx_execution_string(prior, "input_name"), vx_execution_string(binding, "input_name"))) return 0;
        }
    }
    const cJSON* generation = cJSON_GetObjectItemCaseSensitive(document, "generation");
    if (generation) {
        const cJSON* initialize = vx_execution_entry(model, vx_execution_string(generation, "initialize_entrypoint"));
        const cJSON* transition = vx_execution_entry(model, vx_execution_string(generation, "transition_entrypoint"));
        const cJSON* full_prefix = cJSON_GetObjectItemCaseSensitive(generation, "full_prefix");
        if (!vx_json_object_fields(generation, generation_fields, 12, 8, 12) ||
            !vx_incremental_controls_contract_valid(model,generation) ||
            (full_prefix ? state_kind != 1 || !vx_full_prefix_contract_valid(model, generation, initialize, transition) :
             state_kind != 2 || vx_execution_effect(initialize) != VX_EXECUTION_EFFECT_INITIALIZE ||
             vx_execution_effect(transition) != VX_EXECUTION_EFFECT_TRANSITION) ||
            !vx_execution_declared(model->inputs, model->input_count, vx_execution_string(generation, "token_input")) ||
            !vx_execution_declared(model->inputs, model->input_count, vx_execution_string(generation, "keep_input")) ||
            !vx_execution_declared(model->outputs, model->output_count, vx_execution_string(generation, "token_output")) ||
            !vx_execution_string(generation, "output_name")) return 0;
        const cJSON* domain = cJSON_GetObjectItemCaseSensitive(generation, "id_domain_size");
        const cJSON* axis = cJSON_GetObjectItemCaseSensitive(generation, "sequence_axis");
        if (!cJSON_IsNumber(domain) || domain->valuedouble <= 0 || domain->valuedouble > INT32_MAX ||
            trunc(domain->valuedouble) != domain->valuedouble || !cJSON_IsNumber(axis) || axis->valuedouble != 1) return 0;
        const VxDeclaredTensor* tokens = vx_execution_tensor(model->inputs, model->input_count,
            vx_execution_string(generation, "token_input"));
        const VxDeclaredTensor* keep = vx_execution_tensor(model->inputs, model->input_count,
            vx_execution_string(generation, "keep_input"));
        const VxDeclaredTensor* ids = vx_execution_tensor(model->outputs, model->output_count,
            vx_execution_string(generation, "token_output"));
        const VxDeclaredTensor* logits = vx_execution_tensor(model->outputs, model->output_count,
            vx_execution_string(generation, "logits_output"));
        const cJSON* logits_axis = cJSON_GetObjectItemCaseSensitive(generation, "logits_axis");
        if (!tokens || tokens->dtype != VX_DTYPE_I32 || tokens->rank != 2 ||
            tokens->minimums[0] != 1 || tokens->maximums[0] != 1 ||
            !vx_execution_same_shape(tokens, keep) || !vx_execution_same_shape(tokens, ids) ||
            !logits || logits->rank != 3 ||
            (full_prefix && logits->dtype != VX_DTYPE_F32) ||
            (logits->dtype != VX_DTYPE_F32 && logits->dtype != VX_DTYPE_I8 && logits->dtype != VX_DTYPE_U8) ||
            logits->minimums[0] != 1 || logits->maximums[0] != 1 ||
            logits->minimums[1] != tokens->minimums[1] || logits->maximums[1] != tokens->maximums[1] ||
            logits->minimums[2] != domain->valueint || logits->maximums[2] != domain->valueint ||
            !cJSON_IsNumber(logits_axis) || logits_axis->valuedouble != 2 ||
            !strcmp(tokens->name, keep->name) || !strcmp(tokens->name, ids->name) ||
            !strcmp(keep->name, ids->name)) return 0;
    }
    return 1;
}
