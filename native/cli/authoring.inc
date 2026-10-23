/* Full-profile CLI file orchestration. Numerical and policy decisions remain in
 * the generated services. All JSON keys are checked before any engine call. */
static int cli_json_failed;

static int cli_json_keys(const cJSON* object, const char* const* allowed) {
    const cJSON *item, *other;
    if (!cJSON_IsObject(object)) return -1;
    for (item = object->child; item; item = item->next) {
        size_t i;
        for (other = object->child; other != item; other = other->next) {
            if (!strcmp(item->string, other->string)) {
                fprintf(stderr, "Duplicate JSON key: %s\n", item->string);
                return -1;
            }
        }
        for (i = 0; allowed[i] && strcmp(item->string, allowed[i]); i++) {}
        if (!allowed[i]) {
            fprintf(stderr, "Unknown JSON key: %s\n", item->string);
            return -1;
        }
    }
    return 0;
}

static const cJSON* cli_json_field(const cJSON* object, const char* snake,
                                  const char* camel) {
    const cJSON* a = cJSON_GetObjectItemCaseSensitive(object, snake);
    const cJSON* b = camel ? cJSON_GetObjectItemCaseSensitive(object, camel) : NULL;
    if (a && b) {
        fprintf(stderr, "Duplicate JSON aliases: %s and %s\n", snake, camel);
        cli_json_failed = 1;
        return NULL;
    }
    return a ? a : b;
}

static int cli_json_u32(const cJSON* value, uint32_t* out) {
    if (!cJSON_IsNumber(value) || !isfinite(value->valuedouble) ||
        value->valuedouble < 0 || value->valuedouble > UINT32_MAX ||
        floor(value->valuedouble) != value->valuedouble) return -1;
    *out = (uint32_t)value->valuedouble;
    return 0;
}

static cJSON* cli_read_json(const char* path) {
    void* bytes = NULL;
    size_t size = 0;
    const char* end = NULL;
    cJSON* value = NULL;
    if (read_whole_file(path, &bytes, &size) || !size) {
        fprintf(stderr, "JSON file must be nonempty and readable: %s\n", path);
        free(bytes);
        return NULL;
    }
    /* cJSON strings are NUL-terminated. Refuse embedded NUL and its escape so
     * keys/paths cannot be silently truncated before strict schema validation. */
    if (memchr(bytes, 0, size)) goto done;
    for (size_t i = 0; i + 5 < size; i++) {
        if (!memcmp((char*)bytes + i, "\\u0000", 6)) {
            size_t preceding=0;
            while (preceding<i && ((char*)bytes)[i-preceding-1]=='\\') preceding++;
            if (!(preceding&1u)) goto done;
        }
    }
    value = cJSON_ParseWithLengthOpts(bytes, size, &end, 0);
    if (value) {
        const char* limit = (char*)bytes + size;
        while (end < limit && (*end==' ' || *end=='\t' || *end=='\r' || *end=='\n')) end++;
        if (end != limit) { cJSON_Delete(value); value = NULL; }
    }
done:
    free(bytes);
    if (!value) fprintf(stderr, "Invalid JSON document: %s\n", path);
    return value;
}

static int cli_relative_path(char* out, size_t capacity, const char* manifest,
                              const char* value) {
    const char* slash;
    char parent[PATH_MAX];
    if (!value || !value[0]) return -1;
    if (value[0]=='/') return copy_path(out, capacity, value);
    slash = strrchr(manifest, '/');
    if (!slash) return copy_path(out, capacity, value);
    size_t length = (size_t)(slash-manifest);
    if (length >= sizeof(parent)) return -1;
    memcpy(parent, manifest, length); parent[length]=0;
    return join_path(out, capacity, length ? parent : "/", value);
}

static int cli_new_output(const char* path) {
    struct stat st;
    if (!path || !path[0] || lstat(path, &st)==0 || errno!=ENOENT) {
        fprintf(stderr, "Output must be a new file: %s\n", path ? path : "(missing)");
        return -1;
    }
    return 0;
}

static int cli_write_new(const char* path, const void* data, size_t size) {
    FILE* f = fopen(path, "wbx");
    int failed;
    if (!f) { fprintf(stderr,"Cannot create new output: %s (%s)\n",path,strerror(errno)); return -1; }
    failed = size && fwrite(data, 1, size, f)!=size;
    if (fclose(f)) failed=1;
    if (failed) fprintf(stderr,"Failed writing output: %s\n",path);
    return failed ? -1 : 0;
}

static int cli_write_json(const char* path, cJSON* value) {
    char* text;
    int result;
    if (!path) return 0;
    text=cJSON_PrintUnformatted(value);
    if (!text) return -1;
    result=cli_write_new(path,text,strlen(text));
    cJSON_free(text);
    return result;
}

static int cli_algorithm(const char* name, int friendly,
                         VolvoxaiV1WeightAlgorithm* out) {
    if (!name) return -1;
    if (!strcmp(name,"WEIGHT_ALGORITHM_RTN") || (friendly && !strcmp(name,"rtn")))
        *out=VOLVOXAI_V1_WEIGHT_ALGORITHM_RTN;
    else if (!strcmp(name,"WEIGHT_ALGORITHM_MSE_CLIP") || (friendly && !strcmp(name,"mse-clip")))
        *out=VOLVOXAI_V1_WEIGHT_ALGORITHM_MSE_CLIP;
    else if (!strcmp(name,"WEIGHT_ALGORITHM_IMPORTANCE") || (friendly && !strcmp(name,"importance")))
        *out=VOLVOXAI_V1_WEIGHT_ALGORITHM_IMPORTANCE;
    else return -1;
    return 0;
}

static int cli_parse_weights(const cJSON* json, VolvoxaiV1WeightQuantization* q) {
    static const char* const keys[]={"format","algorithm",NULL};
    static const char* const format_keys[]={"bits","group_size","groupSize",NULL};
    const cJSON *format, *algorithm, *group;
    uint32_t enum_value;
    if (cli_json_keys(json,keys)) return -1;
    format=cli_json_field(json,"format",NULL);
    algorithm=cli_json_field(json,"algorithm",NULL);
    if (cli_json_keys(format,format_keys)) return -1;
    q->field_format=q->_allocator->allocate(q->_allocator->context,sizeof(*q->field_format));
    if (!q->field_format) return -1;
    volvoxai_v1_weight_format_init_with_allocator(q->field_format,q->_allocator);
    if (cli_json_u32(cli_json_field(format,"bits",NULL),&q->field_format->field_bits)) return -1;
    group=cli_json_field(format,"group_size","groupSize");
    if (group && cli_json_u32(group,&q->field_format->field_group_size)) return -1;
    if (!algorithm) q->field_algorithm=VOLVOXAI_V1_WEIGHT_ALGORITHM_RTN;
    else if (cJSON_IsString(algorithm)) {
        if (cli_algorithm(algorithm->valuestring,0,&q->field_algorithm)) return -1;
    }
    else if (!cli_json_u32(algorithm,&enum_value)) q->field_algorithm=(VolvoxaiV1WeightAlgorithm)enum_value;
    else return -1;
    return !cli_json_failed && q->field_algorithm >= VOLVOXAI_V1_WEIGHT_ALGORITHM_RTN &&
        q->field_algorithm <= VOLVOXAI_V1_WEIGHT_ALGORITHM_IMPORTANCE ? 0 : -1;
}

static int cli_parse_rules(const cJSON* json, VolvoxaiV1QuantizationConfig* config) {
    static const char* const wrapper_keys[]={"rules",NULL};
    static const char* const keys[]={"node_id","nodeId","operator_name","operatorName",
        "node_glob","nodeGlob","weights","keep_float","keepFloat",NULL};
    if (cJSON_IsObject(json)) {
        if (cli_json_keys(json,wrapper_keys)) return -1;
        json=cli_json_field(json,"rules",NULL);
    }
    if (!cJSON_IsArray(json)) return -1;
    for (const cJSON* item=json->child;item;item=item->next) {
        const cJSON *id,*op,*glob,*weight,*keep;
        VolvoxaiV1QuantizationRule* rule;
        if (cli_json_keys(item,keys)) return -1;
        id=cli_json_field(item,"node_id","nodeId");
        op=cli_json_field(item,"operator_name","operatorName");
        glob=cli_json_field(item,"node_glob","nodeGlob");
        weight=cli_json_field(item,"weights",NULL);
        keep=cli_json_field(item,"keep_float","keepFloat");
        if (cli_json_failed || !!id+!!op+!!glob!=1 || !!weight+!!keep!=1) return -1;
        const cJSON* selector=id?id:op?op:glob;
        if (!cJSON_IsString(selector) || !selector->valuestring[0]) return -1;
        rule=volvoxai_v1_quantization_config_add_rules(config);
        if (!rule) return -1;
        rule->which_selector=id?1:op?2:3;
        if (assign_text(rule->_allocator,id?&rule->field_node_id:op?&rule->field_operator_name:&rule->field_node_glob,selector->valuestring)) return -1;
        if (weight) {
            rule->which_action=4;
            rule->field_weights=rule->_allocator->allocate(rule->_allocator->context,sizeof(*rule->field_weights));
            if (!rule->field_weights) return -1;
            volvoxai_v1_weight_quantization_init_with_allocator(rule->field_weights,rule->_allocator);
            if (cli_parse_weights(weight,rule->field_weights)) return -1;
        } else {
            if (!cJSON_IsObject(keep) || keep->child) return -1;
            rule->which_action=5;
            rule->field_keep_float=rule->_allocator->allocate(rule->_allocator->context,sizeof(*rule->field_keep_float));
            if (!rule->field_keep_float) return -1;
            volvoxai_v1_empty_init_with_allocator(rule->field_keep_float,rule->_allocator);
        }
    }
    return 0;
}

static int cli_read_qat(const char* path, VolvoxaiV1QatOptions* policy) {
    static const char* const keys[]={"nodes",NULL};
    static const char* const node_keys[]={"node_id","nodeId","weights",NULL};
    cJSON* json=cli_read_json(path);
    int result=-1;
    const cJSON* nodes;
    cli_json_failed=0;
    if (!json || cli_json_keys(json,keys)) goto done;
    nodes=cli_json_field(json,"nodes",NULL);
    if (!cJSON_IsArray(nodes) || !nodes->child) goto done;
    for (const cJSON* item=nodes->child;item;item=item->next) {
        VolvoxaiV1QatNode* node;
        const cJSON* id;
        if (cli_json_keys(item,node_keys)) goto done;
        id=cli_json_field(item,"node_id","nodeId");
        if (cli_json_failed || !cJSON_IsString(id) || !id->valuestring[0]) goto done;
        node=volvoxai_v1_qat_options_add_nodes(policy);
        if (!node || assign_text(node->_allocator,&node->field_node_id,id->valuestring)) goto done;
        node->field_weights=node->_allocator->allocate(node->_allocator->context,sizeof(*node->field_weights));
        if (!node->field_weights) goto done;
        volvoxai_v1_weight_quantization_init_with_allocator(node->field_weights,node->_allocator);
        if (cli_parse_weights(cli_json_field(item,"weights",NULL),node->field_weights)) goto done;
    }
    result=0;
done:
    cJSON_Delete(json);
    if (result) fprintf(stderr,"Invalid QatOptions JSON: %s\n",path);
    return result;
}

/* Metadata stays in cJSON; one batch's raw tensors are read only for its call. */
static int cli_batch(const cJSON* item, const char* manifest, int training,
                     RunOptions* run, char* targets, const char** sample,
                     const char** profile, uint32_t* sample_count) {
    static const char* const train_keys[]={"inputs","targets",NULL};
    static const char* const calibration_keys[]={"inputs","sample_name","sampleName",
        "profile_name","profileName","sample_count","sampleCount",NULL};
    const cJSON* inputs;
    run->input_count=0;
    if (cli_json_keys(item,training?train_keys:calibration_keys)) return -1;
    inputs=cli_json_field(item,"inputs",NULL);
    if (!cJSON_IsArray(inputs) || !inputs->child) return -1;
    for (const cJSON* input=inputs->child;input;input=input->next) {
        Binding* binding;
        char relative[PATH_MAX];
        if (!cJSON_IsString(input) || run->input_count==MAX_BINDINGS) return -1;
        binding=&run->inputs[run->input_count];
        if (parse_binding(input->valuestring,binding,1) ||
            cli_relative_path(relative,sizeof(relative),manifest,binding->path) ||
            copy_path(binding->path,sizeof(binding->path),relative) || !file_exists(binding->path)) return -1;
        for (size_t i=0;i<run->input_count;i++) if (!strcmp(run->inputs[i].name,binding->name)) return -1;
        run->input_count++;
    }
    if (training) {
        const cJSON* target=cli_json_field(item,"targets",NULL);
        if (!cJSON_IsString(target) || cli_relative_path(targets,PATH_MAX,manifest,target->valuestring) ||
            !file_exists(targets) || !has_suffix(targets,".i32")) return -1;
    } else {
        const cJSON *s=cli_json_field(item,"sample_name","sampleName"),
                    *p=cli_json_field(item,"profile_name","profileName"),
                    *n=cli_json_field(item,"sample_count","sampleCount");
        if ((s && (!cJSON_IsString(s) || !s->valuestring[0])) || (p && !cJSON_IsString(p))) return -1;
        *sample=s?s->valuestring:NULL;*profile=p?p->valuestring:"";*sample_count=1;
        if (n && (cli_json_u32(n,sample_count) || !*sample_count)) return -1;
    }
    return cli_json_failed?-1:0;
}

static cJSON* cli_read_batches(const char* path, int training) {
    static const char* const keys[]={"batches",NULL};
    cJSON* root=cli_read_json(path);
    const cJSON* batches;
    RunOptions* scratch=calloc(1,sizeof(*scratch));
    char targets[PATH_MAX];const char *sample,*profile;uint32_t count;
    cli_json_failed=0;
    if (!root || !scratch || cli_json_keys(root,keys)) goto failed;
    batches=cli_json_field(root,"batches",NULL);
    if (!cJSON_IsArray(batches) || !batches->child) goto failed;
    for (const cJSON* item=batches->child;item;item=item->next)
        if (cli_batch(item,path,training,scratch,targets,&sample,&profile,&count)) goto failed;
    free(scratch);return root;
failed:
    fprintf(stderr,"Invalid batch manifest or missing typed input file: %s\n",path);
    free(scratch);cJSON_Delete(root);return NULL;
}

static cJSON* cli_weights_json(const VolvoxaiV1WeightQuantization* q) {
    cJSON* value=cJSON_CreateObject();
    cJSON* format=cJSON_CreateObject();
    if (!value || !format || !q || !q->field_format) {cJSON_Delete(value);cJSON_Delete(format);return NULL;}
    cJSON_AddNumberToObject(format,"bits",q->field_format->field_bits);
    cJSON_AddNumberToObject(format,"groupSize",q->field_format->field_group_size);
    cJSON_AddItemToObject(value,"format",format);
    cJSON_AddStringToObject(value,"algorithm",volvoxai_v1_weight_algorithm_name(q->field_algorithm));
    return value;
}

static cJSON* cli_qat_json(const VolvoxaiV1QatOptions* policy) {
    cJSON* value=cJSON_CreateObject();cJSON* nodes=cJSON_CreateArray();
    if (!value || !nodes) {cJSON_Delete(value);cJSON_Delete(nodes);return NULL;}
    if (policy) for (size_t i=0;i<policy->field_nodes.len;i++) {
        const VolvoxaiV1QatNode* q=&policy->field_nodes.data[i];cJSON* node=cJSON_CreateObject();
        if (!node) {cJSON_Delete(value);cJSON_Delete(nodes);return NULL;}
        json_add_bytes_copy(node,"nodeId",&q->field_node_id);
        cJSON_AddItemToObject(node,"weights",cli_weights_json(q->field_weights));
        cJSON_AddItemToArray(nodes,node);
    }
    cJSON_AddItemToObject(value,"nodes",nodes);return value;
}

#define CLI_RELEASE_FUNCTION(name, type, codec, field, method) \
static void name(VxCallClient* client,int64_t id) { \
    type request;uint8_t* payload=NULL;int32_t length=0; \
    if (!id) return;codec##_init(&request);request.field=id; \
    VX_CALL_MESSAGE(client,method,codec,&request,payload,length); \
    codec##_free(&request);if(payload)vx_call_free(client,payload); \
}
CLI_RELEASE_FUNCTION(cli_release_plan,VolvoxaiV1QuantizationPlanRef,volvoxai_v1_quantization_plan_ref,field_quantization_plan_id,VX_RPC_VX_QUANTIZATION_SERVICE_RELEASE_QUANTIZATION_PLAN)
CLI_RELEASE_FUNCTION(cli_release_trainer,VolvoxaiV1TrainerRef,volvoxai_v1_trainer_ref,field_trainer_id,VX_RPC_VX_TRAINING_SERVICE_RELEASE_TRAINER)
CLI_RELEASE_FUNCTION(cli_release_model,VolvoxaiV1ModelRef,volvoxai_v1_model_ref,field_model_id,VX_RPC_VX_INFERENCE_SERVICE_RELEASE_MODEL)
CLI_RELEASE_FUNCTION(cli_release_runtime,VolvoxaiV1RuntimeRef,volvoxai_v1_runtime_ref,field_runtime_id,VX_RPC_VX_INFERENCE_SERVICE_RELEASE_RUNTIME)
#undef CLI_RELEASE_FUNCTION
