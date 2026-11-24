/* Compare original-storage computation with the established F32 routes, and
 * verify that two contexts never acquire a full converted/prepacked matrix. */
static const char compact_compute_graph[] =
    "{\"format\":\"volvox-graph/v1\",\"dimensions\":{},\"inputs\":{"
    "\"x\":{\"dtype\":\"float32\",\"shape\":[3,65]},"
    "\"mx\":{\"dtype\":\"float32\",\"shape\":[2,4]},"
    "\"ids\":{\"dtype\":\"float32\",\"shape\":[2,2]},"
    "\"routes\":{\"dtype\":\"float32\",\"shape\":[2,2]},"
    "\"tokens\":{\"dtype\":\"int32\",\"shape\":[2]},"
    "\"edge\":{\"dtype\":\"float32\",\"shape\":[1,2]}},\"nodes\":["
    "{\"id\":\"w\",\"opType\":\"Linear\",\"params\":{},\"inputs\":{\"input\":\"x\",\"weight\":\"w\",\"bias\":\"b\"},\"outputs\":{\"out\":{\"tensor\":\"d0\",\"dtype\":\"float32\",\"shape\":[3,17]}}},"
    "{\"id\":\"d1\",\"opType\":\"MatMul\",\"params\":{},\"inputs\":{\"input\":\"x\",\"weight\":\"wt\",\"bias\":\"b\"},\"outputs\":{\"out\":{\"tensor\":\"d1\",\"dtype\":\"float32\",\"shape\":[3,17]}}},"
    "{\"id\":\"d2\",\"opType\":\"Gemm\",\"inputs\":{\"input\":\"x\",\"weight\":\"w\"},\"params\":{\"weight_layout\":\"dout_din\"},\"outputs\":{\"out\":{\"tensor\":\"d2\",\"dtype\":\"float32\",\"shape\":[3,17]}}},"
    "{\"id\":\"emb\",\"opType\":\"Embedding\",\"params\":{},\"inputs\":{\"input\":\"tokens\",\"weight\":\"w\"},\"outputs\":{\"out\":{\"tensor\":\"emb\",\"dtype\":\"float32\",\"shape\":[2,65]}}},"
    "{\"id\":\"ml\",\"opType\":\"MoELinear\",\"params\":{},\"inputs\":{\"input\":\"mx\",\"expert_weight\":\"ew\",\"expert_bias\":\"eb\",\"route_indices\":\"ids\",\"route_weights\":\"routes\"},\"outputs\":{\"out\":{\"tensor\":\"ml\",\"dtype\":\"float32\",\"shape\":[2,3]}}},"
    "{\"id\":\"mg\",\"opType\":\"MoEGatedMLP\",\"inputs\":{\"input\":\"mx\",\"gate_weight\":\"gw\",\"up_weight\":\"uw\",\"down_weight\":\"dw\",\"route_indices\":\"ids\",\"route_weights\":\"routes\"},\"params\":{\"route_input_scale\":true,\"swiglu_limit\":2},\"outputs\":{\"out\":{\"tensor\":\"mg\",\"dtype\":\"float32\",\"shape\":[2,3]}}},"
    "{\"id\":\"wide\",\"opType\":\"Linear\",\"params\":{},\"inputs\":{\"input\":\"edge\",\"weight\":\"Linear\"},\"outputs\":{\"out\":{\"tensor\":\"wide\",\"dtype\":\"float32\",\"shape\":[1,1]}}}],"
    "\"outputs\":[\"d0\",\"d1\",\"d2\",\"emb\",\"ml\",\"mg\",\"wide\"],\"execution\":{\"stateless\":{},\"entrypoints\":[{\"name\":\"evaluate\",\"effect\":1,\"allowed_ranges\":[1],\"inputs\":[\"x\",\"mx\",\"ids\",\"routes\",\"tokens\",\"edge\"],\"outputs\":[\"d0\",\"d1\",\"d2\",\"emb\",\"ml\",\"mg\",\"wide\"]}]}}";

static void compact_compute_weights(SafetensorsFile* compact, SafetensorsFile* reference,
        VxDataType dtype) {
    const uint16_t half[] = {0xb800,0xb400,0xb000,0,0x3000,0x3400,0x3800,0x3a00,0x3c00,0x3d00};
    const uint16_t brain[] = {0xbf00,0xbe80,0xbe00,0,0x3e00,0x3e80,0x3f00,0x3f40,0x3f80,0x3fa0};
    const float values[] = {-0.5f,-0.25f,-0.125f,0,0.125f,0.25f,0.5f,0.75f,1,1.25f};
    const char* names[] = {"w","wt","b","ew","eb","gw","uw","dw"};
    const int shapes[][3] = {{17,65,0},{65,17,0},{17,0,0},{3,4,3},{3,3,0},{3,4,5},{3,4,5},{3,5,3}};
    CHECK(!safetensors_init_empty(compact, SAFETENSORS_OPEN_READ_WRITE));
    CHECK(!safetensors_init_empty(reference, SAFETENSORS_OPEN_READ_WRITE));
    for (size_t part = 0; part < 8; ++part) {
        int rank = shapes[part][2] ? 3 : shapes[part][1] ? 2 : 1;
        size_t count = 1;
        for (int axis = 0; axis < rank; ++axis) count *= (size_t)shapes[part][axis];
        uint16_t bits[1105]; float expanded[1105];
        for (size_t i = 0; i < count; ++i) {
            size_t source = part == 1 ? (i % 17u) * 65u + i / 17u : i;
            size_t code = (source * 7u + (part <= 1 ? 0u : part)) % 10u;
            bits[i] = dtype == VX_DTYPE_F16 ? half[code] : brain[code];
            expanded[i] = values[code];
        }
        CHECK(!safetensors_add_tensor(compact,names[part],dtype,shapes[part],rank,bits,count*2));
        CHECK(!safetensors_add_tensor(reference,names[part],VX_DTYPE_F32,shapes[part],rank,expanded,count*4));
    }
    const uint16_t extreme[] = {0x7f7f, 0x0001};
    const float wide[] = {0x1.fep127f, 0x1p-133f};
    const int shape[] = {1,2};
    CHECK(!safetensors_add_tensor(compact,"Linear",VX_DTYPE_BF16,shape,2,extreme,sizeof(extreme)));
    CHECK(!safetensors_add_tensor(reference,"Linear",VX_DTYPE_F32,shape,2,wide,sizeof(wide)));
}

static VxCompiledModel* compact_compute_compile(VxRuntime* runtime, const VxBackendPolicy* policy,
        SafetensorsFile* weights, const char* document) {
    unsigned char* bytes = NULL; size_t size = 0;
    CHECK(!safetensors_serialize(weights,&bytes,&size));
    VxSourceBytes source_bytes = {bytes,size};
    VxModelPackageSource package = {{document,strlen(document)},&source_bytes,1};
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxModel* model = NULL; VxCompiledModel* compiled = NULL;
    CHECK(vx_runtime_internal_load_model_package(runtime,&source,&package,&model,&report,0)==VX_STATUS_OK);
    CHECK(vx_model_compile_options_revision(model,policy,0,NULL,VX_PACKED_ARITHMETIC_F32,
        &compiled,&report)==VX_STATUS_OK);
    vx_model_release(model); free(bytes);
    return compiled;
}

static void compact_compute_inventory(VxExecutionContext* context, int mixed) {
    VxMemoryAllocation allocations[512];
    VxMemoryInventory inventory = {.items=allocations,.capacity=512}; VxMemoryView view;
    CHECK(vx_context_memory_inventory(context,&inventory,&view)==VX_STATUS_OK && !inventory.truncated);
    for (size_t i=0;i<inventory.count;i++) {
        const VxMemoryAllocation* a=&inventory.items[i];
        if (a->allocator!=VX_MEMORY_HOST_PACKED_WEIGHTS) continue;
        if (!strcmp(a->name,"b") || !strcmp(a->name,"eb")) continue;
        if (mixed && (!strcmp(a->name,"w") || strstr(a->name,"F32") || !strcmp(a->name,"d0") || !strcmp(a->name,"d2"))) continue;
        CHECK(a->capacity==0);
    }
}

static void compact_compute_adapter_test(VxRuntime* runtime, const VxBackendPolicy* policy) {
#if defined(__unix__) || defined(__APPLE__)
    const char graph[] =
        "{\"format\":\"volvox-graph/v1\",\"dimensions\":{},\"inputs\":{\"x\":{\"dtype\":\"float32\",\"shape\":[1,2]}},"
        "\"nodes\":[{\"id\":\"linear\",\"opType\":\"Linear\",\"params\":{},\"inputs\":{\"input\":\"x\",\"weight\":\"w\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\",\"dtype\":\"float32\",\"shape\":[1,2]}}}],\"outputs\":[\"y\"],"
        "\"execution\":{\"stateless\":{},\"entrypoints\":[{\"name\":\"evaluate\",\"effect\":1,\"allowed_ranges\":[1],\"inputs\":[\"x\"],\"outputs\":[\"y\"]}]}}";
    const char manifest[] =
        "{\"format\":\"volvox.adapter.v1\",\"adapter_id\":\"compact-lora\",\"version_id\":\"v1\",\"kind\":\"lora\","
        "\"targets\":[{\"weight\":\"w\",\"a\":\"a\",\"b\":\"b\",\"rank\":1,\"alpha\":1,\"layout\":\"din_r_r_dout\"}]}";
    char directory[] = "/tmp/volvoxai-compact-adapter-XXXXXX";
    CHECK(mkdtemp(directory));
    char graph_path[256], weight_path[256], adapter_path[256];
    snprintf(graph_path,sizeof(graph_path),"%s/graph.json",directory);
    snprintf(weight_path,sizeof(weight_path),"%s/base.safetensors",directory);
    snprintf(adapter_path,sizeof(adapter_path),"%s/adapter.safetensors",directory);
    const uint16_t bits[] = {0x3f80,0x4000,0x4040,0x4080};
    const float a[] = {0.5f,1}, b[] = {2,-1};
    const int shape[] = {2,2}, a_shape[] = {2,1}, b_shape[] = {1,2};
    SafetensorsFile weights={0}, adapter={0};
    unsigned char* source_bytes=NULL; size_t source_size=0;
    CHECK(!safetensors_init_empty(&weights,SAFETENSORS_OPEN_READ_WRITE));
    CHECK(!safetensors_add_tensor(&weights,"w",VX_DTYPE_BF16,shape,2,bits,sizeof(bits)));
    CHECK(!safetensors_serialize(&weights,&source_bytes,&source_size));
    mapping_write(graph_path,graph,sizeof(graph)-1u);
    mapping_write(weight_path,source_bytes,source_size);
    CHECK(!safetensors_init_empty(&adapter,SAFETENSORS_OPEN_READ_WRITE));
    CHECK(!safetensors_add_tensor(&adapter,"a",VX_DTYPE_F32,a_shape,2,a,sizeof(a)));
    CHECK(!safetensors_add_tensor(&adapter,"b",VX_DTYPE_F32,b_shape,2,b,sizeof(b)));
    cJSON* metadata=cJSON_CreateObject(); CHECK(metadata);
    CHECK(cJSON_AddStringToObject(metadata,"volvox_adapter_manifest",manifest));
    char* metadata_json=cJSON_PrintUnformatted(metadata); CHECK(metadata_json);
    CHECK(!safetensors_set_metadata_json(&adapter,metadata_json));
    cJSON_free(metadata_json); cJSON_Delete(metadata);
    CHECK(!safetensors_save(adapter_path,&adapter));
    safetensors_free(&adapter); safetensors_free(&weights);

    const char* weight_paths[]={weight_path};
    VxModelSource source=VX_MODEL_SOURCE_INIT;
    source.graph_path=graph_path; source.weight_paths=weight_paths; source.weight_path_count=1;
    VxModel* model=NULL;
    CHECK(vx_runtime_load_model(runtime,&source,&model,&report)==VX_STATUS_OK);
    VxAdapterSource adapter_source=VX_ADAPTER_SOURCE_INIT;
    adapter_source.adapter_name="compact-lora"; adapter_source.version_name="v1";
    adapter_source.package_path=adapter_path;
    VxAdapterRevision revision=VX_ADAPTER_REVISION_INIT;
    CHECK(vx_model_publish_adapter(model,&adapter_source,&revision,&report)==VX_STATUS_OK);
    VxExecutionConfig config={.context=VX_CONTEXT_OPTIONS_INIT};
    VxExecutionContext* contexts[2]={NULL,NULL};
    for(int adapted=0;adapted<2;adapted++) {
        VxCompiledModel* compiled=NULL;
        CHECK(vx_model_compile_options_revision(model,policy,0,adapted?&revision:NULL,
            VX_PACKED_ARITHMETIC_F32,&compiled,&report)==VX_STATUS_OK);
        CHECK(vx_compiled_model_create_execution_context(compiled,&config,&contexts[adapted],&report)==VX_STATUS_OK);
        vx_compiled_model_release(compiled);
    }
    vx_model_release(model);
    CHECK(!remove(adapter_path)); /* Published revision owns its accepted package. */
    const float x[]={2,3}, expected[][2]={{8,18},{16,14}};
    VxTensorBinding input=shaped("x",VX_DTYPE_F32,x,2,1,2);
    VxGraphExecutionOptions options={.entrypoint="evaluate",.range=VX_EXECUTION_RANGE_FULL};
    for(int adapted=0;adapted<2;adapted++) {
        compact_compute_inventory(contexts[adapted],0);
        VxResult* result=NULL;
        CHECK(vx_execution_context_execute_graph(contexts[adapted],&input,1,&options,&result,&report)==VX_STATUS_OK);
        compact_compute_inventory(contexts[adapted],0);
        vx_execution_context_release(contexts[adapted]);
        output_is(result,"y",expected[adapted]); vx_result_release(result);
        mapping_file_is(weight_path,source_bytes,source_size);
    }
    free(source_bytes);
    CHECK(!remove(graph_path) && !remove(weight_path) && !rmdir(directory));
#else
    (void)runtime; (void)policy;
#endif
}

static void compact_compute_test(VxRuntime* runtime, const VxBackendPolicy* policy) {
    float x[195]; for(size_t i=0;i<195;i++) x[i]=(float)((int)(i%13)-6)*0.125f;
    const float mx[]={0.25f,-1,0.5f,2,-0.5f,0.75f,1,-0.25f};
    const float ids[]={2,0,1,2}, routes[]={0.75f,0.25f,0.5f,0.25f}, edge[]={0x1p-126f,0x1p120f};
    const int32_t tokens[]={0,16};
    VxTensorBinding inputs[]={shaped("x",VX_DTYPE_F32,x,2,3,65),shaped("mx",VX_DTYPE_F32,mx,2,2,4),
        shaped("ids",VX_DTYPE_F32,ids,2,2,2),shaped("routes",VX_DTYPE_F32,routes,2,2,2),
        shaped("tokens",VX_DTYPE_I32,tokens,1,2,0),shaped("edge",VX_DTYPE_F32,edge,2,1,2)};
    const char* outputs[]={"d0","d1","d2","emb","ml","mg","wide"};
    VxExecutionConfig config={.context=VX_CONTEXT_OPTIONS_INIT};
    VxGraphExecutionOptions options={.entrypoint="evaluate",.range=VX_EXECUTION_RANGE_FULL};
    for(int type=0;type<2;type++) {
        SafetensorsFile compact={0},reference={0};
        compact_compute_weights(&compact,&reference,type?VX_DTYPE_BF16:VX_DTYPE_F16);
        VxCompiledModel* base=compact_compute_compile(runtime,policy,&reference,compact_compute_graph);
        VxExecutionContext* ref=NULL; CHECK(vx_compiled_model_create_execution_context(base,&config,&ref,&report)==VX_STATUS_OK);
        vx_compiled_model_release(base);
        VxResult* expected=NULL; CHECK(vx_execution_context_execute_graph(ref,inputs,6,&options,&expected,&report)==VX_STATUS_OK);
        vx_execution_context_release(ref);
        for(int mixed=0;mixed<2;mixed++) {
            cJSON* graph=cJSON_Parse(compact_compute_graph); CHECK(graph);
            if(mixed) CHECK(cJSON_AddItemToArray(cJSON_GetObjectItemCaseSensitive(graph,"nodes"),cJSON_Parse(
                "{\"id\":\"copy\",\"opType\":\"Identity\",\"params\":{},\"inputs\":{\"input\":\"w\"},\"outputs\":{\"out\":{\"tensor\":\"copy\",\"dtype\":\"float32\",\"shape\":[17,65]}}}")));
            VxCompiledWeightConversions* conversions=NULL;
            CHECK(!vx_compiled_weight_conversions_create(&compact,1,NULL,0,VX_PORTABLE_BACKEND_KIND,graph,&conversions));
            CHECK(conversions->materialized_bytes==(26u+(mixed?1105u:0u))*4u);
            CHECK(!vx_compiled_weight_conversions_find(conversions,"ew"));
            CHECK(safetensors_find_tensor(&compact,"ew")->dtype==(type?VX_DTYPE_BF16:VX_DTYPE_F16));
            vx_compiled_weight_conversions_destroy(conversions);
            char* document=cJSON_PrintUnformatted(graph); CHECK(document); cJSON_Delete(graph);
            VxCompiledModel* compiled=compact_compute_compile(runtime,policy,&compact,document); cJSON_free(document);
            VxExecutionContext* contexts[2]={NULL,NULL};
            for(int i=0;i<2;i++) CHECK(vx_compiled_model_create_execution_context(compiled,&config,&contexts[i],&report)==VX_STATUS_OK);
            vx_compiled_model_release(compiled);
            for(int i=0;i<2;i++) {
                compact_compute_inventory(contexts[i],mixed);
                VxResult* actual=NULL; CHECK(vx_execution_context_execute_graph(contexts[i],inputs,6,&options,&actual,&report)==VX_STATUS_OK);
                compact_compute_inventory(contexts[i],mixed); vx_execution_context_release(contexts[i]);
                for(size_t output=0;output<7;output++) {
                    float a[130],b[130]; size_t an=0,bn=0;
                    CHECK(vx_result_read(actual,outputs[output],a,sizeof(a),&an,&report)==VX_STATUS_OK);
                    CHECK(vx_result_read(expected,outputs[output],b,sizeof(b),&bn,&report)==VX_STATUS_OK && an==bn);
                    for(size_t j=0;j<an/sizeof(float);j++) CHECK(isfinite(a[j]) && fabsf(a[j]-b[j])<=1e-5f*(1.0f+fabsf(b[j])));
                }
                vx_result_release(actual);
            }
        }
        vx_result_release(expected); safetensors_free(&compact); safetensors_free(&reference);
    }
    compact_compute_adapter_test(runtime,policy);
    puts("compact dense/MoE: layouts, BF16 range, LoRA, shared consumers, memory and ownership passed");
}
