/* Whole-prefix generation deliberately compares with an independently
 * computed token/position oracle and exercises public context transactions. */
static void full_prefix_generation_test(VxRuntime* runtime, const char* path, const VxBackendPolicy* policy) {
    const char* graph =
      "{\"format\":\"volvox-graph/v1\",\"dimensions\":{\"S\":{\"min\":1,\"max\":8}},\"inputs\":{"
      "\"tokens\":{\"dtype\":\"int32\",\"shape\":[1,\"S\"]},\"keep\":{\"dtype\":\"int32\",\"shape\":[1,\"S\"]},"
      "\"positions\":{\"dtype\":\"int32\",\"shape\":[\"S\"]},\"sequence\":{\"dtype\":\"int32\",\"shape\":[\"S\"]},"
      "\"chunks\":{\"dtype\":\"int32\",\"shape\":[1,\"S\",\"S\"]},\"mrope\":{\"dtype\":\"int32\",\"shape\":[3,\"S\"]},"
      "\"image_ids\":{\"dtype\":\"int32\",\"shape\":[\"S\"]},\"image_mask\":{\"dtype\":\"int32\",\"shape\":[1,\"S\"]},"
      "\"embedding\":{\"dtype\":\"float32\",\"shape\":[2,2]},\"position_weights\":{\"dtype\":\"float32\",\"shape\":[8,2]}},"
      "\"nodes\":[{\"id\":\"embedding\",\"opType\":\"Embedding\",\"inputs\":{\"input\":\"tokens\",\"weight\":\"embedding\"},"
      "\"outputs\":{\"out\":{\"tensor\":\"token_scores\",\"dtype\":\"float32\",\"shape\":[1,\"S\",2]}},\"params\":{}},"
      "{\"id\":\"positions\",\"opType\":\"Embedding\",\"inputs\":{\"input\":\"positions\",\"weight\":\"position_weights\"},"
      "\"outputs\":{\"out\":{\"tensor\":\"position_scores\",\"dtype\":\"float32\",\"shape\":[\"S\",2]}},\"params\":{}},"
      "{\"id\":\"scores\",\"opType\":\"Add\",\"inputs\":{\"a\":\"token_scores\",\"b\":\"position_scores\"},"
      "\"outputs\":{\"out\":{\"tensor\":\"logits\",\"dtype\":\"float32\",\"shape\":[1,\"S\",2]}},\"params\":{}},"
      "{\"id\":\"select\",\"opType\":\"ArgMax\",\"inputs\":{\"input\":\"logits\"},"
      "\"outputs\":{\"out\":{\"tensor\":\"ids\",\"dtype\":\"int32\",\"shape\":[1,\"S\"]}},\"params\":{\"axis\":2,\"keepdims\":false,\"select_last_index\":0}}],"
      "\"outputs\":[\"ids\",\"logits\"],\"execution\":{\"stateless\":{},\"invariant_inputs\":[\"embedding\",\"position_weights\"],"
      "\"entrypoints\":[{\"name\":\"evaluate\",\"effect\":1,\"allowed_ranges\":[1],"
      "\"inputs\":[\"tokens\",\"keep\",\"positions\",\"sequence\",\"chunks\",\"mrope\",\"image_ids\",\"image_mask\",\"embedding\",\"position_weights\"],\"outputs\":[\"ids\",\"logits\"]}],"
      "\"generation\":{\"initialize_entrypoint\":\"evaluate\",\"transition_entrypoint\":\"evaluate\",\"token_input\":\"tokens\",\"keep_input\":\"keep\","
      "\"token_output\":\"ids\",\"logits_output\":\"logits\",\"logits_axis\":2,\"id_domain_size\":2,\"output_name\":\"generated\",\"sequence_axis\":1,"
      "\"full_prefix\":{\"position_input\":\"positions\",\"sequence_input\":\"sequence\",\"chunk_mask_input\":\"chunks\",\"chunk_size\":2,"
      "\"multimodal_position_input\":\"mrope\",\"zero_append_inputs\":[\"image_ids\",\"image_mask\"]}}}}";
    FILE* fixture = fopen(path, "wb"); CHECK(fixture);
    CHECK(fwrite(graph, 1, strlen(graph), fixture) == strlen(graph)); CHECK(!fclose(fixture));
    VxModelSource source = VX_MODEL_SOURCE_INIT; source.graph_path = path;
    VxModel* model = NULL; VxCompiledModel* compiled = NULL;
    CHECK(vx_runtime_load_model(runtime, &source, &model, &report) == VX_STATUS_OK);
    CHECK(vx_model_compile_options_revision(model, policy, 0, NULL, VX_PACKED_ARITHMETIC_F32, &compiled, &report) == VX_STATUS_OK);
    int32_t tokens[]={0,1}, keep[]={1,1}, positions[]={0,1}, sequence[]={0,1}, chunks[]={1,1,1,1},
        mrope[]={0,1,0,2,0,3}, image_ids[]={0,0}, image_mask[]={0,0};
    float embedding[]={0,2,2,0}, position_weights[]={0,0,0,3,0,0,3,0,0,0,0,3,0,0,0,0};
    VxTensorBinding inputs[]={shaped("tokens",VX_DTYPE_I32,tokens,2,1,2),shaped("keep",VX_DTYPE_I32,keep,2,1,2),
        shaped("positions",VX_DTYPE_I32,positions,1,2,0),shaped("sequence",VX_DTYPE_I32,sequence,1,2,0),
        shaped("chunks",VX_DTYPE_I32,chunks,2,2,2),shaped("mrope",VX_DTYPE_I32,mrope,2,3,2),
        shaped("image_ids",VX_DTYPE_I32,image_ids,1,2,0),shaped("image_mask",VX_DTYPE_I32,image_mask,2,1,2),
        shaped("embedding",VX_DTYPE_F32,embedding,2,2,2),shaped("position_weights",VX_DTYPE_F32,position_weights,2,8,2)};
    inputs[4].rank=3; inputs[4].shape[0]=1; inputs[4].shape[1]=inputs[4].shape[2]=2;
    const int32_t expected[]={1,0,0,1,1};
    for (int owned=0;owned<2;++owned) for (int chunk=1;chunk<=5;chunk+=4) {
        VxExecutionConfig config={.context=VX_CONTEXT_OPTIONS_INIT,.autoregressive=1,.max_steps=5,.sampling=VX_SAMPLING_GREEDY};
        VxExecutionContext* context=NULL; VxResult* result=NULL,*retained=NULL;
        CHECK(vx_compiled_model_create_execution_context(compiled,&config,&context,&report)==VX_STATUS_OK);
        VxGenerationOptions options={.initial_inputs_present=1,.max_steps=(uint32_t)chunk}; options.tensors.owned_output=owned;
        keep[1]=0;
        CHECK(vx_execution_context_generate(context,inputs,10,&options,&result,&report)==VX_STATUS_INVALID_ARGUMENT);
        CHECK(inspect(context).state==VX_EXECUTION_CONTEXT_STATE_READY && !inspect(context).initialized);
        vx_result_release(result);result=NULL;keep[1]=1;
        int emitted=0;
        while(emitted<5) {
            options.initial_inputs_present=emitted==0;
            CHECK(vx_execution_context_generate(context,emitted?NULL:inputs,emitted?0:10,&options,&result,&report)==VX_STATUS_OK);
            int32_t ids[5];size_t bytes=0;
            CHECK(vx_result_read(result,"generated",ids,sizeof(ids),&bytes,&report)==VX_STATUS_OK);
            CHECK(bytes && bytes<=sizeof(expected)-(size_t)emitted*4u && !memcmp(ids,expected+emitted,bytes));
            emitted+=(int)(bytes/4u);
            VxExecutionInfo info;vx_result_execution_info(result,&info);
            CHECK(info.total_steps==(uint64_t)emitted && info.mode==VX_DECODE_MODE_DISABLED && !info.has_cache);
            if(!retained) retained=result;else vx_result_release(result);result=NULL;
        }
        CHECK(inspect(context).state==VX_EXECUTION_CONTEXT_STATE_COMPLETED);
        CHECK(tokens[0]==0&&tokens[1]==1&&positions[1]==1&&mrope[5]==3);
        vx_execution_context_release(context);
        int32_t ids[5];size_t bytes=0;
        CHECK(vx_result_read(retained,"generated",ids,sizeof(ids),&bytes,&report)==VX_STATUS_OK);
        CHECK(!memcmp(ids,expected,bytes));vx_result_release(retained);
    }
    /* The stop ID is included; the next context starts independently. */
    int32_t eos=1; VxExecutionContext* context=NULL; VxResult* result=NULL;
    VxExecutionConfig config={.context=VX_CONTEXT_OPTIONS_INIT,.autoregressive=1,.max_steps=5,
        .sampling=VX_SAMPLING_GREEDY,.eos_ids=&eos,.eos_id_count=1};
    CHECK(vx_compiled_model_create_execution_context(compiled,&config,&context,&report)==VX_STATUS_OK);
    VxGenerationOptions options={.initial_inputs_present=1,.max_steps=5};uint64_t reservation=0;
    CHECK(vx_execution_context_reserve_generation(context,inputs,10,&options,&reservation,&report)==VX_STATUS_OK);
    CHECK(inspect(context).state==VX_EXECUTION_CONTEXT_STATE_QUEUED);
    CHECK(vx_execution_context_run_reserved_generation(context,reservation,inputs,10,&options,&result,&report)==VX_STATUS_OK);
    VxExecutionInfo info;vx_result_execution_info(result,&info);
    CHECK(info.total_steps==1&&info.stop_reason==VX_EXECUTION_STOP_EOS&&info.state==VX_EXECUTION_CONTEXT_STATE_COMPLETED);
    vx_execution_context_finish_reserved(context,reservation,0);vx_result_release(result);vx_execution_context_release(context);
    /* A byte budget trims a chunk before execution. The engine retains the
     * initial values, even after the caller overwrites all source arrays. */
    config.eos_id_count=0;config.eos_ids=NULL;config.max_steps=9;context=NULL;result=NULL;
    CHECK(vx_compiled_model_create_execution_context(compiled,&config,&context,&report)==VX_STATUS_OK);
    options=(VxGenerationOptions){.initial_inputs_present=1,.max_steps=8};
    CHECK(vx_execution_context_generate(context,inputs,10,&options,&result,&report)==VX_STATUS_INVALID_ARGUMENT);
    CHECK(!inspect(context).initialized && inspect(context).state==VX_EXECUTION_CONTEXT_STATE_READY);
    vx_result_release(result);result=NULL;
    options.max_output_bytes=4;
    CHECK(vx_execution_context_generate(context,inputs,10,&options,&result,&report)==VX_STATUS_OK);
    CHECK(inspect(context).total_steps==1);vx_result_release(result);result=NULL;
    memset(embedding,0,sizeof(embedding));positions[1]=7;tokens[1]=0;
    options=(VxGenerationOptions){.max_steps=3};
    CHECK(vx_execution_context_generate(context,NULL,0,&options,&result,&report)==VX_STATUS_OK);
    int32_t retained_ids[3];size_t retained_bytes=0;
    CHECK(vx_result_read(result,"generated",retained_ids,sizeof(retained_ids),&retained_bytes,&report)==VX_STATUS_OK);
    CHECK(retained_bytes==sizeof(retained_ids)&&!memcmp(retained_ids,expected+1,retained_bytes));
    vx_result_release(result);result=NULL;
    options=(VxGenerationOptions){.max_steps=1,.tensors={.select_outputs=1}};
    CHECK(vx_execution_context_generate(context,NULL,0,&options,&result,&report)==VX_STATUS_OK);
    CHECK(vx_result_output_count(result)==0 && inspect(context).total_steps==5);
    vx_result_release(result);result=NULL;
    CHECK(vx_execution_context_cancel_execution(context,&report)==VX_STATUS_OK);
    CHECK(inspect(context).state==VX_EXECUTION_CONTEXT_STATE_CANCELLED);
    CHECK(vx_execution_context_generate(context,NULL,0,&options,&result,&report)!=VX_STATUS_OK && !result);
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);vx_model_release(model);
}

static void llm_workspace_budget_test(VxRuntime* runtime, const char* path, const VxBackendPolicy* policy) {
    /* No published state tensor: the 512x512 recurrent workspace cannot be
     * deduced from the small q/k/v/output buffers. */
    const char* graph = "{\"format\":\"volvox-graph/v1\",\"dimensions\":{},\"inputs\":{"
      "\"q\":{\"dtype\":\"float32\",\"shape\":[1,1,1,512]},\"k\":{\"dtype\":\"float32\",\"shape\":[1,1,1,512]},"
      "\"v\":{\"dtype\":\"float32\",\"shape\":[1,1,1,512]},\"a\":{\"dtype\":\"float32\",\"shape\":[1,1,1]},"
      "\"b\":{\"dtype\":\"float32\",\"shape\":[1,1,1]},\"a_log\":{\"dtype\":\"float32\",\"shape\":[1]},\"dt_bias\":{\"dtype\":\"float32\",\"shape\":[1]}},"
      "\"nodes\":[{\"id\":\"delta\",\"opType\":\"GatedDeltaNet\",\"inputs\":{\"q\":\"q\",\"k\":\"k\",\"v\":\"v\",\"a\":\"a\",\"b\":\"b\",\"a_log\":\"a_log\",\"dt_bias\":\"dt_bias\"},"
      "\"outputs\":{\"out\":{\"tensor\":\"y\",\"dtype\":\"float32\",\"shape\":[1,1,1,512]}},\"params\":{}}],"
      "\"outputs\":[\"y\"],\"execution\":{\"stateless\":{},\"entrypoints\":[{\"name\":\"evaluate\",\"effect\":1,\"allowed_ranges\":[1],"
      "\"inputs\":[\"q\",\"k\",\"v\",\"a\",\"b\",\"a_log\",\"dt_bias\"],\"outputs\":[\"y\"]}]}}";
    FILE* fixture=fopen(path,"wb");CHECK(fixture);
    CHECK(fwrite(graph,1,strlen(graph),fixture)==strlen(graph));CHECK(!fclose(fixture));
    VxModelSource source=VX_MODEL_SOURCE_INIT;source.graph_path=path;
    VxModel* model=NULL;VxCompiledModel* compiled=NULL;VxExecutionContext* context=NULL;
    CHECK(vx_runtime_load_model(runtime,&source,&model,&report)==VX_STATUS_OK);
    CHECK(vx_model_compile_options_revision(model,policy,0,NULL, VX_PACKED_ARITHMETIC_F32,&compiled,&report)==VX_STATUS_OK);
    CHECK(strstr(report.route_evidence,"max_typed_scratch=1048576"));
    VxExecutionConfig config={.context=VX_CONTEXT_OPTIONS_INIT,.max_memory_bytes=1048575};
    CHECK(vx_compiled_model_create_execution_context(compiled,&config,&context,&report)==VX_STATUS_OUT_OF_MEMORY && !context);
    config.max_memory_bytes=UINT64_C(1)<<32;
    CHECK(vx_compiled_model_create_execution_context(compiled,&config,&context,&report)==VX_STATUS_OK);
    vx_execution_context_release(context);vx_compiled_model_release(compiled);vx_model_release(model);
}
