static int cli_checkpoint_read(const char* path,VolvoxaiV1TrainerCheckpoint* checkpoint) {
    void* bytes=NULL;size_t size=0;
    int failed=read_whole_file(path,&bytes,&size) || !size;
    if(!failed)failed=volvoxai_v1_trainer_checkpoint_decode(checkpoint,bytes,size)!=SYNURANG_LITE_OK;
    free(bytes);
    if(failed)fprintf(stderr,"Invalid TrainerCheckpoint protobuf file: %s\n",path);
    return failed?-1:0;
}

static int cli_checkpoint_save(VxCallClient* client,int64_t id,const char* path,cJSON* report) {
    VolvoxaiV1ExportTrainerCheckpointRequest request;
    VolvoxaiV1TrainerCheckpointInfo response;
    uint8_t *payload=NULL,*encoded=NULL;int32_t length=0;size_t size=0;int result=-1;
    volvoxai_v1_export_trainer_checkpoint_request_init(&request);
    volvoxai_v1_trainer_checkpoint_info_init(&response);
    request.field_trainer_id=id;
    VX_CALL_MESSAGE(client,VX_RPC_VX_TRAINING_SERVICE_EXPORT_TRAINER_CHECKPOINT,volvoxai_v1_export_trainer_checkpoint_request,&request,payload,length);
    if(!payload || volvoxai_v1_trainer_checkpoint_info_decode(&response,payload,(size_t)length)!=SYNURANG_LITE_OK)goto done;
    cJSON_AddItemToObject(report,"checkpointReport",json_build_report(response.field_report));
    if(!report_ok("ExportTrainerCheckpoint",response.field_report) || !response.field_checkpoint)goto done;
    if(volvoxai_v1_trainer_checkpoint_encode(response.field_checkpoint,&encoded,&size)!=SYNURANG_LITE_OK)goto done;
    if(cli_write_new(path,encoded,size))goto done;
    cJSON_AddNumberToObject(report,"checkpointBytes",(double)size);result=0;
done:
    if(encoded)response._allocator->deallocate(response._allocator->context,encoded);
    if(payload)vx_call_free(client,payload);
    volvoxai_v1_trainer_checkpoint_info_free(&response);
    volvoxai_v1_export_trainer_checkpoint_request_free(&request);
    return result;
}

static int cli_training_state(VxCallClient* client,int64_t id,cJSON* report,const char* key,int* private_updates) {
    VolvoxaiV1TrainerRef request;VolvoxaiV1TrainerState state;
    uint8_t* payload=NULL;int32_t length=0;int result=-1;
    volvoxai_v1_trainer_ref_init(&request);volvoxai_v1_trainer_state_init(&state);
    request.field_trainer_id=id;
    VX_CALL_MESSAGE(client,VX_RPC_VX_TRAINING_SERVICE_GET_TRAINER_STATE,volvoxai_v1_trainer_ref,&request,payload,length);
    if(!payload || volvoxai_v1_trainer_state_decode(&state,payload,(size_t)length)!=SYNURANG_LITE_OK)goto done;
    cJSON* value=cJSON_CreateObject();
    if(!value)goto done;
    cJSON_AddItemToObject(report,key,value);
    cJSON_AddItemToObject(value,"report",json_build_report(state.field_report));
    if(!report_ok("GetTrainerState",state.field_report))goto done;
    if(private_updates)*private_updates=state.field_has_private_updates;
    cJSON_AddBoolToObject(value,"hasPrivateUpdates",state.field_has_private_updates);
    cJSON_AddNumberToObject(value,"accumulatedMicrobatches",state.field_accumulated_microbatches);
    cJSON_AddNumberToObject(value,"optimizerStep",(double)state.field_optimizer_step);
    cJSON_AddNumberToObject(value,"optimizerBytes",(double)state.field_optimizer_bytes);
    cJSON_AddNumberToObject(value,"gradientBytes",(double)state.field_gradient_bytes);
    cJSON_AddNumberToObject(value,"qatWorkspaceHighWaterBytes",(double)state.field_qat_workspace_high_water_bytes);
    cJSON_AddItemToObject(value,"qat",state.field_qat?cli_qat_json(state.field_qat):cJSON_CreateNull());
    if(state.field_optimizer_config) {
        cJSON* config=cJSON_CreateObject();const VolvoxaiV1TrainerOptimizerOptions* c=state.field_optimizer_config;
        if(!config)goto done;
        cJSON_AddStringToObject(config,"kind",volvoxai_v1_training_optimizer_kind_name(c->field_kind));
        cJSON_AddNumberToObject(config,"learningRate",c->field_learning_rate);
        cJSON_AddNumberToObject(config,"beta1",c->field_beta1);cJSON_AddNumberToObject(config,"beta2",c->field_beta2);
        cJSON_AddNumberToObject(config,"epsilon",c->field_epsilon);cJSON_AddNumberToObject(config,"weightDecay",c->field_weight_decay);
        cJSON_AddNumberToObject(config,"maxGradientNorm",c->field_max_gradient_norm);
        cJSON_AddItemToObject(value,"optimizerConfig",config);
    }
    result=0;
done:
    if(payload)vx_call_free(client,payload);
    volvoxai_v1_trainer_state_free(&state);volvoxai_v1_trainer_ref_free(&request);return result;
}

static int cli_reset_empty_window(VxCallClient* client,int64_t id,cJSON* report) {
    VolvoxaiV1TrainerRef request;VolvoxaiV1TrainerState state;
    uint8_t* payload=NULL;int32_t length=0;int result=-1;
    volvoxai_v1_trainer_ref_init(&request);volvoxai_v1_trainer_state_init(&state);
    request.field_trainer_id=id;
    VX_CALL_MESSAGE(client,VX_RPC_VX_TRAINING_SERVICE_RESET_TRAINER_ACCUMULATION,volvoxai_v1_trainer_ref,&request,payload,length);
    if(payload && volvoxai_v1_trainer_state_decode(&state,payload,(size_t)length)==SYNURANG_LITE_OK) {
        cJSON_AddItemToObject(report,"emptyWindowResetReport",json_build_report(state.field_report));
        if(report_ok("ResetTrainerAccumulation",state.field_report))result=0;
    }
    if(payload)vx_call_free(client,payload);
    volvoxai_v1_trainer_state_free(&state);volvoxai_v1_trainer_ref_free(&request);return result;
}
