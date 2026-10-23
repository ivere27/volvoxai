static void print_ptq_help(const char* argv0) {
    printf("Usage: %s ptq <model-dir|graph.json> [options]\n",argv0);
    printf("  --weight-bits <2|3|4|8>     Default weight format (default 4).\n");
    printf("  --group-size <0|32|64|128|256> (default 32; static default 0).\n");
    printf("  --algorithm <rtn|mse-clip|importance> (default mse-clip; static rtn).\n");
    printf("  --rules <rules.json>         Array of protobuf QuantizationRule objects.\n");
    printf("  --calibration <batches.json> Stream typed raw input batches once.\n");
    printf("  --static-activations <i8|u8> Static asymmetric byte-domain regions.\n");
    printf("  --weights <file> --threads <n> --report-json <new-file>\n");
    printf("  --cpu | --cuda              Exact calibration backend (default CPU).\n");
    printf("  --out <directory>           New graph.json and model.safetensors.\n");
    printf("  --dry-run                   Inspect resolved policy without calibration/export.\n");
}

static cJSON* cli_plan_json(const VolvoxaiV1QuantizationPlanInfo* info) {
    cJSON *root=cJSON_CreateObject(),*nodes=cJSON_CreateArray();
    if (!root || !nodes || !info) {cJSON_Delete(root);cJSON_Delete(nodes);return NULL;}
    cJSON_AddBoolToObject(root,"requiresCalibration",info->field_requires_calibration);
    cJSON_AddBoolToObject(root,"ready",info->field_ready);
    json_add_bytes_copy(root,"calibrationBackend",&info->field_calibration_backend);
    cJSON_AddNumberToObject(root,"floatNodes",(double)info->field_float_nodes);
    if (info->field_calibration) {
        const VolvoxaiV1CalibrationState* c=info->field_calibration;
        cJSON* state=cJSON_CreateObject();
        if (!state) {cJSON_Delete(root);cJSON_Delete(nodes);return NULL;}
        cJSON_AddNumberToObject(state,"batches",(double)c->field_batches);
        cJSON_AddNumberToObject(state,"samples",(double)c->field_samples);
        cJSON_AddNumberToObject(state,"importanceTensors",(double)c->field_importance_tensors);
        cJSON_AddNumberToObject(state,"affineCount",(double)c->field_affines.len);
        cJSON_AddBoolToObject(state,"coverageComplete",c->field_coverage && c->field_coverage->field_complete);
        if(c->field_coverage)json_add_bytes_copy(state,"logicalFingerprint",&c->field_coverage->field_logical_fingerprint);
        cJSON_AddItemToObject(root,"calibration",state);
    }
    for (size_t i=0;i<info->field_resolved_nodes.len;i++) {
        const VolvoxaiV1QuantizationNodeInfo* n=&info->field_resolved_nodes.data[i];
        cJSON* node=cJSON_CreateObject();
        if (!node) {cJSON_Delete(root);cJSON_Delete(nodes);return NULL;}
        json_add_bytes_copy(node,"nodeId",&n->field_node_id);
        if (n->which_action==2) cJSON_AddItemToObject(node,"weights",cli_weights_json(n->field_weights));
        else cJSON_AddItemToObject(node,"keepFloat",cJSON_CreateObject());
        cJSON_AddItemToArray(nodes,node);
    }
    cJSON_AddItemToObject(root,"resolvedNodes",nodes);
    return root;
}

static int command_ptq(VxCallClient* client,int argc,char** argv) {
    RunOptions options={0};ModelPaths paths;
    const char *rules_path=NULL,*batches_path=NULL,*out=NULL,*static_name=NULL;
    uint32_t bits=4,group=32;int group_set=0,algorithm_set=0,dry_run=0,result=1;
    VolvoxaiV1WeightAlgorithm algorithm=VOLVOXAI_V1_WEIGHT_ALGORITHM_MSE_CLIP;
    VolvoxaiV1CreateQuantizationPlanRequest request;
    VolvoxaiV1QuantizationPlanHandle plan;
    VolvoxaiV1QuantizationPlanResponse inspected;
    uint8_t* payload=NULL;int32_t length=0;int64_t plan_id=0;
    cJSON *rules=NULL,*batches=NULL,*report=cJSON_CreateObject(),*progress=cJSON_CreateArray();
    char output_graph[PATH_MAX]={0},output_weight[PATH_MAX]={0};

    volvoxai_v1_create_quantization_plan_request_init(&request);
    volvoxai_v1_quantization_plan_handle_init(&plan);
    volvoxai_v1_quantization_plan_response_init(&inspected);
    cli_json_failed=0;
    if (argc<3 || !strcmp(argv[2],"--help") || !strcmp(argv[2],"-h")) {
        print_ptq_help(argv[0]);result=argc<3?2:0;goto done;
    }
    if (!report || !progress) goto done;
    cJSON_AddStringToObject(report,"format","volvoxai-native-ptq/v1");
    cJSON_AddItemToObject(report,"calibrationProgress",progress);progress=NULL;
    for (int i=3;i<argc;i++) {
        const char* arg=argv[i];int parsed;
        if (!strcmp(arg,"--dry-run")) {dry_run=1;continue;}
        if (!strcmp(arg,"--weight-bits") || !strcmp(arg,"--group-size")) {
            if (++i>=argc || parse_nonnegative_int(argv[i],0,&parsed)) goto usage;
            if (!strcmp(arg,"--weight-bits")) {if(parsed!=2&&parsed!=3&&parsed!=4&&parsed!=8)goto usage;bits=(uint32_t)parsed;}
            else {if(parsed!=0&&parsed!=32&&parsed!=64&&parsed!=128&&parsed!=256)goto usage;group=(uint32_t)parsed;group_set=1;}
        } else if (!strcmp(arg,"--algorithm")) {
            if (++i>=argc || cli_algorithm(argv[i],1,&algorithm))goto usage;algorithm_set=1;
        } else if (!strcmp(arg,"--rules") || !strcmp(arg,"--calibration") || !strcmp(arg,"--out") || !strcmp(arg,"--static-activations")) {
            const char** dest=!strcmp(arg,"--rules")?&rules_path:!strcmp(arg,"--calibration")?&batches_path:!strcmp(arg,"--out")?&out:&static_name;
            if (++i>=argc || *dest || !argv[i][0])goto usage;*dest=argv[i];
        } else {
            /* Backend controls the private calibration executor only. */
            if (strcmp(arg,"--weights") && strcmp(arg,"--threads") && strcmp(arg,"--report-json") &&
                strcmp(arg,"--cpu") && strcmp(arg,"--cuda"))goto usage;
            if (parse_run_option(argc,argv,&i,&options)!=1)goto usage;
        }
    }
    if ((!out && !dry_run) || (static_name && strcmp(static_name,"i8") && strcmp(static_name,"u8")))goto usage;
    if (static_name) {if(!group_set)group=0;if(!algorithm_set)algorithm=VOLVOXAI_V1_WEIGHT_ALGORITHM_RTN;}
    if (options.report_json && cli_new_output(options.report_json))goto done;
    if (resolve_model_paths(argv[2],&paths))goto done;
    if (!options.weight_path_count && paths.default_weights[0])options.weight_paths[options.weight_path_count++]=paths.default_weights;
    if (!options.weight_path_count) {fprintf(stderr,"PTQ requires --weights or model.safetensors.\n");goto done;}
    if (out && !dry_run) {
        if (!is_directory(out) && mkdir(out,0777)) {fprintf(stderr,"Cannot create output directory: %s\n",out);goto done;}
        if (join_path(output_graph,sizeof(output_graph),out,"graph.json") || join_path(output_weight,sizeof(output_weight),out,"model.safetensors") || cli_new_output(output_graph) || cli_new_output(output_weight))goto done;
        if (options.report_json && (!strcmp(options.report_json,output_graph) || !strcmp(options.report_json,output_weight)))goto usage;
    }
    request.which_source=1;
    request.field_files=request._allocator->allocate(request._allocator->context,sizeof(*request.field_files));
    if (!request.field_files)goto done;
    volvoxai_v1_model_files_init_with_allocator(request.field_files,request._allocator);
    if(assign_text(request._allocator,&request.field_files->field_graph_path,paths.graph))goto done;
    for(size_t i=0;i<options.weight_path_count;i++) {
        SynurangLiteBytes* slot=volvoxai_v1_model_files_add_weight_paths(request.field_files);
        if(!slot || assign_text(request._allocator,slot,options.weight_paths[i]))goto done;
    }
    request.field_config=request._allocator->allocate(request._allocator->context,sizeof(*request.field_config));
    if(!request.field_config)goto done;
    volvoxai_v1_quantization_config_init_with_allocator(request.field_config,request._allocator);
    VolvoxaiV1QuantizationConfig* config=request.field_config;
    config->field_weights=request._allocator->allocate(request._allocator->context,sizeof(*config->field_weights));
    if(!config->field_weights)goto done;
    volvoxai_v1_weight_quantization_init_with_allocator(config->field_weights,request._allocator);
    config->field_weights->field_algorithm=algorithm;
    config->field_weights->field_format=request._allocator->allocate(request._allocator->context,sizeof(*config->field_weights->field_format));
    if(!config->field_weights->field_format)goto done;
    volvoxai_v1_weight_format_init_with_allocator(config->field_weights->field_format,request._allocator);
    config->field_weights->field_format->field_bits=bits;config->field_weights->field_format->field_group_size=group;
    if(rules_path && (!(rules=cli_read_json(rules_path)) || cli_parse_rules(rules,config)))goto usage;
    if(static_name) {
        config->field_static_activations=request._allocator->allocate(request._allocator->context,sizeof(*config->field_static_activations));
        if(!config->field_static_activations)goto done;
        volvoxai_v1_static_activations_init_with_allocator(config->field_static_activations,request._allocator);
        config->field_static_activations->field_dtype=!strcmp(static_name,"i8")?VOLVOXAI_V1_DATA_TYPE_I8:VOLVOXAI_V1_DATA_TYPE_U8;
        config->field_static_activations->field_scheme=VOLVOXAI_V1_AFFINE_SCHEME_ASYMMETRIC;
    }
    if(batches_path && !(batches=cli_read_batches(batches_path,0)))goto usage;
    request.field_calibration=request._allocator->allocate(request._allocator->context,sizeof(*request.field_calibration));
    if(!request.field_calibration)goto done;
    volvoxai_v1_calibration_options_init_with_allocator(request.field_calibration,request._allocator);
    request.field_calibration->field_cpu_threads=(uint32_t)options.cpu_threads;
    if(options.backend && assign_text(request._allocator,&request.field_calibration->field_backend,options.backend))goto done;
    if(batches) for(const cJSON* b=cJSON_GetObjectItemCaseSensitive(batches,"batches")->child;b;b=b->next) {
        const cJSON* p=cli_json_field(b,"profile_name","profileName");const char* name=p&&p->valuestring[0]?p->valuestring:"default";
        size_t i;for(i=0;i<request.field_calibration->field_profile_names.len;i++) {
            SynurangLiteBytes* v=&request.field_calibration->field_profile_names.data[i];
            if(v->len==strlen(name)&&!memcmp(v->data,name,v->len))break;
        }
        if(i==request.field_calibration->field_profile_names.len) {
            SynurangLiteBytes* v=volvoxai_v1_calibration_options_add_profile_names(request.field_calibration);
            if(!v||assign_text(request._allocator,v,name))goto done;
        }
    }
    VX_CALL_MESSAGE(client,VX_RPC_VX_QUANTIZATION_SERVICE_CREATE_QUANTIZATION_PLAN,volvoxai_v1_create_quantization_plan_request,&request,payload,length);
    if(!payload || volvoxai_v1_quantization_plan_handle_decode(&plan,payload,(size_t)length)!=SYNURANG_LITE_OK)goto transport;
    plan_id=plan.field_quantization_plan_id;
    cJSON_AddItemToObject(report,"createReport",json_build_report(plan.field_report));
    if(!report_ok("CreateQuantizationPlan",plan.field_report))goto done;
    vx_call_free(client,payload);payload=NULL;
    if(!plan.field_info)goto done;
    printf("PTQ resolved %zu nodes; calibration=%s ready=%s\n",plan.field_info->field_resolved_nodes.len,plan.field_info->field_requires_calibration?"required":"no",plan.field_info->field_ready?"yes":"no");
    if(!dry_run && plan.field_info->field_requires_calibration && !batches) {
        fprintf(stderr,"Resolved policy requires --calibration batches.json.\n");goto done;
    }
    if(!dry_run && batches && !plan.field_info->field_requires_calibration) {
        fprintf(stderr,"Resolved policy does not accept calibration; omit --calibration.\n");goto done;
    }
    if(!dry_run && batches) {
        size_t batch_index=0;
        for(const cJSON* b=cJSON_GetObjectItemCaseSensitive(batches,"batches")->child;b;b=b->next,batch_index++) {
            const char *sample,*profile;uint32_t count;char default_name[64];
            VolvoxaiV1CalibrateQuantizationPlanRequest call;VolvoxaiV1CalibrationProgress ack;
            volvoxai_v1_calibrate_quantization_plan_request_init(&call);volvoxai_v1_calibration_progress_init(&ack);
            if(cli_batch(b,batches_path,0,&options,NULL,&sample,&profile,&count)) {volvoxai_v1_calibrate_quantization_plan_request_free(&call);goto done;}
            snprintf(default_name,sizeof(default_name),"batch-%zu",batch_index);
            call.field_quantization_plan_id=plan_id;call.field_sample_count=count;
            int failed=assign_text(call._allocator,&call.field_sample_name,sample?sample:default_name) || assign_text(call._allocator,&call.field_profile_name,profile);
            if(options.input_count!=plan.field_info->field_calibration_inputs.len)failed=1;
            for(size_t i=0;i<options.input_count && !failed;i++) {
                const VolvoxaiV1TensorSpec* spec=find_tensor_spec(plan.field_info->field_calibration_inputs.data,plan.field_info->field_calibration_inputs.len,options.inputs[i].name);
                VolvoxaiV1Tensor* t=volvoxai_v1_calibrate_quantization_plan_request_add_inputs(&call);
                if(!spec||!t||fill_tensor_from_binding(&options.inputs[i],spec,t,call._allocator))failed=1;
            }
            if(!failed)VX_CALL_MESSAGE(client,VX_RPC_VX_QUANTIZATION_SERVICE_CALIBRATE_QUANTIZATION_PLAN,volvoxai_v1_calibrate_quantization_plan_request,&call,payload,length);
            volvoxai_v1_calibrate_quantization_plan_request_free(&call);
            if(failed || !payload || volvoxai_v1_calibration_progress_decode(&ack,payload,(size_t)length)!=SYNURANG_LITE_OK) {volvoxai_v1_calibration_progress_free(&ack);goto transport;}
            cJSON* item=cJSON_CreateObject();
            if(!item){volvoxai_v1_calibration_progress_free(&ack);goto done;}
            cJSON_AddStringToObject(item,"sampleName",sample?sample:default_name);
            cJSON_AddNumberToObject(item,"batches",(double)ack.field_batches);cJSON_AddNumberToObject(item,"samples",(double)ack.field_samples);
            cJSON_AddItemToObject(item,"report",json_build_report(ack.field_report));
            cJSON_AddItemToArray(cJSON_GetObjectItemCaseSensitive(report,"calibrationProgress"),item);
            int ok=report_ok("CalibrateQuantizationPlan",ack.field_report);volvoxai_v1_calibration_progress_free(&ack);
            vx_call_free(client,payload);payload=NULL;if(!ok)goto done;
        }
    }
    {
        VolvoxaiV1QuantizationPlanRef ref;volvoxai_v1_quantization_plan_ref_init(&ref);ref.field_quantization_plan_id=plan_id;
        VX_CALL_MESSAGE(client,VX_RPC_VX_QUANTIZATION_SERVICE_INSPECT_QUANTIZATION_PLAN,volvoxai_v1_quantization_plan_ref,&ref,payload,length);volvoxai_v1_quantization_plan_ref_free(&ref);
        if(!payload||volvoxai_v1_quantization_plan_response_decode(&inspected,payload,(size_t)length)!=SYNURANG_LITE_OK)goto transport;
        cJSON_AddItemToObject(report,"inspectReport",json_build_report(inspected.field_report));
        if(!report_ok("InspectQuantizationPlan",inspected.field_report))goto done;
        cJSON_AddItemToObject(report,"plan",cli_plan_json(inspected.field_info));vx_call_free(client,payload);payload=NULL;
    }
    if(!dry_run) {
        VolvoxaiV1ExportQuantizedModelRequest call;VolvoxaiV1ModelFiles files;VolvoxaiV1QuantizedModelInfo exported;
        volvoxai_v1_export_quantized_model_request_init(&call);volvoxai_v1_model_files_init(&files);volvoxai_v1_quantized_model_info_init(&exported);
        call.field_quantization_plan_id=plan_id;call.field_output_files=&files;
        SynurangLiteBytes* weight=volvoxai_v1_model_files_add_weight_paths(&files);
        int failed=!weight||assign_text(files._allocator,&files.field_graph_path,output_graph)||assign_text(files._allocator,weight,output_weight);
        if(!failed)VX_CALL_MESSAGE(client,VX_RPC_VX_QUANTIZATION_SERVICE_EXPORT_QUANTIZED_MODEL,volvoxai_v1_export_quantized_model_request,&call,payload,length);
        call.field_output_files=NULL;volvoxai_v1_export_quantized_model_request_free(&call);volvoxai_v1_model_files_free(&files);
        if(failed||!payload||volvoxai_v1_quantized_model_info_decode(&exported,payload,(size_t)length)!=SYNURANG_LITE_OK){volvoxai_v1_quantized_model_info_free(&exported);goto transport;}
        cJSON_AddItemToObject(report,"exportReport",json_build_report(exported.field_report));int ok=report_ok("ExportQuantizedModel",exported.field_report);volvoxai_v1_quantized_model_info_free(&exported);if(!ok)goto done;
        printf("Exported %s and %s\n",output_graph,output_weight);
    }
    result=0;goto done;
transport:
    print_dispatch_error("PTQ",client);goto done;
usage:
    fprintf(stderr,"Invalid PTQ options or policy JSON. Use '%s ptq --help'.\n",argv[0]);result=2;
done:
    if(plan_id)cli_release_plan(client,plan_id);
    if(payload)vx_call_free(client,payload);
    volvoxai_v1_quantization_plan_handle_free(&plan);
    volvoxai_v1_quantization_plan_response_free(&inspected);
    volvoxai_v1_create_quantization_plan_request_free(&request);
    if(report && argc>=3 && strcmp(argv[2],"--help") && strcmp(argv[2],"-h")) {
        cJSON_AddBoolToObject(report,"passed",result==0);cJSON_AddBoolToObject(report,"dryRun",dry_run);
        if(options.report_json && cli_write_json(options.report_json,report))result=1;
    }
    cJSON_Delete(report);cJSON_Delete(progress);cJSON_Delete(rules);cJSON_Delete(batches);return result;
}
