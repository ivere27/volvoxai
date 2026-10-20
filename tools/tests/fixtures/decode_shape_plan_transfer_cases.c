/* CPU transaction fixture; private shape graph and backend services are mocked.
 * Production paged storage, value preflight, COW and commit wrappers are tested. */

enum {N=7,MAXSEQ=16,CASES=50};
static T tensors[N];static Node nodes[2];static Ref node_inputs[2][3];
static VxDeclaredTensor declared[N],outputs[3];static VxModel model;
static VxCompiledModel compiled,alternate;static VxAdapterRevisionRecord adapter;
static float tensor_data[N][MAXSEQ*4],x[MAXSEQ*4],y[MAXSEQ*4];
static int32_t ids[MAXSEQ],position[1];
static int64_t shapes[4*VX_MAX_TENSOR_RANK],prepared_shapes[4*VX_MAX_TENSOR_RANK];
static VxTensorBinding bindings[4];static VxTensorReference references[2];
static VxNativeStorage* leases[2];static uint32_t lengths[1];static const char* cache_names[2];
static VxPublicValuePredicate predicates[3];
static uint64_t hash_bytes(const void* p,size_t n){
    uint64_t h=UINT64_C(1469598103934665603);const unsigned char* b=p;
    for(size_t i=0;i<n;i++){h^=b[i];h*=UINT64_C(1099511628211);}return h;
}
static void setup(VxExecutionContext* c,VxContextRun* run,VxResult* result,int seq,int cow,int policy){
    assert(live_allocs==0);alloc_calls=0;fail_alloc_at=0;live_plans=peak_plans=0;
    resolve_calls=commits=cow_copies=batch_calls=binds=unbindings=0;
    fail_snapshot=fail_batch=fail_commit=fail_shape=fail_value_plan=fail_cache_create=0;
    memset(c,0,sizeof(*c));memset(run,0,sizeof(*run));memset(result,0,sizeof(*result));
    memset(tensors,0,sizeof(tensors));memset(declared,0,sizeof(declared));
    memset(shapes,0,sizeof(shapes));memset(bindings,0,sizeof(bindings));memset(leases,0,sizeof(leases));
    const char* names[N]={"ids","pos","x","y","cache_k","cache_v","logits"};
    for(int i=0;i<N;i++){
        snprintf(tensors[i].name,sizeof(tensors[i].name),"%s",names[i]);
        declared[i].name=tensors[i].name;declared[i].dtype=i<2?VX_DTYPE_I32:VX_DTYPE_F32;
        declared[i].rank=i==0?2:i==1?1:3;
        declared[i].maximums[0]=1;declared[i].maximums[1]=MAXSEQ;
        declared[i].maximums[2]=i==6?3:4;
        tensors[i].data=tensor_data[i];tensors[i].ndim=(int)declared[i].rank;
        tensors[i].dtype=declared[i].dtype;tensors[i].elem_size=4;tensors[i].numel=1;
        for(uint32_t a=0;a<declared[i].rank;a++){
            tensors[i].shape[a]=a==0?1:a==1?seq:i==6?3:4;
            tensors[i].numel*=tensors[i].shape[a];
        }
        tensors[i].is_graph_input=i<4;
        for(int j=0;j<MAXSEQ*4;j++)tensor_data[i][j]=(float)(i*100+j)/16.f;
    }
    for(int i=0;i<MAXSEQ;i++)ids[i]=i%4;
    position[0]=3;
    for(int i=0;i<MAXSEQ*4;i++){x[i]=(float)(i+3)/16.f;y[i]=(float)(31-i)/32.f;}
    for(int i=0;i<4;i++){
        bindings[i].name=declared[i].name;bindings[i].dtype=declared[i].dtype;
        bindings[i].rank=declared[i].rank;bindings[i].location=VX_MEMORY_HOST;
        size_t count=1;
        for(uint32_t a=0;a<declared[i].rank;a++){
            shapes[i*8+a]=bindings[i].shape[a]=tensors[i].shape[a];count*=tensors[i].shape[a];
        }
        bindings[i].byte_size=count*4;
        bindings[i].data=i==0?(void*)ids:i==1?(void*)position:i==2?(void*)x:(void*)y;
    }
    memcpy(prepared_shapes,shapes,sizeof(shapes));
    references[0]=(VxTensorReference){"x","y"};references[1]=(VxTensorReference){"y","x"};
    if(cow){bindings[2].data=tensor_data[3];bindings[3].data=tensor_data[2];}
    outputs[0]=declared[6];outputs[1]=declared[2];outputs[2]=declared[3];
    model=(VxModel){4,3,declared,outputs};compiled=(VxCompiledModel){&model,VX_BACKEND_KIND_CUDA,"cuda"};
    alternate=compiled;adapter.name="__base__";
    state=(VxEngineState){tensors,N,nodes,2,19,0,1,{1}};
    for(int i=0;i<2;i++){
        nodes[i]=(Node){VX_OP_CROSS_SDPA,3,1,node_inputs[i],0,NULL};
        for(int j=0;j<3;j++){
            int selected=j==1?4+i:2;snprintf(node_inputs[i][j].name,128,"%s",tensors[selected].name);
            node_inputs[i][j].tensor_index=selected;node_inputs[i][j].port=j==1?(i?VX_PORT_V:VX_PORT_K):VX_PORT_Q;
        }
    }
    predicates[0]=(VxPublicValuePredicate){0,0,0,4,UINT32_MAX,1,3,VX_STATUS_INVALID_ARGUMENT,VX_PUBLIC_VALUE_I32_RANGE,VX_PUBLIC_BOUND_RESOLVED_AXIS,0};
    predicates[1]=(VxPublicValuePredicate){0,0,2,0,UINT32_MAX,0,0,VX_STATUS_INVALID_ARGUMENT,VX_PUBLIC_VALUE_F32_FINITE,VX_PUBLIC_BOUND_STATIC,0};
    predicates[2]=(VxPublicValuePredicate){0,0,1,4,UINT32_MAX,1,3,VX_STATUS_INVALID_ARGUMENT,VX_PUBLIC_VALUE_I32_RANGE,VX_PUBLIC_BOUND_RESOLVED_AXIS,0};
    c->compiled=&compiled;c->engine_state=&state;c->adapter=&adapter;c->logical_tensors=declared;c->logical_tensor_count=N;
    c->execution_contract_enabled=1;c->decode_row_mode=VX_DECODE_ROW_REQUIRED;c->decode_slots=1;
    lengths[0]=4;c->decode_active_lengths=lengths;c->native_execution_valid=1;
    c->public_value_plan=(VxPublicValuePlan){predicates,3};
    c->execution_state=VX_EXECUTION_CONTEXT_STATE_RUNNING;c->execution_call_id=123;c->execution_scheduler_token=123;
    c->execution_reserved_budget.active=1;c->execution_max_steps=2;
    atomic_init(&c->execution_cancel_requested,0);
    assert(pthread_mutex_init(&c->cache_mutex,NULL)==0&&pthread_mutex_init(&c->queue_mutex,NULL)==0);
    cache_names[0]="cache_k";cache_names[1]="cache_v";
    c->execution_cache=(VxDecodeCacheCommand){cache_names,2,4,0,0,policy,0};
    result->input_shapes=shapes;
    run->context=c;run->result=result;run->kind=VX_CONTEXT_RUN_DECODE_PREFILL;run->input_count=cow?2:4;
    run->staged=(VxStagedBindingBatch){bindings,4,cow?2:0,cow?references:NULL,leases,0};
    plan_size=N*sizeof(VolvoxAIEngineResolvedTensor);
}
static void numeric_forward(const VxStagedBindingBatch* staged,int seq){
    const float* expected_x=staged->reference_count?tensor_data[3]:bindings[2].data;
    const float* expected_y=staged->reference_count?tensor_data[2]:bindings[3].data;
    (void)expected_x;(void)expected_y;
    float* actual_x=tensor_data[2];float* actual_y=tensor_data[3];float* out=tensor_data[6];
    for(int row=0;row<seq;row++)for(int col=0;col<3;col++){
        float sum=0;double reference=0;
        for(int k=0;k<4;k++){
            float w=(float)(k+col+1)/32.f;
            float term=actual_x[row*4+k]+actual_y[row*4+k]*.5f;
            sum+=term*w;
            reference+=((double)actual_x[row*4+k]+(double)actual_y[row*4+k]*.5)*(double)w;
        }
        out[row*3+col]=sum;
        assert(fabs((double)sum-reference)<=1e-5);
    }
}
static void cleanup(VxExecutionContext* c,VxStagedBindingBatch* staged){
    vx_context_decode_cache_clear(c);
    for(size_t i=0;i<staged->native_lease_count;i++){free(staged->native_leases[i]->data);free(staged->native_leases[i]);}
    free(c->native_tensor_pool);free(c->committed_shape_signature);
    assert(pthread_mutex_destroy(&c->cache_mutex)==0&&pthread_mutex_destroy(&c->queue_mutex)==0);
    assert(live_allocs==0&&live_plans==0);
}
static void run_case(int seq,int case_id,int policy){
    int cow=case_id==1||case_id==23||case_id==24||case_id==25;
    VxExecutionContext context;VxContextRun run;VxResult result;VxReport report={.struct_size=sizeof(report)};
    setup(&context,&run,&result,seq,cow,policy);
    float original_x[MAXSEQ*4],original_y[MAXSEQ*4];
    memcpy(original_x,tensor_data[2],sizeof(original_x));memcpy(original_y,tensor_data[3],sizeof(original_y));
    if(case_id==34)fail_cache_create=1;
    if(case_id==35)fail_shape=1;
    if(case_id==37)fail_alloc_at=1;
    if(case_id>=41&&case_id<=48)fail_alloc_at=case_id-39;
    if(case_id==38)context.execution_memory_limit=1;
    VxStatus prepared=case_id==17?VX_STATUS_OK:vx_cache_prepare_initialize(&context,prepared_shapes,position);
    if(prepared==VX_STATUS_OK&&case_id==18)assert(vx_cache_prepare_initialize(&context,prepared_shapes,position)==VX_STATUS_OK);
    switch(case_id){
    case 2:position[0]=2;break;
    case 3:for(int i=0;i<seq*4;i++){x[i]+=1.f;y[i]-=1.f;}break;
    case 4:{
        int actual=seq==16?8:16;
        for(int i=0;i<4;i++)if(i!=1){shapes[i*8+1]=actual;bindings[i].shape[1]=actual;bindings[i].byte_size=(size_t)actual*(i==0?1:4)*4;}
        break;}
    case 5:state.model_generation++;break;
    case 6:context.compiled=&alternate;break;
#if CANDIDATE
    case 7:if(context.execution_prepared_cache)context.execution_prepared_cache->prepared_shape.count--;break;
    case 8:if(context.execution_prepared_cache)context.execution_prepared_cache->prepared_shape.tensors[2].rank=2;break;
    case 9:if(context.execution_prepared_cache)context.execution_prepared_cache->prepared_shape.tensors[2].dtype=VX_DTYPE_I32;break;
    case 10:if(context.execution_prepared_cache)context.execution_prepared_cache->prepared_shape.tensors[2].shape[1]++;break;
#endif
    case 11:context.execution_initialized=1;break;
    case 12:context.decode_row_mode=VX_DECODE_ROW_DISABLED;break;
    case 13:compiled.builtin_backend=VX_BACKEND_KIND_CPU;break;
    case 14:adapter.name="other";break;
    case 15:context.execution_contract_enabled=0;break;
    case 16:run.kind=VX_CONTEXT_RUN_INFER;break;
    case 19:ids[0]=seq;break;
    case 20:x[0]=NAN;break;
    case 22:fail_value_plan=1;break;
    case 23:fail_snapshot=1;break;
    case 24:fail_snapshot=2;break;
    case 25:fail_batch=1;break;
    case 26:fail_commit=-1;break;
    case 27:fail_commit=-2;break;
    case 28:fail_commit=-3;break;
    case 29:atomic_store(&context.execution_cancel_requested,1);break;
    case 30:context.execution_deadline_ns=1;break;
    case 36:fail_alloc_at=alloc_calls+1;break;
    case 39:shapes[0*8+1]=INT64_MAX;break;
    case 40:result.input_shapes=NULL;break;
    case 49:fail_alloc_at=alloc_calls+2;break;
    }
    VxStatus status=prepared;
    int unsettled=case_id>=31&&case_id<=33;
    if(unsettled){
        context.execution_state=VX_EXECUTION_CONTEXT_STATE_QUEUED;
        vx_execution_context_unreserve(&context,case_id==33?124:123,case_id==31);
        status=VX_STATUS_OK;
    }else if(status==VX_STATUS_OK){
        if(case_id==21)status=vx_commit_builtin_bindings(&context,&run.staged,result.input_shapes,run.kind,"wrong-signature",0);
        else{int prepared_result=vx_context_run_prepare(&run);status=run.binding_status;assert((prepared_result==0)==(status==VX_STATUS_OK));}
        if(status==VX_STATUS_OK){
            int actual=(int)shapes[1];
            if(cow){assert(memcmp(tensor_data[2],original_y,(size_t)actual*4*4)==0);
                assert(memcmp(tensor_data[3],original_x,(size_t)actual*4*4)==0);}
            numeric_forward(&run.staged,actual);
            if(case_id!=4 && context.execution_prepared_cache){
                context.decode_prefilled=1;int mutated=0;
                status=vx_cache_configure(&context,&context.execution_cache,&mutated);
                assert(status==VX_STATUS_OK&&mutated==1);
#if CANDIDATE
                assert(context.decode_cache->prepared_shape.tensors==NULL&&context.decode_cache->prepared_shape.signature==NULL);
#endif
            }
        }
        status=vx_execution_settle(&context,&result,status,&report);
    }else status=vx_execution_settle(&context,&result,status,&report);
    int retained=context.execution_prepared_cache!=NULL;
    uint64_t digest=hash_bytes(tensor_data,sizeof(tensor_data));
    printf("{\"sequence\":%d,\"case\":%d,\"policy\":%d,\"status\":%d,\"state\":%d,\"stop\":%d,\"committed\":%d,\"commits\":%d,"
           "\"prepared_cache\":%d,\"active_cache\":%d,\"budget_active\":%d,\"generation\":%" PRIu64 ","
           "\"digest\":\"%016" PRIx64 "\",\"signature\":\"%s\",\"cow_copies\":%d,\"batches\":%d,\"binds\":%d,\"resolve_calls\":%d,\"peak_plans\":%d}\n",
           seq,case_id,policy,status,context.execution_state,context.execution_stop_reason,context.execution_committed,commits,
           retained,context.decode_cache!=NULL,context.execution_reserved_budget.active,context.decode_cache_generation,
           digest,context.committed_shape_signature?context.committed_shape_signature:"",cow_copies,batch_calls,binds,resolve_calls,peak_plans);
    if(case_id==19||case_id==20||case_id==21||case_id==22||case_id==29||case_id==30||case_id==36)
        assert(commits==0&&cow_copies==0&&batch_calls==0);
    if(case_id==23||case_id==24||case_id==25)assert(commits==0&&memcmp(tensor_data[2],original_x,sizeof(original_x))==0&&memcmp(tensor_data[3],original_y,sizeof(original_y))==0);
    assert(peak_plans<=1);
    cleanup(&context,&run.staged);
}
int main(void){
    const int sequences[4]={4,7,12,16};
    for(int i=0;i<4;i++)for(int policy=0;policy<2;policy++)for(int c=0;c<CASES;c++)run_case(sequences[i],c,policy);
    return 0;
}

