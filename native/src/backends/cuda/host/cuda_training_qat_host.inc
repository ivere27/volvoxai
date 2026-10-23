/* A QAT mask lives only through one microbatch. Masters remain in their
 * existing CUDA weight slots; fake values use distinct graph identities. */
struct CudaTrainingQatWeight { CUdeviceptr pass, status; size_t bytes; };
void cuda_training_qat_weight_destroy(CudaTrainingQatWeight* weight) {
    if(!weight)return;
    cuda_training_step_raw_free(&weight->pass);
    cuda_training_step_raw_free(&weight->status);
    free(weight);
}
CudaTrainingQatWeight* cuda_training_qat_weight_prepare(
        const float* master,float* fake,uint32_t rows,uint32_t columns,
        uint32_t group_size,uint32_t bits,uint32_t transpose,
        uint32_t convolution,uint32_t algorithm) {
    uint64_t count=(uint64_t)rows*columns;
    if(!cuda_ready || !cuda_training_module || cuda_forward_scope_active ||
        cuda_training_active || !master || !fake || master==fake || !count ||
        count>UINT32_MAX || count>SIZE_MAX/sizeof(float) || transpose>1u ||
        convolution>1u || algorithm>1u || (bits!=2u && bits!=3u && bits!=4u) ||
        (convolution ? group_size!=0u : (group_size!=32u && group_size!=64u && group_size!=128u && group_size!=256u)) ||
        !CUDA_TRAINING_FUNCTION(k_training_qat_fake) ||
        !CUDA_TRAINING_FUNCTION(k_training_qat_ste))return NULL;
    CudaTrainingQatWeight* weight=calloc(1,sizeof(*weight));
    CudaTensorView source,output;
    if(!weight)return NULL;
    weight->bytes=(size_t)count*sizeof(float);
    uint32_t groups=convolution ? 1u : columns/group_size+(columns%group_size!=0u);
    uint32_t fits=rows*groups,status=0;
    if(!cuda_training_step_raw_alloc(&weight->pass,(size_t)count) ||
        !cuda_training_step_raw_alloc(&weight->status,sizeof(status)) ||
        !cuda_training_step_zero_device_u32(weight->status,1u) ||
        !graph_ensure_input_view(master,weight->bytes,1,&source) ||
        !graph_output_view(fake,weight->bytes,&output))goto fail;
    void* args[]={&source.device,&output.device,&weight->pass,&weight->status,
        &rows,&columns,&group_size,&bits,&transpose,&convolution,&algorithm};
    if(!cuda_launch_1d(CUDA_TRAINING_FUNCTION(k_training_qat_fake),fits,args) ||
        !cuda_training_step_sync_stream() ||
        !cuda_training_step_copy_dtoh(&status,weight->status,sizeof(status)) || status)goto fail;
    graph_mark_device(output.slot);
    return weight;
fail:
    cuda_training_qat_weight_destroy(weight);
    return NULL;
}
int cuda_training_qat_weight_backward(CudaTrainingQatWeight* weight,float* gradient) {
    CudaTensorView view;
    if(!weight || !gradient || !cuda_training_active ||
        !graph_ensure_input_view(gradient,weight->bytes,0,&view))return -1;
    uint32_t count=(uint32_t)(weight->bytes/sizeof(float));
    void* args[]={&view.device,&weight->pass,&count};
    if(!cuda_launch_1d(CUDA_TRAINING_FUNCTION(k_training_qat_ste),count,args))return -1;
    graph_mark_device(view.slot);
    return 0;
}
