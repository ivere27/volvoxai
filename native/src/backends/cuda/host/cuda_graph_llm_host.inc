/* Packed execution plans use the existing forward function registry. */
static CUfunction cuda_packed_function(VxCudaPackedVariantId id) {
    switch(id) {
#define VOLVOXAI_CUDA_FORWARD_FUNCTION(requirement, handle, symbol, storage)
#define VOLVOXAI_CUDA_PACKED_VARIANT(id, handle, route, bm, bn, bk, threads, shared, static_shared, arch, arithmetic, group, prepared, fast, cache) \
    case VX_CUDA_PACKED_VARIANT_##id:return CUDA_FORWARD_FUNCTION(handle);
#include "cuda_forward_function_registry_host.inc"
#undef VOLVOXAI_CUDA_PACKED_VARIANT
#undef VOLVOXAI_CUDA_FORWARD_FUNCTION
    default:return 0;
    }
}
/* Reuse bounded context scratch. A tactic miss precedes every launch; once
 * preparation starts, a failure is terminal for the operation. */
static int cuda_graph_packed_prepared(const VxLLMOpArgs* source,
        const VxLLMOpArgs* device,uint64_t work,const VxCudaPackedLaunchPlan* ordinary) {
    if(source->kind!=VX_OP_PACKED_LINEAR || source->rate!=128 || source->key_dim<=0 || source->value_dim<=0 ||
       !vx_cuda_packed_linear_prepares_input((uint32_t)source->normalize,
           (uint32_t)source->value_dim,ordinary->execution.route) ||
       !source->x[0] || !source->x[1] || !source->x[2] || !source->y[0] ||
       source->first_unit%(uint32_t)source->value_dim || work%(uint32_t)source->value_dim ||
       source->mask || source->positions || source->y[1] || source->y[2])return 0;
    for(int i=4;i<8;i++)if(source->x[i])return 0;
    uint64_t rows=work/(uint32_t)source->value_dim;
    int matrix=ordinary->execution.route==VX_CUDA_PACKED_A8_MATRIX ||
        ordinary->execution.route==VX_CUDA_PACKED_A16_MATRIX;
    uint32_t arch=0u;
#if defined(VOLVOXAI_CUDA_ARCH)
    arch=VOLVOXAI_CUDA_ARCH;
#endif
    VxCudaPackedLaunchPlan selected=*ordinary;
    VxCudaPackedVariantId special=matrix?vx_cuda_packed_prepared_variant(rows,
        source->key_dim,source->value_dim,source->normalize,source->groups,arch):VX_CUDA_PACKED_VARIANT_COUNT;
    VxCudaPackedLaunchPlan candidate;
    if(vx_cuda_packed_plan(special,rows,work,source->key_dim,source->value_dim,source->rate,
            source->normalize,source->groups,arch,source->interleaved,1,
            cuda_max_shared_memory_per_block,&candidate) && cuda_packed_function(special))selected=candidate;
    if(!vx_cuda_packed_workspace(&selected,rows,source->key_dim,source->value_dim,
            source->normalize,source->groups,arch,matrix))return 0;
    uint64_t first_row=source->first_unit/(uint32_t)source->value_dim;
    if(first_row>UINT64_MAX-rows || first_row+rows>source->x_elements[0]/(uint32_t)source->key_dim ||
       first_row+rows>source->y_elements[0]/(uint32_t)source->value_dim)return 0;
    uint32_t capacity=selected.chunk_capacity;
    /* Each chunk's plan is validated before any preparation is submitted.
     * Only the largest chunk and final remainder can differ in geometry. */
    for(unsigned which=0;which<2u;which++) {
        uint32_t count=which?(uint32_t)(rows%capacity):(rows<capacity?(uint32_t)rows:capacity);
        if(!count)continue;
        VxCudaPackedVariantId id=selected.variant;
        if(matrix && !vx_cuda_packed_variants[id].prepared_only)id=vx_cuda_packed_matrix_variant(
            source->normalize,vx_cuda_packed_linear_tile_columns(count,source->value_dim,
                source->normalize,cuda_device_state.multiprocessors));
        VxCudaPackedLaunchPlan preparation;
        if(!vx_cuda_packed_plan(VX_CUDA_PACKED_VARIANT_PREPARE,count,(uint64_t)count*(uint32_t)source->value_dim,
                source->key_dim,source->value_dim,source->rate,source->normalize,source->groups,
                arch,source->interleaved,1,cuda_max_shared_memory_per_block,&preparation) ||
           !vx_cuda_packed_plan(id,count,(uint64_t)count*(uint32_t)source->value_dim,
                source->key_dim,source->value_dim,source->rate,source->normalize,source->groups,
                arch,source->interleaved,1,cuda_max_shared_memory_per_block,&candidate) ||
           !cuda_packed_function(id) ||
           !cuda_launch_dimensions_valid(preparation.grid,1u,1u,preparation.threads,1u,1u,preparation.total_shared_bytes) ||
           !cuda_launch_dimensions_valid(candidate.grid,1u,1u,candidate.threads,1u,1u,candidate.total_shared_bytes))return 0;
    }
    if(!cuda_packed_function(VX_CUDA_PACKED_VARIANT_PREPARE))return 0;
    CudaContextState* state=cuda_context_state_get();
    if(!state || !state->matmul_workspace ||
       state->matmul_workspace>UINT64_MAX-VX_CUDA_CONTEXT_WORKSPACE_BYTES)return 0;
    size_t output_bytes=(size_t)source->y_elements[0]*sizeof(float);
    for(int i=0;i<4;i++)if(source->x[i]) {
        size_t bytes=(size_t)source->x_elements[i]*sizeof(float);
        if(ranges_overlap((const void*)(uintptr_t)state->matmul_workspace,
                VX_CUDA_CONTEXT_WORKSPACE_BYTES,device->x[i],bytes) ||
           ranges_overlap(source->y[0],output_bytes,source->x[i],bytes) ||
           ranges_overlap(device->y[0],output_bytes,device->x[i],bytes))return 0;
    }
    if(ranges_overlap((const void*)(uintptr_t)state->matmul_workspace,
            VX_CUDA_CONTEXT_WORKSPACE_BYTES,device->y[0],output_bytes))return 0;
    for(uint64_t completed=0u;completed<rows;) {
        uint32_t count=rows-completed<capacity?(uint32_t)(rows-completed):capacity;
        VxLLMOpArgs chunk=*device;
        chunk.batch=1;chunk.sequence=(int32_t)count;
        chunk.first_unit=0;chunk.unit_count=0;
        chunk.x[0]=device->x[0]+(first_row+completed)*(uint32_t)source->key_dim;
        chunk.y[0]=device->y[0]+(first_row+completed)*(uint32_t)source->value_dim;
        chunk.x[4]=(const float*)(uintptr_t)state->matmul_workspace;
        chunk.x[5]=(const float*)(uintptr_t)(state->matmul_workspace+(uint64_t)count*selected.workspace_stride);
        VxCudaPackedVariantId id=selected.variant;
        if(matrix && !vx_cuda_packed_variants[id].prepared_only)id=vx_cuda_packed_matrix_variant(
            source->normalize,vx_cuda_packed_linear_tile_columns(count,source->value_dim,
                source->normalize,cuda_device_state.multiprocessors));
        VxCudaPackedLaunchPlan compute,prepare;
        if(!vx_cuda_packed_plan(id,count,(uint64_t)count*(uint32_t)source->value_dim,
                source->key_dim,source->value_dim,source->rate,source->normalize,source->groups,
                arch,source->interleaved,1,cuda_max_shared_memory_per_block,&compute) ||
           !vx_cuda_packed_plan(VX_CUDA_PACKED_VARIANT_PREPARE,count,(uint64_t)count*(uint32_t)source->value_dim,
                source->key_dim,source->value_dim,source->rate,source->normalize,source->groups,
                arch,source->interleaved,1,cuda_max_shared_memory_per_block,&prepare))return -1;
        void* arguments[2]={&chunk,&prepare.execution};
        if(!cuda_launch(cuda_packed_function(prepare.variant),prepare.grid,1u,1u,
                prepare.threads,1u,1u,prepare.shared_bytes,arguments))return -1;
        arguments[1]=&compute.execution;
        if(!cuda_launch(cuda_packed_function(compute.variant),compute.grid,1u,1u,
                compute.threads,1u,1u,compute.shared_bytes,arguments))return -1;
        completed+=count;
    }
    return 1;
}

/* Single-row inference has scalar replay arguments. Refusal occurs before
 * any compute launch; after submission an error is terminal for the node. */
int cuda_graph_packed_linear_decode_f32(const float* input,const int32_t* packed_weights,
        const float* weight_scales,const float* bias,float* output,
        uint32_t input_width,uint32_t output_width,uint32_t weight_bits,uint32_t activation_bits) {
    if(!input || !packed_weights || !weight_scales || !output ||
       !input_width || input_width>INT32_MAX || !output_width || output_width>INT32_MAX ||
       (weight_bits!=4u && weight_bits!=8u) ||
       (activation_bits!=8u && activation_bits!=16u && activation_bits!=32u) ||
       !CUDA_FORWARD_FUNCTION(k_packed_linear_decode))return 0;
    uint32_t per_word=32u/weight_bits,groups=(input_width+127u)/128u;
    uint64_t weight_words=(uint64_t)((input_width+per_word-1u)/per_word)*output_width;
    uint64_t scale_words=(uint64_t)groups*output_width;
    if(weight_words>SIZE_MAX/4u || scale_words>SIZE_MAX/4u)return 0;
#if SIZE_MAX < UINT64_MAX
    if(input_width>SIZE_MAX/4u || output_width>SIZE_MAX/4u)return 0;
#endif
    const void* sources[4]={input,packed_weights,weight_scales,bias};
    size_t sizes[4]={(size_t)input_width*4u,(size_t)weight_words*4u,
        (size_t)scale_words*4u,(size_t)output_width*4u};
    size_t output_bytes=(size_t)output_width*4u;
    CudaTensorView inputs[4],result;
    for(int i=0;i<4;i++)if(sources[i]) {
        if(ranges_overlap(output,output_bytes,sources[i],sizes[i]) ||
           !graph_ensure_input_view(sources[i],sizes[i],0,&inputs[i]))return 0;
    }
    if(!graph_output_view(output,output_bytes,&result))return 0;
    for(int i=0;i<4;i++)if(sources[i] &&
        ranges_overlap((const void*)(uintptr_t)result.device,output_bytes,
            (const void*)(uintptr_t)inputs[i].device,sizes[i]))return 0;
    uint32_t threads=64u,shared_bytes=0u,window=0u;
    uint32_t grid=(output_width+1u)/2u;
    if(activation_bits==8u && output_width>=32u) {
        threads=256u;grid=(output_width+31u)/32u;window=1u;
        uint32_t cache=vx_cuda_packed_linear_row_cache_bytes(input_width);
        if(cache && cuda_max_shared_memory_per_block>=cache+192u) {
            shared_bytes=cache;window=4u;
        }
    } else if((activation_bits==16u || activation_bits==32u) && weight_bits==4u && output_width>=32u) {
        uint32_t cache=vx_cuda_packed_linear_row_cache_bytes(input_width);
        uint32_t cache_act=activation_bits==32u ? cache*4u : cache*2u;
        if(cache && cuda_max_shared_memory_per_block>=cache_act+192u) {
            threads=256u;grid=(output_width+31u)/32u;
            shared_bytes=cache_act;window=4u;
        }
    }
    CUdeviceptr workspace=0;
    CudaContextState* state=cuda_context_state_get();
    uint64_t prepared_bytes=(uint64_t)groups*128u+4u;
    if(activation_bits==8u && output_width>=8192u &&
       prepared_bytes<=VX_CUDA_MATMUL_WORKSPACE_BYTES && state && state->matmul_workspace &&
       state->matmul_workspace<=UINT64_MAX-VX_CUDA_CONTEXT_WORKSPACE_BYTES) {
        workspace=state->matmul_workspace;
        for(int i=0;i<4;i++)if(sources[i] &&
            ranges_overlap((const void*)(uintptr_t)workspace,VX_CUDA_CONTEXT_WORKSPACE_BYTES,
                (const void*)(uintptr_t)inputs[i].device,sizes[i]))workspace=0;
        if(workspace && ranges_overlap((const void*)(uintptr_t)workspace,
                VX_CUDA_CONTEXT_WORKSPACE_BYTES,(const void*)(uintptr_t)result.device,output_bytes))workspace=0;
        if(workspace)shared_bytes=0u;
    }
    if(!cuda_launch_dimensions_valid(grid,1u,1u,threads,1u,1u,shared_bytes+192u) ||
       (workspace && !cuda_launch_dimensions_valid(1u,1u,1u,256u,1u,1u,224u)))return 0;
    CUdeviceptr bias_device=bias?inputs[3].device:0;
    uint32_t phase=0u;CUdeviceptr argument_storage=0;
    void* arguments[13]={&inputs[0].device,&inputs[1].device,&inputs[2].device,&bias_device,
        &result.device,&input_width,&output_width,&weight_bits,&activation_bits,&window,
        &workspace,&phase,&argument_storage};
    if(workspace) {
        phase=1u;
        if(!cuda_launch(CUDA_FORWARD_FUNCTION(k_packed_linear_decode),1u,1u,1u,
                256u,1u,1u,32u,arguments))return -1;
        phase=0u;
    }
    if(!cuda_launch(CUDA_FORWARD_FUNCTION(k_packed_linear_decode),grid,1u,1u,
            threads,1u,1u,shared_bytes,arguments))return -1;
    graph_mark_device(result.slot);
    return 1;
}

/* Adjacent same-input PackedLinear row tactic. Both linear projections share
 * a single input load and quantization step; outputs remain separate. */
int cuda_graph_packed_linear_pair_decode_f32(const float* input,
        const int32_t* packed_weights0, const float* weight_scales0, const float* bias0, float* output0,
        const int32_t* packed_weights1, const float* weight_scales1, const float* bias1, float* output1,
        int sequence, int row, int d_in, int d_out, uint32_t weight_bits, uint32_t activation_bits) {
    if (!input || !packed_weights0 || !weight_scales0 || !output0 ||
        !packed_weights1 || !weight_scales1 || !output1 ||
        sequence <= 0 || row < 0 || row >= sequence || d_in <= 0 || d_out <= 0 ||
        ((weight_bits != 4u && (weight_bits != 8u || (activation_bits != 16u && activation_bits != 32u))) || (activation_bits != 8u && activation_bits != 16u && activation_bits != 32u)) ||
        !CUDA_FORWARD_FUNCTION(k_packed_linear_pair_decode)) return 0;

    uint32_t input_width = (uint32_t)d_in;
    uint32_t output_width = (uint32_t)d_out;
    uint32_t per_word = 32u / weight_bits;
    uint32_t groups = (input_width + 127u) / 128u;
    uint64_t weight_words = (uint64_t)((input_width + per_word - 1u) / per_word) * output_width;
    uint64_t scale_words = (uint64_t)groups * output_width;

    size_t input_bytes = (size_t)sequence * input_width * sizeof(float);
    size_t output_bytes = (size_t)sequence * output_width * sizeof(float);
    size_t weight_bytes = (size_t)weight_words * sizeof(uint32_t);
    size_t scale_bytes = (size_t)scale_words * sizeof(float);
    size_t bias_bytes = (size_t)output_width * sizeof(float);

    CudaTensorView in_view, w0_view, s0_view, b0_view = {0}, out0_view;
    CudaTensorView w1_view, s1_view, b1_view = {0}, out1_view;

    if (!graph_ensure_input_view(input, input_bytes, 0, &in_view) ||
        !graph_ensure_input_view(packed_weights0, weight_bytes, 0, &w0_view) ||
        !graph_ensure_input_view(weight_scales0, scale_bytes, 0, &s0_view) ||
        !graph_ensure_input_view(packed_weights1, weight_bytes, 0, &w1_view) ||
        !graph_ensure_input_view(weight_scales1, scale_bytes, 0, &s1_view)) return 0;

    if (bias0 && !graph_ensure_input_view(bias0, bias_bytes, 0, &b0_view)) return 0;
    if (bias1 && !graph_ensure_input_view(bias1, bias_bytes, 0, &b1_view)) return 0;

    if (!graph_output_view(output0, output_bytes, &out0_view) ||
        !graph_output_view(output1, output_bytes, &out1_view)) return 0;

    uint32_t threads = 256u;
    uint32_t grid = (output_width + 31u) / 32u;
    uint32_t cache = vx_cuda_packed_linear_row_cache_bytes(input_width);
    uint32_t shared_bytes = activation_bits == 32u ? cache * 4u : (activation_bits == 16u ? cache * 2u : cache);
    if (!cache || cuda_max_shared_memory_per_block < shared_bytes + 160u) return 0;

    if (!cuda_launch_dimensions_valid(grid, 1u, 1u, threads, 1u, 1u, shared_bytes + 160u)) return 0;

    in_view.device += (uint64_t)row * input_width * sizeof(float);
    out0_view.device += (uint64_t)row * output_width * sizeof(float);
    out1_view.device += (uint64_t)row * output_width * sizeof(float);

    CUdeviceptr b0_device = bias0 ? b0_view.device : 0;
    CUdeviceptr b1_device = bias1 ? b1_view.device : 0;
    CUdeviceptr argument_storage = 0;

    void* args[14] = {
        &in_view.device,
        &w0_view.device, &s0_view.device, &b0_device, &out0_view.device,
        &w1_view.device, &s1_view.device, &b1_device, &out1_view.device,
        &input_width, &output_width, &weight_bits, &activation_bits,
        &argument_storage
    };

    if (!cuda_launch(CUDA_FORWARD_FUNCTION(k_packed_linear_pair_decode), grid, 1u, 1u,
                     threads, 1u, 1u, shared_bytes, args)) return -1;

    graph_mark_device(out0_view.slot);
    graph_mark_device(out1_view.slot);
    return 1;
}

/* Adjacent same-input PackedLinear pair with fused SwiGLU: computes Gate and Up
 * projections and immediately evaluates silu(gate) * up in registers, writing all
 * four logical outputs without extra kernel launches or memory round trips. */
int cuda_graph_packed_linear_pair_silu_mul_decode_f32(const float* input,
        const int32_t* packed_weights0, const float* weight_scales0, const float* bias0, float* output0,
        const int32_t* packed_weights1, const float* weight_scales1, const float* bias1, float* output1,
        float* output_silu, float* output_product,
        int sequence, int row, int d_in, int d_out, uint32_t weight_bits, uint32_t activation_bits) {
    if (!input || !packed_weights0 || !weight_scales0 || !output0 ||
        !packed_weights1 || !weight_scales1 || !output1 ||
        !output_silu || !output_product ||
        sequence <= 0 || row < 0 || row >= sequence || d_in <= 0 || d_out <= 0 ||
        ((weight_bits != 4u && (weight_bits != 8u || (activation_bits != 16u && activation_bits != 32u))) || (activation_bits != 8u && activation_bits != 16u && activation_bits != 32u)) ||
        !CUDA_FORWARD_FUNCTION(k_packed_linear_pair_silu_mul_decode)) return 0;

    uint32_t input_width = (uint32_t)d_in;
    uint32_t output_width = (uint32_t)d_out;
    uint32_t per_word = 32u / weight_bits;
    uint32_t groups = (input_width + 127u) / 128u;
    uint64_t weight_words = (uint64_t)((input_width + per_word - 1u) / per_word) * output_width;
    uint64_t scale_words = (uint64_t)groups * output_width;

    size_t input_bytes = (size_t)sequence * input_width * sizeof(float);
    size_t output_bytes = (size_t)sequence * output_width * sizeof(float);
    size_t weight_bytes = (size_t)weight_words * sizeof(uint32_t);
    size_t scale_bytes = (size_t)scale_words * sizeof(float);
    size_t bias_bytes = (size_t)output_width * sizeof(float);

    CudaTensorView in_view, w0_view, s0_view, b0_view = {0}, out0_view;
    CudaTensorView w1_view, s1_view, b1_view = {0}, out1_view;
    CudaTensorView out_silu_view, out_prod_view;

    if (!graph_ensure_input_view(input, input_bytes, 0, &in_view) ||
        !graph_ensure_input_view(packed_weights0, weight_bytes, 0, &w0_view) ||
        !graph_ensure_input_view(weight_scales0, scale_bytes, 0, &s0_view) ||
        !graph_ensure_input_view(packed_weights1, weight_bytes, 0, &w1_view) ||
        !graph_ensure_input_view(weight_scales1, scale_bytes, 0, &s1_view)) return 0;

    if (bias0 && !graph_ensure_input_view(bias0, bias_bytes, 0, &b0_view)) return 0;
    if (bias1 && !graph_ensure_input_view(bias1, bias_bytes, 0, &b1_view)) return 0;

    if (!graph_output_view(output0, output_bytes, &out0_view) ||
        !graph_output_view(output1, output_bytes, &out1_view) ||
        !graph_output_view(output_silu, output_bytes, &out_silu_view) ||
        !graph_output_view(output_product, output_bytes, &out_prod_view)) return 0;

    uint32_t threads = 256u;
    uint32_t grid = (output_width + 31u) / 32u;
    uint32_t cache = vx_cuda_packed_linear_row_cache_bytes(input_width);
    uint32_t shared_bytes = activation_bits == 32u ? cache * 4u : (activation_bits == 16u ? cache * 2u : cache);
    if (!cache || cuda_max_shared_memory_per_block < shared_bytes + 160u) return 0;

    if (!cuda_launch_dimensions_valid(grid, 1u, 1u, threads, 1u, 1u, shared_bytes + 160u)) return 0;

    in_view.device += (uint64_t)row * input_width * sizeof(float);
    out0_view.device += (uint64_t)row * output_width * sizeof(float);
    out1_view.device += (uint64_t)row * output_width * sizeof(float);
    out_silu_view.device += (uint64_t)row * output_width * sizeof(float);
    out_prod_view.device += (uint64_t)row * output_width * sizeof(float);

    CUdeviceptr b0_device = bias0 ? b0_view.device : 0;
    CUdeviceptr b1_device = bias1 ? b1_view.device : 0;
    CUdeviceptr argument_storage = 0;

    void* args[16] = {
        &in_view.device,
        &w0_view.device, &s0_view.device, &b0_device, &out0_view.device,
        &w1_view.device, &s1_view.device, &b1_device, &out1_view.device,
        &out_silu_view.device, &out_prod_view.device,
        &input_width, &output_width, &weight_bits, &activation_bits,
        &argument_storage
    };

    if (!cuda_launch(CUDA_FORWARD_FUNCTION(k_packed_linear_pair_silu_mul_decode), grid, 1u, 1u,
                     threads, 1u, 1u, shared_bytes, args)) return -1;

    graph_mark_device(out0_view.slot);
    graph_mark_device(out1_view.slot);
    graph_mark_device(out_silu_view.slot);
    graph_mark_device(out_prod_view.slot);
    return 1;
}

int cuda_graph_packed_linear_triple_decode_f32(const float* input,
        const int32_t* packed_weights0, const float* weight_scales0, const float* bias0, float* output0, int d_out0,
        const int32_t* packed_weights1, const float* weight_scales1, const float* bias1, float* output1, int d_out1,
        const int32_t* packed_weights2, const float* weight_scales2, const float* bias2, float* output2, int d_out2,
        int sequence, int row, int d_in, uint32_t weight_bits, uint32_t activation_bits) {
    if (!input || !packed_weights0 || !weight_scales0 || !output0 ||
        !packed_weights1 || !weight_scales1 || !output1 ||
        !packed_weights2 || !weight_scales2 || !output2 ||
        sequence <= 0 || row < 0 || row >= sequence || d_in <= 0 ||
        d_out0 <= 0 || d_out1 <= 0 || d_out2 <= 0 ||
        ((weight_bits != 4u && (weight_bits != 8u || (activation_bits != 16u && activation_bits != 32u))) || (activation_bits != 8u && activation_bits != 16u && activation_bits != 32u)) ||
        !CUDA_FORWARD_FUNCTION(k_packed_linear_triple_decode)) return 0;

    uint32_t input_width = (uint32_t)d_in;
    uint32_t output_width0 = (uint32_t)d_out0;
    uint32_t output_width1 = (uint32_t)d_out1;
    uint32_t output_width2 = (uint32_t)d_out2;
    uint32_t per_word = 32u / weight_bits;
    uint32_t groups = (input_width + 127u) / 128u;

    uint64_t weight_words0 = (uint64_t)((input_width + per_word - 1u) / per_word) * output_width0;
    uint64_t scale_words0 = (uint64_t)groups * output_width0;
    uint64_t weight_words1 = (uint64_t)((input_width + per_word - 1u) / per_word) * output_width1;
    uint64_t scale_words1 = (uint64_t)groups * output_width1;
    uint64_t weight_words2 = (uint64_t)((input_width + per_word - 1u) / per_word) * output_width2;
    uint64_t scale_words2 = (uint64_t)groups * output_width2;

    size_t input_bytes = (size_t)sequence * input_width * sizeof(float);
    size_t output_bytes0 = (size_t)sequence * output_width0 * sizeof(float);
    size_t weight_bytes0 = (size_t)weight_words0 * sizeof(uint32_t);
    size_t scale_bytes0 = (size_t)scale_words0 * sizeof(float);
    size_t bias_bytes0 = (size_t)output_width0 * sizeof(float);

    size_t output_bytes1 = (size_t)sequence * output_width1 * sizeof(float);
    size_t weight_bytes1 = (size_t)weight_words1 * sizeof(uint32_t);
    size_t scale_bytes1 = (size_t)scale_words1 * sizeof(float);
    size_t bias_bytes1 = (size_t)output_width1 * sizeof(float);

    size_t output_bytes2 = (size_t)sequence * output_width2 * sizeof(float);
    size_t weight_bytes2 = (size_t)weight_words2 * sizeof(uint32_t);
    size_t scale_bytes2 = (size_t)scale_words2 * sizeof(float);
    size_t bias_bytes2 = (size_t)output_width2 * sizeof(float);

    CudaTensorView in_view, w0_view, s0_view, b0_view = {0}, out0_view;
    CudaTensorView w1_view, s1_view, b1_view = {0}, out1_view;
    CudaTensorView w2_view, s2_view, b2_view = {0}, out2_view;

    if (!graph_ensure_input_view(input, input_bytes, 0, &in_view) ||
        !graph_ensure_input_view(packed_weights0, weight_bytes0, 0, &w0_view) ||
        !graph_ensure_input_view(weight_scales0, scale_bytes0, 0, &s0_view) ||
        !graph_ensure_input_view(packed_weights1, weight_bytes1, 0, &w1_view) ||
        !graph_ensure_input_view(weight_scales1, scale_bytes1, 0, &s1_view) ||
        !graph_ensure_input_view(packed_weights2, weight_bytes2, 0, &w2_view) ||
        !graph_ensure_input_view(weight_scales2, scale_bytes2, 0, &s2_view)) return 0;

    if (bias0 && !graph_ensure_input_view(bias0, bias_bytes0, 0, &b0_view)) return 0;
    if (bias1 && !graph_ensure_input_view(bias1, bias_bytes1, 0, &b1_view)) return 0;
    if (bias2 && !graph_ensure_input_view(bias2, bias_bytes2, 0, &b2_view)) return 0;

    if (!graph_output_view(output0, output_bytes0, &out0_view) ||
        !graph_output_view(output1, output_bytes1, &out1_view) ||
        !graph_output_view(output2, output_bytes2, &out2_view)) return 0;

    uint32_t threads = 256u;
    uint32_t tiles0 = (output_width0 + 31u) / 32u;
    uint32_t tiles1 = (output_width1 + 31u) / 32u;
    uint32_t tiles2 = (output_width2 + 31u) / 32u;
    uint32_t grid = tiles0 + tiles1 + tiles2;
    uint32_t cache = vx_cuda_packed_linear_row_cache_bytes(input_width);
    uint32_t shared_bytes = activation_bits == 32u ? cache * 4u : (activation_bits == 16u ? cache * 2u : cache);
    if (!cache || cuda_max_shared_memory_per_block < shared_bytes + 160u) return 0;

    if (!cuda_launch_dimensions_valid(grid, 1u, 1u, threads, 1u, 1u, shared_bytes + 160u)) return 0;

    in_view.device += (uint64_t)row * input_width * sizeof(float);
    out0_view.device += (uint64_t)row * output_width0 * sizeof(float);
    out1_view.device += (uint64_t)row * output_width1 * sizeof(float);
    out2_view.device += (uint64_t)row * output_width2 * sizeof(float);

    CUdeviceptr b0_device = bias0 ? b0_view.device : 0;
    CUdeviceptr b1_device = bias1 ? b1_view.device : 0;
    CUdeviceptr b2_device = bias2 ? b2_view.device : 0;
    CUdeviceptr argument_storage = 0;

    void* args[20] = {
        &in_view.device,
        &w0_view.device, &s0_view.device, &b0_device, &out0_view.device, &output_width0,
        &w1_view.device, &s1_view.device, &b1_device, &out1_view.device, &output_width1,
        &w2_view.device, &s2_view.device, &b2_device, &out2_view.device, &output_width2,
        &input_width, &weight_bits, &activation_bits,
        &argument_storage
    };

    if (!cuda_launch(CUDA_FORWARD_FUNCTION(k_packed_linear_triple_decode), grid, 1u, 1u,
                     threads, 1u, 1u, shared_bytes, args)) return -1;

    graph_mark_device(out0_view.slot);
    graph_mark_device(out1_view.slot);
    graph_mark_device(out2_view.slot);
    return 1;
}

int cuda_graph_rmsnorm_packed_linear_triple_decode_f32(const float* input,
        const float* norm_weight, float* norm_output, float epsilon,
        const int32_t* packed_weights0, const float* weight_scales0, const float* bias0, float* output0, int d_out0,
        const int32_t* packed_weights1, const float* weight_scales1, const float* bias1, float* output1, int d_out1,
        const int32_t* packed_weights2, const float* weight_scales2, const float* bias2, float* output2, int d_out2,
        int sequence, int row, int d_in, uint32_t weight_bits, uint32_t activation_bits) {
    if (!input || !norm_weight || !norm_output || epsilon <= 0.0f ||
        !packed_weights0 || !weight_scales0 || !output0 ||
        !packed_weights1 || !weight_scales1 || !output1 ||
        !packed_weights2 || !weight_scales2 || !output2 ||
        sequence <= 0 || row < 0 || row >= sequence ||
        d_in <= 0 || d_out0 <= 0 || d_out1 <= 0 || d_out2 <= 0 ||
        ((weight_bits != 4u && (weight_bits != 8u || (activation_bits != 16u && activation_bits != 32u))) || (activation_bits != 8u && activation_bits != 16u && activation_bits != 32u)) ||
        !CUDA_FORWARD_FUNCTION(k_rmsnorm_packed_linear_triple_decode)) return 0;

    uint32_t input_width = (uint32_t)d_in;
    uint32_t output_width0 = (uint32_t)d_out0;
    uint32_t output_width1 = (uint32_t)d_out1;
    uint32_t output_width2 = (uint32_t)d_out2;
    uint32_t per_word = 32u / weight_bits;
    uint32_t groups = (input_width + 127u) / 128u;

    uint64_t weight_words0 = (uint64_t)((input_width + per_word - 1u) / per_word) * output_width0;
    uint64_t scale_words0 = (uint64_t)groups * output_width0;
    uint64_t weight_words1 = (uint64_t)((input_width + per_word - 1u) / per_word) * output_width1;
    uint64_t scale_words1 = (uint64_t)groups * output_width1;
    uint64_t weight_words2 = (uint64_t)((input_width + per_word - 1u) / per_word) * output_width2;
    uint64_t scale_words2 = (uint64_t)groups * output_width2;

    size_t input_bytes = (size_t)sequence * input_width * sizeof(float);
    size_t norm_weight_bytes = (size_t)input_width * sizeof(float);
    size_t norm_output_bytes = (size_t)sequence * input_width * sizeof(float);

    size_t output_bytes0 = (size_t)sequence * output_width0 * sizeof(float);
    size_t weight_bytes0 = (size_t)weight_words0 * sizeof(uint32_t);
    size_t scale_bytes0 = (size_t)scale_words0 * sizeof(float);
    size_t bias_bytes0 = (size_t)output_width0 * sizeof(float);

    size_t output_bytes1 = (size_t)sequence * output_width1 * sizeof(float);
    size_t weight_bytes1 = (size_t)weight_words1 * sizeof(uint32_t);
    size_t scale_bytes1 = (size_t)scale_words1 * sizeof(float);
    size_t bias_bytes1 = (size_t)output_width1 * sizeof(float);

    size_t output_bytes2 = (size_t)sequence * output_width2 * sizeof(float);
    size_t weight_bytes2 = (size_t)weight_words2 * sizeof(uint32_t);
    size_t scale_bytes2 = (size_t)scale_words2 * sizeof(float);
    size_t bias_bytes2 = (size_t)output_width2 * sizeof(float);

    CudaTensorView in_view, nw_view, no_view;
    CudaTensorView w0_view, s0_view, b0_view = {0}, out0_view;
    CudaTensorView w1_view, s1_view, b1_view = {0}, out1_view;
    CudaTensorView w2_view, s2_view, b2_view = {0}, out2_view;

    if (!graph_ensure_input_view(input, input_bytes, 0, &in_view) ||
        !graph_ensure_input_view(norm_weight, norm_weight_bytes, 0, &nw_view) ||
        !graph_output_view(norm_output, norm_output_bytes, &no_view) ||
        !graph_ensure_input_view(packed_weights0, weight_bytes0, 0, &w0_view) ||
        !graph_ensure_input_view(weight_scales0, scale_bytes0, 0, &s0_view) ||
        !graph_ensure_input_view(packed_weights1, weight_bytes1, 0, &w1_view) ||
        !graph_ensure_input_view(weight_scales1, scale_bytes1, 0, &s1_view) ||
        !graph_ensure_input_view(packed_weights2, weight_bytes2, 0, &w2_view) ||
        !graph_ensure_input_view(weight_scales2, scale_bytes2, 0, &s2_view)) return 0;

    if (bias0 && !graph_ensure_input_view(bias0, bias_bytes0, 0, &b0_view)) return 0;
    if (bias1 && !graph_ensure_input_view(bias1, bias_bytes1, 0, &b1_view)) return 0;
    if (bias2 && !graph_ensure_input_view(bias2, bias_bytes2, 0, &b2_view)) return 0;

    if (!graph_output_view(output0, output_bytes0, &out0_view) ||
        !graph_output_view(output1, output_bytes1, &out1_view) ||
        !graph_output_view(output2, output_bytes2, &out2_view)) return 0;

    uint32_t threads = 256u;
    uint32_t tiles0 = (output_width0 + 31u) / 32u;
    uint32_t tiles1 = (output_width1 + 31u) / 32u;
    uint32_t tiles2 = (output_width2 + 31u) / 32u;
    uint32_t grid = tiles0 + tiles1 + tiles2;
    uint32_t cache = vx_cuda_packed_linear_row_cache_bytes(input_width);
    uint32_t shared_bytes = activation_bits == 32u ? cache * 4u : (activation_bits == 16u ? cache * 2u : cache);
    if (!cache || cuda_max_shared_memory_per_block < shared_bytes + 160u) return 0;

    if (!cuda_launch_dimensions_valid(grid, 1u, 1u, threads, 1u, 1u, shared_bytes + 160u)) return 0;

    in_view.device += (uint64_t)row * input_width * sizeof(float);
    no_view.device += (uint64_t)row * input_width * sizeof(float);
    out0_view.device += (uint64_t)row * output_width0 * sizeof(float);
    out1_view.device += (uint64_t)row * output_width1 * sizeof(float);
    out2_view.device += (uint64_t)row * output_width2 * sizeof(float);

    CUdeviceptr b0_device = bias0 ? b0_view.device : 0;
    CUdeviceptr b1_device = bias1 ? b1_view.device : 0;
    CUdeviceptr b2_device = bias2 ? b2_view.device : 0;
    CUdeviceptr argument_storage = 0;

    void* args[23] = {
        &in_view.device, &nw_view.device, &no_view.device, &epsilon,
        &w0_view.device, &s0_view.device, &b0_device, &out0_view.device, &output_width0,
        &w1_view.device, &s1_view.device, &b1_device, &out1_view.device, &output_width1,
        &w2_view.device, &s2_view.device, &b2_device, &out2_view.device, &output_width2,
        &input_width, &weight_bits, &activation_bits,
        &argument_storage
    };

    if (!cuda_launch(CUDA_FORWARD_FUNCTION(k_rmsnorm_packed_linear_triple_decode), grid, 1u, 1u,
                     threads, 1u, 1u, shared_bytes, args)) return -1;

    graph_mark_device(no_view.slot);
    graph_mark_device(out0_view.slot);
    graph_mark_device(out1_view.slot);
    graph_mark_device(out2_view.slot);
    return 1;
}

int cuda_graph_rmsnorm_packed_linear_pair_silu_mul_decode_f32(const float* input,
        const float* norm_weight, float* norm_output, float epsilon,
        const int32_t* packed_weights0, const float* weight_scales0, const float* bias0, float* output0,
        const int32_t* packed_weights1, const float* weight_scales1, const float* bias1, float* output1,
        float* silu_output, float* mul_output,
        int sequence, int row, int d_in, int d_out, uint32_t weight_bits, uint32_t activation_bits) {
    if (!input || !norm_weight || !norm_output || epsilon <= 0.0f ||
        !packed_weights0 || !weight_scales0 || !output0 ||
        !packed_weights1 || !weight_scales1 || !output1 ||
        !silu_output || !mul_output ||
        sequence <= 0 || row < 0 || row >= sequence || d_in <= 0 || d_out <= 0 ||
        ((weight_bits != 4u && (weight_bits != 8u || (activation_bits != 16u && activation_bits != 32u))) || (activation_bits != 8u && activation_bits != 16u && activation_bits != 32u)) ||
        !CUDA_FORWARD_FUNCTION(k_rmsnorm_packed_linear_pair_silu_mul_decode)) return 0;

    uint32_t input_width = (uint32_t)d_in;
    uint32_t output_width = (uint32_t)d_out;
    uint32_t per_word = 32u / weight_bits;
    uint32_t groups = (input_width + 127u) / 128u;
    uint64_t weight_words = (uint64_t)((input_width + per_word - 1u) / per_word) * output_width;
    uint64_t scale_words = (uint64_t)groups * output_width;

    size_t input_bytes = (size_t)sequence * input_width * sizeof(float);
    size_t norm_weight_bytes = (size_t)input_width * sizeof(float);
    size_t norm_output_bytes = (size_t)sequence * input_width * sizeof(float);
    size_t output_bytes = (size_t)sequence * output_width * sizeof(float);
    size_t weight_bytes = (size_t)weight_words * sizeof(uint32_t);
    size_t scale_bytes = (size_t)scale_words * sizeof(float);
    size_t bias_bytes = (size_t)output_width * sizeof(float);

    CudaTensorView in_view, nw_view, no_view;
    CudaTensorView w0_view, s0_view, b0_view = {0}, out0_view;
    CudaTensorView w1_view, s1_view, b1_view = {0}, out1_view;
    CudaTensorView silu_view, mul_view;

    if (!graph_ensure_input_view(input, input_bytes, 0, &in_view) ||
        !graph_ensure_input_view(norm_weight, norm_weight_bytes, 0, &nw_view) ||
        !graph_output_view(norm_output, norm_output_bytes, &no_view) ||
        !graph_ensure_input_view(packed_weights0, weight_bytes, 0, &w0_view) ||
        !graph_ensure_input_view(weight_scales0, scale_bytes, 0, &s0_view) ||
        !graph_ensure_input_view(packed_weights1, weight_bytes, 0, &w1_view) ||
        !graph_ensure_input_view(weight_scales1, scale_bytes, 0, &s1_view)) return 0;

    if (bias0 && !graph_ensure_input_view(bias0, bias_bytes, 0, &b0_view)) return 0;
    if (bias1 && !graph_ensure_input_view(bias1, bias_bytes, 0, &b1_view)) return 0;

    if (!graph_output_view(output0, output_bytes, &out0_view) ||
        !graph_output_view(output1, output_bytes, &out1_view) ||
        !graph_output_view(silu_output, output_bytes, &silu_view) ||
        !graph_output_view(mul_output, output_bytes, &mul_view)) return 0;

    uint32_t threads = 256u;
    uint32_t grid = (output_width + 31u) / 32u;
    uint32_t cache = vx_cuda_packed_linear_row_cache_bytes(input_width);
    uint32_t shared_bytes = activation_bits == 32u ? cache * 4u : (activation_bits == 16u ? cache * 2u : cache);
    if (!cache || cuda_max_shared_memory_per_block < shared_bytes + 160u) return 0;

    if (!cuda_launch_dimensions_valid(grid, 1u, 1u, threads, 1u, 1u, shared_bytes + 160u)) return 0;

    in_view.device += (uint64_t)row * input_width * sizeof(float);
    no_view.device += (uint64_t)row * input_width * sizeof(float);
    out0_view.device += (uint64_t)row * output_width * sizeof(float);
    out1_view.device += (uint64_t)row * output_width * sizeof(float);
    silu_view.device += (uint64_t)row * output_width * sizeof(float);
    mul_view.device += (uint64_t)row * output_width * sizeof(float);

    CUdeviceptr b0_device = bias0 ? b0_view.device : 0;
    CUdeviceptr b1_device = bias1 ? b1_view.device : 0;
    CUdeviceptr argument_storage = 0;

    void* args[19] = {
        &in_view.device, &nw_view.device, &no_view.device, &epsilon,
        &w0_view.device, &s0_view.device, &b0_device, &out0_view.device,
        &w1_view.device, &s1_view.device, &b1_device, &out1_view.device,
        &silu_view.device, &mul_view.device,
        &input_width, &output_width,
        &weight_bits, &activation_bits,
        &argument_storage
    };

    if (!cuda_launch(CUDA_FORWARD_FUNCTION(k_rmsnorm_packed_linear_pair_silu_mul_decode), grid, 1u, 1u,
                     threads, 1u, 1u, shared_bytes, args)) return -1;

    graph_mark_device(no_view.slot);
    graph_mark_device(out0_view.slot);
    graph_mark_device(out1_view.slot);
    graph_mark_device(silu_view.slot);
    graph_mark_device(mul_view.slot);
    return 1;
}

int cuda_graph_packed_linear_add_decode_f32(const float* input,
        const int32_t* packed_weights, const float* weight_scales, const float* bias, float* output,
        const float* residual, float* final_output,
        int sequence, int row, int d_in, int d_out, uint32_t weight_bits, uint32_t activation_bits,
        int residual_first) {
    if (!input || !packed_weights || !weight_scales || !output || !residual || !final_output ||
        sequence <= 0 || row < 0 || row >= sequence || d_in <= 0 || d_out <= 0 ||
        ((weight_bits != 4u && (weight_bits != 8u || (activation_bits != 16u && activation_bits != 32u))) || (activation_bits != 8u && activation_bits != 16u && activation_bits != 32u)) ||
        !CUDA_FORWARD_FUNCTION(k_packed_linear_add_decode)) return 0;

    uint32_t input_width = (uint32_t)d_in;
    uint32_t output_width = (uint32_t)d_out;
    uint32_t per_word = 32u / weight_bits;
    uint32_t groups = (input_width + 127u) / 128u;
    uint64_t weight_words = (uint64_t)((input_width + per_word - 1u) / per_word) * output_width;
    uint64_t scale_words = (uint64_t)groups * output_width;

    size_t input_bytes = (size_t)sequence * input_width * sizeof(float);
    size_t output_bytes = (size_t)sequence * output_width * sizeof(float);
    size_t weight_bytes = (size_t)weight_words * sizeof(uint32_t);
    size_t scale_bytes = (size_t)scale_words * sizeof(float);
    size_t bias_bytes = (size_t)output_width * sizeof(float);

    CudaTensorView in_view, w_view, s_view, b_view = {0}, out_view, res_view, final_view;

    if (!graph_ensure_input_view(input, input_bytes, 0, &in_view) ||
        !graph_ensure_input_view(packed_weights, weight_bytes, 0, &w_view) ||
        !graph_ensure_input_view(weight_scales, scale_bytes, 0, &s_view) ||
        !graph_ensure_input_view(residual, output_bytes, 0, &res_view)) return 0;

    if (bias && !graph_ensure_input_view(bias, bias_bytes, 0, &b_view)) return 0;

    if (!graph_output_view(output, output_bytes, &out_view) ||
        !graph_output_view(final_output, output_bytes, &final_view)) return 0;

    uint32_t threads = 256u;
    uint32_t grid = (output_width + 31u) / 32u;
    uint32_t cache = vx_cuda_packed_linear_row_cache_bytes(input_width);
    uint32_t shared_bytes = activation_bits == 32u ? cache * 4u : (activation_bits == 16u ? cache * 2u : cache);
    if (!cache || cuda_max_shared_memory_per_block < shared_bytes + 160u) return 0;

    if (!cuda_launch_dimensions_valid(grid, 1u, 1u, threads, 1u, 1u, shared_bytes + 160u)) return 0;

    in_view.device += (uint64_t)row * input_width * sizeof(float);
    res_view.device += (uint64_t)row * output_width * sizeof(float);
    out_view.device += (uint64_t)row * output_width * sizeof(float);
    final_view.device += (uint64_t)row * output_width * sizeof(float);

    CUdeviceptr b_device = bias ? b_view.device : 0;
    uint32_t res_first = (uint32_t)residual_first;
    CUdeviceptr argument_storage = 0;

    void* args[13] = {
        &in_view.device, &w_view.device, &s_view.device, &b_device, &out_view.device,
        &res_view.device, &final_view.device,
        &input_width, &output_width, &weight_bits, &activation_bits,
        &res_first, &argument_storage
    };

    if (!cuda_launch(CUDA_FORWARD_FUNCTION(k_packed_linear_add_decode), grid, 1u, 1u,
                     threads, 1u, 1u, shared_bytes, args)) return -1;

    graph_mark_device(out_view.slot);
    graph_mark_device(final_view.slot);
    return 1;
}

int cuda_graph_llm_f32(const VxLLMOpArgs* source) {
    VxLLMOpArgs device;
    CudaTensorView inputs[8],outputs[3],mask,positions;
    void* arguments[2];
    uint64_t work;
    if (!source || source->key_dim<=0 ||
        (source->kind==VX_OP_PACKED_LINEAR && source->first_unit && !source->unit_count) ||
        ((source->kind==VX_OP_GATED_DELTA_NET || source->kind==VX_OP_SPARSE_ATTENTION ||
          source->kind==VX_OP_ROTARY_EMBEDDING || source->kind==VX_OP_QSA_INDEXER ||
          source->kind==VX_OP_SPARSE_INDEXER) && source->key_dim>512)) return 0;
    device=*source;
    for (int i=0;i<8;i++) if (source->x[i]) {
        if (!source->x_elements[i] || source->x_elements[i]>SIZE_MAX/sizeof(float) ||
            !graph_ensure_input_view(source->x[i],(size_t)source->x_elements[i]*sizeof(float),0,&inputs[i])) return 0;
        device.x[i]=(const float*)(uintptr_t)inputs[i].device;
    }
    for (int i=0;i<3;i++) if (source->y[i]) {
        if (!source->y_elements[i] || source->y_elements[i]>SIZE_MAX/sizeof(float) ||
            !graph_output_view(source->y[i],(size_t)source->y_elements[i]*sizeof(float),&outputs[i])) return 0;
        device.y[i]=(float*)(uintptr_t)outputs[i].device;
    }
    if (source->mask) {
        if (!source->mask_elements || source->mask_elements>SIZE_MAX/sizeof(int32_t) ||
            !graph_ensure_input_view(source->mask,(size_t)source->mask_elements*sizeof(int32_t),0,&mask)) return 0;
        device.mask=(const int32_t*)(uintptr_t)mask.device;
    }
    if (source->positions) {
        if (!source->position_elements || source->position_elements>SIZE_MAX/sizeof(int32_t) ||
            !graph_ensure_input_view(source->positions,(size_t)source->position_elements*sizeof(int32_t),0,&positions)) return 0;
        device.positions=(const int32_t*)(uintptr_t)positions.device;
    }
    work=vx_llm_work_items(source);
    if (source->kind==VX_OP_GATED_DELTA_NET || source->kind==VX_OP_SPARSE_ATTENTION) {
        if (source->value_dim<=0 || work>UINT64_MAX/(uint32_t)source->value_dim) return 0;
        work*=(uint32_t)source->value_dim;
    }
    if (source->first_unit>work || (source->unit_count && source->unit_count>work-source->first_unit)) return 0;
    if (source->unit_count) work=source->unit_count;
    uint64_t groups=(work+63u)/64u;
    uint32_t threads=64u,shared_bytes=0u;
    CUfunction function=CUDA_FORWARD_FUNCTION(k_llm);
    VxCudaPackedLaunchPlan packed_plan;
    if(source->kind==VX_OP_PACKED_LINEAR) {
        uint32_t arch=0u;
#if defined(VOLVOXAI_CUDA_ARCH)
        arch=VOLVOXAI_CUDA_ARCH;
#endif
        if(source->value_dim<=0)return 0;
        int aligned=source->first_unit%(uint32_t)source->value_dim==0u && work%(uint32_t)source->value_dim==0u;
        uint64_t rows=aligned?work/(uint32_t)source->value_dim:(uint64_t)source->batch*source->sequence;
        if(!vx_cuda_packed_select(rows,work,source->key_dim,source->value_dim,source->rate,
                source->normalize,source->groups,arch,source->interleaved,
                !source->first_unit&&!source->unit_count,aligned,cuda_device_state.multiprocessors,
                cuda_max_shared_memory_per_block,&packed_plan))return 0;
        function=cuda_packed_function(packed_plan.variant);
        if(!function)return 0;
        int prepared=cuda_graph_packed_prepared(source,&device,work,&packed_plan);
        if(prepared<0)return -1;
        if(prepared) {
            for(int i=0;i<3;i++)if(source->y[i])graph_mark_device(outputs[i].slot);
            return 1;
        }
        groups=packed_plan.grid;threads=packed_plan.threads;shared_bytes=packed_plan.shared_bytes;
    }
    if (!work || groups>UINT32_MAX) return 0;
    arguments[0]=&device;arguments[1]=&packed_plan.execution;
    if (!cuda_launch(function,(uint32_t)groups,1,1,threads,1,1,shared_bytes,arguments)) return 0;
    for (int i=0;i<3;i++) if(source->y[i]) graph_mark_device(outputs[i].slot);
    return 1;
}

int cuda_graph_ngram_ids_rows_i32(const int32_t* tokens,const int32_t* multipliers,const int32_t* sizes,
        const int32_t* offsets,int32_t* output,uint32_t batch,uint32_t sequence,uint32_t n,
        uint32_t p,int32_t eos,int32_t pad,int32_t vocabulary,uint32_t row_start,uint32_t row_count) {
    CudaTensorView views[5];void* arguments[14];
    uint64_t heads=n>=2?(uint64_t)(n-1)*p:0,rows=(uint64_t)batch*sequence;
    if(!tokens||!multipliers||!sizes||!offsets||!output||!batch||!sequence||n<2||!p||
        !heads||heads>INT32_MAX||vocabulary<=0||eos< -1||eos>=vocabulary||pad<0||pad>=vocabulary||rows>UINT32_MAX/heads||
        (uint64_t)n*2>SIZE_MAX/4 || !row_count || row_start>=rows || row_count>rows-row_start) return 0;
    if(!graph_ensure_input_view(tokens,(size_t)rows*4,0,&views[0]) ||
       !graph_ensure_input_view(multipliers,(size_t)n*8,1,&views[1]) ||
       !graph_ensure_input_view(sizes,(size_t)heads*4,1,&views[2]) ||
       !graph_ensure_input_view(offsets,(size_t)heads*4,1,&views[3]) ||
       !graph_output_view(output,(size_t)(rows*heads)*4,&views[4])) return 0;
    for(int i=0;i<5;i++) arguments[i]=&views[i].device;
    arguments[5]=&batch;arguments[6]=&sequence;arguments[7]=&n;arguments[8]=&p;arguments[9]=&eos;arguments[10]=&pad;arguments[11]=&vocabulary;
    arguments[12]=&row_start;arguments[13]=&row_count;
    if(!cuda_launch(CUDA_FORWARD_FUNCTION(k_ngram_ids),(uint32_t)(((uint64_t)row_count*heads+63)/64),1,1,64,1,1,0,arguments)) return 0;
    graph_mark_device(views[4].slot);return 1;
}

int cuda_graph_ngram_ids_i32(const int32_t* tokens,const int32_t* multipliers,const int32_t* sizes,
        const int32_t* offsets,int32_t* output,uint32_t batch,uint32_t sequence,uint32_t n,
        uint32_t p,int32_t eos,int32_t pad,int32_t vocabulary) {
    if((uint64_t)batch*sequence>UINT32_MAX) return 0;
    return cuda_graph_ngram_ids_rows_i32(tokens,multipliers,sizes,offsets,output,batch,sequence,n,p,eos,pad,vocabulary,0,batch*sequence);
}
