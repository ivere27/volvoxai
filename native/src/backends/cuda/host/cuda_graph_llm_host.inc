/* Whole-prefix modern decoder kernels use bounded thread-local state and
 * the existing graph tensor views. There is no host compute fallback. */
/* Reuse the context's fixed dense scratch between serial launches. No pointer
 * escapes this context or grows during replay. A tactic miss precedes every
 * launch; once preparation starts, failure is terminal for the operation. */
static int cuda_graph_packed_prepared(const VxLLMOpArgs* source,
        const VxLLMOpArgs* device,uint64_t work,uint32_t threads,uint32_t shared_bytes) {
    if(source->kind!=VX_OP_PACKED_LINEAR || source->key_dim<=0 || source->value_dim<=0 ||
       !vx_cuda_packed_linear_prepares_input((uint32_t)source->normalize,
           (uint32_t)source->value_dim,(uint32_t)device->window) ||
       !source->x[0] || !source->x[1] || !source->x[2] || !source->y[0] ||
       source->first_unit%(uint32_t)source->value_dim || work%(uint32_t)source->value_dim ||
       source->mask || source->positions || source->y[1] || source->y[2])return 0;
    for(int i=4;i<8;i++)if(source->x[i])return 0;
    uint64_t rows=work/(uint32_t)source->value_dim;
    int matrix=device->window==2 || device->window==3;
    uint32_t compiled_arch=0u;
#if defined(VOLVOXAI_CUDA_ARCH)
    compiled_arch=VOLVOXAI_CUDA_ARCH;
#endif
    uint32_t tile_rows=matrix?vx_cuda_packed_linear_prepared_tile_rows(rows,
        (uint32_t)source->key_dim,(uint32_t)source->value_dim,
        (uint32_t)source->normalize,(uint32_t)source->groups,compiled_arch):0u;
    CUfunction compute=CUDA_FORWARD_FUNCTION(k_llm);
    if(tile_rows) {
        uint32_t prepared_threads=vx_cuda_packed_linear_prepared_threads(rows,
            (uint32_t)source->key_dim,(uint32_t)source->value_dim,
            (uint32_t)source->normalize,(uint32_t)source->groups,compiled_arch);
        compute=source->normalize==16?(prepared_threads==256u?
            CUDA_FORWARD_FUNCTION(k_packed_a16_m64_warps8):CUDA_FORWARD_FUNCTION(k_packed_a16_m64)):
            tile_rows==64u?CUDA_FORWARD_FUNCTION(k_packed_a8_m64):
            CUDA_FORWARD_FUNCTION(k_packed_a8_m16);
        if(!compute)tile_rows=0u;
        else threads=prepared_threads;
    }
    if(!tile_rows)compute=CUDA_FORWARD_FUNCTION(k_llm);
    /* Qualified narrow A16 prefixes retain the ordinary compute entry while
     * sharing the complete prepared input. Other shapes keep their 2MiB policy. */
    int narrow_workspace=matrix && vx_cuda_packed_linear_prepared_narrow_workspace(rows,
        (uint32_t)source->key_dim,(uint32_t)source->value_dim,
        (uint32_t)source->normalize,(uint32_t)source->groups,compiled_arch);
    uint64_t workspace_bytes=(tile_rows && rows>=512u) || narrow_workspace?
        VX_CUDA_PACKED_WORKSPACE_BYTES:VX_CUDA_MATMUL_WORKSPACE_BYTES;
    uint64_t first_row=source->first_unit/(uint32_t)source->value_dim;
    uint64_t stride=(((uint64_t)(uint32_t)source->key_dim+127u)/128u)*128u*
        (source->normalize==8?1u:2u);
    if(!rows || stride>workspace_bytes-4u ||
       first_row>UINT64_MAX-rows ||
       first_row+rows>source->x_elements[0]/(uint32_t)source->key_dim ||
       first_row+rows>source->y_elements[0]/(uint32_t)source->value_dim)return 0;
    uint32_t chunk_capacity=(uint32_t)(workspace_bytes/(stride+4u));
    if(matrix && device->heads!=32 && device->heads!=64 && device->heads!=128)return 0;
    if(rows>chunk_capacity) {
        if(workspace_bytes==VX_CUDA_MATMUL_WORKSPACE_BYTES &&
           !vx_cuda_packed_linear_prepares_chunks((uint32_t)source->normalize,
                (uint32_t)source->key_dim,(uint32_t)source->value_dim))return 0;
        if(matrix)chunk_capacity-=chunk_capacity%16u;
    }
    if(!chunk_capacity)return 0;
    uint32_t largest_chunk=rows<chunk_capacity?(uint32_t)rows:chunk_capacity;
    /* The final chunk can choose a narrower tile than the largest one. Use
     * the smallest tile for a conservative bound before either launch. */
    uint64_t column_tiles=((uint64_t)(uint32_t)source->value_dim+31u)/32u;
    uint64_t largest_grid=matrix?((largest_chunk+15u)/16u)*column_tiles:
        source->normalize==8?(uint64_t)largest_chunk*column_tiles:
        ((uint64_t)largest_chunk*(uint32_t)source->value_dim+1u)/2u;
    uint32_t compute_shared=source->normalize==8&&!matrix?0u:shared_bytes;
    if(tile_rows)compute_shared=source->normalize==8?
        tile_rows*36u*4u*(tile_rows==64u?2u:1u):17408u;
    if(largest_grid>UINT32_MAX || compute_shared>UINT32_MAX-160u ||
       !cuda_launch_dimensions_valid(largest_chunk,1u,1u,256u,1u,1u,192u) ||
       !cuda_launch_dimensions_valid((uint32_t)largest_grid,1u,1u,threads,1u,1u,compute_shared+160u))return 0;
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
        uint32_t count=rows-completed<chunk_capacity?(uint32_t)(rows-completed):chunk_capacity;
        VxLLMOpArgs chunk=*device;
        chunk.batch=1;chunk.sequence=(int32_t)count;
        chunk.first_unit=0;chunk.unit_count=0;
        chunk.x[0]=device->x[0]+(first_row+completed)*(uint32_t)source->key_dim;
        chunk.y[0]=device->y[0]+(first_row+completed)*(uint32_t)source->value_dim;
        chunk.x[4]=(const float*)(uintptr_t)state->matmul_workspace;
        chunk.x[5]=(const float*)(uintptr_t)(state->matmul_workspace+(uint64_t)count*stride);
        if(matrix)chunk.heads=(int32_t)vx_cuda_packed_linear_tile_columns(count,
            (uint32_t)source->value_dim,(uint32_t)source->normalize,
            cuda_device_state.multiprocessors);
        if(tile_rows && (source->normalize==8 || tile_rows==64u))chunk.heads=64;
        uint32_t grid=matrix?((count+15u)/16u)*
            (uint32_t)(((uint64_t)(uint32_t)source->value_dim+(uint32_t)chunk.heads-1u)/(uint32_t)chunk.heads):
            source->normalize==8?count*(uint32_t)column_tiles:
            (uint32_t)(((uint64_t)count*(uint32_t)source->value_dim+1u)/2u);
        if(tile_rows)grid=((count+tile_rows-1u)/tile_rows)*
            (uint32_t)(((uint64_t)(uint32_t)source->value_dim+(uint32_t)chunk.heads-1u)/(uint32_t)chunk.heads);
        int window=chunk.window;
        void* arguments[1]={&chunk};
        chunk.window=5;
        if(!cuda_launch(CUDA_FORWARD_FUNCTION(k_llm),count,1u,1u,256u,1u,1u,32u,arguments))return -1;
        chunk.window=window;
        if(!cuda_launch(compute,grid,1u,1u,threads,1u,1u,compute_shared,arguments))return -1;
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
       (activation_bits!=8u && activation_bits!=16u) ||
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
        if(cache && cuda_max_shared_memory_per_block>=cache+160u) {
            shared_bytes=cache;window=4u;
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
    if(!cuda_launch_dimensions_valid(grid,1u,1u,threads,1u,1u,shared_bytes+160u) ||
       (workspace && !cuda_launch_dimensions_valid(1u,1u,1u,256u,1u,1u,192u)))return 0;
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

int cuda_graph_llm_f32(const VxLLMOpArgs* source) {
    VxLLMOpArgs device;
    CudaTensorView inputs[8],outputs[3],mask,positions;
    void* arguments[1];
    uint64_t work;
    if (!source || source->key_dim<=0 ||
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
    uint64_t groups=source->kind==VX_OP_PACKED_LINEAR?(work+1u)/2u:(work+63u)/64u;
    uint32_t threads=64u,shared_bytes=0u;
    if(source->kind==VX_OP_PACKED_LINEAR)device.window=0;
    if(source->kind==VX_OP_PACKED_LINEAR && source->normalize==8 && source->value_dim>=32 &&
       source->first_unit%(uint32_t)source->value_dim==0u && work%(uint32_t)source->value_dim==0u) {
        groups=(work/(uint32_t)source->value_dim)*(((uint32_t)source->value_dim+31u)/32u);
        threads=256u;device.window=1;
        uint32_t cache_bytes=vx_cuda_packed_linear_row_cache_bytes((uint32_t)source->key_dim);
        if(cache_bytes && cuda_max_shared_memory_per_block>=cache_bytes+160u) {
            shared_bytes=cache_bytes;device.window=4;
        }
    }
    if(source->kind==VX_OP_PACKED_LINEAR &&
       (source->normalize==8 || source->normalize==16) &&
       (uint64_t)source->batch*source->sequence>=16u && !source->first_unit && !source->unit_count) {
        uint32_t compiled_arch=0u;
#if defined(VOLVOXAI_CUDA_ARCH)
        compiled_arch=VOLVOXAI_CUDA_ARCH;
#endif
        uint64_t tiles=vx_cuda_packed_linear_matrix_tiles(
            (uint64_t)source->batch*source->sequence,(uint32_t)source->value_dim,
            (uint32_t)source->normalize,cuda_device_state.multiprocessors,compiled_arch);
        /* Each shared tile is one CTA; the selector covers every SM before
         * reducing output parallelism. Scratch is bounded independently of K. */
        if(tiles) {
            groups=tiles;
            threads=source->normalize==8?256u:128u;
            shared_bytes=source->normalize==8?2368u:4416u;
            device.window=source->normalize==8?2:3;
            device.heads=(int32_t)vx_cuda_packed_linear_tile_columns(
                (uint64_t)source->batch*source->sequence,(uint32_t)source->value_dim,
                (uint32_t)source->normalize,cuda_device_state.multiprocessors);
        }
    }
    int prepared=cuda_graph_packed_prepared(source,&device,work,threads,shared_bytes);
    if(prepared<0)return -1;
    if(prepared) {
        for(int i=0;i<3;i++)if(source->y[i])graph_mark_device(outputs[i].slot);
        return 1;
    }
    if (!work || groups>UINT32_MAX) return 0;
    arguments[0]=&device;
    if (!cuda_launch(CUDA_FORWARD_FUNCTION(k_llm),(uint32_t)groups,1,1,threads,1,1,shared_bytes,arguments)) return 0;
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
