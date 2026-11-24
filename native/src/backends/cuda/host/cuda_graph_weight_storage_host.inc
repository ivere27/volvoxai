/* Compact immutable sources bind their physical byte span. Host mirrors and
 * device allocations retain the original storage; only kernel loads widen. */
int cuda_graph_matmul_storage_f32(const float* input, const void* weight,
        uint32_t dtype, const float* bias, float* output,
        int rows, int d_in, int d_out, int out_in) {
    if(dtype==VX_DTYPE_F32)return cuda_graph_matmul_layout_f32(input,weight,bias,
        output,rows,d_in,d_out,out_in);
    uint32_t size=cuda_weight_storage_bytes(dtype);
    size_t ib,wb,ob; CudaTensorView iv,wv,bv,ov;
    if(!size||!input||!weight||!output||rows<=0||d_in<=0||d_out<=0||
       (out_in!=0&&out_in!=1)||
       !size_product_u32(rows,d_in,4,&ib)||!size_product_u32(d_in,d_out,size,&wb)||
       !size_product_u32(rows,d_out,4,&ob)||ranges_overlap(output,ob,input,ib)||
       ranges_overlap(output,ob,weight,wb)||
       (bias&&ranges_overlap(output,ob,bias,(size_t)d_out*4)))return 0;
    if(!graph_ensure_input_view(input,ib,0,&iv)||!graph_ensure_input_view(weight,wb,1,&wv)||
       !graph_output_view(output,ob,&ov))return 0;
    CUdeviceptr bd=0;
    if(bias){if(!graph_ensure_input_view(bias,(size_t)d_out*4,1,&bv))return 0;bd=bv.device;}
    uint32_t r=rows,k=d_in,n=d_out,layout=out_in,has_bias=bias!=NULL;
    void* args[]={&iv.device,&wv.device,&bd,&ov.device,&r,&k,&n,&layout,&has_bias,&dtype};
    if(!cuda_launch(CUDA_FORWARD_FUNCTION(k_matmul_storage),(n+15u)/16u,(r+15u)/16u,
        1,16,16,1,0,args))return 0;
    graph_mark_device(ov.slot);return 1;
}

int cuda_graph_embedding_storage_f32(const int32_t* tokens,const void* weight,
        uint32_t dtype,float* output,int token_count,int width,int vocab_size) {
    if(dtype==VX_DTYPE_F32)return cuda_graph_embedding_f32(tokens,weight,output,
        token_count,width,vocab_size);
    uint32_t size=cuda_weight_storage_bytes(dtype);
    size_t wb,ob;CudaTensorView tv,wv,ov;
    if(!size||!tokens||!weight||!output||token_count<=0||width<=0||vocab_size<=0||
       !size_product_u32(vocab_size,width,size,&wb)||
       !size_product_u32(token_count,width,4,&ob)||
       ranges_overlap(output,ob,weight,wb)||ranges_overlap(output,ob,tokens,(size_t)token_count*4))return 0;
    if(!vx_engine_state_current()->bounded_gpu_value_domain_proven)
        for(int i=0;i<token_count;i++)if(tokens[i]<0||tokens[i]>=vocab_size)return 0;
    if(!graph_ensure_input_view(tokens,(size_t)token_count*4,0,&tv)||
       !graph_ensure_input_view(weight,wb,1,&wv)||!graph_output_view(output,ob,&ov))return 0;
    uint32_t count=token_count,row_width=width;
    void* args[]={&tv.device,&wv.device,&ov.device,&count,&row_width,&dtype};
    if(!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_embedding_storage),count*row_width,args))return 0;
    graph_mark_device(ov.slot);return 1;
}

int cuda_graph_gather_storage_f32(const void* input,uint32_t dtype,
        const int32_t* indices,float* output,int outer,int axis_size,
        int inner,int indices_elements,int output_elements) {
    if(dtype==VX_DTYPE_F32)return cuda_graph_gather_i32_f32(input,indices,output,
        outer,axis_size,inner,indices_elements,output_elements);
    uint32_t size=cuda_weight_storage_bytes(dtype),count;
    size_t ib,ob;CudaTensorView iv,xv,ov;
    if(!size||!input||!indices||!output||outer<=0||axis_size<=0||inner<=0||
       indices_elements<=0||output_elements<=0||
       !checked_product4_bytes(outer,axis_size,inner,1,size,&count,&ib)||
       !checked_product4_bytes(outer,indices_elements,inner,1,4,&count,&ob)||
       count!=(uint32_t)output_elements||ranges_overlap(output,ob,input,ib)||
       ranges_overlap(output,ob,indices,(size_t)indices_elements*4))return 0;
    if(!vx_engine_state_current()->bounded_gpu_value_domain_proven)
        for(int i=0;i<indices_elements;i++)if(indices[i]<-axis_size||indices[i]>=axis_size)return 0;
    if(!graph_ensure_input_view(input,ib,1,&iv)||
       !graph_ensure_input_view(indices,(size_t)indices_elements*4,0,&xv)||
       !graph_output_view(output,ob,&ov))return 0;
    uint32_t o=outer,a=axis_size,n=inner,k=indices_elements;
    void* args[]={&iv.device,&xv.device,&ov.device,&o,&a,&n,&k,&dtype};
    if(!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_gather_storage),count,args))return 0;
    graph_mark_device(ov.slot);return 1;
}

int cuda_graph_moe_linear_storage_f32(const float* input,const void* weight,
        uint32_t dtype,const float* bias,const float* indices,const float* routes,
        float* output,int rows,int d_in,int d_out,int experts,int top_k,
        const unsigned int* slot_rows,unsigned int slot_domain) {
    if(dtype==VX_DTYPE_F32)return cuda_graph_moe_linear_f32(input,weight,bias,indices,
        routes,output,rows,d_in,d_out,experts,top_k,slot_rows,slot_domain);
    uint32_t size=cuda_weight_storage_bytes(dtype);
    if(!size||!input||!weight||!indices||!routes||!output||rows<=0||d_in<=0||d_out<=0||
       experts<=0||top_k<=0||top_k>8||(slot_rows ? slot_domain<(uint32_t)experts||
       (uint32_t)top_k>slot_domain : top_k>experts))return 0;
    CudaMoeCache* cache=cuda_moe_cache_find(weight);
    if(cache) {
        if(slot_rows||cache->hidden||cache->bias!=bias||cache->dtype[0]!=dtype||
           cache->experts!=(uint32_t)experts||cache->d_in!=(uint32_t)d_in||
           cache->d_out!=(uint32_t)d_out)return 0;
        return cuda_moe_cache_execute(cache,input,indices,routes,output,rows,top_k,0,0);
    }
    const void* pointers[]={input,weight,indices,routes};
    size_t bytes[4],ob,bb;uint32_t count;CudaTensorView views[4],ov,bv,sv;
    if(!size_product_u32(rows,d_in,4,&bytes[0])||
       !checked_product4_bytes(experts,d_in,d_out,1,size,&count,&bytes[1])||
       !size_product_u32(rows,top_k,4,&bytes[2])||
       !size_product_u32(rows,d_out,4,&ob)||!size_product_u32(experts,d_out,4,&bb))return 0;
    bytes[3]=bytes[2];
    for(int i=0;i<4;i++)if(ranges_overlap(output,ob,pointers[i],bytes[i]))return 0;
    if(bias&&ranges_overlap(output,ob,bias,bb))return 0;
    for(int i=0;i<4;i++)if(!graph_ensure_input_view(pointers[i],bytes[i],i==1,&views[i]))return 0;
    if(!graph_output_view(output,ob,&ov))return 0;
    CUdeviceptr bd=0,sd=0;
    if(bias){if(!graph_ensure_input_view(bias,bb,1,&bv))return 0;bd=bv.device;}
    uint32_t domain=slot_rows?slot_domain:0;
    if(domain){if(!graph_ensure_input_view(slot_rows,(size_t)domain*4,1,&sv))return 0;sd=sv.device;}
    uint32_t r=rows,k=d_in,n=d_out,e=experts,top=top_k,has_bias=bias!=NULL;
    void* args[]={&views[0].device,&views[1].device,&bd,&views[2].device,&views[3].device,
        &ov.device,&r,&k,&n,&e,&top,&has_bias,&sd,&domain,&dtype};
    if(!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_moe_linear_storage),r*n,args))return 0;
    graph_mark_device(ov.slot);return 1;
}

int cuda_graph_moe_gated_mlp_storage_f32(const float* input,
        const void* gate,uint32_t gate_dtype,const void* up,uint32_t up_dtype,
        const void* down,uint32_t down_dtype,const float* indices,const float* routes,
        float* output,uint32_t rows,uint32_t d_in,uint32_t hidden,uint32_t d_out,
        uint32_t experts,uint32_t top_k,uint32_t input_scale,float limit) {
    if(gate_dtype==VX_DTYPE_F32&&up_dtype==VX_DTYPE_F32&&down_dtype==VX_DTYPE_F32)
        return cuda_graph_moe_gated_mlp_f32(input,gate,up,down,indices,routes,output,
            rows,d_in,hidden,d_out,experts,top_k,input_scale,limit);
    uint32_t sizes[]={4,cuda_weight_storage_bytes(gate_dtype),cuda_weight_storage_bytes(up_dtype),
        cuda_weight_storage_bytes(down_dtype),4,4,4};
    if(!input||!gate||!up||!down||!indices||!routes||!output||!sizes[1]||!sizes[2]||!sizes[3]||
       !rows||!d_in||!hidden||hidden>8192||!d_out||!experts||!top_k||top_k>experts||
       input_scale>1||!isfinite(limit)||limit<0)return 0;
    CudaMoeCache* cache=cuda_moe_cache_find(gate);
    if(cache) {
        if(cache->source[1]!=up||cache->source[2]!=down||cache->dtype[0]!=gate_dtype||
           cache->dtype[1]!=up_dtype||cache->dtype[2]!=down_dtype||cache->experts!=experts||
           cache->d_in!=d_in||cache->hidden!=hidden||cache->d_out!=d_out)return 0;
        return cuda_moe_cache_execute(cache,input,indices,routes,output,rows,top_k,input_scale,limit);
    }
    const void* pointers[]={input,gate,up,down,indices,routes,output};
    uint32_t a[]={rows,experts,experts,experts,rows,rows,rows};
    uint32_t b[]={d_in,d_in,d_in,hidden,top_k,top_k,d_out};
    uint32_t c[]={1,hidden,hidden,d_out,1,1,1};
    size_t bytes[7];CudaTensorView views[7];uint32_t count;
    for(int i=0;i<7;i++)if(!checked_product4_bytes(a[i],b[i],c[i],1,sizes[i],&count,&bytes[i]))return 0;
    for(int i=0;i<6;i++)if(ranges_overlap(output,bytes[6],pointers[i],bytes[i]))return 0;
    for(int i=0;i<7;i++)if(i<6 ?
        !graph_ensure_input_view(pointers[i],bytes[i],i>=1&&i<=3,&views[i]):
        !graph_output_view(output,bytes[i],&views[i]))return 0;
    void* args[]={&views[0].device,&views[1].device,&views[2].device,&views[3].device,
        &views[4].device,&views[5].device,&views[6].device,&rows,&d_in,&hidden,&d_out,
        &experts,&top_k,&input_scale,&limit,&gate_dtype,&up_dtype,&down_dtype};
    if(!cuda_launch(CUDA_FORWARD_FUNCTION(k_moe_gated_mlp_storage),rows,1,1,128,1,1,0,args))return 0;
    graph_mark_device(views[6].slot);return 1;
}
