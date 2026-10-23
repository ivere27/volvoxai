/* Complete-domain F32 contracts for recurrent/sparse attention and hyper
 * connections. Token/batch symbols are preserved; feature banks are fixed.
 * Numerical values (positions, masks and state contents) are never inferred
 * from a bootstrap execution. */
static int32_t vx_sd_llm_error(VxShapeDomainBuild* b, VxShapeDomainErrorCode code) {
    return vx_sd_fail(b,code,VX_SHAPE_DOMAIN_INDEX_NONE,VX_SHAPE_DOMAIN_INDEX_NONE);
}
static int vx_sd_llm_equal(VxShapeDomainBuild* b, VxShapeDomainDimension a, VxShapeDomainDimension c) {
    return vx_shape_domain_dimensions_equal(&a,&c,b->request->environment,b->request->environment_count);
}
static int vx_sd_llm_fixed(VxShapeDomainBuild* b,VxShapeDomainDimension d,uint64_t* value) {
    return vx_shape_domain_fixed_value(&d,b->request->environment,b->request->environment_count,value) &&
        *value>0 && *value<=INT32_MAX;
}
static const VxShapeDomainTensorDescriptor* vx_sd_llm_input(VxShapeDomainBuild* b,const char* name) {
    const VxShapeDomainNamedTensor* t=vx_sd_input(b->request,name);
    return t?&t->descriptor:NULL;
}
static int vx_sd_llm_shape(VxShapeDomainBuild* b,const VxShapeDomainTensorDescriptor* d,
        const VxShapeDomainDimension* shape,uint32_t rank) {
    if (!d || d->rank!=rank) return 0;
    for (uint32_t axis=0;axis<rank;++axis) if (!vx_sd_llm_equal(b,d->dimensions[axis],shape[axis])) return 0;
    return 1;
}
static int32_t vx_sd_llm_float_inputs(VxShapeDomainBuild* b) {
    for(uint32_t i=0;i<b->request->input_count;++i) {
        const VxShapeDomainNamedTensor* t=&b->request->inputs[i];
        if (vx_sd_string_equal(&t->name,"mask") || vx_sd_string_equal(&t->name,"position_ids") || vx_sd_string_equal(&t->name,"candidates") || vx_sd_string_equal(&t->name,"compressed_mask")) {
            if(t->descriptor.dtype!=VX_DTYPE_I32 || t->descriptor.quantization.scheme)
                return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE);
        } else if(vx_sd_assert_float(b,&t->descriptor,i)) return VX_SHAPE_DOMAIN_STATUS_INVALID;
    }
    return VX_SHAPE_DOMAIN_STATUS_OK;
}
static int32_t vx_sd_llm_integer(VxShapeDomainBuild* b,const char* name,int64_t fallback,
        int64_t minimum,int64_t* result) {
    if(vx_sd_number_integer(b,vx_sd_param(b->request,name),fallback,result) ||
        *result<minimum || *result>INT32_MAX) return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}
static int32_t vx_sd_llm_gdn(VxShapeDomainBuild* b) {
    static const char* const required[]={"q","k","v","a","b","a_log","dt_bias"};
    static const char* const optional[]={"initial_state"};
    static const char* const outputs[]={"out","state"};
    static const char* const params[]={"normalize","eps","scale"};
    uint32_t output_count=b->request->declared_output_count;
    if((output_count!=1 && output_count!=2) ||
        vx_sd_optional_ports(b,required,7,optional,1,outputs,output_count) ||
        vx_sd_allowed_params(b,params,3) || vx_sd_llm_float_inputs(b)) return VX_SHAPE_DOMAIN_STATUS_INVALID;
    const VxShapeDomainTensorDescriptor* q=vx_sd_llm_input(b,"q"),*k=vx_sd_llm_input(b,"k"),*v=vx_sd_llm_input(b,"v");
    if(q->rank!=4 || v->rank!=4 || !vx_sd_shapes_equal(q,k,b->request->environment,b->request->environment_count) ||
        !vx_sd_llm_equal(b,q->dimensions[0],v->dimensions[0]) || !vx_sd_llm_equal(b,q->dimensions[1],v->dimensions[1]))
        return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH);
    uint64_t hk,hv,dk,dv;
    if(!vx_sd_llm_fixed(b,q->dimensions[2],&hk) || !vx_sd_llm_fixed(b,v->dimensions[2],&hv) ||
        !vx_sd_llm_fixed(b,q->dimensions[3],&dk) || !vx_sd_llm_fixed(b,v->dimensions[3],&dv) || hv%hk)
        return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_UNPROVABLE);
    VxShapeDomainDimension gates[]={q->dimensions[0],q->dimensions[1],vx_sd_fixed(hv)};
    VxShapeDomainDimension channels[]={vx_sd_fixed(hv)};
    VxShapeDomainDimension state[]={q->dimensions[0],vx_sd_fixed(hv),vx_sd_fixed(dk),vx_sd_fixed(dv)};
    if(!vx_sd_llm_shape(b,vx_sd_llm_input(b,"a"),gates,3) || !vx_sd_llm_shape(b,vx_sd_llm_input(b,"b"),gates,3) ||
        !vx_sd_llm_shape(b,vx_sd_llm_input(b,"a_log"),channels,1) || !vx_sd_llm_shape(b,vx_sd_llm_input(b,"dt_bias"),channels,1) ||
        (vx_sd_llm_input(b,"initial_state") && !vx_sd_llm_shape(b,vx_sd_llm_input(b,"initial_state"),state,4)))
        return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH);
    int normalize;
    if(vx_sd_ext_model_boolean(b,"normalize",1,&normalize) ||
        vx_sd_ext_model_optional_positive_number(b,"eps") || vx_sd_ext_model_optional_positive_number(b,"scale")) return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if(vx_sd_set_output(b,"out",VX_DTYPE_F32,v->dimensions,4,NULL)) return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if(output_count==2) return vx_sd_set_output(b,"state",VX_DTYPE_F32,state,4,NULL);
    return VX_SHAPE_DOMAIN_STATUS_OK;
}
static int32_t vx_sd_llm_sparse(VxShapeDomainBuild* b) {
    static const char* const required[]={"q","k","v"};
    static const char* const optional[]={"mask","position_ids","compressed_kv","sinks","index_q","index_k","index_weights","compressed_mask"};
    static const char* const outputs[]={"out"};
    static const char* const params[]={"causal","scale","window_size","compress_rate","top_k"};
    if(vx_sd_optional_ports(b,required,3,optional,8,outputs,1) || vx_sd_allowed_params(b,params,5) || vx_sd_llm_float_inputs(b)) return VX_SHAPE_DOMAIN_STATUS_INVALID;
    const VxShapeDomainTensorDescriptor* q=vx_sd_llm_input(b,"q"),*k=vx_sd_llm_input(b,"k"),*v=vx_sd_llm_input(b,"v");
    if(q->rank!=4 || k->rank!=4 || v->rank!=4 ||
        !vx_sd_llm_equal(b,q->dimensions[0],k->dimensions[0]) ||
        !vx_sd_llm_equal(b,q->dimensions[3],k->dimensions[3])) return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH);
    for(uint32_t axis=0;axis<3;++axis) if(!vx_sd_llm_equal(b,k->dimensions[axis],v->dimensions[axis])) return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH);
    uint64_t hq,hk,dk,dv;
    if(!vx_sd_llm_fixed(b,q->dimensions[2],&hq) || !vx_sd_llm_fixed(b,k->dimensions[2],&hk) ||
        !vx_sd_llm_fixed(b,q->dimensions[3],&dk) || !vx_sd_llm_fixed(b,v->dimensions[3],&dv) || hq%hk)
        return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_UNPROVABLE);
    int causal;int64_t window,rate,topk;
    if(vx_sd_ext_model_boolean(b,"causal",1,&causal) || vx_sd_ext_model_optional_positive_number(b,"scale") ||
        vx_sd_llm_integer(b,"window_size",0,0,&window) || vx_sd_llm_integer(b,"compress_rate",1,1,&rate) ||
        vx_sd_llm_integer(b,"top_k",0,0,&topk)) return VX_SHAPE_DOMAIN_STATUS_INVALID;
    const VxShapeDomainTensorDescriptor* positions=vx_sd_llm_input(b,"position_ids"),*mask=vx_sd_llm_input(b,"mask");
    VxShapeDomainDimension pos[]={q->dimensions[1]};
    if(positions && !vx_sd_llm_shape(b,positions,pos,1)) return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH);
    if(mask) {
        VxShapeDomainDimension ks[]={k->dimensions[1]},bk[]={q->dimensions[0],k->dimensions[1]},
            qk[]={q->dimensions[1],k->dimensions[1]},bqk[]={q->dimensions[0],q->dimensions[1],k->dimensions[1]};
        if(!vx_sd_llm_shape(b,mask,ks,1) && !vx_sd_llm_shape(b,mask,bk,2) &&
            !vx_sd_llm_shape(b,mask,qk,2) && !vx_sd_llm_shape(b,mask,bqk,3)) return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH);
    }
    const VxShapeDomainTensorDescriptor* compressed=vx_sd_llm_input(b,"compressed_kv"),*sinks=vx_sd_llm_input(b,"sinks"),
        *iq=vx_sd_llm_input(b,"index_q"),*ik=vx_sd_llm_input(b,"index_k"),*iw=vx_sd_llm_input(b,"index_weights");
    if((compressed && !vx_sd_param(b->request,"compress_rate")) ||
        (!compressed && (vx_sd_param(b->request,"compress_rate") || vx_sd_param(b->request,"top_k"))) ||
        (iq && (!vx_sd_param(b->request,"top_k") || topk<=0)))
        return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS);
    VxShapeDomainDimension cs[]={q->dimensions[0],k->dimensions[1],vx_sd_fixed(dk)},ss[]={vx_sd_fixed(hq)};
    if((sinks && !vx_sd_llm_shape(b,sinks,ss,1)) ||
        (compressed && (!vx_sd_llm_shape(b,compressed,cs,3) || hk!=1 || dv!=dk || !vx_sd_llm_equal(b,q->dimensions[1],k->dimensions[1]))))
        return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH);
    if((!!iq+!!ik+!!iw)!=0 && (!!iq+!!ik+!!iw)!=3) return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_INVALID_PORTS);
    if(iq) {
        uint64_t hi,di;
        if(!compressed || iq->rank!=4 || !vx_sd_llm_equal(b,iq->dimensions[0],q->dimensions[0]) ||
            !vx_sd_llm_equal(b,iq->dimensions[1],q->dimensions[1]) || !vx_sd_llm_fixed(b,iq->dimensions[2],&hi) ||
            !vx_sd_llm_fixed(b,iq->dimensions[3],&di)) return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH);
        VxShapeDomainDimension iks[]={q->dimensions[0],k->dimensions[1],vx_sd_fixed(di)},iws[]={q->dimensions[0],q->dimensions[1],vx_sd_fixed(hi)};
        if(!vx_sd_llm_shape(b,ik,iks,3) || !vx_sd_llm_shape(b,iw,iws,3)) return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH);
    }
    const VxShapeDomainTensorDescriptor* selected=vx_sd_llm_input(b,"compressed_mask");
    VxShapeDomainDimension selected_shape[]={q->dimensions[0],q->dimensions[1],k->dimensions[1]};
    if(selected && (!compressed || iq || ik || iw || !vx_sd_llm_shape(b,selected,selected_shape,3)))
        return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH);
    VxShapeDomainDimension out[]={q->dimensions[0],q->dimensions[1],vx_sd_fixed(hq),vx_sd_fixed(dv)};
    return vx_sd_set_output(b,"out",VX_DTYPE_F32,out,4,NULL);
}
static int32_t vx_sd_llm_sparse_indexer(VxShapeDomainBuild* b) {
    static const char* const required[]={"q","k","index_weights"};
    static const char* const optional[]={"candidates"};
    static const char* const outputs[]={"out","candidates"};
    static const char* const params[]={"compress_rate","top_k","candidate_block_size","candidate_top_k"};
    uint32_t count=b->request->declared_output_count;
    if((count!=1&&count!=2)||vx_sd_optional_ports(b,required,3,optional,1,outputs,count)||
       vx_sd_allowed_params(b,params,4)||vx_sd_llm_float_inputs(b))return VX_SHAPE_DOMAIN_STATUS_INVALID;
    const VxShapeDomainTensorDescriptor* q=vx_sd_llm_input(b,"q");
    uint64_t heads,dim;
    if(q->rank!=4||!vx_sd_llm_fixed(b,q->dimensions[2],&heads)||!vx_sd_llm_fixed(b,q->dimensions[3],&dim))
        return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH);
    VxShapeDomainDimension k[]={q->dimensions[0],q->dimensions[1],vx_sd_fixed(dim)},
        weights[]={q->dimensions[0],q->dimensions[1],vx_sd_fixed(heads)},
        out[]={q->dimensions[0],q->dimensions[1],q->dimensions[1]};
    const VxShapeDomainTensorDescriptor* candidates=vx_sd_llm_input(b,"candidates");
    if(!vx_sd_llm_shape(b,vx_sd_llm_input(b,"k"),k,3)||!vx_sd_llm_shape(b,vx_sd_llm_input(b,"index_weights"),weights,3)||
       (candidates&&!vx_sd_llm_shape(b,candidates,out,3)))return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH);
    int64_t rate,topk,block,budget;
    if(vx_sd_llm_integer(b,"compress_rate",0,1,&rate)||vx_sd_llm_integer(b,"top_k",0,1,&topk)||
       vx_sd_llm_integer(b,"candidate_block_size",0,0,&block)||vx_sd_llm_integer(b,"candidate_top_k",0,0,&budget))return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if((count==2&&(!block||!budget||candidates)) ||
       (count==1&&(vx_sd_param(b->request,"candidate_block_size")||vx_sd_param(b->request,"candidate_top_k"))))
        return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS);
    if(vx_sd_set_output(b,"out",VX_DTYPE_I32,out,3,NULL))return VX_SHAPE_DOMAIN_STATUS_INVALID;
    return count==2?vx_sd_set_output(b,"candidates",VX_DTYPE_I32,out,3,NULL):VX_SHAPE_DOMAIN_STATUS_OK;
}
static int32_t vx_sd_llm_compress(VxShapeDomainBuild* b) {
    static const char* const required[]={"input","a","bias","weight"};
    static const char* const optional[]={"cos","sin"};
    static const char* const outputs[]={"out"};
    static const char* const params[]={"num_groups","compress_rate","eps","rotary_dim","position_offset","interleaved"};
    if(vx_sd_optional_ports(b,required,4,optional,2,outputs,1) || vx_sd_allowed_params(b,params,6) || vx_sd_llm_float_inputs(b)) return VX_SHAPE_DOMAIN_STATUS_INVALID;
    const VxShapeDomainTensorDescriptor* x=vx_sd_llm_input(b,"input"),*a=vx_sd_llm_input(b,"a"),*bias=vx_sd_llm_input(b,"bias"),*w=vx_sd_llm_input(b,"weight");
    uint64_t width,feature;int64_t groups,rate;
    if(!vx_sd_param(b->request,"compress_rate") || vx_sd_llm_integer(b,"num_groups",1,1,&groups) || groups>2 || vx_sd_llm_integer(b,"compress_rate",0,1,&rate) ||
        vx_sd_ext_model_optional_positive_number(b,"eps")) return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS);
    if(x->rank!=3 || w->rank!=1 || !vx_sd_shapes_equal(x,a,b->request->environment,b->request->environment_count) ||
        !vx_sd_llm_fixed(b,x->dimensions[2],&feature) || !vx_sd_llm_fixed(b,w->dimensions[0],&width) ||
        feature!=width*(uint64_t)groups) return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH);
    VxShapeDomainDimension bs[]={vx_sd_fixed((uint64_t)rate),vx_sd_fixed(feature)},out[]={x->dimensions[0],x->dimensions[1],vx_sd_fixed(width)};
    if(!vx_sd_llm_shape(b,bias,bs,2)) return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH);
    const VxShapeDomainTensorDescriptor* cos=vx_sd_llm_input(b,"cos"),*sin=vx_sd_llm_input(b,"sin");
    if(!!cos!=!!sin) return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_INVALID_PORTS);
    if(cos) {
        uint64_t capacity,table_width,sequence_max;int64_t rotary,offset;int interleaved;
        if(vx_sd_llm_integer(b,"rotary_dim",(int64_t)width,1,&rotary) ||
            vx_sd_llm_integer(b,"position_offset",0,0,&offset) || vx_sd_ext_model_boolean(b,"interleaved",0,&interleaved) ||
            rotary%2 || (uint64_t)rotary+(uint64_t)offset>width) return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS);
        if(cos->rank!=2 || !vx_sd_shapes_equal(cos,sin,b->request->environment,b->request->environment_count) ||
            !vx_sd_llm_fixed(b,cos->dimensions[0],&capacity) || !vx_sd_llm_fixed(b,cos->dimensions[1],&table_width) ||
            table_width!=(uint64_t)rotary/(interleaved?2u:1u) ||
            vx_sd_ext_model_dimension_maximum(b,&x->dimensions[1],0,1,&sequence_max) || capacity<sequence_max)
            return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH);
    } else if(vx_sd_param(b->request,"rotary_dim") || vx_sd_param(b->request,"position_offset") || vx_sd_param(b->request,"interleaved"))
        return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS);
    return vx_sd_set_output(b,"out",VX_DTYPE_F32,out,3,NULL);
}
static int32_t vx_sd_llm_hyper(VxShapeDomainBuild* b,int head) {
    static const char* const required[]={"input","weight","bias","scale"};
    static const char* const outputs[]={"pre","post","comb"},*const head_outputs[]={"out"};
    static const char* const params[]={"eps","normalization_epsilon","iterations"};
    if(vx_sd_exact_ports(b,required,4,head?head_outputs:outputs,head?1:3) || vx_sd_allowed_params(b,params,head?2:3) || vx_sd_llm_float_inputs(b)) return VX_SHAPE_DOMAIN_STATUS_INVALID;
    const VxShapeDomainTensorDescriptor* x=vx_sd_llm_input(b,"input");
    uint64_t streams,width;
    if(x->rank!=4 || !vx_sd_llm_fixed(b,x->dimensions[2],&streams) || !vx_sd_llm_fixed(b,x->dimensions[3],&width) ||
        streams>INT32_MAX/width || streams>46338) return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_UNPROVABLE);
    uint64_t channels=head?streams:2*streams+streams*streams;
    VxShapeDomainDimension ws[]={vx_sd_fixed(channels),vx_sd_fixed(streams*width)},bs[]={vx_sd_fixed(channels)},ss[]={vx_sd_fixed(head?1:3)};
    if(!vx_sd_llm_shape(b,vx_sd_llm_input(b,"weight"),ws,2) || !vx_sd_llm_shape(b,vx_sd_llm_input(b,"bias"),bs,1) ||
        !vx_sd_llm_shape(b,vx_sd_llm_input(b,"scale"),ss,1)) return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH);
    int64_t iterations;
    if(vx_sd_ext_model_optional_positive_number(b,"eps") || vx_sd_ext_model_optional_positive_number(b,"normalization_epsilon") ||
        (!head && vx_sd_llm_integer(b,"iterations",20,1,&iterations))) return VX_SHAPE_DOMAIN_STATUS_INVALID;
    VxShapeDomainDimension out[]={x->dimensions[0],x->dimensions[1],vx_sd_fixed(width)},post[]={x->dimensions[0],x->dimensions[1],vx_sd_fixed(streams)},
        comb[]={x->dimensions[0],x->dimensions[1],vx_sd_fixed(streams),vx_sd_fixed(streams)};
    if(vx_sd_set_output(b,head?"out":"pre",VX_DTYPE_F32,head?out:post,3,NULL)) return VX_SHAPE_DOMAIN_STATUS_INVALID;
    if(head) return VX_SHAPE_DOMAIN_STATUS_OK;
    if(vx_sd_set_output(b,"post",VX_DTYPE_F32,post,3,NULL)) return VX_SHAPE_DOMAIN_STATUS_INVALID;
    return vx_sd_set_output(b,"comb",VX_DTYPE_F32,comb,4,NULL);
}
static int32_t vx_sd_llm_residual(VxShapeDomainBuild* b) {
    static const char* const required[]={"input","update","post","comb"};
    static const char* const outputs[]={"out"};
    if(vx_sd_exact_ports(b,required,4,outputs,1) || vx_sd_allowed_params(b,NULL,0) || vx_sd_llm_float_inputs(b)) return VX_SHAPE_DOMAIN_STATUS_INVALID;
    const VxShapeDomainTensorDescriptor* x=vx_sd_llm_input(b,"input");uint64_t streams,width;
    if(x->rank!=4 || !vx_sd_llm_fixed(b,x->dimensions[2],&streams) || !vx_sd_llm_fixed(b,x->dimensions[3],&width)) return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_UNPROVABLE);
    VxShapeDomainDimension update[]={x->dimensions[0],x->dimensions[1],vx_sd_fixed(width)},post[]={x->dimensions[0],x->dimensions[1],vx_sd_fixed(streams)},
        comb[]={x->dimensions[0],x->dimensions[1],vx_sd_fixed(streams),vx_sd_fixed(streams)};
    if(!vx_sd_llm_shape(b,vx_sd_llm_input(b,"update"),update,3) || !vx_sd_llm_shape(b,vx_sd_llm_input(b,"post"),post,3) ||
        !vx_sd_llm_shape(b,vx_sd_llm_input(b,"comb"),comb,4)) return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH);
    return vx_sd_set_output(b,"out",VX_DTYPE_F32,x->dimensions,4,NULL);
}
static int32_t vx_sd_llm_rotary(VxShapeDomainBuild* b) {
    static const char* const required[]={"input","cos","sin"};
    static const char* const outputs[]={"out"};
    static const char* const params[]={"rotary_dim","position_offset","interleaved"};
    if(vx_sd_exact_ports(b,required,3,outputs,1) || vx_sd_allowed_params(b,params,3) || vx_sd_llm_float_inputs(b)) return VX_SHAPE_DOMAIN_STATUS_INVALID;
    const VxShapeDomainTensorDescriptor* x=vx_sd_llm_input(b,"input"),*cos=vx_sd_llm_input(b,"cos"),*sin=vx_sd_llm_input(b,"sin");
    uint64_t heads,width;int64_t rotary,offset;int interleaved;
    if(x->rank!=4 || !vx_sd_llm_fixed(b,x->dimensions[2],&heads) || !vx_sd_llm_fixed(b,x->dimensions[3],&width)) return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_UNPROVABLE);
    if(vx_sd_llm_integer(b,"rotary_dim",(int64_t)width,1,&rotary) || vx_sd_llm_integer(b,"position_offset",0,0,&offset) ||
        vx_sd_ext_model_boolean(b,"interleaved",0,&interleaved) || rotary%2 || (uint64_t)rotary+(uint64_t)offset>width)
        return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS);
    VxShapeDomainDimension tables[]={x->dimensions[0],x->dimensions[1],vx_sd_fixed((uint64_t)rotary/(interleaved?2u:1u))};
    if(!vx_sd_llm_shape(b,cos,tables,3) || !vx_sd_llm_shape(b,sin,tables,3)) return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH);
    return vx_sd_set_output(b,"out",VX_DTYPE_F32,x->dimensions,4,NULL);
}
static int32_t vx_sd_llm_ngram(VxShapeDomainBuild* b) {
    static const char* const inputs[]={"token_ids","multipliers","vocab_sizes","offsets"};
    static const char* const outputs[]={"out"};
    static const char* const params[]={"ngram_size","heads_per_ngram","eos_token_id","vocab_size","pad_token_id"};
    if(vx_sd_exact_ports(b,inputs,4,outputs,1) || vx_sd_allowed_params(b,params,5)) return VX_SHAPE_DOMAIN_STATUS_INVALID;
    for(uint32_t i=0;i<4;++i) {
        const VxShapeDomainTensorDescriptor* d=vx_sd_llm_input(b,inputs[i]);
        if(d->dtype!=VX_DTYPE_I32 || d->quantization.scheme) return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_INVALID_DTYPE);
    }
    for(uint32_t i=0;i<4;++i) if(!vx_sd_param(b->request,params[i]))
        return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS);
    int64_t n,p,eos,vocab,pad;
    if(vx_sd_llm_integer(b,"ngram_size",2,2,&n) || vx_sd_llm_integer(b,"heads_per_ngram",1,1,&p) ||
        vx_sd_llm_integer(b,"eos_token_id",-1,-1,&eos) || vx_sd_llm_integer(b,"vocab_size",0,1,&vocab) ||
        vx_sd_llm_integer(b,"pad_token_id",eos,0,&pad) || eos>=vocab || pad>=vocab ||
        (uint64_t)(n-1)*(uint64_t)p>INT32_MAX) return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS);
    const VxShapeDomainTensorDescriptor* tokens=vx_sd_llm_input(b,"token_ids");
    if(tokens->rank!=2) return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_INVALID_RANK);
    uint64_t heads=(uint64_t)(n-1)*(uint64_t)p;
    VxShapeDomainDimension ms[]={vx_sd_fixed((uint64_t)n),vx_sd_fixed(2)},hs[]={vx_sd_fixed(heads)},
        result[]={tokens->dimensions[0],tokens->dimensions[1],vx_sd_fixed(heads)};
    if(!vx_sd_llm_shape(b,vx_sd_llm_input(b,"multipliers"),ms,2) ||
        !vx_sd_llm_shape(b,vx_sd_llm_input(b,"vocab_sizes"),hs,1) || !vx_sd_llm_shape(b,vx_sd_llm_input(b,"offsets"),hs,1))
        return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH);
    return vx_sd_set_output(b,"out",VX_DTYPE_I32,result,3,NULL);
}
static int32_t vx_sd_llm_qsa(VxShapeDomainBuild* b) {
    static const char* const required[]={"q","k","weight","cos","sin"};
    static const char* const optional[]={"mask"},*const outputs[]={"out"};
    static const char* const params[]={"compress_rate","top_k","rotary_dim","eps","causal"};
    if(vx_sd_optional_ports(b,required,5,optional,1,outputs,1) || vx_sd_allowed_params(b,params,5) ||
        vx_sd_llm_float_inputs(b)) return VX_SHAPE_DOMAIN_STATUS_INVALID;
    const VxShapeDomainTensorDescriptor* q=vx_sd_llm_input(b,"q"),*k=vx_sd_llm_input(b,"k");
    uint64_t heads,width;int64_t rate,topk,rotary;int causal;
    if(q->rank!=4 || k->rank!=3 || !vx_sd_llm_equal(b,q->dimensions[0],k->dimensions[0]) ||
        !vx_sd_llm_equal(b,q->dimensions[3],k->dimensions[2]) ||
        !vx_sd_llm_fixed(b,q->dimensions[2],&heads) || !vx_sd_llm_fixed(b,q->dimensions[3],&width) || width>512)
        return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH);
    if(!vx_sd_param(b->request,"compress_rate") || !vx_sd_param(b->request,"top_k") ||
        vx_sd_llm_integer(b,"compress_rate",0,1,&rate) || vx_sd_llm_integer(b,"top_k",0,1,&topk) ||
        vx_sd_llm_integer(b,"rotary_dim",(int64_t)width,1,&rotary) || (rotary&1) || (uint64_t)rotary>width ||
        vx_sd_ext_model_optional_positive_number(b,"eps") || vx_sd_ext_model_boolean(b,"causal",1,&causal))
        return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS);
    VxShapeDomainDimension ws[]={vx_sd_fixed(width)},tables[]={k->dimensions[0],k->dimensions[1],vx_sd_fixed((uint64_t)rotary)},
        out[]={q->dimensions[0],q->dimensions[1],k->dimensions[1]},bk[]={q->dimensions[0],k->dimensions[1]};
    if(!vx_sd_llm_shape(b,vx_sd_llm_input(b,"weight"),ws,1) ||
        !vx_sd_llm_shape(b,vx_sd_llm_input(b,"cos"),tables,3) || !vx_sd_llm_shape(b,vx_sd_llm_input(b,"sin"),tables,3))
        return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH);
    const VxShapeDomainTensorDescriptor* mask=vx_sd_llm_input(b,"mask");
    if(mask && !vx_sd_llm_shape(b,mask,bk,2) && !vx_sd_llm_shape(b,mask,out,3))
        return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH);
    return vx_sd_set_output(b,"out",VX_DTYPE_I32,out,3,NULL);
}
static int32_t vx_sd_prove_llm(VxShapeDomainBuild* b) {
    b->result->fact_kind=VX_INTERNAL_SHAPE_DOMAIN_FACT_DIRECT_PROJECT;
    switch(b->request->operator_kind) {
        case VX_OP_PACKED_LINEAR: {
            static const char* const required[]={"input","weight","scale"};
            static const char* const optional[]={"bias"},* const outputs[]={"out"};
            static const char* const params[]={"group_size","weight_bits"};
            if(vx_sd_optional_ports(b,required,3,optional,1,outputs,1)||vx_sd_allowed_params(b,params,2))
                return VX_SHAPE_DOMAIN_STATUS_INVALID;
            const VxShapeDomainTensorDescriptor* input=vx_sd_llm_input(b,"input");
            const VxShapeDomainTensorDescriptor* weight=vx_sd_llm_input(b,"weight");
            const VxShapeDomainTensorDescriptor* scale=vx_sd_llm_input(b,"scale");
            const VxShapeDomainTensorDescriptor* bias=vx_sd_llm_input(b,"bias");
            uint64_t k,n; int64_t group,weight_bits;
            if(input->rank<1||input->rank>8||weight->rank!=2||weight->dtype!=VX_DTYPE_I32||weight->quantization.scheme||
                vx_sd_assert_float(b,input,0)||vx_sd_assert_float(b,scale,2)||(bias&&vx_sd_assert_float(b,bias,3))||
                !vx_sd_llm_fixed(b,input->dimensions[input->rank-1],&k)||!vx_sd_llm_fixed(b,weight->dimensions[0],&n))
                return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH);
            if(vx_sd_llm_integer(b,"group_size",0,1,&group)||group>INT32_MAX||
                !vx_packed_group_size_supported((int)group)||
                vx_sd_llm_integer(b,"weight_bits",4,0,&weight_bits)||(weight_bits!=2&&weight_bits!=3&&weight_bits!=4&&weight_bits!=8))
                return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS);
            VxShapeDomainDimension packed[]={vx_sd_fixed(n),vx_sd_fixed((k/32u)*(uint64_t)weight_bits+((k%32u)*(uint64_t)weight_bits+31u)/32u)};
            VxShapeDomainDimension scales[]={vx_sd_fixed(n),vx_sd_fixed(k/(uint64_t)group+(k%(uint64_t)group!=0))};
            if(!vx_sd_llm_shape(b,weight,packed,2)||!vx_sd_llm_shape(b,scale,scales,2)||
                (bias&&!vx_sd_llm_shape(b,bias,packed,1)))
                return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH);
            VxShapeDomainDimension out[8];
            for(uint32_t axis=0;axis<input->rank;axis++)out[axis]=input->dimensions[axis];
            out[input->rank-1]=vx_sd_fixed(n);
            return vx_sd_set_output(b,"out",VX_DTYPE_F32,out,input->rank,NULL);
        }
        case VX_OP_EMBEDDING_SHARD: {
            static const char* const required[]={"input","weight"},* const outputs[]={"out"},* const params[]={"vocab_size","vocab_offset"};
            if(vx_sd_exact_ports(b,required,2,outputs,1)||vx_sd_allowed_params(b,params,2)) return VX_SHAPE_DOMAIN_STATUS_INVALID;
            const VxShapeDomainTensorDescriptor* input=vx_sd_llm_input(b,"input"),*weight=vx_sd_llm_input(b,"weight");
            uint64_t rows,width;int64_t offset,vocab;
            if(input->dtype!=VX_DTYPE_I32||input->quantization.scheme||input->rank>7||weight->rank!=2||
                vx_sd_assert_float(b,weight,1)||!vx_sd_llm_fixed(b,weight->dimensions[0],&rows)||!vx_sd_llm_fixed(b,weight->dimensions[1],&width)) return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_SHAPE_MISMATCH);
            if(!vx_sd_param(b->request,"vocab_offset")||!vx_sd_param(b->request,"vocab_size")||
                vx_sd_llm_integer(b,"vocab_offset",-1,0,&offset)||vx_sd_llm_integer(b,"vocab_size",0,1,&vocab)||vocab>INT32_MAX||(uint64_t)offset+rows>(uint64_t)vocab) return vx_sd_llm_error(b,VX_SHAPE_DOMAIN_ERROR_INVALID_PARAMS);
            VxShapeDomainDimension out[8];
            for(uint32_t axis=0;axis<input->rank;axis++) out[axis]=input->dimensions[axis];
            out[input->rank]=vx_sd_fixed(width);
            return vx_sd_set_output(b,"out",VX_DTYPE_F32,out,input->rank+1,NULL);
        }
        case VX_OP_SIGNED_SQRT: {
            static const char* const required[]={"input"},* const outputs[]={"out"},* const params[]={"eps"};
            if(vx_sd_exact_ports(b,required,1,outputs,1)||vx_sd_allowed_params(b,params,1)||vx_sd_llm_float_inputs(b)||vx_sd_ext_model_optional_positive_number(b,"eps")) return VX_SHAPE_DOMAIN_STATUS_INVALID;
            const VxShapeDomainTensorDescriptor* input=vx_sd_llm_input(b,"input");
            return vx_sd_set_output(b,"out",VX_DTYPE_F32,input->dimensions,input->rank,NULL);
        }
        case VX_OP_N_GRAM_IDS:return vx_sd_llm_ngram(b);
        case VX_OP_SPARSE_INDEXER:return vx_sd_llm_sparse_indexer(b);
        case VX_OP_QSA_INDEXER:return vx_sd_llm_qsa(b);
        case VX_OP_GATED_DELTA_NET:return vx_sd_llm_gdn(b);
        case VX_OP_SPARSE_ATTENTION:return vx_sd_llm_sparse(b);
        case VX_OP_KV_COMPRESS:return vx_sd_llm_compress(b);
        case VX_OP_HYPER_CONNECTION:return vx_sd_llm_hyper(b,0);
        case VX_OP_HYPER_HEAD:return vx_sd_llm_hyper(b,1);
        case VX_OP_HYPER_RESIDUAL:return vx_sd_llm_residual(b);
        case VX_OP_ROTARY_EMBEDDING:return vx_sd_llm_rotary(b);
        default:return VX_SHAPE_DOMAIN_STATUS_UNSUPPORTED;
    }
}
