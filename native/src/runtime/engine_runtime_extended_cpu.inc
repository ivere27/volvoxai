/* Inference primitives previously invoked directly by the WASM TS planner.
 * Resolve their complete descriptors here for both native and WASM owners. */
static int run_extended_cpu(Node* n,T* in,T* out) {
    VxOperatorKind op=n->operator_kind;
    if (op == VX_OP_CROSS_ATTENTION) {
        T* q=nin(n,VX_PORT_Q),*kv=nin(n,VX_PORT_KV),*w=nin(n,VX_PORT_WEIGHT),*scale=nin(n,VX_PORT_SCALE),*bias=nin(n,VX_PORT_BIAS);
        if(!q || !kv || !w || !out || (q->ndim!=2 && q->ndim!=3) ||
            kv->ndim!=q->ndim || out->ndim!=q->ndim)return -1;
        materialize_tensor_f32(q);materialize_tensor_f32(kv);materialize_tensor_f32(w);
        if(scale)materialize_tensor_f32(scale);if(bias)materialize_tensor_f32(bias);
        int rank=q->ndim,d=q->shape[rank-1],heads=n->parsed_params.heads,batch=rank==3?q->shape[0]:1;
        if(q->dtype!=T_F32 || kv->dtype!=T_F32 || w->dtype!=T_F32 || out->dtype!=T_F32 ||
            d<=0 || heads<=0 || d%heads || d>64 || w->numel!=3L*d*d ||
            !tensor_same_shape(q,out) || kv->shape[rank-1]!=d || (rank==3 && kv->shape[0]!=batch) ||
            (scale && (scale->dtype!=T_F32 || scale->numel!=3L*d)) ||
            (bias && (bias->dtype!=T_F32 || bias->numel!=3L*d)))return -1;
        cross_attention_f32(q->data,kv->data,w->data,scale?scale->data:NULL,bias?bias->data:NULL,out->data,
            batch,q->shape[rank-2],kv->shape[rank-2],d,heads,d/heads);
        return 1;
    }
    if (op == VX_OP_DEQUANTIZE_LINEAR) {
        T* scale=nin(n,VX_PORT_SCALE),*zero=nin(n,VX_PORT_ZERO_POINT);
        if(!in || !out || !scale || out->dtype!=T_F32 || scale->dtype!=T_F32 || scale->numel!=1 ||
            !tensor_same_shape(in,out) || out->numel>INT_MAX || (zero && zero->numel!=1))return -1;
        return dequantize_linear_typed(in->data,in->dtype,scale->data,zero?zero->data:NULL,
            zero?zero->dtype:T_F32,out->data,(int)out->numel)==1?1:-1;
    }
    int conv1d=(op == VX_OP_CONV_1D), conv_transpose=(op == VX_OP_CONV_TRANSPOSE_2D);
    int interp=(op == VX_OP_INTERPOLATE_1D),pad=(op == VX_OP_PAD);
    int px=(op == VX_OP_PROFILE_X),py=(op == VX_OP_PROFILE_Y),mean=(op == VX_OP_MEAN_HEIGHT),soft=(op == VX_OP_SPATIAL_SOFTARGMAX_Y);
    if(!conv1d && !conv_transpose && !interp && !pad && !px && !py && !mean && !soft)return 0;
    if(!in || !out || !in->data || !out->data || out->dtype!=T_F32 || in->numel<=0 || out->numel<=0)return -1;
    materialize_tensor_f32(in);if(in->dtype!=T_F32)return -1;
    if(conv1d || conv_transpose) {
        T* w=nin(n,VX_PORT_WEIGHT),*bias=nin(n,VX_PORT_BIAS);int rank=conv1d?3:4,sy,sx,pt,pl;
        if(!w || w->ndim!=rank || in->ndim!=rank || out->ndim!=rank || in->shape[0]!=out->shape[0] ||
            run_node_param_pair(n,VX_NODE_PARAM_STRIDE,VX_NODE_ARRAY_STRIDE,1,&sy,&sx) ||
            run_node_param_pair(n,VX_NODE_PARAM_PADDING,VX_NODE_ARRAY_PADDING,0,&pt,&pl) ||
            sy<=0 || sx<=0 || pt<0 || pl<0)return -1;
        materialize_tensor_f32(w);if(bias)materialize_tensor_f32(bias);
        if(w->dtype!=T_F32 || (bias && (bias->dtype!=T_F32 || bias->numel!=out->shape[rank-1])))return -1;
        if(conv1d) {
            int groups=vx_node_param_i32(n,VX_NODE_PARAM_GROUPS,1),ci=in->shape[2],co=out->shape[2],kernel=w->shape[0];
            if(groups<=0 || ci%groups || co%groups || w->shape[1]!=ci/groups || w->shape[2]!=co ||
                (int64_t)in->shape[1]+2LL*pt<kernel || ((int64_t)in->shape[1]+2LL*pt-kernel)/sy+1!=out->shape[1])return -1;
            for(int batch=0;batch<in->shape[0];batch++)
                conv1d_f32((uintptr_t)(in->data+(long)batch*in->shape[1]*ci),
                    (uintptr_t)(out->data+(long)batch*out->shape[1]*co),(uintptr_t)w->data,(uintptr_t)(bias?bias->data:NULL),
                    ci,in->shape[1],co,w->shape[1],kernel,sy,pt,groups,vx_node_param_i32(n,VX_NODE_PARAM_RELU,0));
        } else {
            if(w->shape[2]!=in->shape[3] || w->shape[3]!=out->shape[3] ||
                (int64_t)(in->shape[1]-1)*sy+w->shape[0]-2LL*pt!=out->shape[1] ||
                (int64_t)(in->shape[2]-1)*sx+w->shape[1]-2LL*pl!=out->shape[2])return -1;
            conv_transpose2d_f32(in->data,w->data,bias?bias->data:NULL,out->data,in->shape[0],in->shape[1],in->shape[2],in->shape[3],
                out->shape[1],out->shape[2],out->shape[3],w->shape[0],w->shape[1],sy,sx,pt,pl);
        }
    } else if(interp) {
        int rank=in->ndim;
        if((rank!=2 && rank!=3) || out->ndim!=rank)return -1;
        for(int d=0;d<rank-1;d++)if(in->shape[d]!=out->shape[d])return -1;
        long channels=in->numel/in->shape[rank-1];if(channels>INT_MAX)return -1;
        interp1d_f32((uintptr_t)in->data,(uintptr_t)out->data,(int)channels,in->shape[rank-1],out->shape[rank-1]);
    } else if(pad) {
        int rank=in->ndim,count=0;const int32_t* pads=vx_node_param_array(n,VX_NODE_PARAM_PADS,VX_NODE_ARRAY_PADS,&count);
        int before[8]={0};long strides[8];
        if(rank<1 || rank>8 || out->ndim!=rank || !pads || (count!=2*rank && !(rank==4 && count==4)) ||
            tensor_contiguous_strides(in,strides))return -1;
        for(int d=0;d<rank;d++) {
            int after;
            if(rank==4 && count==4) {before[d]=d==1?pads[0]:d==2?pads[1]:0;after=d==1?pads[2]:d==2?pads[3]:0;}
            else {before[d]=pads[d];after=pads[rank+d];}
            if(before[d]<0 || after<0 || (int64_t)in->shape[d]+before[d]+after!=out->shape[d])return -1;
        }
        float value=vx_node_param_f32(n,VX_NODE_PARAM_VALUE,0);
        for(long i=0;i<out->numel;i++) {
            long rem=i,index=0;int inside=1;
            for(int d=rank-1;d>=0;d--) {
                long coord=rem%out->shape[d]-before[d];rem/=out->shape[d];
                if(coord<0 || coord>=in->shape[d])inside=0;
                index+=coord*strides[d];
            }
            out->data[i]=inside?in->data[index]:value;
        }
    } else {
        if(in->ndim!=4 || out->ndim!=3)return -1;
        int batch=in->shape[0],h=in->shape[1],w=in->shape[2],c=in->shape[3],extent=py?h:w,channels=px||py?2*c:c;
        if(out->shape[0]!=batch || out->shape[1]!=channels || out->shape[2]!=extent)return -1;
        for(int i=0;i<batch;i++) {
            uintptr_t input=(uintptr_t)(in->data+(long)i*h*w*c),output=(uintptr_t)(out->data+(long)i*channels*extent);
            if(px)profile_x_f32(input,output,c,h,w);
            else if(py)profile_y_f32(input,output,c,h,w);
            else if(mean)mean_height_f32(input,output,c,h,w);
            else spatial_softargmax_y_f32(input,output,c,h,w);
        }
    }
    return 1;
}
