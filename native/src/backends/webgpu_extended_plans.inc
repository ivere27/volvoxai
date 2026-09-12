/* Operator plans shared by ordinary forward execution and training tapes.
 * Included by webgpu_backend.c; every operand remains device-resident. */
static T* vx_webgpu_output_port(const Node* node, VxPortKind key) {
    for (int i = 0; i < node->nout; i++)
        if (node->outs[i].port == key) return t_find(node->outs[i].name);
    return NULL;
}

static int vx_webgpu_grid(VxWebGpuNodePlan* plan, uint32_t shader,
                          uint32_t x, uint32_t y, uint32_t z) {
    if (!x || !y || !z || x > VX_WEBGPU_WORKGROUP_LIMIT ||
        y > VX_WEBGPU_WORKGROUP_LIMIT || z > VX_WEBGPU_WORKGROUP_LIMIT) return 0;
    plan->variant_count = 1;
    plan->variants[0].shader_id = shader;
    plan->variants[0].workgroup[0] = x;
    plan->variants[0].workgroup[1] = y;
    plan->variants[0].workgroup[2] = z;
    return 1;
}

static int vx_webgpu_plan_conv1d(const Node* n, T* in, T* out, VxWebGpuNodePlan* plan) {
    T* w = vx_webgpu_port(n, VX_PORT_WEIGHT), *b = vx_webgpu_port(n, VX_PORT_BIAS);
    uint32_t ni, no, nw, nb, p[12] = {0};
    size_t bi, bo, bw, bb = 0;
    if (!in || !out || !w || (in->ndim != 2 && in->ndim != 3) ||
        out->ndim != in->ndim || w->ndim != 3 ||
        !vx_webgpu_f32_extent(in, &ni, &bi) || !vx_webgpu_f32_extent(out, &no, &bo) ||
        !vx_webgpu_f32_extent(w, &nw, &bw) || (b && !vx_webgpu_f32_extent(b, &nb, &bb))) return 0;
    int rank = in->ndim, groups = vx_node_param_i32(n, VX_NODE_PARAM_GROUPS, 1);
    int stride, unused, pad;
    if (vx_webgpu_param_pair(n, VX_NODE_PARAM_STRIDE, VX_NODE_ARRAY_STRIDE, 1, &stride, &unused) ||
        vx_webgpu_param_pair(n, VX_NODE_PARAM_PADDING, VX_NODE_ARRAY_PADDING, 0, &pad, &unused)) return 0;
    p[0] = in->shape[rank-1]; p[1] = in->shape[rank-2];
    p[2] = out->shape[rank-1]; p[3] = w->shape[0];
    p[4] = stride; p[5] = pad; p[6] = vx_node_param_bool(n, VX_NODE_PARAM_RELU, 0);
    p[7] = rank == 3 ? in->shape[0] : 1; p[8] = groups;
    p[9] = w->shape[1]; p[10] = out->shape[rank-2];
    if (groups <= 0 || stride <= 0 || pad < 0 || p[0] % p[8] || p[2] % p[8] ||
        p[9] != p[0]/p[8] || w->shape[2] != (int)p[2] ||
        (rank == 3 && out->shape[0] != in->shape[0]) || (b && nb != p[2]) ||
        (uint64_t)p[1] + 2u*p[5] < p[3] ||
        ((uint64_t)p[1] + 2u*p[5] - p[3])/p[4] + 1u != p[10]) return 0;
    if (!vx_webgpu_bind(plan, in->data, bi) || !vx_webgpu_bind(plan, w->data, bw)) return 0;
    if (b) { if (!vx_webgpu_bind(plan, b->data, bb)) return 0; }
    else {
        uint32_t* zeros = vx_webgpu_arena((size_t)p[2] * sizeof(uint32_t));
        if (!zeros) return 0;
        memset(zeros, 0, (size_t)p[2] * sizeof(uint32_t));
        if (!vx_webgpu_bind(plan, zeros, (size_t)p[2] * sizeof(uint32_t))) return 0;
    }
    return vx_webgpu_bind(plan, out->data, bo) && vx_webgpu_params(plan, p, 12) &&
        vx_webgpu_grid(plan, VX_SHADER_INFERENCE_CONV1_D, (p[10]+63u)/64u, p[2], p[7]);
}

static int vx_webgpu_plan_conv_transpose(const Node* n, T* in, T* out, VxWebGpuNodePlan* plan) {
    T* w = vx_webgpu_port(n, VX_PORT_WEIGHT), *b = vx_webgpu_port(n, VX_PORT_BIAS);
    uint32_t ni, no, nw, nb, p[14], zero = 0;
    size_t bi, bo, bw, bb = 0;
    if (!in || !out || !w || in->ndim != 4 || out->ndim != 4 || w->ndim != 4 ||
        !vx_webgpu_f32_extent(in, &ni, &bi) || !vx_webgpu_f32_extent(out, &no, &bo) ||
        !vx_webgpu_f32_extent(w, &nw, &bw) || (b && !vx_webgpu_f32_extent(b, &nb, &bb))) return 0;
    int sy, sx, py, px;
    if (vx_webgpu_param_pair(n, VX_NODE_PARAM_STRIDE, VX_NODE_ARRAY_STRIDE, 1, &sy, &sx) ||
        vx_webgpu_param_pair(n, VX_NODE_PARAM_PADDING, VX_NODE_ARRAY_PADDING, 0, &py, &px) ||
        sy <= 0 || sx <= 0 || py < 0 || px < 0 ||
        in->shape[0] != out->shape[0] || w->shape[2] != in->shape[3] || w->shape[3] != out->shape[3] ||
        (b && nb != (uint32_t)out->shape[3]) ||
        (int64_t)(in->shape[1]-1)*sy + w->shape[0] - 2LL*py != out->shape[1] ||
        (int64_t)(in->shape[2]-1)*sx + w->shape[1] - 2LL*px != out->shape[2]) return 0;
    for (int i=0; i<4; i++) p[i] = in->shape[i];
    p[4]=out->shape[1]; p[5]=out->shape[2]; p[6]=out->shape[3];
    p[7]=w->shape[0]; p[8]=w->shape[1]; p[9]=sy; p[10]=sx; p[11]=py; p[12]=px; p[13]=b!=NULL;
    return vx_webgpu_bind(plan,in->data,bi) && vx_webgpu_bind(plan,w->data,bw) &&
        (b ? vx_webgpu_bind(plan,b->data,bb) : vx_webgpu_bind_scratch(plan,&zero,1)) &&
        vx_webgpu_bind(plan,out->data,bo) && vx_webgpu_params(plan,p,14) &&
        vx_webgpu_grid(plan,VX_SHADER_INFERENCE_CONV_TRANSPOSE2_D,(p[5]+7u)/8u,(p[4]+7u)/8u,p[0]*p[6]);
}

static int vx_webgpu_plan_vision(const Node* n, T* in, T* out, VxWebGpuNodePlan* plan) {
    uint32_t ni, no, p[4], shader, extent, channels;
    size_t bi, bo;
    if (!in || !out || in->ndim != 4 || out->ndim != 3 ||
        !vx_webgpu_f32_extent(in,&ni,&bi) || !vx_webgpu_f32_extent(out,&no,&bo)) return 0;
    p[0]=in->shape[1]; p[1]=in->shape[2]; p[2]=in->shape[3]; p[3]=in->shape[0];
    extent=p[1]; channels=p[2];
    if (n->operator_kind == VX_OP_PROFILE_X) { shader=VX_SHADER_INFERENCE_PROFILE_X; channels*=2; }
    else if (n->operator_kind == VX_OP_PROFILE_Y) { shader=VX_SHADER_INFERENCE_PROFILE_Y; channels*=2; extent=p[0]; }
    else if (n->operator_kind == VX_OP_MEAN_HEIGHT) shader=VX_SHADER_INFERENCE_MEAN_HEIGHT;
    else shader=VX_SHADER_INFERENCE_SPATIAL_SOFTARGMAX_Y;
    if ((uint32_t)out->shape[0]!=p[3] || (uint32_t)out->shape[1]!=channels ||
        (uint32_t)out->shape[2]!=extent) return 0;
    return vx_webgpu_bind(plan,in->data,bi) && vx_webgpu_bind(plan,out->data,bo) &&
        vx_webgpu_params(plan,p,4) && vx_webgpu_grid(plan,shader,(extent+63u)/64u,p[2],p[3]);
}

static int vx_webgpu_plan_interp(T* in, T* out, VxWebGpuNodePlan* plan) {
    uint32_t ni,no,p[4]; size_t bi,bo;
    if (!in || !out || (in->ndim!=2 && in->ndim!=3) || out->ndim!=in->ndim ||
        !vx_webgpu_f32_extent(in,&ni,&bi) || !vx_webgpu_f32_extent(out,&no,&bo)) return 0;
    int rank=in->ndim;
    for(int d=0;d<rank-1;d++) if(in->shape[d]!=out->shape[d]) return 0;
    p[0]=in->shape[rank-2]; p[1]=in->shape[rank-1]; p[2]=out->shape[rank-1];
    p[3]=rank==3?in->shape[0]:1;
    return vx_webgpu_bind(plan,in->data,bi) && vx_webgpu_bind(plan,out->data,bo) &&
        vx_webgpu_params(plan,p,4) && vx_webgpu_grid(plan,VX_SHADER_INFERENCE_INTERP1_D,(p[2]+63u)/64u,p[0],p[3]);
}

static int vx_webgpu_plan_pad(const Node* n, T* in, T* out, VxWebGpuNodePlan* plan) {
    uint32_t ni,no,p[16]={0}; size_t bi,bo;
    if (!in || !out || in->ndim<1 || in->ndim>4 || out->ndim!=in->ndim ||
        !vx_webgpu_f32_extent(in,&ni,&bi) || !vx_webgpu_f32_extent(out,&no,&bo)) return 0;
    int count=0;
    const int32_t* pads=vx_node_param_array(n,VX_NODE_PARAM_PADS,VX_NODE_ARRAY_PADS,&count);
    if (!pads) return 0;
    int rank=in->ndim, spatial=count==4 && rank==4;
    if (!spatial && count!=2*rank) return 0;
    for(int d=0;d<4;d++) {
        int axis=d-(4-rank), before=0, after=0;
        p[d]=p[4+d]=1;
        if(axis<0)continue;
        if(spatial) {
            if(axis==1) {before=pads[0];after=pads[2];}
            if(axis==2) {before=pads[1];after=pads[3];}
        } else {before=pads[axis];after=pads[rank+axis];}
        if(before<0 || after<0 || (int64_t)in->shape[axis]+before+after!=out->shape[axis])return 0;
        p[d]=in->shape[axis];p[4+d]=out->shape[axis];p[8+d]=before;
    }
    p[12]=no;p[13]=vx_webgpu_f32_bits(vx_node_param_f32(n,VX_NODE_PARAM_VALUE,0));
    return vx_webgpu_bind(plan,in->data,bi) && vx_webgpu_bind(plan,out->data,bo) &&
        vx_webgpu_params(plan,p,16) && vx_webgpu_grid(plan,VX_SHADER_INFERENCE_PAD,(no+63u)/64u,1,1);
}

static int vx_webgpu_plan_cross_attention(const Node* n, T* out, VxWebGpuNodePlan* plan) {
    T* q=vx_webgpu_port(n,VX_PORT_Q), *kv=vx_webgpu_port(n,VX_PORT_KV), *w=vx_webgpu_port(n,VX_PORT_WEIGHT);
    T* scale=vx_webgpu_port(n,VX_PORT_SCALE), *bias=vx_webgpu_port(n,VX_PORT_BIAS);
    uint32_t nq,nkv,nw,no,ns,nb,p[12]={0},zero=0;
    size_t bq,bkv,bw,bo,bs=0,bb=0;
    if(!q || !kv || !out || !w || (q->ndim!=2 && q->ndim!=3) || kv->ndim!=q->ndim || out->ndim!=q->ndim ||
       !vx_webgpu_f32_extent(q,&nq,&bq) || !vx_webgpu_f32_extent(kv,&nkv,&bkv) ||
       !vx_webgpu_f32_extent(out,&no,&bo) || !vx_webgpu_extent(w,&nw,&bw) ||
       (scale && !vx_webgpu_f32_extent(scale,&ns,&bs)) || (bias && !vx_webgpu_f32_extent(bias,&nb,&bb)))return 0;
    int rank=q->ndim, heads=n->parsed_params.heads;
    p[0]=q->shape[rank-2];p[1]=kv->shape[rank-2];p[2]=q->shape[rank-1];p[3]=heads;
    p[8]=rank==3?q->shape[0]:1;
    if(heads<=0 || p[2]%p[3] || no!=nq || out->shape[rank-2]!=(int)p[0] ||
       out->shape[rank-1]!=(int)p[2] || kv->shape[rank-1]!=(int)p[2] ||
       (rank==3 && (kv->shape[0]!=(int)p[8] || out->shape[0]!=(int)p[8])) ||
       (uint64_t)nw!=3ULL*p[2]*p[2] || (scale && ns!=3u*p[2]) || (bias && nb!=3u*p[2]))return 0;
    p[4]=p[2]/p[3];p[5]=vx_webgpu_f32_bits(1.0f/sqrtf((float)p[4]));p[6]=scale!=NULL;p[7]=bias!=NULL;
    if(!p[4] || p[4]>64 || (w->dtype!=T_F32 && w->dtype!=T_I8) ||
       (w->dtype==T_F32 ? p[2]>64 : p[2]%4u))return 0;
    return vx_webgpu_bind(plan,q->data,bq) && vx_webgpu_bind(plan,kv->data,bkv) && vx_webgpu_bind(plan,w->data,bw) &&
        (scale?vx_webgpu_bind(plan,scale->data,bs):vx_webgpu_bind_scratch(plan,&zero,1)) &&
        (bias?vx_webgpu_bind(plan,bias->data,bb):vx_webgpu_bind_scratch(plan,&zero,1)) &&
        vx_webgpu_bind(plan,out->data,bo) && vx_webgpu_params(plan,p,12) &&
        vx_webgpu_grid(plan,w->dtype==T_F32?VX_SHADER_INFERENCE_CROSS_ATTENTION_F32:VX_SHADER_INFERENCE_CROSS_ATTENTION,
                       (p[0]+63u)/64u,p[3],p[8]);
}

static int vx_webgpu_plan_qembedding(const Node* n,T* input,T* out,VxWebGpuNodePlan* plan) {
    T* w=vx_webgpu_port(n,VX_PORT_WEIGHT);
    if(!g_qembedding_meta || !input || !w || !out || input->dtype!=T_I32 || w->ndim!=2 ||
       n->resident_slot_domain || !out->quantization.valid)return 0;
    const QEmbeddingMetadata* m=&g_qembedding_meta[n-g_n];
    uint32_t ni,nw,no,p[8];size_t bi,bw,bo;
    if(!m->valid || !m->weight_scales || !m->weight_zero_points ||
       !vx_webgpu_extent(input,&ni,&bi) || !vx_webgpu_extent(w,&nw,&bw) || !vx_webgpu_extent(out,&no,&bo) ||
       (w->dtype!=T_I8 && w->dtype!=T_U8) || (out->dtype!=T_I8 && out->dtype!=T_U8) ||
       !m->vocab || !m->hidden || (uint64_t)ni*m->hidden!=no || (uint64_t)m->vocab*m->hidden!=nw)return 0;
    if(!g_bounded_gpu_value_domain_proven)for(uint32_t i=0;i<ni;i++) {
        int32_t id;memcpy(&id,(unsigned char*)input->data+4u*i,4);
        if(id<0 || (uint32_t)id>=m->vocab)return 0;
    }
    p[0]=ni;p[1]=m->vocab;p[2]=m->hidden;p[3]=w->dtype;p[4]=out->dtype;
    p[5]=out->quantization.zero_point;p[6]=vx_webgpu_f32_bits(out->quantization.scale);p[7]=0;
    return vx_webgpu_bind(plan,input->data,bi) && vx_webgpu_bind(plan,w->data,bw) &&
        vx_webgpu_bind(plan,m->weight_scales,(size_t)m->vocab*4) &&
        vx_webgpu_bind(plan,m->weight_zero_points,(size_t)m->vocab*4) &&
        vx_webgpu_bind(plan,out->data,bo) && vx_webgpu_params(plan,p,8) &&
        vx_webgpu_grid(plan,VX_SHADER_INFERENCE_Q_EMBEDDING_INT8,(no+255u)/256u,1,1);
}

static int vx_webgpu_plan_qsdpa_masked(const Node* n,T* out,VxWebGpuNodePlan* plan,T* mask,int row) {
    T* q=vx_webgpu_port(n,VX_PORT_Q), *k=vx_webgpu_port(n,VX_PORT_K), *v=vx_webgpu_port(n,VX_PORT_V);
    T* values[4]={q,k,v,out};
    uint32_t counts[4],p[20]={0},mask_count,zero=0;size_t sizes[4],mask_bytes=0;
    for(int i=0;i<4;i++)if(!values[i] || !vx_webgpu_extent(values[i],&counts[i],&sizes[i]) ||
        (values[i]->dtype!=T_I8 && values[i]->dtype!=T_U8) || !values[i]->quantization.valid)return 0;
    int rank=q->ndim,heads=vx_node_param_i32(n,VX_NODE_PARAM_HEADS,1);
    if((rank!=2 && rank!=3) || k->ndim!=rank || v->ndim!=rank || out->ndim!=rank || heads<=0)return 0;
    p[0]=q->shape[rank-2];p[1]=k->shape[rank-2];p[2]=q->shape[rank-1];p[3]=heads;p[4]=rank==3?q->shape[0]:1;
    if(p[2]%p[3] || p[2]%4 || !(p[2]/p[3]) || p[2]/p[3]>64 || (p[2]/p[3])%4)return 0;
    for(int i=0;i<4;i++) {
        T* t=values[i];
        if(t->shape[rank-1]!=(int)p[2] || t->shape[rank-2]!=(int)p[i==1||i==2?1:0] ||
           (rank==3 && t->shape[0]!=(int)p[4]))return 0;
        p[7]|=(uint32_t)t->dtype<<(8*i);p[8+i]=t->quantization.zero_point;
        p[12+i]=vx_webgpu_f32_bits(t->quantization.scale);
    }
    if(mask) {
        if(mask->dtype!=T_I32 || !vx_webgpu_extent(mask,&mask_count,&mask_bytes))return 0;
        if(mask->ndim==1 && mask->shape[0]==(int)p[1])p[5]=1;
        else if(mask->ndim==2 && mask->shape[1]==(int)p[1]) {
            if(mask->shape[0]==(int)p[4])p[5]=2;
            else if(mask->shape[0]==(int)p[0])p[5]=3;
        } else if(mask->ndim==3 && mask->shape[0]==(int)p[4] && mask->shape[1]==(int)p[0] && mask->shape[2]==(int)p[1])p[5]=4;
        if(!p[5])return 0;
    }
    p[6]=!row && vx_node_param_bool(n,VX_NODE_PARAM_CAUSAL,0);
    float scale=n->parsed_params.has_scale?n->parsed_params.scale:1.0f/sqrtf((float)(p[2]/p[3]));
    if(!isfinite(scale))return 0;p[16]=vx_webgpu_f32_bits(scale);
    for(int i=0;i<3;i++)if(!vx_webgpu_bind(plan,values[i]->data,sizes[i]))return 0;
    return (mask?vx_webgpu_bind(plan,mask->data,mask_bytes):vx_webgpu_bind_scratch(plan,&zero,1)) &&
        vx_webgpu_bind(plan,out->data,sizes[3]) && vx_webgpu_params(plan,p,20) &&
        vx_webgpu_grid(plan,VX_SHADER_INFERENCE_Q_SDPA_INT8,p[0],p[3],p[4]);
}

static int vx_webgpu_plan_moe_router(const Node* node, T* input,
                                     VxWebGpuNodePlan* plan) {
    T* weight = vx_webgpu_port(node, VX_PORT_WEIGHT);
    T* bias = vx_webgpu_port(node, VX_PORT_BIAS);
    T* indices = vx_webgpu_output_port(node, VX_PORT_INDICES);
    T* routes = vx_webgpu_output_port(node, VX_PORT_WEIGHTS);
    if (!weight) weight = vx_webgpu_port(node, VX_PORT_ROUTER_WEIGHT);
    if (!indices) indices = vx_webgpu_output_port(node, VX_PORT_ROUTE_INDICES);
    if (!routes) routes = vx_webgpu_output_port(node, VX_PORT_ROUTE_WEIGHTS);
    uint32_t inputs, weights, outputs, route_count, bias_count, params[7], zero = 0;
    size_t input_bytes, weight_bytes, index_bytes, route_bytes, bias_bytes = 0;
    if (!input || !weight || !indices || !routes || input->ndim < 1 || weight->ndim != 2 ||
        !vx_webgpu_f32_extent(input, &inputs, &input_bytes) ||
        !vx_webgpu_f32_extent(weight, &weights, &weight_bytes) ||
        !vx_webgpu_f32_extent(indices, &outputs, &index_bytes) ||
        !vx_webgpu_f32_extent(routes, &route_count, &route_bytes) || outputs != route_count)
        return 0;
    int width = input->shape[input->ndim - 1];
    int experts = vx_node_param_i32(node, VX_NODE_PARAM_NUM_EXPERTS, weight->shape[1]);
    int top_k = vx_node_param_i32(node, VX_NODE_PARAM_TOP_K, 2);
    float temperature = vx_node_param_f32(node, VX_NODE_PARAM_TEMPERATURE, 1.0f);
    if (width <= 0 || experts <= 0 || top_k <= 0 || top_k > 8 || top_k > experts ||
        weight->shape[0] != width || weight->shape[1] != experts ||
        inputs % (uint32_t)width || (uint64_t)outputs != (uint64_t)(inputs / width) * top_k ||
        !isfinite(temperature) || temperature <= 0 ||
        (bias && (!vx_webgpu_f32_extent(bias, &bias_count, &bias_bytes) || bias_count != (uint32_t)experts)))
        return 0;
    params[0] = inputs / width; params[1] = (uint32_t)width;
    params[2] = (uint32_t)experts; params[3] = (uint32_t)top_k;
    params[4] = vx_node_param_bool(node, VX_NODE_PARAM_NORMALIZE, 1) ? 1u : 0u;
    params[5] = bias != NULL; params[6] = vx_webgpu_f32_bits(temperature);
    if (!vx_webgpu_bind(plan, input->data, input_bytes) ||
        !vx_webgpu_bind(plan, weight->data, weight_bytes) ||
        !(bias ? vx_webgpu_bind(plan, bias->data, bias_bytes) : vx_webgpu_bind_scratch(plan, &zero, 1u)) ||
        !vx_webgpu_bind(plan, indices->data, index_bytes) ||
        !vx_webgpu_bind(plan, routes->data, route_bytes) ||
        !vx_webgpu_params(plan, params, 7u) ||
        !vx_webgpu_linear_1d(params[0], 64u, plan->variants[0].workgroup)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_MOE_ROUTER;
    plan->variant_count = 1u;
    return 1;
}

static int vx_webgpu_plan_moe_linear(const Node* node, T* input, T* output,
                                     VxWebGpuNodePlan* plan) {
    T* weight = vx_webgpu_port(node, VX_PORT_EXPERT_WEIGHT);
    T* bias = vx_webgpu_port(node, VX_PORT_EXPERT_BIAS);
    T* indices = vx_webgpu_port(node, VX_PORT_ROUTE_INDICES);
    T* gates = vx_webgpu_port(node, VX_PORT_ROUTE_WEIGHTS);
    if (!weight) weight = vx_webgpu_port(node, VX_PORT_WEIGHT);
    if (!bias) bias = vx_webgpu_port(node, VX_PORT_BIAS);
    if (!indices) indices = vx_webgpu_port(node, VX_PORT_INDICES);
    if (!gates) gates = vx_webgpu_port(node, VX_PORT_WEIGHTS);
    uint32_t input_count, weight_count, output_count, index_count, gate_count, bias_count;
    size_t input_bytes, weight_bytes, output_bytes, index_bytes, gate_bytes, bias_bytes = 0;
    uint32_t params[7], zero = 0;
    if (!input || !output || !weight || !indices || !gates || input->ndim < 1 ||
        output->ndim < 1 || weight->ndim != 3 ||
        !vx_webgpu_f32_extent(input, &input_count, &input_bytes) ||
        !vx_webgpu_f32_extent(output, &output_count, &output_bytes) ||
        !vx_webgpu_f32_extent(weight, &weight_count, &weight_bytes) ||
        !vx_webgpu_f32_extent(indices, &index_count, &index_bytes) ||
        !vx_webgpu_f32_extent(gates, &gate_count, &gate_bytes) || index_count != gate_count) return 0;
    uint32_t experts = (uint32_t)weight->shape[0], din = (uint32_t)weight->shape[1];
    uint32_t dout = (uint32_t)weight->shape[2];
    if (!experts || !din || !dout || (uint32_t)input->shape[input->ndim - 1] != din ||
        (uint32_t)output->shape[output->ndim - 1] != dout || input_count % din) return 0;
    uint32_t rows = input_count / din;
    if (!rows || index_count % rows || !index_count || (uint64_t)rows * dout != output_count ||
        (bias && (!vx_webgpu_f32_extent(bias, &bias_count, &bias_bytes) ||
                  (uint64_t)bias_count != (uint64_t)experts * dout))) return 0;
    params[0] = rows; params[1] = din; params[2] = dout; params[3] = experts;
    params[4] = index_count / rows; params[5] = bias != NULL;
    params[6] = node->resident_slot_domain;
    if (!vx_webgpu_bind(plan, input->data, input_bytes) ||
        !vx_webgpu_bind(plan, weight->data, weight_bytes) ||
        !(bias ? vx_webgpu_bind(plan, bias->data, bias_bytes) : vx_webgpu_bind_scratch(plan, &zero, 1u)) ||
        !vx_webgpu_bind(plan, indices->data, index_bytes) ||
        !vx_webgpu_bind(plan, gates->data, gate_bytes) ||
        !vx_webgpu_bind(plan, output->data, output_bytes) ||
        !(node->resident_slot_domain
          ? node->resident_slot_rows && vx_webgpu_bind(plan, node->resident_slot_rows,
              (size_t)node->resident_slot_domain * sizeof(uint32_t))
          : vx_webgpu_bind_scratch(plan, &zero, 1u)) ||
        !vx_webgpu_params(plan, params, 7u)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_MOE_LINEAR;
    plan->variants[0].workgroup[0] = (dout + 63u) / 64u;
    plan->variants[0].workgroup[1] = rows;
    plan->variants[0].workgroup[2] = 1u;
    if (plan->variants[0].workgroup[0] > VX_WEBGPU_WORKGROUP_LIMIT || rows > VX_WEBGPU_WORKGROUP_LIMIT)
        return 0;
    plan->variant_count = 1u;
    return 1;
}
