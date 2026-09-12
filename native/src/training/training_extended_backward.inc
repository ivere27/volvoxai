/* Common descriptors for the portable CPU and GPU backward paths. Numerical
 * CPU functions and WGSL share these shapes and parameters; no host-language
 * graph interpretation is involved. Included by training_gpu_plan.inc. */
#include "../kernels/training_kernels.h"

typedef struct {
    void* hosts[GPU_TRAINING_MAX_BINDINGS];
    size_t bytes[GPU_TRAINING_MAX_BINDINGS];
    unsigned char access[GPU_TRAINING_MAX_BINDINGS];
    uint32_t params[36];
} TrainingExtendedBindings;

static int training_extended_gradient(TrainingExtendedBindings* b, int slot,
                                      float** grads, T* tensor) {
    float* gradient = grad_for(grads, tensor);
    if (!gradient) return -1;
    gpu_binding_gradient(b->hosts,b->bytes,slot,gradient,tensor->numel);
    b->access[slot]=GPU_TRAINING_READ_WRITE;
    return 0;
}

static int training_extended_add(GpuTrainingPlan* plan, TrainingExtendedBindings* b,
    const char* shader, const char* entry, int slot, int words,
    uint32_t x, uint32_t y, uint32_t z) {
    uint32_t* params=gpu_training_plan_alloc(plan,(size_t)words*4);
    if(!params)return -1;
    memcpy(params,b->params,(size_t)words*4);
    gpu_binding_params(b->hosts,b->bytes,slot,params,(size_t)words*4);
    return gpu_training_plan_add(plan,shader,entry,b->hosts,b->bytes,b->access,slot+1,x,y,z);
}

static int training_extended_tensor(const T* t) {
    if(!t || !t->data || t->ndim<0 || t->ndim>8 || t->numel<=0 || (uint64_t)t->numel>UINT32_MAX)return 0;
    uint64_t size=1;
    for(int i=0;i<t->ndim;i++) {
        if(t->shape[i]<=0 || size>UINT32_MAX/(uint32_t)t->shape[i])return 0;
        size*=t->shape[i];
    }
    return size==(uint64_t)t->numel;
}

static int training_extended_backward(GpuTrainingPlan* plan,float** grads,Node* n,T* out,float* go) {
    T* in=node_input_default(n);
    VxOperatorKind op=n->operator_kind;
    if((op == VX_OP_WHERE) || (op == VX_OP_MASK)) {
        in=nin(n,VX_PORT_X);if(!in)in=nin(n,VX_PORT_A);
    }
    TrainingExtendedBindings b={0};
    gpu_bindings_init(b.hosts,b.bytes,b.access,GPU_TRAINING_MAX_BINDINGS);
    uint32_t* p=b.params;
    if(!training_extended_tensor(out) || out->dtype!=T_F32 || !go)return -1;
    if (op == VX_OP_CROSS_ATTENTION) {
        T* q=nin(n,VX_PORT_Q),*kv=nin(n,VX_PORT_KV),*w=nin(n,VX_PORT_WEIGHT),*s=nin(n,VX_PORT_SCALE),*bias=nin(n,VX_PORT_BIAS);
        T* tensors[5]={q,kv,w,s,bias};
        for(int i=0;i<5;i++)if((i<3 || tensors[i]) &&
            (!training_extended_tensor(tensors[i]) || tensors[i]->dtype!=T_F32))return -1;
        int rank=q->ndim,heads=vx_node_param_i32(n,VX_NODE_PARAM_HEADS,8);
        if((rank!=2 && rank!=3) || kv->ndim!=rank || out->ndim!=rank || heads<=0)return -1;
        p[0]=q->shape[rank-2];p[1]=kv->shape[rank-2];p[2]=q->shape[rank-1];p[3]=heads;
        if(p[2]%p[3] || p[2]>64 || (uint64_t)w->numel!=3ULL*p[2]*p[2] ||
            (s && s->numel!=3L*p[2]) || (bias && bias->numel!=3L*p[2]) ||
            kv->shape[rank-1]!=(int)p[2] || !tensor_same_shape(q,out) ||
            (rank==3 && kv->shape[0]!=q->shape[0]))return -1;
        p[4]=p[2]/p[3];p[5]=rank==3?q->shape[0]:1;p[6]=q->numel;p[7]=kv->numel;
        p[8]=w->numel;p[9]=3*p[2];p[10]=s!=NULL;p[11]=bias!=NULL;p[12]=f32_bits(1.0f/sqrtf((float)p[4]));
        for(int i=0;i<5;i++) {
            gpu_binding_tensor(b.hosts,b.bytes,i,tensors[i]);
            if(tensors[i] && training_extended_gradient(&b,6+i,grads,tensors[i]))return -1;
        }
        gpu_binding_gradient(b.hosts,b.bytes,5,go,out->numel);
        if(!plan)return volvoxai_training_cross_attention_backward_f32(q->data,kv->data,w->data,
            s?s->data:NULL,bias?bias->data:NULL,go,b.hosts[6],b.hosts[7],b.hosts[8],
            s?b.hosts[9]:NULL,bias?b.hosts[10]:NULL,p[5],p[0],p[1],p[2],p[3])?0:-1;
        const char* entries[5]={"q_main","kv_main","weight_main","scale_main","bias_main"};
        for(int i=0;i<5;i++)if(tensors[i] && training_extended_add(plan,&b,"crossAttentionBackward",entries[i],11,16,
            gpu_groups_64(tensors[i]->numel),1,1))return -1;
        return 0;
    }
    if(!training_extended_tensor(in))return -1;
    if (op == VX_OP_CAST) {
        if(in->numel!=out->numel)return -1;
        if(in->dtype!=T_F32)return 0;
        if(!plan) {float* gi=grad_for(grads,in);if(!gi)return -1;grad_add_full(gi,go,in->numel,1);return 0;}
        return gpu_plan_copy(plan,grads,n,out,go);
    }
    if (op == VX_OP_DEQUANTIZE_LINEAR) {
        T* scale=nin(n,VX_PORT_SCALE),*zero=nin(n,VX_PORT_ZERO_POINT);
        if(!scale || scale->dtype!=T_F32 || scale->numel!=1 || in->numel!=out->numel ||
            (zero && (!training_extended_tensor(zero) || zero->numel!=1)))return -1;
        if(in->dtype!=T_F32 && in->dtype!=T_I32 && in->dtype!=T_I8 && in->dtype!=T_U8)return -1;
        if(zero && zero->dtype!=T_F32 && zero->dtype!=T_I32 && zero->dtype!=T_I8 && zero->dtype!=T_U8)return -1;
        p[0]=in->numel;p[1]=in->dtype;p[2]=zero?zero->dtype:T_F32;p[3]=zero!=NULL;
        gpu_binding_tensor(b.hosts,b.bytes,0,in);gpu_binding_tensor(b.hosts,b.bytes,1,zero);
        gpu_binding_tensor(b.hosts,b.bytes,2,scale);gpu_binding_gradient(b.hosts,b.bytes,3,go,out->numel);
        if((in->dtype==T_F32 && training_extended_gradient(&b,4,grads,in)) || training_extended_gradient(&b,5,grads,scale))return -1;
        if(!plan)return volvoxai_training_dequantize_linear_backward_f32(in->data,in->dtype,scale->data,
            zero?zero->data:NULL,p[2],go,in->dtype==T_F32?b.hosts[4]:NULL,b.hosts[5],p[0])?0:-1;
        if(in->dtype==T_F32 && training_extended_add(plan,&b,"dequantizeLinearBackward","input_main",6,4,gpu_groups_64(in->numel),1,1))return -1;
        return training_extended_add(plan,&b,"dequantizeLinearBackward","scale_main",6,4,1,1,1);
    }
    if(in->dtype!=T_F32)return -1;
    if((op == VX_OP_SUB) || (op == VX_OP_DIV)) {
        T* right=nin(n,VX_PORT_B);
        if(!training_extended_tensor(right) || right->dtype!=T_F32)return -1;
        uint32_t kind=(op == VX_OP_SUB)?2:3;
        if(plan)return gpu_plan_binary(plan,grads,n,out,go,kind);
        float *gi=grad_for(grads,in),*gr=grad_for(grads,right);
        if(!gi || !gr)return -1;
        return volvoxai_training_binary_broadcast_backward_f32(in->data,right->data,go,gi,gr,
            (uint32_t*)in->shape,(uint32_t*)right->shape,(uint32_t*)out->shape,in->ndim,right->ndim,out->ndim,out->numel,kind)?0:-1;
    }
    if((op == VX_OP_WHERE) || (op == VX_OP_MASK)) {
        T* condition=nin(n,VX_PORT_CONDITION);if(!condition)condition=nin(n,VX_PORT_COND);if(!condition)condition=nin(n,VX_PORT_MASK);
        T* x=nin(n,VX_PORT_X);if(!x)x=nin(n,VX_PORT_A);T* y=nin(n,VX_PORT_Y);if(!y)y=nin(n,VX_PORT_B);
        if(!x || !y || !condition || x->dtype!=T_F32 || y->dtype!=T_F32 ||
            (condition->dtype!=T_F32 && condition->dtype!=T_I32) || !tensor_same_shape(x,out) ||
            !tensor_same_shape(y,out) || !tensor_same_shape(condition,out))return -1;
        p[0]=out->numel;p[1]=condition->dtype==T_F32;
        gpu_binding_tensor(b.hosts,b.bytes,0,condition);gpu_binding_gradient(b.hosts,b.bytes,1,go,out->numel);
        if(training_extended_gradient(&b,2,grads,x) || training_extended_gradient(&b,3,grads,y))return -1;
        if(!plan)return volvoxai_training_where_backward_f32(condition->data,go,b.hosts[2],b.hosts[3],condition->dtype,p[0])?0:-1;
        return training_extended_add(plan,&b,"whereBackward","main",4,4,gpu_groups_64(out->numel),1,1);
    }
    if((op == VX_OP_CONV_1D) || (op == VX_OP_CONV_TRANSPOSE_2D)) {
        T* w=nin(n,VX_PORT_WEIGHT),*bias=nin(n,VX_PORT_BIAS);
        int conv1d=(op == VX_OP_CONV_1D),rank=conv1d?3:4,stride[2],pad[2];
        if(!training_extended_tensor(w) || w->dtype!=T_F32 ||
            (bias && (!training_extended_tensor(bias) || bias->dtype!=T_F32)) ||
            in->ndim!=rank || out->ndim!=rank || w->ndim!=rank ||
            in->shape[0]!=out->shape[0] ||
            training_node_pair(n,VX_NODE_PARAM_STRIDE,VX_NODE_ARRAY_STRIDE,1,1,0,stride) ||
            training_node_pair(n,VX_NODE_PARAM_PADDING,VX_NODE_ARRAY_PADDING,0,0,0,pad))return -1;
        if(conv1d) {
            int groups=vx_node_param_i32(n,VX_NODE_PARAM_GROUPS,1);
            p[0]=in->shape[2];p[1]=in->shape[1];p[2]=out->shape[2];p[3]=w->shape[0];p[4]=stride[0];p[5]=pad[0];
            p[6]=vx_node_param_bool(n,VX_NODE_PARAM_RELU,0);p[7]=in->shape[0];p[8]=groups;p[9]=w->shape[1];p[10]=out->shape[1];
            if(groups<=0 || p[0]%p[8] || p[2]%p[8] || p[9]!=p[0]/p[8] || w->shape[2]!=(int)p[2] ||
                (bias && bias->numel!=p[2]) || (uint64_t)p[1]+2u*p[5]<p[3] ||
                ((uint64_t)p[1]+2u*p[5]-p[3])/p[4]+1!=p[10])return -1;
        } else {
            for(int i=0;i<4;i++)p[i]=in->shape[i];
            p[4]=out->shape[1];p[5]=out->shape[2];p[6]=out->shape[3];p[7]=stride[1];
            p[8]=w->shape[0];p[9]=w->shape[1];p[10]=stride[0];p[11]=pad[0];p[12]=pad[1];
            if(w->shape[2]!=(int)p[3] || w->shape[3]!=(int)p[6] || (bias && bias->numel!=p[6]) ||
                (int64_t)(p[1]-1)*p[10]+p[8]-2LL*p[11]!=p[4] ||
                (int64_t)(p[2]-1)*p[7]+p[9]-2LL*p[12]!=p[5])return -1;
        }
        int grad_slot=conv1d?4:3, dy_slot=conv1d?3:2, parameter_slot=conv1d?7:6;
        gpu_binding_tensor(b.hosts,b.bytes,0,in);gpu_binding_tensor(b.hosts,b.bytes,1,w);
        if(conv1d)gpu_binding_tensor(b.hosts,b.bytes,2,out);
        gpu_binding_gradient(b.hosts,b.bytes,dy_slot,go,out->numel);
        if(training_extended_gradient(&b,grad_slot,grads,in) || training_extended_gradient(&b,grad_slot+1,grads,w) ||
            (bias && training_extended_gradient(&b,grad_slot+2,grads,bias)))return -1;
        if(!plan) {
            if(conv1d)return volvoxai_training_conv1d_backward_f32(in->data,w->data,out->data,go,b.hosts[4],b.hosts[5],
                bias?b.hosts[6]:NULL,p[7],p[0],p[1],p[2],p[9],p[3],p[10],p[4],p[5],p[8],p[6])?0:-1;
            return volvoxai_training_conv_transpose2d_backward_f32(in->data,w->data,go,b.hosts[3],b.hosts[4],bias?b.hosts[5]:NULL,
                p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[8],p[9],p[10],p[7],p[11],p[12])?0:-1;
        }
        const char* shader=conv1d?"conv1DBackward":"convTranspose2DBackward";
        int words=conv1d?12:16;
        if(training_extended_add(plan,&b,shader,"input_main",parameter_slot,words,gpu_groups_64(in->numel),1,1) ||
            training_extended_add(plan,&b,shader,"weight_main",parameter_slot,words,gpu_groups_64(w->numel),1,1))return -1;
        return bias?training_extended_add(plan,&b,shader,"bias_main",parameter_slot,words,gpu_groups_64(bias->numel),1,1):0;
    }
    if((op == VX_OP_GATHER) || (op == VX_OP_GATHER_ELEMENTS)) {
        T* ids=nin(n,VX_PORT_INDICES);int rank=in->ndim,axis=vx_node_param_i32(n,VX_NODE_PARAM_AXIS,0);
        if(axis<0)axis+=rank;
        if(!training_extended_tensor(ids) || ids->dtype!=T_I32 || rank<1 || axis<0 || axis>=rank)return -1;
        int elements=(op == VX_OP_GATHER_ELEMENTS);
        if(elements) {
            if(ids->ndim!=rank || !tensor_same_shape(ids,out))return -1;
            p[0]=rank;p[1]=axis;p[2]=in->numel;p[3]=out->numel;
            for(int d=0;d<rank;d++) {
                if(d!=axis && ids->shape[d]>in->shape[d])return -1;
                p[4+d]=in->shape[d];p[12+d]=ids->shape[d];
            }
        } else {
            if(out->ndim!=rank-1+ids->ndim)return -1;
            for(int d=0;d<out->ndim;d++) {
                int expected=d<axis?in->shape[d]:d<axis+ids->ndim?ids->shape[d-axis]:in->shape[d-ids->ndim+1];
                if(out->shape[d]!=expected)return -1;
            }
            p[0]=1;p[1]=in->shape[axis];p[2]=1;p[3]=ids->numel;
            for(int d=0;d<axis;d++)p[0]*=in->shape[d];
            for(int d=axis+1;d<rank;d++)p[2]*=in->shape[d];
        }
        gpu_binding_tensor(b.hosts,b.bytes,0,ids);gpu_binding_gradient(b.hosts,b.bytes,1,go,out->numel);
        if(training_extended_gradient(&b,2,grads,in))return -1;
        if(!plan)return (elements?volvoxai_training_gather_elements_backward_f32((int32_t*)ids->data,go,b.hosts[2],p+4,p+12,p[0],p[1],out->numel):
            volvoxai_training_gather_backward_f32((int32_t*)ids->data,go,b.hosts[2],p[0],p[1],p[2],p[3],out->numel))?0:-1;
        return training_extended_add(plan,&b,elements?"gatherElementsBackward":"gatherBackward","main",3,elements?20:4,
            gpu_groups_64(in->numel),1,1);
    }
    if((op == VX_OP_PROFILE_X) || (op == VX_OP_PROFILE_Y) || (op == VX_OP_MEAN_HEIGHT) || (op == VX_OP_SPATIAL_SOFTARGMAX_Y)) {
        if(in->ndim!=4 || out->ndim!=3)return -1;
        p[0]=in->shape[1];p[1]=in->shape[2];p[2]=in->shape[3];p[3]=in->shape[0];
        int px=(op == VX_OP_PROFILE_X),py=(op == VX_OP_PROFILE_Y),mean=(op == VX_OP_MEAN_HEIGHT);
        uint32_t extent=py?p[0]:p[1],channels=(px||py)?2*p[2]:p[2];
        if(out->shape[0]!=(int)p[3] || out->shape[1]!=(int)channels || out->shape[2]!=(int)extent)return -1;
        gpu_binding_tensor(b.hosts,b.bytes,0,in);gpu_binding_tensor(b.hosts,b.bytes,1,out);
        gpu_binding_gradient(b.hosts,b.bytes,2,go,out->numel);
        if(training_extended_gradient(&b,3,grads,in))return -1;
        if(!plan) {
            uint32_t ok=px?volvoxai_training_profile_x_backward_f32(in->data,go,b.hosts[3],p[3],p[0],p[1],p[2]):
                py?volvoxai_training_profile_y_backward_f32(in->data,go,b.hosts[3],p[3],p[0],p[1],p[2]):
                mean?volvoxai_training_mean_height_backward_f32(go,b.hosts[3],p[3],p[0],p[1],p[2]):
                volvoxai_training_spatial_softargmax_y_backward_f32(in->data,out->data,go,b.hosts[3],p[3],p[0],p[1],p[2]);
            return ok?0:-1;
        }
        return training_extended_add(plan,&b,"visionBackward",px?"profile_x_main":py?"profile_y_main":mean?"mean_height_main":"spatial_softargmax_y_main",
            4,4,(extent+63)/64,p[2],p[3]);
    }
    gpu_binding_gradient(b.hosts,b.bytes,0,go,out->numel);
    if(training_extended_gradient(&b,1,grads,in))return -1;
    if((op == VX_OP_EXPAND) || (op == VX_OP_BROADCAST)) {
        uint32_t strides[8];
        if(in->ndim<1 || gpu_broadcast_strides(in,out,strides))return -1;
        p[0]=in->ndim;p[1]=out->ndim;p[2]=in->numel;p[3]=out->numel;
        for(int d=0;d<in->ndim;d++)p[4+d]=in->shape[d];
        for(int d=0;d<out->ndim;d++)p[12+d]=out->shape[d];
        if(!plan)return volvoxai_training_expand_backward_f32(go,b.hosts[1],p+4,p+12,p[0],p[1],p[3])?0:-1;
        return training_extended_add(plan,&b,"expandBackward","main",2,20,gpu_groups_64(in->numel),1,1);
    }
    if (op == VX_OP_SLICE) {
        int rank=in->ndim;unsigned selected=0;
        if(rank<1 || out->ndim!=rank || !n->parsed_params.has_slice || n->parsed_params.slice_rank<1 || n->parsed_params.slice_rank>rank)return -1;
        p[0]=rank;p[1]=out->numel;
        uint32_t stride=1;
        for(int d=rank-1;d>=0;d--) {p[4+d]=out->shape[d];p[20+d]=1;p[28+d]=stride;stride*=in->shape[d];}
        for(int i=0;i<n->parsed_params.slice_rank;i++) {
            int axis=n->parsed_params.axes[i];if(axis<0)axis+=rank;
            int step=n->parsed_params.steps[i];
            if(axis<0 || axis>=rank || (selected&(1u<<axis)) || step<=0)return -1;
            int64_t start=n->parsed_params.starts[i];if(start<0)start+=in->shape[axis];
            if(start<0)start=0;if(start>in->shape[axis])start=in->shape[axis];
            if(start+(int64_t)(out->shape[axis]-1)*step>=in->shape[axis])return -1;
            p[12+axis]=start;p[20+axis]=step;selected|=1u<<axis;
        }
        for(int d=0;d<rank;d++)if(!(selected&(1u<<d)) && in->shape[d]!=out->shape[d])return -1;
        if(!plan)return volvoxai_training_slice_backward_f32(go,b.hosts[1],(uint32_t*)in->shape,p+4,p+12,p+20,p[0],p[1])?0:-1;
        return training_extended_add(plan,&b,"sliceBackward","main",2,36,gpu_groups_64(out->numel),1,1);
    }
    if (op == VX_OP_PAD) {
        int count=0;const int32_t* pads=vx_node_param_array(n,VX_NODE_PARAM_PADS,VX_NODE_ARRAY_PADS,&count);
        if(in->ndim!=4 || out->ndim!=4 || !pads || (count!=4 && count!=8))return -1;
        if(count==8 && (pads[0] || pads[3] || pads[4] || pads[7]))return -1;
        int pt=pads[count==8?1:0],pl=pads[count==8?2:1],pb=pads[count==8?5:2],pr=pads[count==8?6:3];
        if(pt<0 || pl<0 || pb<0 || pr<0 || out->shape[0]!=in->shape[0] || out->shape[3]!=in->shape[3] ||
            (int64_t)in->shape[1]+pt+pb!=out->shape[1] || (int64_t)in->shape[2]+pl+pr!=out->shape[2])return -1;
        for(int i=0;i<4;i++)p[i]=in->shape[i];
        p[4]=out->shape[1];p[5]=out->shape[2];p[6]=out->shape[3];p[8]=pt;p[9]=pl;p[10]=pb;p[11]=pr;
        if(!plan)return volvoxai_training_pad2d_backward_f32(go,b.hosts[1],p[0],p[1],p[2],p[3],p[8],p[10],p[9],p[11])?0:-1;
        return training_extended_add(plan,&b,"padBackward","main",2,12,gpu_groups_64(in->numel),1,1);
    }
    if (op == VX_OP_INTERPOLATE_1D) {
        if((in->ndim!=2 && in->ndim!=3) || out->ndim!=in->ndim)return -1;
        for(int d=0;d<in->ndim-1;d++)if(in->shape[d]!=out->shape[d])return -1;
        p[0]=in->ndim==3?in->shape[0]:1;p[1]=in->shape[in->ndim-2];p[2]=in->shape[in->ndim-1];p[3]=out->shape[out->ndim-1];
        if(!plan)return volvoxai_training_interp1d_backward_f32(go,b.hosts[1],p[0],p[1],p[2],p[3])?0:-1;
        return training_extended_add(plan,&b,"interp1DBackward","main",2,4,gpu_groups_64(in->numel),1,1);
    }
    return -1;
}
