/* Private whole-prefix modern decoder dispatch. Canonical shape admission
 * precedes this descriptor construction; every physical extent is checked
 * again before the kernel receives a pointer. */
#include "../kernels/llm_ops.h"

static int vx_llm_operator(VxOperatorKind op) {
    return (op >= VX_OP_GATED_DELTA_NET && op <= VX_OP_ROTARY_EMBEDDING) || op==VX_OP_SPARSE_INDEXER || op==VX_OP_QSA_INDEXER || op==VX_OP_SIGNED_SQRT || op==VX_OP_EMBEDDING_SHARD || op==VX_OP_PACKED_LINEAR;
}

static int vx_llm_tensor_valid(const T* t, int dtype) {
    uint64_t count = 1;
    if (!t || t->dtype != dtype || !t->data || t->ndim < 0 || t->ndim > 8) return 0;
    for (int i = 0; i < t->ndim; ++i) {
        if (t->shape[i] <= 0 || count > SIZE_MAX / sizeof(float) / (uint32_t)t->shape[i]) return 0;
        count *= (uint32_t)t->shape[i];
    }
    return t->numel > 0 && count == (uint64_t)t->numel;
}

int vx_engine_llm_node_args(Node* n, VxLLMOpArgs* a) {
    static const VxPortKind delta_ports[] = {VX_PORT_Q,VX_PORT_K,VX_PORT_V,VX_PORT_A,VX_PORT_B,VX_PORT_A_LOG,VX_PORT_DT_BIAS,VX_PORT_INITIAL_STATE};
    static const VxPortKind attention_ports[] = {VX_PORT_Q,VX_PORT_K,VX_PORT_V,VX_PORT_COMPRESSED_KV,VX_PORT_SINKS,VX_PORT_INDEX_Q,VX_PORT_INDEX_K,VX_PORT_INDEX_WEIGHTS};
    static const VxPortKind compress_ports[] = {VX_PORT_INPUT,VX_PORT_A,VX_PORT_BIAS,VX_PORT_WEIGHT,VX_PORT_COS,VX_PORT_SIN};
    static const VxPortKind hyper_ports[] = {VX_PORT_INPUT,VX_PORT_WEIGHT,VX_PORT_BIAS,VX_PORT_SCALE};
    static const VxPortKind residual_ports[] = {VX_PORT_INPUT,VX_PORT_UPDATE,VX_PORT_POST,VX_PORT_COMB};
    static const VxPortKind rotary_ports[] = {VX_PORT_INPUT,VX_PORT_COS,VX_PORT_SIN};
    static const VxPortKind sparse_index_ports[] = {VX_PORT_Q,VX_PORT_K,VX_PORT_INDEX_WEIGHTS,VX_PORT_CANDIDATES};
    static const VxPortKind qsa_ports[] = {VX_PORT_Q,VX_PORT_K,VX_PORT_WEIGHT,VX_PORT_COS,VX_PORT_SIN};
    static const VxPortKind unary_ports[] = {VX_PORT_INPUT};
    static const VxPortKind shard_ports[] = {VX_PORT_INPUT,VX_PORT_WEIGHT};
    static const VxPortKind packed_ports[] = {VX_PORT_INPUT,VX_PORT_WEIGHT,VX_PORT_SCALE,VX_PORT_BIAS};
    const VxPortKind* ports = NULL;
    int count = 0;
    T* x[8] = {0};
    T* y[3] = {nout(n,VX_PORT_OUT),NULL,NULL};
    if (!n || !a || !vx_llm_operator(n->operator_kind)) return 0;
    memset(a,0,sizeof(*a)); a->kind = n->operator_kind;
    switch (a->kind) {
        case VX_OP_GATED_DELTA_NET: ports=delta_ports;count=8;y[1]=nout(n,VX_PORT_STATE);break;
        case VX_OP_SPARSE_ATTENTION: ports=attention_ports;count=8;break;
        case VX_OP_KV_COMPRESS: ports=compress_ports;count=6;break;
        case VX_OP_HYPER_CONNECTION: y[0]=nout(n,VX_PORT_PRE);y[1]=nout(n,VX_PORT_POST);y[2]=nout(n,VX_PORT_COMB); /* fall through */
        case VX_OP_HYPER_HEAD: ports=hyper_ports;count=4;break;
        case VX_OP_HYPER_RESIDUAL: ports=residual_ports;count=4;break;
        case VX_OP_ROTARY_EMBEDDING: ports=rotary_ports;count=3;break;
        case VX_OP_SPARSE_INDEXER: ports=sparse_index_ports;count=4;y[1]=nout(n,VX_PORT_CANDIDATES);break;
        case VX_OP_QSA_INDEXER: ports=qsa_ports;count=5;break;
        case VX_OP_SIGNED_SQRT: ports=unary_ports;count=1;break;
        case VX_OP_EMBEDDING_SHARD: ports=shard_ports;count=2;break;
        case VX_OP_PACKED_LINEAR: ports=packed_ports;count=4;break;
    }
    for (int i=0;i<count;i++) {
        x[i]=nin(n,ports[i]);
        if(a->kind==VX_OP_SPARSE_ATTENTION && i==5 && nin(n,VX_PORT_COMPRESSED_MASK)) {
            if(x[i]||nin(n,VX_PORT_INDEX_K)||nin(n,VX_PORT_INDEX_WEIGHTS))return 0;
            x[i]=nin(n,VX_PORT_COMPRESSED_MASK);
        }
        if (x[i]) {
            if (!vx_llm_tensor_valid(x[i],((a->kind==VX_OP_PACKED_LINEAR&&i==1)||(a->kind==VX_OP_EMBEDDING_SHARD&&i==0)||(a->kind==VX_OP_SPARSE_INDEXER&&i==3)||(a->kind==VX_OP_SPARSE_ATTENTION&&i==5&&nin(n,VX_PORT_COMPRESSED_MASK)))?T_I32:T_F32)) return 0;
            a->x[i]=x[i]->data; a->x_elements[i]=(uint64_t)x[i]->numel;
        }
    }
    for (int i=0;i<3;i++) if (y[i]) {
        if (!vx_llm_tensor_valid(y[i],(a->kind==VX_OP_QSA_INDEXER||a->kind==VX_OP_SPARSE_INDEXER)?T_I32:T_F32)) return 0;
        a->y[i]=y[i]->data;a->y_elements[i]=(uint64_t)y[i]->numel;
        /* WebGPU domain planning temporarily projects maximum descriptors
         * over the bootstrap host addresses. Those ranges are not allocated
         * yet; physical arena aliasing is proved by the sealed GPU planner. */
#if VOLVOXAI_ENABLE_WEBGPU
        if (vx_engine_state_current()->webgpu_validation) continue;
#endif
        uintptr_t start=(uintptr_t)y[i]->data;
        size_t size=(size_t)y[i]->numel*sizeof(float);
        if (start>UINTPTR_MAX-size) return 0;
        for (int j=0;j<count;j++) if (x[j]) {
            uintptr_t other=(uintptr_t)x[j]->data;
            size_t other_size=(size_t)x[j]->numel*sizeof(float);
            if (other>UINTPTR_MAX-other_size || (start<other+other_size && other<start+size)) return 0;
        }
        for (int j=0;j<i;j++) if (y[j]) {
            uintptr_t other=(uintptr_t)y[j]->data;
            if (start<other+(size_t)y[j]->numel*sizeof(float) && other<start+size) return 0;
        }
    }
    if(a->kind==VX_OP_PACKED_LINEAR) {
        if(!x[0]||!x[1]||!x[2]||!y[0]||x[0]->ndim<1||x[1]->ndim!=2||x[2]->ndim!=2||y[0]->ndim!=x[0]->ndim) return 0;
        a->key_dim=x[0]->shape[x[0]->ndim-1];a->value_dim=x[1]->shape[0];
        a->rate=vx_node_param_i32(n,VX_NODE_PARAM_GROUP_SIZE,0);
        a->normalize=vx_node_param_i32(n,VX_NODE_PARAM_ACTIVATION_BITS,0);
        a->groups=vx_node_param_i32(n,VX_NODE_PARAM_WEIGHT_BITS,4);
        if(a->groups!=4&&a->groups!=8)return 0;
        int per_word=32/a->groups;
        if(a->rate!=128||(a->normalize!=8&&a->normalize!=16)||
           x[1]->shape[1]!=(a->key_dim/per_word+(a->key_dim%per_word!=0))||
           x[2]->shape[0]!=a->value_dim||x[2]->shape[1]!=(a->key_dim/128+(a->key_dim%128!=0))||
           y[0]->shape[y[0]->ndim-1]!=a->value_dim||
           (x[3]&&(x[3]->ndim!=1||x[3]->shape[0]!=a->value_dim))) return 0;
        for(int d=0;d<x[0]->ndim-1;d++)if(x[0]->shape[d]!=y[0]->shape[d])return 0;
        long rows=x[0]->numel/a->key_dim;
        if(rows<1||rows>INT_MAX||rows>LONG_MAX/a->value_dim||y[0]->numel!=rows*a->value_dim)return 0;
        a->batch=1;a->sequence=(int32_t)rows;
        /* Required-row initialization owns only the active token prefix.
         * The common decoder shape is [1,capacity,K]; processing its padding
         * made packed prefill scale with cache capacity instead of prompt
         * length. Keep physical spans intact: growing a shortened GPU view
         * later would make its stale host bytes authoritative again. */
        VxEngineState* state=vx_engine_state_current();
        int prefix=state->modern_prefill_rows;
        if(g_active_row<0 && prefix>0 && x[0]->ndim==3 && x[0]->shape[0]==1) {
            if(prefix>x[0]->shape[1])return 0;
            a->sequence=prefix;
            /* Generate's terminal-consumer proof permits just the selected
             * logits row. Keep base addresses and physical owner spans: only
             * the logical work interval changes. Execute retains all rows. */
            if(state->generation_prefill_rows==prefix &&
               state->generation_prefill_capacity==x[0]->shape[1] &&
               state->generation_prefill_logits &&
               !strcmp(y[0]->name,state->generation_prefill_logits)) {
                a->first_unit=(uint64_t)(prefix-1)*(uint32_t)a->value_dim;
                a->unit_count=(uint32_t)a->value_dim;
            }
        }
        return 1;
    }
    if(a->kind==VX_OP_EMBEDDING_SHARD) {
        if(!x[0]||!x[1]||!y[0]||x[1]->ndim!=2||y[0]->ndim!=x[0]->ndim+1||
            !vx_node_param_has(n,VX_NODE_PARAM_VOCAB_OFFSET)||!vx_node_param_has(n,VX_NODE_PARAM_VOCAB_SIZE)) return 0;
        a->batch=a->sequence=1;a->key_dim=x[1]->shape[1];a->value_dim=x[1]->shape[0];
        a->rotary_offset=vx_node_param_i32(n,VX_NODE_PARAM_VOCAB_OFFSET,-1);a->keys=vx_node_param_i32(n,VX_NODE_PARAM_VOCAB_SIZE,0);
        if(a->rotary_offset<0||a->keys<=0||(int64_t)a->rotary_offset+a->value_dim>a->keys||
            y[0]->shape[x[0]->ndim]!=a->key_dim||x[0]->numel>LONG_MAX/a->key_dim||y[0]->numel!=x[0]->numel*a->key_dim) return 0;
        for(int d=0;d<x[0]->ndim;d++) if(x[0]->shape[d]!=y[0]->shape[d]) return 0;
        return 1;
    }
    if (a->kind==VX_OP_SIGNED_SQRT) {
        if(!x[0]||!y[0]||x[0]->ndim!=y[0]->ndim||x[0]->numel!=y[0]->numel) return 0;
        for(int d=0;d<x[0]->ndim;d++) if(x[0]->shape[d]!=y[0]->shape[d]) return 0;
        a->batch=a->sequence=a->key_dim=1;
        a->eps=vx_node_param_f32(n,VX_NODE_PARAM_EPS,1e-6f);
        return isfinite(a->eps)&&a->eps>0.0f;
    }
    if (!x[0] || !y[0] || x[0]->ndim < 3) return 0;
    a->batch=x[0]->shape[0];a->sequence=x[0]->shape[1];
    a->eps=vx_node_param_f32(n,VX_NODE_PARAM_EPS,1e-6f);
    a->hc_eps=vx_node_param_f32(n,VX_NODE_PARAM_NORMALIZATION_EPSILON,1e-6f);
    if (!isfinite(a->eps) || a->eps<=0 || !isfinite(a->hc_eps) || a->hc_eps<=0) return 0;
    if (a->kind==VX_OP_GATED_DELTA_NET || a->kind==VX_OP_SPARSE_ATTENTION) {
        if (!x[1] || !x[2] || x[0]->ndim!=4 || x[1]->ndim!=4 || x[2]->ndim!=4) return 0;
        a->key_dim=x[0]->shape[3];a->value_dim=x[2]->shape[3];
        a->kv_heads=x[1]->shape[2];
        a->heads=a->kind==VX_OP_GATED_DELTA_NET?x[2]->shape[2]:x[0]->shape[2];
        a->scale=vx_node_param_f32(n,VX_NODE_PARAM_SCALE,1.0f/sqrtf((float)a->key_dim));
        if (!isfinite(a->scale) || a->scale<=0 || a->heads%a->kv_heads) return 0;
    }
    if (a->kind==VX_OP_SPARSE_INDEXER) {
        if(!x[1]||!x[2]||x[0]->ndim!=4||x[1]->ndim!=3||x[2]->ndim!=3)return 0;
        a->keys=a->sequence;a->heads=x[0]->shape[2];a->key_dim=x[0]->shape[3];
        a->rate=vx_node_param_i32(n,VX_NODE_PARAM_COMPRESS_RATE,0);
        a->top_k=vx_node_param_i32(n,VX_NODE_PARAM_TOP_K,0);
        a->window=vx_node_param_i32(n,VX_NODE_PARAM_CANDIDATE_BLOCK_SIZE,0);
        a->groups=vx_node_param_i32(n,VX_NODE_PARAM_CANDIDATE_TOP_K,0);
        if(a->rate<=0||a->top_k<=0||a->window<0||a->groups<0||((a->window>0)!=(a->groups>0))||
           (y[1] && (!a->window||x[3]))||(!y[1]&&(a->window||a->groups))||
           x[1]->shape[0]!=a->batch||x[1]->shape[1]!=a->keys||x[1]->shape[2]!=a->key_dim||
           x[2]->shape[0]!=a->batch||x[2]->shape[1]!=a->sequence||x[2]->shape[2]!=a->heads)return 0;
        for(int i=0;i<2;i++)if(y[i]&&(y[i]->ndim!=3||y[i]->shape[0]!=a->batch||y[i]->shape[1]!=a->sequence||y[i]->shape[2]!=a->keys))return 0;
        if(x[3]&&(x[3]->ndim!=3||x[3]->shape[0]!=a->batch||x[3]->shape[1]!=a->sequence||x[3]->shape[2]!=a->keys))return 0;
    } else if (a->kind==VX_OP_QSA_INDEXER) {
        if(!x[1]||!x[2]||!x[3]||!x[4]||x[0]->ndim!=4||x[1]->ndim!=3) return 0;
        a->keys=x[1]->shape[1];a->heads=x[0]->shape[2];a->key_dim=x[0]->shape[3];
        a->rate=vx_node_param_i32(n,VX_NODE_PARAM_COMPRESS_RATE,0);
        a->top_k=vx_node_param_i32(n,VX_NODE_PARAM_TOP_K,0);
        a->rotary_dim=vx_node_param_i32(n,VX_NODE_PARAM_ROTARY_DIM,a->key_dim);
        a->causal=vx_node_param_bool(n,VX_NODE_PARAM_CAUSAL,1);
        if(a->key_dim>512||a->rate<=0||a->top_k<=0||a->rotary_dim<=0||(a->rotary_dim&1)||a->rotary_dim>a->key_dim) return 0;
        T* mask=nin(n,VX_PORT_MASK);
        if(mask) {
            if(!vx_llm_tensor_valid(mask,T_I32)) return 0;
            if(mask->ndim==2&&mask->shape[0]==a->batch&&mask->shape[1]==a->keys) a->mask_mode=2;
            else if(mask->ndim==3&&mask->shape[0]==a->batch&&mask->shape[1]==a->sequence&&mask->shape[2]==a->keys) a->mask_mode=4;
            else return 0;
            a->mask=(const int32_t*)mask->data;a->mask_elements=(uint64_t)mask->numel;
        }
    } else if (a->kind==VX_OP_GATED_DELTA_NET) {
        if (!x[3]||!x[4]||!x[5]||!x[6]||x[0]->shape[2]!=a->kv_heads) return 0;
        a->normalize=vx_node_param_bool(n,VX_NODE_PARAM_NORMALIZE,1);
    } else if (a->kind==VX_OP_SPARSE_ATTENTION) {
        a->keys=x[1]->shape[1];
        a->causal=vx_node_param_bool(n,VX_NODE_PARAM_CAUSAL,1);
        a->window=vx_node_param_i32(n,VX_NODE_PARAM_WINDOW_SIZE,0);
        a->rate=vx_node_param_i32(n,VX_NODE_PARAM_COMPRESS_RATE,1);
        a->top_k=vx_node_param_i32(n,VX_NODE_PARAM_TOP_K,0);
        if (a->window<0||a->rate<=0||a->top_k<0) return 0;
        if (x[3]&&(a->kv_heads!=1||a->value_dim!=a->key_dim||a->sequence!=a->keys)) return 0;
        if(nin(n,VX_PORT_COMPRESSED_MASK)) {
            if(!x[3]||x[5]->ndim!=3||x[5]->shape[0]!=a->batch||x[5]->shape[1]!=a->sequence||x[5]->shape[2]!=a->keys)return 0;
        } else if (x[5]) {
            if (!x[6]||!x[7]||x[5]->ndim!=4) return 0;
            a->index_heads=x[5]->shape[2];a->index_dim=x[5]->shape[3];
        } else if (x[6]||x[7]) return 0;
        T* mask=nin(n,VX_PORT_MASK),*positions=nin(n,VX_PORT_POSITION_IDS);
        if (mask) {
            if (!vx_llm_tensor_valid(mask,T_I32)) return 0;
            if (mask->ndim==1&&mask->shape[0]==a->keys) a->mask_mode=1;
            else if (mask->ndim==2&&mask->shape[0]==a->batch&&mask->shape[1]==a->keys) a->mask_mode=2;
            else if (mask->ndim==2&&mask->shape[0]==a->sequence&&mask->shape[1]==a->keys) a->mask_mode=3;
            else if (mask->ndim==3&&mask->shape[0]==a->batch&&mask->shape[1]==a->sequence&&mask->shape[2]==a->keys) a->mask_mode=4;
            else return 0;
            a->mask=(const int32_t*)mask->data;a->mask_elements=(uint64_t)mask->numel;
        }
        if (positions) {
            if (!vx_llm_tensor_valid(positions,T_I32)||positions->ndim!=1||positions->shape[0]!=a->sequence) return 0;
            a->positions=(const int32_t*)positions->data;a->position_elements=(uint64_t)positions->numel;
        }
    } else if (a->kind==VX_OP_KV_COMPRESS) {
        if (!x[1]||!x[2]||!x[3]||x[0]->ndim!=3||x[3]->ndim!=1) return 0;
        a->key_dim=x[3]->shape[0];a->groups=vx_node_param_i32(n,VX_NODE_PARAM_NUM_GROUPS,1);
        a->rate=vx_node_param_i32(n,VX_NODE_PARAM_COMPRESS_RATE,0);
        if ((a->groups!=1&&a->groups!=2)||a->rate<=0||a->key_dim>INT_MAX/a->groups||x[0]->shape[2]!=a->key_dim*a->groups) return 0;
        if ((x[4]!=NULL)!=(x[5]!=NULL)) return 0;
        if (x[4]) {
            a->rotary_dim=vx_node_param_i32(n,VX_NODE_PARAM_ROTARY_DIM,a->key_dim);
            a->rotary_offset=vx_node_param_i32(n,VX_NODE_PARAM_POSITION_OFFSET,0);
            a->interleaved=vx_node_param_bool(n,VX_NODE_PARAM_INTERLEAVED,0);
            if (a->rotary_dim<=0||(a->rotary_dim&1)||a->rotary_offset<0||a->rotary_dim>a->key_dim||
                a->rotary_offset>a->key_dim-a->rotary_dim||x[4]->ndim!=2||x[5]->ndim!=2||
                x[4]->shape[0]<a->sequence||x[4]->shape[1]!=(a->interleaved?a->rotary_dim/2:a->rotary_dim)||
                x[4]->shape[0]!=x[5]->shape[0]||x[4]->shape[1]!=x[5]->shape[1]) return 0;
        }
    } else if (a->kind==VX_OP_ROTARY_EMBEDDING) {
        if (!x[1]||!x[2]||x[0]->ndim!=4) return 0;
        a->heads=x[0]->shape[2];a->key_dim=x[0]->shape[3];
        a->rotary_dim=vx_node_param_i32(n,VX_NODE_PARAM_ROTARY_DIM,a->key_dim);
        a->rotary_offset=vx_node_param_i32(n,VX_NODE_PARAM_POSITION_OFFSET,0);
        a->interleaved=vx_node_param_bool(n,VX_NODE_PARAM_INTERLEAVED,0);
        if (a->rotary_dim<=0||(a->rotary_dim&1)||a->rotary_offset<0||a->rotary_dim>a->key_dim||a->rotary_offset>a->key_dim-a->rotary_dim) return 0;
    } else {
        if (!x[1]||!x[2]||!x[3]||x[0]->ndim!=4) return 0;
        a->streams=x[0]->shape[2];a->key_dim=x[0]->shape[3];
        if (a->streams>INT_MAX/a->key_dim||a->streams>46338) return 0;
        a->iterations=vx_node_param_i32(n,VX_NODE_PARAM_ITERATIONS,20);
        if (a->iterations<=0 || (a->kind==VX_OP_HYPER_CONNECTION&&(!y[1]||!y[2]))) return 0;
    }
    if(a->kind==VX_OP_GATED_DELTA_NET && vx_engine_state_current()->modern_prefill_rows>0) {
        int prefix=vx_engine_state_current()->modern_prefill_rows;
        if(a->batch!=1 || prefix>a->sequence || !a->y[1] || a->x[7]) return 0;
        a->sequence=prefix;
    }
    return vx_llm_work_items(a)>0;
}

int vx_engine_llm_row_args(Node* n, int row, VxLLMOpArgs* a) {
    if (!n || row < 0 || !a || g_decode_slots > 1 || !vx_engine_llm_node_args(n,a)) return 0;
    if(a->kind==VX_OP_SPARSE_INDEXER || (a->kind==VX_OP_SPARSE_ATTENTION && a->x[5]&&!a->index_heads))return 0;
    T* output=nout(n,n->operator_kind==VX_OP_HYPER_CONNECTION?VX_PORT_PRE:VX_PORT_OUT);
    /* Explicit row probes need the physical capacity even during prefill,
     * before the execution loop has installed g_active_row. */
    if(a->kind==VX_OP_PACKED_LINEAR) {
        if(!output)return 0;
        a->sequence=(int32_t)(output->numel/a->value_dim);
        a->first_unit=0;a->unit_count=0;
    }
    int sequence=a->sequence;
    /* A full prefill descriptor caps recurrence work to the active prompt.
     * Row admission still addresses the physical retained token capacity. */
    if(a->kind==VX_OP_GATED_DELTA_NET) {
        if(!output || output->ndim!=4) return 0;
        sequence=output->shape[1];
    }
    if (a->kind==VX_OP_SIGNED_SQRT || a->kind==VX_OP_EMBEDDING_SHARD) {
        if (!output || output->ndim<2 || output->shape[0]!=1) return 0;
        sequence=output->shape[1];
    }
    if (a->batch!=1 || row>=sequence) return 0;
    if (a->kind==VX_OP_KV_COMPRESS) {
        if ((row+1)%a->rate) return 2;
        a->first_unit=(uint64_t)(row+1)/a->rate-1;
        a->unit_count=1;
        return 1;
    }
    unsigned varying=0;
    switch(a->kind) {
        case VX_OP_GATED_DELTA_NET:
            if (!a->y[1] || a->x[7]) return 0;
            varying=31u;
            /* Every CPU head / CUDA column reads its independent state into
             * private scratch before any write. No cross-column alias exists. */
            a->x[7]=a->y[1];a->x_elements[7]=a->y_elements[1];break;
        case VX_OP_SPARSE_ATTENTION: varying=(1u<<0)|(1u<<5)|(1u<<7);break;
        case VX_OP_QSA_INDEXER: varying=1u;break;
        case VX_OP_HYPER_RESIDUAL: varying=15u;break;
        case VX_OP_ROTARY_EMBEDDING: varying=7u;break;
        default: varying=1u;break;
    }
    for (int i=0;i<8;i++) if ((varying&(1u<<i)) && a->x[i]) {
        if (a->x_elements[i]%(uint32_t)sequence) return 0;
        uint64_t width=a->x_elements[i]/(uint32_t)sequence;
        a->x[i]+=width*(uint32_t)row;a->x_elements[i]=width;
    }
    for (int i=0;i<3;i++) if (a->y[i] && !(a->kind==VX_OP_GATED_DELTA_NET && i==1)) {
        if (a->y_elements[i]%(uint32_t)sequence) return 0;
        uint64_t width=a->y_elements[i]/(uint32_t)sequence;
        a->y[i]+=width*(uint32_t)row;a->y_elements[i]=width;
    }
    if(a->mask && (a->mask_mode==3 || a->mask_mode==4)) {
        a->mask+=(uint64_t)row*a->keys;
        a->mask_elements=a->keys;a->mask_mode=1;
    }
    if(a->positions) {a->positions+=row;a->position_elements=1;}
    a->sequence=1;a->query_offset=row;
    return 1;
}

/* Preserve ordinary validation/unsupported failures as -1 while distinguishing
 * a device submission failure that must stop backend fallback. */
enum { VX_LLM_NODE_SUBMITTED_ERROR = -2 };

static int vx_llm_node_run(Node* n, int cuda) {
    VxLLMOpArgs args;
    if (!vx_llm_operator(n->operator_kind)) return 0;
    if (g_prefix_rows>0 &&
        (!cuda || n->operator_kind!=VX_OP_PACKED_LINEAR ||
         g_active_row>=0 ||
         vx_engine_state_current()->generation_prefill_rows!=g_prefix_rows ||
         vx_engine_state_current()->modern_prefill_rows!=g_prefix_rows)) return -1;
#if VOLVOXAI_ENABLE_TRAINING
    if (g_native_training_mode) return -1;
#endif
    int described=g_active_row>=0?vx_engine_llm_row_args(n,g_active_row,&args):vx_engine_llm_node_args(n,&args);
    if (!described) return -1;
    if (described==2) return 1;
    int prefix=vx_engine_state_current()->modern_prefill_rows;
    if (g_active_row<0 && prefix>0 && args.kind==VX_OP_GATED_DELTA_NET) {
        if(args.batch!=1 || prefix>args.sequence || !args.y[1] || args.x[7]) return -1;
        args.sequence=prefix;
    }
    if(!cuda && !g_vx_cpu_dispatch) {
#if VOLVOXAI_ENABLE_VULKAN || VOLVOXAI_ENABLE_OPENGL || VOLVOXAI_ENABLE_METAL
        VxLLMOpArgs full;
        if(g_active_row>=0 && !vx_engine_llm_node_args(n,&full)) return -1;
#if VOLVOXAI_ENABLE_VULKAN
        if(g_use_vulkan) return (g_active_row>=0?vk_graph_llm_row_f32(&full,&args):vk_graph_llm_f32(&args))?1:-1;
#endif
#if VOLVOXAI_ENABLE_OPENGL
        if(g_use_opengl) return (g_active_row>=0?opengl_graph_llm_row_f32(&full,&args):opengl_graph_llm_f32(&args))?1:-1;
#endif
#if VOLVOXAI_ENABLE_METAL
        if(g_use_metal) return (g_active_row>=0?metal_graph_llm_row_f32(&full,&args):metal_graph_llm_f32(&args))?1:-1;
#endif
#endif
    }
    if (!cuda) {
        for (int i=0;i<n->nin;i++) { T* input=t_find(n->ins[i].name); if (input&&input->dtype==T_F32) materialize_tensor_f32(input); }
        if (args.positions) for (uint64_t i=0;i<args.position_elements;i++) if (args.positions[i]<0||args.positions[i]==INT_MAX) return -1;
        if(args.kind==VX_OP_EMBEDDING_SHARD) for(uint64_t i=0;i<args.x_elements[0];i++) {
            int32_t id=((const int32_t*)args.x[0])[i];if(id<0||id>=args.keys) return -1;
        }
        return vx_llm_forward_f32(&args)==0?1:-1;
    }
#if VOLVOXAI_ENABLE_CUDA
    int status=cuda_graph_llm_f32(&args);
    return status<0?VX_LLM_NODE_SUBMITTED_ERROR:status?1:-1;
#else
    return -1;
#endif
}

#if VOLVOXAI_ENABLE_VULKAN || VOLVOXAI_ENABLE_OPENGL || VOLVOXAI_ENABLE_METAL || VOLVOXAI_ENABLE_WEBGPU
int vx_engine_llm_router_args(Node* n, VxLlmGpuRouterArgs* a) {
    T *input=nin(n,VX_PORT_INPUT), *weight=nin(n,VX_PORT_WEIGHT), *bias=nin(n,VX_PORT_BIAS);
    T *correction=nin(n,VX_PORT_CORRECTION_BIAS), *tokens=nin(n,VX_PORT_TOKEN_IDS), *hash=nin(n,VX_PORT_HASH_INDICES);
    T *indices=nout(n,VX_PORT_INDICES), *routes=nout(n,VX_PORT_WEIGHTS);
    if(!input) input=nin(n,VX_PORT_X);
    if(!weight) weight=nin(n,VX_PORT_ROUTER_WEIGHT);
    if(!indices) indices=nout(n,VX_PORT_ROUTE_INDICES);
    if(!routes) routes=nout(n,VX_PORT_ROUTE_WEIGHTS);
    if(!a || !n || n->operator_kind!=VX_OP_MOE_ROUTER ||
       !vx_llm_tensor_valid(input,T_F32) || !vx_llm_tensor_valid(weight,T_F32) ||
       !vx_llm_tensor_valid(indices,T_F32) || !vx_llm_tensor_valid(routes,T_F32) ||
       input->ndim<1 || weight->ndim!=2 ||
       (bias&&!vx_llm_tensor_valid(bias,T_F32)) || (correction&&!vx_llm_tensor_valid(correction,T_F32))) return 0;
    memset(a,0,sizeof(*a));
    a->d_in=input->shape[input->ndim-1]; a->experts=vx_node_param_i32(n,VX_NODE_PARAM_NUM_EXPERTS,weight->shape[1]);
    if(weight->shape[1]!=(int)a->experts)return 0;
    if(weight->shape[0]!=(int)a->d_in || input->numel/a->d_in>UINT32_MAX) return 0;
    a->rows=(uint32_t)(input->numel/a->d_in);
    a->top_k=vx_node_param_i32(n,VX_NODE_PARAM_TOP_K,2);
    if((uint64_t)a->rows*a->top_k!=(uint64_t)indices->numel || routes->numel!=indices->numel ||
       (bias&&bias->numel!=(long)a->experts) || (correction&&correction->numel!=(long)a->experts)) return 0;
    a->input=input->data; a->weight=weight->data; a->bias=bias?bias->data:NULL;
    a->correction=correction?correction->data:NULL; a->indices=indices->data; a->routes=routes->data;
    a->temperature=vx_node_param_f32(n,VX_NODE_PARAM_TEMPERATURE,1.0f);
    a->normalize=vx_node_param_bool(n,VX_NODE_PARAM_NORMALIZE,1);
    VxNodeParamSymbol scoring=vx_node_param_symbol(n,VX_NODE_PARAM_SCORING_FUNC,VX_NODE_SYMBOL_SOFTMAX);
    a->scoring=scoring==VX_NODE_SYMBOL_SOFTMAX?0u:scoring==VX_NODE_SYMBOL_SIGMOID?1u:scoring==VX_NODE_SYMBOL_SQRTSOFTPLUS?2u:3u;
    a->groups=vx_node_param_i32(n,VX_NODE_PARAM_NUM_GROUPS,1);
    a->selected_groups=vx_node_param_i32(n,VX_NODE_PARAM_TOPK_GROUPS,(int)a->groups);
    a->group_top_k=vx_node_param_i32(n,VX_NODE_PARAM_GROUP_SCORE_TOP_K,1);
    a->scale=vx_node_param_f32(n,VX_NODE_PARAM_ROUTED_SCALING_FACTOR,1.0f);
    a->epsilon=vx_node_param_f32(n,VX_NODE_PARAM_NORMALIZATION_EPSILON,0.0f);
    if((tokens!=NULL)!=(hash!=NULL)) return 0;
    if(hash) {
        if(!vx_llm_tensor_valid(tokens,T_I32) || !vx_llm_tensor_valid(hash,T_I32) ||
           hash->ndim!=2 || hash->shape[1]!=(int)a->top_k || tokens->numel!=(long)a->rows) return 0;
        a->tokens=(const int32_t*)tokens->data; a->hash=(const int32_t*)hash->data;
        a->vocabulary=hash->shape[0];
    }
    return vx_llm_gpu_router_plan(a,NULL,NULL);
}

int vx_engine_llm_gated_args(Node* n, VxLlmGpuGatedArgs* a) {
    T *input=nin(n,VX_PORT_INPUT), *gate=nin(n,VX_PORT_GATE_WEIGHT), *up=nin(n,VX_PORT_UP_WEIGHT), *down=nin(n,VX_PORT_DOWN_WEIGHT);
    T *ids=nin(n,VX_PORT_ROUTE_INDICES), *routes=nin(n,VX_PORT_ROUTE_WEIGHTS), *output=nout(n,VX_PORT_OUT);
    if(!input) input=nin(n,VX_PORT_X);
    if(!n || !a || n->operator_kind!=VX_OP_MOE_GATED_MLP ||
       !vx_llm_tensor_valid(input,T_F32) || !vx_llm_tensor_valid(gate,T_F32) ||
       !vx_llm_tensor_valid(up,T_F32) || !vx_llm_tensor_valid(down,T_F32) ||
       !vx_llm_tensor_valid(ids,T_F32) || !vx_llm_tensor_valid(routes,T_F32) ||
       !vx_llm_tensor_valid(output,T_F32) || input->ndim<1 || output->ndim<1 ||
       gate->ndim!=3 || up->ndim!=3 || down->ndim!=3) return 0;
    memset(a,0,sizeof(*a));
    a->experts=gate->shape[0];a->d_in=gate->shape[1];a->hidden=gate->shape[2];a->d_out=down->shape[2];
    for(int i=0;i<3;i++) if(up->shape[i]!=gate->shape[i]) return 0;
    if(down->shape[0]!=(int)a->experts || down->shape[1]!=(int)a->hidden ||
       input->shape[input->ndim-1]!=(int)a->d_in || output->shape[output->ndim-1]!=(int)a->d_out ||
       input->numel/a->d_in>UINT32_MAX) return 0;
    a->rows=(uint32_t)(input->numel/a->d_in);
    if(!a->rows || output->numel!=(long)a->rows*a->d_out || ids->numel!=routes->numel || ids->numel%a->rows ||
       ids->numel/a->rows>UINT32_MAX) return 0;
    a->top_k=(uint32_t)(ids->numel/a->rows);a->input=input->data;a->gate=gate->data;a->up=up->data;a->down=down->data;
    a->indices=ids->data;a->routes=routes->data;a->output=output->data;
    a->input_scale=vx_node_param_bool(n,VX_NODE_PARAM_ROUTE_INPUT_SCALE,0);
    a->limit=vx_node_param_f32(n,VX_NODE_PARAM_SWIGLU_LIMIT,0.0f);
    uint64_t scratch[2];return vx_llm_gpu_gated_plan(a,scratch,NULL,NULL);
}

int vx_engine_llm_ngram_args(Node* n, VxLlmGpuNgramArgs* a) {
    T *tokens=nin(n,VX_PORT_TOKEN_IDS), *multipliers=nin(n,VX_PORT_MULTIPLIERS), *sizes=nin(n,VX_PORT_VOCAB_SIZES), *offsets=nin(n,VX_PORT_OFFSETS), *output=nout(n,VX_PORT_OUT);
    if(!n || !a || n->operator_kind!=VX_OP_N_GRAM_IDS ||
       !vx_llm_tensor_valid(tokens,T_I32) || !vx_llm_tensor_valid(multipliers,T_I32) ||
       !vx_llm_tensor_valid(sizes,T_I32) || !vx_llm_tensor_valid(offsets,T_I32) ||
       !vx_llm_tensor_valid(output,T_I32) || tokens->ndim!=2 || multipliers->ndim!=2 ||
       sizes->ndim!=1 || offsets->ndim!=1 || output->ndim!=3) return 0;
    memset(a,0,sizeof(*a));a->batch=tokens->shape[0];a->sequence=tokens->shape[1];
    a->ngram=vx_node_param_i32(n,VX_NODE_PARAM_NGRAM_SIZE,0);a->heads_per_ngram=vx_node_param_i32(n,VX_NODE_PARAM_HEADS_PER_NGRAM,0);
    a->eos=vx_node_param_i32(n,VX_NODE_PARAM_EOS_TOKEN_ID,-1);a->vocabulary=vx_node_param_i32(n,VX_NODE_PARAM_VOCAB_SIZE,0);
    a->pad=vx_node_param_i32(n,VX_NODE_PARAM_PAD_TOKEN_ID,a->eos);
    uint64_t heads=a->ngram>=2?(uint64_t)(a->ngram-1)*a->heads_per_ngram:0;
    if(heads>INT32_MAX || multipliers->shape[0]!=(int)a->ngram || multipliers->shape[1]!=2 ||
       sizes->numel!=(long)heads || offsets->numel!=(long)heads || output->shape[0]!=(int)a->batch ||
       output->shape[1]!=(int)a->sequence || output->shape[2]!=(int)heads) return 0;
    a->tokens=(const int32_t*)tokens->data;a->multipliers=(const int32_t*)multipliers->data;a->sizes=(const int32_t*)sizes->data;
    a->offsets=(const int32_t*)offsets->data;a->output=(int32_t*)output->data;
    if(g_active_row>=0) {
        if(a->batch!=1 || g_decode_slots>1 || (uint32_t)g_active_row>=a->sequence) return 0;
        a->first_unit=(uint64_t)(uint32_t)g_active_row*heads;a->unit_count=heads;
    }
    return vx_llm_gpu_ngram_plan(a,NULL,NULL);
}
#endif
