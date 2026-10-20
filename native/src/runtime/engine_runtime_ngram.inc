#include "../kernels/ngram_ids.h"

static int ngram_ids_forward(Node* node,T* out) {
    T* tokens=nin(node,VX_PORT_TOKEN_IDS),*multipliers=nin(node,VX_PORT_MULTIPLIERS),
        *sizes=nin(node,VX_PORT_VOCAB_SIZES),*offsets=nin(node,VX_PORT_OFFSETS);
    if(!vx_node_param_has(node,VX_NODE_PARAM_NGRAM_SIZE) || !vx_node_param_has(node,VX_NODE_PARAM_HEADS_PER_NGRAM) ||
        !vx_node_param_has(node,VX_NODE_PARAM_EOS_TOKEN_ID) || !vx_node_param_has(node,VX_NODE_PARAM_VOCAB_SIZE)) return -1;
    int n=vx_node_param_i32(node,VX_NODE_PARAM_NGRAM_SIZE,2),p=vx_node_param_i32(node,VX_NODE_PARAM_HEADS_PER_NGRAM,1),
        eos=vx_node_param_i32(node,VX_NODE_PARAM_EOS_TOKEN_ID,-1),vocab=vx_node_param_i32(node,VX_NODE_PARAM_VOCAB_SIZE,0);
    int pad=vx_node_param_i32(node,VX_NODE_PARAM_PAD_TOKEN_ID,eos);
    if(!tokens || !multipliers || !sizes || !offsets || !out || tokens->dtype!=T_I32 ||
        multipliers->dtype!=T_I32 || sizes->dtype!=T_I32 || offsets->dtype!=T_I32 || out->dtype!=T_I32 ||
        tokens->ndim!=2 || multipliers->ndim!=2 || sizes->ndim!=1 || offsets->ndim!=1 || out->ndim!=3 ||
        n<2 || p<1 || (uint64_t)(n-1)*(uint32_t)p>INT32_MAX) return -1;
    int heads=(n-1)*p;
    if(multipliers->shape[0]!=n || multipliers->shape[1]!=2 || sizes->shape[0]!=heads || offsets->shape[0]!=heads ||
        out->shape[0]!=tokens->shape[0] || out->shape[1]!=tokens->shape[1] || out->shape[2]!=heads) return -1;
    if(g_active_row>=0) {
        int row=g_active_row;
        if(g_decode_slots>1 || tokens->shape[0]!=1 || row>=tokens->shape[1] || vocab<=0 || eos< -1 || eos>=vocab || pad<0 || pad>=vocab) return -1;
#if VOLVOXAI_ENABLE_CUDA
        if(g_use_cuda && !g_vx_cpu_dispatch) return cuda_graph_ngram_ids_rows_i32(
            (const int32_t*)tokens->data,(const int32_t*)multipliers->data,(const int32_t*)sizes->data,
            (const int32_t*)offsets->data,(int32_t*)out->data,1,(uint32_t)tokens->shape[1],(uint32_t)n,(uint32_t)p,eos,pad,vocab,(uint32_t)row,1)?0:-1;
#endif
        const int32_t* ids=(const int32_t*)tokens->data;
        for(int shift=0;shift<n && shift<=row;shift++) if(ids[row-shift]<0 || ids[row-shift]>=vocab) return -1;
        for(int head=0;head<heads;head++) {
            const int32_t* sizes_data=(const int32_t*)sizes->data,*offsets_data=(const int32_t*)offsets->data;
            if(sizes_data[head]<=0 || offsets_data[head]<0 || (int64_t)offsets_data[head]+sizes_data[head]>INT32_MAX) return -1;
            ((int32_t*)out->data)[(size_t)row*heads+head]=vx_ngram_id_at(ids,(const int32_t*)multipliers->data,
                sizes_data,offsets_data,(uint32_t)tokens->shape[1],(uint32_t)p,eos,pad,0,(uint32_t)row,(uint32_t)head);
        }
        return 0;
    }
    return vx_ngram_ids_i32((const int32_t*)tokens->data,(const int32_t*)multipliers->data,
        (const int32_t*)sizes->data,(const int32_t*)offsets->data,(int32_t*)out->data,
        (uint32_t)tokens->shape[0],(uint32_t)tokens->shape[1],(uint32_t)n,(uint32_t)p,eos,pad,vocab) ? 0 : -1;
}
