/* Preserved source009 host lowering, before allallowed-mask selection. */
static int reference_vx_decode_cuda_attention(Node* node, const VxDecodeRowProjection* p,
    T* output, const int32_t* mask) {
    T* q = nin(node, VX_PORT_Q), *k = nin(node, VX_PORT_K), *v = nin(node, VX_PORT_V);
    if (node->operator_kind != VX_OP_CROSS_SDPA || !q || !k || !v ||
        q->dtype != T_F32 || k->dtype != T_F32 || v->dtype != T_F32 || output->dtype != T_F32)
        return -1;
    int width = q->shape[q->ndim - 1], kv_width = k->shape[k->ndim - 1];
    int heads = node->parsed_params.heads, kv_heads = node->parsed_params.kv_heads;
    if (heads <= 0 || kv_heads <= 0 || heads % kv_heads || width <= 0 ||
        width % heads || kv_width <= 0 || kv_width % kv_heads ||
        width / heads != kv_width / kv_heads || width / heads > 128 ||
        v->shape[v->ndim - 1] != kv_width || output->shape[output->ndim - 1] != width) return -1;
    float scale = node->parsed_params.has_scale ? node->parsed_params.scale : 1.f / sqrtf((float)(width / heads));
    if (!isfinite(scale)) return -1;
    /* Absolute query position and per-slot length were projected into mask.
     * A local query row numbered zero must not reapply causal clipping. */
    return cuda_graph_cross_sdpa_gqa_f32(q->data, k->data, v->data, mask,
        (long)p->rows.slots * p->keys, output->data, 1, p->keys, width,
        kv_width, heads, kv_heads, width / heads, p->rows.slots, scale,
        0, ATTN_MASK_BATCH_KEY) ? 0 : -1;
}
