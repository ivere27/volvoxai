/* Immutable policy copying and generated checkpoint/state messages. */
static void trainer_qat_free(VxTrainingQatPolicy* policy) {
    if (policy) {
        for (size_t i = 0; policy->nodes && i < policy->count; i++) free(policy->nodes[i].node_id);
        free(policy->nodes); free(policy);
    }
}

static VxStatus trainer_qat_copy(const VolvoxaiV1QatOptions* input,
                                VxTrainingQatPolicy** output) {
    *output = NULL;
    if (!input) return VX_STATUS_OK;
    if (!input->field_nodes.len || !input->field_nodes.data ||
        input->field_nodes.len > INT_MAX ||
        input->field_nodes.len > SIZE_MAX / sizeof(VxTrainingQatNode)) return VX_STATUS_INVALID_ARGUMENT;
    VxTrainingQatPolicy* policy = calloc(1, sizeof(*policy));
    if (!policy) return VX_STATUS_OUT_OF_MEMORY;
    policy->count = input->field_nodes.len;
    policy->nodes = calloc(policy->count, sizeof(*policy->nodes));
    if (!policy->nodes) { trainer_qat_free(policy); return VX_STATUS_OUT_OF_MEMORY; }
    for (size_t i = 0; i < policy->count; i++) {
        const VolvoxaiV1QatNode* node = &input->field_nodes.data[i];
        VxTrainingQatNode* target = &policy->nodes[i];
        if (!node->field_node_id.len || node->field_node_id.len == SIZE_MAX ||
            !node->field_node_id.data || memchr(node->field_node_id.data, 0, node->field_node_id.len) ||
            !node->field_weights || !node->field_weights->field_format) goto invalid;
        const VolvoxaiV1WeightQuantization* weights = node->field_weights;
        target->bits = weights->field_format->field_bits;
        target->group_size = weights->field_format->field_group_size;
        target->algorithm = weights->field_algorithm;
        if ((target->bits != 2 && target->bits != 3 && target->bits != 4) ||
            (target->algorithm != VX_WEIGHT_ALGORITHM_RTN &&
             target->algorithm != VX_WEIGHT_ALGORITHM_MSE_CLIP) ||
            (target->group_size && target->group_size != 32 && target->group_size != 64 &&
             target->group_size != 128 && target->group_size != 256)) goto invalid;
        target->node_id = malloc(node->field_node_id.len + 1u);
        if (!target->node_id) { trainer_qat_free(policy); return VX_STATUS_OUT_OF_MEMORY; }
        memcpy(target->node_id, node->field_node_id.data, node->field_node_id.len);
        target->node_id[node->field_node_id.len] = 0;
        for (size_t j = 0; j < i; j++)
            if (!strcmp(target->node_id, policy->nodes[j].node_id)) goto invalid;
    }
    *output = policy;
    return VX_STATUS_OK;
invalid:
    trainer_qat_free(policy);
    return VX_STATUS_INVALID_ARGUMENT;
}

static int trainer_qat_matches(const VxTrainingQatPolicy* policy,
                                const VolvoxaiV1QatOptions* message) {
    if (!policy || !message) return !policy && !message;
    if (policy->count != message->field_nodes.len || !message->field_nodes.data) return 0;
    for (size_t i = 0; i < policy->count; i++) {
        const VxTrainingQatNode* node = &policy->nodes[i];
        int found = 0;
        for (size_t j = 0; j < message->field_nodes.len; j++) {
            const VolvoxaiV1QatNode* candidate = &message->field_nodes.data[j];
            if (candidate->field_node_id.len != strlen(node->node_id) ||
                !candidate->field_node_id.data ||
                memcmp(candidate->field_node_id.data, node->node_id, candidate->field_node_id.len)) continue;
            if (found || !candidate->field_weights || !candidate->field_weights->field_format ||
                candidate->field_weights->field_algorithm != node->algorithm ||
                candidate->field_weights->field_format->field_bits != node->bits ||
                candidate->field_weights->field_format->field_group_size != node->group_size) return 0;
            found = 1;
        }
        if (!found) return 0;
    }
    return 1;
}

static VolvoxaiV1QatOptions* trainer_qat_message(const SynurangLiteAllocator* allocator,
                                               const VxTrainingQatPolicy* policy) {
    if (!policy) return NULL;
    VolvoxaiV1QatOptions* out = allocator->allocate(allocator->context, sizeof(*out));
    if (!out) return NULL;
    volvoxai_v1_qat_options_init_with_allocator(out, allocator);
    for (size_t i = 0; i < policy->count; i++) {
        VolvoxaiV1QatNode* node = volvoxai_v1_qat_options_add_nodes(out);
        if (!node || synurang_lite_bytes_assign(allocator, &node->field_node_id,
            policy->nodes[i].node_id, strlen(policy->nodes[i].node_id))) goto fail;
        node->field_weights = allocator->allocate(allocator->context, sizeof(*node->field_weights));
        if (!node->field_weights) goto fail;
        volvoxai_v1_weight_quantization_init_with_allocator(node->field_weights, allocator);
        node->field_weights->field_algorithm = policy->nodes[i].algorithm;
        node->field_weights->field_format = allocator->allocate(allocator->context,
            sizeof(*node->field_weights->field_format));
        if (!node->field_weights->field_format) goto fail;
        volvoxai_v1_weight_format_init_with_allocator(node->field_weights->field_format, allocator);
        node->field_weights->field_format->field_bits = policy->nodes[i].bits;
        node->field_weights->field_format->field_group_size = policy->nodes[i].group_size;
    }
    return out;
fail:
    volvoxai_v1_qat_options_free(out);
    allocator->deallocate(allocator->context, out);
    return NULL;
}
