/* Load-time weight-bank residency.
 *
 * Each private engine state owns its tensor descriptor table and the matching
 * residency maps. A standalone mutable engine compacts its loaded file bytes
 * in place. A compiled-model context instead borrows immutable file bytes and
 * copies only selected bank rows into a context-owned overlay. Both happen
 * before build_graph, so every downstream shape check sees the staged extent.
 *
 * Route indices keep the exporter's global slot ids at execution; each node that
 * reads a sliced bank carries a global-id -> staged-row table. */

static void vx_bank_residency_clear(void) {
    for (size_t index = 0; index < g_bank_residency_count; index++) {
        free(g_bank_residency[index].slot_rows);
        g_bank_residency[index].slot_rows = NULL;
    }
    free(g_bank_residency);
    g_bank_residency = NULL;
    g_bank_residency_count = 0;
    g_bank_residency_capacity = 0;
}

/* Record one requested bank. Slots must be ascending and unique; the caller
 * (public_api) has already checked that against the graph's "banks" table. */
int volvoxai_engine_add_bank_residency(const char* bank, const uint32_t* slots,
                                       size_t slot_count) {
    VxBankResidencyState* grown;
    VxBankResidencyState* state;
    uint32_t* slot_rows;
    uint32_t domain;
    if (!bank || !bank[0] || !slots || !slot_count ||
        strlen(bank) >= sizeof(state->bank) ||
        slot_count > UINT32_MAX) return -1;
    for (size_t index = 1; index < slot_count; index++)
        if (slots[index] <= slots[index - 1]) return -1;
    if (slots[slot_count - 1] == UINT32_MAX) return -1;
    domain = slots[slot_count - 1] + 1u;
    if ((size_t)domain > SIZE_MAX / sizeof(*slot_rows)) return -1;
    slot_rows = (uint32_t*)malloc((size_t)domain * sizeof(*slot_rows));
    if (!slot_rows) return -1;
    for (uint32_t slot = 0; slot < domain; slot++)
        slot_rows[slot] = VX_MOE_SLOT_ABSENT;
    for (size_t index = 0; index < slot_count; index++)
        slot_rows[slots[index]] = (uint32_t)index;
    if (g_bank_residency_count == g_bank_residency_capacity) {
        size_t next_capacity = g_bank_residency_capacity
            ? g_bank_residency_capacity * 2u : 8u;
        if (next_capacity < g_bank_residency_capacity ||
            next_capacity > SIZE_MAX / sizeof(*g_bank_residency)) {
            free(slot_rows);
            return -1;
        }
        grown = (VxBankResidencyState*)realloc(
            g_bank_residency, next_capacity * sizeof(*g_bank_residency));
        if (!grown) {
            free(slot_rows);
            return -1;
        }
        g_bank_residency = grown;
        g_bank_residency_capacity = next_capacity;
    }
    state = &g_bank_residency[g_bank_residency_count];
    memset(state, 0, sizeof(*state));
    memcpy(state->bank, bank, strlen(bank) + 1);
    state->slot_rows = slot_rows;
    state->slot_domain = domain;
    state->staged_rows = (uint32_t)slot_count;
    g_bank_residency_count++;
    return 0;
}

static VxBankResidencyState* vx_bank_residency_find(const char* name) {
    if (!name) return NULL;
    for (size_t index = 0; index < g_bank_residency_count; index++)
        if (!strcmp(g_bank_residency[index].bank, name))
            return &g_bank_residency[index];
    return NULL;
}

/* Stage each requested bank. Mutable standalone engines compact in place:
 * resident slot s moves to row r, and r is never greater than s. Compiled
 * contexts keep the owner's source bytes untouched and allocate only the
 * selected rows; the borrowed descriptor table is private to the context. */
static int vx_bank_residency_apply(void) {
    for (size_t index = 0; index < g_bank_residency_count; index++) {
        VxBankResidencyState* state = &g_bank_residency[index];
        SafetensorsTensor* stored = NULL;
        T* tensor;
        size_t slot_bytes;
        int file_index;
        state->applied = 0;
        for (file_index = 0; file_index < g_weight_file_count; file_index++) {
            stored = safetensors_find_tensor_mutable(&g_weight_files[file_index],
                                                     state->bank);
            if (stored) break;
        }
        if (!stored) {
            fprintf(stderr, "[engine] bank '%s' is not a loaded weight\n", state->bank);
            return -1;
        }
        if (stored->ndim < 2 || stored->shape[0] <= 0 ||
            (uint32_t)stored->shape[0] < state->slot_domain) {
            fprintf(stderr,
                    "[engine] bank '%s' has %d slots and cannot satisfy a residency "
                    "domain of %u\n",
                    state->bank, stored->ndim >= 1 ? stored->shape[0] : 0,
                    state->slot_domain);
            return -1;
        }
        if (!stored->data || !stored->nbytes ||
            stored->nbytes % (size_t)stored->shape[0]) return -1;
        slot_bytes = stored->nbytes / (size_t)stored->shape[0];
        if ((size_t)state->staged_rows > SIZE_MAX / slot_bytes) return -1;
        tensor = t_find(state->bank);
        if (!tensor || tensor->owns || tensor->data != stored->data) return -1;
        if (g_weight_files_borrowed) {
            size_t staged_bytes = (size_t)state->staged_rows * slot_bytes;
            unsigned char* staged = (unsigned char*)malloc(staged_bytes);
            if (!staged) return -1;
            for (uint32_t slot = 0; slot < state->slot_domain; slot++) {
                uint32_t row = state->slot_rows[slot];
                if (row == VX_MOE_SLOT_ABSENT) continue;
                memcpy(staged + (size_t)row * slot_bytes,
                       (const unsigned char*)stored->data +
                           (size_t)slot * slot_bytes,
                       slot_bytes);
            }
            tensor->data = (float*)staged;
            tensor->owns = 1;
            stored->data = staged;
        } else {
            for (uint32_t slot = 0; slot < state->slot_domain; slot++) {
                uint32_t row = state->slot_rows[slot];
                if (row == VX_MOE_SLOT_ABSENT || row == slot) continue;
                memmove((unsigned char*)stored->data +
                            (size_t)row * slot_bytes,
                        (const unsigned char*)stored->data +
                            (size_t)slot * slot_bytes,
                        slot_bytes);
            }
        }
        stored->shape[0] = (int)state->staged_rows;
        stored->nbytes = (size_t)state->staged_rows * slot_bytes;
        /* The engine registers a T for each safetensors entry while loading, so
         * the already-published tensor has to follow the staged extent. The
         * standalone path still aliases its blob; the compiled path points both
         * private descriptors at the context-owned overlay. */
        {
            long numel = 1;
            tensor->shape[0] = (int)state->staged_rows;
            for (int axis = 0; axis < tensor->ndim; axis++)
                numel *= tensor->shape[axis];
            tensor->numel = numel;
        }
        state->applied = 1;
    }
    return 0;
}

/* Attach the slot table to every node that reads a sliced bank, so the kernels
 * can keep interpreting route indices as global ids. */
static void vx_bank_residency_bind_nodes(void) {
    if (!g_bank_residency_count) return;
    for (int node_index = 0; node_index < g_nn; node_index++) {
        Node* node = &g_n[node_index];
        for (int input = 0; input < node->nin; input++) {
            const VxBankResidencyState* state =
                vx_bank_residency_find(node->ins[input].name);
            if (!state || !state->applied) continue;
            node->resident_slot_rows = state->slot_rows;
            node->resident_slot_domain = state->slot_domain;
            break;
        }
    }
}
