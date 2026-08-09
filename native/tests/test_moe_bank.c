/* Partially resident MoE/LoRA expert banks.
 *
 * The staged expert weight holds only the slots a context materialized, while
 * route indices stay in the model's global slot space. The banked kernel maps
 * one to the other and refuses a route to an absent slot. */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "inference_kernels.h"
#include "training_kernels.h"

#define CHECK(condition)                                                       \
    do {                                                                       \
        if (!(condition)) {                                                    \
            fprintf(stderr, "check failed: %s (%s:%d)\n", #condition,          \
                    __FILE__, __LINE__);                                       \
            return 1;                                                          \
        }                                                                      \
    } while (0)

#define EXPERTS 4u
#define D_IN 2u
#define D_OUT 1u

/* Expert k maps every input feature to the constant k + 1. */
static void fill_bank(float *bank, uint32_t experts) {
    for (uint32_t expert = 0; expert < experts; expert++) {
        for (uint32_t feature = 0; feature < D_IN; feature++) {
            bank[expert * D_IN * D_OUT + feature * D_OUT] = (float)expert + 1.0f;
        }
    }
}

int main(void) {
    float full_bank[EXPERTS * D_IN * D_OUT];
    const float input[D_IN] = {1.0f, 0.0f};
    const float route_weights[1] = {1.0f};
    float route_indices[1];
    float output[D_OUT];
    uint32_t slot_rows[EXPERTS];
    float staged[D_IN * D_OUT];

    fill_bank(full_bank, EXPERTS);

    /* A NULL slot table must reproduce the fully resident kernel exactly. */
    for (uint32_t expert = 0; expert < EXPERTS; expert++) {
        route_indices[0] = (float)expert;
        output[0] = 0.0f;
        CHECK(moe_linear_f32(input, full_bank, NULL, route_indices, route_weights,
                             output, 1, D_IN, D_OUT, EXPERTS, 1) == 1);
        CHECK(output[0] == (float)expert + 1.0f);

        output[0] = 0.0f;
        CHECK(vx_moe_linear_banked_f32(input, full_bank, NULL, route_indices,
                                       route_weights, output, 1, D_IN, D_OUT,
                                       EXPERTS, 1, NULL, 0) == 1);
        CHECK(output[0] == (float)expert + 1.0f);
    }

    /* Stage only slot 2 and route to it by its global id. */
    memcpy(staged, &full_bank[2u * D_IN * D_OUT], sizeof staged);
    for (uint32_t slot = 0; slot < EXPERTS; slot++) slot_rows[slot] = VX_MOE_SLOT_ABSENT;
    slot_rows[2] = 0u;

    route_indices[0] = 2.0f;
    output[0] = 0.0f;
    CHECK(vx_moe_linear_banked_f32(input, staged, NULL, route_indices,
                                   route_weights, output, 1, D_IN, D_OUT, 1, 1,
                                   slot_rows, EXPERTS) == 1);
    CHECK(output[0] == 3.0f);

    /* Every non-resident slot must be refused, not silently remapped. */
    for (uint32_t slot = 0; slot < EXPERTS; slot++) {
        if (slot == 2u) continue;
        route_indices[0] = (float)slot;
        CHECK(vx_moe_linear_banked_f32(input, staged, NULL, route_indices,
                                       route_weights, output, 1, D_IN, D_OUT, 1,
                                       1, slot_rows, EXPERTS) == 0);
    }

    /* A slot id outside the declared domain is refused too. */
    route_indices[0] = (float)EXPERTS;
    CHECK(vx_moe_linear_banked_f32(input, staged, NULL, route_indices,
                                   route_weights, output, 1, D_IN, D_OUT, 1, 1,
                                   slot_rows, EXPERTS) == 0);

    /* A domain smaller than the staged rows is an inconsistent request. */
    CHECK(vx_moe_linear_banked_f32(input, full_bank, NULL, route_indices,
                                   route_weights, output, 1, D_IN, D_OUT,
                                   EXPERTS, 1, slot_rows, 2) == 0);

    /* Two resident slots keep their staged order under top-k mixing. */
    {
        float pair[2u * D_IN * D_OUT];
        float pair_indices[2] = {1.0f, 3.0f};
        float pair_gates[2] = {0.25f, 0.75f};
        memcpy(&pair[0], &full_bank[1u * D_IN * D_OUT], sizeof(float) * D_IN * D_OUT);
        memcpy(&pair[D_IN * D_OUT], &full_bank[3u * D_IN * D_OUT],
               sizeof(float) * D_IN * D_OUT);
        for (uint32_t slot = 0; slot < EXPERTS; slot++) slot_rows[slot] = VX_MOE_SLOT_ABSENT;
        slot_rows[1] = 0u;
        slot_rows[3] = 1u;
        output[0] = 0.0f;
        CHECK(vx_moe_linear_banked_f32(input, pair, NULL, pair_indices, pair_gates,
                                       output, 1, D_IN, D_OUT, 2, 2, slot_rows,
                                       EXPERTS) == 1);
        /* 0.25 * 2 + 0.75 * 4 */
        CHECK(output[0] == 3.5f);
    }

    /* Training forward/backward map global routes the same way, so gradients
     * land on the staged rows rather than on whichever slot shares the index. */
    {
        float staged_pair[2u * D_IN * D_OUT];
        float dweight[2u * D_IN * D_OUT];
        float dinput[D_IN];
        float droute[1];
        float train_out[D_OUT];
        const float dy[D_OUT] = {1.0f};
        route_indices[0] = 3.0f;
        memcpy(&staged_pair[0], &full_bank[1u * D_IN * D_OUT],
               sizeof(float) * D_IN * D_OUT);
        memcpy(&staged_pair[D_IN * D_OUT], &full_bank[3u * D_IN * D_OUT],
               sizeof(float) * D_IN * D_OUT);
        for (uint32_t slot = 0; slot < EXPERTS; slot++) slot_rows[slot] = VX_MOE_SLOT_ABSENT;
        slot_rows[1] = 0u;
        slot_rows[3] = 1u;

        CHECK(volvoxai_training_moe_linear_banked_f32(
            input, staged_pair, NULL, route_indices, route_weights, train_out,
            1, D_IN, D_OUT, 2, 1, slot_rows, EXPERTS) == 1);
        CHECK(train_out[0] == 4.0f);

        memset(dweight, 0, sizeof dweight);
        memset(dinput, 0, sizeof dinput);
        memset(droute, 0, sizeof droute);
        CHECK(volvoxai_training_moe_linear_backward_banked_f32(
            input, staged_pair, NULL, route_indices, route_weights, dy,
            dinput, dweight, NULL, droute, 1, D_IN, D_OUT, 2, 1,
            slot_rows, EXPERTS) == 1);
        /* Row 1 is global slot 3; row 0 (slot 1) must stay untouched. */
        CHECK(dweight[0] == 0.0f && dweight[1] == 0.0f);
        CHECK(dweight[D_IN * D_OUT] == 1.0f);
        CHECK(droute[0] == 4.0f);

        /* A non-resident route is refused in training too. */
        route_indices[0] = 2.0f;
        CHECK(volvoxai_training_moe_linear_backward_banked_f32(
            input, staged_pair, NULL, route_indices, route_weights, dy,
            dinput, dweight, NULL, droute, 1, D_IN, D_OUT, 2, 1,
            slot_rows, EXPERTS) == 0);
    }

    printf("test_moe_bank ok\n");
    return 0;
}
