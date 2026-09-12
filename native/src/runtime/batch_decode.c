#include "batch_decode.h"

VxBatchDecodeStatus vx_batch_decode_positions(const VxContinuousStepWork* works,
                                              int count, int lanes,
                                              int* positions, int* live_out) {
    VxBatchDecodeStatus status = VX_BATCH_DECODE_OK;
    int live = 0;
    if (!positions || lanes < 1 ||
        count < 0 || (count > 0 && !works)) {
        return VX_BATCH_DECODE_INVALID_ARGUMENT;
    }
    for (int lane = 0; lane < lanes; lane++) positions[lane] = VX_DECODE_ROW_PARKED;
    if (live_out) *live_out = 0;
    for (int index = 0; index < count && status == VX_BATCH_DECODE_OK; index++) {
        const VxContinuousStepWork* work = &works[index];
        /* A prefill round carries one request and the whole prompt; it has no
         * single row, so it is not this conversion's business. Saying so here
         * rather than letting `position` through keeps a prompt from being
         * decoded as if it were one token at its first position. */
        if (work->phase != VX_CONTINUOUS_PHASE_DECODE || work->tokens != 1 ||
            work->position < 0) status = VX_BATCH_DECODE_INVALID_ARGUMENT;
        else if (work->slot < 0 || work->slot >= lanes)
            status = VX_BATCH_DECODE_SLOT_OUT_OF_RANGE;
        else if (positions[work->slot] != VX_DECODE_ROW_PARKED)
            status = VX_BATCH_DECODE_DUPLICATE_SLOT;
        else {
            positions[work->slot] = work->position;
            live++;
        }
    }
    if (status == VX_BATCH_DECODE_OK && live == 0) {
        status = VX_BATCH_DECODE_INVALID_ARGUMENT;
    }
    if (status != VX_BATCH_DECODE_OK) {
        /*
         * A refused round leaves every lane parked rather than half-filled.
         *
         * The refusal is the caller's to act on, but a caller that does not
         * gets an array in which nothing advances -- and `vx_decode_row_set_init`
         * refuses an all-parked set, so the step fails closed. The alternative
         * is an array that decodes whichever lanes happened to be reached
         * before the bad one, which is a partial round nobody asked for.
         */
        for (int lane = 0; lane < lanes; lane++) {
            positions[lane] = VX_DECODE_ROW_PARKED;
        }
        return status;
    }
    if (live_out) *live_out = live;
    return VX_BATCH_DECODE_OK;
}
