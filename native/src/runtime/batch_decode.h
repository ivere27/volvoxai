#ifndef VOLVOX_RUNTIME_BATCH_DECODE_H
#define VOLVOX_RUNTIME_BATCH_DECODE_H

/*
 * The join between a scheduler round and one engine decode step.
 *
 * `vx_continuous_batch_scheduler_step` hands its `run_step` callback every lane of a round
 * together, and `volvoxai_engine_forward_incremental_rows` takes every lane of
 * a batch together. They do not fit as they stand, and the mismatch is not
 * arithmetic: the scheduler names *occupied* lanes and says nothing about the
 * rest, while the engine addresses a dense `[lanes,S,W]` activation in which
 * every row exists whether or not a request is behind it.
 *
 * So the conversion is the one thing this file does: a sparse list of works
 * becomes a dense `positions` array with `VX_DECODE_ROW_PARKED` wherever the
 * round has no request. A parked lane still occupies its row -- the operand's
 * shape says so -- and nothing is written back for it.
 *
 * Engine-free, like `continuous_batch_scheduler.c` and `paged_kv.c`. It converts a
 * description of a round; running one stays with the caller, which is what
 * keeps the scheduler from acquiring an opinion about what a token is.
 */

#include "continuous_batch_scheduler.h"
#include "decode_row_set.h"

typedef enum {
    VX_BATCH_DECODE_OK = 0,
    VX_BATCH_DECODE_INVALID_ARGUMENT = -1,
    /* Two works claiming one lane. The scheduler does not produce this, and a
     * caller that filtered or reordered a round might; letting it through
     * would decode one request twice and another not at all. */
    VX_BATCH_DECODE_DUPLICATE_SLOT = -2,
    /* A slot outside the batch the graph declares. */
    VX_BATCH_DECODE_SLOT_OUT_OF_RANGE = -3
} VxBatchDecodeStatus;

/*
 * The dense positions for one round.
 *
 * `positions` holds `lanes` entries and is written in full: every lane the
 * round does not mention is parked, so a caller cannot accidentally carry a
 * previous round's position into a slot that has since been released.
 *
 * `works` are the round's lanes as `run_step` received them; only the decode
 * phase is accepted, because a prefill is a whole-sequence forward and has no
 * single row to name. `live_out` is optional and reports how many lanes
 * actually advance -- the numerator of the batching win, whose denominator is
 * the one call this round becomes.
 */
VxBatchDecodeStatus vx_batch_decode_positions(const VxContinuousStepWork* works,
                                              int count, int lanes,
                                              int* positions, int* live_out);

#endif /* VOLVOX_RUNTIME_BATCH_DECODE_H */
