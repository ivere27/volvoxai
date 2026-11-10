#ifndef VOLVOXAI_LSTM_WORKSPACE_H
#define VOLVOXAI_LSTM_WORKSPACE_H
#include <limits.h>
#include <stddef.h>
#include <stdint.h>

/* XW[T,B,4H], HR[B,4H], hidden[B,H], cell[B,H]; directions reuse it. */
static inline size_t vx_lstm_workspace_bytes(uint64_t time, uint64_t batch,
        uint64_t hidden) {
    if (!time || !batch || !hidden || time > INT_MAX || batch > INT_MAX ||
        time > (uint64_t)INT_MAX / batch || hidden > INT_MAX / 8u) return 0u;
    uint64_t state = batch * hidden;
    uint64_t factors = time * 4u + 6u;
    if (state > (uint64_t)SIZE_MAX / sizeof(float) / factors) return 0u;
    return (size_t)(state * factors) * sizeof(float);
}
#endif
