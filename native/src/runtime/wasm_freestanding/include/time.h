#ifndef VOLVOXAI_WASM_FREESTANDING_TIME_H
#define VOLVOXAI_WASM_FREESTANDING_TIME_H

/* A wasm32 module has no clock of its own. The host supplies one monotonic
 * source; everything else the engine asks a <time.h> for is unavailable here
 * and must not be reachable. Platform samples, scheduler deadlines and aging
 * share this source. Numerical kernels and shape proofs do not depend on it. */

#include <stddef.h>

#ifndef VOLVOXAI_WASM_FREESTANDING_SCOPE
#define VOLVOXAI_WASM_FREESTANDING_SCOPE extern
#endif

#define CLOCK_MONOTONIC 1
#define TIME_UTC 1

typedef long time_t;

struct timespec {
    time_t tv_sec;
    long tv_nsec;
};

VOLVOXAI_WASM_FREESTANDING_SCOPE int clock_gettime(
    int clock_id, struct timespec* out);

static inline int timespec_get(struct timespec* ts, int base) {
    if (!ts || base != TIME_UTC) return 0;
    if (clock_gettime(CLOCK_MONOTONIC, ts) != 0) return 0;
    return base;
}

static inline time_t time(time_t* tloc) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return (time_t)-1;
    if (tloc) *tloc = ts.tv_sec;
    return ts.tv_sec;
}

#endif
