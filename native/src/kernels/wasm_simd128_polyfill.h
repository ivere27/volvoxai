#ifndef WASM_SIMD128_H
#define WASM_SIMD128_H

#include <string.h>

// Polyfill for compiling WASM SIMD kernels natively using GCC Vector Extensions.
// This allows kernels.c to compile unmodified on x86/ARM natively.
typedef float v128_t __attribute__((vector_size(16)));

static inline v128_t wasm_v128_load(const void *p) {
    v128_t r;
    memcpy(&r, p, 16);
    return r;
}

static inline void wasm_v128_store(void *p, v128_t v) {
    memcpy(p, &v, 16);
}

static inline v128_t wasm_f32x4_splat(float x) {
    return (v128_t){x, x, x, x};
}

static inline v128_t wasm_f32x4_make(float x, float y, float z, float w) {
    return (v128_t){x, y, z, w};
}

static inline v128_t wasm_f32x4_add(v128_t a, v128_t b) { return a + b; }
static inline v128_t wasm_f32x4_sub(v128_t a, v128_t b) { return a - b; }
static inline v128_t wasm_f32x4_mul(v128_t a, v128_t b) { return a * b; }
static inline v128_t wasm_f32x4_div(v128_t a, v128_t b) { return a / b; }

static inline v128_t wasm_f32x4_max(v128_t a, v128_t b) {
    return (v128_t){
        a[0] > b[0] ? a[0] : b[0],
        a[1] > b[1] ? a[1] : b[1],
        a[2] > b[2] ? a[2] : b[2],
        a[3] > b[3] ? a[3] : b[3]
    };
}

static inline v128_t wasm_f32x4_min(v128_t a, v128_t b) {
    return (v128_t){
        a[0] < b[0] ? a[0] : b[0],
        a[1] < b[1] ? a[1] : b[1],
        a[2] < b[2] ? a[2] : b[2],
        a[3] < b[3] ? a[3] : b[3]
    };
}

#endif
