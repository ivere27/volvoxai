#ifndef VOLVOXAI_PACKED_LINEAR_H
#define VOLVOXAI_PACKED_LINEAR_H

#include <stdint.h>

/* Private CPU/WASM tuning, independent of the public storage/arithmetic
 * contract. Build the kernel and its memory proof with the same settings. */
#ifndef VX_PACKED_LINEAR_TILE_ROWS
#if defined(__wasm_simd128__)
#define VX_PACKED_LINEAR_TILE_ROWS 32
#elif defined(__SSE2__)
#define VX_PACKED_LINEAR_TILE_ROWS 64
#else
#define VX_PACKED_LINEAR_TILE_ROWS 4
#endif
#endif
#ifndef VX_PACKED_LINEAR_TILE2
#define VX_PACKED_LINEAR_TILE2 1
#endif
#ifndef VX_PACKED_LINEAR_TILE_ALL_BITS
#define VX_PACKED_LINEAR_TILE_ALL_BITS 0
#endif
#if VX_PACKED_LINEAR_TILE_ROWS != 4 && VX_PACKED_LINEAR_TILE_ROWS != 8 && \
    VX_PACKED_LINEAR_TILE_ROWS != 16 && VX_PACKED_LINEAR_TILE_ROWS != 32 && \
    VX_PACKED_LINEAR_TILE_ROWS != 64
#error "VX_PACKED_LINEAR_TILE_ROWS must be 4, 8, 16, 32 or 64"
#endif
#if VX_PACKED_LINEAR_TILE2 != 0 && VX_PACKED_LINEAR_TILE2 != 1
#error "VX_PACKED_LINEAR_TILE2 must be 0 or 1"
#endif
#if VX_PACKED_LINEAR_TILE_ALL_BITS != 0 && VX_PACKED_LINEAR_TILE_ALL_BITS != 1
#error "VX_PACKED_LINEAR_TILE_ALL_BITS must be 0 or 1"
#endif

static inline uint32_t vx_packed_linear_tile_rows(uint64_t complete_rows) {
    uint32_t rows = complete_rows < VX_PACKED_LINEAR_TILE_ROWS
        ? (uint32_t)complete_rows : VX_PACKED_LINEAR_TILE_ROWS;
    if (rows >= 4u) return rows & ~3u;
    return VX_PACKED_LINEAR_TILE2 && rows >= 2u ? 2u : 0u;
}

static inline int vx_packed_linear_panel_eligible(uint64_t k,uint64_t n,uint32_t bits) {
    /* A partial output panel retains the existing SIMD-dot tail route. The
     * experimental decoded panel's scalar column tail is slower for small N. */
    return k>=32u && n>=4u && n%4u==0u && (VX_PACKED_LINEAR_TILE_ALL_BITS || bits==3u || bits==4u);
}

/* A bounded input-row tile holds exact I8 values in adjacent I16 pairs. One
 * group of four weight columns is decoded at a time; no full decoded bank is
 * retained. The ordinary F32/F16/direct path still needs K floats. This is a
 * conservative bound across arithmetic modes and partial-row dispatches. */
static inline uint64_t vx_packed_linear_scratch_elements(uint64_t k,uint64_t rows,uint32_t bits,uint64_t n) {
    if (!vx_packed_linear_panel_eligible(k,n,bits)) return k;
#if defined(__wasm_simd128__) || defined(__SSE2__)
    uint32_t tile = vx_packed_linear_tile_rows(rows);
    if (tile > 4u || (VX_PACKED_LINEAR_TILE2 && rows >= 2u)) {
        uint64_t pairs = k / 2u + k % 2u;
        uint64_t prepared;
        if (tile < 2u) tile = 2u;
        if (pairs > (UINT64_MAX - 3u - 512u) / tile) return UINT64_MAX;
        prepared = (pairs * tile + 3u) & ~UINT64_C(3);
        /* 256 K terms * four outputs * two bytes, in float-sized units. */
        return prepared + 512u > k ? prepared + 512u : k;
    }
#else
    (void)rows;
#endif
    return k;
}

/* Storage groups partition K independently of any backend execution tile.
 * Every admitted size is word aligned for I4 and I8. Zero belongs to the
 * separate per-channel byte format and is never a PackedLinear group. */
enum { VX_PACKED_MAX_GROUP_SIZE = 256 };
static inline int vx_packed_group_size_supported(int group_size) {
    return group_size == 32 || group_size == 64 ||
           group_size == 128 || group_size == 256;
}

/* Arithmetic boundaries for PackedLinear. These helpers do not depend on the
 * host rounding mode, native half support, or a CUDA math library. */
static inline int vx_packed_finite(float value) {
    union { float f; uint32_t u; } bits = {value};
    return (bits.u & 0x7f800000u) != 0x7f800000u;
}

static inline float vx_packed_half(float value) {
    union { float f; uint32_t u; } bits = {value};
    uint32_t sign=bits.u&0x80000000u, absolute=bits.u&0x7fffffffu;
    if(absolute>=0x7f800000u) {
        bits.u=sign|0x7f800000u|(absolute>0x7f800000u?0x00400000u:0u);
        return bits.f;
    }
    if(absolute<0x38800000u) {
        uint32_t exponent=absolute>>23u;
        if(exponent<102u) {bits.u=sign;return bits.f;}
        uint32_t shift=126u-exponent, mantissa=(absolute&0x007fffffu)|0x00800000u;
        uint32_t rounded=(mantissa+((1u<<(shift-1u))-1u)+((mantissa>>shift)&1u))>>shift;
        bits.f=(float)rounded*0x1p-24f;
        bits.u|=sign;
        return bits.f;
    }
    absolute+=0xfffu+((absolute>>13u)&1u);
    bits.u=sign|(absolute>=0x47800000u?0x7f800000u:(absolute&0x7fffe000u));
    return bits.f;
}

static inline int32_t vx_packed_round_i8(float value) {
    if(value>=127.0f)return 127;
    if(value<=-127.0f)return -127;
    int32_t integer=(int32_t)value;
    float remainder=value-(float)integer;
    if(remainder>0.5f||(remainder==0.5f&&(integer&1)))integer++;
    if(remainder<-0.5f||(remainder==-0.5f&&(integer&1)))integer--;
    return integer;
}

#endif
