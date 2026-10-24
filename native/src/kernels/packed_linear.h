#ifndef VOLVOXAI_PACKED_LINEAR_H
#define VOLVOXAI_PACKED_LINEAR_H

#include <stdint.h>

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
