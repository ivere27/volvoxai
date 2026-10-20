#ifndef VOLVOXAI_PACKED_LINEAR_H
#define VOLVOXAI_PACKED_LINEAR_H

#include <stdint.h>

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
