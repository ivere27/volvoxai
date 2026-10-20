#ifndef VOLVOXAI_NGRAM_IDS_H
#define VOLVOXAI_NGRAM_IDS_H
#include <stdint.h>
#ifndef VX_NGRAM_INLINE
#define VX_NGRAM_INLINE static inline
#endif

/* Multipliers use [low32,high32] I32 storage, interpreted as exact unsigned
 * words. Products/XOR have torch.int64 wrap semantics; the final remainder is
 * Euclidean even when the wrapped signed value is negative. eos=-1 disables
 * token resets; pad fills lookbacks outside the sequence or past a reset. */
VX_NGRAM_INLINE uint64_t vx_ngram_multiplier(const int32_t* words, uint32_t index) {
    return (uint64_t)(uint32_t)words[(uint64_t)index*2] |
        ((uint64_t)(uint32_t)words[(uint64_t)index*2+1] << 32);
}
VX_NGRAM_INLINE uint32_t vx_ngram_signed_remainder(uint64_t bits,uint32_t divisor) {
    uint64_t remainder=bits%divisor;
    if(bits>>63) {
        /* 2^64 cannot be represented, so obtain its residue via UINT64_MAX. */
        uint64_t modulus=((UINT64_MAX%divisor)+1u)%divisor;
        remainder=(remainder+divisor-modulus)%divisor;
    }
    return (uint32_t)remainder;
}
VX_NGRAM_INLINE int32_t vx_ngram_id_at(const int32_t* tokens,const int32_t* multipliers,
        const int32_t* sizes,const int32_t* offsets,uint32_t sequence,
        uint32_t heads_per_ngram,int32_t eos,int32_t pad,uint32_t batch,uint32_t position,uint32_t head) {
    uint32_t length=head/heads_per_ngram+2u;
    const int32_t* row=tokens+(uint64_t)batch*sequence;
    uint64_t mixed=(uint64_t)(uint32_t)row[position]*vx_ngram_multiplier(multipliers,0);
    int boundary=0;
    for(uint32_t shift=1;shift<length;++shift) {
        int32_t token=pad;
        if(shift<=position && !boundary) {
            token=row[position-shift];
            if(eos>=0 && token==eos) boundary=1;
        }
        mixed^=(uint64_t)(uint32_t)token*vx_ngram_multiplier(multipliers,shift);
    }
    return (int32_t)(vx_ngram_signed_remainder(mixed,(uint32_t)sizes[head])+(uint32_t)offsets[head]);
}
int vx_ngram_ids_i32(const int32_t* tokens,const int32_t* multipliers,
    const int32_t* sizes,const int32_t* offsets,int32_t* output,
    uint32_t batch,uint32_t sequence,uint32_t ngram_size,uint32_t heads_per_ngram,
    int32_t eos,int32_t pad,int32_t vocabulary);
#endif
