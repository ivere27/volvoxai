#include "ngram_ids.h"
#include <stddef.h>
#include <limits.h>

int vx_ngram_ids_i32(const int32_t* tokens,const int32_t* multipliers,
        const int32_t* sizes,const int32_t* offsets,int32_t* output,
        uint32_t batch,uint32_t sequence,uint32_t ngram_size,uint32_t heads_per_ngram,
        int32_t eos,int32_t pad,int32_t vocabulary) {
    if(!tokens || !multipliers || !sizes || !offsets || !output || !batch || !sequence ||
        ngram_size<2 || !heads_per_ngram || vocabulary<=0 || eos< -1 || eos>=vocabulary || pad<0 || pad>=vocabulary ||
        (uint64_t)(ngram_size-1)*heads_per_ngram>INT32_MAX) return 0;
    uint32_t heads=(ngram_size-1)*heads_per_ngram;
    uint64_t rows=(uint64_t)batch*sequence;
    if(rows>SIZE_MAX/sizeof(int32_t)/heads) return 0;
#if SIZE_MAX <= UINT32_MAX
    if(ngram_size>SIZE_MAX/(2u*sizeof(int32_t))) return 0;
#endif
    for(uint32_t i=0;i<ngram_size;++i) {
        uint64_t multiplier=vx_ngram_multiplier(multipliers,i);
        if(!multiplier || multiplier>INT64_MAX) return 0;
    }
    for(uint32_t h=0;h<heads;++h)
        if(sizes[h]<=0 || offsets[h]<0 || (uint64_t)(uint32_t)offsets[h]+(uint32_t)sizes[h]-1u>INT32_MAX) return 0;
    for(uint64_t i=0;i<rows;++i) if(tokens[i]<0 || tokens[i]>=vocabulary) return 0;
    for(uint32_t b=0;b<batch;++b) for(uint32_t t=0;t<sequence;++t) for(uint32_t h=0;h<heads;++h)
        output[((uint64_t)b*sequence+t)*heads+h]=vx_ngram_id_at(tokens,multipliers,sizes,offsets,sequence,heads_per_ngram,eos,pad,b,t,h);
    return 1;
}
