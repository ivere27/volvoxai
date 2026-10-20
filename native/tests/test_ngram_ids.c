#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "../src/kernels/ngram_ids.inc"
static int failures;
#define CHECK(x) do {if(!(x)){fprintf(stderr,"line %d: %s\n",__LINE__,#x);++failures;}} while(0)
int main(void) {
    int32_t tokens[]={2,3,9,4,5,6,9,1,2,9,3,4};
    uint64_t multipliers64[]={UINT64_C(0x7fffffffffffffff),UINT64_C(0x100000001),UINT64_C(1234567890123)};
    int32_t multipliers[6];
    for(int i=0;i<3;++i){multipliers[2*i]=(int32_t)(uint32_t)multipliers64[i];multipliers[2*i+1]=(int32_t)(uint32_t)(multipliers64[i]>>32);}
    int32_t sizes[]={17,19,23,29},offsets[]={0,17,36,59},output[48];
    const int32_t expected[]={16,30,52,87,6,22,48,82,11,24,45,81,14,28,54,60,4,29,45,73,11,24,53,84,15,17,41,79,7,28,56,71,15,29,56,83,13,31,50,84,5,26,58,73,13,17,57,86};
    CHECK(vx_ngram_ids_i32(tokens,multipliers,sizes,offsets,output,2,6,3,2,9,9,10));
    CHECK(memcmp(output,expected,sizeof(expected))==0);
    /* Engram uses start padding without resetting on occurrences of that ID.
     * Values are independently evaluated integer XOR/modulo expressions. */
    {
        int32_t words[]={1,0,3,0,5,0}, ids[]={2,1,2,4};
        int32_t moduli[]={17,23}, starts[]={0,17}, actual[8];
        const int32_t reference[]={4,31,7,30,1,28,2,24};
        CHECK(vx_ngram_ids_i32(ids,words,moduli,starts,actual,1,4,3,1,-1,2,10));
        CHECK(memcmp(actual,reference,sizeof(reference))==0);
        CHECK(!vx_ngram_ids_i32(ids,words,moduli,starts,actual,1,4,3,1,-1,-1,10));
    }
    /* Validate before publication, including banks and token ranges. */
    for(int i=0;i<48;++i)output[i]=-777;
    tokens[11]=10;
    CHECK(!vx_ngram_ids_i32(tokens,multipliers,sizes,offsets,output,2,6,3,2,9,9,10));
    for(int i=0;i<48;++i)CHECK(output[i]==-777);
    tokens[11]=4;sizes[2]=0;
    CHECK(!vx_ngram_ids_i32(tokens,multipliers,sizes,offsets,output,2,6,3,2,9,9,10));
    sizes[2]=23;offsets[2]=INT32_MAX;
    CHECK(!vx_ngram_ids_i32(tokens,multipliers,sizes,offsets,output,2,6,3,2,9,9,10));
    offsets[2]=36;multipliers[1]=(int32_t)0x80000000u;
    CHECK(!vx_ngram_ids_i32(tokens,multipliers,sizes,offsets,output,2,6,3,2,9,9,10));
    if(!failures)puts("NGramIds exact int64/EOS oracle PASS");
    return failures?1:0;
}
