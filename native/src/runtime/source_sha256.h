/* Private streaming SHA-256 for immutable model source integrity. */
#ifndef VX_SOURCE_SHA256_H
#define VX_SOURCE_SHA256_H
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct VxSourceSha256 {
    uint32_t state[8];
    uint64_t bytes;
    unsigned char block[64];
    size_t used;
} VxSourceSha256;

static uint32_t vx_source_sha256_rotate(uint32_t x, unsigned n) {
    return (x >> n) | (x << (32u - n));
}

static void vx_source_sha256_block(VxSourceSha256* hash, const unsigned char* input) {
    static const uint32_t k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };
    uint32_t w[64];
    uint32_t a=hash->state[0], b=hash->state[1], c=hash->state[2], d=hash->state[3];
    uint32_t e=hash->state[4], f=hash->state[5], g=hash->state[6], h=hash->state[7];
    for (unsigned i=0;i<16;i++)
        w[i]=((uint32_t)input[i*4]<<24)|((uint32_t)input[i*4+1]<<16)|
             ((uint32_t)input[i*4+2]<<8)|input[i*4+3];
    for (unsigned i=16;i<64;i++) {
        uint32_t x=w[i-15], y=w[i-2];
        w[i]=w[i-16]+(vx_source_sha256_rotate(x,7)^vx_source_sha256_rotate(x,18)^(x>>3))+
             w[i-7]+(vx_source_sha256_rotate(y,17)^vx_source_sha256_rotate(y,19)^(y>>10));
    }
    for (unsigned i=0;i<64;i++) {
        uint32_t t1=h+(vx_source_sha256_rotate(e,6)^vx_source_sha256_rotate(e,11)^vx_source_sha256_rotate(e,25))+
                    ((e&f)^((~e)&g))+k[i]+w[i];
        uint32_t t2=(vx_source_sha256_rotate(a,2)^vx_source_sha256_rotate(a,13)^vx_source_sha256_rotate(a,22))+
                    ((a&b)^(a&c)^(b&c));
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    hash->state[0]+=a; hash->state[1]+=b; hash->state[2]+=c; hash->state[3]+=d;
    hash->state[4]+=e; hash->state[5]+=f; hash->state[6]+=g; hash->state[7]+=h;
}

static void vx_source_sha256_init(VxSourceSha256* hash) {
    static const uint32_t initial[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                                    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    memset(hash,0,sizeof(*hash)); memcpy(hash->state,initial,sizeof(initial));
}

static void vx_source_sha256_update(VxSourceSha256* hash, const void* data, size_t bytes) {
    const unsigned char* cursor=(const unsigned char*)data;
    hash->bytes+=bytes;
    while (bytes) {
        size_t count=64u-hash->used;
        if (count>bytes) count=bytes;
        memcpy(hash->block+hash->used,cursor,count);
        hash->used+=count; cursor+=count; bytes-=count;
        if (hash->used==64u) { vx_source_sha256_block(hash,hash->block); hash->used=0; }
    }
}

static void vx_source_sha256_finish(VxSourceSha256* hash, unsigned char digest[32]) {
    uint64_t bits=hash->bytes*8u;
    hash->block[hash->used++]=0x80;
    if (hash->used>56u) {
        memset(hash->block+hash->used,0,64u-hash->used);
        vx_source_sha256_block(hash,hash->block); hash->used=0;
    }
    memset(hash->block+hash->used,0,56u-hash->used);
    for (unsigned i=0;i<8;i++) hash->block[63u-i]=(unsigned char)(bits>>(i*8));
    vx_source_sha256_block(hash,hash->block);
    for (unsigned i=0;i<8;i++) for (unsigned j=0;j<4;j++)
        digest[i*4+j]=(unsigned char)(hash->state[i]>>(24u-j*8u));
}
#endif
