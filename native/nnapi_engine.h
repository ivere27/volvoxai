#ifndef NNAPI_ENGINE_H
#define NNAPI_ENGINE_H

int nnapi_init(void);
void nnapi_cleanup(void);
void nnapi_matmul(const float* in, const float* w, const float* b, float* out, int seq, int d_in, int d_out);
void nnapi_free_weight_cache(void);

#endif
