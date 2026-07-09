#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>
#include <stddef.h>

typedef enum {
    DTYPE_F32,
    DTYPE_F16,
    DTYPE_INT8,
    DTYPE_INT32
} DType;

typedef struct {
    char name[64];
    int shape[4];
    int ndim;
    DType dtype;
    void* data;
    size_t size_bytes;
} Tensor;

#endif
