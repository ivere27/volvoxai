#ifndef VOLVOXAI_IMAGE_IO_H
#define VOLVOXAI_IMAGE_IO_H

#include <stddef.h>

enum {
    VOLVOX_IMAGE_ZERO_ONE = 0,
    VOLVOX_IMAGE_MINUS_ONE_ONE = 1,
    VOLVOX_IMAGE_RAW_255 = 2
};

int volvox_load_image_to_tensor(const char* path, float* dst, const int* shape, int ndim,
                                int normalize, char* err, size_t err_size);

#endif
