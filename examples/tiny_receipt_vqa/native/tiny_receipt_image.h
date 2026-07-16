#ifndef VOLVOXAI_EXAMPLE_TINY_RECEIPT_IMAGE_H
#define VOLVOXAI_EXAMPLE_TINY_RECEIPT_IMAGE_H

#include <stddef.h>

int tiny_receipt_rgb_to_grayscale_bilinear(const unsigned char* rgb,
                                            int source_width,
                                            int source_height,
                                            float* dst,
                                            int target_width,
                                            int target_height);

int tiny_receipt_load_image_to_tensor(const char* path,
                                      float* dst,
                                      const int* shape,
                                      int ndim,
                                      char* err,
                                      size_t err_size);

#endif
