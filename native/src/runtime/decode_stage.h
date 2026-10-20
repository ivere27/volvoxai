#ifndef VOLVOXAI_DECODE_STAGE_H
#define VOLVOXAI_DECODE_STAGE_H

#include "generated/backend_vocabulary.h"

#define VX_CUDA_DECODE_NUMERIC_STAGE_COUNT 4u
#define VX_DECODE_NUMERIC_STAGE_COUNT 12u

static inline unsigned vx_decode_numeric_stage_count(VxBackendKind backend) {
    if (backend == VX_BACKEND_KIND_CUDA) return VX_CUDA_DECODE_NUMERIC_STAGE_COUNT;
    if (backend == VX_BACKEND_KIND_VULKAN || backend == VX_BACKEND_KIND_OPENGL ||
        backend == VX_BACKEND_KIND_METAL) return VX_DECODE_NUMERIC_STAGE_COUNT;
    return 0u;
}

#endif
