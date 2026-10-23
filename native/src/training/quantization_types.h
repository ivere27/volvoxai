/* Private authoring implementation vocabulary; the public API is protobuf. */
#ifndef VOLVOXAI_QUANTIZATION_TYPES_H
#define VOLVOXAI_QUANTIZATION_TYPES_H
#include <stdint.h>
typedef int32_t VxPTQMode;
enum { VX_PTQ_MODE_W8A8 = 0, VX_PTQ_MODE_W4A8 = 1,
       VX_PTQ_MODE_W4A16 = 2, VX_PTQ_MODE_W8A16 = 3,
       VX_PTQ_MODE_W2A8 = 4, VX_PTQ_MODE_W3A8 = 5 };
static inline uint32_t vx_ptq_weight_bits(VxPTQMode mode) {
    return mode == VX_PTQ_MODE_W2A8 ? 2u : mode == VX_PTQ_MODE_W3A8 ? 3u :
        (mode == VX_PTQ_MODE_W4A8 || mode == VX_PTQ_MODE_W4A16 ? 4u : 8u);
}
typedef int32_t VxPTQLayerKind;
enum { VX_PTQ_LAYER_QLINEAR = 0, VX_PTQ_LAYER_QCONV2D = 1,
       VX_PTQ_LAYER_PACKED_LINEAR = 2 };
#endif
