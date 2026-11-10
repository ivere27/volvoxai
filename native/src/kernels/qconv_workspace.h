#ifndef VOLVOX_QCONV_WORKSPACE_H
#define VOLVOX_QCONV_WORKSPACE_H

#include <stddef.h>
#include <stdint.h>

/* One request-local NHWC patch matrix; packed immutable weights are charged
 * separately to the model. Keep allocation and domain admission on one cap. */
#define VX_W8A8_QCONV_IM2COL_MAX_BYTES (128u * 1024u * 1024u)

static inline int vx_qconv_im2col_geometry(uint64_t height, uint64_t width) {
    return width && (height == 1u || (height == 3u && width == 3u));
}

static inline size_t vx_qconv_im2col_allocation_bytes(uint64_t rows,
                                                    uint64_t columns) {
    if (!rows || !columns || rows > UINT32_MAX || columns > UINT32_MAX ||
        rows > VX_W8A8_QCONV_IM2COL_MAX_BYTES / columns) return 0u;
    return (size_t)(rows * columns);
}

#endif
