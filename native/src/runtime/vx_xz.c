#include "vx_xz.h"

#include "vx_thread.h"
#include "xz.h"

void vx_xz_init(void) {
    static VxOnce crc_once = VX_ONCE_INITIALIZER;
    vx_once(&crc_once, xz_crc32_init);
}
