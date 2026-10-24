#ifndef VOLVOXAI_RUNTIME_VX_XZ_H
#define VOLVOXAI_RUNTIME_VX_XZ_H

/* Initialize XZ's process-wide CRC table before any decoder uses it.
 * All engine consumers share the same once state, including metadata and
 * shader decoding. This is private runtime support, not a public operation. */
void vx_xz_init(void);

#endif
