#ifndef VOLVOXAI_WASM_FREESTANDING_ERRNO_H
#define VOLVOXAI_WASM_FREESTANDING_ERRNO_H

/* A freestanding module makes no system calls, so nothing here ever sets
 * errno. It exists because callers that also compile for a hosted target
 * inspect it after a libc call; on wasm32 those calls are unreachable and the
 * value stays zero. */

#ifndef VOLVOXAI_WASM_FREESTANDING_SCOPE
#define VOLVOXAI_WASM_FREESTANDING_SCOPE extern
#endif

VOLVOXAI_WASM_FREESTANDING_SCOPE int volvoxai_wasm_errno;
#define errno volvoxai_wasm_errno

#define EDOM 33
#define ERANGE 34
#define EINVAL 22
#define ENOMEM 12
#define EBUSY 16
#define ETIMEDOUT 110

#endif
