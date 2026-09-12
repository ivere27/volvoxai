#ifndef VOLVOXAI_EXAMPLE_HOST_BACKEND_H
#define VOLVOXAI_EXAMPLE_HOST_BACKEND_H

#include "volvoxai_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Adds this provider to process composition. Call once before generated
 * CreateRuntime; every later Runtime owns an independent provider instance. */
VxStatus volvoxai_example_host_backend_register(VxReport* report);


#ifdef __cplusplus
}
#endif

#endif /* VOLVOXAI_EXAMPLE_HOST_BACKEND_H */
