#ifndef VOLVOXAI_EXAMPLE_HOST_BACKEND_H
#define VOLVOXAI_EXAMPLE_HOST_BACKEND_H

#include "volvoxai_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Registers an independent provider instance in `runtime`. The runtime copies
 * the descriptor and owns the provider instance through final release. */
VxStatus volvoxai_example_host_backend_register(VxRuntime* runtime,
                                                VxReport* report);

#if defined(VOLVOXAI_PUBLIC_API_TESTING)
int volvoxai_example_host_backend_test_oversized_descriptors(void);
#endif

#ifdef __cplusplus
}
#endif

#endif /* VOLVOXAI_EXAMPLE_HOST_BACKEND_H */
