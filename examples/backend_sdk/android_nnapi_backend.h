#ifndef VOLVOXAI_EXAMPLE_ANDROID_NNAPI_BACKEND_H
#define VOLVOXAI_EXAMPLE_ANDROID_NNAPI_BACKEND_H

#include "volvoxai_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Registers the Android NNAPI Add provider in one runtime. On non-Android
 * hosts, or Android systems without an NNAPI device, registration returns
 * VX_STATUS_BACKEND_UNAVAILABLE without installing a descriptor. */
VxStatus volvoxai_example_nnapi_backend_register(VxRuntime* runtime,
                                                 VxReport* report);

#if defined(VOLVOXAI_PUBLIC_API_TESTING)
int volvoxai_example_nnapi_backend_test_oversized_descriptors(void);
#endif

#ifdef __cplusplus
}
#endif

#endif /* VOLVOXAI_EXAMPLE_ANDROID_NNAPI_BACKEND_H */
