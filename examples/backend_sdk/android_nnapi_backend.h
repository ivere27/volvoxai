#ifndef VOLVOXAI_EXAMPLE_ANDROID_NNAPI_BACKEND_H
#define VOLVOXAI_EXAMPLE_ANDROID_NNAPI_BACKEND_H

/*
 * Example Android NNAPI backend built entirely against VolvoxAI's public ABI.
 * This header is part of the example, not the installed VolvoxAI C API.
 */

#include "volvoxai_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Exact-shape F32 Add eligibility shared by the example callback and host
 * contract tests.  The node view is borrowed and must come from a backend
 * callback.  Returns VX_HANDLED or VX_DECLINED. */
int volvoxai_example_nnapi_add_supports(const VxNode* node);

/* Register the example as "android-nnapi-add".  Select it separately with
 * volvoxai_engine_configure_backend() before model load. */
int volvoxai_example_nnapi_backend_register(void);

#ifdef __cplusplus
}
#endif

#endif /* VOLVOXAI_EXAMPLE_ANDROID_NNAPI_BACKEND_H */
