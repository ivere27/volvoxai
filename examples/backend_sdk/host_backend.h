#ifndef VOLVOXAI_EXAMPLE_HOST_BACKEND_H
#define VOLVOXAI_EXAMPLE_HOST_BACKEND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int volvoxai_example_host_backend_register(void);
uint64_t volvoxai_example_host_backend_handled_nodes(void);

#ifdef __cplusplus
}
#endif

#endif
