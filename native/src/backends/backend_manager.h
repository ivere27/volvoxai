#ifndef VOLVOXAI_BACKEND_MANAGER_H
#define VOLVOXAI_BACKEND_MANAGER_H

#include "engine_core.h"

/*
 * The portable implementation ships under the provider name its target
 * deploys as: native callers select `cpu`, browser callers select the same
 * code compiled into their `wasm` provider.
 */

int vx_backend_manager_activate(VxBackendKind backend);
void vx_backend_manager_deactivate(void);
VxBackendKind vx_backend_manager_current(void);

/*
 * The backend name space, in both directions.
 *
 * Selection and reporting share one vocabulary on purpose: a caller that
 * reads a backend out of a report and hands it back to a policy has to get
 * the backend it just saw. Every name the engine answers to is a row in one
 * table, so adding a backend is one row rather than an edit in each of the
 * places that used to spell the list out.
 *
 * A name maps whether or not this build can run it -- availability is what
 * `activate` answers, and keeping the two questions apart is what lets a
 * policy name `webgpu` on a target that then declines it.
 */
const char* vx_backend_manager_name_of(VxBackendKind backend);
const char* vx_backend_manager_name(void);
int vx_backend_manager_by_name(const char* name, VxBackendKind* backend);

#endif
