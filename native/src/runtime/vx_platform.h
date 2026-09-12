#ifndef VOLVOXAI_RUNTIME_VX_PLATFORM_H
#define VOLVOXAI_RUNTIME_VX_PLATFORM_H

/*
 * What the target the engine is compiled for can actually do.
 *
 * The engine runs the same graph the same way everywhere. Native paths resolve
 * through the operating system while wasm32 paths resolve through the
 * module-local VFS, so path handling needs no capability split here. A
 * freestanding module does lack a process environment and stderr; this header
 * abstracts only those two surroundings.
 */

#if defined(__wasm__)

/* A freestanding module has no stderr stream to write to. Diagnostics are
 * dropped rather than routed to a host import: they are unstructured developer
 * text, and the structured operation report is what a caller is meant to read. */
#define vx_engine_log(...) ((void)0)

/* No process environment either. A developer override that is unreachable is
 * reported as absent rather than faked. */
#define vx_engine_env(name) ((const char*)0)

#else

#include <stdio.h>
#include <stdlib.h>
#define vx_engine_log(...) ((void)fprintf(stderr, __VA_ARGS__))
#define vx_engine_env(name) getenv(name)

#endif

#endif
