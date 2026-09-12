#ifndef VOLVOXAI_BACKEND_CONFIG_H
#define VOLVOXAI_BACKEND_CONFIG_H

/* Native backend composition is a build-time decision.  Keep defaults
 * compatible with direct source builds while allowing release builds to
 * remove an integration and every reference to it from the binary. */
#if defined(__wasm__)
#ifndef VOLVOXAI_ENABLE_VULKAN
#define VOLVOXAI_ENABLE_VULKAN 0
#endif
#ifndef VOLVOXAI_ENABLE_OPENGL
#define VOLVOXAI_ENABLE_OPENGL 0
#endif
#else
#ifndef VOLVOXAI_ENABLE_VULKAN
#define VOLVOXAI_ENABLE_VULKAN 1
#endif
#ifndef VOLVOXAI_ENABLE_OPENGL
#define VOLVOXAI_ENABLE_OPENGL 1
#endif
#endif

/* WebGPU is reachable only from a browser module, and only through the host
 * device bridge. It is never a native backend, and the ordinary inference
 * profile is deliberately WASM-only, so the full browser build is the one
 * that asks for it. Default off; that build defines it. */
#ifndef VOLVOXAI_ENABLE_WEBGPU
#define VOLVOXAI_ENABLE_WEBGPU 0
#endif

#ifndef VOLVOXAI_ENABLE_METAL
#if defined(__APPLE__)
#define VOLVOXAI_ENABLE_METAL 1
#else
#define VOLVOXAI_ENABLE_METAL 0
#endif
#endif

/* CUDA requires generated kernels and its driver integration, so direct
 * source builds keep it disabled unless their build composition opts in. */
#ifndef VOLVOXAI_ENABLE_CUDA
#define VOLVOXAI_ENABLE_CUDA 0
#endif

#if VOLVOXAI_ENABLE_METAL && !defined(__APPLE__)
#error "VOLVOXAI_ENABLE_METAL requires an Apple target"
#endif

#endif
