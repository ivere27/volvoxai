#ifndef MATHCOMPAT_H
#define MATHCOMPAT_H
// Native builds use libm; the freestanding wasm32 build (-nostdlib) has no <math.h>,
// so declare the few functions we use — clang lowers sqrtf/floorf/fabsf to wasm
// instructions and expf/logf/powf resolve to host imports (see WasmEngine env).
#ifdef __wasm__
extern float expf(float);
extern float logf(float);
extern float powf(float, float);
extern float sqrtf(float);
extern float floorf(float);
extern float fabsf(float);
extern float tanhf(float);
#else
#include <math.h>
#endif
#endif
