#ifndef VOLVOXAI_WASM_FREESTANDING_ASSERT_H
#define VOLVOXAI_WASM_FREESTANDING_ASSERT_H

/* A freestanding module has nowhere to print an assertion and no abort worth
 * taking: a failed invariant here must already be a rejected status on the
 * engine's own error path. Compile assertions out and keep the expression
 * unevaluated, matching <assert.h> under NDEBUG. */
#define assert(expression) ((void)0)

#endif
