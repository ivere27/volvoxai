/* Private module integration. Application operations remain generated from
 * proto/volvoxai.proto and are registered separately for every module owner. */
#ifndef VOLVOXAI_API_H
#define VOLVOXAI_API_H

#include "vx_api_handles.h"
#include <synurang/call.h>

/* Used only by the private WASM preflight transport. */
VxApiRegistry* vx_api_module_registry(SynurangInstance* instance);

/* A unary response queue is empty on entry and has capacity >= 1. Encoding
 * copies every response byte before this callback releases its message. Each
 * call finishes exactly once; domain outcomes stay in OperationReport. */
#define VX_API_UNARY(handler, request_type, response_type, codec, respond) \
    static void handler##_call(SynurangStream* stream, \
                               const request_type* request, void* registry) { \
        response_type response; \
        codec##_init(&response); \
        int result = handler(request, &response, registry); \
        SynurangStatus status = result == 0 ? respond(stream, &response) \
                                           : SYNURANG_INTERNAL; \
        codec##_free(&response); \
        if (status == SYNURANG_OK) \
            (void)synurang_stream_finish(stream); \
        else \
            (void)synurang_stream_fail_error(stream, status, 13, \
                                            "Response construction failed"); \
    }

#endif /* VOLVOXAI_API_H */
