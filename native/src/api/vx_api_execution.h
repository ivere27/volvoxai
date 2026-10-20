/* Private common execution projection. Public contracts are generated. */
#ifndef VOLVOXAI_API_EXECUTION_H
#define VOLVOXAI_API_EXECUTION_H

#include "vx_api_convert.h"
#include "public_api_internal.h"
#include "cJSON.h"

int vx_api_execution_contract_from_json(const SynurangLiteAllocator*,
    VolvoxaiV1ModelExecutionContract**, const cJSON*);
cJSON* vx_api_execution_contract_json(const VolvoxaiV1ModelExecutionContract*);

typedef struct {
    int64_t context_id;
    int kind;
    VxTensorBinding* inputs;
    size_t input_count;
    VxGraphExecutionOptions graph;
    VxGenerationOptions generation;
} VxApiExecutionInput;

VxStatus vx_api_parse_execute(VxApiScratch*, const VolvoxaiV1ExecuteRequest*,
    VxApiExecutionInput*, VxReport*);
VxStatus vx_api_parse_generate(VxApiScratch*, const VolvoxaiV1GenerateRequest*,
    VxApiExecutionInput*, VxReport*);
/* Does not consume result. On failure the caller must discard every temporary
 * buffer capability before freeing the generated packet. */
int vx_api_execution_packet(VxApiRegistry*, VxResult*, VxExecutionContext*,
    const VxReport*, const VxApiHandleLineage*, int owned, int kind,
    VolvoxaiV1ExecutionPacket*);
void vx_api_execution_packet_discard(VxApiRegistry*, VolvoxaiV1ExecutionPacket*);

#if VOLVOXAI_ENABLE_TRAINING
/* Cancel an attached debugger and drain its device fence without taking its
 * packet. Returns zero only when the observer could not be allocated. */
int vx_api_cancel_attached_debug(VxApiRegistry*, VxExecutionContext*);
#endif

#endif
