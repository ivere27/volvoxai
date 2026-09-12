#ifndef VOLVOXAI_RUNTIME_GRAPH_BIND_DEFINITION_H
#define VOLVOXAI_RUNTIME_GRAPH_BIND_DEFINITION_H

#include "safetensors.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cJSON cJSON;

/* Private host adapter result.  Semantic failures are intentionally distinct
 * from allocation/argument failures: model loading may cache the former and
 * preserve the established compile-time unsupported boundary. */
typedef int32_t VxGraphBindDefinitionStatusV1;
enum {
    VX_GRAPH_BIND_DEFINITION_STATUS_OK = 0,
    VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_ARGUMENT = -1,
    VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH_PLAN = -2,
    VX_GRAPH_BIND_DEFINITION_STATUS_INVALID_GRAPH = -3,
    VX_GRAPH_BIND_DEFINITION_STATUS_UNSUPPORTED_PARAMETER = -4,
    VX_GRAPH_BIND_DEFINITION_STATUS_OUT_OF_MEMORY = -5,
    VX_GRAPH_BIND_DEFINITION_STATUS_INTERNAL = -6
};

typedef uint32_t VxGraphBindDefinitionErrorSectionV1;
enum {
    VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_NONE = 0,
    VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_GRAPH_PLAN = 1,
    VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_DIMENSION = 2,
    VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_TENSOR = 3,
    VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_NODE = 4,
    VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_EDGE = 5,
    VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_PARAM = 6,
    VX_GRAPH_BIND_DEFINITION_ERROR_SECTION_QUANTIZATION = 7
};

typedef struct VxGraphBindDefinitionErrorV1 {
    VxGraphBindDefinitionStatusV1 status;
    VxGraphBindDefinitionErrorSectionV1 section;
    uint32_t index;
    uint32_t subindex;
} VxGraphBindDefinitionErrorV1;

/* Encode the exact normalized definition consumed by graph_bind.h.  `files`
 * and every tensor payload remain borrowed for the duration of the call.
 * Later files take precedence for duplicate tensor names, matching native
 * package semantics.  On every failure the output pair is reset to NULL/zero.
 * A successful output is malloc-owned by the caller. */
VxGraphBindDefinitionStatusV1 vx_graph_bind_definition_encode_cjson_v1(
    const cJSON* root,
    const SafetensorsFile* files,
    size_t file_count,
    const uint8_t* graph_plan_request,
    uint32_t graph_plan_request_bytes,
    const uint8_t* graph_plan_response,
    uint32_t graph_plan_response_bytes,
    uint8_t** out_definition,
    uint32_t* out_definition_bytes,
    VxGraphBindDefinitionErrorV1* out_error);

#ifdef __cplusplus
}
#endif

#endif
