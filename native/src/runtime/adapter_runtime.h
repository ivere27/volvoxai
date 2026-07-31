#ifndef VOLVOX_ADAPTER_RUNTIME_H
#define VOLVOX_ADAPTER_RUNTIME_H

#include "volvoxai_enums.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VX_ADAPTER_LORA = 0
} VxAdapterKind;

enum {
    VX_ADAPTER_DTYPE_F32 = VX_DTYPE_F32,
    VX_ADAPTER_DTYPE_F16 = VX_DTYPE_F16
};

typedef enum {
    VX_ADAPTER_UPDATE_ASSIGN = 0,
    VX_ADAPTER_UPDATE_ADD = 1
} VxAdapterUpdateMode;

typedef struct {
    const void* data;
    size_t nbytes;
    VxDataType dtype;
    int rows;
    int cols;
} VxAdapterTensorSpec;

typedef struct {
    const char* weight_name;
    VxAdapterKind kind;
    int d_in;
    int d_out;
    int rank;
    float alpha;
    float scale;
    const char* a_name;
    const char* b_name;
    VxAdapterTensorSpec a;
    VxAdapterTensorSpec b;
} VxAdapterTargetSpec;

typedef struct {
    const char* adapter_id;
    const char* version_id;
    const VxAdapterTargetSpec* targets;
    int target_count;
    const char* manifest_json;
} VxAdapterVersionSpec;

typedef struct {
    const char* tensor_name;
    const void* data;
    size_t nbytes;
    VxDataType dtype;
    VxAdapterUpdateMode mode;
} VxAdapterTensorUpdate;

typedef struct {
    char weight_name[128];
    VxAdapterKind kind;
    int d_in;
    int d_out;
    int rank;
    float alpha;
    float scale;
} VxAdapterTargetInfo;

#ifdef __cplusplus
}
#endif

#endif
