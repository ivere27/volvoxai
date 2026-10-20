#ifndef VOLVOXAI_MODERN_ROW_H
#define VOLVOXAI_MODERN_ROW_H
#include "engine_core.h"
#include <stddef.h>
/* Pure storage geometry. Backend adapters supply resident narrowed spans and
 * restore the original tensor descriptors before returning to the planner. */
typedef struct {
    T* tensor;
    T saved,narrow;
    int axis,first,rows,stage;
    size_t width;
} VxModernRowSpan;
typedef struct {
    VxModernRowSpan inputs[3],outputs[2];
    int inputs_count,outputs_count,convolution,padding_left,copy,attention,rope,query;
} VxModernRowProjection;
int vx_engine_modern_row_projection(Node*,int,VxModernRowProjection*);
#endif
