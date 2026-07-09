#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include "tensor.h"

// Load a single tensor from a safetensors file by name.
// In a full implementation, this parses the JSON header.
Tensor* load_safetensor(const char* filepath, const char* tensor_name);

#endif
