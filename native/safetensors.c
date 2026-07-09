#include "safetensors.h"
#include "cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Tensor* load_safetensor(const char* filepath, const char* tensor_name) {
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open safetensors file: %s\n", filepath);
        return NULL;
    }

    uint64_t header_size;
    if (fread(&header_size, sizeof(uint64_t), 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    // Read the JSON header
    char* json_str = (char*)malloc(header_size + 1);
    if (fread(json_str, 1, header_size, f) != header_size) {
        free(json_str);
        fclose(f);
        return NULL;
    }
    json_str[header_size] = '\0';

    cJSON* root = cJSON_Parse(json_str);
    free(json_str);
    
    if (!root) {
        fprintf(stderr, "Failed to parse safetensors JSON header\n");
        fclose(f);
        return NULL;
    }

    cJSON* tensor_node = cJSON_GetObjectItemCaseSensitive(root, tensor_name);
    if (!tensor_node) {
        fprintf(stderr, "Tensor '%s' not found in file.\n", tensor_name);
        cJSON_Delete(root);
        fclose(f);
        return NULL;
    }

    cJSON* dtype_node = cJSON_GetObjectItemCaseSensitive(tensor_node, "dtype");
    cJSON* shape_node = cJSON_GetObjectItemCaseSensitive(tensor_node, "shape");
    cJSON* offsets_node = cJSON_GetObjectItemCaseSensitive(tensor_node, "data_offsets");

    if (!dtype_node || !shape_node || !offsets_node) {
        fprintf(stderr, "Invalid tensor metadata for '%s'.\n", tensor_name);
        cJSON_Delete(root);
        fclose(f);
        return NULL;
    }

    long start_offset = cJSON_GetArrayItem(offsets_node, 0)->valuedouble;
    long end_offset = cJSON_GetArrayItem(offsets_node, 1)->valuedouble;
    size_t size_bytes = end_offset - start_offset;

    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    strncpy(t->name, tensor_name, 63);
    t->name[63] = '\0';
    t->size_bytes = size_bytes;
    
    // Parse dtype
    const char* dtype_str = dtype_node->valuestring;
    if (strcmp(dtype_str, "F32") == 0) t->dtype = DTYPE_F32;
    else if (strcmp(dtype_str, "F16") == 0) t->dtype = DTYPE_F16;
    else if (strcmp(dtype_str, "I8") == 0) t->dtype = DTYPE_INT8;
    else if (strcmp(dtype_str, "I32") == 0) t->dtype = DTYPE_INT32;
    else t->dtype = DTYPE_F32; // Default fallback

    // Parse shape
    t->ndim = cJSON_GetArraySize(shape_node);
    for (int i = 0; i < t->ndim && i < 4; i++) {
        t->shape[i] = cJSON_GetArrayItem(shape_node, i)->valueint;
    }

    // Allocate and read binary data
    t->data = malloc(size_bytes);
    fseek(f, 8 + header_size + start_offset, SEEK_SET);
    if (fread(t->data, 1, size_bytes, f) != size_bytes) {
        fprintf(stderr, "Failed to read binary data for '%s'.\n", tensor_name);
        free(t->data);
        free(t);
        t = NULL;
    }

    cJSON_Delete(root);
    fclose(f);
    return t;
}
