/* Generic immutable SafeTensors I/O. The existing header authority validates
 * storage independently of the executable runtime's fixed tensor descriptors. */
#include "safetensors.h"

static VxStatus vx_storage_inspect(const SynurangLiteBytes* source,
                                     VolvoxaiV1SafetensorsInfo* info) {
    uint8_t* wire = NULL;
    uint32_t wire_bytes = 0;
    if (!source || !source->data || source->len < 8) return VX_STATUS_INVALID_ARGUMENT;
    uint64_t header = vx_plan_u64(source->data);
    if (header > UINT32_MAX - 8u || header > source->len - 8u) return VX_STATUS_INVALID_ARGUMENT;
    VxStatus status = vx_plan_safetensors_run(source->data, (uint32_t)header + 8u,
        source->len, &wire, &wire_bytes);
    if (status == VX_STATUS_OK && !vx_plan_project_safetensors(info->_allocator, wire, info))
        status = VX_STATUS_OUT_OF_MEMORY;
    free(wire);
    return status;
}

static float vx_storage_f16(uint16_t bits) {
    uint32_t sign = (uint32_t)(bits & 0x8000u) << 16u;
    uint32_t exponent = (bits >> 10u) & 31u, fraction = bits & 1023u, value;
    if (!exponent) {
        if (!fraction) value = sign;
        else {
            int shift = 0;
            while (!(fraction & 1024u)) { fraction <<= 1u; ++shift; }
            value = sign | (uint32_t)(113 - shift) << 23u | (fraction & 1023u) << 13u;
        }
    } else if (exponent == 31) value = sign | 0x7f800000u | fraction << 13u;
    else value = sign | (exponent + 112u) << 23u | fraction << 13u;
    float result;
    memcpy(&result, &value, sizeof(result));
    return result;
}

static int vx_storage_copy_metadata(VolvoxaiV1SafetensorsMetadata* target,
        const VolvoxaiV1SafetensorsMetadataEntry* entries, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        VolvoxaiV1SafetensorsMetadataEntry* entry = volvoxai_v1_safetensors_metadata_add_entries(target);
        if (!entry || !vx_plan_bytes(target->_allocator, &entry->field_key,
                entries[i].field_key.data, entries[i].field_key.len) ||
            !vx_plan_bytes(target->_allocator, &entry->field_value,
                entries[i].field_value.data, entries[i].field_value.len)) return 0;
    }
    return 1;
}

static int vx_storage_fail(const SynurangLiteAllocator* a,
        VolvoxaiV1OperationReport** report, VxStatus status, VxStage stage) {
    return vx_api_report_fail(a, report, status, stage,
        status == VX_STATUS_NOT_FOUND ? VX_CODE_NOT_FOUND :
        status == VX_STATUS_TRANSPORT_UNSUPPORTED ? VX_CODE_TRANSPORT_UNSUPPORTED :
        status == VX_STATUS_OUT_OF_MEMORY ? VX_CODE_OUT_OF_MEMORY : VX_CODE_INVALID_ARGUMENT,
        status == VX_STATUS_NOT_FOUND ? "selected tensor does not exist" :
        status == VX_STATUS_OUT_OF_MEMORY ? "SafeTensors allocation failed" :
        status == VX_STATUS_TRANSPORT_UNSUPPORTED ? "SafeTensors storage requires host bytes" :
        "invalid SafeTensors bytes, tensor descriptor or edit") ? 0 : -1;
}

static int vx_api_read_safetensors(const VolvoxaiV1ReadSafetensorsRequest* request,
        VolvoxaiV1SafetensorsContent* response, void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* a = response->_allocator;
    VolvoxaiV1SafetensorsInfo info;
    volvoxai_v1_safetensors_info_init_with_allocator(&info, a);
    VxStatus status = vx_storage_inspect(&request->field_source, &info);
    if (status != VX_STATUS_OK) goto done;
    if (info.field_has_metadata) {
        response->field_metadata = a->allocate(a->context, sizeof(*response->field_metadata));
        if (!response->field_metadata) goto oom;
        volvoxai_v1_safetensors_metadata_init_with_allocator(response->field_metadata, a);
        if (!vx_storage_copy_metadata(response->field_metadata, info.field_metadata.data, info.field_metadata.len)) goto oom;
    }
    size_t count = request->field_names.len ? request->field_names.len : info.field_tensors.len;
    for (size_t i = 0; i < count; ++i) {
        size_t index = i;
        if (request->field_names.len) {
            for (size_t j = 0; j < i; ++j)
                if (vx_graph_bytes_equal(&request->field_names.data[i], &request->field_names.data[j])) {
                    status = VX_STATUS_INVALID_ARGUMENT; goto done;
                }
            for (index = 0; index < info.field_tensors.len; ++index)
                if (vx_graph_bytes_equal(&request->field_names.data[i], &info.field_tensors.data[index].field_name)) break;
            if (index == info.field_tensors.len) { status = VX_STATUS_NOT_FOUND; goto done; }
        }
        const VolvoxaiV1SafetensorsTensorInfo* source = &info.field_tensors.data[index];
        VolvoxaiV1Tensor* target = volvoxai_v1_safetensors_content_add_tensors(response);
        if (!target || !vx_plan_bytes(a, &target->field_name, source->field_name.data, source->field_name.len)) goto oom;
        target->field_dtype = source->field_dtype;
        target->which_payload = 5;
        for (size_t d = 0; d < source->field_shape.len; ++d) {
            int64_t* axis = volvoxai_v1_tensor_add_shape(target);
            if (!axis) goto oom;
            *axis = source->field_shape.data[d];
        }
        const uint8_t* payload = request->field_source.data + (size_t)source->field_file_offset;
        if (request->field_normalize_f16 && source->field_dtype == VOLVOXAI_V1_DATA_TYPE_F16) {
            if (source->field_byte_size > SIZE_MAX / 2u) goto oom;
            size_t n = (size_t)source->field_byte_size / 2u;
            float* floats = n ? malloc(n * sizeof(float)) : NULL;
            if (n && !floats) goto oom;
            for (size_t j = 0; j < n; ++j) floats[j] = vx_storage_f16(
                (uint16_t)payload[j * 2] | (uint16_t)payload[j * 2 + 1] << 8u);
            int ok = vx_plan_bytes(a, &target->field_inline, floats, n * sizeof(float));
            free(floats);
            if (!ok) goto oom;
            target->field_dtype = VOLVOXAI_V1_DATA_TYPE_F32;
        } else if (!vx_plan_bytes(a, &target->field_inline, payload, (size_t)source->field_byte_size)) goto oom;
    }
    goto done;
oom:
    status = VX_STATUS_OUT_OF_MEMORY;
done:
    volvoxai_v1_safetensors_info_free(&info);
    if (status == VX_STATUS_OK) return vx_api_report_ok(a, &response->field_report, VX_STAGE_SAFETENSORS_READ) ? 0 : -1;
    volvoxai_v1_safetensors_content_free(response);
    volvoxai_v1_safetensors_content_init_with_allocator(response, a);
    return vx_storage_fail(a, &response->field_report, status, VX_STAGE_SAFETENSORS_READ);
}

typedef struct VxStorageBuffer {
    uint8_t* data;
    size_t size, capacity;
    VxStatus status;
} VxStorageBuffer;

static void vx_storage_append(VxStorageBuffer* b, const void* bytes, size_t size) {
    if (b->status != VX_STATUS_OK) return;
    if (size > SIZE_MAX - b->size || (size && !bytes)) { b->status = VX_STATUS_INVALID_ARGUMENT; return; }
    size_t required = b->size + size;
    if (required > b->capacity) {
        size_t capacity = b->capacity <= SIZE_MAX / 2u ? b->capacity * 2u : SIZE_MAX;
        if (capacity < 256u) capacity = 256u;
        if (capacity < required) capacity = required;
        void* next = realloc(b->data, capacity);
        if (!next) { b->status = VX_STATUS_OUT_OF_MEMORY; return; }
        b->data = next; b->capacity = capacity;
    }
    if (size) memcpy(b->data + b->size, bytes, size);
    b->size = required;
}

static void vx_storage_literal(VxStorageBuffer* b, const char* text) {
    vx_storage_append(b, text, strlen(text));
}

static void vx_storage_string(VxStorageBuffer* b, const SynurangLiteBytes* value) {
    static const char hex[] = "0123456789abcdef";
    vx_storage_literal(b, "\"");
    for (size_t i = 0; i < value->len; ++i) {
        uint8_t c = value->data[i];
        if (c < 32u) {
            char escape[] = "\\u0000";
            escape[4] = hex[c >> 4u]; escape[5] = hex[c & 15u];
            vx_storage_append(b, escape, 6);
        } else {
            if (c == '"' || c == '\\') vx_storage_literal(b, "\\");
            vx_storage_append(b, &c, 1);
        }
    }
    vx_storage_literal(b, "\"");
}

static void vx_storage_integer(VxStorageBuffer* b, uint64_t value) {
    char text[32];
    int n = snprintf(text, sizeof(text), "%" PRIu64, value);
    if (n <= 0 || (size_t)n >= sizeof(text)) { b->status = VX_STATUS_INTERNAL; return; }
    vx_storage_append(b, text, (size_t)n);
}

static VxStatus vx_storage_set_tensor(VolvoxaiV1SafetensorsContent* working,
                                        const VolvoxaiV1Tensor* source) {
    if (!source || !source->field_name.len || !source->field_name.data ||
        (source->field_name.len == 12u && !memcmp(source->field_name.data, "__metadata__", 12u))) return VX_STATUS_INVALID_ARGUMENT;
    if (source->field_location != VOLVOXAI_V1_MEMORY_LOCATION_HOST) return VX_STATUS_TRANSPORT_UNSUPPORTED;
    uint64_t elements = 1;
    size_t bits = safetensors_dtype_bit_width((VxDataType)source->field_dtype);
    if (!bits) return VX_STATUS_INVALID_ARGUMENT;
    for (size_t i = 0; i < source->field_shape.len; ++i) {
        int64_t d = source->field_shape.data[i];
        if (d < 0 || (uint64_t)d > VOLVOXAI_SAFETENSORS_MAX_EXACT_INTEGER ||
            (d && elements > VOLVOXAI_SAFETENSORS_MAX_EXACT_INTEGER / (uint64_t)d)) return VX_STATUS_INVALID_ARGUMENT;
        elements *= (uint64_t)d;
    }
    if (elements > VOLVOXAI_SAFETENSORS_MAX_EXACT_INTEGER / bits || (elements * bits) % 8u)
        return VX_STATUS_INVALID_ARGUMENT;
    uint64_t expected = elements * bits / 8u;
    if (expected > SIZE_MAX) return VX_STATUS_OUT_OF_MEMORY;
    const void* data = NULL;
    void* zeroes = NULL;
    size_t data_size = 0;
    VxStatus status = VX_STATUS_OK;
    if (source->which_payload == 5) { data = source->field_inline.data; data_size = source->field_inline.len; }
    else if (source->which_payload == 6) {
        void* pointer = NULL;
        status = vx_api_buffer_view_resolve(source->field_view, &pointer, &data_size);
        data = pointer;
        if (status != VX_STATUS_OK) return status;
    } else if (!source->which_payload) {
        zeroes = expected ? calloc(1u, (size_t)expected) : NULL;
        if (expected && !zeroes) return VX_STATUS_OUT_OF_MEMORY;
        data = zeroes; data_size = (size_t)expected;
    } else return VX_STATUS_INVALID_ARGUMENT;
    if (data_size != expected || (expected && !data)) { free(zeroes); return VX_STATUS_INVALID_ARGUMENT; }
    VolvoxaiV1Tensor* target = NULL;
    for (size_t i = 0; i < working->field_tensors.len; ++i)
        if (vx_graph_bytes_equal(&source->field_name, &working->field_tensors.data[i].field_name)) {
            target = &working->field_tensors.data[i];
            volvoxai_v1_tensor_free(target);
            volvoxai_v1_tensor_init_with_allocator(target, working->_allocator);
            break;
        }
    if (!target) target = volvoxai_v1_safetensors_content_add_tensors(working);
    if (!target || !vx_plan_bytes(working->_allocator, &target->field_name, source->field_name.data, source->field_name.len) ||
        !vx_plan_bytes(working->_allocator, &target->field_inline, data, data_size)) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
    target->field_dtype = source->field_dtype; target->which_payload = 5;
    for (size_t i = 0; i < source->field_shape.len; ++i) {
        int64_t* axis = volvoxai_v1_tensor_add_shape(target);
        if (!axis) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
        *axis = source->field_shape.data[i];
    }
done:
    free(zeroes);
    return status;
}

static void vx_storage_remove_tensor(VolvoxaiV1SafetensorsContent* working,
                                       const SynurangLiteBytes* name) {
    for (size_t i = 0; i < working->field_tensors.len; ++i) {
        VolvoxaiV1Tensor* tensors = working->field_tensors.data;
        if (!vx_graph_bytes_equal(&tensors[i].field_name, name)) continue;
        volvoxai_v1_tensor_free(&tensors[i]);
        --working->field_tensors.len;
        memmove(&tensors[i], &tensors[i + 1], (working->field_tensors.len - i) * sizeof(*tensors));
        memset(&tensors[working->field_tensors.len], 0, sizeof(*tensors));
        return;
    }
}

static VxStatus vx_storage_serialize(const VolvoxaiV1SafetensorsContent* working,
                                       VolvoxaiV1SafetensorsArtifact* response) {
    VxStorageBuffer header = {0};
    uint64_t cursor = 0;
    vx_storage_literal(&header, "{");
    if (working->field_metadata) {
        vx_storage_literal(&header, "\"__metadata__\":{");
        for (size_t i = 0; i < working->field_metadata->field_entries.len; ++i) {
            const VolvoxaiV1SafetensorsMetadataEntry* e = &working->field_metadata->field_entries.data[i];
            if (i) vx_storage_literal(&header, ",");
            vx_storage_string(&header, &e->field_key); vx_storage_literal(&header, ":");
            vx_storage_string(&header, &e->field_value);
        }
        vx_storage_literal(&header, "}");
    }
    for (size_t i = 0; i < working->field_tensors.len; ++i) {
        const VolvoxaiV1Tensor* t = &working->field_tensors.data[i];
        if (i || working->field_metadata) vx_storage_literal(&header, ",");
        vx_storage_string(&header, &t->field_name);
        vx_storage_literal(&header, ":{\"dtype\":\"");
        vx_storage_literal(&header, safetensors_dtype_name((VxDataType)t->field_dtype));
        vx_storage_literal(&header, "\",\"shape\":[");
        for (size_t d = 0; d < t->field_shape.len; ++d) {
            if (d) vx_storage_literal(&header, ",");
            vx_storage_integer(&header, (uint64_t)t->field_shape.data[d]);
        }
        vx_storage_literal(&header, "],\"data_offsets\":["); vx_storage_integer(&header, cursor);
        if (t->field_inline.len > VOLVOXAI_SAFETENSORS_MAX_EXACT_INTEGER - cursor) {
            header.status = VX_STATUS_INVALID_ARGUMENT; break;
        }
        cursor += t->field_inline.len;
        vx_storage_literal(&header, ","); vx_storage_integer(&header, cursor);
        vx_storage_literal(&header, "]}");
    }
    vx_storage_literal(&header, "}");
    while (header.status == VX_STATUS_OK && header.size % 8u) vx_storage_literal(&header, " ");
    VxStatus status = header.status;
    uint8_t* blob = NULL;
    VolvoxaiV1SafetensorsInfo info;
    volvoxai_v1_safetensors_info_init(&info);
    if (status != VX_STATUS_OK) goto done;
    if (header.size > SIZE_MAX - 8u || cursor > SIZE_MAX - header.size - 8u) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
    size_t bytes = header.size + 8u + (size_t)cursor;
    blob = malloc(bytes);
    if (!blob) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
    for (unsigned i = 0; i < 8; ++i) blob[i] = (uint8_t)((uint64_t)header.size >> (i * 8u));
    memcpy(blob + 8u, header.data, header.size);
    size_t offset = header.size + 8u;
    for (size_t i = 0; i < working->field_tensors.len; ++i) {
        const SynurangLiteBytes* data = &working->field_tensors.data[i].field_inline;
        if (data->len) memcpy(blob + offset, data->data, data->len);
        offset += data->len;
    }
    /* Includes duplicate metadata keys, UTF-8 and exact coverage. No artifact
     * is published until the same reader accepts the entire resulting file. */
    SynurangLiteBytes source = {0}; source.data = blob; source.len = bytes;
    status = vx_storage_inspect(&source, &info);
    if (status == VX_STATUS_OK && !vx_plan_bytes(response->_allocator, &response->field_data, blob, bytes)) status = VX_STATUS_OUT_OF_MEMORY;
done:
    volvoxai_v1_safetensors_info_free(&info);
    free(blob); free(header.data);
    return status;
}

static int vx_api_write_safetensors(const VolvoxaiV1WriteSafetensorsRequest* request,
        VolvoxaiV1SafetensorsArtifact* response, void* user_data) {
    VolvoxaiV1SafetensorsContent working;
    volvoxai_v1_safetensors_content_init(&working);
    VxStatus status = VX_STATUS_OK;
    if (request->field_source.len) {
        VolvoxaiV1ReadSafetensorsRequest read;
        volvoxai_v1_read_safetensors_request_init(&read);
        read.field_source = request->field_source;
        int result = vx_api_read_safetensors(&read, &working, user_data);
        if (result != 0) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
        status = (VxStatus)working.field_report->field_status;
        if (status != VX_STATUS_OK) goto done;
    }
    if (request->which_metadata_update) {
        if (working.field_metadata) {
            volvoxai_v1_safetensors_metadata_free(working.field_metadata);
            working._allocator->deallocate(working._allocator->context, working.field_metadata);
            working.field_metadata = NULL;
        }
        if (request->which_metadata_update == 3 && request->field_metadata) {
            working.field_metadata = working._allocator->allocate(working._allocator->context, sizeof(*working.field_metadata));
            if (!working.field_metadata) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
            volvoxai_v1_safetensors_metadata_init_with_allocator(working.field_metadata, working._allocator);
            if (!vx_storage_copy_metadata(working.field_metadata, request->field_metadata->field_entries.data,
                                           request->field_metadata->field_entries.len)) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
        } else if (request->which_metadata_update != 4 || !request->field_remove_metadata) { status = VX_STATUS_INVALID_ARGUMENT; goto done; }
    }
    for (size_t i = 0; i < request->field_edits.len; ++i) {
        const VolvoxaiV1SafetensorsEdit* e = &request->field_edits.data[i];
        if (e->which_operation == 1) status = vx_storage_set_tensor(&working, e->field_set_tensor);
        else if (e->which_operation == 2) vx_storage_remove_tensor(&working, &e->field_remove_tensor);
        else status = VX_STATUS_INVALID_ARGUMENT;
        if (status != VX_STATUS_OK) goto done;
    }
    status = vx_storage_serialize(&working, response);
done:
    volvoxai_v1_safetensors_content_free(&working);
    if (status != VX_STATUS_OK) return vx_storage_fail(response->_allocator,
        &response->field_report, status, VX_STAGE_SAFETENSORS_WRITE);
    return vx_api_report_ok(response->_allocator, &response->field_report, VX_STAGE_SAFETENSORS_WRITE) ? 0 : -1;
}
