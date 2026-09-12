/* Descriptor-derived metadata shares the public dispatch and owner allocator.
 * Only the active profile's tables are compiled into the library. */
typedef struct VxApiMethodEntry {
    const char* service;
    const char* name;
    const unsigned char* data;
    size_t size;
    size_t request_type;
    size_t response_type;
} VxApiMethodEntry;

typedef struct VxApiTypeEntry {
    int is_enum;
    const unsigned char* data;
    size_t size;
    const size_t* dependencies;
    size_t dependency_count;
} VxApiTypeEntry;

#if VOLVOXAI_ENABLE_TRAINING
#include "generated/api_contract_full.inc"
#else
#include "generated/api_contract_inference.inc"
#endif

static int vx_api_contract_matches(const SynurangLiteBytes* filter, const char* name) {
    return !filter->len || (filter->len == strlen(name) &&
                            !memcmp(filter->data, name, filter->len));
}

static int vx_api_describe_api(const VolvoxaiV1DescribeApiRequest* request,
                               VolvoxaiV1ApiDescription* response,
                               void* user_data) {
    const SynurangLiteAllocator* allocator = response->_allocator;
    unsigned char selected[sizeof(vx_api_type_entries) / sizeof(vx_api_type_entries[0])] = {0};
    size_t count = 0;
    (void)user_data;
    if (request->field_method.len && !request->field_service.len)
        return vx_api_report_fail(allocator, &response->field_report,
            VX_STATUS_INVALID_ARGUMENT, VX_STAGE_NONE, VX_CODE_INVALID_ARGUMENT,
            "method requires a service filter") ? 0 : -1;
    for (size_t i = 0; i < sizeof(vx_api_method_entries) / sizeof(vx_api_method_entries[0]); i++) {
        const VxApiMethodEntry* entry = &vx_api_method_entries[i];
        VolvoxaiV1ApiMethod* method;
        if (!vx_api_contract_matches(&request->field_service, entry->service) ||
            !vx_api_contract_matches(&request->field_method, entry->name)) continue;
        method = volvoxai_v1_api_description_add_methods(response);
        if (!method || volvoxai_v1_api_method_decode(method, entry->data, entry->size) != SYNURANG_LITE_OK)
            return -1;
        selected[entry->request_type] = selected[entry->response_type] = 1;
        count++;
    }
    if (!count)
        return vx_api_report_fail(allocator, &response->field_report,
            VX_STATUS_NOT_FOUND, VX_STAGE_NONE, VX_CODE_NOT_FOUND,
            "service or method is not available in this build") ? 0 : -1;
    if (request->field_include_types) {
        int changed;
        do {
            changed = 0;
            for (size_t i = 0; i < sizeof(selected); i++) {
                const VxApiTypeEntry* entry = &vx_api_type_entries[i];
                if (!selected[i]) continue;
                for (size_t j = 0; j < entry->dependency_count; j++) {
                    size_t dependency = entry->dependencies[j];
                    if (!selected[dependency]) {
                        selected[dependency] = 1;
                        changed = 1;
                    }
                }
            }
        } while (changed);
        for (size_t i = 0; i < sizeof(selected); i++) {
            const VxApiTypeEntry* entry = &vx_api_type_entries[i];
            if (!selected[i]) continue;
            if (entry->is_enum) {
                VolvoxaiV1ApiEnum* value = volvoxai_v1_api_description_add_enums(response);
                if (!value || volvoxai_v1_api_enum_decode(value, entry->data, entry->size) != SYNURANG_LITE_OK)
                    return -1;
            } else {
                VolvoxaiV1ApiMessage* value = volvoxai_v1_api_description_add_messages(response);
                if (!value || volvoxai_v1_api_message_decode(value, entry->data, entry->size) != SYNURANG_LITE_OK)
                    return -1;
            }
        }
    }
    if (synurang_lite_bytes_assign(allocator, &response->field_schema_sha256,
            VX_API_SCHEMA_SHA256, sizeof(VX_API_SCHEMA_SHA256) - 1u) != SYNURANG_LITE_OK)
        return -1;
    return vx_api_report_ok(allocator, &response->field_report, VX_STAGE_NONE) ? 0 : -1;
}
