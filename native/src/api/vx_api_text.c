/* Generated VxTextService dispatch: ownership and wire projection only. */
#include "vx_api_convert.h"
#include "vx_api_handles.h"
#include "tokenizer.h"
#include "volvoxai_ffi.h"
#include "vx_api.h"
#include <stdlib.h>
#include <string.h>

static int vx_text_report(const SynurangLiteAllocator* allocator,
                           VolvoxaiV1OperationReport** report,
                           VxStatus status, VxStage stage) {
    if (*report) {
        volvoxai_v1_operation_report_free(*report);
        allocator->deallocate(allocator->context, *report);
        *report = NULL;
    }
    if (status == VX_STATUS_OK) return vx_api_report_ok(allocator, report, stage) ? 0 : -1;
    VxOperationCode code = status == VX_STATUS_HANDLE_DISPOSED ? VX_CODE_HANDLE_DISPOSED :
        status == VX_STATUS_OUT_OF_MEMORY ? VX_CODE_OUT_OF_MEMORY : VX_CODE_INVALID_ARGUMENT;
    const char* message = status == VX_STATUS_HANDLE_DISPOSED ? "unknown or released tokenizer" :
        status == VX_STATUS_OUT_OF_MEMORY ? "tokenizer allocation failed" :
        "invalid vocabulary, UTF-8 text or tokenization options";
    return vx_api_report_fail(allocator, report, status, stage, code, message) ? 0 : -1;
}

static int vx_api_create_tokenizer(const VolvoxaiV1CreateTokenizerRequest* request,
                                    VolvoxaiV1TokenizerHandle* response, void* user_data) {
    (void)user_data;
    VxTokenizerSource source = {0}; VxTokenizer* tokenizer = NULL;
    VxVocabularyToken* tokens = NULL; VxStatus status = VX_STATUS_OK;
    source.format = request->which_vocabulary;
    source.json = request->field_vocabulary_json.data;
    source.json_bytes = request->field_vocabulary_json.len;
    source.binary = request->field_vocabulary_binary.data;
    source.binary_bytes = request->field_vocabulary_binary.len;
    source.merges = request->field_merges.data;
    source.merge_bytes = request->field_merges.len;
    if (source.format == 1) {
        const VolvoxaiV1TokenizerVocabulary* vocabulary = request->field_tokens;
        if (!vocabulary || vocabulary->field_tokens.len > SIZE_MAX / sizeof(*tokens))
            status = VX_STATUS_INVALID_ARGUMENT;
        else {
            source.token_count = vocabulary->field_tokens.len;
            if (source.token_count) {
                tokens = calloc(source.token_count, sizeof(*tokens));
                if (!tokens) status = VX_STATUS_OUT_OF_MEMORY;
            }
            for (size_t i = 0; status == VX_STATUS_OK && i < source.token_count; ++i) {
                const VolvoxaiV1VocabularyToken* token = &vocabulary->field_tokens.data[i];
                tokens[i] = (VxVocabularyToken){token->field_id, token->field_value.data, token->field_value.len};
            }
            source.tokens = tokens;
        }
    }
    if (status == VX_STATUS_OK) status = vx_tokenizer_create(&source, &tokenizer);
    free(tokens);
    if (status != VX_STATUS_OK)
        return vx_text_report(response->_allocator, &response->field_report, status, VX_STAGE_TOKENIZER_CREATE);
    /* Allocate the complete response before publishing ownership. */
    if (vx_text_report(response->_allocator, &response->field_report, VX_STATUS_OK, VX_STAGE_TOKENIZER_CREATE)) {
        vx_tokenizer_release(tokenizer); return -1;
    }
    int64_t id = vx_api_handle_insert(user_data, VX_API_HANDLE_TOKENIZER, tokenizer, vx_tokenizer_retain, vx_tokenizer_release);
    if (!id) {
        vx_tokenizer_release(tokenizer);
        return vx_text_report(response->_allocator, &response->field_report, VX_STATUS_OUT_OF_MEMORY, VX_STAGE_TOKENIZER_CREATE);
    }
    response->field_tokenizer_id = id;
    response->field_vocabulary_size = vx_tokenizer_size(tokenizer);
    response->field_merge_count = vx_tokenizer_merges(tokenizer);
    return 0;
}

static int vx_api_encode_text(const VolvoxaiV1EncodeTextRequest* request,
                               VolvoxaiV1EncodedText* response, void* user_data) {
    (void)user_data;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    uint32_t* tokens = NULL; size_t count = 0;
    VxStatus status = VX_STATUS_HANDLE_DISPOSED;
    if (vx_api_handle_acquire(user_data, VX_API_HANDLE_TOKENIZER, request->field_tokenizer_id, &lease))
        status = vx_tokenizer_encode(lease.pointer, request->field_text.data, request->field_text.len,
            request->has_max_tokens ? request->field_max_tokens : 256u,
            (int)request->field_mode, &tokens, &count);
    vx_api_handle_lease_release(&lease);
    if (status == VX_STATUS_OK && count) {
        response->field_tokens.data = response->_allocator->allocate(response->_allocator->context, count * sizeof(uint32_t));
        if (!response->field_tokens.data) status = VX_STATUS_OUT_OF_MEMORY;
        else {
            memcpy(response->field_tokens.data, tokens, count * sizeof(uint32_t));
            response->field_tokens.len = response->field_tokens.cap = count;
        }
    }
    free(tokens);
    return vx_text_report(response->_allocator, &response->field_report, status, VX_STAGE_TOKENIZE);
}

static int vx_api_decode_tokens(const VolvoxaiV1DecodeTokensRequest* request,
                                 VolvoxaiV1DecodedText* response, void* user_data) {
    (void)user_data;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    uint8_t* text = NULL; size_t bytes = 0;
    VxStatus status = VX_STATUS_HANDLE_DISPOSED;
    if (vx_api_handle_acquire(user_data, VX_API_HANDLE_TOKENIZER, request->field_tokenizer_id, &lease))
        status = vx_tokenizer_decode(lease.pointer, request->field_tokens.data, request->field_tokens.len, &text, &bytes);
    vx_api_handle_lease_release(&lease);
    if (status == VX_STATUS_OK && bytes && synurang_lite_bytes_assign(response->_allocator, &response->field_text,
            text, bytes) != SYNURANG_LITE_OK) status = VX_STATUS_OUT_OF_MEMORY;
    free(text);
    return vx_text_report(response->_allocator, &response->field_report, status, VX_STAGE_DETOKENIZE);
}

static int vx_api_release_tokenizer(const VolvoxaiV1TokenizerRef* request,
                                     VolvoxaiV1OperationReport* response, void* user_data) {
    (void)user_data;
    (void)vx_api_handle_remove(user_data, VX_API_HANDLE_TOKENIZER, request->field_tokenizer_id);
    VxReport report = VX_REPORT_INIT;
    report.status = VX_STATUS_OK; report.stage = VX_STAGE_CLOSE;
    return vx_api_report_from_native(response, &report) ? 0 : -1;
}

VX_API_UNARY(vx_api_create_tokenizer, VolvoxaiV1CreateTokenizerRequest, VolvoxaiV1TokenizerHandle,
    volvoxai_v1_tokenizer_handle, vx_text_create_tokenizer_respond)
VX_API_UNARY(vx_api_encode_text, VolvoxaiV1EncodeTextRequest, VolvoxaiV1EncodedText,
    volvoxai_v1_encoded_text, vx_text_encode_text_respond)
VX_API_UNARY(vx_api_decode_tokens, VolvoxaiV1DecodeTokensRequest, VolvoxaiV1DecodedText,
    volvoxai_v1_decoded_text, vx_text_decode_tokens_respond)
VX_API_UNARY(vx_api_release_tokenizer, VolvoxaiV1TokenizerRef, VolvoxaiV1OperationReport,
    volvoxai_v1_operation_report, vx_text_release_tokenizer_respond)

int vx_api_install_text_handlers(SynurangInstance* instance, VxApiRegistry* registry) {
    VxTextServiceHandlers handlers;
    memset(&handlers, 0, sizeof(handlers));
    handlers.create_tokenizer.message = vx_api_create_tokenizer_call;
    handlers.encode_text.message = vx_api_encode_text_call;
    handlers.decode_tokens.message = vx_api_decode_tokens_call;
    handlers.release_tokenizer.message = vx_api_release_tokenizer_call;
    return vx_text_register(instance, &handlers, registry);
}
