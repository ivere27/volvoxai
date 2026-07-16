#ifndef VOLVOXAI_EXAMPLES_TINY_RECEIPT_VQA_GRPC_FFI_CLIENT_H_
#define VOLVOXAI_EXAMPLES_TINY_RECEIPT_VQA_GRPC_FFI_CLIENT_H_

#include <dlfcn.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <google/protobuf/message_lite.h>

// A protobuf/gRPC service client transported through the Synurang plugin ABI.
// The request and response types are generated from proto/volvoxai.proto; only
// the transport is in-process. Method names are the canonical gRPC paths.
namespace volvoxai_example {

struct RpcStatus {
    int32_t code = 0;
    int32_t grpc_code = 0;
    std::string message;
};

class RpcError final : public std::runtime_error {
public:
    explicit RpcError(RpcStatus status)
        : std::runtime_error("VolvoxAI RPC failed (grpc=" +
                             std::to_string(status.grpc_code) + "): " + status.message),
          status_(std::move(status)) {}

    const RpcStatus& status() const noexcept { return status_; }

private:
    RpcStatus status_;
};

class GrpcFfiClient final {
public:
    explicit GrpcFfiClient(const std::string& library_path) {
        library_ = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!library_) {
            throw std::runtime_error("dlopen(" + library_path + ") failed: " +
                                     std::string(dlerror()));
        }
        invoke_ = reinterpret_cast<InvokeFn>(
            dlsym(library_, "Synurang_Invoke_VolvoxAiService"));
        free_ = reinterpret_cast<FreeFn>(dlsym(library_, "Synurang_Free"));
        InitFn init = reinterpret_cast<InitFn>(dlsym(library_, "VolvoxAI_Init"));
        if (!invoke_ || !free_) {
            throw std::runtime_error(
                "libvolvoxai is missing Synurang_Invoke_VolvoxAiService/Synurang_Free");
        }
        if (init) init();
    }

    GrpcFfiClient(const GrpcFfiClient&) = delete;
    GrpcFfiClient& operator=(const GrpcFfiClient&) = delete;

    ~GrpcFfiClient() {
        // Intentionally retain the plugin until process exit. GPU drivers may
        // own TLS destructors whose code must remain mapped after model cleanup.
    }

    template <class Request, class Response>
    Response Unary(const std::string& method, const Request& request) const {
        std::string request_bytes;
        if (!request.SerializeToString(&request_bytes)) {
            throw std::runtime_error("failed to serialize request for " + method);
        }
        const std::string response_bytes = Invoke(method, request_bytes);
        Response response;
        if (!response.ParseFromString(response_bytes)) {
            throw std::runtime_error("failed to parse response for " + method);
        }
        return response;
    }

private:
    using InvokeFn = char* (*)(const char*, const char*, int, int*);
    using FreeFn = void (*)(void*);
    using InitFn = void (*)();

    static bool ReadVarint(const std::string& bytes, size_t* position, uint64_t* value) {
        *value = 0;
        for (int shift = 0; shift <= 63 && *position < bytes.size(); shift += 7) {
            const uint8_t byte = static_cast<uint8_t>(bytes[(*position)++]);
            *value |= static_cast<uint64_t>(byte & 0x7f) << shift;
            if ((byte & 0x80) == 0) return true;
        }
        return false;
    }

    static bool SkipField(const std::string& bytes, size_t* position, int wire_type) {
        uint64_t length = 0;
        switch (wire_type) {
            case 0:
                return ReadVarint(bytes, position, &length);
            case 1:
                if (bytes.size() - *position < 8) return false;
                *position += 8;
                return true;
            case 2:
                if (!ReadVarint(bytes, position, &length) || length > bytes.size() - *position) {
                    return false;
                }
                *position += static_cast<size_t>(length);
                return true;
            case 5:
                if (bytes.size() - *position < 4) return false;
                *position += 4;
                return true;
            default:
                return false;
        }
    }

    static RpcStatus DecodeStatus(const std::string& bytes) {
        RpcStatus status;
        // This function is called only for the ABI's negative-length failure
        // envelope. If that envelope is truncated, never misreport success
        // (gRPC OK/0); INTERNAL is the conservative transport fallback.
        status.grpc_code = 13;
        size_t position = 0;
        while (position < bytes.size()) {
            uint64_t key = 0;
            if (!ReadVarint(bytes, &position, &key)) break;
            const int field = static_cast<int>(key >> 3);
            const int wire = static_cast<int>(key & 7);
            uint64_t value = 0;
            if ((field == 1 || field == 3) && wire == 0 &&
                ReadVarint(bytes, &position, &value)) {
                if (field == 1) status.code = static_cast<int32_t>(value);
                else status.grpc_code = static_cast<int32_t>(value);
            } else if (field == 2 && wire == 2) {
                uint64_t length = 0;
                if (!ReadVarint(bytes, &position, &length) ||
                    length > bytes.size() - position) {
                    break;
                }
                status.message.assign(bytes.data() + position, static_cast<size_t>(length));
                position += static_cast<size_t>(length);
            } else if (!SkipField(bytes, &position, wire)) {
                break;
            }
        }
        if (status.message.empty()) status.message = "unknown service error";
        return status;
    }

    std::string Invoke(const std::string& method, const std::string& request) const {
        if (request.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("request is too large for the Synurang ABI");
        }
        int response_length = 0;
        char* response = invoke_(
            method.c_str(), request.empty() ? nullptr : request.data(),
            static_cast<int>(request.size()), &response_length);
        if (!response && response_length != 0) {
            throw std::runtime_error("null response from " + method);
        }
        const bool failed = response_length < 0;
        const size_t size = static_cast<size_t>(failed ? -static_cast<int64_t>(response_length)
                                                       : response_length);
        std::string bytes;
        if (response && size) bytes.assign(response, size);
        if (response) free_(response);
        if (failed) throw RpcError(DecodeStatus(bytes));
        return bytes;
    }

    void* library_ = nullptr;
    InvokeFn invoke_ = nullptr;
    FreeFn free_ = nullptr;
};

}  // namespace volvoxai_example

#endif  // VOLVOXAI_EXAMPLES_TINY_RECEIPT_VQA_GRPC_FFI_CLIENT_H_
