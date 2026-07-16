// Standalone C++ FFI client for the VolvoxAI runtime shared library.
//
// It loads the Synurang plugin server ABI with dlopen/dlsym and uses the
// Synurang-generated dependency-free C-lite messages for protobuf payloads. It
// executes:
//   * models/efficientdet_lite0_fp32
//   * models/tinystories_1m
//
// Build from repo root:
//   make -C examples cpp_ffi_example
//
// Run from repo root:
//   ./examples/target/bin/cpp_ffi_client runtime/target/release/libvolvoxai.so .

#include <dlfcn.h>

#include "volvoxai_lite.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Bytes = std::vector<uint8_t>;

constexpr VolvoxaiV1DataType DATA_TYPE_I32 = VOLVOXAI_V1_DATA_TYPE_DATA_TYPE_I32;
constexpr VolvoxaiV1DataType DATA_TYPE_F32 = VOLVOXAI_V1_DATA_TYPE_DATA_TYPE_F32;

// Synurang's ABI error payload is core.v1.Error, which is separate from the
// VolvoxAI contract. Keep its tiny reader isolated from all service messages;
// every VolvoxAI request/response below uses generated C-lite codecs.
namespace synurang_error_wire {

struct Cursor {
    const uint8_t* data = nullptr;
    size_t size = 0;
    size_t pos = 0;

    Cursor() = default;
    explicit Cursor(const Bytes& b) : data(b.data()), size(b.size()) {}
    Cursor(const uint8_t* p, size_t n) : data(p), size(n) {}

    bool eof() const { return pos >= size; }

    bool read_varint(uint64_t& value) {
        value = 0;
        int shift = 0;
        while (pos < size && shift <= 63) {
            uint8_t byte = data[pos++];
            value |= static_cast<uint64_t>(byte & 0x7f) << shift;
            if ((byte & 0x80) == 0) return true;
            shift += 7;
        }
        return false;
    }

    bool read_len(Bytes& out) {
        uint64_t len = 0;
        if (!read_varint(len) || len > size - pos) return false;
        if (len == 0) {
            out.clear();
            return true;
        }
        out.assign(data + pos, data + pos + len);
        pos += static_cast<size_t>(len);
        return true;
    }

    bool read_string(std::string& out) {
        Bytes bytes;
        if (!read_len(bytes)) return false;
        if (bytes.empty()) {
            out.clear();
            return true;
        }
        out.assign(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        return true;
    }

    bool skip(int wire) {
        uint64_t value = 0;
        Bytes bytes;
        switch (wire) {
            case 0:
                return read_varint(value);
            case 1:
                if (size - pos < 8) return false;
                pos += 8;
                return true;
            case 2:
                return read_len(bytes);
            case 5:
                if (size - pos < 4) return false;
                pos += 4;
                return true;
            default:
                return false;
        }
    }
};

}  // namespace synurang_error_wire

struct FfiErrorPayload {
    int32_t code = 0;
    std::string message;
    int32_t grpc_code = 0;
};

struct ModelInfo {
    int32_t num_ops = 0;
    bool has_tokenizer = false;
};

struct LoadModelResponse {
    std::string model_id;
    ModelInfo info;
};

struct Tensor {
    std::string name;
    std::vector<int64_t> shape;
    int32_t dtype = 0;
    Bytes data;
};

struct RunResponse {
    std::vector<Tensor> outputs;
};

struct TokenEvent {
    int32_t id = 0;
    std::string text;
    int32_t position = 0;
};

struct DoneEvent {
    std::string text;
    int32_t num_tokens = 0;
    int32_t reason = 0;
};

struct GenerateEvent {
    bool has_token = false;
    bool has_done = false;
    TokenEvent token;
    DoneEvent done;
};

struct Detection {
    int class_id = 0;
    float score = 0.0f;
    float y0 = 0.0f;
    float x0 = 0.0f;
    float y1 = 0.0f;
    float x1 = 0.0f;
};

static std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty() || a == ".") return b;
    if (a.back() == '/') return a + b;
    return a + "/" + b;
}

static Bytes read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return Bytes(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

static bool file_exists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return static_cast<bool>(f);
}

static uint32_t read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

static int32_t read_i32_le(const uint8_t* p) {
    return static_cast<int32_t>(read_u32_le(p));
}

static float read_f32_le(const uint8_t* p) {
    uint32_t bits = read_u32_le(p);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static size_t numel(const std::vector<int64_t>& shape) {
    size_t n = 1;
    for (int64_t dim : shape) {
        n *= static_cast<size_t>(dim);
    }
    return n;
}

static const char* lite_status_name(SynurangLiteStatus status) {
    switch (status) {
        case SYNURANG_LITE_OK:
            return "ok";
        case SYNURANG_LITE_INVALID_ARGUMENT:
            return "invalid argument";
        case SYNURANG_LITE_OUT_OF_MEMORY:
            return "out of memory";
        case SYNURANG_LITE_MALFORMED:
            return "malformed protobuf";
        case SYNURANG_LITE_OVERFLOW:
            return "overflow";
    }
    return "unknown error";
}

static void require_lite(SynurangLiteStatus status, const char* operation) {
    if (status != SYNURANG_LITE_OK) {
        throw std::runtime_error(std::string(operation) + ": " + lite_status_name(status));
    }
}

template <typename T>
struct LiteMessageDeleter {
    void (*free_message)(T*) = nullptr;

    void operator()(T* message) const noexcept {
        if (message) {
            free_message(message);
            delete message;
        }
    }
};

template <typename T>
using LiteMessage = std::unique_ptr<T, LiteMessageDeleter<T>>;

template <typename T>
static LiteMessage<T> make_lite_message(void (*init_message)(T*),
                                        void (*free_message)(T*)) {
    T* message = new T;
    init_message(message);
    return LiteMessage<T>(message, LiteMessageDeleter<T>{free_message});
}

template <typename T>
static T* allocate_lite_array(const SynurangLiteAllocator* allocator, size_t count) {
    if (count == 0) return nullptr;
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
        throw std::runtime_error("generated message allocation overflow");
    }
    allocator = synurang_lite_allocator_or_default(allocator);
    void* storage = allocator->allocate(allocator->context, count * sizeof(T));
    if (!storage) throw std::runtime_error("generated message allocation failed");
    std::memset(storage, 0, count * sizeof(T));
    return static_cast<T*>(storage);
}

template <typename T>
static T* allocate_lite_message(const SynurangLiteAllocator* allocator,
                                void (*init_message)(T*, const SynurangLiteAllocator*)) {
    T* message = allocate_lite_array<T>(allocator, 1);
    init_message(message, allocator);
    return message;
}

static void assign_lite_bytes(const SynurangLiteAllocator* allocator,
                              SynurangLiteBytes* destination,
                              const std::string& value,
                              const char* field) {
    const void* data = value.empty() ? nullptr : static_cast<const void*>(value.data());
    require_lite(
        synurang_lite_bytes_assign(allocator, destination, data, value.size()), field);
}

static void assign_lite_bytes(const SynurangLiteAllocator* allocator,
                              SynurangLiteBytes* destination,
                              const Bytes& value,
                              const char* field) {
    const void* data = value.empty() ? nullptr : static_cast<const void*>(value.data());
    require_lite(
        synurang_lite_bytes_assign(allocator, destination, data, value.size()), field);
}

static std::string lite_string(const SynurangLiteBytes& value, const char* field) {
    if (value.len == 0) return {};
    if (!value.data) throw std::runtime_error(std::string(field) + ": null string data");
    return std::string(reinterpret_cast<const char*>(value.data), value.len);
}

static Bytes lite_bytes(const SynurangLiteBytes& value, const char* field) {
    if (value.len == 0) return {};
    if (!value.data) throw std::runtime_error(std::string(field) + ": null bytes data");
    return Bytes(value.data, value.data + value.len);
}

template <typename T>
static Bytes encode_lite_message(
    const T& message,
    SynurangLiteStatus (*encode_message)(const T*, uint8_t**, size_t*),
    const char* operation) {
    uint8_t* encoded = nullptr;
    size_t encoded_len = 0;
    SynurangLiteStatus status = encode_message(&message, &encoded, &encoded_len);
    if (status != SYNURANG_LITE_OK) {
        synurang_lite_release(message._allocator, encoded);
        require_lite(status, operation);
    }

    try {
        Bytes result;
        if (encoded_len != 0) result.assign(encoded, encoded + encoded_len);
        synurang_lite_release(message._allocator, encoded);
        return result;
    } catch (...) {
        synurang_lite_release(message._allocator, encoded);
        throw;
    }
}

template <typename T>
static LiteMessage<T> decode_lite_message(
    const Bytes& payload,
    void (*init_message)(T*),
    void (*free_message)(T*),
    SynurangLiteStatus (*decode_message)(T*, const uint8_t*, size_t),
    const char* operation) {
    LiteMessage<T> message = make_lite_message(init_message, free_message);
    const uint8_t* data = payload.empty() ? nullptr : payload.data();
    require_lite(decode_message(message.get(), data, payload.size()), operation);
    return message;
}

static void assign_tensor(VolvoxaiV1Tensor* destination, const Tensor& source) {
    assign_lite_bytes(destination->_allocator, &destination->field_name, source.name,
                      "Tensor.name");
    destination->field_shape.data =
        allocate_lite_array<int64_t>(destination->_allocator, source.shape.size());
    destination->field_shape.cap = source.shape.size();
    for (int64_t dimension : source.shape) {
        destination->field_shape.data[destination->field_shape.len++] = dimension;
    }
    destination->field_dtype = static_cast<VolvoxaiV1DataType>(source.dtype);
    assign_lite_bytes(destination->_allocator, &destination->field_data, source.data,
                      "Tensor.data");
}

static Tensor from_lite_tensor(const VolvoxaiV1Tensor& source) {
    Tensor tensor;
    tensor.name = lite_string(source.field_name, "Tensor.name");
    if (source.field_shape.len != 0) {
        if (!source.field_shape.data) throw std::runtime_error("Tensor.shape: null data");
        tensor.shape.assign(source.field_shape.data,
                            source.field_shape.data + source.field_shape.len);
    }
    tensor.dtype = source.field_dtype;
    tensor.data = lite_bytes(source.field_data, "Tensor.data");
    return tensor;
}

static Bytes encode_empty_request() {
    auto request = make_lite_message(volvoxai_v1_empty_init, volvoxai_v1_empty_free);
    return encode_lite_message(*request, volvoxai_v1_empty_encode, "encode Empty");
}

static Bytes encode_load_model_request(const std::string& config, const std::string& weights,
                                       const std::string& tokenizer = "") {
    auto request = make_lite_message(volvoxai_v1_load_model_request_init,
                                     volvoxai_v1_load_model_request_free);
    request->field_paths = allocate_lite_message(
        request->_allocator, volvoxai_v1_model_paths_init_with_allocator);
    request->which_source = 1;
    assign_lite_bytes(request->_allocator, &request->field_paths->field_config_path, config,
                      "LoadModelRequest.paths.config_path");
    assign_lite_bytes(request->_allocator, &request->field_paths->field_weights_path, weights,
                      "LoadModelRequest.paths.weights_path");
    if (!tokenizer.empty()) {
        assign_lite_bytes(request->_allocator, &request->field_paths->field_tokenizer_path,
                          tokenizer, "LoadModelRequest.paths.tokenizer_path");
    }
    return encode_lite_message(*request, volvoxai_v1_load_model_request_encode,
                               "encode LoadModelRequest");
}

static Bytes encode_model_ref(const std::string& model_id) {
    auto request =
        make_lite_message(volvoxai_v1_model_ref_init, volvoxai_v1_model_ref_free);
    assign_lite_bytes(request->_allocator, &request->field_model_id, model_id,
                      "ModelRef.model_id");
    return encode_lite_message(*request, volvoxai_v1_model_ref_encode, "encode ModelRef");
}

static Bytes encode_run_request(const std::string& model_id, const std::vector<Tensor>& inputs,
                                const std::vector<std::string>& output_names,
                                int32_t last_token = -1) {
    auto request =
        make_lite_message(volvoxai_v1_run_request_init, volvoxai_v1_run_request_free);
    assign_lite_bytes(request->_allocator, &request->field_model_id, model_id,
                      "RunRequest.model_id");

    request->field_inputs.data =
        allocate_lite_array<VolvoxaiV1Tensor>(request->_allocator, inputs.size());
    request->field_inputs.cap = inputs.size();
    for (const Tensor& input : inputs) {
        VolvoxaiV1Tensor* destination =
            &request->field_inputs.data[request->field_inputs.len];
        volvoxai_v1_tensor_init_with_allocator(destination, request->_allocator);
        ++request->field_inputs.len;
        assign_tensor(destination, input);
    }

    request->field_output_names.data =
        allocate_lite_array<SynurangLiteBytes>(request->_allocator, output_names.size());
    request->field_output_names.cap = output_names.size();
    for (const std::string& output_name : output_names) {
        SynurangLiteBytes* destination =
            &request->field_output_names.data[request->field_output_names.len++];
        assign_lite_bytes(request->_allocator, destination, output_name,
                          "RunRequest.output_names");
    }

    if (last_token >= 0) {
        request->has_last_token = 1;
        request->field_last_token = last_token;
    }
    return encode_lite_message(*request, volvoxai_v1_run_request_encode,
                               "encode RunRequest");
}

static Bytes encode_generate_request(const std::string& model_id, const std::string& prompt,
                                     int32_t max_new_tokens) {
    auto request = make_lite_message(volvoxai_v1_generate_request_init,
                                     volvoxai_v1_generate_request_free);
    assign_lite_bytes(request->_allocator, &request->field_model_id, model_id,
                      "GenerateRequest.model_id");
    assign_lite_bytes(request->_allocator, &request->field_prompt, prompt,
                      "GenerateRequest.prompt");
    request->field_gen = allocate_lite_message(
        request->_allocator, volvoxai_v1_generation_config_init_with_allocator);
    request->has_gen = 1;
    request->field_gen->has_max_new_tokens = 1;
    request->field_gen->field_max_new_tokens = max_new_tokens;
    request->field_gen->has_eos_token = 1;
    request->field_gen->field_eos_token = 50256;
    request->field_gen->has_pad_token = 1;
    request->field_gen->field_pad_token = 50256;
    return encode_lite_message(*request, volvoxai_v1_generate_request_encode,
                               "encode GenerateRequest");
}

static FfiErrorPayload decode_error(const Bytes& data) {
    FfiErrorPayload err;
    synurang_error_wire::Cursor c(data);
    while (!c.eof()) {
        uint64_t key = 0;
        if (!c.read_varint(key)) break;
        int field = static_cast<int>(key >> 3);
        int wire = static_cast<int>(key & 7);
        uint64_t value = 0;
        switch (field) {
            case 1:
                if (wire == 0 && c.read_varint(value)) err.code = static_cast<int32_t>(value);
                else if (!c.skip(wire)) return err;
                break;
            case 2:
                if (wire == 2) c.read_string(err.message);
                else if (!c.skip(wire)) return err;
                break;
            case 3:
                if (wire == 0 && c.read_varint(value)) err.grpc_code = static_cast<int32_t>(value);
                else if (!c.skip(wire)) return err;
                break;
            default:
                if (!c.skip(wire)) return err;
                break;
        }
    }
    return err;
}

static LoadModelResponse decode_load_model_response(const Bytes& data) {
    auto message = decode_lite_message(
        data, volvoxai_v1_load_model_response_init, volvoxai_v1_load_model_response_free,
        volvoxai_v1_load_model_response_decode, "decode LoadModelResponse");
    LoadModelResponse resp;
    resp.model_id = lite_string(message->field_model_id, "LoadModelResponse.model_id");
    if (message->field_info) {
        resp.info.num_ops = message->field_info->field_num_ops;
        resp.info.has_tokenizer = message->field_info->field_has_tokenizer != 0;
    }
    return resp;
}

static RunResponse decode_run_response(const Bytes& data) {
    auto message = decode_lite_message(
        data, volvoxai_v1_run_response_init, volvoxai_v1_run_response_free,
        volvoxai_v1_run_response_decode, "decode RunResponse");
    RunResponse resp;
    resp.outputs.reserve(message->field_outputs.len);
    for (size_t i = 0; i < message->field_outputs.len; ++i) {
        resp.outputs.push_back(from_lite_tensor(message->field_outputs.data[i]));
    }
    return resp;
}

static GenerateEvent decode_generate_event(const Bytes& data) {
    auto message = decode_lite_message(
        data, volvoxai_v1_generate_event_init, volvoxai_v1_generate_event_free,
        volvoxai_v1_generate_event_decode, "decode GenerateEvent");
    GenerateEvent event;
    if (message->which_event == 1 && message->field_token) {
        event.has_token = true;
        event.token.id = message->field_token->field_id;
        event.token.text = lite_string(message->field_token->field_text, "Token.text");
        event.token.position = message->field_token->field_position;
    } else if (message->which_event == 2 && message->field_done) {
        event.has_done = true;
        event.done.text = lite_string(message->field_done->field_text, "GenerateDone.text");
        event.done.num_tokens = message->field_done->field_num_tokens;
        event.done.reason = message->field_done->field_reason;
    }
    return event;
}

static int ffi_size(size_t size, const char* payload) {
    if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(payload) + " exceeds the Synurang ABI limit");
    }
    return static_cast<int>(size);
}

struct FfiLibrary {
    using InvokeFn = char* (*)(const char*, const char*, int, int*);
    using FreeFn = void (*)(void*);
    using InitFn = void (*)();
    using StreamOpenFn = uint64_t (*)(const char*);
    using StreamSendFn = int (*)(uint64_t, const char*, int);
    using StreamRecvFn = char* (*)(uint64_t, int*, int*);
    using StreamCloseSendFn = void (*)(uint64_t);
    using StreamCloseFn = void (*)(uint64_t);

    void* handle = nullptr;
    InvokeFn invoke = nullptr;
    FreeFn free_fn = nullptr;
    InitFn init = nullptr;
    StreamOpenFn stream_open = nullptr;
    StreamSendFn stream_send = nullptr;
    StreamRecvFn stream_recv = nullptr;
    StreamCloseSendFn stream_close_send = nullptr;
    StreamCloseFn stream_close = nullptr;

    explicit FfiLibrary(const std::string& path) {
        handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
        invoke = reinterpret_cast<InvokeFn>(dlsym(handle, "Synurang_Invoke_VolvoxAiService"));
        free_fn = reinterpret_cast<FreeFn>(dlsym(handle, "Synurang_Free"));
        init = reinterpret_cast<InitFn>(dlsym(handle, "VolvoxAI_Init"));
        stream_open = reinterpret_cast<StreamOpenFn>(dlsym(handle, "Synurang_Stream_VolvoxAiService_Open"));
        stream_send = reinterpret_cast<StreamSendFn>(dlsym(handle, "Synurang_Stream_Send"));
        stream_recv = reinterpret_cast<StreamRecvFn>(dlsym(handle, "Synurang_Stream_Recv"));
        stream_close_send = reinterpret_cast<StreamCloseSendFn>(dlsym(handle, "Synurang_Stream_CloseSend"));
        stream_close = reinterpret_cast<StreamCloseFn>(dlsym(handle, "Synurang_Stream_Close"));
        if (!invoke || !free_fn) throw std::runtime_error("missing Synurang FFI symbols");
        if (!stream_open || !stream_send || !stream_recv || !stream_close_send || !stream_close) {
            throw std::runtime_error("missing Synurang stream FFI symbols");
        }
        if (init) init();
    }

    ~FfiLibrary() {
        if (handle) dlclose(handle);
    }

    Bytes call(const std::string& method, const Bytes& request) {
        int resp_len = 0;
        const char* data = request.empty() ? nullptr : reinterpret_cast<const char*>(request.data());
        char* ptr = invoke(method.c_str(), data, ffi_size(request.size(), "request"), &resp_len);
        std::unique_ptr<char, FreeFn> owned_response(ptr, free_fn);
        if (!ptr && resp_len != 0) throw std::runtime_error("null FFI response");

        size_t len = resp_len < 0 ? static_cast<size_t>(-static_cast<int64_t>(resp_len))
                                  : static_cast<size_t>(resp_len);
        Bytes response;
        if (len > 0) response.assign(reinterpret_cast<uint8_t*>(ptr), reinterpret_cast<uint8_t*>(ptr) + len);

        if (resp_len < 0) {
            FfiErrorPayload err = decode_error(response);
            throw std::runtime_error("service error: " + err.message);
        }
        return response;
    }

    std::vector<Bytes> server_stream(const std::string& method, const Bytes& request) {
        uint64_t handle = stream_open(method.c_str());
        if (handle == 0) throw std::runtime_error("failed to open stream " + method);
        bool closed = false;
        try {
            const char* data = request.empty() ? nullptr : reinterpret_cast<const char*>(request.data());
            if (stream_send(handle, data, ffi_size(request.size(), "stream request")) != 0) {
                throw std::runtime_error("failed to send stream request");
            }
            stream_close_send(handle);

            std::vector<Bytes> events;
            while (true) {
                int resp_len = 0;
                int status = 0;
                char* ptr = stream_recv(handle, &resp_len, &status);
                std::unique_ptr<char, FreeFn> owned_response(ptr, free_fn);
                if (status == 1) {
                    break;
                }
                if (!ptr && resp_len != 0) throw std::runtime_error("null stream response");
                if (resp_len < 0) throw std::runtime_error("negative stream response length");
                Bytes payload;
                if (resp_len > 0) {
                    payload.assign(reinterpret_cast<uint8_t*>(ptr),
                                   reinterpret_cast<uint8_t*>(ptr) + static_cast<size_t>(resp_len));
                }
                if (status < 0) {
                    FfiErrorPayload err = decode_error(payload);
                    throw std::runtime_error("stream service error: " + err.message);
                }
                events.push_back(std::move(payload));
            }
            stream_close(handle);
            closed = true;
            return events;
        } catch (...) {
            if (!closed) stream_close(handle);
            throw;
        }
    }
};

static Tensor zero_input(std::string name, std::vector<int64_t> shape, int32_t dtype, size_t width) {
    Tensor tensor;
    tensor.name = std::move(name);
    tensor.shape = std::move(shape);
    tensor.dtype = dtype;
    tensor.data.assign(numel(tensor.shape) * width, 0);
    return tensor;
}

static void print_output_summary(const std::string& model, const Tensor& tensor) {
    if (tensor.dtype != DATA_TYPE_F32 || tensor.data.size() % 4 != 0) {
        throw std::runtime_error(model + " output " + tensor.name + " is not F32");
    }
    size_t n = tensor.data.size() / 4;
    size_t finite = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::isfinite(read_f32_le(&tensor.data[i * 4]))) ++finite;
    }
    std::cout << "[" << model << "] output " << tensor.name << " shape=[";
    for (size_t i = 0; i < tensor.shape.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << tensor.shape[i];
    }
    std::cout << "] n=" << n << " finite=" << finite << "\n";
    if (finite != n) throw std::runtime_error(model + " output " + tensor.name + " has non-finite values");
}

static size_t argmax_f32(const Bytes& data) {
    if (data.empty() || data.size() % 4 != 0) throw std::runtime_error("bad f32 payload");
    size_t n = data.size() / 4;
    size_t best = 0;
    float best_value = read_f32_le(data.data());
    for (size_t i = 1; i < n; ++i) {
        float value = read_f32_le(&data[i * 4]);
        if (value > best_value) {
            best = i;
            best_value = value;
        }
    }
    return best;
}

static std::vector<Detection> top_detections(const Tensor& boxes, const Tensor& scores, size_t limit) {
    if (boxes.dtype != DATA_TYPE_F32 || scores.dtype != DATA_TYPE_F32) throw std::runtime_error("detection outputs are not F32");
    if (boxes.shape.size() != 3 || scores.shape.size() != 3 || boxes.shape[2] != 4) {
        throw std::runtime_error("unexpected EfficientDet output shapes");
    }
    size_t anchors = static_cast<size_t>(scores.shape[1]);
    size_t classes = static_cast<size_t>(scores.shape[2]);
    if (static_cast<size_t>(boxes.shape[1]) != anchors) {
        throw std::runtime_error("boxes/scores anchor count mismatch");
    }

    std::vector<Detection> all;
    all.reserve(anchors);
    for (size_t anchor = 0; anchor < anchors; ++anchor) {
        int best_class = 0;
        float best_score = read_f32_le(&scores.data[(anchor * classes) * 4]);
        for (size_t cls = 1; cls < classes; ++cls) {
            float score = read_f32_le(&scores.data[(anchor * classes + cls) * 4]);
            if (score > best_score) {
                best_score = score;
                best_class = static_cast<int>(cls);
            }
        }
        const uint8_t* b = &boxes.data[anchor * 4 * 4];
        all.push_back(Detection{
            best_class,
            best_score,
            read_f32_le(b + 0),
            read_f32_le(b + 4),
            read_f32_le(b + 8),
            read_f32_le(b + 12),
        });
    }
    size_t keep = std::min(limit, all.size());
    std::partial_sort(all.begin(), all.begin() + keep, all.end(),
                      [](const Detection& a, const Detection& b) { return a.score > b.score; });
    all.resize(keep);
    return all;
}

static int32_t last_non_pad_token(const Bytes& token_i32, int32_t pad_token) {
    if (token_i32.size() % 4 != 0) throw std::runtime_error("tokens.i32 is not i32 data");
    size_t count = token_i32.size() / 4;
    for (size_t i = 0; i < count; ++i) {
        if (read_i32_le(&token_i32[i * 4]) == pad_token) {
            return i == 0 ? -1 : static_cast<int32_t>(i - 1);
        }
    }
    return count == 0 ? -1 : static_cast<int32_t>(count - 1);
}

static LoadModelResponse load_model(FfiLibrary& lib, const std::string& dir, const std::string& tokenizer = "") {
    Bytes resp = lib.call("/volvoxai.v1.VolvoxAiService/LoadModel",
                          encode_load_model_request(join_path(dir, "config.json"),
                                                    join_path(dir, "model.safetensors"),
                                                    tokenizer));
    LoadModelResponse load = decode_load_model_response(resp);
    if (load.model_id.empty()) throw std::runtime_error("LoadModel returned empty model_id");
    std::cout << "[" << dir << "] loaded " << load.model_id << " ops=" << load.info.num_ops
              << " tokenizer=" << (load.info.has_tokenizer ? "true" : "false") << "\n";
    return load;
}

static void unload_model(FfiLibrary& lib, const std::string& model_id) {
    lib.call("/volvoxai.v1.VolvoxAiService/UnloadModel", encode_model_ref(model_id));
}

static void run_efficientdet(FfiLibrary& lib, const std::string& root) {
    std::string model = "efficientdet_lite0_fp32";
    std::string dir = join_path(root, "models/" + model);
    LoadModelResponse load = load_model(lib, dir);
    std::vector<Tensor> inputs = {
        zero_input("input0", {1, 320, 320, 3}, DATA_TYPE_F32, 4),
    };
    Bytes resp = lib.call("/volvoxai.v1.VolvoxAiService/Run",
                          encode_run_request(load.model_id, inputs, {"boxes", "scores"}));
    RunResponse run = decode_run_response(resp);
    if (run.outputs.size() != 2) throw std::runtime_error(model + " expected boxes and scores outputs");
    for (const Tensor& output : run.outputs) print_output_summary(model, output);

    const Tensor* boxes = nullptr;
    const Tensor* scores = nullptr;
    for (const Tensor& output : run.outputs) {
        if (output.name == "boxes") boxes = &output;
        if (output.name == "scores") scores = &output;
    }
    if (!boxes || !scores) throw std::runtime_error(model + " missing boxes/scores outputs");
    std::vector<Detection> detections = top_detections(*boxes, *scores, 5);
    std::cout << "[" << model << "] top detections from zero image:\n";
    for (size_t i = 0; i < detections.size(); ++i) {
        const Detection& d = detections[i];
        std::cout << "  #" << (i + 1) << " class_id=" << d.class_id
                  << " score=" << d.score
                  << " box[y0,x0,y1,x1]=[" << d.y0 << "," << d.x0 << "," << d.y1 << "," << d.x1
                  << "]\n";
    }
    unload_model(lib, load.model_id);
}

static void run_tinystories(FfiLibrary& lib, const std::string& root) {
    std::string model = "tinystories_1m";
    std::string dir = join_path(root, "models/" + model);
    std::string vocab = join_path(dir, "vocab.bin");
    LoadModelResponse load = load_model(lib, dir, file_exists(vocab) ? vocab : "");

    Bytes token_data = read_file(join_path(dir, "tokens.i32"));
    Bytes position_data = read_file(join_path(dir, "positions.i32"));
    int32_t last_token = last_non_pad_token(token_data, 50256);
    if (last_token < 0) throw std::runtime_error("no non-pad token in tokens.i32");

    std::vector<Tensor> inputs = {
        Tensor{"tokens", {1, 256}, DATA_TYPE_I32, token_data},
        Tensor{"positions", {1, 256}, DATA_TYPE_I32, position_data},
    };
    Bytes resp = lib.call("/volvoxai.v1.VolvoxAiService/Run",
                          encode_run_request(load.model_id, inputs, {}, last_token));
    RunResponse run = decode_run_response(resp);
    if (run.outputs.size() != 1) throw std::runtime_error(model + " expected one logits output");
    print_output_summary(model, run.outputs[0]);

    size_t got = argmax_f32(run.outputs[0].data);
    std::cout << "[" << model << "] logits argmax=" << got << " last_token=" << last_token;
    std::string ref_path = join_path(dir, "output.f32");
    if (file_exists(ref_path)) {
        Bytes ref = read_file(ref_path);
        size_t expected = argmax_f32(ref);
        std::cout << " reference_argmax=" << expected;
        if (got != expected) throw std::runtime_error(model + " argmax differs from output.f32");
    }
    std::cout << "\n";

    std::string prompt = "Once upon a time, Lily";
    std::vector<Bytes> events = lib.server_stream(
        "/volvoxai.v1.VolvoxAiService/Generate",
        encode_generate_request(load.model_id, prompt, 24));
    std::string generated;
    int32_t generated_tokens = 0;
    int32_t finish_reason = 0;
    for (const Bytes& payload : events) {
        GenerateEvent event = decode_generate_event(payload);
        if (event.has_token) {
            generated += event.token.text;
        }
        if (event.has_done) {
            generated_tokens = event.done.num_tokens;
            finish_reason = event.done.reason;
        }
    }
    std::cout << "[" << model << "] generated text:\n";
    std::cout << "  prompt: " << prompt << "\n";
    std::cout << "  output: " << generated << "\n";
    std::cout << "  tokens: " << generated_tokens << " finish_reason=" << finish_reason << "\n";
    unload_model(lib, load.model_id);
}

int main(int argc, char** argv) {
    try {
        std::string lib_path = argc > 1 ? argv[1] : "runtime/target/release/libvolvoxai.so";
        std::string repo_root = argc > 2 ? argv[2] : ".";

        FfiLibrary lib(lib_path);
        Bytes ping = lib.call("/volvoxai.v1.VolvoxAiService/Ping", encode_empty_request());
        (void)ping;
        std::cout << "Loaded FFI server: " << lib_path << "\n";

        run_efficientdet(lib, repo_root);
        run_tinystories(lib, repo_root);
        std::cout << "C++ FFI example completed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
