// Standalone C++ FFI client for runtime/target/release/libvolvoxai.so.
//
// It loads the Synurang plugin server ABI with dlopen/dlsym, builds the minimal
// protobuf wire payloads needed for Ping/LoadModel/Run/UnloadModel, and executes:
//   * models/efficientdet_lite0_fp32
//   * models/tinystories_1m
//
// Build from repo root:
//   c++ -std=c++17 -O2 -Wall -Wextra runtime/examples/cpp_ffi_client.cpp -o runtime/target/examples/cpp_ffi_client -ldl
//
// Run from repo root:
//   ./runtime/target/examples/cpp_ffi_client runtime/target/release/libvolvoxai.so

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Bytes = std::vector<uint8_t>;

namespace pb {

static void put_varint(Bytes& out, uint64_t value) {
    while (value >= 0x80) {
        out.push_back(static_cast<uint8_t>(value | 0x80));
        value >>= 7;
    }
    out.push_back(static_cast<uint8_t>(value));
}

static void put_key(Bytes& out, int field, int wire) {
    put_varint(out, (static_cast<uint64_t>(field) << 3) | static_cast<uint64_t>(wire));
}

static void put_len(Bytes& out, int field, const Bytes& data) {
    put_key(out, field, 2);
    put_varint(out, data.size());
    out.insert(out.end(), data.begin(), data.end());
}

static void put_string(Bytes& out, int field, const std::string& value) {
    put_key(out, field, 2);
    put_varint(out, value.size());
    out.insert(out.end(), value.begin(), value.end());
}

static void put_bytes(Bytes& out, int field, const Bytes& value) {
    put_len(out, field, value);
}

static void put_i32(Bytes& out, int field, int32_t value) {
    put_key(out, field, 0);
    put_varint(out, static_cast<uint32_t>(value));
}

static void put_packed_i64(Bytes& out, int field, const std::vector<int64_t>& values) {
    Bytes packed;
    for (int64_t value : values) {
        put_varint(packed, static_cast<uint64_t>(value));
    }
    put_len(out, field, packed);
}

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
        out.assign(data + pos, data + pos + len);
        pos += static_cast<size_t>(len);
        return true;
    }

    bool read_string(std::string& out) {
        Bytes bytes;
        if (!read_len(bytes)) return false;
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

}  // namespace pb

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

static Bytes encode_model_paths(const std::string& config, const std::string& weights,
                                const std::string& tokenizer = "") {
    Bytes out;
    pb::put_string(out, 1, config);
    pb::put_string(out, 2, weights);
    if (!tokenizer.empty()) pb::put_string(out, 3, tokenizer);
    return out;
}

static Bytes encode_load_model_request(const std::string& config, const std::string& weights,
                                       const std::string& tokenizer = "") {
    Bytes out;
    pb::put_len(out, 1, encode_model_paths(config, weights, tokenizer));
    return out;
}

static Bytes encode_model_ref(const std::string& model_id) {
    Bytes out;
    pb::put_string(out, 1, model_id);
    return out;
}

static Bytes encode_tensor(const Tensor& tensor) {
    Bytes out;
    pb::put_string(out, 1, tensor.name);
    pb::put_packed_i64(out, 2, tensor.shape);
    pb::put_i32(out, 3, tensor.dtype);
    pb::put_bytes(out, 4, tensor.data);
    return out;
}

static Bytes encode_run_request(const std::string& model_id, const std::vector<Tensor>& inputs,
                                const std::vector<std::string>& output_names,
                                int32_t last_token = -1) {
    Bytes out;
    pb::put_string(out, 1, model_id);
    for (const Tensor& input : inputs) pb::put_len(out, 2, encode_tensor(input));
    for (const std::string& name : output_names) pb::put_string(out, 3, name);
    if (last_token >= 0) pb::put_i32(out, 4, last_token);
    return out;
}

static Bytes encode_generation_config(int32_t max_new_tokens, int32_t eos_token, int32_t pad_token) {
    Bytes out;
    pb::put_i32(out, 1, max_new_tokens);
    pb::put_i32(out, 2, eos_token);
    pb::put_i32(out, 3, pad_token);
    return out;
}

static Bytes encode_generate_request(const std::string& model_id, const std::string& prompt,
                                     int32_t max_new_tokens) {
    Bytes out;
    pb::put_string(out, 1, model_id);
    pb::put_string(out, 2, prompt);
    pb::put_len(out, 4, encode_generation_config(max_new_tokens, 50256, 50256));
    return out;
}

static FfiErrorPayload decode_error(const Bytes& data) {
    FfiErrorPayload err;
    pb::Cursor c(data);
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

static ModelInfo decode_model_info(const Bytes& data) {
    ModelInfo info;
    pb::Cursor c(data);
    while (!c.eof()) {
        uint64_t key = 0;
        if (!c.read_varint(key)) break;
        int field = static_cast<int>(key >> 3);
        int wire = static_cast<int>(key & 7);
        uint64_t value = 0;
        switch (field) {
            case 3:
                if (wire == 0 && c.read_varint(value)) info.num_ops = static_cast<int32_t>(value);
                else if (!c.skip(wire)) return info;
                break;
            case 5:
                if (wire == 0 && c.read_varint(value)) info.has_tokenizer = value != 0;
                else if (!c.skip(wire)) return info;
                break;
            default:
                if (!c.skip(wire)) return info;
                break;
        }
    }
    return info;
}

static LoadModelResponse decode_load_model_response(const Bytes& data) {
    LoadModelResponse resp;
    pb::Cursor c(data);
    while (!c.eof()) {
        uint64_t key = 0;
        if (!c.read_varint(key)) break;
        int field = static_cast<int>(key >> 3);
        int wire = static_cast<int>(key & 7);
        Bytes nested;
        switch (field) {
            case 1:
                if (wire == 2) c.read_string(resp.model_id);
                else if (!c.skip(wire)) return resp;
                break;
            case 2:
                if (wire == 2 && c.read_len(nested)) resp.info = decode_model_info(nested);
                else if (!c.skip(wire)) return resp;
                break;
            default:
                if (!c.skip(wire)) return resp;
                break;
        }
    }
    return resp;
}

static Tensor decode_tensor(const Bytes& data) {
    Tensor tensor;
    pb::Cursor c(data);
    while (!c.eof()) {
        uint64_t key = 0;
        if (!c.read_varint(key)) break;
        int field = static_cast<int>(key >> 3);
        int wire = static_cast<int>(key & 7);
        uint64_t value = 0;
        Bytes packed;
        pb::Cursor packed_cursor;
        switch (field) {
            case 1:
                if (wire == 2) c.read_string(tensor.name);
                else if (!c.skip(wire)) return tensor;
                break;
            case 2:
                if (wire == 0 && c.read_varint(value)) {
                    tensor.shape.push_back(static_cast<int64_t>(value));
                } else if (wire == 2 && c.read_len(packed)) {
                    packed_cursor = pb::Cursor(packed);
                    while (!packed_cursor.eof() && packed_cursor.read_varint(value)) {
                        tensor.shape.push_back(static_cast<int64_t>(value));
                    }
                } else if (!c.skip(wire)) {
                    return tensor;
                }
                break;
            case 3:
                if (wire == 0 && c.read_varint(value)) tensor.dtype = static_cast<int32_t>(value);
                else if (!c.skip(wire)) return tensor;
                break;
            case 4:
                if (wire == 2) c.read_len(tensor.data);
                else if (!c.skip(wire)) return tensor;
                break;
            default:
                if (!c.skip(wire)) return tensor;
                break;
        }
    }
    return tensor;
}

static RunResponse decode_run_response(const Bytes& data) {
    RunResponse resp;
    pb::Cursor c(data);
    while (!c.eof()) {
        uint64_t key = 0;
        if (!c.read_varint(key)) break;
        int field = static_cast<int>(key >> 3);
        int wire = static_cast<int>(key & 7);
        Bytes nested;
        if (field == 1 && wire == 2 && c.read_len(nested)) {
            resp.outputs.push_back(decode_tensor(nested));
        } else if (!c.skip(wire)) {
            break;
        }
    }
    return resp;
}

static TokenEvent decode_token(const Bytes& data) {
    TokenEvent token;
    pb::Cursor c(data);
    while (!c.eof()) {
        uint64_t key = 0;
        if (!c.read_varint(key)) break;
        int field = static_cast<int>(key >> 3);
        int wire = static_cast<int>(key & 7);
        uint64_t value = 0;
        switch (field) {
            case 1:
                if (wire == 0 && c.read_varint(value)) token.id = static_cast<int32_t>(value);
                else if (!c.skip(wire)) return token;
                break;
            case 2:
                if (wire == 2) c.read_string(token.text);
                else if (!c.skip(wire)) return token;
                break;
            case 3:
                if (wire == 0 && c.read_varint(value)) token.position = static_cast<int32_t>(value);
                else if (!c.skip(wire)) return token;
                break;
            default:
                if (!c.skip(wire)) return token;
                break;
        }
    }
    return token;
}

static DoneEvent decode_done(const Bytes& data) {
    DoneEvent done;
    pb::Cursor c(data);
    while (!c.eof()) {
        uint64_t key = 0;
        if (!c.read_varint(key)) break;
        int field = static_cast<int>(key >> 3);
        int wire = static_cast<int>(key & 7);
        uint64_t value = 0;
        switch (field) {
            case 1:
                if (wire == 2) c.read_string(done.text);
                else if (!c.skip(wire)) return done;
                break;
            case 2:
                if (wire == 0 && c.read_varint(value)) done.num_tokens = static_cast<int32_t>(value);
                else if (!c.skip(wire)) return done;
                break;
            case 3:
                if (wire == 0 && c.read_varint(value)) done.reason = static_cast<int32_t>(value);
                else if (!c.skip(wire)) return done;
                break;
            default:
                if (!c.skip(wire)) return done;
                break;
        }
    }
    return done;
}

static GenerateEvent decode_generate_event(const Bytes& data) {
    GenerateEvent event;
    pb::Cursor c(data);
    while (!c.eof()) {
        uint64_t key = 0;
        if (!c.read_varint(key)) break;
        int field = static_cast<int>(key >> 3);
        int wire = static_cast<int>(key & 7);
        Bytes nested;
        if (field == 1 && wire == 2 && c.read_len(nested)) {
            event.has_token = true;
            event.token = decode_token(nested);
        } else if (field == 2 && wire == 2 && c.read_len(nested)) {
            event.has_done = true;
            event.done = decode_done(nested);
        } else if (!c.skip(wire)) {
            break;
        }
    }
    return event;
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
        invoke = reinterpret_cast<InvokeFn>(dlsym(handle, "Synurang_Invoke_VolvoxAIService"));
        free_fn = reinterpret_cast<FreeFn>(dlsym(handle, "Synurang_Free"));
        init = reinterpret_cast<InitFn>(dlsym(handle, "VolvoxAI_Init"));
        stream_open = reinterpret_cast<StreamOpenFn>(dlsym(handle, "Synurang_Stream_VolvoxAIService_Open"));
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
        char* ptr = invoke(method.c_str(), data, static_cast<int>(request.size()), &resp_len);
        if (!ptr && resp_len != 0) throw std::runtime_error("null FFI response");

        size_t len = resp_len < 0 ? static_cast<size_t>(-resp_len) : static_cast<size_t>(resp_len);
        Bytes response;
        if (len > 0) response.assign(reinterpret_cast<uint8_t*>(ptr), reinterpret_cast<uint8_t*>(ptr) + len);
        if (ptr) free_fn(ptr);

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
            if (stream_send(handle, data, static_cast<int>(request.size())) != 0) {
                throw std::runtime_error("failed to send stream request");
            }
            stream_close_send(handle);

            std::vector<Bytes> events;
            while (true) {
                int resp_len = 0;
                int status = 0;
                char* ptr = stream_recv(handle, &resp_len, &status);
                if (status == 1) {
                    break;
                }
                if (!ptr && resp_len != 0) throw std::runtime_error("null stream response");
                Bytes payload;
                if (resp_len > 0) {
                    payload.assign(reinterpret_cast<uint8_t*>(ptr),
                                   reinterpret_cast<uint8_t*>(ptr) + static_cast<size_t>(resp_len));
                }
                if (ptr) free_fn(ptr);
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
    if (tensor.dtype != 1 || tensor.data.size() % 4 != 0) {
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
    if (boxes.dtype != 1 || scores.dtype != 1) throw std::runtime_error("detection outputs are not F32");
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
    Bytes resp = lib.call("/volvoxai.v1.VolvoxAIService/LoadModel",
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
    lib.call("/volvoxai.v1.VolvoxAIService/UnloadModel", encode_model_ref(model_id));
}

static void run_efficientdet(FfiLibrary& lib, const std::string& root) {
    std::string model = "efficientdet_lite0_fp32";
    std::string dir = join_path(root, "models/" + model);
    LoadModelResponse load = load_model(lib, dir);
    std::vector<Tensor> inputs = {
        zero_input("input0", {1, 320, 320, 3}, 1, 4),
    };
    Bytes resp = lib.call("/volvoxai.v1.VolvoxAIService/Run",
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
        Tensor{"tokens", {1, 256}, 10, token_data},
        Tensor{"positions", {1, 256}, 10, position_data},
    };
    Bytes resp = lib.call("/volvoxai.v1.VolvoxAIService/Run",
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
        "/volvoxai.v1.VolvoxAIService/Generate",
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
        Bytes ping = lib.call("/volvoxai.v1.VolvoxAIService/Ping", {});
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
