// C++ client for the Synurang VolvoxAI RuntimeService.
//
// Build from the repository root:
//   make -C examples cpp_ffi_example
//
// Run:
//   ./examples/target/bin/cpp_ffi_client runtime/target/release/libvolvoxai.so
//       models/efficientdet_lite0_fp32/graph.json
//       models/efficientdet_lite0_fp32/model.safetensors

#include <dlfcn.h>

#include "volvoxai_lite.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using InvokeFn = char* (*)(const char*, const char*, int, int*);
using FreeFn = void (*)(void*);

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
    return "unknown";
}

static void require_lite(SynurangLiteStatus status, const char* operation) {
    if (status != SYNURANG_LITE_OK) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 lite_status_name(status));
    }
}

static void assign_bytes(const SynurangLiteAllocator* allocator,
                         SynurangLiteBytes* field,
                         const void* data,
                         size_t size,
                         const char* operation) {
    require_lite(synurang_lite_bytes_assign(allocator, field, data, size), operation);
}

static void assign_string(const SynurangLiteAllocator* allocator,
                          SynurangLiteBytes* field,
                          const std::string& value,
                          const char* operation) {
    assign_bytes(allocator, field, value.data(), value.size(), operation);
}

static std::string as_string(const SynurangLiteBytes& value) {
    if (value.len == 0) return {};
    if (!value.data) throw std::runtime_error("message contains a null string");
    return std::string(reinterpret_cast<const char*>(value.data), value.len);
}

static size_t runtime_dtype_size(VolvoxaiRuntimeDataType dtype) {
    switch (dtype) {
        case VOLVOXAI_RUNTIME_DATA_TYPE_DATA_TYPE_U8:
        case VOLVOXAI_RUNTIME_DATA_TYPE_DATA_TYPE_I8:
            return 1;
        case VOLVOXAI_RUNTIME_DATA_TYPE_DATA_TYPE_I32:
        case VOLVOXAI_RUNTIME_DATA_TYPE_DATA_TYPE_F32:
            return 4;
        default:
            throw std::runtime_error("input uses an unsupported runtime dtype");
    }
}

static bool read_varint(const uint8_t* data,
                        size_t size,
                        size_t* position,
                        uint64_t* value) {
    uint64_t result = 0;
    for (unsigned shift = 0; shift < 70 && *position < size; shift += 7) {
        const uint8_t byte = data[(*position)++];
        result |= static_cast<uint64_t>(byte & 0x7f) << shift;
        if ((byte & 0x80) == 0) {
            *value = result;
            return true;
        }
    }
    return false;
}

static std::string decode_service_error(const uint8_t* data, size_t size) {
    size_t position = 0;
    while (position < size) {
        uint64_t tag = 0;
        if (!read_varint(data, size, &position, &tag)) break;
        const uint32_t field = static_cast<uint32_t>(tag >> 3);
        const uint8_t wire = static_cast<uint8_t>(tag & 7);
        if (wire == 0) {
            uint64_t ignored = 0;
            if (!read_varint(data, size, &position, &ignored)) break;
        } else if (wire == 2) {
            uint64_t length = 0;
            if (!read_varint(data, size, &position, &length) ||
                length > size - position) {
                break;
            }
            if (field == 2) {
                return std::string(reinterpret_cast<const char*>(data + position),
                                   static_cast<size_t>(length));
            }
            position += static_cast<size_t>(length);
        } else if (wire == 1 && size - position >= 8) {
            position += 8;
        } else if (wire == 5 && size - position >= 4) {
            position += 4;
        } else {
            break;
        }
    }
    return "service returned an undecodable error";
}

class RuntimeService {
   public:
    explicit RuntimeService(const std::string& library_path) {
        library_ = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!library_) throw std::runtime_error(dlerror());
        invoke_ = reinterpret_cast<InvokeFn>(
            dlsym(library_, "Synurang_Invoke_RuntimeService"));
        free_ = reinterpret_cast<FreeFn>(dlsym(library_, "Synurang_Free"));
        if (!invoke_ || !free_) {
            const char* error = dlerror();
            dlclose(library_);
            library_ = nullptr;
            throw std::runtime_error(error ? error : "missing Synurang symbols");
        }
    }

    RuntimeService(const RuntimeService&) = delete;
    RuntimeService& operator=(const RuntimeService&) = delete;

    ~RuntimeService() {
        if (library_) dlclose(library_);
    }

    template <typename Request, typename Response>
    Response call(const char* method,
                  const Request& request,
                  SynurangLiteStatus (*encode)(const Request*, uint8_t**, size_t*),
                  void (*init_response)(Response*),
                  SynurangLiteStatus (*decode)(Response*, const uint8_t*, size_t)) {
        uint8_t* encoded = nullptr;
        size_t encoded_size = 0;
        require_lite(encode(&request, &encoded, &encoded_size), "encode request");
        if (encoded_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
            synurang_lite_release(request._allocator, encoded);
            throw std::runtime_error("request is too large for the Synurang ABI");
        }

        int response_size = 0;
        char* response_data =
            invoke_(method, reinterpret_cast<const char*>(encoded),
                    static_cast<int>(encoded_size), &response_size);
        synurang_lite_release(request._allocator, encoded);
        const size_t payload_size =
            response_size < 0 ? static_cast<size_t>(-static_cast<int64_t>(response_size))
                              : static_cast<size_t>(response_size);
        if (payload_size != 0 && !response_data) {
            throw std::runtime_error("service returned a null response");
        }
        if (response_size < 0) {
            const std::string message = decode_service_error(
                reinterpret_cast<const uint8_t*>(response_data), payload_size);
            if (response_data) free_(response_data);
            throw std::runtime_error(message);
        }

        Response response;
        init_response(&response);
        const SynurangLiteStatus status =
            decode(&response, reinterpret_cast<const uint8_t*>(response_data),
                   payload_size);
        if (response_data) free_(response_data);
        require_lite(status, "decode response");
        return response;
    }

   private:
    void* library_ = nullptr;
    InvokeFn invoke_ = nullptr;
    FreeFn free_ = nullptr;
};

template <typename T>
static T* allocate_message(const SynurangLiteAllocator* allocator,
                           void (*init)(T*, const SynurangLiteAllocator*)) {
    allocator = synurang_lite_allocator_or_default(allocator);
    void* storage = allocator->allocate(allocator->context, sizeof(T));
    if (!storage) throw std::runtime_error("message allocation failed");
    std::memset(storage, 0, sizeof(T));
    T* message = static_cast<T*>(storage);
    init(message, allocator);
    return message;
}

static VolvoxaiRuntimeEmpty release_runtime(RuntimeService& service,
                                            const std::string& runtime_id) {
    VolvoxaiRuntimeRuntimeRef request;
    volvoxai_runtime_runtime_ref_init(&request);
    assign_string(request._allocator, &request.field_runtime_id, runtime_id,
                  "assign runtime_id");
    auto response = service.call(
        "/volvoxai.runtime.RuntimeService/ReleaseRuntime", request,
        volvoxai_runtime_runtime_ref_encode, volvoxai_runtime_empty_init,
        volvoxai_runtime_empty_decode);
    volvoxai_runtime_runtime_ref_free(&request);
    return response;
}

static VolvoxaiRuntimeEmpty release_model(RuntimeService& service,
                                          const std::string& model_id) {
    VolvoxaiRuntimeModelRef request;
    volvoxai_runtime_model_ref_init(&request);
    assign_string(request._allocator, &request.field_model_id, model_id,
                  "assign model_id");
    auto response = service.call(
        "/volvoxai.runtime.RuntimeService/ReleaseModel", request,
        volvoxai_runtime_model_ref_encode, volvoxai_runtime_empty_init,
        volvoxai_runtime_empty_decode);
    volvoxai_runtime_model_ref_free(&request);
    return response;
}

static VolvoxaiRuntimeEmpty release_compiled(RuntimeService& service,
                                             const std::string& compiled_id) {
    VolvoxaiRuntimeCompiledModelRef request;
    volvoxai_runtime_compiled_model_ref_init(&request);
    assign_string(request._allocator, &request.field_compiled_model_id, compiled_id,
                  "assign compiled_model_id");
    auto response = service.call(
        "/volvoxai.runtime.RuntimeService/ReleaseCompiledModel", request,
        volvoxai_runtime_compiled_model_ref_encode, volvoxai_runtime_empty_init,
        volvoxai_runtime_empty_decode);
    volvoxai_runtime_compiled_model_ref_free(&request);
    return response;
}

static VolvoxaiRuntimeEmpty release_context(RuntimeService& service,
                                            const std::string& context_id) {
    VolvoxaiRuntimeExecutionContextRef request;
    volvoxai_runtime_execution_context_ref_init(&request);
    assign_string(request._allocator, &request.field_context_id, context_id,
                  "assign context_id");
    auto response = service.call(
        "/volvoxai.runtime.RuntimeService/ReleaseExecutionContext", request,
        volvoxai_runtime_execution_context_ref_encode,
        volvoxai_runtime_empty_init, volvoxai_runtime_empty_decode);
    volvoxai_runtime_execution_context_ref_free(&request);
    return response;
}

static VolvoxaiRuntimeEmpty release_result(RuntimeService& service,
                                           const std::string& result_id) {
    VolvoxaiRuntimeResultRef request;
    volvoxai_runtime_result_ref_init(&request);
    assign_string(request._allocator, &request.field_result_id, result_id,
                  "assign result_id");
    auto response = service.call(
        "/volvoxai.runtime.RuntimeService/ReleaseResult", request,
        volvoxai_runtime_result_ref_encode, volvoxai_runtime_empty_init,
        volvoxai_runtime_empty_decode);
    volvoxai_runtime_result_ref_free(&request);
    return response;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <libvolvoxai.so> <graph.json> [weights.safetensors ...]\n";
        return 2;
    }

    try {
        RuntimeService service(argv[1]);

        VolvoxaiRuntimeCreateRuntimeRequest create_runtime;
        volvoxai_runtime_create_runtime_request_init(&create_runtime);
        create_runtime.field_cpu_threads = 1;
        auto runtime = service.call(
            "/volvoxai.runtime.RuntimeService/CreateRuntime", create_runtime,
            volvoxai_runtime_create_runtime_request_encode,
            volvoxai_runtime_runtime_handle_init,
            volvoxai_runtime_runtime_handle_decode);
        volvoxai_runtime_create_runtime_request_free(&create_runtime);
        const std::string runtime_id = as_string(runtime.field_runtime_id);
        std::cout << "runtime: " << runtime_id << "\n";

        VolvoxaiRuntimeLoadModelRequest load;
        volvoxai_runtime_load_model_request_init(&load);
        assign_string(load._allocator, &load.field_runtime_id, runtime_id,
                      "assign runtime_id");
        assign_string(load._allocator, &load.field_graph_path, argv[2],
                      "assign graph_path");
        const size_t weight_count = static_cast<size_t>(argc - 3);
        if (weight_count != 0) {
            load.field_weight_paths.data = static_cast<SynurangLiteBytes*>(
                load._allocator->allocate(load._allocator->context,
                                          weight_count * sizeof(SynurangLiteBytes)));
            if (!load.field_weight_paths.data) {
                throw std::runtime_error("weight path allocation failed");
            }
            std::memset(load.field_weight_paths.data, 0,
                        weight_count * sizeof(SynurangLiteBytes));
            load.field_weight_paths.len = weight_count;
            load.field_weight_paths.cap = weight_count;
            for (size_t index = 0; index < weight_count; ++index) {
                assign_string(load._allocator, &load.field_weight_paths.data[index],
                              argv[index + 3], "assign weight path");
            }
        }
        auto model = service.call(
            "/volvoxai.runtime.RuntimeService/LoadModel", load,
            volvoxai_runtime_load_model_request_encode,
            volvoxai_runtime_model_handle_init,
            volvoxai_runtime_model_handle_decode);
        volvoxai_runtime_load_model_request_free(&load);
        const std::string model_id = as_string(model.field_model_id);
        std::cout << "model: " << model_id << "\n";

        VolvoxaiRuntimeCompileModelRequest compile;
        volvoxai_runtime_compile_model_request_init(&compile);
        assign_string(compile._allocator, &compile.field_model_id, model_id,
                      "assign model_id");
        compile.field_policy = allocate_message<VolvoxaiRuntimeBackendPolicy>(
            compile._allocator, volvoxai_runtime_backend_policy_init_with_allocator);
        compile.field_policy->field_mode =
            VOLVOXAI_RUNTIME_BACKEND_POLICY_MODE_BACKEND_POLICY_MODE_REQUIRE;
        compile.field_policy->field_operator_fallback =
            VOLVOXAI_RUNTIME_OPERATOR_FALLBACK_OPERATOR_FALLBACK_FORBID;
        compile.field_policy->field_backends.data =
            static_cast<SynurangLiteBytes*>(compile._allocator->allocate(
                compile._allocator->context, sizeof(SynurangLiteBytes)));
        if (!compile.field_policy->field_backends.data) {
            throw std::runtime_error("backend policy allocation failed");
        }
        std::memset(compile.field_policy->field_backends.data, 0,
                    sizeof(SynurangLiteBytes));
        compile.field_policy->field_backends.len = 1;
        compile.field_policy->field_backends.cap = 1;
        assign_string(compile._allocator,
                      &compile.field_policy->field_backends.data[0], "cpu",
                      "assign backend");
        auto compiled = service.call(
            "/volvoxai.runtime.RuntimeService/CompileModel", compile,
            volvoxai_runtime_compile_model_request_encode,
            volvoxai_runtime_compiled_model_handle_init,
            volvoxai_runtime_compiled_model_handle_decode);
        volvoxai_runtime_compile_model_request_free(&compile);
        const std::string compiled_id =
            as_string(compiled.field_compiled_model_id);
        std::cout << "compiled: " << compiled_id << "\n";

        VolvoxaiRuntimeCreateExecutionContextRequest create_context;
        volvoxai_runtime_create_execution_context_request_init(&create_context);
        assign_string(create_context._allocator,
                      &create_context.field_compiled_model_id, compiled_id,
                      "assign compiled_model_id");
        create_context.field_decode_row_mode =
            VOLVOXAI_RUNTIME_DECODE_ROW_MODE_DECODE_ROW_MODE_DISABLED;
        create_context.field_require_incremental = 0;
        auto context = service.call(
            "/volvoxai.runtime.RuntimeService/CreateExecutionContext",
            create_context,
            volvoxai_runtime_create_execution_context_request_encode,
            volvoxai_runtime_execution_context_handle_init,
            volvoxai_runtime_execution_context_handle_decode);
        volvoxai_runtime_create_execution_context_request_free(&create_context);
        const std::string context_id = as_string(context.field_context_id);
        std::cout << "context: " << context_id << "\n";

        VolvoxaiRuntimeExecuteRequest execute;
        volvoxai_runtime_execute_request_init(&execute);
        assign_string(execute._allocator, &execute.field_context_id, context_id,
                      "assign context_id");
        if (context.field_inputs.len != 0) {
            execute.field_inputs.data =
                static_cast<VolvoxaiRuntimeTensor*>(execute._allocator->allocate(
                    execute._allocator->context,
                    context.field_inputs.len * sizeof(VolvoxaiRuntimeTensor)));
            if (!execute.field_inputs.data) {
                throw std::runtime_error("input batch allocation failed");
            }
            std::memset(execute.field_inputs.data, 0,
                        context.field_inputs.len * sizeof(VolvoxaiRuntimeTensor));
            execute.field_inputs.len = context.field_inputs.len;
            execute.field_inputs.cap = context.field_inputs.len;
        }
        for (size_t index = 0; index < context.field_inputs.len; ++index) {
            const VolvoxaiRuntimeTensorSpec& spec = context.field_inputs.data[index];
            VolvoxaiRuntimeTensor& input = execute.field_inputs.data[index];
            volvoxai_runtime_tensor_init_with_allocator(&input, execute._allocator);
            const std::string name = as_string(spec.field_name);
            assign_string(execute._allocator, &input.field_name, name,
                          "assign input name");
            input.field_dtype = spec.field_dtype;
            input.field_location =
                VOLVOXAI_RUNTIME_MEMORY_LOCATION_MEMORY_LOCATION_HOST;
            size_t elements = 1;
            if (spec.field_dimensions.len != 0) {
                input.field_shape.data =
                    static_cast<int64_t*>(execute._allocator->allocate(
                        execute._allocator->context,
                        spec.field_dimensions.len * sizeof(int64_t)));
                if (!input.field_shape.data) {
                    throw std::runtime_error("input shape allocation failed");
                }
                input.field_shape.len = spec.field_dimensions.len;
                input.field_shape.cap = spec.field_dimensions.len;
            }
            for (size_t axis = 0; axis < spec.field_dimensions.len; ++axis) {
                const VolvoxaiRuntimeDimensionConstraint& dimension =
                    spec.field_dimensions.data[axis];
                const int64_t extent = dimension.field_min;
                if (extent <= 0 || extent > dimension.field_max ||
                    dimension.field_multiple_of <= 0 ||
                    extent % dimension.field_multiple_of != 0 ||
                    static_cast<uint64_t>(extent) >
                        std::numeric_limits<size_t>::max() / elements) {
                    throw std::runtime_error("input has an invalid logical dimension");
                }
                input.field_shape.data[axis] = extent;
                elements *= static_cast<size_t>(extent);
            }
            const size_t element_size = runtime_dtype_size(spec.field_dtype);
            if (element_size > std::numeric_limits<size_t>::max() / elements) {
                throw std::runtime_error("input byte size overflows size_t");
            }
            std::vector<uint8_t> zeroes(elements * element_size, 0);
            assign_bytes(execute._allocator, &input.field_data,
                         zeroes.data(), zeroes.size(), "assign input data");
            std::cout << "  input " << name << ": " << zeroes.size() << " bytes\n";
        }
        auto result = service.call(
            "/volvoxai.runtime.RuntimeService/Execute", execute,
            volvoxai_runtime_execute_request_encode,
            volvoxai_runtime_execution_result_handle_init,
            volvoxai_runtime_execution_result_handle_decode);
        volvoxai_runtime_execute_request_free(&execute);
        const std::string result_id = as_string(result.field_result_id);
        std::cout << "result: " << result_id
                  << " (execution " << result.field_execution_id << ")\n";

        VolvoxaiRuntimeResultRef get_result;
        volvoxai_runtime_result_ref_init(&get_result);
        assign_string(get_result._allocator, &get_result.field_result_id, result_id,
                      "assign result_id");
        auto info = service.call(
            "/volvoxai.runtime.RuntimeService/GetResult", get_result,
            volvoxai_runtime_result_ref_encode, volvoxai_runtime_result_info_init,
            volvoxai_runtime_result_info_decode);
        volvoxai_runtime_result_ref_free(&get_result);

        for (size_t index = 0; index < info.field_outputs.len; ++index) {
            const std::string name = as_string(info.field_outputs.data[index].field_name);
            VolvoxaiRuntimeReadOutputRequest read;
            volvoxai_runtime_read_output_request_init(&read);
            assign_string(read._allocator, &read.field_result_id, result_id,
                          "assign result_id");
            assign_string(read._allocator, &read.field_name, name,
                          "assign output name");
            auto output = service.call(
                "/volvoxai.runtime.RuntimeService/ReadOutput", read,
                volvoxai_runtime_read_output_request_encode,
                volvoxai_runtime_tensor_init, volvoxai_runtime_tensor_decode);
            volvoxai_runtime_read_output_request_free(&read);
            std::cout << "  output " << name << ": " << output.field_data.len
                      << " stable host bytes\n";
            volvoxai_runtime_tensor_free(&output);
        }

        auto released_result = release_result(service, result_id);
        volvoxai_runtime_empty_free(&released_result);

        VolvoxaiRuntimeExecutionContextRef close_context;
        volvoxai_runtime_execution_context_ref_init(&close_context);
        assign_string(close_context._allocator, &close_context.field_context_id,
                      context_id, "assign context_id");
        auto close_context_report = service.call(
            "/volvoxai.runtime.RuntimeService/CloseExecutionContext", close_context,
            volvoxai_runtime_execution_context_ref_encode,
            volvoxai_runtime_operation_report_init,
            volvoxai_runtime_operation_report_decode);
        volvoxai_runtime_operation_report_free(&close_context_report);
        volvoxai_runtime_execution_context_ref_free(&close_context);
        auto released_context = release_context(service, context_id);
        volvoxai_runtime_empty_free(&released_context);

        auto released_compiled = release_compiled(service, compiled_id);
        volvoxai_runtime_empty_free(&released_compiled);
        auto released_model = release_model(service, model_id);
        volvoxai_runtime_empty_free(&released_model);

        VolvoxaiRuntimeRuntimeRef close_runtime;
        volvoxai_runtime_runtime_ref_init(&close_runtime);
        assign_string(close_runtime._allocator, &close_runtime.field_runtime_id,
                      runtime_id, "assign runtime_id");
        auto close_runtime_report = service.call(
            "/volvoxai.runtime.RuntimeService/CloseRuntime", close_runtime,
            volvoxai_runtime_runtime_ref_encode,
            volvoxai_runtime_operation_report_init,
            volvoxai_runtime_operation_report_decode);
        volvoxai_runtime_operation_report_free(&close_runtime_report);
        volvoxai_runtime_runtime_ref_free(&close_runtime);
        auto released_runtime = release_runtime(service, runtime_id);
        volvoxai_runtime_empty_free(&released_runtime);

        volvoxai_runtime_result_info_free(&info);
        volvoxai_runtime_execution_result_handle_free(&result);
        volvoxai_runtime_execution_context_handle_free(&context);
        volvoxai_runtime_compiled_model_handle_free(&compiled);
        volvoxai_runtime_model_handle_free(&model);
        volvoxai_runtime_runtime_handle_free(&runtime);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
