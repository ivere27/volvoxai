// End-to-end TinyReceiptVQA training using VolvoxAI protobuf/gRPC APIs only.
//
// Model creation, forward/backward, AdamW, safetensors, LoRA staging, task
// adapter export, and resumable checkpoints are VolvoxAI RPCs. Dataset record
// construction, image preprocessing, character tokenization, batching,
// scheduling, and metrics intentionally remain in this caller.

#include "volvoxai_grpc_ffi_client.h"
#include "volvoxai.pb.h"

#include "cJSON.h"
extern "C" {
#include "image_io.h"
}

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace api = volvoxai::v1;
using volvoxai_example::GrpcFfiClient;

namespace {

constexpr int kImageHeight = 320;
constexpr int kImageWidth = 672;
constexpr int kImageTokens = 210;
constexpr int kIgnoreId = -100;
constexpr int kF32 = api::DATA_TYPE_F32;
constexpr int kI32 = api::DATA_TYPE_I32;
constexpr const char* kPreprocessingFormat =
    "stb_image.rgb_luma.bilinear.672x320.minus_one_one.v1";
constexpr std::array<const char*, 8> kFamilies = {
    "phone", "address", "store", "item_row", "item_math", "item_lookup", "math", "other",
};

void AppendStringEntry(
        google::protobuf::RepeatedPtrField<api::StringEntry>* entries,
        std::string key, std::string value) {
    api::StringEntry* entry = entries->Add();
    entry->set_key(std::move(key));
    entry->set_value(std::move(value));
}

api::TensorShape* AppendTensorShapeEntry(
        google::protobuf::RepeatedPtrField<api::TensorShapeEntry>* entries,
        std::string key) {
    api::TensorShapeEntry* entry = entries->Add();
    entry->set_key(std::move(key));
    return entry->mutable_value();
}

struct Options {
#if defined(__APPLE__)
    fs::path library = "runtime/target/release/libvolvoxai.dylib";
#else
    fs::path library = "runtime/target/release/libvolvoxai.so";
#endif
    fs::path navercap_root;
    fs::path synth_root;
    fs::path real_annotations;
    fs::path real_images;
    fs::path train_records;
    fs::path validation_records;
    fs::path output = "runs/volvoxai_tiny_receipt_vqa";
    fs::path resume;
    // CPU is deliberately the safe default. GPU training must be selected
    // explicitly because a display GPU reset can terminate the desktop session.
    std::string backend = "cpu";
    // OpenGL and Metal do not expose Vulkan's fixed native arena setting. Keep
    // an explicit, caller-adjustable ceiling for the conservative graph and
    // gradient working-set estimate so a large batch is rejected before the
    // runtime allocates device resources.
    int gpu_memory_mb = 4096;
    std::string task_profile = "structured_qa";
    std::string target_mode = "rationale";
    std::string evaluation_routing = "auto";
    int synth_limit = 24000;
    int validation_synth_limit = 1200;
    int tasks_per_synth = 3;
    int real_repeat = 8;
    double real_validation_fraction = 0.1;
    int epochs = 12;
    // Keep the production effective batch of 24 without making one graph
    // instance retain 24 copies of every activation.  The physical batch is
    // fully configurable; accumulation supplies the safe default.
    int batch_size = 1;
    int max_steps = 0;
    int evaluation_batches = 20;
    int log_every = 100;
    int seed = 71;
    float learning_rate = 2.0e-4f;
    float weight_decay = 0.05f;
    int gradient_accumulation_steps = 24;
    int d_model = 320;
    int heads = 8;
    int encoder_layers = 6;
    int decoder_layers = 4;
    int ff_multiplier = 4;
    float dropout = 0.1f;
    int max_question_length = 192;
    int max_output_length = 192;  // Includes BOS in the Navercap convention.
    bool task_adapters = true;
    bool router = true;
    int adapter_bottleneck = 64;
    int lora_rank = 8;
    float lora_alpha = 16.0f;
    float lora_dropout = 0.05f;
    bool freeze_base_for_lora = false;
    bool smoke = false;
    bool dry_run = false;
    bool debug = false;
};

[[noreturn]] void Usage(const char* program, const std::string& error = {}) {
    if (!error.empty()) std::cerr << "error: " << error << "\n\n";
    std::cerr
        << "Usage: " << program << " [options]\n"
        << "  --lib PATH                 VolvoxAI shared library (.so or .dylib)\n"
        << "  --navercap-root PATH       Navercap checkout for raw-data mode\n"
        << "  --synth-root PATH          synth/data path (alternative to checkout root)\n"
        << "  --real-ann PATH            data/annotations path\n"
        << "  --real-img PATH            data/images path\n"
        << "  --train-records JSONL      consume pre-generated exact Navercap records\n"
        << "  --val-records JSONL        validation record JSONL\n"
        << "  --out PATH                 output/checkpoint directory\n"
        << "  --resume PATH              VolvoxAI training-checkpoint directory\n"
        << "  --backend cpu|vulkan|opengl|metal\n"
        << "  --gpu-memory-mb N          OpenGL/Metal conservative budget (default: 4096)\n"
        << "  --task-profile v1|structured_qa\n"
        << "  --target-mode rationale|answer\n"
        << "  --eval-routing auto|explicit\n"
        << "  --synth-limit N --val-synth-limit N --tasks-per-synth N (0 = all)\n"
        << "  --real-repeat N\n"
        << "  --real-val-frac F          real-receipt validation fraction\n"
        << "  --epochs N --batch-size N --max-steps N --eval-batches N\n"
        << "  --lr F --weight-decay F --accumulation-steps N\n"
        << "  --d-model N --heads N --enc-layers N --dec-layers N\n"
        << "  --max-q-len N --max-out-len N --dropout F\n"
        << "  --lora-r N --lora-alpha F --lora-dropout F\n"
        << "  --no-adapters --no-router --freeze-base-for-lora\n"
        << "  --smoke                    one tiny scaled train/eval step\n"
        << "  --debug                    enable native per-node diagnostics\n"
        << "  --dry-run                  build and validate the request without an RPC\n";
    std::exit(error.empty() ? 0 : 2);
}

int ParseInt(const std::string& value, const std::string& name) {
    size_t consumed = 0;
    long parsed = 0;
    try {
        parsed = std::stol(value, &consumed);
    } catch (...) {
        Usage("tiny_receipt_vqa_train", "invalid integer for " + name + ": " + value);
    }
    if (consumed != value.size() || parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
        Usage("tiny_receipt_vqa_train", "invalid integer for " + name + ": " + value);
    }
    return static_cast<int>(parsed);
}

int64_t ParseInt64(const std::string& value, const std::string& name) {
    size_t consumed = 0;
    long long parsed = 0;
    try {
        parsed = std::stoll(value, &consumed);
    } catch (...) {
        throw std::runtime_error("invalid integer for " + name + ": " + value);
    }
    if (consumed != value.size()) {
        throw std::runtime_error("invalid integer for " + name + ": " + value);
    }
    return static_cast<int64_t>(parsed);
}

double ParseDouble(const std::string& value, const std::string& name) {
    size_t consumed = 0;
    double parsed = 0;
    try {
        parsed = std::stod(value, &consumed);
    } catch (...) {
        throw std::runtime_error("invalid number for " + name + ": " + value);
    }
    if (consumed != value.size() || !std::isfinite(parsed)) {
        throw std::runtime_error("invalid number for " + name + ": " + value);
    }
    return parsed;
}

float ParseFloat(const std::string& value, const std::string& name) {
    size_t consumed = 0;
    float parsed = 0;
    try {
        parsed = std::stof(value, &consumed);
    } catch (...) {
        Usage("tiny_receipt_vqa_train", "invalid float for " + name + ": " + value);
    }
    if (consumed != value.size() || !std::isfinite(parsed)) {
        Usage("tiny_receipt_vqa_train", "invalid float for " + name + ": " + value);
    }
    return parsed;
}

Options ParseOptions(int argc, char** argv) {
    Options options;
    auto next = [&](int* index, const std::string& flag) -> std::string {
        if (*index + 1 >= argc) Usage(argv[0], flag + " requires a value");
        return argv[++*index];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--help" || flag == "-h") Usage(argv[0]);
        else if (flag == "--lib") options.library = next(&i, flag);
        else if (flag == "--navercap-root") options.navercap_root = next(&i, flag);
        else if (flag == "--synth-root") options.synth_root = next(&i, flag);
        else if (flag == "--real-ann") options.real_annotations = next(&i, flag);
        else if (flag == "--real-img") options.real_images = next(&i, flag);
        else if (flag == "--train-records") options.train_records = next(&i, flag);
        else if (flag == "--val-records") options.validation_records = next(&i, flag);
        else if (flag == "--out") options.output = next(&i, flag);
        else if (flag == "--resume") options.resume = next(&i, flag);
        else if (flag == "--backend") options.backend = next(&i, flag);
        else if (flag == "--gpu-memory-mb") options.gpu_memory_mb = ParseInt(next(&i, flag), flag);
        else if (flag == "--task-profile") options.task_profile = next(&i, flag);
        else if (flag == "--target-mode") options.target_mode = next(&i, flag);
        else if (flag == "--eval-routing") options.evaluation_routing = next(&i, flag);
        else if (flag == "--synth-limit") options.synth_limit = ParseInt(next(&i, flag), flag);
        else if (flag == "--val-synth-limit") options.validation_synth_limit = ParseInt(next(&i, flag), flag);
        else if (flag == "--tasks-per-synth") options.tasks_per_synth = ParseInt(next(&i, flag), flag);
        else if (flag == "--real-repeat") options.real_repeat = ParseInt(next(&i, flag), flag);
        else if (flag == "--real-val-frac") options.real_validation_fraction = ParseDouble(next(&i, flag), flag);
        else if (flag == "--epochs") options.epochs = ParseInt(next(&i, flag), flag);
        else if (flag == "--batch-size") options.batch_size = ParseInt(next(&i, flag), flag);
        else if (flag == "--max-steps") options.max_steps = ParseInt(next(&i, flag), flag);
        else if (flag == "--eval-batches") options.evaluation_batches = ParseInt(next(&i, flag), flag);
        else if (flag == "--log-every") options.log_every = ParseInt(next(&i, flag), flag);
        else if (flag == "--seed") options.seed = ParseInt(next(&i, flag), flag);
        else if (flag == "--lr") options.learning_rate = ParseFloat(next(&i, flag), flag);
        else if (flag == "--weight-decay") options.weight_decay = ParseFloat(next(&i, flag), flag);
        else if (flag == "--accumulation-steps") options.gradient_accumulation_steps = ParseInt(next(&i, flag), flag);
        else if (flag == "--d-model") options.d_model = ParseInt(next(&i, flag), flag);
        else if (flag == "--heads") options.heads = ParseInt(next(&i, flag), flag);
        else if (flag == "--enc-layers") options.encoder_layers = ParseInt(next(&i, flag), flag);
        else if (flag == "--dec-layers") options.decoder_layers = ParseInt(next(&i, flag), flag);
        else if (flag == "--max-q-len") options.max_question_length = ParseInt(next(&i, flag), flag);
        else if (flag == "--max-out-len") options.max_output_length = ParseInt(next(&i, flag), flag);
        else if (flag == "--dropout") options.dropout = ParseFloat(next(&i, flag), flag);
        else if (flag == "--adapter-bottleneck") options.adapter_bottleneck = ParseInt(next(&i, flag), flag);
        else if (flag == "--lora-r") options.lora_rank = ParseInt(next(&i, flag), flag);
        else if (flag == "--lora-alpha") options.lora_alpha = ParseFloat(next(&i, flag), flag);
        else if (flag == "--lora-dropout") options.lora_dropout = ParseFloat(next(&i, flag), flag);
        else if (flag == "--no-adapters") options.task_adapters = false;
        else if (flag == "--no-router") options.router = false;
        else if (flag == "--freeze-base-for-lora") options.freeze_base_for_lora = true;
        else if (flag == "--smoke") options.smoke = true;
        else if (flag == "--debug") options.debug = true;
        else if (flag == "--dry-run") options.dry_run = true;
        else Usage(argv[0], "unknown option: " + flag);
    }
    if (!options.navercap_root.empty()) {
        if (options.synth_root.empty()) options.synth_root = options.navercap_root / "synth/data";
        if (options.real_annotations.empty()) {
            options.real_annotations = options.navercap_root / "data/annotations";
        }
        if (options.real_images.empty()) options.real_images = options.navercap_root / "data/images";
    }
    const bool manifest_mode = !options.train_records.empty() || !options.validation_records.empty();
    if (!manifest_mode && (options.synth_root.empty() || options.real_annotations.empty() ||
                           options.real_images.empty())) {
        Usage(argv[0],
              "raw-data mode requires --navercap-root PATH, or all of --synth-root, "
              "--real-ann, and --real-img");
    }
    if (options.backend != "cpu" && options.backend != "vulkan" &&
        options.backend != "opengl" && options.backend != "metal") {
        Usage(argv[0], "backend must be cpu, vulkan, opengl, or metal");
    }
#if !defined(__APPLE__)
    if (options.backend == "metal" && !options.dry_run) {
        Usage(argv[0], "--backend metal requires an Apple runtime build (use --dry-run to inspect the graph here)");
    }
#endif
    if (options.task_profile != "v1" && options.task_profile != "structured_qa") {
        Usage(argv[0], "task-profile must be v1 or structured_qa");
    }
    if (options.target_mode != "rationale" && options.target_mode != "answer") {
        Usage(argv[0], "target-mode must be rationale or answer");
    }
    if (options.evaluation_routing != "auto" && options.evaluation_routing != "explicit") {
        Usage(argv[0], "eval-routing must be auto or explicit");
    }
    if (options.evaluation_routing == "auto" && !options.router) {
        options.evaluation_routing = "explicit";
    }
    if (options.synth_limit < 0 || options.validation_synth_limit < 0 ||
        options.tasks_per_synth < 0 || options.real_repeat <= 0 ||
        !std::isfinite(options.real_validation_fraction) ||
        options.real_validation_fraction < 0.0 || options.real_validation_fraction >= 1.0 ||
        options.batch_size <= 0 || options.epochs <= 0 || options.max_steps < 0 ||
        options.evaluation_batches < 0 || options.log_every <= 0 || options.seed < 0 ||
        options.d_model <= 0 || options.heads <= 0 || options.d_model % options.heads != 0 ||
        options.max_question_length <= 0 ||
        options.max_question_length > std::numeric_limits<int>::max() - kImageTokens ||
        options.max_output_length < 3 || options.encoder_layers <= 0 ||
        options.decoder_layers <= 0 || options.ff_multiplier <= 0 ||
        options.d_model > std::numeric_limits<int>::max() / options.ff_multiplier ||
        options.gradient_accumulation_steps <= 0 || options.lora_rank < 0 ||
        options.adapter_bottleneck < 0 || options.gpu_memory_mb <= 0) {
        Usage(argv[0], "invalid model/training dimensions");
    }
    if (!std::isfinite(options.learning_rate) || !(options.learning_rate > 0.0f)) {
        Usage(argv[0], "--lr must be finite and greater than zero");
    }
    if (!std::isfinite(options.weight_decay) || options.weight_decay < 0.0f) {
        Usage(argv[0], "--weight-decay must be finite and nonnegative");
    }
    if (!std::isfinite(options.dropout) || options.dropout < 0.0f || options.dropout >= 1.0f) {
        Usage(argv[0], "--dropout must be in [0, 1)");
    }
    if (!std::isfinite(options.lora_dropout) || options.lora_dropout < 0.0f ||
        options.lora_dropout >= 1.0f) {
        Usage(argv[0], "--lora-dropout must be in [0, 1)");
    }
    if (!std::isfinite(options.lora_alpha) ||
        (options.lora_rank > 0 && !(options.lora_alpha > 0.0f))) {
        Usage(argv[0], "--lora-alpha must be greater than zero when --lora-r is nonzero");
    }
    if (options.task_adapters && options.adapter_bottleneck <= 0) {
        Usage(argv[0], "--adapter-bottleneck must be greater than zero when adapters are enabled");
    }
    if (options.freeze_base_for_lora && options.lora_rank == 0) {
        Usage(argv[0], "--freeze-base-for-lora requires --lora-r > 0");
    }
    if (options.smoke) {
        options.synth_limit = std::min(options.synth_limit, 4);
        options.validation_synth_limit = std::min(options.validation_synth_limit, 2);
        options.tasks_per_synth = std::min(options.tasks_per_synth, 2);
        options.real_repeat = 1;
        options.epochs = 1;
        options.batch_size = 1;
        options.gradient_accumulation_steps = 1;
        options.max_steps = 1;
        options.evaluation_batches = 1;
        options.d_model = 32;
        options.heads = 4;
        options.encoder_layers = 1;
        options.decoder_layers = 1;
        options.max_question_length = 32;
        options.max_output_length = 32;
        options.adapter_bottleneck = std::min(options.adapter_bottleneck, 16);
        options.lora_rank = std::min(options.lora_rank, 4);
        options.dropout = 0.0f;
        options.lora_dropout = 0.0f;
    }
    return options;
}

struct JsonDeleter {
    void operator()(cJSON* value) const { cJSON_Delete(value); }
};
using Json = std::unique_ptr<cJSON, JsonDeleter>;

std::string ReadFile(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

Json ParseJson(const std::string& text, const std::string& context) {
    Json root(cJSON_ParseWithLength(text.data(), text.size()));
    if (!root) throw std::runtime_error("invalid JSON in " + context);
    return root;
}

cJSON* Member(cJSON* object, const char* key) {
    return object && cJSON_IsObject(object) ? cJSON_GetObjectItemCaseSensitive(object, key) : nullptr;
}

std::string JsonText(cJSON* object, const char* key) {
    cJSON* value = Member(object, key);
    if (cJSON_IsString(value) && value->valuestring) return value->valuestring;
    if (cJSON_IsNumber(value)) {
        std::ostringstream out;
        if (std::floor(value->valuedouble) == value->valuedouble) out << static_cast<int64_t>(value->valuedouble);
        else out << value->valuedouble;
        return out.str();
    }
    return {};
}

std::vector<uint32_t> Utf8Codepoints(const std::string& text) {
    std::vector<uint32_t> output;
    for (size_t i = 0; i < text.size();) {
        const uint8_t first = static_cast<uint8_t>(text[i]);
        uint32_t codepoint = 0xfffd;
        size_t length = 1;
        if (first < 0x80) {
            codepoint = first;
        } else if ((first & 0xe0) == 0xc0 && i + 1 < text.size()) {
            codepoint = first & 0x1f;
            length = 2;
        } else if ((first & 0xf0) == 0xe0 && i + 2 < text.size()) {
            codepoint = first & 0x0f;
            length = 3;
        } else if ((first & 0xf8) == 0xf0 && i + 3 < text.size()) {
            codepoint = first & 0x07;
            length = 4;
        }
        bool valid = length > 1;
        for (size_t j = 1; j < length && valid; ++j) {
            const uint8_t byte = static_cast<uint8_t>(text[i + j]);
            if ((byte & 0xc0) != 0x80) valid = false;
            else codepoint = (codepoint << 6) | (byte & 0x3f);
        }
        if (!valid && length > 1) {
            codepoint = 0xfffd;
            length = 1;
        }
        output.push_back(codepoint);
        i += length;
    }
    return output;
}

std::string EncodeUtf8(uint32_t codepoint) {
    std::string output;
    if (codepoint <= 0x7f) output.push_back(static_cast<char>(codepoint));
    else if (codepoint <= 0x7ff) {
        output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0xffff) {
        output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
        output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
    return output;
}

bool UnicodeSpace(uint32_t codepoint) {
    return codepoint == ' ' || codepoint == '\t' || codepoint == '\r' || codepoint == '\n' ||
           codepoint == 0x0b || codepoint == 0x0c || codepoint == 0x00a0 || codepoint == 0x3000;
}

std::string CleanText(const std::string& text) {
    std::string output;
    bool pending_space = false;
    for (uint32_t codepoint : Utf8Codepoints(text)) {
        if (UnicodeSpace(codepoint)) {
            pending_space = !output.empty();
        } else {
            if (pending_space) output.push_back(' ');
            pending_space = false;
            output += EncodeUtf8(codepoint);
        }
    }
    return output;
}

std::string DigitsOnly(const std::string& text) {
    std::string output;
    for (unsigned char byte : text) if (byte >= '0' && byte <= '9') output.push_back(static_cast<char>(byte));
    return output;
}

std::string CompactText(const std::string& text) {
    std::string output;
    for (uint32_t codepoint : Utf8Codepoints(CleanText(text))) {
        if (!UnicodeSpace(codepoint)) output += EncodeUtf8(codepoint);
    }
    return output;
}

std::vector<std::string> WordTokens(const std::string& text) {
    std::istringstream input(CleanText(text));
    std::vector<std::string> tokens;
    for (std::string token; input >> token;) tokens.push_back(std::move(token));
    return tokens;
}

std::string StreetNumber(const std::string& address) {
    std::smatch match;
    if (std::regex_search(address, match, std::regex("([0-9]+(?:[[:space:]]+[0-9]+)*)[[:space:]]*$"))) {
        return DigitsOnly(match[1]);
    }
    return {};
}

std::string NormalizeAddress(const std::string& address) {
    const std::string clean = CleanText(address);
    std::smatch match;
    if (!std::regex_search(clean, match, std::regex("([0-9]+(?:[[:space:]]+[0-9]+)*)$"))) return clean;
    return clean.substr(0, static_cast<size_t>(match.position(1))) + DigitsOnly(match[1]);
}

std::string AddressRoadPhrase(const std::string& address) {
    std::vector<std::string> tokens = WordTokens(NormalizeAddress(address));
    const std::string street = StreetNumber(address);
    if (!tokens.empty() && !street.empty() && DigitsOnly(tokens.back()) == street) tokens.pop_back();
    if (tokens.empty()) return {};
    std::string suffix = tokens.back();
    std::transform(suffix.begin(), suffix.end(), suffix.begin(),
                   [](unsigned char byte) { return static_cast<char>(std::tolower(byte)); });
    while (!suffix.empty() && suffix.back() == '.') suffix.pop_back();
    if (tokens.size() >= 2 &&
        (suffix == "st" || suffix == "street" || suffix == "rd" || suffix == "road" ||
         suffix == "ave" || suffix == "avenue" || suffix == "blvd")) {
        return tokens[tokens.size() - 2] + " " + tokens.back();
    }
    return tokens.back();
}

std::string LowerAscii(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char byte) {
        return static_cast<char>(std::tolower(byte));
    });
    return text;
}

bool HasKorean(const std::string& text) {
    for (uint32_t codepoint : Utf8Codepoints(text)) {
        if (codepoint >= 0xac00 && codepoint <= 0xd7a3) return true;
    }
    return false;
}

bool LooksAddressLikeText(const std::string& value) {
    const std::string text = CleanText(value);
    if (text.empty()) return false;
    const std::string lower = LowerAscii(text);
    static const std::regex english_road(
        "(^|[^a-z0-9])(st|street|rd|road|ave|avenue|blvd)\\.?([^a-z0-9]|$)");
    static const std::regex korean_numbered_road("(\ub85c|\uae38)[[:space:]]*[0-9]");
    return std::regex_search(lower, english_road) ||
           text.find("\uc8fc\uc18c") != std::string::npos ||
           text.find("\ub3c4\ub85c\uba85") != std::string::npos ||
           text.find("\ubc88\uc9c0") != std::string::npos ||
           std::regex_search(text, korean_numbered_road);
}

std::string Utf8Prefix(const std::string& text, size_t length) {
    const std::vector<uint32_t> codepoints = Utf8Codepoints(text);
    if (codepoints.size() < length) return {};
    std::string output;
    for (size_t index = 0; index < length; ++index) output += EncodeUtf8(codepoints[index]);
    return output;
}

std::vector<std::string> ItemLookupTerms(const std::string& name) {
    const std::string text = CleanText(name);
    if (text.empty()) return {};
    std::vector<std::string> terms = {text};
    const std::vector<std::string> tokens = WordTokens(text);
    if (!tokens.empty()) {
        terms.push_back(tokens.front());
        if (tokens.size() > 1) {
            terms.push_back(tokens.back());
            terms.push_back(tokens[0] + " " + tokens[1]);
        }
    }
    const std::string compact = CompactText(text);
    if (HasKorean(text)) {
        for (size_t length : {2u, 3u, 4u, 6u}) {
            const std::string prefix = Utf8Prefix(compact, length);
            if (!prefix.empty()) terms.push_back(prefix);
        }
    } else if (tokens.size() == 1) {
        for (size_t length : {4u, 6u}) {
            const std::string prefix = Utf8Prefix(compact, length);
            if (!prefix.empty()) terms.push_back(prefix);
        }
    }
    const std::string lowered = LowerAscii(text);
    if (lowered != text) terms.push_back(lowered);

    std::vector<std::string> output;
    std::unordered_set<std::string> seen;
    for (const std::string& term : terms) {
        const std::string clean = CleanText(term);
        const std::string key = LowerAscii(clean);
        if (!key.empty() && seen.insert(key).second) output.push_back(clean);
    }
    return output;
}

std::string StreetLikeNegativeTerm(bool korean, std::mt19937* random) {
    static const std::array<const char*, 5> korean_terms = {
        "\ub3c4\ub85c\uba85", "\ubc88\uc9c0", "\uc8fc\uc18c \ubc88\ud638", "\uae38 \ubc88\ud638", "\ub3c4\ub85c\uba85 \ubc88\ud638",
    };
    static const std::array<const char*, 6> english_terms = {
        "Something St.", "Sample ST", "Example Street", "Sample Road",
        "Address road number", "Location street number",
    };
    if (korean) {
        return korean_terms[std::uniform_int_distribution<size_t>(0, korean_terms.size() - 1)(*random)];
    }
    return english_terms[std::uniform_int_distribution<size_t>(0, english_terms.size() - 1)(*random)];
}

std::string Utf8CharAt(const std::string& text, int position, bool back = false) {
    const std::vector<uint32_t> codepoints = Utf8Codepoints(CompactText(text));
    if (position <= 0 || static_cast<size_t>(position) > codepoints.size()) return {};
    const size_t index = back ? codepoints.size() - static_cast<size_t>(position)
                              : static_cast<size_t>(position - 1);
    return EncodeUtf8(codepoints[index]);
}

std::string Target(const Options& options, const std::string& field, const std::string& value,
                   const std::string& operation, const std::string& answer) {
    const std::string answer_tag = "<answer>" + CleanText(answer) + "</answer>";
    if (options.target_mode == "answer") return answer_tag;
    return "<field>" + field + "</field><value>" + CleanText(value) + "</value><op>" +
           operation + "</op>" + answer_tag;
}

struct Record {
    fs::path image;
    std::string question;
    std::string target;
    std::string answer;
    std::string task;
    std::string field;
    std::string operation;
    std::string source;
    std::string receipt_id;
    std::string family;
};

int FamilyId(const std::string& family) {
    for (size_t i = 0; i < kFamilies.size(); ++i) if (family == kFamilies[i]) return static_cast<int>(i);
    return 7;
}

std::string TaskFamily(const std::string& task, const std::string& operation) {
    static const std::unordered_set<std::string> item_math_tasks = {
        "item_total_product", "item_total_sum", "item_quantity_sum", "item_max_total",
        "item_min_total", "item_first_two_sum", "item_first_two_diff", "phone_digit_sum",
        "phone_length", "store_name_length", "item_count", "item_unit_price_sum",
        "item_max_unit_price", "item_min_unit_price", "item_max_quantity", "item_min_quantity",
    };
    static const std::unordered_set<std::string> math_operations = {
        "product", "sum", "sum_quantity", "max_total", "min_total", "sum_first_2",
        "abs_diff_first_2", "digit_sum", "digit_count", "char_count", "count",
        "sum_unit_price", "max_unit_price", "min_unit_price", "max_quantity", "min_quantity",
    };
    if (item_math_tasks.count(task) || math_operations.count(operation)) {
        if (task.rfind("item", 0) == 0) return "item_math";
        const size_t separator = task.find('_');
        return separator == std::string::npos ? (task.empty() ? "math" : task)
                                               : task.substr(0, separator);
    }
    if (task.rfind("phone", 0) == 0) return "phone";
    if (task.rfind("address", 0) == 0) return "address";
    if (task.rfind("store", 0) == 0) return "store";
    if (task == "item_name" || task == "item_unit_price" || task == "item_quantity" ||
        task == "item_total" || task == "item_name_char") return "item_row";
    if (task.rfind("item_", 0) == 0) return "item_lookup";
    return task.empty() ? "other" : task;
}

void AddRecord(std::vector<Record>* records, const fs::path& image, std::string question,
               std::string target, std::string answer, std::string task, std::string field,
               std::string operation, std::string source, std::string receipt_id,
               std::string family) {
    question = CleanText(question);
    target = CleanText(target);
    answer = CleanText(answer);
    if (question.empty() || target.empty() || answer.empty()) return;
    records->push_back(Record{image, std::move(question), std::move(target), std::move(answer),
                              std::move(task), std::move(field), std::move(operation),
                              std::move(source), std::move(receipt_id), std::move(family)});
}

std::vector<fs::path> JsonFiles(const fs::path& directory) {
    std::vector<fs::path> paths;
    if (!fs::is_directory(directory)) return paths;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".json") paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

std::string PhoneQuestion(bool korean, bool back, int position) {
    if (korean) {
        return "가게 전화번호의 " + std::string(back ? "뒤에서 " : "앞에서 ") +
               std::to_string(position) + "번째 숫자는 무엇입니까?";
    }
    return "What is digit " + std::to_string(position) +
           std::string(back ? " from the back" : " from the front") +
           " of the store phone number?";
}

std::string AddressQuestion(bool korean) {
    return korean ? "영수증의 가게 주소에서 도로명 뒤 숫자는 무엇입니까?"
                  : "What is the street number in the store address?";
}

std::vector<Record> CandidateRecords(const Options& options, const fs::path& image, cJSON* answers,
                                     const std::string& source, const std::string& receipt_id,
                                     std::mt19937* random) {
    std::vector<Record> records;
    const bool korean = std::uniform_int_distribution<int>(0, 1)(*random) != 0;
    const std::string store = JsonText(answers, "store");
    const std::string address = NormalizeAddress(JsonText(answers, "address"));
    std::string street = JsonText(answers, "street_no");
    if (street.empty()) street = StreetNumber(address);
    std::string phone = DigitsOnly(JsonText(answers, "phone_digits"));
    if (phone.empty()) phone = DigitsOnly(JsonText(answers, "phone"));
    if (!street.empty()) {
        AddRecord(&records, image, AddressQuestion(korean),
                  Target(options, "addr", address, "street_no", street), street,
                  "address_number", "addr", "street_no", source, receipt_id, "address");
    }
    std::vector<std::pair<bool, int>> positions;
    for (int n = 1; n <= std::min<int>(4, phone.size()); ++n) {
        positions.emplace_back(false, n);
        positions.emplace_back(true, n);
    }
    std::shuffle(positions.begin(), positions.end(), *random);
    for (const auto& [back, position] : positions) {
        const size_t index = back ? phone.size() - static_cast<size_t>(position)
                                  : static_cast<size_t>(position - 1);
        const std::string answer(1, phone[index]);
        const std::string operation = std::string(back ? "back_" : "front_") + std::to_string(position);
        AddRecord(&records, image, PhoneQuestion(korean, back, position),
                  Target(options, "phone", phone, operation, answer), answer,
                  "phone_digit", "phone", operation, source, receipt_id, "phone");
    }
    if (options.task_profile == "v1") return records;

    auto add = [&](std::string question, const std::string& field, const std::string& value,
                   const std::string& operation, const std::string& answer,
                   const std::string& task, const std::string& family) {
        AddRecord(&records, image, std::move(question),
                  Target(options, field, value, operation, answer), answer, task, field,
                  operation, source, receipt_id, family);
    };
    auto question = [&](std::string korean_text, std::string english_text) {
        return korean ? std::move(korean_text) : std::move(english_text);
    };
    auto add_character_tasks = [&](const std::string& field, const std::string& value,
                                   const std::string& task, const std::string& family,
                                   const std::string& korean_subject,
                                   const std::string& english_subject, int maximum) {
        const int count = std::min<int>(maximum, Utf8Codepoints(CompactText(value)).size());
        for (int position = 1; position <= count; ++position) {
            add(question(korean_subject + "의 " + std::to_string(position) + "번째 글자는 무엇입니까?",
                         "What is character " + std::to_string(position) + " of the " +
                             english_subject + "?"),
                field, CompactText(value), "char_front_" + std::to_string(position),
                Utf8CharAt(value, position), task, family);
            add(question(korean_subject + "의 뒤에서 " + std::to_string(position) +
                             "번째 글자는 무엇입니까?",
                         "What is character " + std::to_string(position) + " from the end of the " +
                             english_subject + "?"),
                field, CompactText(value), "char_back_" + std::to_string(position),
                Utf8CharAt(value, position, true), task, family);
        }
    };

    if (!store.empty()) {
        add(question("영수증의 가게 이름은 무엇입니까?", "What is the store name?"),
            "store", store, "identity", store, "store_name", "store");
        add_character_tasks("store", store, "store_name_char", "store", "상호", "store name", 4);
        const std::vector<std::string> words = WordTokens(store);
        if (!words.empty()) {
            add(question("상호의 첫 단어는 무엇입니까?", "What is the first word of the store name?"),
                "store", store, "first_word", words.front(), "store_name_word", "store");
            add(question("상호의 마지막 단어는 무엇입니까?", "What is the last word of the store name?"),
                "store", store, "last_word", words.back(), "store_name_word", "store");
        }
        const int length = static_cast<int>(Utf8Codepoints(CompactText(store)).size());
        add(question("가게 이름은 몇 글자입니까?", "How many characters are in the store name?"),
            "store", CompactText(store), "char_count", std::to_string(length),
            "store_name_length", "store");
    }
    if (!phone.empty()) {
        add(question("가게 전화번호는 무엇입니까?", "What is the store phone number?"),
            "phone", phone, "identity", phone, "phone_number", "phone");
        for (int count : {2, 3, 4}) {
            if (static_cast<int>(phone.size()) < count) continue;
            add(question("전화번호 앞 " + std::to_string(count) + "자리는 무엇입니까?",
                         "What are the first " + std::to_string(count) + " phone digits?"),
                "phone", phone, "prefix_" + std::to_string(count), phone.substr(0, count),
                "phone_prefix", "phone");
            add(question("전화번호 끝 " + std::to_string(count) + "자리는 무엇입니까?",
                         "What are the last " + std::to_string(count) + " phone digits?"),
                "phone", phone, "suffix_" + std::to_string(count),
                phone.substr(phone.size() - static_cast<size_t>(count)), "phone_suffix", "phone");
        }
        add(question("전화번호 숫자는 모두 몇 개입니까?", "How many digits are in the phone number?"),
            "phone", phone, "digit_count", std::to_string(phone.size()), "phone_length", "phone");
        int sum = 0;
        for (char digit : phone) sum += digit - '0';
        add(question("전화번호 숫자를 모두 더하면 얼마입니까?", "What is the sum of the phone digits?"),
            "phone", phone, "digit_sum", std::to_string(sum), "phone_digit_sum", "phone");
    }
    if (!address.empty()) {
        add(question("가게 주소는 무엇입니까?", "What is the store address?"),
            "addr", address, "identity", address, "address_full", "address");
        add_character_tasks("addr", address, "address_char", "address", "주소", "store address", 3);
        const std::vector<std::string> words = WordTokens(address);
        if (!words.empty()) {
            add(question("주소의 첫 단어는 무엇입니까?", "What is the first word of the address?"),
                "addr", address, "first_token", words.front(), "address_first_token", "address");
        }
        if (words.size() >= 2) {
            add(question("주소의 두 번째 단어는 무엇입니까?", "What is the second word of the address?"),
                "addr", address, "second_token", words[1], "address_second_token", "address");
        }
        const std::string road = AddressRoadPhrase(address);
        if (!road.empty()) {
            add(question("주소의 도로명은 무엇입니까?", "What is the road name in the address?"),
                "addr", address, "road_name", road, "address_road_name", "address");
        }
    }

    struct ItemRow {
        std::string name;
        std::string unit_price;
        std::string quantity;
        std::string total;
    };
    auto number_text = [](const std::string& input) {
        const std::string clean = CleanText(input);
        if (clean.empty()) return std::string{};
        size_t consumed = 0;
        try {
            const int64_t value = std::stoll(clean, &consumed);
            if (consumed == clean.size()) return std::to_string(value);
        } catch (...) {
        }
        return DigitsOnly(clean);
    };
    auto integer = [](const std::string& value) -> int64_t {
        try { return std::stoll(value); } catch (...) { return 0; }
    };
    std::vector<ItemRow> item_rows;
    cJSON* items = Member(answers, "items");
    if (cJSON_IsArray(items)) {
        cJSON* item = nullptr;
        cJSON_ArrayForEach(item, items) {
            if (!cJSON_IsObject(item)) continue;
            item_rows.push_back(ItemRow{CleanText(JsonText(item, "name")),
                                        number_text(JsonText(item, "unit_price")),
                                        number_text(JsonText(item, "quantity")),
                                        number_text(JsonText(item, "total"))});
        }
    }
    if (!item_rows.empty()) {
        add(question("영수증의 품목 수는 몇 개입니까?", "How many item rows are on the receipt?"),
            "items", std::to_string(item_rows.size()), "count", std::to_string(item_rows.size()),
            "item_count", "item_math");
        for (size_t index = 0; index < std::min<size_t>(4, item_rows.size()); ++index) {
            const ItemRow& item = item_rows[index];
            const std::string position = std::to_string(index + 1);
            if (!item.name.empty()) {
                add(question(position + "번째 품목명은 무엇입니까?",
                             "What is the name of item " + position + "?"),
                    "item", item.name, "item_" + position + "_name", item.name,
                    "item_name", "item_row");
                add(question(position + "번째 품목명의 첫 글자는 무엇입니까?",
                             "What is the first character of item " + position + "'s name?"),
                    "item", CompactText(item.name), "item_" + position + "_name_char_front_1",
                    Utf8CharAt(item.name, 1), "item_name_char", "item_row");
            }
            const std::array<std::tuple<const char*, const std::string*, const char*>, 3> fields = {{
                {"unit_price", &item.unit_price, "item_unit_price"},
                {"quantity", &item.quantity, "item_quantity"},
                {"total", &item.total, "item_total"},
            }};
            for (const auto& [key, value, task] : fields) {
                if (value->empty()) continue;
                add(question(position + "번째 품목의 " +
                                 std::string(key == std::string("quantity") ? "수량" :
                                             key == std::string("total") ? "총합" : "가격") +
                                 "은 얼마입니까?",
                             "What is item " + position + "'s " +
                                 std::string(key == std::string("quantity") ? "quantity" :
                                             key == std::string("total") ? "total amount" : "unit price") + "?"),
                    "item", *value, "item_" + position + "_" + key, *value, task, "item_row");
            }
            if (!item.unit_price.empty() && !item.quantity.empty() && !item.total.empty()) {
                add(question(position + "번째 품목의 가격 곱하기 수량은 얼마입니까?",
                             "What is unit price times quantity for item " + position + "?"),
                    "item", item.unit_price + "*" + item.quantity, "product", item.total,
                    "item_total_product", "item_math");
            }
        }

        std::vector<const ItemRow*> named_items;
        for (size_t index = 0; index < std::min<size_t>(6, item_rows.size()); ++index) {
            const ItemRow& item = item_rows[index];
            if (!item.name.empty() && !LooksAddressLikeText(item.name)) named_items.push_back(&item);
        }
        for (const ItemRow* item : named_items) {
            std::vector<std::string> terms;
            for (const std::string& term : ItemLookupTerms(item->name)) {
                if (!LooksAddressLikeText(term)) terms.push_back(term);
            }
            if (terms.empty()) continue;
            const std::string& term = terms[
                std::uniform_int_distribution<size_t>(0, terms.size() - 1)(*random)];
            if (!item->unit_price.empty()) {
                add(question(term + " 가격은 얼마입니까?", "What is the price of " + term + "?"),
                    "item", item->name, "lookup_price", item->unit_price,
                    "item_lookup_price_by_name", "item_lookup");
            }
            if (!item->quantity.empty()) {
                add(question(term + " 수량은 몇 개입니까?", "What is the quantity of " + term + "?"),
                    "item", item->name, "lookup_quantity", item->quantity,
                    "item_lookup_quantity_by_name", "item_lookup");
            }
            if (!item->total.empty()) {
                add(question(term + " 총합은 얼마입니까?", "What is the total amount for " + term + "?"),
                    "item", item->name, "lookup_total", item->total,
                    "item_lookup_total_by_name", "item_lookup");
            }
        }
        if (!named_items.empty()) {
            const int absent_id = std::uniform_int_distribution<int>(100, 999)(*random);
            const std::string absent = korean ? "없는상품" + std::to_string(absent_id)
                                              : "Missing Item " + std::to_string(absent_id);
            add(question(absent + " 가격은 얼마입니까?", "What is the price of " + absent + "?"),
                "item", "none", "absent", "없음", "item_lookup_absent", "item_lookup");

            // Address-shaped negatives prevent road text from being learned as
            // an item merely because it is another prominent receipt row.
            std::vector<std::string> street_like_terms = {
                StreetLikeNegativeTerm(korean, random),
            };
            const std::string road = AddressRoadPhrase(address);
            if (!road.empty()) {
                street_like_terms.push_back(road);
                if (!street.empty()) street_like_terms.push_back(road + " " + street);
            }
            std::unordered_set<std::string> seen_street_like;
            for (const std::string& raw_term : street_like_terms) {
                const std::string term = CleanText(raw_term);
                if (term.empty() || !seen_street_like.insert(LowerAscii(term)).second) continue;
                add(question(term + " 가격은 얼마입니까?", "What is the price of " + term + "?"),
                    "item", "none", "absent", "없음", "item_lookup_absent", "item_lookup");
            }
        }

        auto add_metric_tasks = [&](const std::string& metric, const std::string& amount_task,
                                    const std::string& sum_operation,
                                    const std::string& sum_task) {
            std::vector<std::pair<std::string, std::string>> rows;
            for (const ItemRow& item : item_rows) {
                const std::string* value = metric == "total" ? &item.total
                    : metric == "unit_price" ? &item.unit_price : &item.quantity;
                if (!item.name.empty() && !value->empty()) rows.emplace_back(item.name, *value);
            }
            if (rows.empty()) return;
            std::string values;
            int64_t sum = 0;
            for (const auto& [name, value] : rows) {
                if (!values.empty()) values += ',';
                values += value;
                sum += integer(value);
            }
            add(question(metric == "quantity" ? "품목 수량을 모두 더하면 몇 개입니까?" :
                             metric == "unit_price" ? "품목 가격을 모두 더하면 얼마입니까?" :
                                                      "품목 총합을 모두 더하면 얼마입니까?",
                         metric == "quantity" ? "What is the sum of item quantities?" :
                             metric == "unit_price" ? "What is the sum of item unit prices?" :
                                                      "What is the sum of item totals?"),
                "items", values, sum_operation, std::to_string(sum), sum_task, "item_math");

            std::map<std::string, int> counts;
            for (const auto& row : rows) ++counts[row.second];
            std::vector<const std::pair<std::string, std::string>*> unique_rows;
            for (const auto& row : rows) {
                if (counts[row.second] == 1) unique_rows.push_back(&row);
            }
            if (!unique_rows.empty()) {
                const auto& [name, value] = *unique_rows[
                    std::uniform_int_distribution<size_t>(0, unique_rows.size() - 1)(*random)];
                add(question(value + " 값인 품목명은 무엇입니까?",
                             "Which item has " + metric + " " + value + "?"),
                    "items", value + ":" + name, "lookup_" + metric + "_name", name,
                    "item_lookup_name_by_" + metric, "item_lookup");
            }
            for (bool largest : {true, false}) {
                auto selected = rows.begin();
                for (auto candidate = std::next(rows.begin()); candidate != rows.end(); ++candidate) {
                    if ((largest && integer(candidate->second) > integer(selected->second)) ||
                        (!largest && integer(candidate->second) < integer(selected->second))) {
                        selected = candidate;
                    }
                }
                const std::string operation = std::string(largest ? "max_" : "min_") + metric;
                const std::string task = std::string("item_") + (largest ? "max_" : "min_") + amount_task;
                add(question(largest ? "가장 큰 값은 얼마입니까?" : "가장 작은 값은 얼마입니까?",
                             largest ? "What is the largest value?" : "What is the smallest value?"),
                    "items", values, operation, selected->second, task, "item_math");
                std::string pairs;
                for (const auto& [name, value] : rows) {
                    if (!pairs.empty()) pairs += ';';
                    pairs += value + ':' + name;
                }
                if (Utf8Codepoints(Target(options, "items", pairs, operation + "_name",
                                           selected->first)).size() > 180) {
                    pairs = selected->second + ':' + selected->first;
                }
                add(question(largest ? "가장 큰 값의 품목명은 무엇입니까?"
                                     : "가장 작은 값의 품목명은 무엇입니까?",
                             largest ? "Which item has the largest value?"
                                     : "Which item has the smallest value?"),
                    "items", pairs, operation + "_name", selected->first, task + "_name", "item_lookup");
            }
        };
        add_metric_tasks("total", "total", "sum", "item_total_sum");
        add_metric_tasks("unit_price", "unit_price", "sum_unit_price", "item_unit_price_sum");
        add_metric_tasks("quantity", "quantity", "sum_quantity", "item_quantity_sum");

        std::vector<std::string> totals;
        for (const ItemRow& item : item_rows) {
            if (!item.name.empty() && !item.total.empty()) totals.push_back(item.total);
        }
        if (totals.size() >= 2) {
            add(question("첫 두 품목 총합을 더하면 얼마입니까?",
                         "What is the sum of the first two item totals?"),
                "items", totals[0] + "," + totals[1], "sum_first_2",
                std::to_string(integer(totals[0]) + integer(totals[1])),
                "item_first_two_sum", "item_math");
            add(question("첫 두 품목 총합의 차이는 얼마입니까?",
                         "What is the difference between the first two item totals?"),
                "items", totals[0] + "," + totals[1], "abs_diff_first_2",
                std::to_string(std::llabs(integer(totals[0]) - integer(totals[1]))),
                "item_first_two_diff", "item_math");
        }
    }
    return records;
}

std::vector<Record> StratifiedSample(std::vector<Record> candidates, int limit, std::mt19937* random) {
    if (limit <= 0 || static_cast<int>(candidates.size()) <= limit) return candidates;
    std::array<std::vector<Record>, 8> buckets;
    for (Record& record : candidates) buckets[FamilyId(record.family)].push_back(std::move(record));
    for (auto& bucket : buckets) std::shuffle(bucket.begin(), bucket.end(), *random);
    std::vector<Record> output;
    for (size_t round = 0; static_cast<int>(output.size()) < limit; ++round) {
        bool added = false;
        for (auto& bucket : buckets) {
            if (round < bucket.size()) {
                output.push_back(std::move(bucket[round]));
                added = true;
                if (static_cast<int>(output.size()) == limit) break;
            }
        }
        if (!added) break;
    }
    return output;
}

std::vector<Record> LoadSynthetic(const Options& options, const std::string& split, int limit) {
    std::mt19937 random(options.seed + (split == "train" ? 0 : 1009));
    std::vector<fs::path> annotations = JsonFiles(options.synth_root / split / "ann");
    if (limit > 0 && static_cast<int>(annotations.size()) > limit) annotations.resize(limit);
    std::vector<Record> records;
    for (const fs::path& annotation : annotations) {
        const fs::path image = options.synth_root / split / "img" / (annotation.stem().string() + ".jpg");
        if (!fs::exists(image)) continue;
        Json root = ParseJson(ReadFile(annotation), annotation.string());
        cJSON* answers = Member(root.get(), "answers");
        if (!cJSON_IsObject(answers)) continue;
        std::vector<Record> candidates = CandidateRecords(
            options, fs::absolute(image), answers, "synth", annotation.stem().string(), &random);
        if (options.task_profile == "v1") {
            // CandidateRecords emits the address first and already shuffles
            // phone positions, matching train.py's address-plus-phone-cap
            // policy instead of allowing family stratification to displace
            // the address when the cap is small.
            if (options.tasks_per_synth > 0 &&
                static_cast<int>(candidates.size()) > options.tasks_per_synth) {
                candidates.resize(static_cast<size_t>(options.tasks_per_synth));
            }
        } else {
            candidates = StratifiedSample(std::move(candidates), options.tasks_per_synth, &random);
        }
        records.insert(records.end(), std::make_move_iterator(candidates.begin()),
                       std::make_move_iterator(candidates.end()));
    }
    return records;
}

bool ContainsAny(const std::string& text, std::initializer_list<const char*> needles) {
    for (const char* needle : needles) if (text.find(needle) != std::string::npos) return true;
    return false;
}

std::string RealPhoneOperation(const std::string& question) {
    std::smatch match;
    if (std::regex_search(question, match, std::regex("(?:back|end|right|뒤에서|뒷자리|끝자리)[^0-9]*([0-9]+)",
                                                      std::regex::icase))) {
        return "back_" + match[1].str();
    }
    if (std::regex_search(question, match, std::regex("([0-9]+)"))) return "front_" + match[1].str();
    static const std::array<const char*, 8> words = {
        "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
    };
    std::string lower = question;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
    for (size_t i = 0; i < words.size(); ++i) {
        if (lower.find(words[i]) != std::string::npos) {
            return (ContainsAny(lower, {"back", "end", "last"}) ? "back_" : "front_") +
                   std::to_string(i + 1);
        }
    }
    return "phone_digit";
}

std::vector<Record> LoadReal(const Options& options, const std::string& split) {
    std::vector<fs::path> annotations = JsonFiles(options.real_annotations);
    std::mt19937 random(options.seed);
    std::shuffle(annotations.begin(), annotations.end(), random);
    const size_t validation_count = annotations.empty()
        ? 0
        : std::max<size_t>(1, static_cast<size_t>(annotations.size() * options.real_validation_fraction));
    const auto begin = split == "val" ? annotations.begin() : annotations.begin() + validation_count;
    const auto end = split == "val" ? annotations.begin() + validation_count : annotations.end();
    std::vector<Record> records;
    for (auto it = begin; it != end; ++it) {
        const fs::path& annotation = *it;
        const fs::path image = options.real_images / (annotation.stem().string() + ".jpg");
        if (!fs::exists(image)) continue;
        Json root = ParseJson(ReadFile(annotation), annotation.string());
        const std::string question = CleanText(JsonText(root.get(), "question"));
        const std::string answer = CleanText(JsonText(root.get(), "answer"));
        cJSON* receipt = Member(root.get(), "receipt");
        cJSON* store = Member(receipt, "store");
        if (options.task_profile == "v1") {
            if (question.empty() || answer.empty()) continue;
            if (ContainsAny(question, {"phone number", "telephone", "전화번호"})) {
                const std::string phone = DigitsOnly(JsonText(store, "phone"));
                const std::string operation = RealPhoneOperation(question);
                AddRecord(&records, fs::absolute(image), question,
                          Target(options, "phone", phone, operation, answer), answer,
                          "phone_digit", "phone", operation, "real", annotation.stem().string(), "phone");
            } else if (ContainsAny(question, {"location", "address", "street", "가게 위치", "주소"})) {
                const std::string address = JsonText(store, "address");
                AddRecord(&records, fs::absolute(image), question,
                          Target(options, "addr", address, "street_no", answer), answer,
                          "address_number", "addr", "street_no", "real", annotation.stem().string(),
                          "address");
            }
        } else {
            // CandidateRecords reads an object without taking ownership. Build a
            // compact temporary object from the real receipt fields.
            Json answers(cJSON_CreateObject());
            cJSON_AddStringToObject(answers.get(), "store", JsonText(store, "name").c_str());
            cJSON_AddStringToObject(answers.get(), "address", JsonText(store, "address").c_str());
            cJSON_AddStringToObject(answers.get(), "street_no", StreetNumber(JsonText(store, "address")).c_str());
            cJSON_AddStringToObject(answers.get(), "phone_digits", DigitsOnly(JsonText(store, "phone")).c_str());
            cJSON* items = Member(receipt, "items");
            if (items) cJSON_AddItemReferenceToObject(answers.get(), "items", items);
            std::vector<Record> candidates = CandidateRecords(
                options, fs::absolute(image), answers.get(), "real", annotation.stem().string(), &random);
            records.insert(records.end(), std::make_move_iterator(candidates.begin()),
                           std::make_move_iterator(candidates.end()));
        }
    }
    if (split == "train" && options.real_repeat > 1) {
        const std::vector<Record> original = records;
        for (int repeat = 1; repeat < options.real_repeat; ++repeat) {
            records.insert(records.end(), original.begin(), original.end());
        }
        std::shuffle(records.begin(), records.end(), random);
    }
    return records;
}

std::vector<Record> LoadRecordJsonl(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open record JSONL " + path.string());
    std::vector<Record> records;
    std::string line;
    size_t line_number = 0;
    while (std::getline(stream, line)) {
        ++line_number;
        if (line.empty()) continue;
        Json root = ParseJson(line, path.string() + ":" + std::to_string(line_number));
        Record record;
        const std::string image_text = JsonText(root.get(), "image");
        record.image = image_text;
        if (record.image.is_relative()) record.image = path.parent_path() / record.image;
        record.image = fs::absolute(record.image).lexically_normal();
        record.question = CleanText(JsonText(root.get(), "question"));
        record.target = CleanText(JsonText(root.get(), "target"));
        record.answer = CleanText(JsonText(root.get(), "answer"));
        record.task = JsonText(root.get(), "task");
        record.field = JsonText(root.get(), "field");
        record.operation = JsonText(root.get(), "op");
        record.source = JsonText(root.get(), "source");
        record.receipt_id = JsonText(root.get(), "receipt_id");
        record.family = JsonText(root.get(), "family");
        if (record.family.empty()) record.family = TaskFamily(record.task, record.operation);
        if (image_text.empty() || record.question.empty() || record.target.empty() ||
            record.answer.empty() || record.task.empty() || record.operation.empty()) {
            throw std::runtime_error("incomplete Navercap record in " + path.string() + ":" +
                                     std::to_string(line_number));
        }
        records.push_back(std::move(record));
    }
    return records;
}

struct Dataset {
    std::vector<Record> train;
    std::vector<Record> validation;
};

Dataset LoadDataset(const Options& options) {
    Dataset dataset;
    if (!options.train_records.empty() || !options.validation_records.empty()) {
        if (options.train_records.empty() || options.validation_records.empty()) {
            throw std::runtime_error("--train-records and --val-records must be supplied together");
        }
        dataset.train = LoadRecordJsonl(options.train_records);
        dataset.validation = LoadRecordJsonl(options.validation_records);
    } else {
        const std::array<std::pair<const char*, fs::path>, 6> required_directories = {{
            {"synthetic train annotations", options.synth_root / "train/ann"},
            {"synthetic train images", options.synth_root / "train/img"},
            {"synthetic validation annotations", options.synth_root / "val/ann"},
            {"synthetic validation images", options.synth_root / "val/img"},
            {"real annotations", options.real_annotations},
            {"real images", options.real_images},
        }};
        for (const auto& [label, path] : required_directories) {
            if (!fs::is_directory(path)) {
                throw std::runtime_error(std::string(label) + " directory does not exist: " +
                                         path.string());
            }
        }
        dataset.train = LoadSynthetic(options, "train", options.synth_limit);
        std::vector<Record> real_train = LoadReal(options, "train");
        dataset.train.insert(dataset.train.end(), std::make_move_iterator(real_train.begin()),
                             std::make_move_iterator(real_train.end()));
        dataset.validation = LoadSynthetic(options, "val", options.validation_synth_limit);
        std::vector<Record> real_validation = LoadReal(options, "val");
        dataset.validation.insert(dataset.validation.end(),
                                  std::make_move_iterator(real_validation.begin()),
                                  std::make_move_iterator(real_validation.end()));
    }
    std::mt19937 train_random(options.seed);
    std::mt19937 validation_random(options.seed + 1);
    std::shuffle(dataset.train.begin(), dataset.train.end(), train_random);
    std::shuffle(dataset.validation.begin(), dataset.validation.end(), validation_random);
    if (dataset.train.size() < static_cast<size_t>(options.batch_size)) {
        throw std::runtime_error("not enough training records for one fixed-shape batch");
    }
    if (dataset.validation.empty()) throw std::runtime_error("validation dataset is empty");
    return dataset;
}

class StableFingerprint {
public:
    void AddBytes(const void* data, size_t size) {
        const auto* bytes = static_cast<const uint8_t*>(data);
        for (size_t index = 0; index < size; ++index) {
            first_ = (first_ ^ bytes[index]) * 1099511628211ULL;
            second_ = (second_ ^ bytes[index]) * 14029467366897019727ULL;
        }
    }

    void AddUint64(uint64_t value) {
        std::array<uint8_t, 8> bytes{};
        for (size_t index = 0; index < bytes.size(); ++index) {
            bytes[index] = static_cast<uint8_t>(value >> (index * 8));
        }
        AddBytes(bytes.data(), bytes.size());
    }

    void AddString(const std::string& value) {
        AddUint64(value.size());
        AddBytes(value.data(), value.size());
    }

    std::string Hex() const {
        std::ostringstream output;
        output << std::hex << std::setfill('0') << std::setw(16) << first_
               << std::setw(16) << second_;
        return output.str();
    }

private:
    uint64_t first_ = 1469598103934665603ULL;
    uint64_t second_ = 7809847782465536322ULL;
};

void FingerprintRecords(StableFingerprint* fingerprint, const char* split,
                        const std::vector<Record>& records, std::set<fs::path>* images) {
    fingerprint->AddString(split);
    fingerprint->AddUint64(records.size());
    std::vector<const Record*> canonical;
    canonical.reserve(records.size());
    for (const Record& record : records) canonical.push_back(&record);
    std::sort(canonical.begin(), canonical.end(), [](const Record* a, const Record* b) {
        return std::tie(a->image, a->receipt_id, a->task, a->question, a->target, a->answer,
                        a->family, a->field, a->operation, a->source) <
               std::tie(b->image, b->receipt_id, b->task, b->question, b->target, b->answer,
                        b->family, b->field, b->operation, b->source);
    });
    for (const Record* record : canonical) {
        const std::string path = fs::absolute(record->image).lexically_normal().generic_string();
        fingerprint->AddString(path);
        fingerprint->AddString(record->question);
        fingerprint->AddString(record->target);
        fingerprint->AddString(record->answer);
        fingerprint->AddString(record->task);
        fingerprint->AddString(record->field);
        fingerprint->AddString(record->operation);
        fingerprint->AddString(record->source);
        fingerprint->AddString(record->receipt_id);
        fingerprint->AddString(record->family);
        images->insert(path);
    }
}

std::string DatasetFingerprint(const Dataset& dataset) {
    StableFingerprint fingerprint;
    std::set<fs::path> images;
    FingerprintRecords(&fingerprint, "train", dataset.train, &images);
    FingerprintRecords(&fingerprint, "validation", dataset.validation, &images);
    std::array<char, 64 * 1024> buffer{};
    for (const fs::path& image : images) {
        fingerprint.AddString(image.generic_string());
        std::ifstream stream(image, std::ios::binary);
        if (!stream) throw std::runtime_error("cannot fingerprint image " + image.string());
        const uint64_t size = fs::file_size(image);
        fingerprint.AddUint64(size);
        while (stream) {
            stream.read(buffer.data(), buffer.size());
            const std::streamsize count = stream.gcount();
            if (count > 0) fingerprint.AddBytes(buffer.data(), static_cast<size_t>(count));
        }
        if (!stream.eof()) throw std::runtime_error("failed to fingerprint image " + image.string());
    }
    return fingerprint.Hex();
}

std::string JsonEscape(const std::string& value) {
    std::ostringstream output;
    for (unsigned char byte : value) {
        switch (byte) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (byte < 0x20) output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                                        << static_cast<int>(byte) << std::dec;
                else output << static_cast<char>(byte);
        }
    }
    return output.str();
}

class CharVocab {
public:
    static CharVocab Build(const Dataset& dataset) {
        std::set<uint32_t> codepoints;
        auto add = [&](const std::vector<Record>& records) {
            for (const Record& record : records) {
                for (uint32_t codepoint : Utf8Codepoints(record.question)) codepoints.insert(codepoint);
                for (uint32_t codepoint : Utf8Codepoints(record.target)) codepoints.insert(codepoint);
            }
        };
        add(dataset.train);
        add(dataset.validation);
        CharVocab vocab;
        vocab.tokens_ = {"<pad>", "<bos>", "<eos>", "<unk>"};
        for (uint32_t codepoint : codepoints) vocab.tokens_.push_back(EncodeUtf8(codepoint));
        for (size_t i = 0; i < vocab.tokens_.size(); ++i) {
            if (i >= 4) vocab.ids_[Utf8Codepoints(vocab.tokens_[i]).front()] = static_cast<int32_t>(i);
        }
        return vocab;
    }

    static CharVocab FromJson(const std::string& text, const std::string& context) {
        Json root = ParseJson(text, context);
        cJSON* items = Member(root.get(), "itos");
        if (!cJSON_IsArray(items) || cJSON_GetArraySize(items) < 4) {
            throw std::runtime_error(context + " must contain an itos array");
        }
        CharVocab vocab;
        const int count = cJSON_GetArraySize(items);
        vocab.tokens_.reserve(static_cast<size_t>(count));
        for (int index = 0; index < count; ++index) {
            cJSON* item = cJSON_GetArrayItem(items, index);
            if (!cJSON_IsString(item) || !item->valuestring) {
                throw std::runtime_error(context + " contains a non-string vocabulary token");
            }
            vocab.tokens_.emplace_back(item->valuestring);
        }
        if (vocab.tokens_[0] != "<pad>" || vocab.tokens_[1] != "<bos>" ||
            vocab.tokens_[2] != "<eos>" || vocab.tokens_[3] != "<unk>") {
            throw std::runtime_error(context + " has incompatible special-token ids");
        }
        std::unordered_set<uint32_t> seen;
        for (size_t index = 4; index < vocab.tokens_.size(); ++index) {
            const std::vector<uint32_t> codepoints = Utf8Codepoints(vocab.tokens_[index]);
            if (codepoints.size() != 1 || !seen.insert(codepoints.front()).second) {
                throw std::runtime_error(context + " must contain unique single-codepoint tokens");
            }
            vocab.ids_[codepoints.front()] = static_cast<int32_t>(index);
        }
        return vocab;
    }

    int32_t pad() const { return 0; }
    int32_t bos() const { return 1; }
    int32_t eos() const { return 2; }
    int32_t unk() const { return 3; }
    int size() const { return static_cast<int>(tokens_.size()); }

    std::vector<int32_t> Encode(const std::string& text, bool add_bos, bool add_eos,
                                int max_length) const {
        std::vector<int32_t> output;
        if (add_bos) output.push_back(bos());
        for (uint32_t codepoint : Utf8Codepoints(text)) {
            const auto found = ids_.find(codepoint);
            output.push_back(found == ids_.end() ? unk() : found->second);
        }
        if (add_eos) output.push_back(eos());
        if (max_length > 0 && static_cast<int>(output.size()) > max_length) {
            output.resize(max_length);
            if (add_eos) output.back() = eos();
        }
        return output;
    }

    std::string Decode(const std::vector<int32_t>& ids) const {
        std::string output;
        for (int32_t id : ids) {
            if (id == eos()) break;
            if (id == pad() || id == bos()) continue;
            if (id >= 0 && id < static_cast<int32_t>(tokens_.size())) output += tokens_[id];
        }
        return output;
    }

    std::string ToJson() const {
        std::ostringstream output;
        output << "{\"itos\":[";
        for (size_t i = 0; i < tokens_.size(); ++i) {
            if (i) output << ',';
            output << '"' << JsonEscape(tokens_[i]) << '"';
        }
        output << "]}";
        return output.str();
    }

private:
    std::vector<std::string> tokens_;
    std::unordered_map<uint32_t, int32_t> ids_;
};

enum class WeightRole { kBase, kToken, kLora, kAdapter, kRouter, kConstant };

struct TensorRef {
    std::string name;
    std::vector<int64_t> shape;
};

struct RouterRef {
    TensorRef logits;
    TensorRef route_indices;
    TensorRef route_weights;
};

struct LoraBinding {
    std::string target_id;
    std::string op_id;
    std::string base_tensor;
    std::string a_tensor;
    std::string b_tensor;
    std::vector<int64_t> a_shape;
    std::vector<int64_t> b_shape;
};

struct ModelPlan {
    api::CreateModelRequest request;
    std::vector<std::string> trainable_tensors;
    std::vector<LoraBinding> lora_bindings;
    std::vector<std::string> task_adapter_tensors;
    std::string logits = "vqa.logits";
    std::string router_logits = "vqa.router.logits";
    std::string automatic_route_indices;
    int target_length = 0;
    int memory_length = 0;
};

int64_t ElementCount(const std::vector<int64_t>& shape) {
    int64_t count = 1;
    for (int64_t dimension : shape) {
        if (dimension <= 0 || count > std::numeric_limits<int64_t>::max() / dimension) {
            throw std::runtime_error("invalid or overflowing tensor shape");
        }
        count *= dimension;
    }
    return count;
}

// These values mirror native/src/backends/vulkan_engine.c. Vulkan currently keeps every
// graph tensor in one mapped arena; it does not apply the CPU liveness arena to
// device tensors.  Consequently the forward sum is shape-derived, and the
// training requirement adds a conservative F32 gradient slot for every
// differentiable graph value; neither depends on host RAM or advertised VRAM.
constexpr uint64_t kMebibyte = 1024ULL * 1024ULL;
constexpr uint64_t kVulkanDefaultArenaBytes = 512ULL * kMebibyte;
constexpr uint64_t kVulkanMaximumArenaBytes = 4096ULL * kMebibyte;
constexpr uint64_t kVulkanGraphBaseBytes = 128ULL * kMebibyte;
constexpr uint64_t kVulkanGraphScratchBytes = 1ULL * kMebibyte;
constexpr uint64_t kVulkanGraphAlignment = 256ULL;

uint64_t CheckedResourceAdd(uint64_t left, uint64_t right) {
    if (right > std::numeric_limits<uint64_t>::max() - left) {
        throw std::runtime_error("graph resource size overflows uint64");
    }
    return left + right;
}

uint64_t AlignResourceBytes(uint64_t bytes) {
    const uint64_t remainder = bytes % kVulkanGraphAlignment;
    return remainder == 0 ? bytes
                          : CheckedResourceAdd(bytes, kVulkanGraphAlignment - remainder);
}

uint64_t TensorElementBytes(api::DataType dtype) {
    switch (dtype) {
        case api::DATA_TYPE_F32:
        case api::DATA_TYPE_I32:
        case api::DATA_TYPE_U32:
            return 4;
        case api::DATA_TYPE_F16:
        case api::DATA_TYPE_BF16:
        case api::DATA_TYPE_I16:
        case api::DATA_TYPE_U16:
            return 2;
        case api::DATA_TYPE_BOOL:
        case api::DATA_TYPE_U8:
        case api::DATA_TYPE_I8:
        case api::DATA_TYPE_F8_E5M2:
        case api::DATA_TYPE_F8_E4M3:
        case api::DATA_TYPE_F8_E8M0:
        case api::DATA_TYPE_F8_E4M3FNUZ:
        case api::DATA_TYPE_F8_E5M2FNUZ:
            return 1;
        case api::DATA_TYPE_C64:
        case api::DATA_TYPE_F64:
        case api::DATA_TYPE_I64:
        case api::DATA_TYPE_U64:
            return 8;
        default:
            throw std::runtime_error("resource preflight does not support sub-byte/unspecified dtype " +
                                     std::to_string(static_cast<int>(dtype)));
    }
}

template <typename Shape>
uint64_t ResourceTensorBytes(const Shape& shape, api::DataType dtype) {
    uint64_t elements = 1;
    for (int64_t dimension : shape) {
        if (dimension <= 0 || static_cast<uint64_t>(dimension) >
                                  std::numeric_limits<uint64_t>::max() / elements) {
            throw std::runtime_error("invalid or overflowing tensor shape in resource preflight");
        }
        elements *= static_cast<uint64_t>(dimension);
    }
    const uint64_t width = TensorElementBytes(dtype);
    if (elements > std::numeric_limits<uint64_t>::max() / width) {
        throw std::runtime_error("tensor byte size overflows uint64 in resource preflight");
    }
    return elements * width;
}

struct GraphResourceEstimate {
    uint64_t input_bytes = 0;
    uint64_t weight_bytes = 0;
    uint64_t output_bytes = 0;
    uint64_t gradient_bytes = 0;
    uint64_t vulkan_training_arena_required_bytes = 0;
};

GraphResourceEstimate EstimateGraphResources(const api::CreateModelRequest& request) {
    GraphResourceEstimate estimate;
    uint64_t arena_cursor = kVulkanGraphBaseBytes;
    auto add_slot = [&](uint64_t bytes, uint64_t* subtotal) {
        *subtotal = CheckedResourceAdd(*subtotal, bytes);
        arena_cursor = AlignResourceBytes(arena_cursor);
        arena_cursor = CheckedResourceAdd(arena_cursor, bytes);
    };
    for (const api::TensorSpec& input : request.inputs()) {
        add_slot(ResourceTensorBytes(input.shape(), input.dtype()), &estimate.input_bytes);
    }
    for (const api::Tensor& tensor : request.tensors()) {
        add_slot(ResourceTensorBytes(tensor.shape(), tensor.dtype()), &estimate.weight_bytes);
    }
    // CreateModel graph outputs are F32 in the current native graph contract.
    for (const api::GraphNode& node : request.graph().nodes()) {
        for (const auto& entry : node.output_shapes()) {
            add_slot(ResourceTensorBytes(entry.value().dims(), api::DATA_TYPE_F32),
                     &estimate.output_bytes);
        }
    }
    // Strict GPU TrainStep keeps gradients in the same device arena.  Every
    // differentiable F32 graph value can acquire a gradient during autograd;
    // reserve all of them so passing this preflight is not merely enough for
    // forward and then hazardous at backward-plan creation.
    for (const api::TensorSpec& input : request.inputs()) {
        if (input.dtype() == api::DATA_TYPE_F32) {
            add_slot(ResourceTensorBytes(input.shape(), input.dtype()), &estimate.gradient_bytes);
        }
    }
    for (const api::Tensor& tensor : request.tensors()) {
        if (tensor.dtype() == api::DATA_TYPE_F32) {
            add_slot(ResourceTensorBytes(tensor.shape(), tensor.dtype()), &estimate.gradient_bytes);
        }
    }
    for (const api::GraphNode& node : request.graph().nodes()) {
        for (const auto& entry : node.output_shapes()) {
            add_slot(ResourceTensorBytes(entry.value().dims(), api::DATA_TYPE_F32),
                     &estimate.gradient_bytes);
        }
    }
    estimate.vulkan_training_arena_required_bytes =
        CheckedResourceAdd(AlignResourceBytes(arena_cursor), kVulkanGraphScratchBytes);
    return estimate;
}

uint64_t ConfiguredVulkanArenaBytes() {
    uint64_t arena = kVulkanDefaultArenaBytes;
    const char* value = std::getenv("VOLVOX_VULKAN_MB");
    if (value && value[0]) {
        const long megabytes = std::strtol(value, nullptr, 10);
        // Match native/src/backends/vulkan_engine.c: invalid/out-of-range values leave the
        // 512 MiB default in place.
        if (megabytes >= 192 && megabytes <= 4096) {
            arena = static_cast<uint64_t>(megabytes) * kMebibyte;
        }
    }
    return arena;
}

void ValidateGraphResources(const Options& options, const api::CreateModelRequest& request) {
    const GraphResourceEstimate estimate = EstimateGraphResources(request);
    const uint64_t configured_vulkan_bytes = ConfiguredVulkanArenaBytes();
    const uint64_t configured_gpu_budget_bytes =
        static_cast<uint64_t>(options.gpu_memory_mb) * kMebibyte;
    std::cout << "{\"event\":\"resource_preflight\",\"backend\":\""
              << JsonEscape(options.backend) << "\",\"input_bytes\":" << estimate.input_bytes
              << ",\"weight_bytes\":" << estimate.weight_bytes
              << ",\"output_bytes\":" << estimate.output_bytes
              << ",\"gradient_reserve_bytes\":" << estimate.gradient_bytes
              << ",\"conservative_training_working_set_bytes\":"
              << estimate.vulkan_training_arena_required_bytes
              << ",\"vulkan_training_arena_required_bytes\":"
              << estimate.vulkan_training_arena_required_bytes
              << ",\"vulkan_arena_configured_bytes\":" << configured_vulkan_bytes
              << ",\"opengl_metal_budget_bytes\":" << configured_gpu_budget_bytes << "}\n";
    std::cout.flush();
    if ((options.backend == "opengl" || options.backend == "metal") &&
        estimate.vulkan_training_arena_required_bytes > configured_gpu_budget_bytes) {
        const uint64_t required_mib =
            (estimate.vulkan_training_arena_required_bytes + kMebibyte - 1) / kMebibyte;
        std::ostringstream message;
        message << (options.backend == "opengl" ? "OpenGL" : "Metal")
                << " resource preflight rejected this graph before runtime initialization: "
                << required_mib
                << " MiB is conservatively required for graph tensors, their F32 gradient "
                   "reserve, and backend workspace, but --gpu-memory-mb is "
                << options.gpu_memory_mb
                << "; raise --gpu-memory-mb only when the selected device has enough memory, "
                   "or reduce --batch-size, sequence lengths, or model dimensions";
        throw std::runtime_error(message.str());
    }
    if (options.backend != "vulkan" ||
        estimate.vulkan_training_arena_required_bytes <= configured_vulkan_bytes) {
        return;
    }

    const uint64_t required_mib =
        (estimate.vulkan_training_arena_required_bytes + kMebibyte - 1) / kMebibyte;
    std::ostringstream message;
    message << "Vulkan resource preflight rejected this graph before runtime initialization: "
            << required_mib << " MiB is required for graph tensors and their F32 gradient reserve, but "
            << configured_vulkan_bytes / kMebibyte
            << " MiB is configured by VOLVOX_VULKAN_MB";
    if (estimate.vulkan_training_arena_required_bytes > kVulkanMaximumArenaBytes) {
        message << " and the current Vulkan arena maximum is 4096 MiB; reduce --batch-size, "
                   "sequence lengths, or model dimensions, or use --backend cpu";
    } else {
        message << "; set VOLVOX_VULKAN_MB=" << required_mib
                << " or reduce --batch-size/model dimensions. This check proves conservative "
                   "training-arena fit; backward operator/device support is checked by TrainStep";
    }
    throw std::runtime_error(message.str());
}

std::string FloatBytes(float value) {
    std::string bytes(sizeof(float), '\0');
    std::memcpy(bytes.data(), &value, sizeof(value));
    return bytes;
}

class ReceiptGraphBuilder {
public:
    ReceiptGraphBuilder(const Options& options, const CharVocab& vocab)
        : options_(options), batch_(options.batch_size), d_(options.d_model),
          q_length_(options.max_question_length),
          target_length_(options.max_output_length - 1),
          memory_length_(kImageTokens + q_length_), vocabulary_(vocab.size()) {
        AppendStringEntry(request_.mutable_metadata(), "format", "tiny_receipt_vqa.volvox.v2");
        AppendStringEntry(request_.mutable_metadata(), "task_profile", options.task_profile);
        AppendStringEntry(request_.mutable_metadata(), "target_mode", options.target_mode);
        AppendStringEntry(request_.mutable_metadata(), "preprocessing", kPreprocessingFormat);
        AppendStringEntry(request_.mutable_metadata(), "families",
                          "phone,address,store,item_row,item_math,item_lookup,math,other");
        request_.mutable_exec()->set_backend(Backend(options.backend));
        if (options.debug) request_.mutable_exec()->set_debug(true);
        Build();
    }

    ModelPlan Finish() {
        request_.add_output_names(logits_.name);
        if (options_.router) request_.add_output_names(router_logits_.name);
        if (!automatic_route_indices_.name.empty()) {
            request_.add_output_names(automatic_route_indices_.name);
        }
        ModelPlan plan;
        plan.request = std::move(request_);
        plan.lora_bindings = std::move(lora_bindings_);
        plan.task_adapter_tensors = std::move(task_adapter_tensors_);
        plan.logits = logits_.name;
        plan.router_logits = options_.router ? router_logits_.name : std::string();
        plan.automatic_route_indices = automatic_route_indices_.name;
        plan.target_length = target_length_;
        plan.memory_length = memory_length_;
        for (const auto& [name, role] : weights_) {
            const bool trainable = options_.freeze_base_for_lora
                ? role == WeightRole::kToken || role == WeightRole::kLora ||
                      role == WeightRole::kAdapter || role == WeightRole::kRouter
                : role != WeightRole::kConstant;
            if (trainable) plan.trainable_tensors.push_back(name);
        }
        return plan;
    }

private:
    static api::Backend Backend(const std::string& name) {
        if (name == "vulkan") return api::BACKEND_VULKAN;
        if (name == "opengl") return api::BACKEND_OPENGL;
        if (name == "metal") return api::BACKEND_METAL;
        return api::BACKEND_CPU;
    }

    static int Groups(int channels) {
        for (int groups : {32, 16, 8, 4, 2}) if (channels % groups == 0) return groups;
        return 1;
    }

    TensorRef Input(const std::string& name, std::vector<int64_t> shape, api::DataType dtype) {
        api::TensorSpec* spec = request_.add_inputs();
        spec->set_name(name);
        for (int64_t dimension : shape) spec->add_shape(dimension);
        spec->set_dtype(dtype);
        const int64_t width = dtype == api::DATA_TYPE_F32 || dtype == api::DATA_TYPE_I32 ? 4 : 1;
        spec->set_size_bytes(ElementCount(shape) * width);
        return TensorRef{name, std::move(shape)};
    }

    TensorRef Weight(const std::string& name, std::vector<int64_t> shape, WeightRole role,
                     api::TensorInitializerKind initializer, float stddev = 0.02f) {
        if (!names_.insert(name).second) throw std::runtime_error("duplicate graph tensor " + name);
        api::Tensor* tensor = request_.add_tensors();
        tensor->set_name(name);
        for (int64_t dimension : shape) tensor->add_shape(dimension);
        tensor->set_dtype(api::DATA_TYPE_F32);
        api::TensorInitializer* init = tensor->mutable_initializer();
        init->set_kind(initializer);
        init->set_seed(static_cast<uint64_t>(options_.seed) + initializer_seed_++);
        if (initializer == api::TENSOR_INITIALIZER_NORMAL) init->set_stddev(stddev);
        weights_.emplace_back(name, role);
        return TensorRef{name, std::move(shape)};
    }

    TensorRef Constant(const std::string& name, float value) {
        if (!names_.insert(name).second) throw std::runtime_error("duplicate graph tensor " + name);
        api::Tensor* tensor = request_.add_tensors();
        tensor->set_name(name);
        tensor->add_shape(1);
        tensor->set_dtype(api::DATA_TYPE_F32);
        tensor->set_data(FloatBytes(value));
        weights_.emplace_back(name, WeightRole::kConstant);
        return TensorRef{name, {1}};
    }

    TensorRef Node(const std::string& id, const std::string& op,
                   const std::map<std::string, TensorRef>& inputs,
                   const std::string& output_name, std::vector<int64_t> output_shape,
                   const std::string& params = "{}") {
        if (!names_.insert(output_name).second) {
            throw std::runtime_error("duplicate graph output tensor " + output_name);
        }
        api::GraphNode* node = request_.mutable_graph()->add_nodes();
        node->set_index(node_index_++);
        node->set_id(id);
        node->set_op(api::OP_UNSPECIFIED);
        node->set_op_name(op);
        for (const auto& [key, tensor] : inputs) {
            AppendStringEntry(node->mutable_inputs(), key, tensor.name);
        }
        AppendStringEntry(node->mutable_outputs(), "out", output_name);
        api::TensorShape* shape = AppendTensorShapeEntry(node->mutable_output_shapes(), "out");
        for (int64_t dimension : output_shape) shape->add_dims(dimension);
        node->set_params_json(params);
        return TensorRef{output_name, std::move(output_shape)};
    }

    std::pair<TensorRef, TensorRef> NodePair(
            const std::string& id, const std::string& op,
            const std::map<std::string, TensorRef>& inputs,
            const std::string& first_key, const std::string& first_name,
            std::vector<int64_t> first_shape,
            const std::string& second_key, const std::string& second_name,
            std::vector<int64_t> second_shape, const std::string& params = "{}") {
        if (!names_.insert(first_name).second || !names_.insert(second_name).second) {
            throw std::runtime_error("duplicate graph output tensor in " + id);
        }
        api::GraphNode* node = request_.mutable_graph()->add_nodes();
        node->set_index(node_index_++);
        node->set_id(id);
        node->set_op(api::OP_UNSPECIFIED);
        node->set_op_name(op);
        for (const auto& [key, tensor] : inputs) {
            AppendStringEntry(node->mutable_inputs(), key, tensor.name);
        }
        AppendStringEntry(node->mutable_outputs(), first_key, first_name);
        AppendStringEntry(node->mutable_outputs(), second_key, second_name);
        api::TensorShape* first =
            AppendTensorShapeEntry(node->mutable_output_shapes(), first_key);
        for (int64_t dimension : first_shape) first->add_dims(dimension);
        api::TensorShape* second =
            AppendTensorShapeEntry(node->mutable_output_shapes(), second_key);
        for (int64_t dimension : second_shape) second->add_dims(dimension);
        node->set_params_json(params);
        return {TensorRef{first_name, std::move(first_shape)},
                TensorRef{second_name, std::move(second_shape)}};
    }

    TensorRef Add(const TensorRef& a, const TensorRef& b, const std::string& name,
                  std::vector<int64_t> shape = {}) {
        if (shape.empty()) shape = a.shape;
        return Node(name, "Add", {{"a", a}, {"b", b}}, name + ".out", std::move(shape));
    }

    TensorRef Mul(const TensorRef& a, const TensorRef& b, const std::string& name,
                  std::vector<int64_t> shape = {}) {
        if (shape.empty()) shape = a.shape;
        return Node(name, "Mul", {{"a", a}, {"b", b}}, name + ".out", std::move(shape));
    }

    TensorRef Dropout(const TensorRef& input, const std::string& name, float probability) {
        if (probability == 0.0f) return input;
        std::ostringstream params;
        params << "{\"ratio\":" << probability << ",\"seed\":"
               << (static_cast<uint64_t>(options_.seed) + dropout_seed_++) << '}';
        return Node(name, "Dropout", {{"input", input}}, name + ".out", input.shape, params.str());
    }

    TensorRef Linear(const TensorRef& input, int output_width, const std::string& name,
                     WeightRole role = WeightRole::kBase, bool lora = false) {
        const int input_width = static_cast<int>(input.shape.back());
        TensorRef weight = Weight(name + ".weight", {input_width, output_width}, role,
                                  api::TENSOR_INITIALIZER_XAVIER_UNIFORM);
        TensorRef bias = Weight(name + ".bias", {output_width}, role,
                                api::TENSOR_INITIALIZER_ZEROS);
        std::vector<int64_t> output_shape = input.shape;
        output_shape.back() = output_width;
        const std::string base_id = lora && options_.lora_rank > 0 ? name + ".base" : name;
        TensorRef base = Node(base_id, "Linear", {{"input", input}, {"weight", weight}, {"bias", bias}},
                              base_id + ".out", output_shape,
                              "{\"weight_layout\":\"IN_OUT\"}");
        if (!lora || options_.lora_rank == 0) return base;

        TensorRef branch_input = Dropout(input, name + ".lora.dropout", options_.lora_dropout);
        TensorRef a = Weight(name + ".lora_a", {input_width, options_.lora_rank},
                             WeightRole::kLora, api::TENSOR_INITIALIZER_XAVIER_UNIFORM);
        TensorRef b = Weight(name + ".lora_b", {options_.lora_rank, output_width},
                             WeightRole::kLora, api::TENSOR_INITIALIZER_ZEROS);
        TensorRef reduced = Node(name + ".lora.a", "Linear", {{"input", branch_input}, {"weight", a}},
                                 name + ".lora.a.out",
                                 [&] { auto shape = input.shape; shape.back() = options_.lora_rank; return shape; }(),
                                 "{\"weight_layout\":\"IN_OUT\"}");
        TensorRef expanded = Node(name + ".lora.b", "Linear", {{"input", reduced}, {"weight", b}},
                                  name + ".lora.b.out", output_shape,
                                  "{\"weight_layout\":\"IN_OUT\"}");
        if (!lora_scale_.has_value()) {
            lora_scale_ = Constant("vqa.lora.scale", options_.lora_alpha / options_.lora_rank);
        }
        TensorRef scaled = Node(name + ".lora.scale", "Mul", {{"a", expanded}, {"b", *lora_scale_}},
                                name + ".lora.scaled", output_shape);
        TensorRef output = Add(base, scaled, name, output_shape);
        lora_bindings_.push_back(LoraBinding{name, base_id, weight.name, a.name, b.name,
                                             a.shape, b.shape});
        return output;
    }

    TensorRef Norm(const TensorRef& input, const std::string& name) {
        TensorRef scale = Weight(name + ".weight", {d_}, WeightRole::kBase,
                                 api::TENSOR_INITIALIZER_ONES);
        TensorRef bias = Weight(name + ".bias", {d_}, WeightRole::kBase,
                                api::TENSOR_INITIALIZER_ZEROS);
        std::ostringstream params;
        params << "{\"d_model\":" << d_ << ",\"eps\":1e-5}";
        return Node(name, "LayerNorm", {{"input", input}, {"weight", scale}, {"bias", bias}},
                    name + ".out", input.shape, params.str());
    }

    TensorRef GroupNorm(const TensorRef& input, int channels, const std::string& name) {
        TensorRef scale = Weight(name + ".weight", {channels}, WeightRole::kBase,
                                 api::TENSOR_INITIALIZER_ONES);
        TensorRef bias = Weight(name + ".bias", {channels}, WeightRole::kBase,
                                api::TENSOR_INITIALIZER_ZEROS);
        std::ostringstream params;
        params << "{\"num_groups\":" << Groups(channels) << ",\"eps\":1e-5}";
        return Node(name, "GroupNorm", {{"input", input}, {"weight", scale}, {"bias", bias}},
                    name + ".out", input.shape, params.str());
    }

    TensorRef Conv(const TensorRef& input, int output_channels, int stride, const std::string& name) {
        const int input_channels = static_cast<int>(input.shape.back());
        TensorRef weight = Weight(name + ".weight", {3, 3, input_channels, output_channels},
                                  WeightRole::kBase, api::TENSOR_INITIALIZER_XAVIER_UNIFORM);
        std::vector<int64_t> shape = input.shape;
        shape[1] = (shape[1] + stride - 1) / stride;
        shape[2] = (shape[2] + stride - 1) / stride;
        shape[3] = output_channels;
        std::ostringstream params;
        params << "{\"stride\":[" << stride << ',' << stride
               << "],\"padding\":[1,1],\"weight_layout\":\"HWIO\"}";
        return Node(name, "Conv2D", {{"input", input}, {"weight", weight}},
                    name + ".out", std::move(shape), params.str());
    }

    TensorRef ConvBlock(const TensorRef& input, int output_channels, const std::string& name) {
        TensorRef value = Conv(input, output_channels, 2, name + ".conv");
        value = GroupNorm(value, output_channels, name + ".norm");
        return Node(name + ".activation", "SiLU", {{"input", value}},
                    name + ".activation.out", value.shape);
    }

    TensorRef ResBlock(const TensorRef& input, int channels, const std::string& name) {
        TensorRef value = Conv(input, channels, 1, name + ".conv1");
        value = GroupNorm(value, channels, name + ".norm1");
        value = Node(name + ".activation1", "SiLU", {{"input", value}},
                     name + ".activation1.out", value.shape);
        value = Conv(value, channels, 1, name + ".conv2");
        value = GroupNorm(value, channels, name + ".norm2");
        value = Add(input, value, name + ".residual");
        return Node(name + ".activation2", "SiLU", {{"input", value}},
                    name + ".out", value.shape);
    }

    TensorRef Embedding(const TensorRef& ids, const TensorRef& table, const std::string& name,
                        std::vector<int64_t> shape) {
        return Node(name, "Embedding", {{"input", ids}, {"weight", table}},
                    name + ".out", std::move(shape));
    }

    TensorRef FeedForward(const TensorRef& input, const std::string& name) {
        TensorRef normalized = Norm(input, name + ".norm");
        TensorRef expanded = Linear(normalized, d_ * options_.ff_multiplier, name + ".linear1",
                                    WeightRole::kBase, true);
        TensorRef activated = Node(name + ".gelu", "GELU", {{"input", expanded}},
                                   name + ".gelu.out", expanded.shape);
        activated = Dropout(activated, name + ".activation_dropout", options_.dropout);
        TensorRef projected = Linear(activated, d_, name + ".linear2", WeightRole::kBase, true);
        projected = Dropout(projected, name + ".output_dropout", options_.dropout);
        return Add(input, projected, name + ".residual");
    }

    TensorRef SelfAttention(const TensorRef& input, const TensorRef& mask, bool causal,
                            const std::string& name) {
        TensorRef normalized = Norm(input, name + ".norm");
        TensorRef qkv = Linear(normalized, d_ * 3, name + ".qkv");
        std::ostringstream params;
        params << "{\"heads\":" << options_.heads << ",\"causal\":"
               << (causal ? "true" : "false");
        if (options_.dropout > 0) {
            params << ",\"dropout\":" << options_.dropout << ",\"dropout_seed\":"
                   << (static_cast<uint64_t>(options_.seed) + dropout_seed_++);
        }
        params << '}';
        TensorRef attention = Node(name + ".attention", "SDPA", {{"qkv", qkv}, {"mask", mask}},
                                   name + ".attention.out", input.shape, params.str());
        TensorRef projected = Linear(attention, d_, name + ".output");
        projected = Dropout(projected, name + ".output_dropout", options_.dropout);
        return Add(input, projected, name + ".residual");
    }

    TensorRef CrossAttention(const TensorRef& input, const TensorRef& memory,
                             const TensorRef& source_mask, const std::string& name) {
        TensorRef normalized = Norm(input, name + ".norm");
        TensorRef query = Linear(normalized, d_, name + ".query");
        TensorRef key = Linear(memory, d_, name + ".key");
        TensorRef value = Linear(memory, d_, name + ".value");
        std::ostringstream params;
        params << "{\"heads\":" << options_.heads << ",\"causal\":false";
        if (options_.dropout > 0) {
            params << ",\"dropout\":" << options_.dropout << ",\"dropout_seed\":"
                   << (static_cast<uint64_t>(options_.seed) + dropout_seed_++);
        }
        params << '}';
        TensorRef attention = Node(name + ".attention", "CrossSDPA",
                                   {{"q", query}, {"k", key}, {"v", value}, {"mask", source_mask}},
                                   name + ".attention.out", input.shape, params.str());
        TensorRef projected = Linear(attention, d_, name + ".output");
        projected = Dropout(projected, name + ".output_dropout", options_.dropout);
        return Add(input, projected, name + ".residual");
    }

    TensorRef TaskAdapter(const TensorRef& input, const TensorRef& route_indices,
                          const TensorRef& route_weights, const std::string& name) {
        const int experts = static_cast<int>(kFamilies.size());
        const int hidden = std::min(options_.adapter_bottleneck, d_);
        TensorRef down_weight = Weight(name + ".down.weight", {experts, d_, hidden},
                                       WeightRole::kAdapter, api::TENSOR_INITIALIZER_XAVIER_UNIFORM);
        TensorRef down_bias = Weight(name + ".down.bias", {experts, hidden}, WeightRole::kAdapter,
                                     api::TENSOR_INITIALIZER_ZEROS);
        TensorRef up_weight = Weight(name + ".up.weight", {experts, hidden, d_},
                                     WeightRole::kAdapter, api::TENSOR_INITIALIZER_ZEROS);
        TensorRef up_bias = Weight(name + ".up.bias", {experts, d_}, WeightRole::kAdapter,
                                   api::TENSOR_INITIALIZER_ZEROS);
        for (const TensorRef* tensor : {&down_weight, &down_bias, &up_weight, &up_bias}) {
            task_adapter_tensors_.push_back(tensor->name);
        }
        std::vector<int64_t> hidden_shape = input.shape;
        hidden_shape.back() = hidden;
        TensorRef reduced = Node(name + ".down", "MoELinear",
                                 {{"input", input}, {"expert_weight", down_weight},
                                  {"expert_bias", down_bias}, {"route_indices", route_indices},
                                  {"route_weights", route_weights}},
                                 name + ".down.out", hidden_shape);
        TensorRef activated = Node(name + ".gelu", "GELU", {{"input", reduced}},
                                   name + ".gelu.out", hidden_shape);
        TensorRef expanded = Node(name + ".up", "MoELinear",
                                  {{"input", activated}, {"expert_weight", up_weight},
                                   {"expert_bias", up_bias}, {"route_indices", route_indices},
                                   {"route_weights", route_weights}},
                                  name + ".up.out", input.shape);
        expanded = Dropout(expanded, name + ".dropout", options_.dropout);
        return Add(input, expanded, name + ".residual");
    }

    RouterRef Router(const TensorRef& question, const TensorRef& pooling_weights,
                     bool build_routes) {
        TensorRef transposed = Node("vqa.router.transpose", "Transpose", {{"input", question}},
                                    "vqa.router.transposed", {batch_, d_, q_length_},
                                    "{\"perm\":[0,2,1]}");
        TensorRef expanded = Node("vqa.router.weights", "Unsqueeze", {{"input", pooling_weights}},
                                  "vqa.router.weights.expanded", {batch_, 1, q_length_},
                                  "{\"axes\":[1]}");
        TensorRef weighted = Node("vqa.router.weighted", "Mul", {{"a", transposed}, {"b", expanded}},
                                  "vqa.router.weighted.out", {batch_, d_, q_length_});
        TensorRef pooled = Node("vqa.router.pool", "ReduceSum", {{"input", weighted}},
                                "vqa.router.pooled", {batch_, d_},
                                "{\"axis\":-1,\"keepdims\":false}");
        const int hidden_width = std::max(1, d_ / 2);
        TensorRef hidden = Linear(pooled, hidden_width, "vqa.router.linear1", WeightRole::kRouter);
        hidden = Node("vqa.router.gelu", "GELU", {{"input", hidden}},
                      "vqa.router.gelu.out", hidden.shape);
        const int experts = static_cast<int>(kFamilies.size());
        TensorRef weight = Weight("vqa.router.linear2.weight", {hidden_width, experts},
                                  WeightRole::kRouter, api::TENSOR_INITIALIZER_XAVIER_UNIFORM);
        TensorRef bias = Weight("vqa.router.linear2.bias", {experts}, WeightRole::kRouter,
                                api::TENSOR_INITIALIZER_ZEROS);
        TensorRef logits = Node("vqa.router.linear2", "Linear",
                                {{"input", hidden}, {"weight", weight}, {"bias", bias}},
                                "vqa.router.linear2.out", {batch_, experts},
                                "{\"weight_layout\":\"IN_OUT\"}");
        RouterRef result;
        result.logits = logits;
        if (build_routes) {
            auto routes = NodePair(
                "vqa.router.auto", "MoERouter",
                {{"input", hidden}, {"weight", weight}, {"bias", bias}},
                "indices", "vqa.router.auto.indices", {batch_, 1},
                "weights", "vqa.router.auto.weights", {batch_, 1},
                "{\"num_experts\":8,\"top_k\":1,\"normalize\":1}");
            result.route_indices = std::move(routes.first);
            result.route_weights = std::move(routes.second);
        }
        return result;
    }

    TensorRef BroadcastRoute(const TensorRef& route, int length, const std::string& name) {
        TensorRef expanded = Node(name + ".unsqueeze", "Unsqueeze", {{"input", route}},
                                  name + ".unsqueezed", {batch_, 1, 1}, "{\"axes\":[2]}");
        TensorRef zeros = Weight(name + ".zeros", {1, length, 1}, WeightRole::kConstant,
                                 api::TENSOR_INITIALIZER_ZEROS);
        return Add(expanded, zeros, name, {batch_, length, 1});
    }

    TensorRef SelectRoute(const TensorRef& explicit_route, const TensorRef& automatic_route,
                          const TensorRef& explicit_gate, const TensorRef& automatic_gate,
                          const std::string& name) {
        TensorRef explicit_value = Mul(explicit_route, explicit_gate, name + ".explicit",
                                       explicit_route.shape);
        TensorRef automatic_value = Mul(automatic_route, automatic_gate, name + ".automatic",
                                        automatic_route.shape);
        return Add(explicit_value, automatic_value, name, explicit_route.shape);
    }

    void Build() {
        const std::array<int, 5> channels = options_.smoke
            ? std::array<int, 5>{8, 16, 24, d_, d_}
            : std::array<int, 5>{48, 96, 192, d_, d_};
        TensorRef image = Input("vqa.image", {batch_, kImageHeight, kImageWidth, 1}, api::DATA_TYPE_F32);
        TensorRef question_ids = Input("vqa.question_ids", {batch_, q_length_}, api::DATA_TYPE_I32);
        TensorRef question_positions = Input("vqa.question_positions", {batch_, q_length_}, api::DATA_TYPE_I32);
        TensorRef source_mask = Input("vqa.source_mask", {batch_, memory_length_}, api::DATA_TYPE_I32);
        TensorRef decoder_ids = Input("vqa.decoder_ids", {batch_, target_length_}, api::DATA_TYPE_I32);
        TensorRef decoder_positions = Input("vqa.decoder_positions", {batch_, target_length_}, api::DATA_TYPE_I32);
        TensorRef target_mask = Input("vqa.target_mask", {batch_, target_length_}, api::DATA_TYPE_I32);
        TensorRef pooling_weights;
        if (options_.router) {
            pooling_weights = Input("vqa.question_pool_weights", {batch_, q_length_}, api::DATA_TYPE_F32);
        }
        TensorRef memory_route_indices, memory_route_weights, decoder_route_indices, decoder_route_weights;
        TensorRef explicit_route_gate, automatic_route_gate;
        if (options_.task_adapters) {
            memory_route_indices = Input("vqa.memory_route_indices", {batch_, memory_length_, 1}, api::DATA_TYPE_F32);
            memory_route_weights = Input("vqa.memory_route_weights", {batch_, memory_length_, 1}, api::DATA_TYPE_F32);
            decoder_route_indices = Input("vqa.decoder_route_indices", {batch_, target_length_, 1}, api::DATA_TYPE_F32);
            decoder_route_weights = Input("vqa.decoder_route_weights", {batch_, target_length_, 1}, api::DATA_TYPE_F32);
            if (options_.router) {
                explicit_route_gate = Input("vqa.explicit_route_gate", {batch_, 1, 1}, api::DATA_TYPE_F32);
                automatic_route_gate = Input("vqa.automatic_route_gate", {batch_, 1, 1}, api::DATA_TYPE_F32);
            }
        }

        TensorRef vision = image;
        vision = ConvBlock(vision, channels[0], "vqa.stem.0");
        vision = ConvBlock(vision, channels[1], "vqa.stem.1");
        vision = ResBlock(vision, channels[1], "vqa.stem.2");
        vision = ConvBlock(vision, channels[2], "vqa.stem.3");
        vision = ResBlock(vision, channels[2], "vqa.stem.4");
        vision = ConvBlock(vision, channels[3], "vqa.stem.5");
        vision = ResBlock(vision, channels[3], "vqa.stem.6");
        vision = ConvBlock(vision, channels[4], "vqa.stem.7");
        vision = ResBlock(vision, channels[4], "vqa.stem.8");
        TensorRef image_tokens = Node("vqa.image_tokens", "Reshape", {{"input", vision}},
                                      "vqa.image_tokens.out", {batch_, kImageTokens, d_});
        TensorRef image_position = Weight("vqa.img_pos", {1, kImageTokens, d_}, WeightRole::kBase,
                                          api::TENSOR_INITIALIZER_NORMAL);
        TensorRef image_type = Weight("vqa.type_img", {1, 1, d_}, WeightRole::kBase,
                                      api::TENSOR_INITIALIZER_NORMAL);
        image_tokens = Add(image_tokens, image_position, "vqa.image_positioned");
        image_tokens = Add(image_tokens, image_type, "vqa.image_typed");

        TensorRef token_table = Weight("vqa.tok.weight", {vocabulary_, d_}, WeightRole::kToken,
                                       api::TENSOR_INITIALIZER_NORMAL);
        TensorRef question_position_table = Weight("vqa.q_pos", {q_length_, d_}, WeightRole::kBase,
                                                   api::TENSOR_INITIALIZER_NORMAL);
        TensorRef question_type = Weight("vqa.type_q", {1, 1, d_}, WeightRole::kBase,
                                         api::TENSOR_INITIALIZER_NORMAL);
        TensorRef question = Embedding(question_ids, token_table, "vqa.question_embedding",
                                       {batch_, q_length_, d_});
        TensorRef positioned_question = Embedding(question_positions, question_position_table,
                                                  "vqa.question_position_embedding",
                                                  {batch_, q_length_, d_});
        question = Add(question, positioned_question, "vqa.question_positioned");
        question = Add(question, question_type, "vqa.question_typed");
        RouterRef router;
        if (options_.router) {
            // Always materialize the top-1 route when the router is enabled.
            // Task adapters consume it when present, while router-only models
            // still expose it for honest routing evaluation.
            router = Router(question, pooling_weights, true);
            router_logits_ = router.logits;
            automatic_route_indices_ = router.route_indices;
        }

        if (options_.task_adapters && options_.router) {
            TensorRef automatic_memory_indices = BroadcastRoute(
                router.route_indices, memory_length_, "vqa.router.memory.indices");
            TensorRef automatic_memory_weights = BroadcastRoute(
                router.route_weights, memory_length_, "vqa.router.memory.weights");
            TensorRef automatic_decoder_indices = BroadcastRoute(
                router.route_indices, target_length_, "vqa.router.decoder.indices");
            TensorRef automatic_decoder_weights = BroadcastRoute(
                router.route_weights, target_length_, "vqa.router.decoder.weights");
            memory_route_indices = SelectRoute(
                memory_route_indices, automatic_memory_indices,
                explicit_route_gate, automatic_route_gate, "vqa.memory_route.selected_indices");
            memory_route_weights = SelectRoute(
                memory_route_weights, automatic_memory_weights,
                explicit_route_gate, automatic_route_gate, "vqa.memory_route.selected_weights");
            decoder_route_indices = SelectRoute(
                decoder_route_indices, automatic_decoder_indices,
                explicit_route_gate, automatic_route_gate, "vqa.decoder_route.selected_indices");
            decoder_route_weights = SelectRoute(
                decoder_route_weights, automatic_decoder_weights,
                explicit_route_gate, automatic_route_gate, "vqa.decoder_route.selected_weights");
        }

        TensorRef memory = Node("vqa.memory_concat", "Concat", {{"a", image_tokens}, {"b", question}},
                                "vqa.memory", {batch_, memory_length_, d_}, "{\"axis\":1}");
        for (int layer = 0; layer < options_.encoder_layers; ++layer) {
            const std::string prefix = "vqa.encoder." + std::to_string(layer);
            memory = SelfAttention(memory, source_mask, false, prefix + ".self_attention");
            memory = FeedForward(memory, prefix + ".feed_forward");
        }
        // nn.TransformerEncoder in the source has no final norm.
        if (options_.task_adapters) {
            memory = TaskAdapter(memory, memory_route_indices, memory_route_weights,
                                 "vqa.memory_adapters");
        }

        TensorRef target_position = Weight("vqa.y_pos", {1, target_length_, d_}, WeightRole::kBase,
                                           api::TENSOR_INITIALIZER_NORMAL);
        TensorRef hidden = Embedding(decoder_ids, token_table, "vqa.decoder_embedding",
                                     {batch_, target_length_, d_});
        hidden = Add(hidden, target_position, "vqa.decoder_positioned");
        for (int layer = 0; layer < options_.decoder_layers; ++layer) {
            const std::string prefix = "vqa.decoder." + std::to_string(layer);
            hidden = SelfAttention(hidden, target_mask, true, prefix + ".self_attention");
            hidden = CrossAttention(hidden, memory, source_mask, prefix + ".cross_attention");
            hidden = FeedForward(hidden, prefix + ".feed_forward");
        }
        if (options_.task_adapters) {
            hidden = TaskAdapter(hidden, decoder_route_indices, decoder_route_weights,
                                 "vqa.decoder_adapters");
        }
        hidden = Norm(hidden, "vqa.norm");
        TensorRef projection = Node("vqa.lm_head", "Linear",
                                    {{"input", hidden}, {"weight", token_table}},
                                    "vqa.lm_projection", {batch_, target_length_, vocabulary_},
                                    "{\"weight_layout\":\"OUT_IN\"}");
        TensorRef output_bias = Weight("vqa.out_bias", {vocabulary_}, WeightRole::kToken,
                                       api::TENSOR_INITIALIZER_ZEROS);
        logits_ = Add(projection, output_bias, "vqa.logits", projection.shape);
    }

    const Options& options_;
    int batch_;
    int d_;
    int q_length_;
    int target_length_;
    int memory_length_;
    int vocabulary_;
    api::CreateModelRequest request_;
    int node_index_ = 0;
    uint64_t initializer_seed_ = 0;
    uint64_t dropout_seed_ = 100000;
    std::unordered_set<std::string> names_;
    std::vector<std::pair<std::string, WeightRole>> weights_;
    std::optional<TensorRef> lora_scale_;
    std::vector<LoraBinding> lora_bindings_;
    std::vector<std::string> task_adapter_tensors_;
    TensorRef logits_;
    TensorRef router_logits_;
    TensorRef automatic_route_indices_;
};

template <class T>
api::Tensor TensorMessage(const std::string& name, const std::vector<int64_t>& shape,
                          api::DataType dtype, const std::vector<T>& values) {
    if (ElementCount(shape) != static_cast<int64_t>(values.size())) {
        throw std::runtime_error("tensor " + name + " has the wrong element count");
    }
    api::Tensor tensor;
    tensor.set_name(name);
    for (int64_t dimension : shape) tensor.add_shape(dimension);
    tensor.set_dtype(dtype);
    tensor.set_data(values.data(), values.size() * sizeof(T));
    return tensor;
}

void PreprocessImage(const fs::path& path, float* destination) {
    const int shape[3] = {kImageHeight, kImageWidth, 1};
    std::array<char, 256> error{};
    if (volvox_load_image_to_tensor(path.string().c_str(), destination, shape, 3,
                                    VOLVOX_IMAGE_MINUS_ONE_ONE,
                                    error.data(), error.size()) != 0) {
        throw std::runtime_error("cannot preprocess image " + path.string() + ": " +
                                 (error[0] ? error.data() : "unknown image error"));
    }
}

struct PreparedBatch {
    std::vector<api::Tensor> inputs;
    std::vector<int64_t> labels;
    std::vector<int64_t> family_ids;
    int active_labels = 0;
};

PreparedBatch PrepareBatch(const Options& options, const CharVocab& vocab,
                           const std::vector<const Record*>& records, int target_length,
                           int memory_length, bool training) {
    if (static_cast<int>(records.size()) != options.batch_size) {
        throw std::runtime_error("PrepareBatch requires exactly batch_size records");
    }
    const int batch = options.batch_size;
    const int q_length = options.max_question_length;
    std::vector<float> images(static_cast<size_t>(batch) * kImageHeight * kImageWidth);
    std::vector<int32_t> question_ids(static_cast<size_t>(batch) * q_length, vocab.pad());
    std::vector<int32_t> question_positions(static_cast<size_t>(batch) * q_length);
    std::vector<int32_t> source_mask(static_cast<size_t>(batch) * memory_length, 0);
    std::vector<int32_t> decoder_ids(static_cast<size_t>(batch) * target_length, vocab.pad());
    std::vector<int32_t> decoder_positions(static_cast<size_t>(batch) * target_length);
    std::vector<int32_t> target_mask(static_cast<size_t>(batch) * target_length, 0);
    std::vector<float> pooling_weights;
    if (options.router) pooling_weights.assign(static_cast<size_t>(batch) * q_length, 0.0f);
    std::vector<float> memory_route_indices, memory_route_weights;
    std::vector<float> decoder_route_indices, decoder_route_weights;
    std::vector<float> explicit_route_gate, automatic_route_gate;
    if (options.task_adapters) {
        memory_route_indices.resize(static_cast<size_t>(batch) * memory_length);
        memory_route_weights.assign(static_cast<size_t>(batch) * memory_length, 1.0f);
        decoder_route_indices.resize(static_cast<size_t>(batch) * target_length);
        decoder_route_weights.assign(static_cast<size_t>(batch) * target_length, 1.0f);
        if (options.router) {
            const bool automatic = !training && options.evaluation_routing == "auto";
            explicit_route_gate.assign(batch, automatic ? 0.0f : 1.0f);
            automatic_route_gate.assign(batch, automatic ? 1.0f : 0.0f);
        }
    }

    PreparedBatch batch_data;
    batch_data.labels.assign(static_cast<size_t>(batch) * target_length, kIgnoreId);
    batch_data.family_ids.resize(batch);
    for (int row = 0; row < batch; ++row) {
        const Record& record = *records[row];
        PreprocessImage(record.image,
                        images.data() + static_cast<size_t>(row) * kImageHeight * kImageWidth);

        std::vector<int32_t> question = vocab.Encode(record.question, false, true, q_length);
        for (size_t position = 0; position < question.size(); ++position) {
            question_ids[static_cast<size_t>(row) * q_length + position] = question[position];
        }
        for (int position = 0; position < q_length; ++position) {
            question_positions[static_cast<size_t>(row) * q_length + position] = position;
        }
        std::fill(source_mask.begin() + static_cast<size_t>(row) * memory_length,
                  source_mask.begin() + static_cast<size_t>(row) * memory_length + kImageTokens, 1);
        for (size_t position = 0; position < question.size(); ++position) {
            source_mask[static_cast<size_t>(row) * memory_length + kImageTokens + position] = 1;
            if (options.router) {
                pooling_weights[static_cast<size_t>(row) * q_length + position] =
                    1.0f / static_cast<float>(question.size());
            }
        }

        std::vector<int32_t> target = vocab.Encode(record.target, true, true, options.max_output_length);
        if (training) {
            for (int position = 0; position < target_length; ++position) {
                const size_t offset = static_cast<size_t>(row) * target_length + position;
                if (position < static_cast<int>(target.size()) - 1) {
                    decoder_ids[offset] = target[position];
                    target_mask[offset] = 1;
                    batch_data.labels[offset] = target[position + 1];
                    if (target[position + 1] != vocab.pad()) ++batch_data.active_labels;
                }
            }
        } else {
            decoder_ids[static_cast<size_t>(row) * target_length] = vocab.bos();
            target_mask[static_cast<size_t>(row) * target_length] = 1;
        }
        for (int position = 0; position < target_length; ++position) {
            decoder_positions[static_cast<size_t>(row) * target_length + position] = position;
        }
        const int family = FamilyId(record.family);
        batch_data.family_ids[row] = family;
        if (options.task_adapters) {
            std::fill(memory_route_indices.begin() + static_cast<size_t>(row) * memory_length,
                      memory_route_indices.begin() + static_cast<size_t>(row + 1) * memory_length,
                      static_cast<float>(family));
            std::fill(decoder_route_indices.begin() + static_cast<size_t>(row) * target_length,
                      decoder_route_indices.begin() + static_cast<size_t>(row + 1) * target_length,
                      static_cast<float>(family));
        }
    }

    batch_data.inputs.push_back(TensorMessage("vqa.image", {batch, kImageHeight, kImageWidth, 1},
                                                     api::DATA_TYPE_F32, images));
    batch_data.inputs.push_back(TensorMessage("vqa.question_ids", {batch, q_length},
                                                     api::DATA_TYPE_I32, question_ids));
    batch_data.inputs.push_back(TensorMessage("vqa.question_positions", {batch, q_length},
                                                     api::DATA_TYPE_I32, question_positions));
    batch_data.inputs.push_back(TensorMessage("vqa.source_mask", {batch, memory_length},
                                                     api::DATA_TYPE_I32, source_mask));
    batch_data.inputs.push_back(TensorMessage("vqa.decoder_ids", {batch, target_length},
                                                     api::DATA_TYPE_I32, decoder_ids));
    batch_data.inputs.push_back(TensorMessage("vqa.decoder_positions", {batch, target_length},
                                                     api::DATA_TYPE_I32, decoder_positions));
    batch_data.inputs.push_back(TensorMessage("vqa.target_mask", {batch, target_length},
                                                     api::DATA_TYPE_I32, target_mask));
    if (options.router) {
        batch_data.inputs.push_back(TensorMessage("vqa.question_pool_weights", {batch, q_length},
                                                         api::DATA_TYPE_F32, pooling_weights));
    }
    if (options.task_adapters) {
        batch_data.inputs.push_back(TensorMessage("vqa.memory_route_indices", {batch, memory_length, 1},
                                                         api::DATA_TYPE_F32, memory_route_indices));
        batch_data.inputs.push_back(TensorMessage("vqa.memory_route_weights", {batch, memory_length, 1},
                                                         api::DATA_TYPE_F32, memory_route_weights));
        batch_data.inputs.push_back(TensorMessage("vqa.decoder_route_indices", {batch, target_length, 1},
                                                         api::DATA_TYPE_F32, decoder_route_indices));
        batch_data.inputs.push_back(TensorMessage("vqa.decoder_route_weights", {batch, target_length, 1},
                                                         api::DATA_TYPE_F32, decoder_route_weights));
        if (options.router) {
            batch_data.inputs.push_back(TensorMessage("vqa.explicit_route_gate", {batch, 1, 1},
                                                             api::DATA_TYPE_F32, explicit_route_gate));
            batch_data.inputs.push_back(TensorMessage("vqa.automatic_route_gate", {batch, 1, 1},
                                                             api::DATA_TYPE_F32, automatic_route_gate));
        }
    }
    return batch_data;
}

api::Tensor* FindInput(std::vector<api::Tensor>* inputs, const std::string& name) {
    for (api::Tensor& input : *inputs) if (input.name() == name) return &input;
    return nullptr;
}

std::vector<int32_t> TensorI32(const api::Tensor& tensor) {
    if (tensor.dtype() != api::DATA_TYPE_I32 || tensor.data().size() % sizeof(int32_t) != 0) {
        throw std::runtime_error("tensor " + tensor.name() + " is not I32");
    }
    std::vector<int32_t> output(tensor.data().size() / sizeof(int32_t));
    std::memcpy(output.data(), tensor.data().data(), tensor.data().size());
    return output;
}

std::vector<float> TensorF32(const api::Tensor& tensor) {
    if (tensor.dtype() != api::DATA_TYPE_F32 || tensor.data().size() % sizeof(float) != 0) {
        throw std::runtime_error("tensor " + tensor.name() + " is not F32");
    }
    std::vector<float> output(tensor.data().size() / sizeof(float));
    std::memcpy(output.data(), tensor.data().data(), tensor.data().size());
    return output;
}

std::string ExtractAnswer(const std::string& text) {
    const size_t begin = text.find("<answer>");
    if (begin == std::string::npos) return CleanText(text);
    const size_t content = begin + std::strlen("<answer>");
    const size_t end = text.find("</answer>", content);
    if (end == std::string::npos) return CleanText(text);
    return CleanText(text.substr(content, end - content));
}

using MetricGroups = std::map<std::string, std::array<int, 3>>;

struct Metrics {
    int count = 0;
    int target_exact = 0;
    int answer_exact = 0;
    int router_count = 0;
    int router_correct = 0;
    MetricGroups by_task;
    MetricGroups by_operation;
    MetricGroups by_source;
    MetricGroups by_family;
    MetricGroups by_task_operation;

    double TargetExact() const { return static_cast<double>(target_exact) / std::max(1, count); }
    double AnswerExact() const { return static_cast<double>(answer_exact) / std::max(1, count); }
    double RouterAccuracy() const {
        return static_cast<double>(router_correct) / std::max(1, router_count);
    }
};

void AddMetricGroup(MetricGroups* groups, const std::string& key, bool target_ok, bool answer_ok) {
    auto& group = (*groups)[key.empty() ? "unknown" : key];
    ++group[0];
    group[1] += target_ok;
    group[2] += answer_ok;
}

void PrintMetricGroup(const char* name, const MetricGroups& groups) {
    std::cout << '"' << name << "\":{";
    bool first = true;
    for (const auto& [key, values] : groups) {
        if (!first) std::cout << ',';
        first = false;
        std::cout << '"' << JsonEscape(key) << "\":{\"n\":" << values[0]
                  << ",\"target_exact\":"
                  << static_cast<double>(values[1]) / std::max(1, values[0])
                  << ",\"answer_exact\":"
                  << static_cast<double>(values[2]) / std::max(1, values[0]) << '}';
    }
    std::cout << '}';
}

const char* Method(const char* name) {
    static thread_local std::string method;
    method = "/volvoxai.v1.VolvoxAiService/";
    method += name;
    return method.c_str();
}

api::Backend BackendValue(const std::string& backend) {
    if (backend == "vulkan") return api::BACKEND_VULKAN;
    if (backend == "opengl") return api::BACKEND_OPENGL;
    if (backend == "metal") return api::BACKEND_METAL;
    return api::BACKEND_CPU;
}

int ActiveTargetCount(const CharVocab& vocab, const Record& record, int max_output_length) {
    const std::vector<int32_t> target = vocab.Encode(record.target, true, true, max_output_length);
    return std::max(0, static_cast<int>(target.size()) - 1);
}

api::TrainStepResponse TrainOneBatch(const GrpcFfiClient& client, const Options& options,
                                     const ModelPlan& model, const std::string& model_id,
                                     PreparedBatch batch, float learning_rate,
                                     float token_normalizer, float router_normalizer,
                                     bool flush) {
    api::TrainStepRequest request;
    request.set_model_id(model_id);
    for (api::Tensor& input : batch.inputs) *request.add_inputs() = std::move(input);
    for (const std::string& name : model.trainable_tensors) request.add_trainable_tensors(name);
    api::OptimizerOptions* optimizer = request.mutable_optimizer();
    optimizer->set_learning_rate(learning_rate);
    optimizer->set_beta1(0.9f);
    optimizer->set_beta2(0.999f);
    optimizer->set_epsilon(1.0e-8f);
    optimizer->set_weight_decay(options.weight_decay);
    optimizer->set_max_grad_norm(1.0f);
    request.set_update_mode(api::TENSOR_UPDATE_ADAMW);
    api::CrossEntropyLoss* token_loss = request.add_losses();
    token_loss->set_name("tokens");
    token_loss->set_logits_tensor(model.logits);
    for (int64_t label : batch.labels) token_loss->add_target_ids(label);
    token_loss->set_ignore_id(kIgnoreId);
    token_loss->set_weight(1.0f);
    token_loss->set_normalizer(std::max(1.0f, token_normalizer));
    if (options.router) {
        api::CrossEntropyLoss* router_loss = request.add_losses();
        router_loss->set_name("router");
        router_loss->set_logits_tensor(model.router_logits);
        for (int64_t family : batch.family_ids) router_loss->add_target_ids(family);
        router_loss->set_weight(0.1f);
        router_loss->set_normalizer(std::max(1.0f, router_normalizer));
    }
    api::GradientAccumulationOptions* accumulation = request.mutable_accumulation();
    accumulation->set_steps(options.gradient_accumulation_steps);
    accumulation->set_flush(flush);
    return client.Unary<api::TrainStepRequest, api::TrainStepResponse>(Method("TrainStep"), request);
}

Metrics Evaluate(const GrpcFfiClient& client, const Options& options, const CharVocab& vocab,
                 const ModelPlan& model, const std::string& model_id,
                 const std::vector<Record>& records) {
    Metrics metrics;
    int failure_examples = 0;
    const size_t requested_batches = options.evaluation_batches > 0
        ? static_cast<size_t>(options.evaluation_batches)
        : (records.size() + options.batch_size - 1) / options.batch_size;
    const size_t batch_count = std::min(
        requested_batches, (records.size() + options.batch_size - 1) / options.batch_size);
    for (size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
        const size_t offset = batch_index * options.batch_size;
        const int actual = static_cast<int>(std::min<size_t>(options.batch_size, records.size() - offset));
        std::vector<const Record*> rows;
        for (int row = 0; row < actual; ++row) rows.push_back(&records[offset + row]);
        while (static_cast<int>(rows.size()) < options.batch_size) rows.push_back(rows.front());
        PreparedBatch prepared = PrepareBatch(options, vocab, rows, model.target_length,
                                              model.memory_length, false);
        api::Tensor* decoder_tensor = FindInput(&prepared.inputs, "vqa.decoder_ids");
        api::Tensor* mask_tensor = FindInput(&prepared.inputs, "vqa.target_mask");
        if (!decoder_tensor || !mask_tensor) throw std::runtime_error("decoder inputs are missing");
        std::vector<int32_t> decoder = TensorI32(*decoder_tensor);
        std::vector<int32_t> target_mask = TensorI32(*mask_tensor);
        std::vector<std::vector<int32_t>> generated(options.batch_size);
        std::vector<bool> done(options.batch_size, false);
        for (int position = 0; position < model.target_length; ++position) {
            decoder_tensor->set_data(decoder.data(), decoder.size() * sizeof(int32_t));
            mask_tensor->set_data(target_mask.data(), target_mask.size() * sizeof(int32_t));
            api::RunRequest request;
            request.set_model_id(model_id);
            for (const api::Tensor& input : prepared.inputs) *request.add_inputs() = input;
            request.add_output_names(model.logits);
            const bool inspect_router = position == 0 && !model.automatic_route_indices.empty();
            if (inspect_router) request.add_output_names(model.automatic_route_indices);
            api::RunResponse response =
                client.Unary<api::RunRequest, api::RunResponse>(Method("Run"), request);
            const api::Tensor* logits_tensor_ptr = nullptr;
            const api::Tensor* route_tensor = nullptr;
            for (const api::Tensor& output : response.outputs()) {
                if (output.name() == model.logits) logits_tensor_ptr = &output;
                if (inspect_router && output.name() == model.automatic_route_indices) {
                    route_tensor = &output;
                }
            }
            if (!logits_tensor_ptr || (inspect_router && !route_tensor)) {
                throw std::runtime_error("Run did not return the requested receipt/router outputs");
            }
            const api::Tensor& logits_tensor = *logits_tensor_ptr;
            std::vector<float> logits = TensorF32(logits_tensor);
            const int vocabulary = static_cast<int>(logits_tensor.shape(logits_tensor.shape_size() - 1));
            const size_t expected = static_cast<size_t>(options.batch_size) * model.target_length * vocabulary;
            if (logits.size() != expected) throw std::runtime_error("unexpected receipt logits shape");
            if (inspect_router) {
                const std::vector<float> routes = TensorF32(*route_tensor);
                if (routes.size() != static_cast<size_t>(options.batch_size)) {
                    throw std::runtime_error("unexpected automatic-router output shape");
                }
                for (int row = 0; row < actual; ++row) {
                    const int selected = static_cast<int>(std::lround(routes[row]));
                    ++metrics.router_count;
                    metrics.router_correct += selected == FamilyId(rows[row]->family);
                }
            }
            for (int row = 0; row < actual; ++row) {
                if (done[row]) continue;
                const float* values = logits.data() +
                    (static_cast<size_t>(row) * model.target_length + position) * vocabulary;
                const int token = static_cast<int>(std::max_element(values, values + vocabulary) - values);
                generated[row].push_back(token);
                if (token == vocab.eos()) {
                    done[row] = true;
                } else if (position + 1 < model.target_length) {
                    const size_t next = static_cast<size_t>(row) * model.target_length + position + 1;
                    decoder[next] = token;
                    target_mask[next] = 1;
                }
            }
            if (std::all_of(done.begin(), done.begin() + actual, [](bool value) { return value; })) break;
        }
        for (int row = 0; row < actual; ++row) {
            const Record& record = *rows[row];
            const std::string prediction = CleanText(vocab.Decode(generated[row]));
            const bool target_ok = prediction == CleanText(record.target);
            const bool answer_ok = ExtractAnswer(prediction) == CleanText(record.answer);
            ++metrics.count;
            metrics.target_exact += target_ok;
            metrics.answer_exact += answer_ok;
            AddMetricGroup(&metrics.by_task, record.task, target_ok, answer_ok);
            AddMetricGroup(&metrics.by_operation, record.operation, target_ok, answer_ok);
            AddMetricGroup(&metrics.by_source, record.source, target_ok, answer_ok);
            AddMetricGroup(&metrics.by_family, record.family, target_ok, answer_ok);
            AddMetricGroup(&metrics.by_task_operation,
                           (record.task.empty() ? "unknown" : record.task) + "|" +
                               (record.operation.empty() ? "unknown" : record.operation),
                           target_ok, answer_ok);
            if (!answer_ok && failure_examples++ < 8) {
                std::cout << "{\"event\":\"eval_failure\",\"task\":\""
                          << JsonEscape(record.task) << "\",\"op\":\""
                          << JsonEscape(record.operation) << "\",\"source\":\""
                          << JsonEscape(record.source) << "\",\"family\":\""
                          << JsonEscape(record.family) << "\",\"prediction\":\""
                          << JsonEscape(prediction.substr(0, 240)) << "\",\"target\":\""
                          << JsonEscape(record.target.substr(0, 240)) << "\",\"answer\":\""
                          << JsonEscape(record.answer) << "\"}\n";
            }
        }
    }
    std::cout << "{\"event\":\"eval_groups\",\"n\":" << metrics.count
              << ",\"router_n\":" << metrics.router_count
              << ",\"router_accuracy\":" << metrics.RouterAccuracy() << ',';
    PrintMetricGroup("by_task", metrics.by_task);
    std::cout << ',';
    PrintMetricGroup("by_op", metrics.by_operation);
    std::cout << ',';
    PrintMetricGroup("by_source", metrics.by_source);
    std::cout << ',';
    PrintMetricGroup("by_family", metrics.by_family);
    std::cout << ',';
    PrintMetricGroup("by_task_op", metrics.by_task_operation);
    std::cout << "}\n" << std::flush;
    return metrics;
}

struct TrainingState {
    int64_t training_step = 0;
    int64_t scheduler_total_updates = 0;
    double best_answer = -1.0;
    double best_target = -1.0;
};

struct ArtifactContract {
    std::string config_json;
    std::string vocab_json;
};

int UpdatesPerEpoch(const Options& options, const Dataset& dataset) {
    const int physical_batches = static_cast<int>(dataset.train.size() / options.batch_size);
    const int complete_updates = physical_batches / options.gradient_accumulation_steps;
    return std::min(complete_updates,
                    options.max_steps > 0 ? options.max_steps
                                          : std::numeric_limits<int>::max());
}

int BatchesPerEpoch(const Options& options, const Dataset& dataset) {
    const int updates = UpdatesPerEpoch(options, dataset);
    if (updates <= 0) {
        throw std::runtime_error(
            "not enough records for one complete effective batch; reduce --batch-size or "
            "--accumulation-steps");
    }
    return updates * options.gradient_accumulation_steps;
}

int64_t SchedulerTotalUpdates(const Options& options, const Dataset& dataset) {
    const int updates = UpdatesPerEpoch(options, dataset);
    if (updates <= 0) {
        throw std::runtime_error("scheduler has no complete effective batch");
    }
    return std::max<int64_t>(1, static_cast<int64_t>(updates) * options.epochs);
}

api::TrainingCheckpointInfo SaveCheckpoint(const GrpcFfiClient& client, const std::string& model_id,
                                           const fs::path& directory, int epoch,
                                           const Metrics& metrics, const CharVocab& vocab,
                                           const TrainingState& state,
                                           const ArtifactContract& contract) {
    api::SaveTrainingCheckpointRequest request;
    request.set_model_id(model_id);
    request.set_directory(fs::absolute(directory).string());
    request.set_overwrite(true);
    AppendStringEntry(request.mutable_metadata(), "epoch", std::to_string(epoch));
    AppendStringEntry(request.mutable_metadata(), "answer_exact",
                      std::to_string(metrics.AnswerExact()));
    AppendStringEntry(request.mutable_metadata(), "target_exact",
                      std::to_string(metrics.TargetExact()));
    AppendStringEntry(request.mutable_metadata(), "router_accuracy",
                      std::to_string(metrics.RouterAccuracy()));
    AppendStringEntry(request.mutable_metadata(), "caller_vocab", "../vocab.json");
    AppendStringEntry(request.mutable_metadata(), "caller_config", "../config.json");
    AppendStringEntry(request.mutable_metadata(), "vocab_size", std::to_string(vocab.size()));
    // Caller-owned state is embedded as metadata so a checkpoint remains
    // self-describing even when it is copied away from its output directory.
    // Do not use the tokenizer field: the native tokenizer expects its own
    // binary format, not this example's character-vocabulary JSON.
    AppendStringEntry(request.mutable_metadata(), "caller_vocab_json", contract.vocab_json);
    AppendStringEntry(request.mutable_metadata(), "caller_config_json", contract.config_json);
    AppendStringEntry(request.mutable_metadata(), "scheduler_policy", "cosine.v1");
    AppendStringEntry(request.mutable_metadata(), "scheduler_total_updates",
                      std::to_string(state.scheduler_total_updates));
    AppendStringEntry(request.mutable_metadata(), "best_answer_exact",
                      std::to_string(state.best_answer));
    AppendStringEntry(request.mutable_metadata(), "best_target_exact",
                      std::to_string(state.best_target));
    return client.Unary<api::SaveTrainingCheckpointRequest, api::TrainingCheckpointInfo>(
        Method("SaveTrainingCheckpoint"), request);
}

struct TrainResult {
    Metrics metrics;
    TrainingState state;
};

TrainResult Train(const GrpcFfiClient& client, const Options& options, const CharVocab& vocab,
                  const ModelPlan& model, const std::string& model_id, Dataset* dataset,
                  int start_epoch, TrainingState state,
                  const ArtifactContract& contract) {
    int64_t applied_updates = state.training_step;
    TrainResult result;
    const int batches_per_epoch = BatchesPerEpoch(options, *dataset);
    const int64_t expected_total_updates = SchedulerTotalUpdates(options, *dataset);
    if (state.scheduler_total_updates == 0) {
        state.scheduler_total_updates = expected_total_updates;
    } else if (state.scheduler_total_updates != expected_total_updates) {
        throw std::runtime_error("resume scheduler horizon does not match the caller contract");
    }
    if (applied_updates < 0 || applied_updates > state.scheduler_total_updates) {
        throw std::runtime_error("checkpoint optimizer step is outside the cosine schedule");
    }
    for (int epoch = start_epoch; epoch < options.epochs; ++epoch) {
        // Reconstruct each epoch's order from a canonical record order. This
        // makes an epoch-boundary resume consume the same batches as an
        // uninterrupted run. Image preprocessing itself is deterministic.
        std::sort(dataset->train.begin(), dataset->train.end(), [](const Record& a, const Record& b) {
            return std::tie(a.image, a.receipt_id, a.task, a.question, a.target, a.answer,
                            a.family, a.field, a.operation, a.source) <
                   std::tie(b.image, b.receipt_id, b.task, b.question, b.target, b.answer,
                            b.family, b.field, b.operation, b.source);
        });
        std::mt19937 random(options.seed + epoch * 7919);
        std::shuffle(dataset->train.begin(), dataset->train.end(), random);
        double loss_sum = 0;
        int completed_loss_windows = 0;
        const auto epoch_start = std::chrono::steady_clock::now();
        for (int step = 0; step < batches_per_epoch; ++step) {
            const int window_start = (step / options.gradient_accumulation_steps) *
                                     options.gradient_accumulation_steps;
            const int window_end = std::min(batches_per_epoch,
                                            window_start + options.gradient_accumulation_steps);
            int token_normalizer = 0;
            for (int microbatch = window_start; microbatch < window_end; ++microbatch) {
                for (int row = 0; row < options.batch_size; ++row) {
                    token_normalizer += ActiveTargetCount(
                        vocab, dataset->train[static_cast<size_t>(microbatch) * options.batch_size + row],
                        options.max_output_length);
                }
            }
            const int router_normalizer = (window_end - window_start) * options.batch_size;
            std::vector<const Record*> rows;
            for (int row = 0; row < options.batch_size; ++row) {
                rows.push_back(&dataset->train[static_cast<size_t>(step) * options.batch_size + row]);
            }
            PreparedBatch batch = PrepareBatch(options, vocab, rows, model.target_length,
                                               model.memory_length, true);
            const int64_t scheduled_update = applied_updates;
            const double progress = static_cast<double>(scheduled_update) /
                                    state.scheduler_total_updates;
            const float learning_rate = options.learning_rate * 0.5f *
                static_cast<float>(1.0 + std::cos(3.14159265358979323846 * progress));
            api::TrainStepResponse response = TrainOneBatch(
                client, options, model, model_id, std::move(batch), learning_rate,
                static_cast<float>(token_normalizer), static_cast<float>(router_normalizer), false);
            if (!std::isfinite(response.loss())) throw std::runtime_error("TrainStep returned non-finite loss");
            const int microbatch_in_window = step - window_start + 1;
            const bool expected_update = microbatch_in_window == options.gradient_accumulation_steps;
            if (response.accumulated_microbatches() != microbatch_in_window ||
                response.accumulation_steps() != options.gradient_accumulation_steps ||
                response.update_applied() != expected_update) {
                throw std::runtime_error(
                    "TrainStep returned an inconsistent gradient-accumulation state");
            }
            if (expected_update) {
                if (response.step() != applied_updates + 1) {
                    throw std::runtime_error("TrainStep optimizer step did not advance exactly once");
                }
                applied_updates = response.step();
            } else if (response.step() != applied_updates) {
                throw std::runtime_error("TrainStep optimizer step advanced before its window closed");
            }
            // TrainStep loss is cumulative within an accumulation window.
            // Count it once when the window closes instead of double-counting
            // every prefix of the same objective.
            if (step + 1 == window_end) {
                loss_sum += response.loss();
                ++completed_loss_windows;
            }
            if (step % std::max(1, options.log_every) == 0) {
                const double seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - epoch_start).count();
                std::cout << "{\"event\":\"train\",\"epoch\":" << epoch
                          << ",\"step\":" << step << ",\"optimizer_step\":"
                          << response.step() << ",\"window_loss_so_far\":" << response.loss()
                          << ",\"accumulated_microbatches\":"
                          << response.accumulated_microbatches()
                          << ",\"lr\":" << learning_rate << ",\"update_applied\":"
                          << (response.update_applied() ? "true" : "false")
                          << ",\"seconds\":" << seconds << "}\n" << std::flush;
            }
        }
        result.metrics = Evaluate(client, options, vocab, model, model_id, dataset->validation);
        state.training_step = applied_updates;
        const double average_loss = loss_sum / std::max(1, completed_loss_windows);
        std::cout << "{\"event\":\"eval\",\"epoch\":" << epoch
                  << ",\"n\":" << result.metrics.count << ",\"loss\":" << average_loss
                  << ",\"target_exact\":" << result.metrics.TargetExact()
                  << ",\"answer_exact\":" << result.metrics.AnswerExact()
                  << ",\"router_accuracy\":" << result.metrics.RouterAccuracy()
                  << ",\"by_family\":{";
        bool first = true;
        for (const auto& [family, values] : result.metrics.by_family) {
            if (!first) std::cout << ',';
            first = false;
            std::cout << '"' << JsonEscape(family) << "\":{\"n\":" << values[0]
                      << ",\"target_exact\":" << static_cast<double>(values[1]) / std::max(1, values[0])
                      << ",\"answer_exact\":" << static_cast<double>(values[2]) / std::max(1, values[0])
                      << '}';
        }
        std::cout << "}}\n" << std::flush;
        const bool best = result.metrics.AnswerExact() > state.best_answer ||
            (result.metrics.AnswerExact() == state.best_answer &&
             result.metrics.TargetExact() > state.best_target);
        if (best) {
            state.best_answer = result.metrics.AnswerExact();
            state.best_target = result.metrics.TargetExact();
        }
        SaveCheckpoint(client, model_id, options.output / "last.checkpoint", epoch,
                       result.metrics, vocab, state, contract);
        if (best) {
            SaveCheckpoint(client, model_id, options.output / "best.checkpoint", epoch,
                           result.metrics, vocab, state, contract);
        }
    }
    result.state = state;
    return result;
}

api::Tensor GetTensor(const GrpcFfiClient& client, const std::string& model_id,
                      const std::string& name) {
    api::GetTensorRequest request;
    request.set_model_id(model_id);
    request.set_name(name);
    return client.Unary<api::GetTensorRequest, api::Tensor>(Method("GetTensor"), request);
}

void SaveFullWeights(const GrpcFfiClient& client, const std::string& model_id,
                     const fs::path& path) {
    api::SaveModelWeightsRequest request;
    request.set_model_id(model_id);
    request.add_weight_paths(fs::absolute(path).string());
    request.set_atomic(true);
    client.Unary<api::SaveModelWeightsRequest, api::SaveModelWeightsResponse>(
        Method("SaveModelWeights"), request);
}

void ExportTaskAdapters(const GrpcFfiClient& client, const ModelPlan& model,
                        const std::string& model_id, const fs::path& path) {
    if (model.task_adapter_tensors.empty()) return;
    api::CreateSafetensorsRequest request;
    request.set_path(fs::absolute(path).string());
    request.set_overwrite(true);
    AppendStringEntry(request.mutable_metadata(), "format", "tiny_receipt_vqa.task_adapters.v1");
    AppendStringEntry(request.mutable_metadata(), "families",
                      "phone,address,store,item_row,item_math,item_lookup,math,other");
    AppendStringEntry(
        request.mutable_metadata(), "composition",
        "transfer tensors by name into the matching v2 graph; full_model already contains them");
    AppendStringEntry(request.mutable_metadata(), "full_model", "full_model/model.safetensors");
    for (const std::string& name : model.task_adapter_tensors) {
        *request.add_tensors() = GetTensor(client, model_id, name);
    }
    client.Unary<api::CreateSafetensorsRequest, api::SafetensorsInfo>(
        Method("CreateSafetensors"), request);
}

void FillTensorSpec(api::TensorSpec* spec, const api::Tensor& tensor) {
    spec->set_name(tensor.name());
    for (int64_t dimension : tensor.shape()) spec->add_shape(dimension);
    spec->set_dtype(tensor.dtype());
    spec->set_size_bytes(tensor.data().size());
}

api::AdapterVersionRef ExportLoraAdapter(const GrpcFfiClient& client, const Options& options,
                                         const ModelPlan& model, const std::string& model_id,
                                         const fs::path& path) {
    if (model.lora_bindings.empty()) return {};
    api::StageAdapterRequest stage;
    stage.set_model_id(model_id);
    stage.set_activate(false);
    api::AdapterManifest* manifest = stage.mutable_manifest();
    manifest->set_format("volvox.adapter.v1");
    manifest->set_adapter_id("tiny-receipt-vqa-lora");
    manifest->set_kind(api::ADAPTER_LORA);
    AppendStringEntry(manifest->mutable_metadata(), "source", "tiny_receipt_vqa_train.cc");
    AppendStringEntry(manifest->mutable_metadata(), "paired_base", "lora_base/model.safetensors");
    AppendStringEntry(
        manifest->mutable_metadata(), "composition",
        "load lora_base/config.json + lora_base/model.safetensors, then activate this adapter");
    std::unordered_map<std::string, api::Tensor> tensors;
    for (const LoraBinding& binding : model.lora_bindings) {
        tensors.try_emplace(binding.a_tensor, GetTensor(client, model_id, binding.a_tensor));
        tensors.try_emplace(binding.b_tensor, GetTensor(client, model_id, binding.b_tensor));
        api::AdapterTarget* target = manifest->add_targets();
        target->set_target_id(binding.target_id);
        target->set_op_id(binding.op_id);
        target->set_base_tensor(binding.base_tensor);
        target->set_adapter_layout(api::ADAPTER_MATRIX_LAYOUT_CANONICAL);
        target->set_rank(options.lora_rank);
        target->set_alpha(options.lora_alpha);
        target->set_scale(options.lora_alpha / options.lora_rank);
        api::AdapterTensorBinding* a = target->add_tensors();
        a->set_role(api::ADAPTER_TENSOR_A);
        a->set_tensor_name(binding.a_tensor);
        FillTensorSpec(a->mutable_spec(), tensors.at(binding.a_tensor));
        api::AdapterTensorBinding* b = target->add_tensors();
        b->set_role(api::ADAPTER_TENSOR_B);
        b->set_tensor_name(binding.b_tensor);
        FillTensorSpec(b->mutable_spec(), tensors.at(binding.b_tensor));
    }
    for (auto& [name, tensor] : tensors) *stage.add_tensors() = std::move(tensor);
    api::AdapterInfo info = client.Unary<api::StageAdapterRequest, api::AdapterInfo>(
        Method("StageAdapter"), stage);
    if (!info.has_adapter()) throw std::runtime_error("StageAdapter did not return a version");
    api::SaveAdapterRequest save;
    save.set_model_id(model_id);
    *save.mutable_adapter() = info.adapter();
    save.set_path(fs::absolute(path).string());
    save.set_atomic(true);
    client.Unary<api::SaveAdapterRequest, api::SaveAdapterResponse>(Method("SaveAdapter"), save);
    return info.adapter();
}

void WriteText(const fs::path& path, const std::string& text) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot write " + path.string());
    stream << text;
    if (!stream) throw std::runtime_error("failed to write " + path.string());
}

fs::path StableAbsolutePath(const fs::path& path) {
    std::error_code error;
    fs::path normalized = fs::weakly_canonical(path, error);
    if (!error) return normalized;
    return fs::absolute(path).lexically_normal();
}

bool IsSameOrDescendant(const fs::path& candidate, const fs::path& directory) {
    const fs::path child = StableAbsolutePath(candidate);
    const fs::path parent = StableAbsolutePath(directory);
    const auto mismatch = std::mismatch(parent.begin(), parent.end(), child.begin(), child.end());
    return mismatch.first == parent.end();
}

std::string OptionsJson(const Options& options, size_t training_records,
                        size_t validation_records, int vocabulary,
                        const std::string& dataset_fingerprint) {
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    output << "{\n"
           << "  \"format\": \"tiny_receipt_vqa.volvox.v2\",\n"
           << "  \"task_profile\": \"" << JsonEscape(options.task_profile) << "\",\n"
           << "  \"target_mode\": \"" << JsonEscape(options.target_mode) << "\",\n"
           << "  \"evaluation_routing\": \"" << JsonEscape(options.evaluation_routing)
           << "\",\n"
           << "  \"preprocessing\": \"" << kPreprocessingFormat << "\",\n"
           << "  \"gpu_memory_mb\": " << options.gpu_memory_mb << ",\n"
           << "  \"train_records\": " << training_records << ",\n"
           << "  \"validation_records\": " << validation_records << ",\n"
           << "  \"vocabulary\": " << vocabulary << ",\n"
           << "  \"dataset_fingerprint\": \"" << dataset_fingerprint << "\",\n"
           << "  \"synth_limit\": " << options.synth_limit << ",\n"
           << "  \"validation_synth_limit\": " << options.validation_synth_limit << ",\n"
           << "  \"tasks_per_synth\": " << options.tasks_per_synth << ",\n"
           << "  \"real_repeat\": " << options.real_repeat << ",\n"
           << "  \"real_validation_fraction\": " << options.real_validation_fraction << ",\n"
           << "  \"epochs\": " << options.epochs << ",\n"
           << "  \"batch_size\": " << options.batch_size << ",\n"
           << "  \"max_steps\": " << options.max_steps << ",\n"
           << "  \"evaluation_batches\": " << options.evaluation_batches << ",\n"
           << "  \"seed\": " << options.seed << ",\n"
           << "  \"learning_rate\": " << options.learning_rate << ",\n"
           << "  \"weight_decay\": " << options.weight_decay << ",\n"
           << "  \"gradient_accumulation_steps\": "
           << options.gradient_accumulation_steps << ",\n"
           << "  \"d_model\": " << options.d_model << ",\n"
           << "  \"heads\": " << options.heads << ",\n"
           << "  \"encoder_layers\": " << options.encoder_layers << ",\n"
           << "  \"decoder_layers\": " << options.decoder_layers << ",\n"
           << "  \"ff_multiplier\": " << options.ff_multiplier << ",\n"
           << "  \"dropout\": " << options.dropout << ",\n"
           << "  \"max_question_length\": " << options.max_question_length << ",\n"
           << "  \"max_output_length\": " << options.max_output_length << ",\n"
           << "  \"task_adapters\": " << (options.task_adapters ? "true" : "false") << ",\n"
           << "  \"router\": " << (options.router ? "true" : "false") << ",\n"
           << "  \"adapter_bottleneck\": " << options.adapter_bottleneck << ",\n"
           << "  \"lora_rank\": " << options.lora_rank << ",\n"
           << "  \"lora_alpha\": " << options.lora_alpha << ",\n"
           << "  \"lora_dropout\": " << options.lora_dropout << ",\n"
           << "  \"freeze_base_for_lora\": "
           << (options.freeze_base_for_lora ? "true" : "false") << "\n"
           << "}\n";
    return output.str();
}

ArtifactContract LoadCheckpointContract(const fs::path& directory) {
    const fs::path manifest_path = directory / "manifest.json";
    Json manifest = ParseJson(ReadFile(manifest_path), manifest_path.string());
    cJSON* metadata = Member(manifest.get(), "metadata");
    if (!cJSON_IsObject(metadata)) {
        throw std::runtime_error("checkpoint manifest has no metadata object");
    }
    ArtifactContract contract;
    contract.config_json = JsonText(metadata, "caller_config_json");
    contract.vocab_json = JsonText(metadata, "caller_vocab_json");
    if (contract.config_json.empty() || contract.vocab_json.empty()) {
        throw std::runtime_error(
            "checkpoint lacks the embedded caller config/vocabulary contract");
    }
    return contract;
}

const std::string& RequiredCheckpointMetadata(
    const google::protobuf::RepeatedPtrField<api::StringEntry>& metadata,
    const std::string& key) {
    for (int index = metadata.size() - 1; index >= 0; --index) {
        const api::StringEntry& entry = metadata.Get(index);
        if (entry.key() == key) {
            if (entry.value().empty()) break;
            return entry.value();
        }
    }
    throw std::runtime_error("checkpoint metadata is missing " + key);
}

void WriteModelPackageMetadata(const fs::path& directory, const fs::path& graph_config,
                               const ArtifactContract& contract) {
    fs::create_directories(directory);
    std::error_code error;
    fs::copy_file(graph_config, directory / "config.json",
                  fs::copy_options::overwrite_existing, error);
    if (error) {
        throw std::runtime_error("copy model graph config failed: " + error.message());
    }
    WriteText(directory / "vocab.json", contract.vocab_json + "\n");
}

std::vector<float> SnapshotLogits(const GrpcFfiClient& client, const Options& options,
                                  const CharVocab& vocab, const ModelPlan& model,
                                  const std::string& model_id,
                                  const std::vector<Record>& records) {
    if (records.empty()) throw std::runtime_error("cannot validate artifacts without records");
    std::vector<const Record*> rows;
    const size_t count = std::min<size_t>(options.batch_size, records.size());
    for (size_t index = 0; index < count; ++index) rows.push_back(&records[index]);
    while (static_cast<int>(rows.size()) < options.batch_size) rows.push_back(rows.front());
    PreparedBatch prepared = PrepareBatch(options, vocab, rows, model.target_length,
                                          model.memory_length, false);
    api::RunRequest request;
    request.set_model_id(model_id);
    for (api::Tensor& input : prepared.inputs) *request.add_inputs() = std::move(input);
    request.add_output_names(model.logits);
    api::RunResponse response =
        client.Unary<api::RunRequest, api::RunResponse>(Method("Run"), request);
    if (response.outputs_size() != 1 || response.outputs(0).name() != model.logits) {
        throw std::runtime_error("artifact validation Run did not return receipt logits");
    }
    return TensorF32(response.outputs(0));
}

void RequireEquivalentLogits(const std::vector<float>& expected,
                             const std::vector<float>& actual) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("base+LoRA validation returned a different logits shape");
    }
    float largest_reference = 0.0f;
    float largest_error = 0.0f;
    for (size_t index = 0; index < expected.size(); ++index) {
        if (!std::isfinite(expected[index]) || !std::isfinite(actual[index])) {
            throw std::runtime_error("base+LoRA validation returned non-finite logits");
        }
        largest_reference = std::max(largest_reference, std::abs(expected[index]));
        largest_error = std::max(largest_error, std::abs(expected[index] - actual[index]));
    }
    const float tolerance = 5.0e-4f * (1.0f + largest_reference);
    if (largest_error > tolerance) {
        std::ostringstream error;
        error << "base+LoRA logits mismatch: max error " << largest_error
              << " exceeds " << tolerance;
        throw std::runtime_error(error.str());
    }
    std::cout << "{\"event\":\"artifact_validation\",\"kind\":\"base_plus_lora\","
              << "\"max_abs_error\":" << largest_error << "}\n" << std::flush;
}

void ZeroInlineLora(const GrpcFfiClient& client, const ModelPlan& model,
                    const std::string& model_id) {
    std::unordered_set<std::string> names;
    for (const LoraBinding& binding : model.lora_bindings) {
        names.insert(binding.a_tensor);
        names.insert(binding.b_tensor);
    }
    for (const std::string& name : names) {
        api::Tensor tensor = GetTensor(client, model_id, name);
        tensor.mutable_data()->assign(tensor.data().size(), '\0');
        api::SetTensorRequest request;
        request.set_model_id(model_id);
        *request.mutable_tensor() = std::move(tensor);
        client.Unary<api::SetTensorRequest, api::TensorSpec>(Method("SetTensor"), request);
    }
}

void RemoveAdapter(const GrpcFfiClient& client, const std::string& model_id,
                   const api::AdapterVersionRef& adapter) {
    if (adapter.version_id().empty()) return;
    api::RemoveAdapterRequest request;
    request.set_model_id(model_id);
    *request.mutable_adapter() = adapter;
    client.Unary<api::RemoveAdapterRequest, api::Empty>(
        Method("RemoveAdapter"), request);
}

void UnloadModel(const GrpcFfiClient& client, std::string* model_id) {
    if (!model_id || model_id->empty()) return;
    api::ModelRef request;
    request.set_model_id(*model_id);
    client.Unary<api::ModelRef, api::Empty>(Method("UnloadModel"), request);
    model_id->clear();
}

std::string LoadPackagedModel(const GrpcFfiClient& client, const Options& options,
                              const fs::path& directory) {
    api::LoadModelRequest request;
    api::ModelPaths* paths = request.mutable_paths();
    paths->set_config_path(fs::absolute(directory / "config.json").string());
    paths->set_weights_path(fs::absolute(directory / "model.safetensors").string());
    request.mutable_exec()->set_backend(BackendValue(options.backend));
    if (options.debug) request.mutable_exec()->set_debug(true);
    api::LoadModelResponse response = client.Unary<api::LoadModelRequest, api::LoadModelResponse>(
        Method("LoadModel"), request);
    if (response.model_id().empty()) throw std::runtime_error("LoadModel returned an empty id");
    return response.model_id();
}

void LoadAndActivateLora(const GrpcFfiClient& client, const std::string& model_id,
                         const fs::path& path) {
    api::LoadAdapterRequest request;
    request.set_model_id(model_id);
    request.set_path(fs::absolute(path).string());
    request.set_adapter_id("tiny-receipt-vqa-lora");
    request.set_activate(true);
    api::AdapterInfo response = client.Unary<api::LoadAdapterRequest, api::AdapterInfo>(
        Method("LoadAdapter"), request);
    if (!response.has_adapter() || !response.active()) {
        throw std::runtime_error("saved LoRA adapter did not reload as active");
    }
}

}  // namespace

int main(int argc, char** argv) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    std::string model_id;
    std::unique_ptr<GrpcFfiClient> client;
    try {
        Options options = ParseOptions(argc, argv);
        if (!options.resume.empty() && IsSameOrDescendant(options.output, options.resume)) {
            throw std::runtime_error(
                "--out must be a run directory outside the checkpoint passed to --resume "
                "(the checkpoint may be a child of --out)");
        }
        fs::create_directories(options.output);
        Dataset dataset = LoadDataset(options);
        const std::string dataset_fingerprint = DatasetFingerprint(dataset);
        CharVocab dataset_vocab = CharVocab::Build(dataset);
        CharVocab vocab = dataset_vocab;
        ArtifactContract contract{
            OptionsJson(options, dataset.train.size(), dataset.validation.size(), vocab.size(),
                        dataset_fingerprint),
            vocab.ToJson(),
        };
        if (!options.resume.empty()) {
            const ArtifactContract saved = LoadCheckpointContract(options.resume);
            CharVocab saved_vocab = CharVocab::FromJson(
                saved.vocab_json, (options.resume / "manifest.json").string());
            if (saved_vocab.ToJson() != dataset_vocab.ToJson()) {
                throw std::runtime_error(
                    "resume dataset vocabulary differs from the checkpoint vocabulary");
            }
            const std::string requested_config = OptionsJson(
                options, dataset.train.size(), dataset.validation.size(), saved_vocab.size(),
                dataset_fingerprint);
            if (saved.config_json != requested_config) {
                throw std::runtime_error(
                    "resume caller/model settings differ from the checkpoint contract");
            }
            vocab = std::move(saved_vocab);
            contract = saved;
        }
        // Validation above happens before either file can be overwritten.
        WriteText(options.output / "vocab.json", contract.vocab_json + "\n");
        WriteText(options.output / "config.json", contract.config_json);

        ReceiptGraphBuilder builder(options, vocab);
        ModelPlan model = builder.Finish();
        std::cout << "{\"event\":\"graph\",\"nodes\":" << model.request.graph().nodes_size()
                  << ",\"weights\":" << model.request.tensors_size()
                  << ",\"trainable\":" << model.trainable_tensors.size()
                  << ",\"lora_targets\":" << model.lora_bindings.size()
                  << ",\"vocabulary\":" << vocab.size()
                  << ",\"request_bytes\":" << model.request.ByteSizeLong() << "}\n";
        ValidateGraphResources(options, model.request);
        if (options.dry_run) {
            WriteText(options.output / "create_model.pb", model.request.SerializeAsString());
            std::cout << "{\"event\":\"done\",\"reason\":\"dry_run\"}\n";
            return 0;
        }

        client = std::make_unique<GrpcFfiClient>(options.library.string());
        api::Empty empty;
        api::PingResponse ping =
            client->Unary<api::Empty, api::PingResponse>(Method("Ping"), empty);
        std::cout << "{\"event\":\"ping\",\"version\":\"" << JsonEscape(ping.version())
                  << "\"}\n";
        int start_epoch = 0;
        TrainingState initial_state;
        if (!options.resume.empty()) {
            api::LoadTrainingCheckpointRequest request;
            request.set_directory(fs::absolute(options.resume).string());
            request.mutable_exec()->set_backend(BackendValue(options.backend));
            if (options.debug) request.mutable_exec()->set_debug(true);
            api::LoadTrainingCheckpointResponse response = client->Unary<
                api::LoadTrainingCheckpointRequest, api::LoadTrainingCheckpointResponse>(
                    Method("LoadTrainingCheckpoint"), request);
            if (!response.has_model() || !response.has_checkpoint()) {
                throw std::runtime_error("LoadTrainingCheckpoint returned an incomplete response");
            }
            model_id = response.model().model_id();
            const auto& metadata = response.checkpoint().metadata();
            if (RequiredCheckpointMetadata(metadata, "caller_config_json") !=
                    contract.config_json ||
                RequiredCheckpointMetadata(metadata, "caller_vocab_json") !=
                    contract.vocab_json) {
                throw std::runtime_error("loaded checkpoint caller contract changed during load");
            }
            if (RequiredCheckpointMetadata(metadata, "scheduler_policy") != "cosine.v1") {
                throw std::runtime_error("checkpoint uses an unsupported scheduler policy");
            }
            start_epoch = ParseInt(RequiredCheckpointMetadata(metadata, "epoch"),
                                   "checkpoint epoch") + 1;
            initial_state.training_step = response.checkpoint().training_step();
            initial_state.scheduler_total_updates = ParseInt64(
                RequiredCheckpointMetadata(metadata, "scheduler_total_updates"),
                "scheduler_total_updates");
            initial_state.best_answer = ParseDouble(
                RequiredCheckpointMetadata(metadata, "best_answer_exact"),
                "best_answer_exact");
            initial_state.best_target = ParseDouble(
                RequiredCheckpointMetadata(metadata, "best_target_exact"),
                "best_target_exact");
            const int64_t expected_horizon = SchedulerTotalUpdates(options, dataset);
            const int64_t expected_checkpoint_step =
                static_cast<int64_t>(UpdatesPerEpoch(options, dataset)) * start_epoch;
            if (initial_state.scheduler_total_updates != expected_horizon ||
                initial_state.training_step < 0 ||
                initial_state.training_step > expected_horizon ||
                initial_state.training_step != expected_checkpoint_step) {
                throw std::runtime_error(
                    "checkpoint optimizer/scheduler state is inconsistent with its caller contract");
            }
            if (start_epoch < 0 || start_epoch > options.epochs) {
                throw std::runtime_error("checkpoint epoch is outside the configured training run");
            }
        } else {
            api::LoadModelResponse response = client->Unary<api::CreateModelRequest, api::LoadModelResponse>(
                Method("CreateModel"), model.request);
            model_id = response.model_id();
        }
        if (model_id.empty()) throw std::runtime_error("model RPC returned an empty model_id");
        std::cout << "{\"event\":\"model_ready\",\"model_id\":\""
                  << JsonEscape(model_id) << "\",\"backend\":\""
                  << JsonEscape(options.backend) << "\",\"start_epoch\":" << start_epoch << "}\n";

        TrainResult training;
        if (start_epoch < options.epochs) {
            training = Train(*client, options, vocab, model, model_id, &dataset,
                             start_epoch, initial_state, contract);
        } else {
            training.metrics = Evaluate(*client, options, vocab, model, model_id,
                                        dataset.validation);
            training.state = initial_state;
            SaveCheckpoint(*client, model_id, options.output / "last.checkpoint",
                           std::max(0, start_epoch - 1), training.metrics, vocab,
                           training.state, contract);
        }

        // The standalone full package contains the trained inline LoRA and task
        // adapters. The separate LoRA artifact is paired with a second package
        // whose inline LoRA tensors are zero, so applying it cannot double the
        // learned delta.
        const fs::path graph_config = options.output / "last.checkpoint/config.json";
        const fs::path full_model = options.output / "full_model";
        WriteModelPackageMetadata(full_model, graph_config, contract);
        SaveFullWeights(*client, model_id, full_model / "model.safetensors");
        ExportTaskAdapters(*client, model, model_id, options.output / "task_adapters.safetensors");
        if (!model.lora_bindings.empty()) {
            const std::vector<float> expected = SnapshotLogits(
                *client, options, vocab, model, model_id, dataset.validation);
            const fs::path lora_path = options.output / "lora.safetensors";
            const api::AdapterVersionRef staged =
                ExportLoraAdapter(*client, options, model, model_id, lora_path);
            RemoveAdapter(*client, model_id, staged);
            ZeroInlineLora(*client, model, model_id);
            const fs::path lora_base = options.output / "lora_base";
            WriteModelPackageMetadata(lora_base, graph_config, contract);
            SaveFullWeights(*client, model_id, lora_base / "model.safetensors");

            // Validate the files, not merely the staged in-memory tensors.
            UnloadModel(*client, &model_id);
            model_id = LoadPackagedModel(*client, options, lora_base);
            LoadAndActivateLora(*client, model_id, lora_path);
            const std::vector<float> actual = SnapshotLogits(
                *client, options, vocab, model, model_id, dataset.validation);
            RequireEquivalentLogits(expected, actual);
        }

        std::cout << "{\"event\":\"done\",\"training_step\":"
                  << training.state.training_step
                  << ",\"answer_exact\":" << training.metrics.AnswerExact()
                  << ",\"target_exact\":" << training.metrics.TargetExact()
                  << ",\"model\":\""
                  << JsonEscape((full_model / "model.safetensors").string())
                  << "\",\"checkpoint\":\""
                  << JsonEscape((options.output / "last.checkpoint").string()) << "\"}\n";

        UnloadModel(*client, &model_id);
        google::protobuf::ShutdownProtobufLibrary();
        return 0;
    } catch (const std::exception& error) {
        if (client && !model_id.empty()) {
            try {
                api::ModelRef unload;
                unload.set_model_id(model_id);
                client->Unary<api::ModelRef, api::Empty>(Method("UnloadModel"), unload);
            } catch (...) {
            }
        }
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
