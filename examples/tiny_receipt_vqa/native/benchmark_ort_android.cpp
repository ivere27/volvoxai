/* Android ONNX Runtime CPU benchmark for the TinyReceipt explicit-KV ABI.
 *
 * This intentionally stays outside the VolvoxAI native inference composition.
 * It links against the official onnxruntime-android library and exists only to
 * produce an on-device reference observation for the example benchmark.
 */
extern "C" {
#include "tiny_receipt_image.h"
}

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int64_t kQuestionIds[] = {1038, 54, 1124, 54, 1181, 54, 1031, 2};
constexpr size_t kLayerTensorCount = 8;
constexpr size_t kVocabularySize = 1536;
static_assert(sizeof(bool) == 1,
              "ONNX BOOL tensor storage requires one-byte bool");

constexpr const char *kEncoderInputs[] = {
    "image",
    "question_ids",
    "family_ids",
};
constexpr const char *kEncoderOutputs[] = {
    "memory",    "memory_padding_mask", "router_logits", "selected_family_ids",
    "cross_k_0", "cross_v_0",           "cross_k_1",     "cross_v_1",
    "cross_k_2", "cross_v_2",           "cross_k_3",     "cross_v_3",
};
constexpr const char *kDecoderInputs[] = {
    "decoder_input_ids", "position_ids", "family_ids", "memory_padding_mask",
    "past_padding_mask", "cross_k_0",    "cross_v_0",  "cross_k_1",
    "cross_v_1",         "cross_k_2",    "cross_v_2",  "cross_k_3",
    "cross_v_3",         "past_k_0",     "past_v_0",   "past_k_1",
    "past_v_1",          "past_k_2",     "past_v_2",   "past_k_3",
    "past_v_3",
};
constexpr const char *kDecoderOutputs[] = {
    "logits",      "present_padding_mask", "present_k_0", "present_v_0",
    "present_k_1", "present_v_1",          "present_k_2", "present_v_2",
    "present_k_3", "present_v_3",
};

struct RequestResult {
  double encoder_ms = 0.0;
  std::vector<double> decoder_ms;
  std::vector<int64_t> tokens;
};

double now_ms() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(
             clock::now().time_since_epoch())
      .count();
}

[[noreturn]] void fail(const std::string &message) {
  std::fprintf(stderr, "benchmark_ort_android: %s\n", message.c_str());
  std::exit(1);
}

std::vector<int64_t> shape_of(const Ort::Value &value) {
  return value.GetTensorTypeAndShapeInfo().GetShape();
}

void require_shape(const Ort::Value &value,
                   const std::vector<int64_t> &expected, const char *label) {
  if (shape_of(value) != expected)
    fail(std::string(label) + " has an invalid shape");
}

template <typename T>
Ort::Value tensor(Ort::MemoryInfo &memory, T *data, size_t count,
                  const std::vector<int64_t> &shape) {
  return Ort::Value::CreateTensor<T>(memory, data, count, shape.data(),
                                     shape.size());
}

RequestResult execute_request(Ort::Session &encoder, Ort::Session &decoder,
                              Ort::MemoryInfo &memory,
                              std::vector<float> &image) {
  int64_t family_id = 0;
  std::vector<int64_t> question(std::begin(kQuestionIds),
                                std::end(kQuestionIds));
  std::vector<Ort::Value> encoder_inputs;
  encoder_inputs.emplace_back(
      tensor(memory, image.data(), image.size(), {1, 1, 320, 672}));
  encoder_inputs.emplace_back(
      tensor(memory, question.data(), question.size(), {1, 8}));
  encoder_inputs.emplace_back(tensor(memory, &family_id, 1, {1}));

  const double encoder_started = now_ms();
  std::vector<Ort::Value> encoded =
      encoder.Run(Ort::RunOptions{nullptr}, kEncoderInputs,
                  encoder_inputs.data(), encoder_inputs.size(), kEncoderOutputs,
                  sizeof(kEncoderOutputs) / sizeof(kEncoderOutputs[0]));
  RequestResult result;
  result.encoder_ms = now_ms() - encoder_started;
  if (encoded.size() != 12)
    fail("encoder output count is not 12");
  require_shape(encoded[0], {1, 218, 320}, "memory");
  require_shape(encoded[1], {1, 218}, "memory_padding_mask");
  require_shape(encoded[2], {1, 8}, "router_logits");
  require_shape(encoded[3], {1}, "selected_family_ids");
  if (encoded[3].GetTensorData<int64_t>()[0] != 0) {
    fail("encoder did not preserve requested phone family");
  }
  const bool *memory_mask = encoded[1].GetTensorData<bool>();
  for (size_t index = 0; index < kLayerTensorCount; ++index) {
    require_shape(encoded[4 + index], {1, 8, 218, 40}, "cross cache");
    const float *values = encoded[4 + index].GetTensorData<float>();
    const size_t count = 8u * 218u * 40u;
    for (size_t item = 0; item < count; ++item) {
      if (!std::isfinite(values[item]))
        fail("encoder cross cache is not finite");
    }
  }

  std::vector<uint8_t> past_mask(1, 1);
  std::vector<std::vector<float>> past(kLayerTensorCount,
                                       std::vector<float>(8u * 40u, 0.0f));
  int64_t current = 1;
  for (int64_t position = 0; position < 4; ++position) {
    const int64_t past_length = position + 1;
    std::vector<Ort::Value> inputs;
    inputs.reserve(21);
    inputs.emplace_back(tensor(memory, &current, 1, {1, 1}));
    inputs.emplace_back(tensor(memory, &position, 1, {1}));
    inputs.emplace_back(tensor(memory, &family_id, 1, {1}));
    inputs.emplace_back(
        tensor(memory, const_cast<bool *>(memory_mask), 218, {1, 218}));
    inputs.emplace_back(tensor(memory,
                               reinterpret_cast<bool *>(past_mask.data()),
                               past_mask.size(), {1, past_length}));
    for (size_t index = 0; index < kLayerTensorCount; ++index) {
      float *values = encoded[4 + index].GetTensorMutableData<float>();
      inputs.emplace_back(
          tensor(memory, values, 8u * 218u * 40u, {1, 8, 218, 40}));
    }
    for (size_t index = 0; index < kLayerTensorCount; ++index) {
      inputs.emplace_back(tensor(memory, past[index].data(), past[index].size(),
                                 {1, 8, past_length, 40}));
    }

    const double decoder_started = now_ms();
    std::vector<Ort::Value> decoded = decoder.Run(
        Ort::RunOptions{nullptr}, kDecoderInputs, inputs.data(), inputs.size(),
        kDecoderOutputs, sizeof(kDecoderOutputs) / sizeof(kDecoderOutputs[0]));
    result.decoder_ms.push_back(now_ms() - decoder_started);
    if (decoded.size() != 10)
      fail("decoder output count is not 10");
    require_shape(decoded[0], {1, 1, 1536}, "logits");
    require_shape(decoded[1], {1, past_length + 1}, "present_padding_mask");
    const float *logits = decoded[0].GetTensorData<float>();
    size_t best = 0;
    for (size_t index = 0; index < kVocabularySize; ++index) {
      if (!std::isfinite(logits[index]))
        fail("decoder logits are not finite");
      if (logits[index] > logits[best])
        best = index;
    }
    const bool *present_mask = decoded[1].GetTensorData<bool>();
    for (int64_t index = 0; index < past_length; ++index) {
      if (present_mask[index] != static_cast<bool>(past_mask[index])) {
        fail("decoder changed the padding-mask prefix");
      }
    }
    if (present_mask[past_length] != (current == 0)) {
      fail("decoder appended an invalid padding-mask value");
    }

    std::vector<std::vector<float>> next_past(kLayerTensorCount);
    const size_t present_count =
        8u * static_cast<size_t>(past_length + 1) * 40u;
    for (size_t index = 0; index < kLayerTensorCount; ++index) {
      require_shape(decoded[2 + index], {1, 8, past_length + 1, 40},
                    "present cache");
      const float *values = decoded[2 + index].GetTensorData<float>();
      for (size_t item = 0; item < present_count; ++item) {
        if (!std::isfinite(values[item]))
          fail("present cache is not finite");
      }
      for (size_t head = 0; head < 8u; ++head) {
        const float *old_head =
            past[index].data() + head * static_cast<size_t>(past_length) * 40u;
        const float *new_head =
            values + head * static_cast<size_t>(past_length + 1) * 40u;
        if (std::memcmp(new_head, old_head,
                        static_cast<size_t>(past_length) * 40u *
                            sizeof(float)) != 0) {
          fail("decoder changed a cache prefix");
        }
      }
      next_past[index].assign(values, values + present_count);
    }
    past.swap(next_past);
    past_mask.assign(reinterpret_cast<const uint8_t *>(present_mask),
                     reinterpret_cast<const uint8_t *>(present_mask) +
                         past_length + 1);
    current = static_cast<int64_t>(best);
    result.tokens.push_back(current);
  }
  return result;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  return values.size() % 2 ? values[middle]
                           : (values[middle - 1] + values[middle]) * 0.5;
}

int parse_positive(const char *value, const char *label) {
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (!value[0] || !end || *end || parsed < 1 || parsed > 1000) {
    fail(std::string(label) + " must be an integer from 1 through 1000");
  }
  return static_cast<int>(parsed);
}

} // namespace

int main(int argc, char **argv) try {
  if (argc < 5 || argc > 7) {
    std::fprintf(stderr,
                 "usage: %s <encoder.onnx> <decoder.onnx> <receipt.png> "
                 "<fp32|int8> [repeat=3] [threads=1]\n",
                 argv[0]);
    return 2;
  }
  const std::string precision = argv[4];
  if (precision != "fp32" && precision != "int8")
    fail("precision must be fp32 or int8");
  const int repeat = argc >= 6 ? parse_positive(argv[5], "repeat") : 3;
  const int threads = argc >= 7 ? parse_positive(argv[6], "threads") : 1;
  std::vector<float> image(320u * 672u);
  /* The shared loader names its single-channel storage NHWC. With C=1 its
   * flat bytes are also the ONNX model's required NCHW [1,1,320,672]. */
  const int image_shape[] = {1, 320, 672, 1};
  char image_error[256] = {0};
  if (tiny_receipt_load_image_to_tensor(argv[3], image.data(), image_shape, 4,
                                        image_error,
                                        sizeof(image_error)) != 0) {
    fail(std::string("cannot preprocess image: ") + image_error);
  }

  Ort::Env environment(ORT_LOGGING_LEVEL_WARNING,
                       "tinyreceipt-android-benchmark");
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(threads);
  options.SetInterOpNumThreads(1);
  options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  Ort::Session encoder(environment, argv[1], options);
  Ort::Session decoder(environment, argv[2], options);
  Ort::MemoryInfo memory =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  const std::vector<int64_t> expected_tokens = {4, 1038, 5, 6};
  std::vector<double> encoder_samples;
  std::vector<double> prefill_samples;
  std::vector<double> steady_samples;
  std::vector<double> decoder_samples;
  std::vector<double> component_samples;
  for (int sample = 0; sample < repeat; ++sample) {
    const RequestResult warmup =
        execute_request(encoder, decoder, memory, image);
    const RequestResult measured =
        execute_request(encoder, decoder, memory, image);
    if (warmup.tokens != expected_tokens ||
        measured.tokens != expected_tokens) {
      fail("warmup/measured token parity failed");
    }
    const double decoder_total =
        measured.decoder_ms[0] + measured.decoder_ms[1] +
        measured.decoder_ms[2] + measured.decoder_ms[3];
    const double steady_mean =
        (measured.decoder_ms[1] + measured.decoder_ms[2] +
         measured.decoder_ms[3]) /
        3.0;
    encoder_samples.push_back(measured.encoder_ms);
    prefill_samples.push_back(measured.decoder_ms[0]);
    steady_samples.push_back(steady_mean);
    decoder_samples.push_back(decoder_total);
    component_samples.push_back(measured.encoder_ms + decoder_total);
    std::printf(
        "ORT_ANDROID_SAMPLE precision=%s index=%d encoder_ms=%.3f prefill_ms=%.3f "
        "steady_mean_ms=%.3f decoder_total_ms=%.3f component_ms=%.3f "
        "tokens=4,1038,5,6 cache=P1-R2-R3-R4-R5\n",
        precision.c_str(), sample, measured.encoder_ms, measured.decoder_ms[0],
        steady_mean, decoder_total, measured.encoder_ms + decoder_total);
  }
  std::printf(
      "ORT_ANDROID_RESULT status=pass runtime=%s provider=CPUExecutionProvider "
      "precision=%s repeat=%d warmup_per_sample=1 threads=%d encoder_ms=%.3f "
      "prefill_ms=%.3f steady_mean_ms=%.3f decoder_total_ms=%.3f "
      "component_ms=%.3f tokens=4,1038,5,6 cache=P1-R2-R3-R4-R5\n",
      OrtGetApiBase()->GetVersionString(), precision.c_str(), repeat, threads,
      median(encoder_samples), median(prefill_samples), median(steady_samples),
      median(decoder_samples), median(component_samples));
  return 0;
} catch (const Ort::Exception &error) {
  std::fprintf(stderr, "benchmark_ort_android: ONNX Runtime error: %s\n",
               error.what());
  return 1;
} catch (const std::exception &error) {
  std::fprintf(stderr, "benchmark_ort_android: %s\n", error.what());
  return 1;
}
