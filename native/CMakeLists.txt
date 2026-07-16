# Native engine: profiles, tests, benchmarks and ISA codegen checks.
# Faithful port of the native section of the historical root Makefile.

set(REPO ${CMAKE_SOURCE_DIR})            # repo root (shaders/, tools/, examples/)
set(GEN  ${CMAKE_BINARY_DIR}/gen)        # generated assets (was build/native/*)
set(PY   python3)

# --- Backend toggles (mirror NATIVE_ENABLE_* defaults) ------------------------
option(VOLVOXAI_ENABLE_VULKAN "Link the Vulkan device backend" ON)
option(VOLVOXAI_ENABLE_OPENGL "Link the OpenGL device backend" ON)
if(APPLE)
  option(VOLVOXAI_ENABLE_METAL "Link the Metal device backend" ON)
else()
  option(VOLVOXAI_ENABLE_METAL "Link the Metal device backend" OFF)
endif()
option(VOLVOXAI_ENABLE_NNAPI "Link the NNAPI device backend" OFF)
set(VOLVOXAI_CPU_TARGET "baseline" CACHE STRING "baseline or avx2")

# Android release builds are NNAPI-first (mirrors the old build_android recipe).
# Cross-compile with the NDK toolchain file:
#   cmake -S . -B build/android -DCMAKE_C_COMPILER=clang \
#     -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
#     -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(VOLVOXAI_ENABLE_NNAPI ON)
endif()

# Objective-C is only needed for the Metal backend.
if(VOLVOXAI_ENABLE_METAL)
  enable_language(OBJC)
endif()

# --- Host / ISA detection -----------------------------------------------------
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _arch)
if(_arch MATCHES "x86_64|amd64|i386|i686")
  set(VOLVOX_X86 TRUE)
else()
  set(VOLVOX_X86 FALSE)
endif()
# Armv8.2 SDOT objects are used on aarch64 Linux and Android (not macOS arm64,
# matching the historical Makefile which gated on Linux/Android only).
if(_arch MATCHES "aarch64|arm64" AND
   (CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Android"))
  set(VOLVOX_ARM_DOTPROD TRUE)
else()
  set(VOLVOX_ARM_DOTPROD FALSE)
endif()

# CPU baseline flags (NATIVE_CPU_FLAGS).
set(CPU_FLAGS "")
if(VOLVOXAI_CPU_TARGET STREQUAL "avx2")
  if(NOT VOLVOX_X86)
    message(FATAL_ERROR "VOLVOXAI_CPU_TARGET=avx2 requires an x86 host")
  endif()
  set(CPU_FLAGS -mavx2 -mfma)
elseif(NOT VOLVOXAI_CPU_TARGET STREQUAL "baseline")
  message(FATAL_ERROR "Unsupported VOLVOXAI_CPU_TARGET '${VOLVOXAI_CPU_TARGET}'")
endif()
# Benchmarks are always built with AVX2 on x86 (NATIVE_BENCHMARK_CPU_FLAGS).
if(VOLVOX_X86)
  set(BENCH_CPU_FLAGS -mavx2 -mfma)
else()
  set(BENCH_CPU_FLAGS "")
endif()

# --- Include dirs (NATIVE_INCLUDE_FLAGS) --------------------------------------
set(INC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime
  ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels
  ${CMAKE_CURRENT_SOURCE_DIR}/src/backends
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/xz-embedded)

# Common interface: includes, provenance defines, -pthread, libm/libdl.
# Backend-enable and TRAINING macros are deliberately set PER TARGET so a test
# can request a different profile than the release default without redefinition.
add_library(volvox_common INTERFACE)
target_include_directories(volvox_common INTERFACE ${INC})
target_compile_definitions(volvox_common INTERFACE
  VOLVOXAI_VERSION="${VOLVOXAI_VERSION}"
  VOLVOXAI_GIT_COMMIT="${VOLVOXAI_GIT_COMMIT}"
  VOLVOXAI_BUILD_DATE="${VOLVOXAI_BUILD_DATE}")
target_compile_options(volvox_common INTERFACE ${CPU_FLAGS} -pthread)
target_link_options(volvox_common INTERFACE -pthread)
target_link_libraries(volvox_common INTERFACE m ${CMAKE_DL_LIBS})
# metal_engine.m needs the Apple frameworks; NNAPI needs libneuralnetworks.
if(APPLE AND VOLVOXAI_ENABLE_METAL)
  target_link_libraries(volvox_common INTERFACE "-framework Foundation" "-framework Metal")
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  target_compile_definitions(volvox_common INTERFACE USE_NNAPI)
  target_link_libraries(volvox_common INTERFACE neuralnetworks)
endif()

# Backend-define bundles used by target_compile_definitions below.
set(BK_DEFAULT
  VOLVOXAI_ENABLE_VULKAN=$<IF:$<BOOL:${VOLVOXAI_ENABLE_VULKAN}>,1,0>
  VOLVOXAI_ENABLE_OPENGL=$<IF:$<BOOL:${VOLVOXAI_ENABLE_OPENGL}>,1,0>
  VOLVOXAI_ENABLE_METAL=$<IF:$<BOOL:${VOLVOXAI_ENABLE_METAL}>,1,0>
  VOLVOXAI_ENABLE_NNAPI=$<IF:$<BOOL:${VOLVOXAI_ENABLE_NNAPI}>,1,0>)
set(BK_CPU_ONLY VOLVOXAI_ENABLE_VULKAN=0 VOLVOXAI_ENABLE_OPENGL=0
                VOLVOXAI_ENABLE_METAL=0 VOLVOXAI_ENABLE_NNAPI=0)
set(BK_MANAGER  VOLVOXAI_ENABLE_VULKAN=1 VOLVOXAI_ENABLE_OPENGL=1
                VOLVOXAI_ENABLE_METAL=0 VOLVOXAI_ENABLE_NNAPI=0)
set(BK_NNAPI    VOLVOXAI_ENABLE_VULKAN=0 VOLVOXAI_ENABLE_OPENGL=0
                VOLVOXAI_ENABLE_METAL=0 VOLVOXAI_ENABLE_NNAPI=1)

# --- ARM dotprod objects (Armv8.2 SDOT), separately-attributed like the .o's --
set(ARM_OBJS "")
if(VOLVOX_ARM_DOTPROD)
  add_library(volvox_arm_dotprod OBJECT
    src/kernels/qlinear_w8a8_arm_dotprod.c
    src/kernels/qconv_w8a8_arm_dotprod.c)
  target_include_directories(volvox_arm_dotprod PRIVATE ${INC})
  target_compile_options(volvox_arm_dotprod PRIVATE -O3 -march=armv8.2-a+dotprod -fno-lto)
  set(ARM_OBJS $<TARGET_OBJECTS:volvox_arm_dotprod>)
  add_compile_definitions(VOLVOXAI_ARM_DOTPROD_OBJECT=1)
endif()

# --- Source groups (mirror NATIVE_*_SRCS) -------------------------------------
set(RUNTIME_SRCS
  third_party/cJSON.c
  src/runtime/safetensors.c src/runtime/adapter_runtime.c src/runtime/attention_mask.c
  src/runtime/sequence_runtime.c src/runtime/backend.c src/runtime/arena.c
  src/runtime/backend_sdk.c src/runtime/incremental_runtime.c src/runtime/decode_session.c
  src/backends/w8a8_device_ops.c src/runtime/engine_runtime.c src/runtime/engine.c)
set(KERNEL_SRCS
  src/kernels/kernels.c src/kernels/cpu_features.c src/kernels/quant_cpu_opt.c
  src/kernels/qlinear_w8a8_x86.c src/kernels/qlinear_w8a8_arm.c
  src/kernels/qconv_w8a8_x86.c src/kernels/qconv_w8a8_arm.c
  src/kernels/qsdpa_w8a8_native.c src/kernels/conv_f32_opt.c src/kernels/tensor_f32_opt.c)
set(BACKEND_CORE_SRCS src/backends/backend_manager.c)
set(SHADER_STORE_SRCS
  src/shader_store.c
  third_party/xz-embedded/xz_crc32.c
  third_party/xz-embedded/xz_dec_lzma2.c
  third_party/xz-embedded/xz_dec_stream.c)
set(FULL_PROFILE_SRCS
  src/training/quantization.c src/training/quantization_runtime.c src/training/quantization_package.c)
set(CLI_SRCS cli/main.c)
set(TOKENIZER_SRC src/tokenization/tokenizer.c)

# Device backends selected by the enable toggles (NATIVE_BACKEND_SRCS).
set(BACKEND_DEVICE_SRCS "")
if(VOLVOXAI_ENABLE_VULKAN)
  list(APPEND BACKEND_DEVICE_SRCS src/backends/vulkan_engine.c)
endif()
if(VOLVOXAI_ENABLE_OPENGL)
  list(APPEND BACKEND_DEVICE_SRCS src/backends/opengl_engine.c)
endif()
if(VOLVOXAI_ENABLE_NNAPI)
  list(APPEND BACKEND_DEVICE_SRCS src/backends/nnapi_engine.c)
endif()
if(VOLVOXAI_ENABLE_METAL)
  list(APPEND BACKEND_DEVICE_SRCS src/backends/metal_engine.m)
endif()

# Full multi-backend engine object set (NATIVE_ENGINE_SRCS) and the baseline
# CPU-only set (NATIVE_CPU_ONLY_ENGINE_SRCS).
set(ENGINE_SRCS   ${RUNTIME_SRCS} ${KERNEL_SRCS} ${BACKEND_CORE_SRCS} ${BACKEND_DEVICE_SRCS} ${ARM_OBJS})
set(CPU_ONLY_SRCS ${RUNTIME_SRCS} ${KERNEL_SRCS} ${BACKEND_CORE_SRCS} ${ARM_OBJS})

# --- Shader pipeline: WGSL -> naga -> packed embedded_shaders.c ---------------
# Enabled-backend pack args (NATIVE_SHADER_BACKEND_ARGS).
set(PACK_BACKENDS "")
if(VOLVOXAI_ENABLE_VULKAN)
  list(APPEND PACK_BACKENDS --backend spv)
endif()
if(VOLVOXAI_ENABLE_OPENGL)
  list(APPEND PACK_BACKENDS --backend glsl --backend gles)
endif()
if(VOLVOXAI_ENABLE_METAL)
  list(APPEND PACK_BACKENDS --backend metal)
endif()
if(PACK_BACKENDS STREQUAL "")
  set(PACK_BACKENDS --backend none)
endif()

file(GLOB WGSL_SOURCES CONFIGURE_DEPENDS
  ${REPO}/shaders/inference/*.wgsl ${REPO}/shaders/training/*.wgsl)

set(SHADER_STAMP ${GEN}/.shaders_compiled.stamp)
add_custom_command(
  OUTPUT ${SHADER_STAMP}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GEN}
  # compile_shaders.sh drives `cargo`, which chokes on make's jobserver when
  # run as a build step; clear MAKEFLAGS so cargo builds standalone.
  COMMAND ${CMAKE_COMMAND} -E env MAKEFLAGS= MAKELEVEL= ${REPO}/tools/compile_shaders.sh
  COMMAND ${CMAKE_COMMAND} -E touch ${SHADER_STAMP}
  WORKING_DIRECTORY ${REPO}
  DEPENDS ${WGSL_SOURCES} ${REPO}/tools/compile_shaders.sh
  COMMENT "Compiling WGSL shaders with naga"
  VERBATIM)

# profile: <logical> <pack-profile> <backend-args...> -> ${GEN}/<logical>/embedded_shaders.c
function(volvox_shader_pack logical pack_profile out_var)
  set(dir ${GEN}/${logical})
  set(outc ${dir}/embedded_shaders.c)
  set(outh ${dir}/embedded_shaders.h)
  add_custom_command(
    OUTPUT ${outc} ${outh}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${dir}
    COMMAND ${PY} ${REPO}/tools/pack_native_shaders.py
            --compiled-dir ${REPO}/native/shaders --shader-source-dir ${REPO}/shaders
            --profile ${pack_profile} ${ARGN}
            --output-c ${outc} --output-h ${outh}
    DEPENDS ${SHADER_STAMP} ${REPO}/tools/pack_native_shaders.py
    COMMENT "Packing ${logical} embedded shaders"
    VERBATIM)
  set(${out_var} ${outc} PARENT_SCOPE)
endfunction()

volvox_shader_pack(inference inference EMB_INFERENCE ${PACK_BACKENDS})
volvox_shader_pack(full      full      EMB_FULL      ${PACK_BACKENDS})
volvox_shader_pack(cpu_only  inference EMB_CPU_ONLY  --backend none)

set(INFERENCE_SHADER_SRCS ${SHADER_STORE_SRCS} ${EMB_INFERENCE})
set(FULL_SHADER_SRCS      ${SHADER_STORE_SRCS} ${EMB_FULL})

# --- Release binaries ---------------------------------------------------------
# Placed in native/ (in-source, git-ignored) so the Python CLI tests and the
# profile-boundary checks find them at their historical native/volvoxai paths.
add_executable(volvoxai
  ${ENGINE_SRCS} ${INFERENCE_SHADER_SRCS} ${CLI_SRCS})
target_link_libraries(volvoxai PRIVATE volvox_common)
target_compile_definitions(volvoxai PRIVATE VOLVOXAI_ENABLE_TRAINING=0 ${BK_DEFAULT})
target_include_directories(volvoxai PRIVATE ${GEN}/inference)
target_compile_options(volvoxai PRIVATE -O3 -ffunction-sections -fdata-sections)
target_link_options(volvoxai PRIVATE -Wl,--gc-sections)
set_target_properties(volvoxai PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_NAME volvoxai)
# Android is inference-only and keeps a distinct name (was native/volvoxai_android
# in the old build_android recipe) so it never clobbers the Linux binary.
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  set_target_properties(volvoxai PROPERTIES OUTPUT_NAME volvoxai_android)
endif()

add_executable(volvoxai-full
  ${ENGINE_SRCS} ${FULL_PROFILE_SRCS} ${FULL_SHADER_SRCS} ${CLI_SRCS})
target_link_libraries(volvoxai-full PRIVATE volvox_common)
target_compile_definitions(volvoxai-full PRIVATE VOLVOXAI_ENABLE_TRAINING=1 ${BK_DEFAULT})
target_include_directories(volvoxai-full PRIVATE ${GEN}/full)
target_compile_options(volvoxai-full PRIVATE -O3)
set_target_properties(volvoxai-full PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_NAME volvoxai-full)

# CPU-only composition binary (no device backends; baseline ISA proof).
add_executable(volvoxai_cpu_only
  ${CPU_ONLY_SRCS} ${SHADER_STORE_SRCS} ${EMB_CPU_ONLY} ${CLI_SRCS})
target_link_libraries(volvoxai_cpu_only PRIVATE volvox_common)
target_compile_definitions(volvoxai_cpu_only PRIVATE VOLVOXAI_ENABLE_TRAINING=0 ${BK_CPU_ONLY})
target_include_directories(volvoxai_cpu_only PRIVATE ${GEN}/cpu_only)
target_compile_options(volvoxai_cpu_only PRIVATE -O2 -ffunction-sections -fdata-sections)
target_link_options(volvoxai_cpu_only PRIVATE -Wl,--gc-sections)

# --- Test-executable helper ---------------------------------------------------
# volvox_test(<name> SOURCES ... [DEFS ...] [OPTS ...] [GEN <profile>]
#             [LABELS ...] [RUN_ARGS ...] [NO_AUTOTEST])
function(volvox_test name)
  cmake_parse_arguments(T "NO_AUTOTEST" "GEN" "SOURCES;DEFS;OPTS;LABELS;RUN_ARGS" ${ARGN})
  add_executable(${name} ${T_SOURCES})
  target_link_libraries(${name} PRIVATE volvox_common)
  if(T_DEFS)
    target_compile_definitions(${name} PRIVATE ${T_DEFS})
  endif()
  if(T_OPTS)
    target_compile_options(${name} PRIVATE ${T_OPTS})
  endif()
  if(T_GEN)
    target_include_directories(${name} PRIVATE ${GEN}/${T_GEN})
  endif()
  if(NOT T_NO_AUTOTEST)
    add_test(NAME ${name} COMMAND ${name} ${T_RUN_ARGS})
    if(NOT T_LABELS)
      set(T_LABELS native)
    endif()
    set_tests_properties(${name} PROPERTIES LABELS "${T_LABELS}")
  endif()
endfunction()

# --- Regression tests (label: native) ----------------------------------------
volvox_test(test_backend_registry
  SOURCES src/runtime/backend.c tests/test_backend_registry.c
  OPTS -std=c11 -O2 -Wall -Wextra -Werror)

volvox_test(test_packed_quant_gemm
  SOURCES src/kernels/kernels.c src/kernels/cpu_features.c tests/test_packed_quant_gemm.c
  OPTS -std=c11 -O3)

volvox_test(test_qsdpa_w8a8_native
  SOURCES src/kernels/kernels.c src/kernels/cpu_features.c
          src/kernels/qsdpa_w8a8_native.c tests/test_qsdpa_w8a8_native.c
  OPTS -std=c11 -O3)

volvox_test(test_backend_sdk
  SOURCES ${CPU_ONLY_SRCS}
          ${REPO}/examples/backend_sdk/android_nnapi_backend.c
          ${REPO}/examples/backend_sdk/host_backend.c
          tests/test_backend_sdk.c
  DEFS VOLVOXAI_ENABLE_TRAINING=0 ${BK_CPU_ONLY}
  OPTS -std=c11 -O2)
target_include_directories(test_backend_sdk PRIVATE ${REPO}/examples/backend_sdk)

volvox_test(test_backend_manager
  SOURCES src/backends/backend_manager.c tests/test_backend_manager.c
  DEFS ${BK_MANAGER}
  OPTS -std=c11 -O2 -Wall -Wextra -Werror)

volvox_test(test_tokenizer
  SOURCES ${TOKENIZER_SRC} tests/test_tokenizer.c
  OPTS -std=c11 -O2 -Wall -Wextra -Werror)

volvox_test(test_ptq
  SOURCES src/training/quantization.c tests/test_ptq.c
  DEFS VOLVOXAI_ENABLE_TRAINING=1
  OPTS -std=c11 -O2 -Wall -Wextra -Werror)

volvox_test(test_ptq_package
  SOURCES ${ENGINE_SRCS} ${FULL_PROFILE_SRCS} ${FULL_SHADER_SRCS} tests/test_ptq_package.c
  DEFS VOLVOXAI_ENABLE_TRAINING=1 ${BK_DEFAULT}
  OPTS -O2 GEN full)

volvox_test(test_sequence_ops
  SOURCES src/kernels/sequence_ops.c tests/test_sequence_ops.c
  OPTS -std=c11 -O2 -Wall -Wextra -Werror)

volvox_test(test_sequence_runtime
  SOURCES ${CPU_ONLY_SRCS} tests/test_sequence_runtime.c
  DEFS VOLVOXAI_ENABLE_TRAINING=0 ${BK_CPU_ONLY}
  OPTS -std=c11 -O2)

volvox_test(test_incremental_runtime
  SOURCES ${ENGINE_SRCS} ${INFERENCE_SHADER_SRCS} tests/test_incremental_runtime.c
  DEFS VOLVOXAI_ENABLE_TRAINING=0 ${BK_DEFAULT}
  OPTS -O2 GEN inference)

volvox_test(test_incremental_runtime_nnapi
  SOURCES ${CPU_ONLY_SRCS} tests/test_incremental_runtime.c
  DEFS VOLVOXAI_ENABLE_TRAINING=0 ${BK_NNAPI} VOLVOXAI_INCREMENTAL_FAKE_NNAPI_TEST=1
  OPTS -O2)

volvox_test(test_gemm_f32
  SOURCES src/kernels/gemm_f32.c src/kernels/thread_pool.c tests/test_gemm_f32.c
  OPTS -std=c11 -O3 -Wall -Wextra -Werror)

volvox_test(test_gemm_f32_runtime
  SOURCES ${CPU_ONLY_SRCS} tests/test_gemm_f32_runtime.c
  DEFS VOLVOXAI_ENABLE_TRAINING=0 ${BK_CPU_ONLY}
  OPTS -std=c11 -O3)

# Full-profile engine smoke tests (were built by the `build_native` aggregate).
foreach(_t adapter_runtime training_backward quantized_runtime)
  volvox_test(test_${_t}
    SOURCES ${ENGINE_SRCS} ${FULL_SHADER_SRCS} tests/test_${_t}.c
    DEFS VOLVOXAI_ENABLE_TRAINING=1 ${BK_DEFAULT}
    OPTS -O2 GEN full)
endforeach()

# Shader store (needs the full embedded pack + on-disk native/shaders).
volvox_test(test_shader_store
  SOURCES ${FULL_SHADER_SRCS} tests/test_shader_store.c
  DEFS ${BK_DEFAULT}
  OPTS -std=c11 -O2 -Wall -Wextra -Werror GEN full
  LABELS native
  RUN_ARGS /tmp/volvoxai-no-shaders ${REPO}/native/shaders)

# --- Composition + profile-boundary checks (scripts) --------------------------
add_test(NAME native_backend_composition
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_cpu_only_composition.sh
          $<TARGET_FILE:volvoxai_cpu_only>)
set_tests_properties(native_backend_composition PROPERTIES LABELS native)

add_test(NAME native_profile_boundaries
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_profile_boundaries.sh
          $<TARGET_FILE:volvoxai> $<TARGET_FILE:volvoxai-full>)
set_tests_properties(native_profile_boundaries PROPERTIES LABELS native)

add_test(NAME native_cli
  COMMAND ${PY} ${REPO}/tools/tests/test_native_cli.py $<TARGET_FILE:volvoxai>)
set_tests_properties(native_cli PROPERTIES LABELS native)

add_test(NAME native_training_boundary
  COMMAND ${PY} ${REPO}/tools/tests/test_native_training_boundary.py -v)
set_tests_properties(native_training_boundary PROPERTIES
  LABELS native WORKING_DIRECTORY ${REPO})

# --- Opt-in example policies (label: example) ---------------------------------
add_executable(volvoxai-tasks
  ${ENGINE_SRCS} ${INFERENCE_SHADER_SRCS}
  ${REPO}/examples/native_support/image_io.c
  ${REPO}/examples/native_task_cli/main.c ${TOKENIZER_SRC})
target_link_libraries(volvoxai-tasks PRIVATE volvox_common)
target_compile_definitions(volvoxai-tasks PRIVATE VOLVOXAI_ENABLE_TRAINING=0 ${BK_DEFAULT})
target_include_directories(volvoxai-tasks PRIVATE
  ${GEN}/inference ${REPO}/examples/native_task_cli ${REPO}/examples/native_support)
target_compile_options(volvoxai-tasks PRIVATE -O2)
add_test(NAME test_native_task_cli
  COMMAND ${PY} ${REPO}/examples/native_task_cli/tests/test_cli.py $<TARGET_FILE:volvoxai-tasks>)
set_tests_properties(test_native_task_cli PROPERTIES LABELS example)

foreach(_variant tiny_receipt_w8a8:main test_tiny_receipt_w8a8:test_tiny_receipt_w8a8)
  string(REPLACE ":" ";" _pair ${_variant})
  list(GET _pair 0 _bin)
  list(GET _pair 1 _entry)
  add_executable(${_bin}
    ${ENGINE_SRCS} ${INFERENCE_SHADER_SRCS}
    ${REPO}/examples/tiny_receipt_vqa/native/tiny_receipt_image.c
    ${REPO}/examples/tiny_receipt_vqa/native/tiny_receipt_w8a8.c
    ${REPO}/examples/tiny_receipt_vqa/native/${_entry}.c)
  target_link_libraries(${_bin} PRIVATE volvox_common)
  target_compile_definitions(${_bin} PRIVATE VOLVOXAI_ENABLE_TRAINING=0 ${BK_DEFAULT})
  target_include_directories(${_bin} PRIVATE
    ${GEN}/inference ${REPO}/examples/tiny_receipt_vqa/native)
  target_compile_options(${_bin} PRIVATE -O3)
endforeach()
add_test(NAME test_tiny_receipt_native_example
  COMMAND ${CMAKE_COMMAND}
    -DBIN=$<TARGET_FILE:tiny_receipt_w8a8>
    -DTEST_BIN=$<TARGET_FILE:test_tiny_receipt_w8a8>
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_tiny_receipt.cmake)
set_tests_properties(test_tiny_receipt_native_example PROPERTIES LABELS example)

# --- Benchmarks (label: benchmark; on demand, not a gate) ---------------------
volvox_test(benchmark_w8a8_native
  SOURCES src/kernels/kernels.c src/kernels/quant_cpu_opt.c src/kernels/cpu_features.c
          src/kernels/qlinear_w8a8_x86.c src/kernels/qlinear_w8a8_arm.c
          src/kernels/qconv_w8a8_x86.c src/kernels/qconv_w8a8_arm.c ${ARM_OBJS}
          tests/benchmark_w8a8_native.c
  OPTS -O3 ${BENCH_CPU_FLAGS}
  LABELS benchmark)

volvox_test(benchmark_qsdpa_w8a8_native
  SOURCES src/kernels/kernels.c src/kernels/cpu_features.c
          src/kernels/qsdpa_w8a8_native.c tests/benchmark_qsdpa_w8a8_native.c
  OPTS -std=c11 -O3
  LABELS benchmark)

volvox_test(benchmark_gemm_f32
  SOURCES src/kernels/gemm_f32.c src/kernels/thread_pool.c tests/benchmark_gemm_f32.c
  OPTS -std=c11 -O3 ${BENCH_CPU_FLAGS}
  LABELS benchmark)

# --- GPU training tests (label: gpu; build here, run needs a real device) -----
if(VOLVOXAI_ENABLE_VULKAN)
  volvox_test(test_vulkan_training
    SOURCES src/backends/vulkan_engine.c ${FULL_SHADER_SRCS} tests/test_vulkan_training.c
    DEFS VOLVOXAI_ENABLE_TRAINING=1
    OPTS -O2 GEN full
    LABELS gpu)
endif()
if(VOLVOXAI_ENABLE_OPENGL)
  volvox_test(test_opengl_training
    SOURCES src/backends/opengl_engine.c ${FULL_SHADER_SRCS} tests/test_opengl_training.c
    DEFS VOLVOXAI_ENABLE_TRAINING=1 VOLVOX_OPENGL_TESTING
    OPTS -O2 GEN full
    LABELS gpu)
endif()
if(APPLE AND VOLVOXAI_ENABLE_METAL)
  volvox_test(test_metal_training
    SOURCES src/backends/metal_engine.m ${FULL_SHADER_SRCS} tests/test_metal_training.m
    DEFS VOLVOXAI_ENABLE_TRAINING=1 VOLVOX_METAL_TESTING
    OPTS -O2 GEN full
    LABELS gpu)
endif()

# --- ISA codegen verification (label: verify; x86 only) -----------------------
if(VOLVOX_X86)
  set(_verify ${CMAKE_CURRENT_SOURCE_DIR}/cmake/verify_codegen.sh)
  set(_kern ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels)
  add_test(NAME verify_native_avxvnni
    COMMAND ${_verify} emit-ymm ${_kern}/qlinear_w8a8_x86.c ${_kern})
  add_test(NAME verify_native_avx512vnni
    COMMAND ${_verify} emit-zmm ${_kern}/qlinear_w8a8_x86.c ${_kern})
  add_test(NAME verify_native_qconv_avxvnni
    COMMAND ${_verify} emit-ymm ${_kern}/qconv_w8a8_x86.c ${_kern})
  add_test(NAME verify_native_qconv_avx512vnni
    COMMAND ${_verify} emit-zmm ${_kern}/qconv_w8a8_x86.c ${_kern})
  add_test(NAME verify_native_w8a8_baseline_isa
    COMMAND ${_verify} baseline ${_kern})
  set_tests_properties(
    verify_native_avxvnni verify_native_avx512vnni
    verify_native_qconv_avxvnni verify_native_qconv_avx512vnni
    verify_native_w8a8_baseline_isa
    PROPERTIES LABELS verify)
endif()
