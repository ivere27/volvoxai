# Native engine: profiles, tests, benchmarks and ISA codegen checks.
# Native build composition is defined here for every supported profile.

set(REPO ${CMAKE_SOURCE_DIR})            # repo root (shaders/, tools/, examples/)
set(GEN  ${CMAKE_BINARY_DIR}/gen)        # generated assets (was build/native/*)
set(PY   python3)

# proto/volvoxai.proto owns the public enum/status/stage numbers consumed by
# the native implementation. Fail before compilation if the tracked headers
# were not regenerated with the schema.
execute_process(
  COMMAND ${PY} ${REPO}/tools/generate_proto_enums.py --check
  WORKING_DIRECTORY ${REPO}
  RESULT_VARIABLE VOLVOXAI_PROTO_ENUM_CHECK_RESULT
  OUTPUT_VARIABLE VOLVOXAI_PROTO_ENUM_CHECK_OUTPUT
  ERROR_VARIABLE VOLVOXAI_PROTO_ENUM_CHECK_ERROR)
if(NOT VOLVOXAI_PROTO_ENUM_CHECK_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Generated protobuf enum contract is stale. Run make proto_enum_codegen.\n"
    "${VOLVOXAI_PROTO_ENUM_CHECK_OUTPUT}${VOLVOXAI_PROTO_ENUM_CHECK_ERROR}")
endif()

# proto/kernel_registry.proto owns the internal backend/kernel inventory used
# by native code and the other runtime/tooling projections. Keep this check
# dependency-free so the pinned native build image does not need protoc or the
# Python protobuf package merely to validate tracked generated files.
execute_process(
  COMMAND ${PY} ${REPO}/tools/generate_kernel_registry.py --check
  WORKING_DIRECTORY ${REPO}
  RESULT_VARIABLE VOLVOXAI_KERNEL_REGISTRY_CHECK_RESULT
  OUTPUT_VARIABLE VOLVOXAI_KERNEL_REGISTRY_CHECK_OUTPUT
  ERROR_VARIABLE VOLVOXAI_KERNEL_REGISTRY_CHECK_ERROR)
if(NOT VOLVOXAI_KERNEL_REGISTRY_CHECK_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Generated kernel registry is stale. Run make kernel_registry_codegen.\n"
    "${VOLVOXAI_KERNEL_REGISTRY_CHECK_OUTPUT}${VOLVOXAI_KERNEL_REGISTRY_CHECK_ERROR}")
endif()

# --- Backend toggles (mirror NATIVE_ENABLE_* defaults) ------------------------
option(VOLVOXAI_ENABLE_VULKAN "Link the Vulkan device backend" ON)
option(VOLVOXAI_ENABLE_OPENGL "Link the OpenGL device backend" ON)
option(VOLVOXAI_ENABLE_CUDA "Link the runtime-loaded CUDA device backend" OFF)
if(APPLE)
  option(VOLVOXAI_ENABLE_METAL "Link the Metal device backend" ON)
else()
  option(VOLVOXAI_ENABLE_METAL "Link the Metal device backend" OFF)
endif()
option(VOLVOXAI_ENABLE_NNAPI "Link the NNAPI device backend" OFF)
set(VOLVOXAI_CPU_TARGET "baseline" CACHE STRING "baseline or avx2")
set(VOLVOXAI_CUDA_ARCH "75" CACHE STRING
    "CUDA compute capability used for embedded PTX (for example 75 or 86)")
option(VOLVOXAI_CUDA_FAST_FP32
  "Allow CUDA FP32 fused multiply-add contraction (changes rounding)" OFF)
if(VOLVOXAI_CUDA_FAST_FP32)
  set(_cuda_fp32_contract fast-fma)
else()
  set(_cuda_fp32_contract strict-no-fma)
endif()

# Android release builds enable NNAPI by default.
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
# Armv8.2 SDOT objects are used on aarch64 Linux and Android. macOS arm64 uses
# its platform-specific kernels.
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
  ${GEN}/cuda/${_cuda_fp32_contract}
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
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(volvox_common INTERFACE -fvisibility=hidden)
endif()
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
  VOLVOXAI_ENABLE_CUDA=$<IF:$<BOOL:${VOLVOXAI_ENABLE_CUDA}>,1,0>
  VOLVOXAI_ENABLE_METAL=$<IF:$<BOOL:${VOLVOXAI_ENABLE_METAL}>,1,0>
  VOLVOXAI_ENABLE_NNAPI=$<IF:$<BOOL:${VOLVOXAI_ENABLE_NNAPI}>,1,0>)
set(BK_CPU_ONLY VOLVOXAI_ENABLE_VULKAN=0 VOLVOXAI_ENABLE_OPENGL=0
                VOLVOXAI_ENABLE_CUDA=0 VOLVOXAI_ENABLE_METAL=0
                VOLVOXAI_ENABLE_NNAPI=0)
set(BK_MANAGER  VOLVOXAI_ENABLE_VULKAN=1 VOLVOXAI_ENABLE_OPENGL=1
                VOLVOXAI_ENABLE_CUDA=1 VOLVOXAI_ENABLE_METAL=0
                VOLVOXAI_ENABLE_NNAPI=0)
set(BK_NNAPI    VOLVOXAI_ENABLE_VULKAN=0 VOLVOXAI_ENABLE_OPENGL=0
                VOLVOXAI_ENABLE_CUDA=0 VOLVOXAI_ENABLE_METAL=0
                VOLVOXAI_ENABLE_NNAPI=1)
set(BK_CUDA     VOLVOXAI_ENABLE_VULKAN=0 VOLVOXAI_ENABLE_OPENGL=0
                VOLVOXAI_ENABLE_CUDA=1 VOLVOXAI_ENABLE_METAL=0
                VOLVOXAI_ENABLE_NNAPI=0)

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
  src/runtime/backend_sdk.c src/runtime/runtime_state.c src/runtime/public_api.c
  src/runtime/incremental_runtime.c src/runtime/decode_session.c
  src/backends/w8a8_device_ops.c src/runtime/engine_runtime.c src/runtime/engine.c)
set(KERNEL_SRCS
  src/kernels/kernels.c src/kernels/cpu_features.c src/kernels/quant_cpu_opt.c
  src/kernels/qlinear_w8a8_x86.c src/kernels/qlinear_w8a8_arm.c
  src/kernels/qconv_w8a8_x86.c src/kernels/qconv_w8a8_arm.c
  src/kernels/qbatch_matmul_w8a8_native.c
  src/kernels/qsdpa_w8a8_native.c src/kernels/qgroupnorm_w8a8_native.c
  src/kernels/qnorm_activation_w8a8_native.c
  src/kernels/transpose_w8a8_native.c
  src/kernels/conv_f32_opt_baseline.c src/kernels/conv_f32_opt_avx2.c
  src/kernels/conv_f32_dispatch.c src/kernels/tensor_f32_opt.c)

# conv_f32_opt.c holds hand-written AVX2 conv bodies behind `#if
# defined(__AVX2__)`.  Compiling it only at VOLVOXAI_CPU_TARGET (default
# `baseline`) preprocessed them away, so AVX2 machines ran the scalar fallback —
# 23.5x slower on the TinyReceiptVQA encoder's Conv2D nodes.  Build the same
# source a second time with the higher ISA and let conv_f32_dispatch.c pick at
# runtime.  Only this one file gets the flags, and it is reached only after a
# vx_kernel_platform() check, so the binary's own CPU requirement is unchanged.
#
# These are plain sources, not a shared OBJECT library: conv_f32_opt.c reads
# VxEngineState, whose layout depends on each target's VOLVOXAI_ENABLE_TRAINING.
if(VOLVOX_X86)
  set_source_files_properties(src/kernels/conv_f32_opt_avx2.c
    PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
endif()
set(BACKEND_CORE_SRCS src/backends/backend_manager.c)
set(SHADER_STORE_SRCS
  src/shader_store.c
  third_party/xz-embedded/xz_crc32.c
  third_party/xz-embedded/xz_dec_lzma2.c
  third_party/xz-embedded/xz_dec_stream.c)
set(FULL_PROFILE_SRCS
  src/training/trainer_api.c src/training/ptq_api.c src/training/quantization.c
  src/training/quantization_runtime.c src/training/quantization_package.c)
set(CLI_SRCS cli/main.c)
set(TOKENIZER_SRC src/tokenization/tokenizer.c)

# --- CUDA kernel pipeline: authoritative .cu -> PTX -> embedded C array ------
# CUDA stays opt-in because the pinned default build image has no CUDA toolkit.
# The backend itself uses dlopen/GetProcAddress for the Driver API; compiling
# PTX as a custom command keeps CMake from adding cudart or libcuda linkage.
set(CUDA_BACKEND_SRCS "")
set(CUDA_TRAINING_PTX_SRCS "")
if(VOLVOXAI_ENABLE_CUDA)
  if(NOT VOLVOXAI_CUDA_ARCH MATCHES "^[0-9]+$")
    message(FATAL_ERROR
      "VOLVOXAI_CUDA_ARCH must be a numeric compute capability such as 75 or 86")
  endif()

  if(VOLVOXAI_CUDA_FAST_FP32)
    set(_cuda_nvcc_fp32_flag --fmad=true)
    set(_cuda_clang_fp32_flag -ffp-contract=fast)
    message(STATUS
      "CUDA FP32 contract: fast-fma (multiply-add contraction enabled)")
  else()
    set(_cuda_nvcc_fp32_flag --fmad=false)
    set(_cuda_clang_fp32_flag -ffp-contract=off)
    message(STATUS
      "CUDA FP32 contract: strict-no-fma (multiply-add contraction disabled)")
  endif()

  set(CUDA_KERNEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/backends)
  set(CUDA_DEVICE_KERNEL_DIR ${CUDA_KERNEL_DIR}/cuda/kernels)
  set(CUDA_KERNEL_SOURCE ${CUDA_KERNEL_DIR}/cuda_kernels.cu)
  file(GLOB CUDA_FORWARD_KERNEL_INCLUDES CONFIGURE_DEPENDS
    ${CUDA_DEVICE_KERNEL_DIR}/cuda_forward_*.inc)
  list(FILTER CUDA_FORWARD_KERNEL_INCLUDES EXCLUDE REGEX "_host\\.inc$")
  # Keep mode-specific outputs separate. This makes a contract change rebuild
  # PTX even with generators that do not track custom-command text changes.
  set(CUDA_GEN_DIR ${GEN}/cuda/${_cuda_fp32_contract})
  set(CUDA_PTX ${CUDA_GEN_DIR}/cuda_kernels.ptx)
  set(CUDA_EMBED_C ${CUDA_GEN_DIR}/embedded_cuda_ptx.c)
  set(CUDA_EMBED_H ${CUDA_GEN_DIR}/embedded_cuda_ptx.h)
  set(CUDA_TRAINING_KERNEL_SOURCE
    ${CUDA_KERNEL_DIR}/cuda_training_kernels.cu)
  file(GLOB CUDA_TRAINING_KERNEL_INCLUDES CONFIGURE_DEPENDS
    ${CUDA_DEVICE_KERNEL_DIR}/cuda_training_*.inc
    ${CUDA_DEVICE_KERNEL_DIR}/cuda_quantize_w8_*.inc)
  # Host fragments are compiled through cuda_engine.c and must never become PTX
  # dependencies. The suffix filters keep that profile boundary explicit while
  # CONFIGURE_DEPENDS discovers every current and future device fragment.
  list(FILTER CUDA_TRAINING_KERNEL_INCLUDES EXCLUDE REGEX "_host\\.inc$")
  set(CUDA_TRAINING_PTX ${CUDA_GEN_DIR}/cuda_training_kernels.ptx)
  set(CUDA_TRAINING_EMBED_C
    ${CUDA_GEN_DIR}/embedded_cuda_training_ptx.c)
  set(CUDA_TRAINING_EMBED_H
    ${CUDA_GEN_DIR}/embedded_cuda_training_ptx.h)

  find_program(VOLVOXAI_NVCC_EXECUTABLE NAMES nvcc)
  if(VOLVOXAI_NVCC_EXECUTABLE)
    set(_cuda_ptx_command
      ${VOLVOXAI_NVCC_EXECUTABLE}
      --ptx --std=c++14 -O3 ${_cuda_nvcc_fp32_flag}
      --gpu-architecture=compute_${VOLVOXAI_CUDA_ARCH}
      native/src/backends/cuda_kernels.cu -o ${CUDA_PTX})
    set(_cuda_training_ptx_command
      ${VOLVOXAI_NVCC_EXECUTABLE}
      --ptx --std=c++14 -O3 ${_cuda_nvcc_fp32_flag}
      --gpu-architecture=compute_${VOLVOXAI_CUDA_ARCH}
      native/src/backends/cuda_training_kernels.cu
      -o ${CUDA_TRAINING_PTX})
    message(STATUS "CUDA PTX compiler: ${VOLVOXAI_NVCC_EXECUTABLE} (nvcc)")
  else()
    set(_cuda_clang_candidates "")
    if(DEFINED VOLVOXAI_CUDA_CLANGXX_EXECUTABLE AND
       VOLVOXAI_CUDA_CLANGXX_EXECUTABLE AND
       NOT VOLVOXAI_CUDA_CLANGXX_EXECUTABLE MATCHES "-NOTFOUND$")
      list(APPEND _cuda_clang_candidates
        "${VOLVOXAI_CUDA_CLANGXX_EXECUTABLE}")
    endif()
    foreach(_cuda_clang_name clang++ clang++-17 clang++-16)
      unset(_cuda_clang_candidate CACHE)
      find_program(_cuda_clang_candidate NAMES ${_cuda_clang_name})
      if(_cuda_clang_candidate)
        list(APPEND _cuda_clang_candidates "${_cuda_clang_candidate}")
      endif()
    endforeach()
    list(REMOVE_DUPLICATES _cuda_clang_candidates)
    set(_cuda_clang_selected "")
    foreach(_cuda_clang_candidate IN LISTS _cuda_clang_candidates)
      execute_process(
        COMMAND ${_cuda_clang_candidate} --print-targets
        RESULT_VARIABLE _cuda_clang_targets_status
        OUTPUT_VARIABLE _cuda_clang_targets
        ERROR_VARIABLE _cuda_clang_targets_error)
      string(FIND "${_cuda_clang_targets}${_cuda_clang_targets_error}"
        "nvptx64" _cuda_nvptx64_index)
      if(_cuda_clang_targets_status EQUAL 0 AND
         NOT _cuda_nvptx64_index EQUAL -1)
        set(_cuda_clang_selected "${_cuda_clang_candidate}")
        break()
      endif()
    endforeach()
    if(NOT _cuda_clang_selected)
      message(FATAL_ERROR
        "VOLVOXAI_ENABLE_CUDA=ON requires nvcc or a clang++ build with "
        "NVPTX64 support")
    endif()
    set(VOLVOXAI_CUDA_CLANGXX_EXECUTABLE "${_cuda_clang_selected}"
      CACHE FILEPATH "Clang++ executable used to compile CUDA PTX" FORCE)
    set(_cuda_ptx_command
      ${VOLVOXAI_CUDA_CLANGXX_EXECUTABLE}
      -x cuda --cuda-device-only
      --cuda-gpu-arch=sm_${VOLVOXAI_CUDA_ARCH}
      -nocudainc -nocudalib -std=c++14 -O3 ${_cuda_clang_fp32_flag} -S
      native/src/backends/cuda_kernels.cu -o ${CUDA_PTX})
    set(_cuda_training_ptx_command
      ${VOLVOXAI_CUDA_CLANGXX_EXECUTABLE}
      -x cuda --cuda-device-only
      --cuda-gpu-arch=sm_${VOLVOXAI_CUDA_ARCH}
      -nocudainc -nocudalib -std=c++14 -O3 ${_cuda_clang_fp32_flag} -S
      native/src/backends/cuda_training_kernels.cu
      -o ${CUDA_TRAINING_PTX})
    message(STATUS
      "CUDA PTX compiler: ${VOLVOXAI_CUDA_CLANGXX_EXECUTABLE} (clang NVPTX)")
  endif()

  add_custom_command(
    OUTPUT ${CUDA_PTX}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CUDA_GEN_DIR}
    COMMAND ${_cuda_ptx_command}
    WORKING_DIRECTORY ${REPO}
    DEPENDS ${CUDA_KERNEL_SOURCE} ${CUDA_FORWARD_KERNEL_INCLUDES}
    COMMENT
      "Compiling CUDA kernels to PTX for compute_${VOLVOXAI_CUDA_ARCH} (${_cuda_fp32_contract})"
    VERBATIM)
  add_custom_command(
    OUTPUT ${CUDA_EMBED_C} ${CUDA_EMBED_H}
    COMMAND ${PY} ${REPO}/tools/embed_cuda_ptx.py
            --input ${CUDA_PTX}
            --output-c ${CUDA_EMBED_C}
            --output-h ${CUDA_EMBED_H}
    DEPENDS ${CUDA_PTX} ${REPO}/tools/embed_cuda_ptx.py
    COMMENT "Embedding CUDA PTX module"
    VERBATIM)
  add_custom_command(
    OUTPUT ${CUDA_TRAINING_PTX}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CUDA_GEN_DIR}
    COMMAND ${_cuda_training_ptx_command}
    WORKING_DIRECTORY ${REPO}
    DEPENDS ${CUDA_TRAINING_KERNEL_SOURCE} ${CUDA_TRAINING_KERNEL_INCLUDES}
    COMMENT
      "Compiling CUDA training substrate to PTX for compute_${VOLVOXAI_CUDA_ARCH} (${_cuda_fp32_contract})"
    VERBATIM)
  add_custom_command(
    OUTPUT ${CUDA_TRAINING_EMBED_C} ${CUDA_TRAINING_EMBED_H}
    COMMAND ${PY} ${REPO}/tools/embed_cuda_ptx.py
            --input ${CUDA_TRAINING_PTX}
            --output-c ${CUDA_TRAINING_EMBED_C}
            --output-h ${CUDA_TRAINING_EMBED_H}
            --symbol volvoxai_cuda_training_ptx
    DEPENDS ${CUDA_TRAINING_PTX} ${REPO}/tools/embed_cuda_ptx.py
    COMMENT "Embedding full-profile CUDA training PTX module"
    VERBATIM)
  # Generated PTX is shared by every CUDA-enabled executable.  Route all
  # consumers through one target so parallel Make builds never run duplicate
  # custom commands against the same PTX/C/H outputs.
  add_custom_target(volvoxai_cuda_codegen
    DEPENDS ${CUDA_EMBED_C} ${CUDA_EMBED_H})
  # Training code generation has distinct outputs and a distinct target. Only
  # full-profile consumers depend on it, so building an inference target never
  # compiles or embeds training PTX. The two codegen targets can run safely in
  # parallel because they share no output file.
  add_custom_target(volvoxai_cuda_training_codegen
    DEPENDS ${CUDA_TRAINING_EMBED_C} ${CUDA_TRAINING_EMBED_H})
  set_source_files_properties(
    ${CUDA_EMBED_C} ${CUDA_EMBED_H}
    ${CUDA_TRAINING_EMBED_C} ${CUDA_TRAINING_EMBED_H}
    PROPERTIES GENERATED TRUE)
  list(APPEND CUDA_BACKEND_SRCS
    src/backends/cuda_engine.c ${CUDA_EMBED_C} ${CUDA_EMBED_H})
  list(APPEND CUDA_TRAINING_PTX_SRCS
    ${CUDA_TRAINING_EMBED_C} ${CUDA_TRAINING_EMBED_H})
endif()

function(volvox_cuda_codegen_before target)
  if(TARGET volvoxai_cuda_codegen)
    add_dependencies(${target} volvoxai_cuda_codegen)
  endif()
endfunction()

function(volvox_cuda_training_codegen_before target)
  if(TARGET volvoxai_cuda_training_codegen)
    add_dependencies(${target} volvoxai_cuda_training_codegen)
  endif()
endfunction()

# Device backends selected by the enable toggles (NATIVE_BACKEND_SRCS).
set(BACKEND_DEVICE_SRCS "")
if(VOLVOXAI_ENABLE_VULKAN)
  list(APPEND BACKEND_DEVICE_SRCS src/backends/vulkan_engine.c)
endif()
if(VOLVOXAI_ENABLE_OPENGL)
  list(APPEND BACKEND_DEVICE_SRCS src/backends/opengl_engine.c)
endif()
if(VOLVOXAI_ENABLE_CUDA)
  list(APPEND BACKEND_DEVICE_SRCS ${CUDA_BACKEND_SRCS})
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
set(FULL_ENGINE_SRCS ${ENGINE_SRCS} ${CUDA_TRAINING_PTX_SRCS})
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
add_custom_target(volvoxai_compile_shaders DEPENDS ${SHADER_STAMP})

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
  add_custom_target(volvoxai_pack_${logical}_shaders DEPENDS ${outc} ${outh})
  add_dependencies(volvoxai_pack_${logical}_shaders volvoxai_compile_shaders)
  set(${out_var} ${outc} PARENT_SCOPE)
endfunction()

volvox_shader_pack(inference inference EMB_INFERENCE ${PACK_BACKENDS})
volvox_shader_pack(full      full      EMB_FULL      ${PACK_BACKENDS})
volvox_shader_pack(cpu_only  inference EMB_CPU_ONLY  --backend none)

set(INFERENCE_SHADER_SRCS ${SHADER_STORE_SRCS} ${EMB_INFERENCE})
set(FULL_SHADER_SRCS      ${SHADER_STORE_SRCS} ${EMB_FULL})

# --- Release binaries ---------------------------------------------------------
# Fixed release names and CLI/profile checks require these binaries in native/.
add_executable(volvoxai
  ${ENGINE_SRCS} ${INFERENCE_SHADER_SRCS} ${CLI_SRCS})
add_dependencies(volvoxai volvoxai_pack_inference_shaders)
volvox_cuda_codegen_before(volvoxai)
target_link_libraries(volvoxai PRIVATE volvox_common)
target_compile_definitions(volvoxai PRIVATE VOLVOXAI_ENABLE_TRAINING=0 ${BK_DEFAULT})
target_include_directories(volvoxai PRIVATE ${GEN}/inference)
target_compile_options(volvoxai PRIVATE -O3 -ffunction-sections -fdata-sections)
target_link_options(volvoxai PRIVATE -Wl,--gc-sections)
set_target_properties(volvoxai PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_NAME volvoxai)
# Android is inference-only and uses a distinct output name so cross-builds do
# not clobber the host binary.
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  set_target_properties(volvoxai PROPERTIES OUTPUT_NAME volvoxai_android)
endif()

add_executable(volvoxai-full
  ${FULL_ENGINE_SRCS} ${FULL_PROFILE_SRCS} ${FULL_SHADER_SRCS} ${CLI_SRCS})
add_dependencies(volvoxai-full volvoxai_pack_full_shaders)
volvox_cuda_codegen_before(volvoxai-full)
volvox_cuda_training_codegen_before(volvoxai-full)
target_link_libraries(volvoxai-full PRIVATE volvox_common)
target_compile_definitions(volvoxai-full PRIVATE VOLVOXAI_ENABLE_TRAINING=1 ${BK_DEFAULT})
target_include_directories(volvoxai-full PRIVATE ${GEN}/full)
target_compile_options(volvoxai-full PRIVATE -O3)
set_target_properties(volvoxai-full PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_NAME volvoxai-full)

# CPU-only composition binary (no device backends; baseline ISA proof).
add_executable(volvoxai_cpu_only
  ${CPU_ONLY_SRCS} ${SHADER_STORE_SRCS} ${EMB_CPU_ONLY} ${CLI_SRCS})
add_dependencies(volvoxai_cpu_only volvoxai_pack_cpu_only_shaders)
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
  volvox_cuda_codegen_before(${name})
  target_link_libraries(${name} PRIVATE volvox_common)
  if(T_DEFS)
    target_compile_definitions(${name} PRIVATE ${T_DEFS})
  endif()
  if(T_OPTS)
    target_compile_options(${name} PRIVATE ${T_OPTS})
  endif()
  if(T_GEN)
    add_dependencies(${name} volvoxai_pack_${T_GEN}_shaders)
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
volvox_test(test_runtime_state
  SOURCES src/runtime/runtime_state.c src/kernels/thread_pool.c
          tests/test_runtime_state.c
  DEFS VOLVOXAI_ENABLE_TRAINING=0
  OPTS -std=c11 -O2 -Wall -Wextra -Werror)

volvox_test(test_nnapi_state_ownership
  SOURCES src/runtime/runtime_state.c src/kernels/thread_pool.c
          src/backends/nnapi_engine.c tests/test_nnapi_state_ownership.c
  DEFS VOLVOXAI_ENABLE_TRAINING=0 __ANDROID__=1
  OPTS -std=c11 -O2 -Wall -Wextra -Werror)
target_include_directories(test_nnapi_state_ownership PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/fake_android)

volvox_test(test_backend_registry
  SOURCES src/runtime/backend.c tests/test_backend_registry.c
  OPTS -std=c11 -O2 -Wall -Wextra -Werror)

volvox_test(test_packed_quant_gemm
  SOURCES src/kernels/kernels.c src/kernels/cpu_features.c
          src/kernels/quant_cpu_opt.c tests/test_packed_quant_gemm.c
  OPTS -std=c11 -O3)

volvox_test(test_quantize_linear
  SOURCES src/kernels/kernels.c src/kernels/cpu_features.c
          src/kernels/quant_cpu_opt.c
          tests/test_quantize_linear.c
  OPTS -std=c11 -O3)

volvox_test(test_qsdpa_w8a8_native
  SOURCES src/kernels/kernels.c src/kernels/cpu_features.c
          src/kernels/quant_cpu_opt.c
          src/kernels/qsdpa_w8a8_native.c tests/test_qsdpa_w8a8_native.c
  OPTS -std=c11 -O3)

volvox_test(test_public_api
  SOURCES ${CPU_ONLY_SRCS} ${SHADER_STORE_SRCS} ${EMB_CPU_ONLY}
          tests/test_public_api.c
  DEFS VOLVOXAI_ENABLE_TRAINING=0 VOLVOXAI_PUBLIC_API_TESTING=1 ${BK_CPU_ONLY}
  OPTS -std=c11 -O2 GEN cpu_only)

volvox_test(test_trainer_api
  SOURCES ${CPU_ONLY_SRCS} ${FULL_PROFILE_SRCS} ${FULL_SHADER_SRCS}
          tests/test_trainer_api.c
  DEFS VOLVOXAI_ENABLE_TRAINING=1 ${BK_CPU_ONLY}
  OPTS -std=c11 -O2 GEN full)

volvox_test(test_ptq_api
  SOURCES ${CPU_ONLY_SRCS} ${FULL_PROFILE_SRCS} ${FULL_SHADER_SRCS}
          tests/test_ptq_api.c
  DEFS VOLVOXAI_ENABLE_TRAINING=1 ${BK_CPU_ONLY}
  OPTS -std=c11 -O2 GEN full)

volvox_test(test_backend_provider_examples
  SOURCES ${CPU_ONLY_SRCS} ${SHADER_STORE_SRCS} ${EMB_CPU_ONLY}
          ../examples/backend_sdk/host_backend.c
          ../examples/backend_sdk/android_nnapi_backend.c
          tests/test_backend_provider_examples.c
  DEFS VOLVOXAI_ENABLE_TRAINING=0 ${BK_CPU_ONLY}
  OPTS -std=c11 -O2 GEN cpu_only)
target_include_directories(test_backend_provider_examples PRIVATE
  ${REPO}/examples/backend_sdk)
if(ANDROID)
  target_link_libraries(test_backend_provider_examples PRIVATE neuralnetworks)
endif()

volvox_test(test_backend_manager
  SOURCES src/runtime/runtime_state.c src/kernels/thread_pool.c
          src/backends/backend_manager.c
          tests/test_backend_manager.c
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
  SOURCES ${FULL_ENGINE_SRCS} ${FULL_PROFILE_SRCS} ${FULL_SHADER_SRCS}
          tests/test_ptq_package.c
  DEFS VOLVOXAI_ENABLE_TRAINING=1 ${BK_DEFAULT}
  OPTS -O2 GEN full)
volvox_cuda_training_codegen_before(test_ptq_package)

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

# cpu_features.c backs kernel_platform.h, which gemm_f32.c now consults so its
# microkernel selection honours VOLVOXAI_CPU_ISA like every other kernel.
volvox_test(test_gemm_f32
  SOURCES src/kernels/gemm_f32.c src/kernels/thread_pool.c
          src/kernels/cpu_features.c tests/test_gemm_f32.c
  OPTS -std=c11 -O3 -Wall -Wextra -Werror)

volvox_test(test_gemm_f32_runtime
  SOURCES ${CPU_ONLY_SRCS} tests/test_gemm_f32_runtime.c
  DEFS VOLVOXAI_ENABLE_TRAINING=0 ${BK_CPU_ONLY}
  OPTS -std=c11 -O3)

# Full-profile engine smoke tests (were built by the `build_native` aggregate).
foreach(_t adapter_runtime training_backward quantized_runtime)
  volvox_test(test_${_t}
    SOURCES ${FULL_ENGINE_SRCS} ${FULL_SHADER_SRCS} tests/test_${_t}.c
    DEFS VOLVOXAI_ENABLE_TRAINING=1 ${BK_DEFAULT}
    OPTS -O2 GEN full)
  volvox_cuda_training_codegen_before(test_${_t})
endforeach()
volvox_test(test_dynamic_autograd
  SOURCES ${FULL_ENGINE_SRCS} ${FULL_PROFILE_SRCS} ${FULL_SHADER_SRCS}
          tests/test_dynamic_autograd.c
  DEFS VOLVOXAI_ENABLE_TRAINING=1 ${BK_DEFAULT}
  OPTS -O2 GEN full)
volvox_cuda_training_codegen_before(test_dynamic_autograd)
if(VOLVOXAI_ENABLE_CUDA)
  # Public CUDA TrainStep tests assert transfer-byte boundaries exposed only
  # by the backend's test instrumentation.
  target_compile_definitions(test_training_backward PRIVATE
    VOLVOXAI_CUDA_TESTING=1)
  # Dynamic autograd requires strict CUDA forward and backward dispatch; the
  # test-only launch counter proves neither phase silently used CPU fallback.
  target_compile_definitions(test_dynamic_autograd PRIVATE
    VOLVOXAI_CUDA_TESTING=1)
endif()

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
  COMMAND ${PY} ${REPO}/tools/tests/test_native_cli.py
          $<TARGET_FILE:volvoxai> $<TARGET_FILE:volvoxai-full>)
set_tests_properties(native_cli PROPERTIES LABELS native)

add_test(NAME native_training_boundary
  COMMAND ${PY} ${REPO}/tools/tests/test_native_training_boundary.py -v)
set_tests_properties(native_training_boundary PROPERTIES
  LABELS native WORKING_DIRECTORY ${REPO})

add_test(NAME cuda_ptx_embedding
  COMMAND ${PY} ${REPO}/tools/tests/test_embed_cuda_ptx.py -v)
set_tests_properties(cuda_ptx_embedding PROPERTIES
  LABELS native WORKING_DIRECTORY ${REPO})

add_test(NAME cuda_fp32_build_contract
  COMMAND ${PY} ${REPO}/tools/tests/test_cuda_fp32_contract.py -v
          --cmake ${CMAKE_COMMAND} --c-compiler ${CMAKE_C_COMPILER})
set_tests_properties(cuda_fp32_build_contract PROPERTIES
  LABELS native WORKING_DIRECTORY ${REPO})

add_test(NAME cuda_source_composition
  COMMAND ${PY} ${REPO}/tools/tests/test_cuda_source_composition.py -v)
set_tests_properties(cuda_source_composition PROPERTIES
  LABELS native WORKING_DIRECTORY ${REPO})

add_test(NAME cuda_state_ownership_audit
  COMMAND ${PY} ${REPO}/tools/tests/test_cuda_state_ownership.py -v)
set_tests_properties(cuda_state_ownership_audit PROPERTIES
  LABELS native WORKING_DIRECTORY ${REPO})

add_test(NAME metal_nnapi_state_ownership_audit
  COMMAND ${PY} ${REPO}/tools/tests/test_metal_nnapi_state_ownership.py -v)
set_tests_properties(metal_nnapi_state_ownership_audit PROPERTIES
  LABELS native WORKING_DIRECTORY ${REPO})

# --- Opt-in example policies (label: example) ---------------------------------
add_executable(volvoxai-tasks
  ${ENGINE_SRCS} ${INFERENCE_SHADER_SRCS}
  ${REPO}/examples/native_support/image_io.c
  ${REPO}/examples/native_task_cli/main.c)
add_dependencies(volvoxai-tasks volvoxai_pack_inference_shaders)
volvox_cuda_codegen_before(volvoxai-tasks)
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
  add_dependencies(${_bin} volvoxai_pack_inference_shaders)
  volvox_cuda_codegen_before(${_bin})
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

foreach(_variant tiny_receipt_split_w8a8:main_split
                 test_tiny_receipt_split_w8a8:test_tiny_receipt_split_w8a8)
  string(REPLACE ":" ";" _pair ${_variant})
  list(GET _pair 0 _bin)
  list(GET _pair 1 _entry)
  add_executable(${_bin}
    ${ENGINE_SRCS} ${INFERENCE_SHADER_SRCS}
    ${REPO}/examples/tiny_receipt_vqa/native/tiny_receipt_image.c
    ${REPO}/examples/tiny_receipt_vqa/native/tiny_receipt_split_w8a8.c
    ${REPO}/examples/tiny_receipt_vqa/native/${_entry}.c)
  add_dependencies(${_bin} volvoxai_pack_inference_shaders)
  volvox_cuda_codegen_before(${_bin})
  target_link_libraries(${_bin} PRIVATE volvox_common)
  target_compile_definitions(${_bin} PRIVATE VOLVOXAI_ENABLE_TRAINING=0 ${BK_DEFAULT})
  target_include_directories(${_bin} PRIVATE
    ${GEN}/inference ${REPO}/examples/tiny_receipt_vqa/native)
  target_compile_options(${_bin} PRIVATE -O3)
endforeach()
add_test(NAME test_tiny_receipt_split_native_example
  COMMAND ${CMAKE_COMMAND}
    -DBIN=$<TARGET_FILE:tiny_receipt_split_w8a8>
    -DTEST_BIN=$<TARGET_FILE:test_tiny_receipt_split_w8a8>
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_tiny_receipt.cmake)
set_tests_properties(test_tiny_receipt_split_native_example PROPERTIES LABELS example)

# Canonical W8A8 affine contract. Proves the shared header before any kernel is
# migrated onto it, including bit equality with the device backends' multiplier.
volvox_test(test_w8a8_affine
  SOURCES tests/test_w8a8_affine.c
  OPTS -std=c11 -O2 -Wall -Wextra -Werror
  LABELS native)

# Resolved per-CPU kernel platform: the table every dispatcher reads instead of
# probing CPUID inline. Pins that it never invents a capability and that the
# VOLVOXAI_CPU_ISA clamp only ever removes one.
volvox_test(test_kernel_platform
  SOURCES tests/test_kernel_platform.c src/kernels/cpu_features.c
  OPTS -std=c11 -O2 -Wall -Wextra -Werror
  LABELS native)

# Byte-exact parity of every ISA tier the host can run, for the raw QLinear and
# QConv dispatchers. The VNNI kernels are unreachable through the packed path, so
# this is the only coverage they get; on a pre-VNNI CPU it proves only the tiers
# that exist, which is why it prints the tier list.
volvox_test(test_qlinear_isa_parity
  SOURCES src/kernels/kernels.c src/kernels/cpu_features.c
          src/kernels/quant_cpu_opt.c
          src/kernels/qlinear_w8a8_x86.c src/kernels/qlinear_w8a8_arm.c
          src/kernels/qconv_w8a8_x86.c src/kernels/qconv_w8a8_arm.c ${ARM_OBJS}
          tests/test_qlinear_isa_parity.c
  OPTS -O2 ${BENCH_CPU_FLAGS}
  LABELS native)

# The web build compiles kernel sources no native target reaches. Keep it honest
# from the native suite so a shared-kernel rename cannot break wasm silently.
add_test(NAME verify_wasm_build
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/verify_wasm_build.sh ${REPO})
set_tests_properties(verify_wasm_build PROPERTIES LABELS verify)

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
          src/kernels/quant_cpu_opt.c
          src/kernels/qsdpa_w8a8_native.c tests/benchmark_qsdpa_w8a8_native.c
  OPTS -std=c11 -O3
  LABELS benchmark)

volvox_test(benchmark_gemm_f32
  SOURCES src/kernels/gemm_f32.c src/kernels/thread_pool.c
          src/kernels/cpu_features.c tests/benchmark_gemm_f32.c
  # No BENCH_CPU_FLAGS: gemm_f32.c dispatches its microkernel at runtime, so the
  # flags would only stop VOLVOXAI_CPU_ISA from selecting the scalar tier.
  OPTS -std=c11 -O3
  LABELS benchmark)

# Smallest-unit per-kernel throughput plus exactness against the portable
# reference. Emits CSV for tools/compare_kernel_onnx.py to join against the same
# shapes run by ONNX Runtime.
volvox_test(benchmark_kernel_unit
  SOURCES src/kernels/kernels.c src/kernels/cpu_features.c
          src/kernels/quant_cpu_opt.c
          src/kernels/qlinear_w8a8_x86.c src/kernels/qlinear_w8a8_arm.c
          src/kernels/qconv_w8a8_x86.c src/kernels/qconv_w8a8_arm.c
          src/kernels/qsdpa_w8a8_native.c src/kernels/qgroupnorm_w8a8_native.c
          src/kernels/qnorm_activation_w8a8_native.c
          src/kernels/qbatch_matmul_w8a8_native.c
          src/kernels/transpose_w8a8_native.c src/kernels/tensor_f32_opt.c
          src/kernels/conv_f32_opt_baseline.c src/kernels/conv_f32_opt_avx2.c
          src/kernels/conv_f32_dispatch.c src/runtime/runtime_state.c
          ${ARM_OBJS}
          tests/benchmark_kernel_unit.c
  # Deliberately no BENCH_CPU_FLAGS.  Every kernel this target measures selects
  # its ISA at runtime, so giving the whole target -mavx2 would only make
  # VOLVOXAI_CPU_ISA=baseline meaningless: the "baseline" copy of
  # conv_f32_opt.c would be compiled with AVX2 too and the clamp would compare
  # a tier against itself.  Without the flags the two copies really differ, and
  # one binary can report both tiers.
  OPTS -O3
  LABELS benchmark)

# --- GPU device tests (label: gpu; build here, run needs a real device) -------
if(VOLVOXAI_ENABLE_CUDA)
  volvox_test(test_cuda_kernels
    SOURCES ${CUDA_BACKEND_SRCS} src/runtime/runtime_state.c
            src/kernels/thread_pool.c tests/test_cuda_kernels.c
    DEFS VOLVOXAI_ENABLE_TRAINING=0 VOLVOXAI_ENABLE_CUDA=1
         VOLVOXAI_CUDA_TESTING=1
         VOLVOXAI_CUDA_FAST_FP32=$<BOOL:${VOLVOXAI_CUDA_FAST_FP32}>
    OPTS -std=c11 -O2
    LABELS gpu)
  set_tests_properties(test_cuda_kernels PROPERTIES SKIP_RETURN_CODE 77)
  volvox_test(test_cuda_runtime
    SOURCES ${CPU_ONLY_SRCS} ${CUDA_BACKEND_SRCS} ${SHADER_STORE_SRCS}
            ${EMB_CPU_ONLY} tests/test_cuda_runtime.c
    DEFS VOLVOXAI_ENABLE_TRAINING=0 VOLVOXAI_CUDA_TESTING=1 ${BK_CUDA}
    OPTS -std=c11 -O2 GEN cpu_only
    LABELS gpu)
  set_tests_properties(test_cuda_runtime PROPERTIES SKIP_RETURN_CODE 77)

  volvox_test(test_cuda_training
    SOURCES ${CUDA_BACKEND_SRCS} ${CUDA_TRAINING_PTX_SRCS}
            src/runtime/runtime_state.c src/kernels/thread_pool.c
            tests/test_cuda_training.c
    DEFS VOLVOXAI_ENABLE_TRAINING=1 VOLVOXAI_ENABLE_CUDA=1
         VOLVOXAI_ENABLE_VULKAN=0 VOLVOXAI_ENABLE_OPENGL=0
         VOLVOXAI_ENABLE_METAL=0 VOLVOXAI_ENABLE_NNAPI=0
         VOLVOXAI_CUDA_TESTING=1
         VOLVOXAI_CUDA_FAST_FP32=$<BOOL:${VOLVOXAI_CUDA_FAST_FP32}>
    LABELS "native;gpu")
  volvox_cuda_training_codegen_before(test_cuda_training)
  set_tests_properties(test_cuda_training PROPERTIES
    SKIP_RETURN_CODE 77
    ENVIRONMENT
      "VOLVOXAI_CUDA_PROFILE_PATH=${CMAKE_CURRENT_BINARY_DIR}/test_cuda_training-profile.csv")

  volvox_test(benchmark_cuda_linear
    SOURCES ${CUDA_BACKEND_SRCS} ${CUDA_TRAINING_PTX_SRCS}
            src/runtime/runtime_state.c src/kernels/thread_pool.c
            tests/benchmark_cuda_linear.c
    DEFS VOLVOXAI_ENABLE_TRAINING=1 VOLVOXAI_CUDA_TESTING=1 ${BK_CUDA}
         VOLVOXAI_CUDA_FAST_FP32=$<BOOL:${VOLVOXAI_CUDA_FAST_FP32}>
    OPTS -std=c11 -O3
    LABELS "benchmark;gpu")
  volvox_cuda_training_codegen_before(benchmark_cuda_linear)
  set_tests_properties(benchmark_cuda_linear PROPERTIES SKIP_RETURN_CODE 77)

  volvox_test(test_cuda_state_ownership
    SOURCES ${CUDA_BACKEND_SRCS} src/runtime/runtime_state.c
            src/kernels/thread_pool.c tests/test_cuda_state_ownership.c
    DEFS VOLVOXAI_ENABLE_TRAINING=0 VOLVOXAI_ENABLE_CUDA=1
         VOLVOXAI_CUDA_TESTING=1
         VOLVOXAI_CUDA_FAST_FP32=$<BOOL:${VOLVOXAI_CUDA_FAST_FP32}>
    OPTS -std=c11 -O2 -Wall -Wextra -Werror
    LABELS native)

  volvox_test(test_cuda_ptq
    SOURCES ${FULL_ENGINE_SRCS} ${FULL_PROFILE_SRCS} ${FULL_SHADER_SRCS}
            tests/test_cuda_ptq.c
    DEFS VOLVOXAI_ENABLE_TRAINING=1 VOLVOXAI_CUDA_TESTING=1 ${BK_CUDA}
    OPTS -std=c11 -O2 GEN full
    LABELS "native;gpu")
  volvox_cuda_training_codegen_before(test_cuda_ptq)
  set_tests_properties(test_cuda_ptq PROPERTIES SKIP_RETURN_CODE 77)
endif()
if(VOLVOXAI_ENABLE_VULKAN)
  volvox_test(test_vulkan_training
    SOURCES src/runtime/runtime_state.c src/kernels/thread_pool.c
            src/backends/vulkan_engine.c ${FULL_SHADER_SRCS} tests/test_vulkan_training.c
    DEFS VOLVOXAI_ENABLE_TRAINING=1 VOLVOXAI_VULKAN_TESTING=1
    OPTS -O2 GEN full
    LABELS gpu)
  set_tests_properties(test_vulkan_training PROPERTIES SKIP_RETURN_CODE 77)
endif()
if(VOLVOXAI_ENABLE_OPENGL)
  volvox_test(test_opengl_training
    SOURCES src/runtime/runtime_state.c src/kernels/thread_pool.c
            src/backends/opengl_engine.c ${FULL_SHADER_SRCS}
            tests/test_opengl_training.c
    DEFS VOLVOXAI_ENABLE_TRAINING=1 VOLVOX_OPENGL_TESTING
    OPTS -O2 GEN full
    LABELS gpu)
endif()
if(APPLE AND VOLVOXAI_ENABLE_METAL)
  volvox_test(test_metal_training
    SOURCES src/runtime/runtime_state.c src/kernels/thread_pool.c
            src/backends/metal_engine.m ${FULL_SHADER_SRCS}
            tests/test_metal_training.m
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
  add_test(NAME verify_native_qlinear_maddubs
    COMMAND ${_verify} emit-maddubs ${_kern}/qlinear_w8a8_x86.c ${_kern})
  add_test(NAME verify_native_qbatch_maddubs
    COMMAND ${_verify} emit-maddubs
            ${_kern}/qbatch_matmul_w8a8_native.c ${_kern})
  add_test(NAME verify_native_packed_qgemm_maddubs
    COMMAND ${_verify} emit-maddubs ${_kern}/packed_quant_gemm.c ${_kern})
  add_test(NAME verify_native_packed_qgemm_signed_maddubs
    COMMAND ${_verify} emit-signed-maddubs
            ${_kern}/packed_quant_gemm.c ${_kern})
  add_test(NAME verify_native_w8a8_baseline_isa
    COMMAND ${_verify} baseline ${_kern})
  set_tests_properties(
    verify_native_avxvnni verify_native_avx512vnni
    verify_native_qconv_avxvnni verify_native_qconv_avx512vnni
    verify_native_qlinear_maddubs verify_native_qbatch_maddubs
    verify_native_packed_qgemm_maddubs
    verify_native_packed_qgemm_signed_maddubs
    verify_native_w8a8_baseline_isa
    PROPERTIES LABELS verify)
endif()
