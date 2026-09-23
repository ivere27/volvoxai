# Native engine production profiles and executables.
# Native build composition is defined here for every supported profile.

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.20)
  get_filename_component(_volvox_repo_root
    "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_volvox_repo_root}/package.json")
  file(READ "${_volvox_repo_root}/package.json" _volvox_pkg_json)
  string(JSON VOLVOXAI_VERSION GET "${_volvox_pkg_json}" version)
  find_package(Git QUIET)
  set(VOLVOXAI_GIT_COMMIT "unknown")
  if(GIT_FOUND)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-parse --short=12 HEAD
      WORKING_DIRECTORY "${_volvox_repo_root}"
      OUTPUT_VARIABLE _volvox_git_commit
      RESULT_VARIABLE _volvox_git_commit_result
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if("${_volvox_git_commit_result}" STREQUAL "0" AND
       NOT "${_volvox_git_commit}" STREQUAL "")
      set(VOLVOXAI_GIT_COMMIT "${_volvox_git_commit}")
      execute_process(
        COMMAND ${GIT_EXECUTABLE} status --porcelain
        WORKING_DIRECTORY "${_volvox_repo_root}"
        OUTPUT_VARIABLE _volvox_git_dirty
        RESULT_VARIABLE _volvox_git_dirty_result
        ERROR_QUIET)
      if("${_volvox_git_dirty_result}" STREQUAL "0" AND
         NOT "${_volvox_git_dirty}" STREQUAL "")
        set(VOLVOXAI_GIT_COMMIT "${VOLVOXAI_GIT_COMMIT}-dirty")
      elseif(NOT "${_volvox_git_dirty_result}" STREQUAL "0")
        set(VOLVOXAI_GIT_COMMIT "${VOLVOXAI_GIT_COMMIT}-status-unknown")
      endif()
    endif()
  endif()
  if(NOT DEFINED VOLVOXAI_BUILD_DATE)
    string(TIMESTAMP VOLVOXAI_BUILD_DATE "%Y-%m-%dT%H:%M:%SZ" UTC)
  endif()
  project(volvoxai_native VERSION ${VOLVOXAI_VERSION} LANGUAGES C)
  enable_testing()
endif()

get_filename_component(REPO "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(GEN  ${CMAKE_BINARY_DIR}/gen)        # generated assets (was build/native/*)
get_property(VOLVOXAI_NATIVE_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
set(PY "${Python3_EXECUTABLE}")

# Module sources are vendored from the same pinned revision as the generator.
# Static callers reference Synurang_GetApi; the linker follows its module and
# generated registration dependencies without constructors or bootstrap glue.
set(VOLVOXAI_SYNURANG_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/synurang")
set(VOLVOXAI_SYNURANG_RUNTIME_HEADER
  "${VOLVOXAI_SYNURANG_DIR}/include/synurang/call.h")

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

# Machine-readable contracts and reference docs share the public schema.
execute_process(
  COMMAND ${PY} ${REPO}/tools/generate_api_contract.py --check-manifest
  WORKING_DIRECTORY ${REPO}
  RESULT_VARIABLE VOLVOXAI_API_CONTRACT_CHECK_RESULT)
if(NOT VOLVOXAI_API_CONTRACT_CHECK_RESULT EQUAL 0)
  message(FATAL_ERROR "Generated API contract is stale. Run make proto_codegen.")
endif()

# proto/operator_vocabulary.proto owns the implementation-only graph operator
# IDs and spellings. It must never re-enter a public header or Synurang output.
execute_process(
  COMMAND ${PY} ${REPO}/tools/generate_operator_vocabulary.py --check
  WORKING_DIRECTORY ${REPO}
  RESULT_VARIABLE VOLVOXAI_OPERATOR_VOCABULARY_CHECK_RESULT
  OUTPUT_VARIABLE VOLVOXAI_OPERATOR_VOCABULARY_CHECK_OUTPUT
  ERROR_VARIABLE VOLVOXAI_OPERATOR_VOCABULARY_CHECK_ERROR)
if(NOT VOLVOXAI_OPERATOR_VOCABULARY_CHECK_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Generated operator vocabulary is stale. "
    "Run make operator_vocabulary_codegen.\n"
    "${VOLVOXAI_OPERATOR_VOCABULARY_CHECK_OUTPUT}${VOLVOXAI_OPERATOR_VOCABULARY_CHECK_ERROR}")
endif()

# proto/operator_param_registry.proto owns the internal typed parameter IDs,
# JSON spellings and profile-specific operator allowlists.  Runtime execution
# consumes only the tracked dependency-free projections, never protobuf.
execute_process(
  COMMAND ${PY} ${REPO}/tools/generate_operator_param_registry.py --check
  WORKING_DIRECTORY ${REPO}
  RESULT_VARIABLE VOLVOXAI_OPERATOR_PARAM_CHECK_RESULT
  OUTPUT_VARIABLE VOLVOXAI_OPERATOR_PARAM_CHECK_OUTPUT
  ERROR_VARIABLE VOLVOXAI_OPERATOR_PARAM_CHECK_ERROR)
if(NOT VOLVOXAI_OPERATOR_PARAM_CHECK_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Generated operator parameter registry is stale. "
    "Run make operator_param_registry_codegen.\n"
    "${VOLVOXAI_OPERATOR_PARAM_CHECK_OUTPUT}${VOLVOXAI_OPERATOR_PARAM_CHECK_ERROR}")
endif()

# proto/volvoxai.proto is the public API. Fail before compilation when an RPC
# has no dispatch or handler, when a handler exists for an undeclared RPC, or
# when a public header reintroduces an engine function or a hand-written enum.
execute_process(
  COMMAND ${PY} ${REPO}/tools/check_api_conformance.py --check
  WORKING_DIRECTORY ${REPO}
  RESULT_VARIABLE VOLVOXAI_API_CONFORMANCE_RESULT
  OUTPUT_VARIABLE VOLVOXAI_API_CONFORMANCE_OUTPUT
  ERROR_VARIABLE VOLVOXAI_API_CONFORMANCE_ERROR)
if(NOT VOLVOXAI_API_CONFORMANCE_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Public API conformance failed.\n"
    "${VOLVOXAI_API_CONFORMANCE_OUTPUT}${VOLVOXAI_API_CONFORMANCE_ERROR}")
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
execute_process(
  COMMAND ${PY} ${REPO}/tools/generate_tokenizer_unicode.py --check
  WORKING_DIRECTORY ${REPO}
  RESULT_VARIABLE VOLVOXAI_UNICODE_CHECK_RESULT)
if(NOT VOLVOXAI_UNICODE_CHECK_RESULT EQUAL 0)
  message(FATAL_ERROR "Tokenizer Unicode verification failed.")
endif()

option(VOLVOXAI_ENABLE_VULKAN "Link the Vulkan device backend" ON)
option(VOLVOXAI_ENABLE_OPENGL "Link the OpenGL device backend" ON)
option(VOLVOXAI_ENABLE_CUDA "Link the runtime-loaded CUDA device backend" ON)
if(APPLE)
  option(VOLVOXAI_ENABLE_METAL "Link the Metal device backend" ON)
else()
  option(VOLVOXAI_ENABLE_METAL "Link the Metal device backend" OFF)
endif()
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
# Optional Arm objects are used on aarch64 Linux and Android. macOS arm64 uses
# its platform-specific kernels.  Each higher ISA remains in a separate object
# library so the baseline executable still starts on Armv8-A CPUs.
if(_arch MATCHES "aarch64|arm64" AND
   (CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Android"))
  set(VOLVOX_ARM_DOTPROD TRUE)
  set(VOLVOX_ARM_I8MM TRUE)
else()
  set(VOLVOX_ARM_DOTPROD FALSE)
  set(VOLVOX_ARM_I8MM FALSE)
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

# Common build interface: internal includes, provenance defines, -pthread,
# libm/libdl. Profile-generated API and full-only training includes are added
# explicitly to each target below; putting them here lets inference silently
# compile against the full projection.
# Backend-enable and TRAINING macros are deliberately set per target so every
# production profile carries an explicit composition contract.
add_library(volvox_common INTERFACE)
target_include_directories(volvox_common INTERFACE
  ${INC}
  ${VOLVOXAI_SYNURANG_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src/api
  ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime)
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
# metal_engine.m needs the Apple frameworks.
if(APPLE AND VOLVOXAI_ENABLE_METAL)
  target_link_libraries(volvox_common INTERFACE "-framework Foundation" "-framework Metal")
endif()

# Select the only generated API projection a target may see. Full-only native
# sources also get their private training header directory; inference targets
# never inherit either path through a common interface.
function(volvox_target_api_profile target profile)
  if(profile STREQUAL "full")
    target_include_directories(${target} BEFORE PRIVATE
      ${REPO}/runtime/generated/c)
    target_include_directories(${target} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/training)
  elseif(profile STREQUAL "inference")
    target_include_directories(${target} BEFORE PRIVATE
      ${REPO}/runtime/generated/c/inference)
  else()
    message(FATAL_ERROR "Unknown VolvoxAI API profile '${profile}'")
  endif()
endfunction()
# Backend-define bundles used by target_compile_definitions below.
set(BK_FULL
  VOLVOXAI_ENABLE_VULKAN=$<IF:$<BOOL:${VOLVOXAI_ENABLE_VULKAN}>,1,0>
  VOLVOXAI_ENABLE_OPENGL=$<IF:$<BOOL:${VOLVOXAI_ENABLE_OPENGL}>,1,0>
  VOLVOXAI_ENABLE_CUDA=$<IF:$<BOOL:${VOLVOXAI_ENABLE_CUDA}>,1,0>
  VOLVOXAI_ENABLE_METAL=$<IF:$<BOOL:${VOLVOXAI_ENABLE_METAL}>,1,0>)
set(BK_LITE
  VOLVOXAI_ENABLE_VULKAN=0
  VOLVOXAI_ENABLE_OPENGL=0
  VOLVOXAI_ENABLE_CUDA=0
  VOLVOXAI_ENABLE_METAL=0)
set(BK_DEFAULT ${BK_FULL})
# --- ARM dotprod objects (Armv8.2 SDOT), separately-attributed like the .o's --
set(ARM_OBJS "")
if(VOLVOX_ARM_DOTPROD)
  add_library(volvox_arm_dotprod OBJECT
    src/kernels/qlinear_w8a8_arm_dotprod.c
    src/kernels/qconv_w8a8_arm_dotprod.c)
  target_include_directories(volvox_arm_dotprod PRIVATE ${INC})
  target_compile_options(volvox_arm_dotprod PRIVATE -O3 -march=armv8.2-a+dotprod -fno-lto)
  # These objects also land in libvolvoxai.so.
  set_target_properties(volvox_arm_dotprod PROPERTIES POSITION_INDEPENDENT_CODE ON)
  list(APPEND ARM_OBJS $<TARGET_OBJECTS:volvox_arm_dotprod>)
  add_compile_definitions(VOLVOXAI_ARM_DOTPROD_OBJECT=1)
endif()

# Arm I8MM is an independent architectural extension and is never enabled for
# the baseline or SDOT objects.  Keep the source discovery explicit: adding an
# I8MM implementation under either canonical kernel name makes it available to
# all native compositions without forcing an empty placeholder file.
if(VOLVOX_ARM_I8MM)
  set(_arm_i8mm_sources "")
  foreach(_arm_i8mm_source IN ITEMS
      src/kernels/qlinear_w8a8_arm_i8mm.c
      src/kernels/qconv_w8a8_arm_i8mm.c
      src/kernels/qbatch_matmul_w8a8_arm_i8mm.c)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_arm_i8mm_source}")
      list(APPEND _arm_i8mm_sources ${_arm_i8mm_source})
    endif()
  endforeach()
  if(_arm_i8mm_sources)
    add_library(volvox_arm_i8mm OBJECT ${_arm_i8mm_sources})
    target_include_directories(volvox_arm_i8mm PRIVATE ${INC})
    target_compile_options(volvox_arm_i8mm PRIVATE
      -O3 -march=armv8.2-a+i8mm -fno-lto -ffp-contract=off)
    set_target_properties(volvox_arm_i8mm PROPERTIES POSITION_INDEPENDENT_CODE ON)
    list(APPEND ARM_OBJS $<TARGET_OBJECTS:volvox_arm_i8mm>)
    add_compile_definitions(VOLVOXAI_ARM_I8MM_OBJECT=1)
  endif()
endif()

# --- Source groups (mirror NATIVE_*_SRCS) -------------------------------------
# proto/volvoxai.proto is the public API. The generated dispatch decodes each
# request and calls a handler table. The handler implementation is common, but
# each capability profile compiles its own generated lite codec and dispatch so
# inference contains neither authoring entry points nor authoring message
# codec symbols.
set(API_COMMON_SRCS
  src/api/vx_api.c
  third_party/synurang/src/c_runtime.c
  third_party/synurang/src/call.c
  src/api/vx_api_handles.c
  src/api/vx_api_buffer.c
  src/api/vx_api_convert.c
  src/api/vx_api_memory.c
  src/api/vx_api_profiling.c
  src/api/vx_api_platform.c
  src/api/vx_api_inference.c
  src/api/vx_api_scheduler.c
  src/api/vx_api_text.c)
set(INFERENCE_API_GENERATED_SRCS
  ${REPO}/runtime/generated/c/inference/volvoxai_lite.c
  ${REPO}/runtime/generated/c/inference/volvoxai_ffi.c)
set(FULL_API_GENERATED_SRCS
  ${REPO}/runtime/generated/c/volvoxai_lite.c
  ${REPO}/runtime/generated/c/volvoxai_ffi.c)

set(RUNTIME_SRCS
  src/generated/operator_vocabulary.c
  src/generated/kernel_registry.c
  third_party/cJSON.c
  src/runtime/safetensors_header.c src/runtime/safetensors_bytes.c
  src/runtime/safetensors.c src/runtime/tokenizer.c
  src/runtime/adapter_runtime.c src/runtime/attention_mask.c
  src/runtime/compiled_weight_resources.c
  src/runtime/sequence_runtime.c src/runtime/backend.c src/runtime/arena.c
  src/runtime/backend_sdk.c src/runtime/runtime_state.c src/runtime/public_api.c
  src/runtime/native_tensor.c src/runtime/profiling.c
  src/runtime/scheduler_policy.c
  src/runtime/shape_contract.c src/runtime/shape_domain_contract.c
  src/runtime/graph_plan.c src/runtime/activation_plan.c
  src/runtime/graph_bind.c src/runtime/graph_bind_definition.c
  src/runtime/graph_domain.c src/runtime/independent_batch_proof.c
  src/runtime/call_sequence_policy.c
  src/runtime/paged_kv.c src/runtime/paged_binding.c src/runtime/batch_scheduler.c src/runtime/continuous_batch_scheduler.c
  src/runtime/decode_row_set.c src/runtime/batch_decode.c
  src/runtime/incremental_runtime.c src/runtime/decode_session.c
  src/backends/w8a8_device_ops.c src/runtime/engine_runtime.c src/runtime/engine_state.c)
set(RUNTIME_HOT_RELEASE_SRCS
  src/backends/w8a8_device_ops.c
  src/runtime/engine_runtime.c
  src/runtime/engine_state.c
  src/runtime/runtime_state.c
  src/runtime/adapter_runtime.c
  src/runtime/paged_kv.c
  src/runtime/paged_binding.c
  src/runtime/incremental_runtime.c
  src/runtime/decode_session.c)
set(RUNTIME_COLD_RELEASE_SRCS ${RUNTIME_SRCS})
list(REMOVE_ITEM RUNTIME_COLD_RELEASE_SRCS ${RUNTIME_HOT_RELEASE_SRCS})
set(KERNEL_SRCS
  src/kernels/kernels.c src/kernels/cpu_features.c src/kernels/quant_cpu_isa.c
  src/kernels/qlinear_w8a8_x86.c src/kernels/qlinear_w8a8_arm.c
  src/kernels/qconv_w8a8_x86.c src/kernels/qconv_w8a8_arm.c
  src/kernels/qbatch_matmul_w8a8_native.c
  src/kernels/qsdpa_w8a8_native.c src/kernels/qgroupnorm_w8a8_native.c
  src/kernels/qnorm_activation_w8a8_native.c
  src/kernels/transpose_w8a8_native.c
  src/kernels/paged_attention.c
  src/kernels/conv_f32_isa_baseline.c src/kernels/conv_f32_isa_avx2.c
  src/kernels/conv_f32_dispatch.c src/kernels/tensor_f32_isa.c)

# conv_f32_isa.c holds hand-written AVX2 conv bodies behind `#if
# defined(__AVX2__)`.  Compiling it only at VOLVOXAI_CPU_TARGET (default
# `baseline`) preprocessed them away, so AVX2 machines ran the scalar fallback —
# substantially slower on representative encoder Conv2D nodes.  Build the same
# source a second time with the higher ISA and let conv_f32_dispatch.c pick at
# runtime.  Only this one file gets the flags, and it is reached only after a
# vx_kernel_platform() check, so the binary's own CPU requirement is unchanged.
#
# These are plain sources, not a shared OBJECT library: conv_f32_isa.c reads
# VxEngineState, whose layout depends on each target's VOLVOXAI_ENABLE_TRAINING.
if(VOLVOX_X86)
  set_source_files_properties(src/kernels/conv_f32_isa_avx2.c
    PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
endif()
set(BACKEND_CORE_SRCS src/backends/backend_manager.c)
set(SHADER_STORE_SRCS
  src/shader_store.c
  third_party/xz-embedded/xz_crc32.c
  third_party/xz-embedded/xz_dec_lzma2.c
  third_party/xz-embedded/xz_dec_stream.c)
set(FULL_PROFILE_SRCS
  src/generated/kernel_registry_full.c
  src/kernels/training_kernels.c
  src/api/vx_api_training.c src/api/vx_api_quantization.c
  src/api/vx_api_planning.c
  src/training/trainer_api.c src/training/ptq_api.c src/training/quantization.c
  src/training/quantization_runtime.c src/training/quantization_package.c
  src/training/ptq_names.c src/training/ptq_authoring.c
  src/training/training_control.c)
set(FULL_PROFILE_HOT_RELEASE_SRCS
  src/kernels/training_kernels.c
  src/training/quantization.c)
set(FULL_PROFILE_COLD_RELEASE_SRCS ${FULL_PROFILE_SRCS})
list(REMOVE_ITEM FULL_PROFILE_COLD_RELEASE_SRCS ${FULL_PROFILE_HOT_RELEASE_SRCS})
set(CLI_SRCS cli/main.c)

# Audit every native C/Objective-C source without making a glob the build
# authority. New translation units must be deliberately classified as a
# release root or as a text-included/WASM-only fragment; otherwise configure
# fails instead of silently omitting code from the hot/cold composition.
set(VOLVOXAI_NATIVE_OPTIONAL_RELEASE_ROOT_SRCS
  src/backends/cuda_engine.c
  src/backends/metal_engine.m
  src/backends/opengl_engine.c
  src/backends/vulkan_engine.c
  src/kernels/qlinear_w8a8_arm_dotprod.c
  src/kernels/qconv_w8a8_arm_dotprod.c
  src/kernels/qlinear_w8a8_arm_i8mm.c
  src/kernels/qconv_w8a8_arm_i8mm.c
  src/kernels/qbatch_matmul_w8a8_arm_i8mm.c)
set(VOLVOXAI_NATIVE_NONROOT_SRCS
  # Included by generated api/vx_api_composition.c.
  src/api/vx_api.c
  # Included by the native kernel amalgamation/ISA wrappers.
  src/kernels/conv_f32_isa.c
  src/kernels/gemm_f32.c
  src/kernels/packed_quant_gemm.c
  src/kernels/sequence_ops.c
  src/kernels/thread_pool.c
  src/kernels/wasm_heap_arena.c
  # WebAssembly profile roots, never native release translation units.
  # webgpu_backend.c reaches a device only through the host bridge, which no
  # native build has; it is text-included by the browser amalgamation.
  src/backends/webgpu_backend.c
  src/kernels/qlinear_w8a8_wasm_relaxed.c
  src/runtime/portable_control_wasm.c
  src/runtime/wasm_libc.c
  src/runtime/wasm_decimal.c
  src/training/ptq_authoring_wasm.c
  src/runtime/runtime_lifecycle.c
  # Portable scalar libm for the freestanding WASM profiles.
  src/runtime/wasm_math.c
  # WASM bridge and optional native consumer loader are not engine roots.
  third_party/synurang/src/wasm.c
  third_party/synurang/src/module_host.c)
set(_volvox_native_declared_sources
  ${API_COMMON_SRCS}
  ${RUNTIME_SRCS}
  ${KERNEL_SRCS}
  ${BACKEND_CORE_SRCS}
  ${FULL_PROFILE_SRCS}
  ${SHADER_STORE_SRCS}
  ${CLI_SRCS}
  ${VOLVOXAI_NATIVE_OPTIONAL_RELEASE_ROOT_SRCS}
  ${VOLVOXAI_NATIVE_NONROOT_SRCS})
list(FILTER _volvox_native_declared_sources INCLUDE REGEX
  "^(src|cli|third_party)/.*\\.(c|m)$")
list(REMOVE_DUPLICATES _volvox_native_declared_sources)
list(SORT _volvox_native_declared_sources)
file(GLOB_RECURSE _volvox_native_discovered_sources
  CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.m"
  "${CMAKE_CURRENT_SOURCE_DIR}/cli/*.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/cli/*.m"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/*.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/*.m")
list(SORT _volvox_native_discovered_sources)
foreach(_source IN LISTS _volvox_native_discovered_sources)
  list(FIND _volvox_native_declared_sources "${_source}" _declared_index)
  if(_declared_index EQUAL -1)
    message(FATAL_ERROR
      "Native source '${_source}' has no release-root or non-root classification")
  endif()
endforeach()
foreach(_source IN LISTS _volvox_native_declared_sources)
  list(FIND _volvox_native_discovered_sources "${_source}" _discovered_index)
  if(_discovered_index EQUAL -1)
    message(FATAL_ERROR "Classified native source '${_source}' does not exist")
  endif()
endforeach()
set(VOLVOXAI_CUDA_DEVICE_SOURCE_INVENTORY
  src/backends/cuda_kernels.cu
  src/backends/cuda_training_kernels.cu)
file(GLOB_RECURSE _volvox_cuda_discovered_sources
  CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
list(SORT VOLVOXAI_CUDA_DEVICE_SOURCE_INVENTORY)
list(SORT _volvox_cuda_discovered_sources)
if(NOT "${VOLVOXAI_CUDA_DEVICE_SOURCE_INVENTORY}" STREQUAL
       "${_volvox_cuda_discovered_sources}")
  message(FATAL_ERROR
    "CUDA device-source inventory changed; classify every .cu profile root")
endif()

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
  if(VOLVOXAI_CUDA_ARCH LESS 61)
    message(FATAL_ERROR
      "VOLVOXAI_CUDA_ARCH must be at least 61 for the exact DP4A W8A8 kernels")
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
if(VOLVOXAI_ENABLE_METAL)
  list(APPEND BACKEND_DEVICE_SRCS src/backends/metal_engine.m)
endif()
set(BACKEND_DEVICE_HOT_RELEASE_SRCS ${BACKEND_DEVICE_SRCS})
set(BACKEND_DEVICE_COLD_RELEASE_SRCS "")
if(VOLVOXAI_ENABLE_CUDA)
  # Runtime CUDA dispatch is hot, but embedded PTX is immutable payload data.
  list(REMOVE_ITEM BACKEND_DEVICE_HOT_RELEASE_SRCS
    ${CUDA_EMBED_C} ${CUDA_EMBED_H})
  list(APPEND BACKEND_DEVICE_COLD_RELEASE_SRCS
    ${CUDA_EMBED_C} ${CUDA_EMBED_H})
endif()

# Full multi-backend engine source sets. Keep the separately compiled Arm ISA
# objects out of the corresponding *_COMPILE_SRCS lists and add them only to
# final executable source sets.
set(ENGINE_CORE_COMPILE_SRCS
  ${API_COMMON_SRCS}
  ${RUNTIME_SRCS} ${KERNEL_SRCS} ${BACKEND_CORE_SRCS})
set(ENGINE_COMPILE_SRCS
  ${ENGINE_CORE_COMPILE_SRCS} ${INFERENCE_API_GENERATED_SRCS})
set(FULL_ENGINE_COMPILE_SRCS
  ${ENGINE_CORE_COMPILE_SRCS} ${BACKEND_DEVICE_SRCS} ${FULL_API_GENERATED_SRCS}
  ${CUDA_TRAINING_PTX_SRCS})
set(ENGINE_SRCS ${ENGINE_COMPILE_SRCS} ${ARM_OBJS})
set(FULL_ENGINE_SRCS ${FULL_ENGINE_COMPILE_SRCS} ${ARM_OBJS})

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

file(GLOB WGSL_INFERENCE_SOURCES CONFIGURE_DEPENDS
  ${REPO}/shaders/inference/*.wgsl)
file(GLOB WGSL_TRAINING_SOURCES CONFIGURE_DEPENDS
  ${REPO}/shaders/training/*.wgsl)
set(WGSL_SOURCES ${WGSL_INFERENCE_SOURCES} ${WGSL_TRAINING_SOURCES})

set(SHADER_STAMP ${GEN}/.shaders_compiled.stamp)
add_custom_command(
  OUTPUT ${SHADER_STAMP}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GEN}
  # compile_shaders.sh drives `cargo`, which chokes on make's jobserver when
  # run as a build step; clear MAKEFLAGS so cargo builds standalone.
  # An optional absolute prebuilt compiler is forwarded for hermetic builders
  # that intentionally do not install Cargo.
  COMMAND ${CMAKE_COMMAND} -E env MAKEFLAGS= MAKELEVEL=
          VOLVOXAI_NATIVE_SHADER_COMPILER=$ENV{VOLVOXAI_NATIVE_SHADER_COMPILER}
          ${REPO}/tools/compile_shaders.sh
  COMMAND ${CMAKE_COMMAND} -E touch ${SHADER_STAMP}
  WORKING_DIRECTORY ${REPO}
  DEPENDS ${WGSL_SOURCES} ${REPO}/tools/compile_shaders.sh
  COMMENT "Compiling WGSL shaders with naga"
  VERBATIM)
add_custom_target(volvoxai_compile_shaders DEPENDS ${SHADER_STAMP})

set(INFERENCE_COMPILED_SHADER_DIR ${GEN}/compiled_shaders/inference)
set(INFERENCE_SHADER_STAMP ${GEN}/.inference_shaders_compiled.stamp)
add_custom_command(
  OUTPUT ${INFERENCE_SHADER_STAMP}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GEN}
  # Keep inference compilation physically separate from the compatible full
  # path above so both targets can run concurrently without deleting or
  # replacing one another's generated artifacts.
  COMMAND ${CMAKE_COMMAND} -E env MAKEFLAGS= MAKELEVEL=
          VOLVOXAI_SHADER_PROFILE=inference
          VOLVOXAI_NATIVE_SHADER_OUTPUT_DIR=${INFERENCE_COMPILED_SHADER_DIR}
          VOLVOXAI_NATIVE_SHADER_COMPILER=$ENV{VOLVOXAI_NATIVE_SHADER_COMPILER}
          ${REPO}/tools/compile_shaders.sh
  COMMAND ${CMAKE_COMMAND} -E touch ${INFERENCE_SHADER_STAMP}
  WORKING_DIRECTORY ${REPO}
  DEPENDS ${WGSL_INFERENCE_SOURCES} ${REPO}/tools/compile_shaders.sh
  COMMENT "Compiling inference WGSL shaders with naga"
  VERBATIM)
add_custom_target(volvoxai_compile_inference_shaders
  DEPENDS ${INFERENCE_SHADER_STAMP})

# profile: <logical> <pack-profile> <backend-args...> -> ${GEN}/<logical>/embedded_shaders.c
function(volvox_shader_pack logical pack_profile out_var)
  if(pack_profile STREQUAL "inference")
    set(compiled_dir ${INFERENCE_COMPILED_SHADER_DIR})
    set(shader_stamp ${INFERENCE_SHADER_STAMP})
    set(compile_target volvoxai_compile_inference_shaders)
    set(shader_source_dir ${REPO}/shaders/inference)
    set(source_scope_args --source-scope inference)
  elseif(pack_profile STREQUAL "full")
    set(compiled_dir ${REPO}/native/shaders)
    set(shader_stamp ${SHADER_STAMP})
    set(compile_target volvoxai_compile_shaders)
    set(shader_source_dir ${REPO}/shaders)
    set(source_scope_args "")
  else()
    message(FATAL_ERROR "Unsupported native shader pack profile: ${pack_profile}")
  endif()
  set(dir ${GEN}/${logical})
  set(outc ${dir}/embedded_shaders.c)
  set(outh ${dir}/embedded_shaders.h)
  add_custom_command(
    OUTPUT ${outc} ${outh}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${dir}
    COMMAND ${PY} ${REPO}/tools/pack_native_shaders.py
            --compiled-dir ${compiled_dir}
            --shader-source-dir ${shader_source_dir} ${source_scope_args}
            --profile ${pack_profile} ${ARGN}
            --output-c ${outc} --output-h ${outh}
    DEPENDS ${shader_stamp} ${REPO}/tools/pack_native_shaders.py
    COMMENT "Packing ${logical} embedded shaders"
    VERBATIM)
  add_custom_target(volvoxai_pack_${logical}_shaders DEPENDS ${outc} ${outh})
  add_dependencies(volvoxai_pack_${logical}_shaders ${compile_target})
  set(${out_var} ${outc} PARENT_SCOPE)
endfunction()

volvox_shader_pack(inference inference EMB_INFERENCE --backend none)
volvox_shader_pack(full      full      EMB_FULL      ${PACK_BACKENDS})

set(INFERENCE_SHADER_SRCS ${SHADER_STORE_SRCS} ${EMB_INFERENCE})
set(FULL_SHADER_SRCS      ${SHADER_STORE_SRCS} ${EMB_FULL})

# Release executables use an explicit hot/cold object composition. Numerical
# execution stays at -O3; generated dispatch, validation, package, shader and
# CLI control paths use the compiler's size-oriented optimization. The public
# shared/static libraries below deliberately keep their independent all-O3
# object composition because section GC and executable stripping must not
# narrow the embedding ABI.
if(CMAKE_C_COMPILER_ID MATCHES "Clang")
  set(VOLVOXAI_NATIVE_RELEASE_COLD_OPTIMIZATION -Oz)
elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
  set(VOLVOXAI_NATIVE_RELEASE_COLD_OPTIMIZATION -Os)
else()
  message(FATAL_ERROR
    "Native release hot/cold composition requires a GNU-compatible C compiler")
endif()
set(VOLVOXAI_NATIVE_RELEASE_HOT_OPTIMIZATION -O3)

function(volvox_assert_source_partition label all_var hot_var cold_var)
  set(_all ${${all_var}})
  set(_hot ${${hot_var}})
  set(_cold ${${cold_var}})
  set(_classified ${_hot} ${_cold})

  foreach(_name IN ITEMS _all _hot _cold _classified)
    list(LENGTH ${_name} _before)
    list(REMOVE_DUPLICATES ${_name})
    list(LENGTH ${_name} _after)
    if(NOT _before EQUAL _after)
      message(FATAL_ERROR
        "${label} has a duplicate source in ${_name}; every release source "
        "must have exactly one optimization owner")
    endif()
  endforeach()

  foreach(_source IN LISTS _all)
    list(FIND _hot "${_source}" _hot_index)
    list(FIND _cold "${_source}" _cold_index)
    if(_hot_index EQUAL -1 AND _cold_index EQUAL -1)
      message(FATAL_ERROR
        "${label} source '${_source}' has no hot/cold release classification")
    endif()
  endforeach()
  foreach(_source IN LISTS _classified)
    list(FIND _all "${_source}" _all_index)
    if(_all_index EQUAL -1)
      message(FATAL_ERROR
        "${label} classifies '${_source}' but the profile does not compile it")
    endif()
  endforeach()
endfunction()

set(INFERENCE_RELEASE_OBJECT_SRCS
  ${ENGINE_COMPILE_SRCS}
  ${INFERENCE_SHADER_SRCS})
set(INFERENCE_RELEASE_HOT_SRCS
  ${RUNTIME_HOT_RELEASE_SRCS}
  ${KERNEL_SRCS}
  ${BACKEND_CORE_SRCS})
set(INFERENCE_RELEASE_COLD_SRCS
  ${API_COMMON_SRCS}
  ${RUNTIME_COLD_RELEASE_SRCS}
  ${INFERENCE_API_GENERATED_SRCS}
  ${INFERENCE_SHADER_SRCS})
volvox_assert_source_partition(
  "inference release"
  INFERENCE_RELEASE_OBJECT_SRCS
  INFERENCE_RELEASE_HOT_SRCS
  INFERENCE_RELEASE_COLD_SRCS)

set(FULL_RELEASE_OBJECT_SRCS
  ${FULL_ENGINE_COMPILE_SRCS}
  ${FULL_PROFILE_SRCS}
  ${FULL_SHADER_SRCS})
set(FULL_RELEASE_HOT_SRCS
  ${RUNTIME_HOT_RELEASE_SRCS}
  ${KERNEL_SRCS}
  ${BACKEND_CORE_SRCS}
  ${BACKEND_DEVICE_HOT_RELEASE_SRCS}
  ${FULL_PROFILE_HOT_RELEASE_SRCS})
set(FULL_RELEASE_COLD_SRCS
  ${API_COMMON_SRCS}
  ${RUNTIME_COLD_RELEASE_SRCS}
  ${FULL_API_GENERATED_SRCS}
  ${BACKEND_DEVICE_COLD_RELEASE_SRCS}
  ${CUDA_TRAINING_PTX_SRCS}
  ${FULL_PROFILE_COLD_RELEASE_SRCS}
  ${FULL_SHADER_SRCS})
volvox_assert_source_partition(
  "full release"
  FULL_RELEASE_OBJECT_SRCS
  FULL_RELEASE_HOT_SRCS
  FULL_RELEASE_COLD_SRCS)

function(volvox_add_release_object_profile target profile temperature sources_var pack_target)
  if(profile STREQUAL "full")
    set(_training 1)
    set(_gen ${GEN}/full)
    set(_bk ${BK_FULL})
  elseif(profile STREQUAL "inference")
    set(_training 0)
    set(_gen ${GEN}/inference)
    set(_bk ${BK_LITE})
  else()
    message(FATAL_ERROR "Unknown native release object profile '${profile}'")
  endif()
  if(temperature STREQUAL "hot")
    set(_optimization ${VOLVOXAI_NATIVE_RELEASE_HOT_OPTIMIZATION})
  elseif(temperature STREQUAL "cold")
    set(_optimization ${VOLVOXAI_NATIVE_RELEASE_COLD_OPTIMIZATION})
  else()
    message(FATAL_ERROR "Unknown native release temperature '${temperature}'")
  endif()

  add_library(${target} OBJECT ${${sources_var}})
  add_dependencies(${target} ${pack_target})
  if(profile STREQUAL "full")
    volvox_cuda_codegen_before(${target})
    if(_training)
      volvox_cuda_training_codegen_before(${target})
    endif()
  endif()
  target_link_libraries(${target} PRIVATE volvox_common)
  target_compile_definitions(${target} PRIVATE
    VOLVOXAI_ENABLE_TRAINING=${_training} ${_bk})
  volvox_target_api_profile(${target} ${profile})
  target_include_directories(${target} BEFORE PRIVATE ${_gen})
  target_compile_options(${target} PRIVATE -std=c11 ${_optimization})
  if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE
      -ffunction-sections -fdata-sections)
    # Only Linux release finalization splits DWARF into a build-ID-matched
    # sidecar. Android and Apple publish the linked executable directly, so
    # adding -g there would turn a size optimization into a multi-megabyte
    # release regression.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      target_compile_options(${target} PRIVATE
        -g
        "-fdebug-prefix-map=${CMAKE_CURRENT_BINARY_DIR}=."
        "-fdebug-prefix-map=${REPO}=.")
    else()
      # Release objects remain configuration-independent under Xcode/Ninja
      # Multi-Config as well: a Debug configuration must not reintroduce DWARF
      # into a directly published Android or Apple executable.
      target_compile_options(${target} PRIVATE -g0)
    endif()
  endif()
endfunction()

# Preserve the ordered source authority while still compiling each source at
# its declared hot/cold optimization. A single hot target followed by a single
# cold target changes every function address even though it does not change an
# object byte. Split only when the temperature changes, then link the shards in
# their original order. The flattened shard inventory must remain byte-for-byte
# equal to the input list; source discovery is still only an audit above, never
# a build-order authority.
function(volvox_add_release_object_shards
    prefix profile ordered_var hot_var cold_var pack_target out_targets)
  set(_targets "")
  set(_flattened "")
  set(_shard_sources "")
  set(_shard_temperature "")
  set(_shard_index 0)

  foreach(_source IN LISTS ${ordered_var})
    list(FIND ${hot_var} "${_source}" _hot_index)
    list(FIND ${cold_var} "${_source}" _cold_index)
    if(NOT _hot_index EQUAL -1 AND NOT _cold_index EQUAL -1)
      message(FATAL_ERROR
        "Release shard source '${_source}' is both hot and cold")
    elseif(NOT _hot_index EQUAL -1)
      set(_temperature hot)
    elseif(NOT _cold_index EQUAL -1)
      set(_temperature cold)
    else()
      message(FATAL_ERROR
        "Release shard source '${_source}' has no hot/cold classification")
    endif()

    if(_shard_sources AND NOT _temperature STREQUAL _shard_temperature)
      set(_target
        "${prefix}_shard_${_shard_index}_${_shard_temperature}_objects")
      set(_sources_var "${_target}_sources")
      set(${_sources_var} ${_shard_sources})
      volvox_add_release_object_profile(
        ${_target} ${profile} ${_shard_temperature}
        ${_sources_var} ${pack_target})
      list(APPEND _targets ${_target})
      math(EXPR _shard_index "${_shard_index} + 1")
      set(_shard_sources "")
    endif()
    list(APPEND _shard_sources "${_source}")
    list(APPEND _flattened "${_source}")
    set(_shard_temperature ${_temperature})
  endforeach()

  if(_shard_sources)
    set(_target
      "${prefix}_shard_${_shard_index}_${_shard_temperature}_objects")
    set(_sources_var "${_target}_sources")
    set(${_sources_var} ${_shard_sources})
    volvox_add_release_object_profile(
      ${_target} ${profile} ${_shard_temperature}
      ${_sources_var} ${pack_target})
    list(APPEND _targets ${_target})
  endif()

  if(NOT "${_flattened}" STREQUAL "${${ordered_var}}")
    message(FATAL_ERROR
      "Release shards for ${prefix} changed the ordered source inventory")
  endif()
  set(${out_targets} ${_targets} PARENT_SCOPE)
endfunction()

volvox_add_release_object_shards(
  volvoxai_inference_release_engine inference
  ENGINE_COMPILE_SRCS INFERENCE_RELEASE_HOT_SRCS INFERENCE_RELEASE_COLD_SRCS
  volvoxai_pack_inference_shaders INFERENCE_RELEASE_ENGINE_SHARDS)
volvox_add_release_object_shards(
  volvoxai_inference_release_shader inference
  INFERENCE_SHADER_SRCS INFERENCE_RELEASE_HOT_SRCS INFERENCE_RELEASE_COLD_SRCS
  volvoxai_pack_inference_shaders INFERENCE_RELEASE_SHADER_SHARDS)
set(INFERENCE_RELEASE_CLI_SRCS ${CLI_SRCS})
volvox_add_release_object_profile(
  volvoxai_inference_release_cli_object inference cold
  INFERENCE_RELEASE_CLI_SRCS volvoxai_pack_inference_shaders)

volvox_add_release_object_shards(
  volvoxai_full_release_engine full
  FULL_ENGINE_COMPILE_SRCS FULL_RELEASE_HOT_SRCS FULL_RELEASE_COLD_SRCS
  volvoxai_pack_full_shaders FULL_RELEASE_ENGINE_SHARDS)
volvox_add_release_object_shards(
  volvoxai_full_release_profile full
  FULL_PROFILE_SRCS FULL_RELEASE_HOT_SRCS FULL_RELEASE_COLD_SRCS
  volvoxai_pack_full_shaders FULL_RELEASE_PROFILE_SHARDS)
volvox_add_release_object_shards(
  volvoxai_full_release_shader full
  FULL_SHADER_SRCS FULL_RELEASE_HOT_SRCS FULL_RELEASE_COLD_SRCS
  volvoxai_pack_full_shaders FULL_RELEASE_SHADER_SHARDS)
set(FULL_RELEASE_CLI_SRCS ${CLI_SRCS})
volvox_add_release_object_profile(
  volvoxai_full_release_cli_object full cold
  FULL_RELEASE_CLI_SRCS volvoxai_pack_full_shaders)

# The native W8A8 performance gate compares the production selective profile
# with a current-source all-O3 reference. These objects are build-private and
# excluded from ordinary builds: they exist only to isolate optimization and
# layout effects without publishing another release artifact.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT CMAKE_CROSSCOMPILING AND
   NOT VOLVOXAI_NATIVE_MULTI_CONFIG)
  set(VOLVOXAI_NATIVE_W8A8_BENCHMARK_SOURCE
    "${REPO}/examples/tiny_receipt_vqa/native/benchmark_w8a8_native.c")
  set(VOLVOXAI_NATIVE_W8A8_BASELINE_ENGINE_SRCS ${ENGINE_COMPILE_SRCS})
  set(VOLVOXAI_NATIVE_W8A8_BASELINE_SHADER_SRCS ${INFERENCE_SHADER_SRCS})
  set(VOLVOXAI_NATIVE_W8A8_CLI_ANCHOR_SRCS ${CLI_SRCS})
  volvox_add_release_object_profile(
    volvoxai_native_w8a8_baseline_engine_objects inference hot
    VOLVOXAI_NATIVE_W8A8_BASELINE_ENGINE_SRCS
    volvoxai_pack_inference_shaders)
  volvox_add_release_object_profile(
    volvoxai_native_w8a8_baseline_shader_objects inference hot
    VOLVOXAI_NATIVE_W8A8_BASELINE_SHADER_SRCS
    volvoxai_pack_inference_shaders)
  volvox_add_release_object_profile(
    volvoxai_native_w8a8_candidate_cli_anchor inference cold
    VOLVOXAI_NATIVE_W8A8_CLI_ANCHOR_SRCS
    volvoxai_pack_inference_shaders)
  volvox_add_release_object_profile(
    volvoxai_native_w8a8_baseline_cli_anchor inference hot
    VOLVOXAI_NATIVE_W8A8_CLI_ANCHOR_SRCS
    volvoxai_pack_inference_shaders)
  foreach(_target IN ITEMS
      volvoxai_native_w8a8_candidate_cli_anchor
      volvoxai_native_w8a8_baseline_cli_anchor)
    target_compile_definitions(
      ${_target} PRIVATE main=volvoxai_native_w8a8_release_cli_anchor)
  endforeach()
  set_target_properties(
    volvoxai_native_w8a8_baseline_engine_objects
    volvoxai_native_w8a8_baseline_shader_objects
    volvoxai_native_w8a8_candidate_cli_anchor
    volvoxai_native_w8a8_baseline_cli_anchor
    PROPERTIES EXCLUDE_FROM_ALL TRUE)

  add_library(volvoxai_native_w8a8_benchmark_main OBJECT EXCLUDE_FROM_ALL
    "${VOLVOXAI_NATIVE_W8A8_BENCHMARK_SOURCE}")
  target_link_libraries(volvoxai_native_w8a8_benchmark_main PRIVATE volvox_common)
  target_compile_definitions(volvoxai_native_w8a8_benchmark_main PRIVATE
    VOLVOXAI_ENABLE_TRAINING=0 ${BK_LITE})
  volvox_target_api_profile(volvoxai_native_w8a8_benchmark_main inference)
  target_include_directories(volvoxai_native_w8a8_benchmark_main BEFORE PRIVATE
    ${GEN}/inference)
  target_compile_options(volvoxai_native_w8a8_benchmark_main PRIVATE
    -std=c11 -O2 -g -Wall -Wextra -Werror)
  if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(volvoxai_native_w8a8_benchmark_main PRIVATE
      -ffunction-sections -fdata-sections
      "-fdebug-prefix-map=${CMAKE_CURRENT_BINARY_DIR}=."
      "-fdebug-prefix-map=${REPO}=.")
  endif()
endif()

# The ISA-specific Arm objects are shared by the inference and full release
# executables, but not by the embedding libraries. Keep the library objects
# unchanged and compile a release-only copy with the same O3 ISA contract plus
# the section/DWARF provenance required by executable GC and split debug.
set(ARM_RELEASE_OBJS "")
function(volvox_add_release_arm_object target sources_var)
  add_library(${target} OBJECT ${${sources_var}})
  target_link_libraries(${target} PRIVATE volvox_common)
  target_include_directories(${target} PRIVATE ${INC})
  target_compile_options(${target} PRIVATE
    -std=c11 ${VOLVOXAI_NATIVE_RELEASE_HOT_OPTIMIZATION}
    ${ARGN} -fno-lto)
  if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE
      -ffunction-sections -fdata-sections)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      target_compile_options(${target} PRIVATE
        -g
        "-fdebug-prefix-map=${CMAKE_CURRENT_BINARY_DIR}=."
        "-fdebug-prefix-map=${REPO}=.")
    else()
      target_compile_options(${target} PRIVATE -g0)
    endif()
  endif()
endfunction()

if(VOLVOX_ARM_DOTPROD)
  set(_arm_release_dotprod_sources
    src/kernels/qlinear_w8a8_arm_dotprod.c
    src/kernels/qconv_w8a8_arm_dotprod.c)
  volvox_add_release_arm_object(
    volvox_release_arm_dotprod _arm_release_dotprod_sources
    -march=armv8.2-a+dotprod)
  list(APPEND ARM_RELEASE_OBJS
    $<TARGET_OBJECTS:volvox_release_arm_dotprod>)
endif()
if(VOLVOX_ARM_I8MM AND _arm_i8mm_sources)
  volvox_add_release_arm_object(
    volvox_release_arm_i8mm _arm_i8mm_sources
    -march=armv8.2-a+i8mm -ffp-contract=off)
  list(APPEND ARM_RELEASE_OBJS
    $<TARGET_OBJECTS:volvox_release_arm_i8mm>)
endif()

# Portable graph/shape legality is a standalone C authority. Keep it
# independently buildable so authoring and runtime integrations share it.
add_library(volvoxai_semantics STATIC
  src/generated/operator_vocabulary.c
  src/generated/kernel_registry.c
  src/runtime/shape_contract.c
  src/runtime/shape_domain_contract.c
  src/runtime/graph_plan.c
  src/runtime/activation_plan.c
  src/runtime/graph_bind.c
  src/runtime/graph_domain.c
  src/runtime/independent_batch_proof.c
  src/runtime/call_sequence_policy.c)
target_link_libraries(volvoxai_semantics PRIVATE volvox_common)
target_compile_definitions(volvoxai_semantics PRIVATE
  VOLVOXAI_ENABLE_TRAINING=0)
target_compile_options(volvoxai_semantics PRIVATE -std=c11 -O2)

# --- Distributable libraries --------------------------------------------------
# The engine is a library first. proto/volvoxai.proto is its whole public API,
# and Synurang_GetApi is the entry point every module host loads — Python, Go, Rust, Java, Dart, Swift,
# C#. Shipping only executables put that ABI out of reach of all of them.
#
# One profile pair, two link forms each:
#
#   libvolvoxai-lite    inference   Platform, Inference, Scheduler
#   libvolvoxai         full        + Training, Quantization
#
# The shared and static forms are built from the same objects so they cannot
# disagree about what the engine does.
#
# The generated dispatch has no visibility attribute of its own. Give it
# default visibility and let the profile's version script decide what leaves
# the shared library.
set_source_files_properties(
  ${REPO}/runtime/generated/c/inference/volvoxai_lite.c
  ${REPO}/runtime/generated/c/inference/volvoxai_ffi.c
  ${REPO}/runtime/generated/c/volvoxai_lite.c
  ${REPO}/runtime/generated/c/volvoxai_ffi.c
  PROPERTIES COMPILE_OPTIONS "-fvisibility=default")

set(VOLVOXAI_INFERENCE_VERSION_SCRIPT
  ${CMAKE_CURRENT_SOURCE_DIR}/volvoxai.map)
set(VOLVOXAI_FULL_VERSION_SCRIPT
  ${CMAKE_CURRENT_SOURCE_DIR}/volvoxai.map)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  if(NOT CMAKE_OBJCOPY)
    find_program(CMAKE_OBJCOPY objcopy)
  endif()
  if(NOT CMAKE_STRIP)
    find_program(CMAKE_STRIP strip)
  endif()
endif()

# profile: inference | full
function(volvox_add_library profile)
  if(profile STREQUAL "full")
    set(_name volvoxai)
    set(_srcs ${FULL_ENGINE_COMPILE_SRCS} ${FULL_PROFILE_SRCS} ${FULL_SHADER_SRCS})
    set(_training 1)
    set(_gen ${GEN}/full)
    set(_pack volvoxai_pack_full_shaders)
    set(_version_script ${VOLVOXAI_FULL_VERSION_SCRIPT})
    set(_bk ${BK_FULL})

    add_library(${_name}_objects OBJECT ${_srcs})
    add_dependencies(${_name}_objects ${_pack})
    volvox_cuda_codegen_before(${_name}_objects)
    if(_training)
      volvox_cuda_training_codegen_before(${_name}_objects)
    endif()
    target_link_libraries(${_name}_objects PRIVATE volvox_common)
    target_compile_definitions(${_name}_objects PRIVATE
      VOLVOXAI_ENABLE_TRAINING=${_training} ${_bk})
    volvox_target_api_profile(${_name}_objects ${profile})
    target_include_directories(${_name}_objects BEFORE PRIVATE ${_gen})
    target_compile_options(${_name}_objects PRIVATE -std=c11 -O3)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      target_compile_options(${_name}_objects PRIVATE
        -g
        "-fdebug-prefix-map=${CMAKE_CURRENT_BINARY_DIR}=."
        "-fdebug-prefix-map=${REPO}=.")
    endif()
    set_target_properties(${_name}_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set(_lib_objs $<TARGET_OBJECTS:${_name}_objects>)
  else()
    set(_name volvoxai-lite)
    set(_training 0)
    set(_gen ${GEN}/inference)
    set(_pack volvoxai_pack_inference_shaders)
    set(_version_script ${VOLVOXAI_INFERENCE_VERSION_SCRIPT})
    set(_bk ${BK_LITE})

    # Hot/cold optimization split: Hot paths at -O3, Cold paths at -Oz/-Os for size optimization
    add_library(${_name}_hot_objects OBJECT ${INFERENCE_RELEASE_HOT_SRCS})
    add_dependencies(${_name}_hot_objects ${_pack})
    target_link_libraries(${_name}_hot_objects PRIVATE volvox_common)
    target_compile_definitions(${_name}_hot_objects PRIVATE
      VOLVOXAI_ENABLE_TRAINING=${_training} ${_bk})
    volvox_target_api_profile(${_name}_hot_objects ${profile})
    target_include_directories(${_name}_hot_objects BEFORE PRIVATE ${_gen})
    target_compile_options(${_name}_hot_objects PRIVATE
      -std=c11 ${VOLVOXAI_NATIVE_RELEASE_HOT_OPTIMIZATION})
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      target_compile_options(${_name}_hot_objects PRIVATE
        -g
        "-fdebug-prefix-map=${CMAKE_CURRENT_BINARY_DIR}=."
        "-fdebug-prefix-map=${REPO}=.")
    endif()
    set_target_properties(${_name}_hot_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)

    add_library(${_name}_cold_objects OBJECT ${INFERENCE_RELEASE_COLD_SRCS})
    add_dependencies(${_name}_cold_objects ${_pack})
    target_link_libraries(${_name}_cold_objects PRIVATE volvox_common)
    target_compile_definitions(${_name}_cold_objects PRIVATE
      VOLVOXAI_ENABLE_TRAINING=${_training} ${_bk})
    volvox_target_api_profile(${_name}_cold_objects ${profile})
    target_include_directories(${_name}_cold_objects BEFORE PRIVATE ${_gen})
    target_compile_options(${_name}_cold_objects PRIVATE
      -std=c11 ${VOLVOXAI_NATIVE_RELEASE_COLD_OPTIMIZATION})
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      target_compile_options(${_name}_cold_objects PRIVATE
        -g
        "-fdebug-prefix-map=${CMAKE_CURRENT_BINARY_DIR}=."
        "-fdebug-prefix-map=${REPO}=.")
    endif()
    set_target_properties(${_name}_cold_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)

    set(_lib_objs
      $<TARGET_OBJECTS:${_name}_hot_objects>
      $<TARGET_OBJECTS:${_name}_cold_objects>)
  endif()

  foreach(form shared static)
    if(form STREQUAL "shared")
      add_library(${_name}_shared SHARED
        ${_lib_objs} ${ARM_OBJS})
      set(_target ${_name}_shared)
    else()
      add_library(${_name}_static STATIC
        ${_lib_objs} ${ARM_OBJS})
      set(_target ${_name}_static)
    endif()
    target_link_libraries(${_target} PRIVATE volvox_common)
    set_target_properties(${_target} PROPERTIES
      OUTPUT_NAME ${_name}
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  endforeach()

  set_target_properties(${_name}_shared PROPERTIES
    VERSION ${VOLVOXAI_VERSION} SOVERSION 1)
  # A plugin host resolves symbols by name, so an internal helper that leaks
  # into the dynamic table becomes an accidental contract. Restrict the export
  # list where the platform supports it; elsewhere -fvisibility=hidden is the
  # only fence and the .so carries a wider table.
  if(NOT APPLE AND NOT MSVC)
    target_link_options(${_name}_shared PRIVATE
      "-Wl,--version-script=${_version_script}"
      "-Wl,--build-id=sha1")
    set_target_properties(${_name}_shared PROPERTIES
      LINK_DEPENDS ${_version_script})
  endif()

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_OBJCOPY AND CMAKE_STRIP)
    set(_debug_dir "${CMAKE_CURRENT_SOURCE_DIR}/.debug")
    set(_debug_file "${_debug_dir}/lib${_name}.so.${VOLVOXAI_VERSION}.debug")
    add_custom_command(TARGET ${_name}_shared POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory "${_debug_dir}"
      COMMAND ${CMAKE_OBJCOPY} --only-keep-debug "$<TARGET_FILE:${_name}_shared>" "${_debug_file}"
      COMMAND ${CMAKE_STRIP} --strip-unneeded "$<TARGET_FILE:${_name}_shared>"
      COMMAND ${CMAKE_OBJCOPY} "--add-gnu-debuglink=${_debug_file}" "$<TARGET_FILE:${_name}_shared>"
      COMMAND ${CMAKE_COMMAND} -E create_symlink
              "lib${_name}.so.${VOLVOXAI_VERSION}.debug"
              "${_debug_dir}/lib${_name}.so.debug"
      COMMAND ${CMAKE_COMMAND} -E create_symlink
              "lib${_name}.so.${VOLVOXAI_VERSION}.debug"
              "${_debug_dir}/lib${_name}.so.1.debug"
      BYPRODUCTS "${_debug_file}"
      COMMENT "Splitting debug symbols for lib${_name}.so.${VOLVOXAI_VERSION}"
      VERBATIM)
  endif()
endfunction()

volvox_add_library(inference)
volvox_add_library(full)

# Source-build Python clients load engine modules through this small upstream
# host shim. It is a development library, outside the fixed release inventory.
find_package(Threads REQUIRED)
add_library(synurang_module_host SHARED EXCLUDE_FROM_ALL
  ${VOLVOXAI_SYNURANG_DIR}/src/module_host.c)
target_include_directories(synurang_module_host PRIVATE
  ${VOLVOXAI_SYNURANG_DIR}/include)
target_compile_definitions(synurang_module_host PRIVATE
  SYNURANG_C_RUNTIME_SHARED SYNURANG_C_RUNTIME_BUILDING)
target_link_libraries(synurang_module_host PRIVATE
  Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(synurang_module_host PROPERTIES
  C_STANDARD 11 C_STANDARD_REQUIRED ON
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_dependencies(volvoxai-lite_shared synurang_module_host)
add_dependencies(volvoxai_shared synurang_module_host)

# Proves the shipped .so is loadable and complete: dlopen it the way a plugin
# host does, resolve Synurang_GetApi by name, and confirm no engine lifecycle
# or retired dispatch symbol came along. Links nothing from
# the engine itself, so a missing export fails here rather than in a host.
add_executable(test_shared_library_abi tests/test_shared_library_abi.c)
target_link_libraries(test_shared_library_abi PRIVATE ${CMAKE_DL_LIBS})
target_include_directories(test_shared_library_abi PRIVATE
  ${REPO}/runtime/generated/c ${VOLVOXAI_SYNURANG_DIR}/include)
target_compile_options(test_shared_library_abi PRIVATE
  -std=c11 -O2 -Wall -Wextra -Werror)
add_dependencies(test_shared_library_abi
  volvoxai-lite_shared volvoxai-lite_static
  volvoxai_shared volvoxai_static)
add_test(NAME test_shared_library_abi
  COMMAND $<TARGET_FILE:test_shared_library_abi>
          $<TARGET_FILE:volvoxai-lite_shared> inference)
# Same ABI, wider handler table. Both profiles ship, so both are checked.
add_test(NAME test_shared_library_abi_full
  COMMAND $<TARGET_FILE:test_shared_library_abi>
          $<TARGET_FILE:volvoxai_shared> full)
add_test(NAME check_native_library_profile_boundaries
  COMMAND ${PY}
          ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_library_profile_boundaries.py
          --system-name ${CMAKE_SYSTEM_NAME}
          --nm ${CMAKE_NM}
          --readelf "${CMAKE_READELF}"
          --synurang-header ${VOLVOXAI_SYNURANG_RUNTIME_HEADER}
          --inference-shared $<TARGET_FILE:volvoxai-lite_shared>
          --inference-static $<TARGET_FILE:volvoxai-lite_static>
          --full-shared $<TARGET_FILE:volvoxai_shared>
          --full-static $<TARGET_FILE:volvoxai_static>)

add_executable(test_static_library_abi_inference
  tests/test_static_library_abi.c
  ${REPO}/examples/backend_sdk/host_backend.c)
target_link_libraries(test_static_library_abi_inference PRIVATE volvoxai-lite_static)
target_include_directories(test_static_library_abi_inference PRIVATE
  ${REPO}/examples/backend_sdk
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${REPO}/runtime/generated/c/inference
  ${REPO}/runtime/generated/c
  ${VOLVOXAI_SYNURANG_DIR}/include)
target_compile_options(test_static_library_abi_inference PRIVATE
  -std=c11 -O2 -Wall -Wextra -Werror)
add_test(NAME test_static_library_abi_inference
  COMMAND $<TARGET_FILE:test_static_library_abi_inference>
          ${REPO}/examples/backend_sdk/example.graph.json)

add_executable(test_static_library_abi_full
  tests/test_static_library_abi.c
  ${REPO}/examples/backend_sdk/host_backend.c)
target_link_libraries(test_static_library_abi_full PRIVATE volvoxai_static)
target_include_directories(test_static_library_abi_full PRIVATE
  ${REPO}/examples/backend_sdk
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${REPO}/runtime/generated/c
  ${VOLVOXAI_SYNURANG_DIR}/include)
target_compile_options(test_static_library_abi_full PRIVATE
  -std=c11 -O2 -Wall -Wextra -Werror)
add_test(NAME test_static_library_abi_full
  COMMAND $<TARGET_FILE:test_static_library_abi_full>
          ${REPO}/examples/backend_sdk/example.graph.json)

add_executable(test_api_handle_registry
  tests/test_api_handle_registry.c)
target_link_libraries(test_api_handle_registry PRIVATE volvoxai_static)
target_include_directories(test_api_handle_registry PRIVATE src/api)
target_compile_options(test_api_handle_registry PRIVATE
  -std=c11 -O2 -Wall -Wextra -Werror)
add_test(NAME test_api_handle_registry
  COMMAND $<TARGET_FILE:test_api_handle_registry>)

# Exercise trace ownership, bounded collection and the CUDA Driver adapter.
add_executable(test_profiling tests/test_profiling.c src/runtime/profiling.c)
target_include_directories(test_profiling PRIVATE ${INC})
target_compile_definitions(test_profiling PRIVATE VOLVOXAI_ENABLE_TRAINING=0)
target_compile_options(test_profiling PRIVATE -std=c11 -O2 -UNDEBUG)
target_link_libraries(test_profiling PRIVATE Threads::Threads)
add_test(NAME test_profiling COMMAND $<TARGET_FILE:test_profiling>)
set_tests_properties(test_profiling PROPERTIES LABELS "native;native-invariant" TIMEOUT 60)

add_executable(test_cuda_trace tests/test_cuda_trace.c)
target_include_directories(test_cuda_trace PRIVATE ${INC})
target_compile_definitions(test_cuda_trace PRIVATE VOLVOXAI_ENABLE_TRAINING=0)
target_compile_options(test_cuda_trace PRIVATE -std=c11 -O2 -UNDEBUG)
add_test(NAME test_cuda_trace COMMAND $<TARGET_FILE:test_cuda_trace>)
set_tests_properties(test_cuda_trace PROPERTIES LABELS "native;native-invariant" TIMEOUT 60)

# Exercise pending GPU storage reuse without requiring a physical CUDA device.
add_executable(test_cuda_tensor_streams
  tests/test_cuda_tensor_streams.c src/runtime/native_tensor.c src/runtime/profiling.c)
target_include_directories(test_cuda_tensor_streams PRIVATE ${INC})
target_compile_definitions(test_cuda_tensor_streams PRIVATE
  VOLVOXAI_ENABLE_CUDA=1 VOLVOXAI_ENABLE_VULKAN=0 VOLVOXAI_ENABLE_OPENGL=0
  VOLVOXAI_ENABLE_METAL=0 VOLVOXAI_ENABLE_TRAINING=0)
target_compile_options(test_cuda_tensor_streams PRIVATE -std=c11 -O2 -UNDEBUG)
target_link_libraries(test_cuda_tensor_streams PRIVATE Threads::Threads)
add_test(NAME test_cuda_tensor_streams COMMAND $<TARGET_FILE:test_cuda_tensor_streams>)
set_tests_properties(test_cuda_tensor_streams PROPERTIES LABELS "native;native-invariant" TIMEOUT 60)

# Exercise retained native buffers through the generated C dispatch in both profiles.
foreach(tensor_profile IN ITEMS inference full)
  set(tensor_target test_native_tensor_client_${tensor_profile})
  add_executable(${tensor_target} tests/test_native_tensor_client.c)
  if(tensor_profile STREQUAL "full")
    target_link_libraries(${tensor_target} PRIVATE volvoxai_static volvox_common)
  else()
    target_link_libraries(${tensor_target} PRIVATE volvoxai-lite_static volvox_common)
  endif()
  volvox_target_api_profile(${tensor_target} ${tensor_profile})
  target_compile_options(${tensor_target} PRIVATE -std=c11 -O2)
  add_test(NAME ${tensor_target} COMMAND $<TARGET_FILE:${tensor_target}> cpu)
  set_tests_properties(${tensor_target} PROPERTIES LABELS "native;native-invariant" TIMEOUT 60)
endforeach()

# Exercise the real public_api.c direct-call lifetime logic with the same
# no-op synchronization primitives as wasm32.  The fixture includes that one
# production translation unit so private scope state remains private to
# releases; libvolvoxai supplies its ordinary external dependencies.
add_executable(test_threadless_runtime_release
  tests/test_threadless_runtime_release.c)
target_link_libraries(test_threadless_runtime_release PRIVATE
  volvoxai_static volvox_common)
target_compile_definitions(test_threadless_runtime_release PRIVATE
  VOLVOXAI_NO_THREADS=1 VOLVOXAI_ENABLE_TRAINING=0 ${BK_DEFAULT})
volvox_target_api_profile(test_threadless_runtime_release inference)
target_include_directories(test_threadless_runtime_release BEFORE PRIVATE
  ${GEN}/inference)
target_compile_options(test_threadless_runtime_release PRIVATE
  -std=c11 -O2 -ffunction-sections -fdata-sections)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(test_threadless_runtime_release PRIVATE
    -Wl,--gc-sections)
endif()
add_test(NAME test_threadless_runtime_release
  COMMAND $<TARGET_FILE:test_threadless_runtime_release>)

set_tests_properties(
  test_shared_library_abi test_shared_library_abi_full
  check_native_library_profile_boundaries
  test_static_library_abi_inference test_static_library_abi_full
  test_api_handle_registry test_threadless_runtime_release
  PROPERTIES LABELS "native;native-invariant" TIMEOUT 60)

# --- Release binaries ---------------------------------------------------------
# Fixed release names and CLI/profile checks require these binaries in native/.
# They are command-line applications, not libraries. Public generated ABI
# completeness is checked on libvolvoxai[-full].so/.a above; do not preserve
# the private vx_* lifecycle as an executable symbol contract.

# Size reports need object attribution that cannot be reconstructed reliably
# from the final ELF alone. GNU-compatible Linux linkers can emit a map without
# changing the release artifact or its public ABI. Other platforms still get
# section and symbol reports from their native object format.
set(VOLVOXAI_SIZE_MAP_DIR "${CMAKE_BINARY_DIR}/size-maps")
file(MAKE_DIRECTORY "${VOLVOXAI_SIZE_MAP_DIR}")
set(VOLVOXAI_NATIVE_RELEASE_LINK_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/release-link")
set(VOLVOXAI_NATIVE_RELEASE_STATE_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/release-state")
set(VOLVOXAI_NATIVE_PUBLISHED_DEBUG_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/.debug")
file(MAKE_DIRECTORY
  "${VOLVOXAI_NATIVE_RELEASE_LINK_DIR}"
  "${VOLVOXAI_NATIVE_RELEASE_STATE_DIR}")

function(volvox_release_section_gc target)
  get_target_property(_target_type ${target} TYPE)
  if(NOT _target_type STREQUAL "EXECUTABLE")
    message(FATAL_ERROR "Native section GC is executable-only: ${target}")
  endif()
  if(APPLE)
    target_link_options(${target} PRIVATE -Wl,-dead_strip)
  elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(${target} PRIVATE -Wl,--gc-sections)
  endif()
endfunction()

function(volvox_release_link_map target filename)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set(_map "${VOLVOXAI_SIZE_MAP_DIR}/${filename}")
    target_link_options(${target} PRIVATE
      "-Wl,-Map,${_map}")
    set_property(TARGET ${target} PROPERTY VOLVOXAI_RELEASE_LINK_MAP "${_map}")
  endif()
endfunction()

# Linux release executables carry only runtime-visible symbols. Preserve the
# complete optimized symbol/DWARF image beside them as a CI artifact and attach
# a GNU debuglink so standard symbolizers can resolve it. Shared/static public
# libraries are intentionally excluded from this executable packaging step.
function(volvox_release_artifact target filename)
  set(_build_artifact "${VOLVOXAI_NATIVE_RELEASE_LINK_DIR}/${filename}")
  set(_published_artifact "${CMAKE_CURRENT_SOURCE_DIR}/${filename}")
  set(_stamp "${VOLVOXAI_NATIVE_RELEASE_STATE_DIR}/${filename}.link.stamp")
  if(NOT EXISTS "${_stamp}")
    file(TOUCH "${_stamp}")
  endif()
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${VOLVOXAI_NATIVE_RELEASE_LINK_DIR}"
    OUTPUT_NAME "${filename}")

  set(_guard_target "${target}_release_output_guard")
  set(_guard_script "${REPO}/tools/ensure_native_release_outputs.py")
  set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS
    "${_stamp}" "${_guard_script}")

  # The private artifact, linker map, debug sidecar, evidence and recovery
  # stamp intentionally have one unambiguous path per build tree. A
  # multi-config generator would let Debug/Release overwrite that state and
  # then accept another configuration's newer bundle as a no-op. Keep normal
  # library/test targets usable under Xcode and Ninja Multi-Config, but fail a
  # release executable before it can link or publish mixed evidence.
  set(_configuration_guard "")
  if(VOLVOXAI_NATIVE_MULTI_CONFIG)
    set(_configuration_guard "${target}_release_single_config_guard")
    add_custom_target(${_configuration_guard}
      COMMAND ${CMAKE_COMMAND} -E echo
              "Native release evidence requires a single-config generator"
      COMMAND ${CMAKE_COMMAND} -E false
      COMMENT "Rejecting ambiguous multi-config ${filename} release build"
      VERBATIM)
  endif()

  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_custom_target(${_guard_target}
      COMMAND ${PY} "${_guard_script}"
              --artifact "${_build_artifact}"
              --published-artifact "${_published_artifact}"
              --stamp "${_stamp}"
      COMMENT "Checking recoverable ${filename} release outputs"
      VERBATIM)
    if(_configuration_guard)
      add_dependencies(${_guard_target} ${_configuration_guard})
    endif()
    add_dependencies(${target} ${_guard_target})
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "$<TARGET_FILE:${target}>" "${_published_artifact}"
      COMMENT "Publishing build-private ${filename} release artifact"
      VERBATIM)
    return()
  endif()
  if(NOT CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR
      "Linux native release debug packaging requires a GNU-compatible compiler")
  endif()
  if(NOT CMAKE_OBJCOPY OR NOT CMAKE_STRIP)
    message(FATAL_ERROR
      "Linux native release debug packaging requires objcopy and strip")
  endif()
  get_target_property(_target_type ${target} TYPE)
  if(NOT _target_type STREQUAL "EXECUTABLE")
    message(FATAL_ERROR
      "Native release debug packaging is executable-only: ${target}")
  endif()
  if(NOT CMAKE_READELF OR NOT CMAKE_NM)
    message(FATAL_ERROR
      "Linux native release debug packaging requires readelf and nm")
  endif()
  string(REGEX MATCH "^[0-9]+" _compiler_major "${CMAKE_C_COMPILER_VERSION}")
  find_program(_volvox_addr2line
    NAMES
      llvm-addr2line-${_compiler_major}
      llvm-addr2line-17 llvm-addr2line-16 llvm-addr2line-15
      llvm-addr2line-14 llvm-addr2line)
  if(NOT _volvox_addr2line)
    message(FATAL_ERROR
      "Linux native release debug packaging requires LLVM addr2line")
  endif()
  get_target_property(_map_file ${target} VOLVOXAI_RELEASE_LINK_MAP)
  if(NOT _map_file OR _map_file MATCHES "-NOTFOUND$")
    message(FATAL_ERROR
      "Linux native release ${target} has no build-private linker map")
  endif()
  set(_debug_file
    "${VOLVOXAI_NATIVE_RELEASE_LINK_DIR}/.debug/${filename}.debug")
  set(_evidence_file
    "${VOLVOXAI_NATIVE_RELEASE_LINK_DIR}/.debug/${filename}.debug.json")
  set(_published_debug
    "${VOLVOXAI_NATIVE_PUBLISHED_DEBUG_DIR}/${filename}.debug")
  set(_published_evidence
    "${VOLVOXAI_NATIVE_PUBLISHED_DEBUG_DIR}/${filename}.debug.json")
  file(RELATIVE_PATH _build_directory_relative "${REPO}" "${CMAKE_BINARY_DIR}")
  file(TO_CMAKE_PATH "${_build_directory_relative}" _build_directory_relative)
  set(_run_version_arg "")
  if(NOT CMAKE_CROSSCOMPILING)
    set(_run_version_arg --run-version)
  endif()
  add_custom_target(${_guard_target}
    COMMAND ${PY} "${_guard_script}"
            --artifact "${_build_artifact}"
            --debug "${_debug_file}"
            --evidence "${_evidence_file}"
            --map "${_map_file}"
            --published-artifact "${_published_artifact}"
            --published-debug "${_published_debug}"
            --published-evidence "${_published_evidence}"
            --build-directory "${_build_directory_relative}"
            --repository-root "${REPO}"
            --stamp "${_stamp}"
    COMMENT "Checking recoverable ${filename} release bundle"
    VERBATIM)
  if(_configuration_guard)
    add_dependencies(${_guard_target} ${_configuration_guard})
  endif()
  add_dependencies(${target} ${_guard_target})
  target_link_options(${target} PRIVATE -Wl,--build-id=sha1)
  set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS
    "${REPO}/tools/finalize_native_release.py")
  get_target_property(_release_link_objects ${target} SOURCES)
  if(NOT _release_link_objects)
    message(FATAL_ERROR "Native release ${target} has no link object graph")
  endif()
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${PY} "${REPO}/tools/finalize_native_release.py"
            --artifact "$<TARGET_FILE:${target}>"
            --debug "${_debug_file}"
            --evidence "${_evidence_file}"
            --build-directory "${CMAKE_BINARY_DIR}"
            --link-map "${_map_file}"
            --link-objects ${_release_link_objects}
            --logical-artifact "native/${filename}"
            --logical-debug "native/.debug/${filename}.debug"
            --publish-artifact "${_published_artifact}"
            --publish-debug "${_published_debug}"
            --publish-evidence "${_published_evidence}"
            --repository-root "${REPO}"
            --expected-version "${VOLVOXAI_VERSION}"
            --objcopy "${CMAKE_OBJCOPY}"
            --strip "${CMAKE_STRIP}"
            --readelf "${CMAKE_READELF}"
            --nm "${CMAKE_NM}"
            --addr2line "${_volvox_addr2line}"
            ${_run_version_arg}
    BYPRODUCTS "${_debug_file}" "${_evidence_file}"
    COMMENT "Finalizing and publishing verified ${filename} release bundle"
    COMMAND_EXPAND_LISTS
    VERBATIM)
endfunction()

# Exercises the public protobuf API through the generated C dispatch, so a
# break in the handler tables, converters or schema fails here rather than
# only in a downstream host.
# The roundtrip drives VxPlanningService to author its fixture, so it builds on
# the full profile where that surface lives. Inference dispatch stays covered by
# test_shared_library_abi and the profile boundary checks.
add_executable(test_proto_api_roundtrip
  ${FULL_ENGINE_SRCS} ${FULL_PROFILE_SRCS} ${FULL_SHADER_SRCS}
  tests/test_proto_api_roundtrip.c)
add_dependencies(test_proto_api_roundtrip volvoxai_pack_full_shaders)
volvox_cuda_codegen_before(test_proto_api_roundtrip)
volvox_cuda_training_codegen_before(test_proto_api_roundtrip)
target_link_libraries(test_proto_api_roundtrip PRIVATE volvox_common)
target_compile_definitions(test_proto_api_roundtrip PRIVATE
  VOLVOXAI_ENABLE_TRAINING=1 VOLVOXAI_TEST_IDENTITY_LAYOUT=1 ${BK_FULL})
volvox_target_api_profile(test_proto_api_roundtrip full)
target_include_directories(test_proto_api_roundtrip PRIVATE ${GEN}/full)
target_compile_options(test_proto_api_roundtrip PRIVATE -std=c11 -O2)
add_test(NAME test_proto_api_roundtrip
  COMMAND $<TARGET_FILE:test_proto_api_roundtrip>
          ${REPO}/models/tinystories_1m)

set(INFERENCE_RELEASE_RUNTIME_LINK_OBJECTS "")
foreach(_target IN LISTS INFERENCE_RELEASE_ENGINE_SHARDS)
  list(APPEND INFERENCE_RELEASE_RUNTIME_LINK_OBJECTS
    $<TARGET_OBJECTS:${_target}>)
endforeach()
list(APPEND INFERENCE_RELEASE_RUNTIME_LINK_OBJECTS ${ARM_RELEASE_OBJS})
foreach(_target IN LISTS INFERENCE_RELEASE_SHADER_SHARDS)
  list(APPEND INFERENCE_RELEASE_RUNTIME_LINK_OBJECTS
    $<TARGET_OBJECTS:${_target}>)
endforeach()
set(INFERENCE_RELEASE_LINK_OBJECTS ${INFERENCE_RELEASE_RUNTIME_LINK_OBJECTS})
list(APPEND INFERENCE_RELEASE_LINK_OBJECTS
  $<TARGET_OBJECTS:volvoxai_inference_release_cli_object>)
add_executable(volvoxai-lite ${INFERENCE_RELEASE_LINK_OBJECTS})
set_target_properties(volvoxai-lite PROPERTIES LINKER_LANGUAGE C)
target_link_libraries(volvoxai-lite PRIVATE volvox_common)
volvox_release_section_gc(volvoxai-lite)
volvox_release_link_map(volvoxai-lite volvoxai-lite.map)
# Android is inference-only and uses a distinct output name so cross-builds do
# not clobber the host binary.
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  volvox_release_artifact(volvoxai-lite volvoxai_android)
else()
  volvox_release_artifact(volvoxai-lite volvoxai-lite)
endif()

set(FULL_RELEASE_LINK_OBJECTS "")
foreach(_target IN LISTS FULL_RELEASE_ENGINE_SHARDS)
  list(APPEND FULL_RELEASE_LINK_OBJECTS $<TARGET_OBJECTS:${_target}>)
endforeach()
list(APPEND FULL_RELEASE_LINK_OBJECTS ${ARM_RELEASE_OBJS})
foreach(_target IN LISTS FULL_RELEASE_PROFILE_SHARDS FULL_RELEASE_SHADER_SHARDS)
  list(APPEND FULL_RELEASE_LINK_OBJECTS $<TARGET_OBJECTS:${_target}>)
endforeach()
list(APPEND FULL_RELEASE_LINK_OBJECTS
  $<TARGET_OBJECTS:volvoxai_full_release_cli_object>)
add_executable(volvoxai ${FULL_RELEASE_LINK_OBJECTS})
set_target_properties(volvoxai PROPERTIES LINKER_LANGUAGE C)
target_link_libraries(volvoxai PRIVATE volvox_common)
volvox_release_section_gc(volvoxai)
# The portable training-control state machine is an intentional full-profile
# authority even though the current native CLI reaches training through the
# generated handler implementation. Preserve the pre-GC physical contract;
# the inference executable must continue to contain none of these symbols.
foreach(_symbol IN ITEMS
    vx_training_control_core_abi_version
    vx_training_control_state_init_v1
    vx_training_control_state_validate_v1
    vx_training_control_step_begin_v1
    vx_training_control_step_finish_v1
    vx_training_control_prepare_commit_v1
    vx_training_control_prepare_rollback_v1
    vx_training_control_prepare_reset_accumulation_v1)
  if(APPLE)
    target_link_options(volvoxai PRIVATE "-Wl,-u,_${_symbol}")
  elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(volvoxai PRIVATE "-Wl,-u,${_symbol}")
  endif()
endforeach()
volvox_release_link_map(volvoxai volvoxai.map)
volvox_release_artifact(volvoxai volvoxai)

# Performance executables deliberately bypass release maps, finalization and
# publication. Each graph roots a renamed copy of the production CLI after its
# shader objects so release API/control reachability and hot symbol layout are
# retained before the shared benchmark main. Candidate uses the production
# cold CLI profile; baseline keeps its engine, shader and CLI reference at O3.
# The runner verifies engine -> Arm -> shader -> CLI anchor -> benchmark-main
# order, hot symbol addresses and the immutable release bundle before either.
if(TARGET volvoxai_native_w8a8_benchmark_main)
  set(VOLVOXAI_NATIVE_W8A8_PERFORMANCE_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/performance")

  set(_volvox_w8a8_candidate_objects
    ${INFERENCE_RELEASE_RUNTIME_LINK_OBJECTS}
    $<TARGET_OBJECTS:volvoxai_native_w8a8_candidate_cli_anchor>
    $<TARGET_OBJECTS:volvoxai_native_w8a8_benchmark_main>)
  add_executable(volvoxai_native_w8a8_candidate EXCLUDE_FROM_ALL
    ${_volvox_w8a8_candidate_objects})
  set_target_properties(volvoxai_native_w8a8_candidate PROPERTIES
    LINKER_LANGUAGE C
    RUNTIME_OUTPUT_DIRECTORY "${VOLVOXAI_NATIVE_W8A8_PERFORMANCE_DIR}"
    OUTPUT_NAME volvoxai-native-w8a8-candidate)
  target_link_libraries(volvoxai_native_w8a8_candidate PRIVATE volvox_common)
  volvox_release_section_gc(volvoxai_native_w8a8_candidate)
  target_link_options(volvoxai_native_w8a8_candidate PRIVATE
    -Wl,--build-id=sha1
    -Wl,-u,volvoxai_native_w8a8_release_cli_anchor)

  set(_volvox_w8a8_baseline_objects
    $<TARGET_OBJECTS:volvoxai_native_w8a8_baseline_engine_objects>
    ${ARM_RELEASE_OBJS}
    $<TARGET_OBJECTS:volvoxai_native_w8a8_baseline_shader_objects>
    $<TARGET_OBJECTS:volvoxai_native_w8a8_baseline_cli_anchor>
    $<TARGET_OBJECTS:volvoxai_native_w8a8_benchmark_main>)
  add_executable(volvoxai_native_w8a8_baseline EXCLUDE_FROM_ALL
    ${_volvox_w8a8_baseline_objects})
  set_target_properties(volvoxai_native_w8a8_baseline PROPERTIES
    LINKER_LANGUAGE C
    RUNTIME_OUTPUT_DIRECTORY "${VOLVOXAI_NATIVE_W8A8_PERFORMANCE_DIR}"
    OUTPUT_NAME volvoxai-native-w8a8-baseline)
  target_link_libraries(volvoxai_native_w8a8_baseline PRIVATE volvox_common)
  volvox_release_section_gc(volvoxai_native_w8a8_baseline)
  target_link_options(volvoxai_native_w8a8_baseline PRIVATE
    -Wl,--build-id=sha1
    -Wl,-u,volvoxai_native_w8a8_release_cli_anchor)

  add_custom_target(volvoxai_native_w8a8_performance
    DEPENDS
      volvoxai_native_w8a8_candidate
      volvoxai_native_w8a8_baseline)
endif()

# --- Opt-in generated-API applications ----------------------------------------
# Build applications like external C consumers: they see only the generated
# inference projection plus Synurang's public runtime header, and reach the
# engine through the shipped static library.
add_executable(volvoxai-tasks
  ${REPO}/examples/native_support/image_io.c
  ${REPO}/examples/native_task_cli/main.c)
target_link_libraries(volvoxai-tasks PRIVATE volvoxai_static)
target_compile_definitions(volvoxai-tasks PRIVATE
  VOLVOXAI_VERSION="${VOLVOXAI_VERSION}"
  VOLVOXAI_GIT_COMMIT="${VOLVOXAI_GIT_COMMIT}"
  VOLVOXAI_BUILD_DATE="${VOLVOXAI_BUILD_DATE}")
volvox_target_api_profile(volvoxai-tasks inference)
target_include_directories(volvoxai-tasks PRIVATE
  ${VOLVOXAI_SYNURANG_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  ${REPO}/examples/native_task_cli
  ${REPO}/examples/native_support)
target_compile_options(volvoxai-tasks PRIVATE
  -std=c11 -O2 -Wall -Wextra -Werror)

add_executable(receipt_digit_reader
  ${REPO}/examples/receipt_digit_reader/native/receipt_digit_reader.c
  ${REPO}/examples/receipt_digit_reader/native/main_receipt_digit_reader.c)
target_link_libraries(receipt_digit_reader PRIVATE volvoxai_static)
target_compile_definitions(receipt_digit_reader PRIVATE
  _POSIX_C_SOURCE=200809L)
volvox_target_api_profile(receipt_digit_reader inference)
target_include_directories(receipt_digit_reader PRIVATE
  ${VOLVOXAI_SYNURANG_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  ${REPO}/examples/receipt_digit_reader/native)
target_compile_options(receipt_digit_reader PRIVATE
  -std=c11 -O2 -Wall -Wextra -Werror)

# Opt-in engine-internal scheduler benchmark. It intentionally sits below the
# generated application API to measure coordinator invariants, and needs
# private example artifacts plus a physical GPU. Keep it out of default builds.
if(VOLVOXAI_ENABLE_VULKAN OR VOLVOXAI_ENABLE_OPENGL OR
   VOLVOXAI_ENABLE_CUDA)
  add_executable(native_dynamic_batch_benchmark EXCLUDE_FROM_ALL
    ${FULL_ENGINE_SRCS} ${FULL_SHADER_SRCS}
    ${REPO}/examples/native_dynamic_batch_benchmark/main.c)
  add_dependencies(native_dynamic_batch_benchmark
    volvoxai_pack_full_shaders)
  volvox_cuda_codegen_before(native_dynamic_batch_benchmark)
  target_link_libraries(native_dynamic_batch_benchmark PRIVATE volvox_common)
  target_compile_definitions(native_dynamic_batch_benchmark PRIVATE
    VOLVOXAI_ENABLE_TRAINING=0 ${BK_FULL})
  volvox_target_api_profile(native_dynamic_batch_benchmark full)
  target_include_directories(native_dynamic_batch_benchmark PRIVATE
    ${GEN}/full
    ${REPO}/examples/native_dynamic_batch_benchmark)
  target_compile_options(native_dynamic_batch_benchmark PRIVATE -std=c11 -O3)
endif()

# --- Minimal native inference invariants ------------------------------------
# Tests stay in the inference composition: no training sources or public
# training definitions are linked into these executables.

add_executable(test_kernel_registry
  tests/test_kernel_registry.c tests/kernel_registry_peer.c)
target_link_libraries(test_kernel_registry PRIVATE volvoxai_semantics volvox_common)
target_compile_options(test_kernel_registry PRIVATE
  -std=c11 -O2 -Wall -Wextra -Werror)
add_test(NAME test_kernel_registry COMMAND $<TARGET_FILE:test_kernel_registry>)
set_tests_properties(test_kernel_registry PROPERTIES
  LABELS "native;native-invariant" TIMEOUT 60)

foreach(_profile IN ITEMS inference full)
  add_executable(test_operator_runtime_${_profile} tests/test_operator_runtime.c)
  if(_profile STREQUAL "inference")
    target_link_libraries(test_operator_runtime_${_profile} PRIVATE volvoxai-lite_static volvox_common)
    target_compile_definitions(test_operator_runtime_${_profile} PRIVATE
      VOLVOXAI_ENABLE_TRAINING=0 ${BK_LITE})
  else()
    target_link_libraries(test_operator_runtime_${_profile} PRIVATE volvoxai_static volvox_common)
    target_compile_definitions(test_operator_runtime_${_profile} PRIVATE
      VOLVOXAI_ENABLE_TRAINING=1 ${BK_FULL})
  endif()
  target_compile_options(test_operator_runtime_${_profile} PRIVATE -std=c11 -O2 -Wall -Wextra -Werror)
  add_test(NAME test_operator_runtime_${_profile}
    COMMAND $<TARGET_FILE:test_operator_runtime_${_profile}>
            ${CMAKE_CURRENT_BINARY_DIR}/operator-runtime-${_profile}.graph.json)
  set_tests_properties(test_operator_runtime_${_profile} PROPERTIES
    LABELS "native;native-invariant" TIMEOUT 60)
endforeach()

# The names PTQ authoring introduces are a cross-implementation contract: the
# Python exporter derives them the same way, and a template written by one is
# read by the other. Pin them, and the SHA-256 they come from.
add_executable(test_ptq_names
  tests/test_ptq_names.c src/training/ptq_names.c)
target_include_directories(test_ptq_names PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/training)
target_compile_options(test_ptq_names PRIVATE
  -std=c11 -O2 -Wall -Wextra -Werror)
add_test(NAME test_ptq_names COMMAND $<TARGET_FILE:test_ptq_names>)
set_tests_properties(test_ptq_names PROPERTIES LABELS "native;native-invariant")

# Authors a template from files. With no arguments it checks the cases
# authoring must refuse; python/tests/test_ptq_authoring_parity.py drives it
# with paths and diffs the result against the Python exporter's.
add_executable(test_ptq_authoring
  tests/test_ptq_authoring.c
  src/generated/operator_vocabulary.c
  src/training/ptq_authoring.c src/training/ptq_names.c
  third_party/cJSON.c
  src/runtime/safetensors.c src/runtime/safetensors_header.c
  src/runtime/safetensors_bytes.c)
target_link_libraries(test_ptq_authoring PRIVATE volvox_common)
target_compile_definitions(test_ptq_authoring PRIVATE VOLVOXAI_ENABLE_TRAINING=1)
target_include_directories(test_ptq_authoring PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/training
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_compile_options(test_ptq_authoring PRIVATE
  -std=c11 -O2 -Wall -Wextra -Werror)
add_test(NAME test_ptq_authoring COMMAND $<TARGET_FILE:test_ptq_authoring>)
set_tests_properties(test_ptq_authoring PROPERTIES LABELS "native;native-invariant")

# Exercises AuthorPtqTemplate through the generated native dispatch in both
# mixed modes: path-backed source to response bytes, and byte-backed source to
# a file with an empty PtqTemplateInfo.template_graph field.
add_executable(test_ptq_api_authoring
  ${REPO}/runtime/generated/c/volvoxai_lite.c
  tests/test_ptq_api_authoring.c)
target_link_libraries(test_ptq_api_authoring PRIVATE
  volvoxai_shared volvox_common)
volvox_target_api_profile(test_ptq_api_authoring full)
target_compile_options(test_ptq_api_authoring PRIVATE
  -std=c11 -O2 -Wall -Wextra -Werror)
add_test(NAME test_ptq_api_authoring
  COMMAND $<TARGET_FILE:test_ptq_api_authoring>
          ${CMAKE_CURRENT_BINARY_DIR}/ptq_api_source.graph.json
          ${CMAKE_CURRENT_BINARY_DIR}/ptq_api_authored.graph.json)
set_tests_properties(test_ptq_api_authoring PROPERTIES
  LABELS "native;native-full;native-invariant")

# The WASM parent supplies its own freestanding libc. Compile that implementation
# into a host-side oracle under renamed symbols so cJSON's integer formatting,
# JSON control escapes, and truncation behavior stay pinned to libc semantics.
add_executable(test_wasm_libc_format tests/test_wasm_libc_format.c)
if(UNIX)
  target_link_libraries(test_wasm_libc_format PRIVATE m)
endif()
target_compile_options(test_wasm_libc_format PRIVATE
  -std=c11 -O2 -Wall -Wextra -Werror)
add_test(NAME test_wasm_libc_format
  COMMAND $<TARGET_FILE:test_wasm_libc_format>)
set_tests_properties(test_wasm_libc_format PROPERTIES
  LABELS "native;native-invariant" TIMEOUT 60)

add_executable(test_wasm_math tests/test_wasm_math.c)
if(UNIX)
  target_link_libraries(test_wasm_math PRIVATE m)
endif()
target_compile_options(test_wasm_math PRIVATE
  -std=c11 -O3 -fno-builtin -ffp-contract=off -Wall -Wextra -Werror)
add_test(NAME test_wasm_math COMMAND $<TARGET_FILE:test_wasm_math>)
set_tests_properties(test_wasm_math PROPERTIES
  LABELS "native;native-invariant" TIMEOUT 60)

add_executable(test_paged_decode_cache
  tests/test_paged_decode_cache.c src/runtime/paged_kv.c)
target_include_directories(test_paged_decode_cache PRIVATE src/runtime)
target_compile_options(test_paged_decode_cache PRIVATE
  -std=c11 -O2 -Wall -Wextra -Werror)
add_test(NAME test_paged_decode_cache COMMAND $<TARGET_FILE:test_paged_decode_cache>)
set_tests_properties(test_paged_decode_cache PROPERTIES
  LABELS "native;native-invariant" TIMEOUT 60)

add_executable(test_batch_queue
  tests/test_batch_queue.c src/runtime/batch_scheduler.c src/runtime/paged_kv.c)
target_include_directories(test_batch_queue PRIVATE src/runtime)
target_compile_options(test_batch_queue PRIVATE -std=c11 -O2 -Wall -Wextra -Werror)
add_test(NAME test_batch_queue COMMAND $<TARGET_FILE:test_batch_queue>)
set_tests_properties(test_batch_queue PROPERTIES
  LABELS "native;native-invariant" TIMEOUT 60)

add_executable(test_w8a8_affine tests/test_w8a8_affine.c)
target_link_libraries(test_w8a8_affine PRIVATE volvox_common)
target_compile_definitions(test_w8a8_affine PRIVATE
  VOLVOXAI_ENABLE_TRAINING=0)
target_compile_options(test_w8a8_affine PRIVATE
  -std=c11 -O2 -Wall -Wextra -Werror)
add_test(NAME test_w8a8_affine COMMAND $<TARGET_FILE:test_w8a8_affine>)

add_executable(test_kernel_platform
  tests/test_kernel_platform.c src/kernels/cpu_features.c)
target_link_libraries(test_kernel_platform PRIVATE volvox_common)
target_compile_definitions(test_kernel_platform PRIVATE
  VOLVOXAI_ENABLE_TRAINING=0)
target_compile_options(test_kernel_platform PRIVATE
  -std=c11 -O2 -Wall -Wextra -Werror)
add_test(NAME test_kernel_platform COMMAND $<TARGET_FILE:test_kernel_platform>)

# kernels.c is the production amalgamation and carries the packed implementation
# plus its portable leaf oracle.  Keep its production -O3/baseline flags and
# link the same optional Arm ISA objects advertised by the global definitions.
add_executable(test_packed_quant_gemm
  src/kernels/kernels.c src/kernels/cpu_features.c
  src/kernels/quant_cpu_isa.c ${ARM_OBJS}
  tests/test_packed_quant_gemm.c)
target_link_libraries(test_packed_quant_gemm PRIVATE volvox_common)
target_compile_definitions(test_packed_quant_gemm PRIVATE
  VOLVOXAI_ENABLE_TRAINING=0)
target_compile_options(test_packed_quant_gemm PRIVATE -std=c11 -O3)

add_test(NAME test_packed_quant_gemm_contract
  COMMAND ${CMAKE_COMMAND} -E env --unset=VOLVOXAI_CPU_ISA
          $<TARGET_FILE:test_packed_quant_gemm> --contract)

add_executable(test_thread_pool_lifecycle
  src/kernels/thread_pool.c tests/test_thread_pool_lifecycle.c)
target_link_libraries(test_thread_pool_lifecycle PRIVATE volvox_common)
target_compile_definitions(test_thread_pool_lifecycle PRIVATE
  VOLVOXAI_ENABLE_TRAINING=0)
target_compile_options(test_thread_pool_lifecycle PRIVATE
  -std=c11 -O2 -Wall -Wextra -Werror)
add_test(NAME test_thread_pool_lifecycle
  COMMAND $<TARGET_FILE:test_thread_pool_lifecycle>)

add_executable(test_qsdpa_w8a8_parity
  src/kernels/kernels.c src/kernels/cpu_features.c
  src/kernels/quant_cpu_isa.c src/kernels/qsdpa_w8a8_native.c
  ${ARM_OBJS} tests/test_qsdpa_w8a8_parity.c)
target_link_libraries(test_qsdpa_w8a8_parity PRIVATE volvox_common)
target_compile_definitions(test_qsdpa_w8a8_parity PRIVATE
  VOLVOXAI_ENABLE_TRAINING=0)
target_compile_options(test_qsdpa_w8a8_parity PRIVATE -std=c11 -O3)
add_test(NAME test_qsdpa_w8a8_parity
  COMMAND $<TARGET_FILE:test_qsdpa_w8a8_parity>)

# One clamp process covers the remaining raw CPU dispatchers: canonical
# quantized operators are byte exact, while F32 GEMM/attention use explicit
# relative bounds against their portable arithmetic.
add_executable(test_native_isa_parity
  src/kernels/kernels.c src/kernels/cpu_features.c
  src/kernels/quant_cpu_isa.c
  src/kernels/qlinear_w8a8_x86.c src/kernels/qlinear_w8a8_arm.c
  src/kernels/qconv_w8a8_x86.c src/kernels/qconv_w8a8_arm.c
  src/kernels/qbatch_matmul_w8a8_native.c ${ARM_OBJS}
  tests/test_native_isa_parity.c)
target_link_libraries(test_native_isa_parity PRIVATE volvox_common)
target_compile_definitions(test_native_isa_parity PRIVATE
  VOLVOXAI_ENABLE_TRAINING=0)
target_compile_options(test_native_isa_parity PRIVATE
  -std=c11 -O2 -ffp-contract=off)

# Each clamp is a separate CTest process because vx_kernel_platform() caches
# the environment-derived tier in every translation unit.  A tier unavailable
# on the current runner is a reported skip; baseline always runs.
function(volvox_add_packed_isa_test tier)
  set(_test_name test_packed_quant_gemm_clamp_${tier})
  add_test(NAME ${_test_name}
    COMMAND ${CMAKE_COMMAND} -E env "VOLVOXAI_CPU_ISA=${tier}"
            $<TARGET_FILE:test_packed_quant_gemm> --isa ${tier})
  set_tests_properties(${_test_name} PROPERTIES
    LABELS "native;native-invariant;native-isa-runtime"
    SKIP_RETURN_CODE 77 TIMEOUT 60)

  set(_raw_test_name test_native_isa_parity_clamp_${tier})
  if(tier STREQUAL "avx512vnni")
    set(_f32_avx512 1)
  else()
    set(_f32_avx512 0)
  endif()
  add_test(NAME ${_raw_test_name}
    COMMAND ${CMAKE_COMMAND} -E env "VOLVOXAI_CPU_ISA=${tier}"
            "VOLVOX_F32_AVX512=${_f32_avx512}"
            $<TARGET_FILE:test_native_isa_parity> --isa ${tier})
  set_tests_properties(${_raw_test_name} PROPERTIES
    LABELS "native;native-invariant;native-isa-runtime"
    SKIP_RETURN_CODE 77 TIMEOUT 60)
endfunction()

volvox_add_packed_isa_test(baseline)
if(VOLVOX_X86)
  volvox_add_packed_isa_test(avx2)
  volvox_add_packed_isa_test(avxvnni)
  volvox_add_packed_isa_test(avx512vnni)
elseif(_arch MATCHES "aarch64|arm64|arm")
  volvox_add_packed_isa_test(neon)
  volvox_add_packed_isa_test(neondotprod)
  volvox_add_packed_isa_test(neoni8mm)
  volvox_add_packed_isa_test(sve2)
endif()

set_tests_properties(
  test_w8a8_affine test_kernel_platform test_packed_quant_gemm_contract
  test_thread_pool_lifecycle test_qsdpa_w8a8_parity
  PROPERTIES LABELS "native;native-invariant" TIMEOUT 60)

# Backend-manager semantics do not require a physical GPU. Link the production
# manager and state capsule against four test-owned device stubs so every host
# exercises transactional activation, fail-closed selection, and cleanup/reuse.
# Compile the production manager as a test-private object. Non-Apple runners
# define __APPLE__ only for this one translation unit to admit the Metal branch
# against a stub; no Objective-C, framework, or device implementation is linked,
# and production compositions keep backend_config.h's platform rejection.
add_library(test_native_gpu_backend_manager OBJECT
  src/backends/backend_manager.c)
target_link_libraries(test_native_gpu_backend_manager PRIVATE volvox_common)
target_compile_definitions(test_native_gpu_backend_manager PRIVATE
  VOLVOXAI_ENABLE_TRAINING=0
  VOLVOXAI_ENABLE_VULKAN=1 VOLVOXAI_ENABLE_OPENGL=1
  VOLVOXAI_ENABLE_METAL=1 VOLVOXAI_ENABLE_CUDA=1)
if(NOT APPLE)
  target_compile_definitions(test_native_gpu_backend_manager PRIVATE __APPLE__=1)
endif()
target_compile_options(test_native_gpu_backend_manager PRIVATE
  -std=c11 -O2 -Wall -Wextra -Werror)

add_executable(test_native_gpu_backend_contract
  src/generated/kernel_registry.c
  src/runtime/runtime_state.c src/kernels/thread_pool.c third_party/cJSON.c
  tests/test_native_gpu_backend_contract.c
  $<TARGET_OBJECTS:test_native_gpu_backend_manager>)
target_link_libraries(test_native_gpu_backend_contract PRIVATE volvox_common)
target_compile_definitions(test_native_gpu_backend_contract PRIVATE
  VOLVOXAI_ENABLE_TRAINING=0
  VOLVOXAI_ENABLE_VULKAN=1 VOLVOXAI_ENABLE_OPENGL=1
  VOLVOXAI_ENABLE_METAL=1 VOLVOXAI_ENABLE_CUDA=1)
# The contract test turns every backend on at once, which is a composition no
# real target has. backend_config.h refuses Metal off an Apple target, so this
# claims the target the way the manager object above already does.
if(NOT APPLE)
  target_compile_definitions(test_native_gpu_backend_contract PRIVATE __APPLE__=1)
endif()
target_compile_options(test_native_gpu_backend_contract PRIVATE
  -std=c11 -O2 -Wall -Wextra -Werror)
add_test(NAME test_native_gpu_backend_contract
  COMMAND $<TARGET_FILE:test_native_gpu_backend_contract>)
set_tests_properties(test_native_gpu_backend_contract PROPERTIES
  LABELS "native;native-gpu-static" TIMEOUT 60)

# Linux source-level target-attributed codegen checks complement the runtime
# clamp matrix. They use the configured production compiler to prove that
# higher-tier bodies can be emitted even on runners that cannot execute those
# instructions, and that public dispatchers remain baseline.
if(VOLVOX_X86 AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
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
  add_test(NAME verify_native_qconv_im2col_specialization
    COMMAND ${_verify} emit-qconv-im2col
            ${_kern}/qconv_w8a8_x86.c ${_kern})
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
  add_test(NAME verify_native_packed_qgemm_vnni
    COMMAND ${_verify} emit-packed-vnni
            ${_kern}/packed_quant_gemm.c ${_kern})
  add_test(NAME verify_native_w8a8_baseline_isa
    COMMAND ${_verify} baseline ${_kern})
  set_tests_properties(
    verify_native_avxvnni verify_native_avx512vnni
    verify_native_qconv_avxvnni verify_native_qconv_avx512vnni
    verify_native_qconv_im2col_specialization
    verify_native_qlinear_maddubs verify_native_qbatch_maddubs
    verify_native_packed_qgemm_maddubs
    verify_native_packed_qgemm_signed_maddubs
    verify_native_packed_qgemm_vnni
    verify_native_w8a8_baseline_isa
    PROPERTIES
      ENVIRONMENT "CLANG=${CMAKE_C_COMPILER}"
      LABELS "verify;native-isa-codegen"
      TIMEOUT 120)
endif()
