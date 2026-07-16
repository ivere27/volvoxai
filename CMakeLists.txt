# VolvoxAI native engine — CMake build.
#
# Scope: this file builds ONLY the native C engine (release binaries, tests,
# benchmarks, ISA codegen checks). The web/WASM bundles (npm + esbuild), the
# Rust service (cargo), and proto codegen keep their own toolchains and are
# driven from the thin top-level Makefile — see `make help`.
#
# Reproducible builds still run inside the pinned Docker toolchain image; CMake
# runs *inside* that image, it does not replace it. Native compilation uses
# clang to match the historical Makefile, so configure with:
#   cmake -S . -B build/cmake -DCMAKE_C_COMPILER=clang
cmake_minimum_required(VERSION 3.20)

# --- Version / provenance (mirrors NATIVE_BUILD_DEFS in the old Makefile) -----
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/package.json" _volvox_pkg_json)
string(JSON VOLVOXAI_VERSION GET "${_volvox_pkg_json}" version)

find_package(Git QUIET)
set(VOLVOXAI_GIT_COMMIT "unknown")
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short=12 HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE VOLVOXAI_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} status --porcelain
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE _volvox_git_dirty
    ERROR_QUIET)
  if(NOT "${_volvox_git_dirty}" STREQUAL "")
    set(VOLVOXAI_GIT_COMMIT "${VOLVOXAI_GIT_COMMIT}-dirty")
  endif()
endif()

if(NOT DEFINED VOLVOXAI_BUILD_DATE)
  string(TIMESTAMP VOLVOXAI_BUILD_DATE "%Y-%m-%dT%H:%M:%SZ" UTC)
endif()

project(volvoxai VERSION ${VOLVOXAI_VERSION} LANGUAGES C)

# Do NOT default to a Release config: the historical build never passed
# -DNDEBUG, and several tests rely on asserts. Optimisation levels are set
# per target to match the old per-recipe -O2/-O3 choices exactly.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "" CACHE STRING "" FORCE)
endif()
set(CMAKE_C_STANDARD_REQUIRED OFF)
set(CMAKE_C_EXTENSIONS ON)

enable_testing()
add_subdirectory(native)
