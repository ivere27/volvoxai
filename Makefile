# VolvoxAI — top-level task runner.
#
# The native C engine is built and tested by CMake + CTest (see CMakeLists.txt,
# native/CMakeLists.txt). This Makefile is a thin wrapper: it drives Docker for
# a reproducible toolchain, forwards native targets to cmake/ctest, and keeps
# the web (npm/esbuild), proto codegen, model export, and Rust service targets
# that have their own toolchains. Run `make help` for the common entry points.

.PHONY: all help build_docker compile_shaders \
        build_native test_native test_native_all test_all test_js \
        verify_native_isa benchmark_native test_native_gpu clean_native \
        build_native_task_cli test_native_task_cli \
        build_tiny_receipt_native_example test_tiny_receipt_native_example \
        build_wasm test_wasm_relaxed_simd benchmark_wasm_w8a8_seed build_web \
        publish_npm webtest release_git \
        models models_efficientdet models_tinystories models_deps models_clean \
        proto_codegen_fetch proto_codegen proto_codegen_check \
        proto_all proto_check proto_c proto_c_lite proto_c_native proto_typescript \
        proto_rust build_server clean_server

VERSION := $(shell grep '"version"' package.json | head -n 1 | cut -d '"' -f 4)
GIT_COMMIT := $(shell git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
GIT_DIRTY := $(shell test -n "$$(git status --porcelain 2>/dev/null)" && echo -dirty)
BUILD_DATE ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
HOST_UID := $(shell id -u)
HOST_GID := $(shell id -g)
DOCKER_IMAGE := volvoxai-build:latest
DOCKER_RUN := docker run --rm -v $(PWD):/workspace -w /workspace $(DOCKER_IMAGE)
PY := python3

# Native engine (CMake). Native compilation uses clang to match the historical
# recipes; the build runs inside the pinned Docker toolchain for reproducibility.
CMAKE_BUILD_DIR ?= build/cmake
CMAKE_CONFIG ?= -DCMAKE_C_COMPILER=clang
NATIVE_INCLUDE_FLAGS := -Inative/include -Inative/src -Inative/src/runtime \
                        -Inative/src/kernels -Inative/src/backends -Inative/third_party \
                        -Inative/third_party/xz-embedded

# Web/WASM artifacts.
WEB_DIST_DIR := dist/$(VERSION)
WEB_JS_ARTIFACTS := $(WEB_DIST_DIR)/volvoxai.js $(WEB_DIST_DIR)/volvoxai.min.js \
                    $(WEB_DIST_DIR)/volvoxai.full.js $(WEB_DIST_DIR)/volvoxai.full.min.js \
                    $(WEB_DIST_DIR)/volvoxai.wasm.js $(WEB_DIST_DIR)/volvoxai.wasm.min.js
WEB_WASM_ARTIFACTS := $(WEB_DIST_DIR)/volvoxai.wasm $(WEB_DIST_DIR)/volvoxai.full.wasm
WEB_ARTIFACTS := $(WEB_JS_ARTIFACTS) $(WEB_WASM_ARTIFACTS)
WASM_RELAXED_SECTION_NAME := volvoxai.relaxed_simd.v1
WASM_RELAXED_CHILD := build/wasm/$(WASM_RELAXED_SECTION_NAME).wasm
WASM_CC ?= clang-17
WASM_RELAXED_CC ?= $(WASM_CC)

all: build_docker

help:
	@echo "VolvoxAI targets:"
	@echo "  Native engine (CMake + CTest, in Docker):"
	@echo "    build_native        compile all native binaries"
	@echo "    test_native         native regression suite (ctest -L native)"
	@echo "    test_native_all     regression + opt-in examples"
	@echo "    verify_native_isa   W8A8 ISA codegen checks (x86)"
	@echo "    benchmark_native    microbenchmarks (not a gate)"
	@echo "    test_native_gpu     Vulkan/OpenGL tests (needs a real device)"
	@echo "  Opt-in native example binaries (-> examples/target/bin/):"
	@echo "    build_native_task_cli  volvoxai-tasks (generate/detect/classify/...)"
	@echo "    build_tiny_receipt_native_example  tiny_receipt_w8a8"
	@echo "  Full suite:"
	@echo "    test_js             JS/TS unit suite (npm test)"
	@echo "    test_all            test_js + test_native_all"
	@echo "  Web / service / models:"
	@echo "    build_wasm build_web publish_npm  build_server  models"
	@echo "    proto_all proto_check"

build_docker:
	docker build \
		--build-arg VOLVOXAI_VERSION=$(VERSION) \
		--build-arg VOLVOXAI_GIT_COMMIT=$(GIT_COMMIT)$(GIT_DIRTY) \
		--build-arg VOLVOXAI_BUILD_DATE=$(BUILD_DATE) \
		-t $(DOCKER_IMAGE) .

# Compile WGSL -> native shader formats (naga). Kept here because the Rust
# service build (build_server) consumes native/shaders; the CMake build runs
# the same script internally for the engine.
compile_shaders: build_docker
	$(DOCKER_RUN) ./tools/compile_shaders.sh
	$(DOCKER_RUN) chown -R $(HOST_UID):$(HOST_GID) native/shaders

# --------------------------------------------------------------------------
# Native engine: thin cmake/ctest wrappers. The shader pipeline runs inside
# the CMake build; the whole thing runs as root in Docker (naga/cargo need it)
# and chowns the generated tree back afterwards.
# --------------------------------------------------------------------------
build_native: build_docker
	$(DOCKER_RUN) bash -c 'set -e; \
		cmake -S . -B $(CMAKE_BUILD_DIR) $(CMAKE_CONFIG); \
		cmake --build $(CMAKE_BUILD_DIR) -j"$$(nproc)"; \
		chown -R $(HOST_UID):$(HOST_GID) $(CMAKE_BUILD_DIR) native/shaders \
			native/volvoxai native/volvoxai-full 2>/dev/null || true'

test_native: build_native
	$(DOCKER_RUN) ctest --test-dir $(CMAKE_BUILD_DIR) -L native --output-on-failure

# Regression + opt-in example policies (was test_native_all in the old Makefile).
test_native_all: build_native
	$(DOCKER_RUN) ctest --test-dir $(CMAKE_BUILD_DIR) -L 'native|example' --output-on-failure

verify_native_isa: build_native
	$(DOCKER_RUN) ctest --test-dir $(CMAKE_BUILD_DIR) -L verify --output-on-failure

benchmark_native: build_native
	$(DOCKER_RUN) ctest --test-dir $(CMAKE_BUILD_DIR) -L benchmark --output-on-failure

# Builds in Docker; running needs a real GPU, so run it where one is attached.
test_native_gpu: build_native
	$(DOCKER_RUN) ctest --test-dir $(CMAKE_BUILD_DIR) -L gpu --output-on-failure

clean_native:
	rm -rf $(CMAKE_BUILD_DIR)

# --------------------------------------------------------------------------
# Opt-in native example binaries. These link the full engine, so they build via
# the same CMake-in-Docker path as build_native, then are copied into the
# example tree at the documented examples/target/bin/<tool> paths. The examples/
# Makefile forwards here (native_task_cli / native_receipt_inference_example).
# --------------------------------------------------------------------------
EXAMPLES_BIN_DIR := examples/target/bin

build_native_task_cli: build_docker
	$(DOCKER_RUN) bash -c 'set -e; \
		cmake -S . -B $(CMAKE_BUILD_DIR) $(CMAKE_CONFIG); \
		cmake --build $(CMAKE_BUILD_DIR) --target volvoxai-tasks -j"$$(nproc)"; \
		mkdir -p $(EXAMPLES_BIN_DIR); \
		cp $(CMAKE_BUILD_DIR)/native/volvoxai-tasks $(EXAMPLES_BIN_DIR)/volvoxai-tasks; \
		chown -R $(HOST_UID):$(HOST_GID) $(CMAKE_BUILD_DIR) native/shaders \
			$(EXAMPLES_BIN_DIR) 2>/dev/null || true'
	@echo "Built $(EXAMPLES_BIN_DIR)/volvoxai-tasks"

test_native_task_cli: build_native_task_cli
	$(DOCKER_RUN) ctest --test-dir $(CMAKE_BUILD_DIR) -R '^test_native_task_cli$$' --output-on-failure

build_tiny_receipt_native_example: build_docker
	$(DOCKER_RUN) bash -c 'set -e; \
		cmake -S . -B $(CMAKE_BUILD_DIR) $(CMAKE_CONFIG); \
		cmake --build $(CMAKE_BUILD_DIR) --target tiny_receipt_w8a8 test_tiny_receipt_w8a8 -j"$$(nproc)"; \
		mkdir -p $(EXAMPLES_BIN_DIR); \
		cp $(CMAKE_BUILD_DIR)/native/tiny_receipt_w8a8 $(EXAMPLES_BIN_DIR)/tiny_receipt_w8a8; \
		cp $(CMAKE_BUILD_DIR)/native/test_tiny_receipt_w8a8 $(EXAMPLES_BIN_DIR)/test_tiny_receipt_w8a8; \
		chown -R $(HOST_UID):$(HOST_GID) $(CMAKE_BUILD_DIR) native/shaders \
			$(EXAMPLES_BIN_DIR) 2>/dev/null || true'
	@echo "Built $(EXAMPLES_BIN_DIR)/tiny_receipt_w8a8"

test_tiny_receipt_native_example: build_tiny_receipt_native_example
	$(DOCKER_RUN) ctest --test-dir $(CMAKE_BUILD_DIR) -R '^test_tiny_receipt_native_example$$' --output-on-failure

# JS/TS unit suite (CI runs `npm test` directly; this is the Make alias).
test_js:
	npm test

# Full portable regression: JS suite + native suite.
test_all: test_js test_native_all

release_git: build_native
	@echo "Creating git release..."
	# Example: gh release create v$(VERSION) ./native/volvoxai -t "Release v$(VERSION)"
	@echo "Git release automation should be added here."

# --------------------------------------------------------------------------
# Web build (JS/WASM/WGSL).
# --------------------------------------------------------------------------
build_wasm: build_docker
	$(DOCKER_RUN) mkdir -p $(WEB_DIST_DIR) $(dir $(WASM_RELAXED_CHILD))
	$(DOCKER_RUN) rm -f $(WEB_WASM_ARTIFACTS)
	$(DOCKER_RUN) $(WASM_RELAXED_CC) --target=wasm32 -std=c11 -O3 \
		-Wall -Wextra -Werror -msimd128 -mrelaxed-simd -nostdlib \
		-ffreestanding -Wl,--no-entry -Wl,--import-memory -Wl,--strip-all \
		-o $(WASM_RELAXED_CHILD) \
		native/src/kernels/qlinear_w8a8_wasm_relaxed.c
	$(DOCKER_RUN) $(WASM_CC) --target=wasm32 -O3 -msimd128 -nostdlib \
		-Wl,--no-entry -Wl,--export-all -Wl,--allow-undefined \
		-o $(WEB_DIST_DIR)/volvoxai.wasm native/src/kernels/kernels.c
	$(DOCKER_RUN) $(WASM_CC) --target=wasm32 -O3 -msimd128 -nostdlib \
		-Wl,--no-entry -Wl,--export-all -Wl,--allow-undefined \
		$(NATIVE_INCLUDE_FLAGS) \
		-o $(WEB_DIST_DIR)/volvoxai.full.wasm \
		native/src/kernels/kernels.c native/src/kernels/training_kernels.c \
		native/src/training/quantization.c
	$(DOCKER_RUN) $(PY) tools/embed_wasm_custom_section.py \
		--section-name $(WASM_RELAXED_SECTION_NAME) \
		--payload $(WASM_RELAXED_CHILD) \
		--input $(WEB_DIST_DIR)/volvoxai.wasm \
		--output $(WEB_DIST_DIR)/volvoxai.wasm
	$(DOCKER_RUN) $(PY) tools/embed_wasm_custom_section.py \
		--section-name $(WASM_RELAXED_SECTION_NAME) \
		--payload $(WASM_RELAXED_CHILD) \
		--input $(WEB_DIST_DIR)/volvoxai.full.wasm \
		--output $(WEB_DIST_DIR)/volvoxai.full.wasm
	$(DOCKER_RUN) chmod 0644 $(WEB_WASM_ARTIFACTS)
	$(DOCKER_RUN) chown -R $(HOST_UID):$(HOST_GID) $(WEB_DIST_DIR)
	@echo "Built $(WEB_WASM_ARTIFACTS)"

test_wasm_relaxed_simd: build_wasm
	$(DOCKER_RUN) node --test tests/js_wasm_w8a32_packed_simd.test.mjs
	$(DOCKER_RUN) node tools/test_wasm_relaxed_simd.mjs --baseline-only \
		$(WEB_WASM_ARTIFACTS)
	$(DOCKER_RUN) node --experimental-wasm-relaxed-simd \
		tools/test_wasm_relaxed_simd.mjs $(WEB_WASM_ARTIFACTS)

benchmark_wasm_w8a8_seed: build_wasm
	$(DOCKER_RUN) node tools/benchmark_wasm_w8a8_seed.mjs \
		$(WEB_DIST_DIR)/volvoxai.wasm

build_web: build_docker build_wasm
	$(DOCKER_RUN) npm install
	$(DOCKER_RUN) npm run build:all
	$(DOCKER_RUN) chown -R $(HOST_UID):$(HOST_GID) $(WEB_DIST_DIR)
	@echo "Built $(WEB_ARTIFACTS)"

# Publish NPM from the already-built dist artifacts.
publish_npm:
	@command -v npm >/dev/null 2>&1 || { echo "Error: npm not found in PATH."; exit 1; }
	@set -e; \
	for f in $(WEB_ARTIFACTS); do \
		if [ ! -f "$$f" ]; then \
			echo "Error: $$f not found."; \
			echo "Build web artifacts first: make build_web"; \
			exit 1; \
		fi; \
	done; \
	unexpected=$$(node -e "const f=require('node:fs'),d='$(WEB_DIST_DIR)';const keep=new Set(['volvoxai.js','volvoxai.min.js','volvoxai.full.js','volvoxai.full.min.js','volvoxai.wasm.js','volvoxai.wasm.min.js','volvoxai.wasm','volvoxai.full.wasm']);console.log(f.readdirSync(d).filter(x=>!keep.has(x)).map(x=>d+'/'+x).join('\\n'))"); \
	if [ -n "$$unexpected" ]; then \
		echo "Error: unexpected files remain in $(WEB_DIST_DIR)/:"; \
		echo "$$unexpected"; \
		echo "Run npm run build:all before publishing."; \
		exit 1; \
	fi; \
	if ! npm whoami >/dev/null 2>&1; then \
		echo "Error: npm is not authenticated. Use npm login or provide the existing npm auth environment."; \
		exit 1; \
	fi; \
	echo "Package contents:"; \
	npm pack --dry-run; \
	echo "Publishing volvoxai@$(VERSION) to npm..."; \
	npm publish --access public; \
	echo "npm publish complete."

webtest: build_web
	@echo "Starting local web server at http://localhost:8085"
	@echo "Open http://localhost:8085/examples/efficientdet_lite0.html in your browser"
	python3 -m http.server 8085

# --------------------------------------------------------------------------
# Models: the models/ directory is .gitignored (large weights), so regenerate
# it from public sources. These run on the HOST (need Python ML deps the build
# image does not carry). Install them once with `make models_deps`.
# --------------------------------------------------------------------------
models_deps:
	$(PY) -m pip install -r tools/requirements-export.txt
	$(PY) -m pip install -r examples/tinystories/requirements-export.txt

models: models_efficientdet models_tinystories

models_efficientdet:
	PY=$(PY) bash ./examples/efficientdet_lite0/tools/fetch_model.sh

models_tinystories:
	PY=$(PY) bash ./examples/tinystories/tools/fetch_model.sh

models_clean:
	rm -rf models/efficientdet_lite0_int8 models/efficientdet_lite0_fp16 \
	       models/efficientdet_lite0_fp32 models/tinystories_1m

# --------------------------------------------------------------------------
# Service wrapper: expose the C engine as a Synurang service (Rust runtime).
# Runs on the HOST. Needs Python 3 and protoc; the Rust service also needs a
# Rust toolchain and libvulkan-dev. See docs for PROTOC_GEN_SYNURANG_FFI.
# --------------------------------------------------------------------------
SYNURANG_CODEGEN_TOOL     := tools/generate_proto.py
SYNURANG_CODEGEN_CACHE    ?= build/cache/synurang
SYNURANG_CODEGEN_TARGET   ?=
PROTOC                    ?= protoc
PROTOC_GEN_SYNURANG_FFI   ?=
PROTO_DIR                 := proto
PROTO_SOURCE              := $(PROTO_DIR)/volvoxai.proto
PROTO_C_GEN_DIR           := runtime/generated/c
PROTO_TYPESCRIPT_GEN_DIR  := runtime/generated/typescript
PROTO_RUST_GEN_DIR        := runtime/src/gen
RUNTIME_SHARED_LIBRARY    := $(if $(filter Darwin,$(shell uname -s)),libvolvoxai.dylib,libvolvoxai.so)

SYNURANG_GENERATOR_ARG = $(if $(strip $(PROTOC_GEN_SYNURANG_FFI)),--generator "$(PROTOC_GEN_SYNURANG_FFI)")
SYNURANG_TARGET_ARG = $(if $(strip $(SYNURANG_CODEGEN_TARGET)),--target "$(SYNURANG_CODEGEN_TARGET)")
SYNURANG_CODEGEN = $(PY) $(SYNURANG_CODEGEN_TOOL) \
	--protoc "$(PROTOC)" \
	--cache-dir "$(SYNURANG_CODEGEN_CACHE)" \
	--proto "$(PROTO_SOURCE)" --proto-root "$(PROTO_DIR)" \
	--c-out "$(PROTO_C_GEN_DIR)" \
	--typescript-out "$(PROTO_TYPESCRIPT_GEN_DIR)" \
	--rust-out "$(PROTO_RUST_GEN_DIR)" \
	$(SYNURANG_GENERATOR_ARG) $(SYNURANG_TARGET_ARG)

proto_codegen_fetch:
	$(SYNURANG_CODEGEN) --fetch-only

proto_codegen:
	$(SYNURANG_CODEGEN)

proto_codegen_check:
	$(SYNURANG_CODEGEN) --check

proto_all: proto_codegen

proto_check: proto_codegen_check

proto_c:
	$(SYNURANG_CODEGEN) --language c-lite,c-native

proto_c_lite:
	$(SYNURANG_CODEGEN) --language c-lite

proto_c_native:
	$(SYNURANG_CODEGEN) --language c-native

proto_typescript:
	$(SYNURANG_CODEGEN) --language typescript

proto_rust:
	$(SYNURANG_CODEGEN) --language rust

build_server: compile_shaders proto_rust
	cd runtime && cargo build --release
	@echo "Built runtime/target/release/$(RUNTIME_SHARED_LIBRARY)"

clean_server:
	rm -rf $(PROTO_RUST_GEN_DIR) runtime/target
