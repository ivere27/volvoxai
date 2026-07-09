.PHONY: all build_docker compile_shaders build_wasm build_web publish_npm webtest \
        build_native build_android test_native release_git \
        models models_efficientdet models_tinystories models_deps models_clean \
        build_codegen_tool proto_rust build_server clean_server \
        cpp_ffi_example run_cpp_ffi_example

VERSION := $(shell grep '"version"' package.json | head -n 1 | cut -d '"' -f 4)
GIT_COMMIT := $(shell git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
GIT_DIRTY := $(shell test -n "$$(git status --porcelain 2>/dev/null)" && echo -dirty)
BUILD_DATE ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
NATIVE_BUILD_DEFS := -DVOLVOXAI_VERSION='\"$(VERSION)\"' \
                     -DVOLVOXAI_GIT_COMMIT='\"$(GIT_COMMIT)$(GIT_DIRTY)\"' \
                     -DVOLVOXAI_BUILD_DATE='\"$(BUILD_DATE)\"'
DOCKER_IMAGE := volvoxai-build:latest
DOCKER_RUN := docker run --rm -v $(PWD):/workspace -w /workspace $(DOCKER_IMAGE)
PY := python3

all: build_docker

build_docker:
	docker build \
		--build-arg VOLVOXAI_VERSION=$(VERSION) \
		--build-arg VOLVOXAI_GIT_COMMIT=$(GIT_COMMIT)$(GIT_DIRTY) \
		--build-arg VOLVOXAI_BUILD_DATE=$(BUILD_DATE) \
		-t $(DOCKER_IMAGE) .

# Compile WGSL to Native formats using Naga
compile_shaders: build_docker
	$(DOCKER_RUN) ./tools/compile_shaders.sh

# Build WASM from C kernels
build_wasm: build_docker
	$(DOCKER_RUN) clang --target=wasm32 -O3 -msimd128 -nostdlib \
		-Wl,--no-entry -Wl,--export-all -Wl,--allow-undefined \
		-o volvoxai.wasm native/kernels.c
	@echo "Built volvoxai.wasm"

# Web Build (JS/WASM/WGSL)
build_web: build_docker build_wasm
	$(DOCKER_RUN) npm install
	$(DOCKER_RUN) npm run build:all
	$(DOCKER_RUN) bash -c "mkdir -p dist/v$(VERSION) && cp volvoxai.wasm dist/ 2>/dev/null || true && cp dist/volvoxai.js dist/v$(VERSION)/ && cp dist/volvoxai.min.js dist/v$(VERSION)/ && cp volvoxai.wasm dist/v$(VERSION)/ 2>/dev/null || true"

# Publish NPM (JS/WASM/WebGPU engine) from the already-built dist artifacts.
publish_npm:
	@command -v npm >/dev/null 2>&1 || { echo "Error: npm not found in PATH."; exit 1; }
	@set -e; \
	for f in dist/volvoxai.js dist/volvoxai.min.js dist/volvoxai.wasm volvoxai.wasm; do \
		if [ ! -f "$$f" ]; then \
			echo "Error: $$f not found."; \
			echo "Build web artifacts first: make build_web"; \
			exit 1; \
		fi; \
	done; \
	if ! npm whoami >/dev/null 2>&1; then \
		echo "Error: npm is not authenticated. Use npm login or provide the existing npm auth environment."; \
		exit 1; \
	fi; \
	echo "Package contents:"; \
	npm pack --dry-run; \
	echo "Publishing volvoxai@$(VERSION) to npm..."; \
	npm publish --access public; \
	echo "npm publish complete."

# Run local web server for testing browser examples
webtest: build_web
	@echo "Starting local web server at http://localhost:8085"
	@echo "Open http://localhost:8085/examples/efficientdet_lite0.html in your browser"
	python3 -m http.server 8085

# Build Native Bare-Metal Engine (C/C++)
build_native: build_docker
	@echo "Building native bare-metal executable and tests..."
	$(DOCKER_RUN) bash -c "cd native && clang -O3 -mavx2 -mfma -pthread $(NATIVE_BUILD_DEFS) -I. cJSON.c safetensors.c kernels.c quant_cpu_opt.c conv_f32_opt.c tensor_f32_opt.c engine_runtime.c engine.c image_io.c kie_runtime.c vulkan_engine.c opengl_engine.c tokenizer.c nnapi_engine.c main.c -o volvoxai -lm -ldl"
	$(DOCKER_RUN) bash -c "cd native && clang -O3 -mavx2 -mfma -pthread -I. cJSON.c kernels.c test_ops.c -o test_ops -lm 2>/dev/null || true"
	$(DOCKER_RUN) bash -c "clang -O3 -mavx2 -mfma -pthread -Inative native/cJSON.c native/safetensors.c native/kernels.c examples/bert_native.c -o examples/bert_native -lm 2>/dev/null || true"

# Build Android Executable (with NNAPI support)
build_android: build_docker
	@echo "Building android executable..."
	$(DOCKER_RUN) bash -c "cd native && clang -O3 -DUSE_NNAPI -pthread $(NATIVE_BUILD_DEFS) -I. cJSON.c safetensors.c kernels.c quant_cpu_opt.c conv_f32_opt.c tensor_f32_opt.c engine_runtime.c engine.c image_io.c kie_runtime.c vulkan_engine.c opengl_engine.c tokenizer.c nnapi_engine.c main.c -o volvoxai_android -lm -ldl"

# Run tests
test_native: build_native
	$(DOCKER_RUN) bash -c "cd native && ./test_ops"
release_git: build_native
	@echo "Creating git release..."
	# Normally we'd use gh CLI or a similar tool here.
	# Example: $(DOCKER_RUN) gh release create v$(VERSION) ./native/volvoxai -t "Release v$(VERSION)" -n "Native bare-metal engine"
	@echo "Git release automation should be added here."

# --------------------------------------------------------------------------
# Models: the models/ directory is .gitignored (large weights), so regenerate
# it from public sources by downloading + exporting to the Volvox blueprint.
# These run on the HOST (not in Docker): they need Python ML deps that the
# build image does not carry. Install them once with `make models_deps`.
# --------------------------------------------------------------------------

# Install the Python deps needed only for model export (runtime has none).
models_deps:
	$(PY) -m pip install -r tools/requirements-export.txt

# Download + export every bundled model (EfficientDet fp32/fp16/int8 + TinyStories).
models: models_efficientdet models_tinystories

# EfficientDet-Lite0 object detector, all three precisions, from MediaPipe.
models_efficientdet:
	PY=$(PY) ./tools/fetch_models.sh efficientdet

# TinyStories-1M GPT-style language model, from HuggingFace (weights + tokenizer).
models_tinystories:
	PY=$(PY) ./tools/fetch_models.sh tinystories

# Remove the regenerated model directories.
models_clean:
	rm -rf models/efficientdet_lite0_int8 models/efficientdet_lite0_fp16 \
	       models/efficientdet_lite0_fp32 models/tinystories_1m

# --------------------------------------------------------------------------
# Service wrapper: expose the C engine as a Synurang service (Rust runtime).
#
# Follows the volvoxgrid pattern with the Synurang *Rust* generator. The result
# is ONE shared library, libvolvoxai.so, that:
#   * implements the generated VolvoxAIServicePlugin trait,
#   * cc-compiles ALL of native/*.c (the engine) straight into the same .so, and
#   * exposes the Synurang_* FFI entry points (serves FFI and TCP, streaming OK).
# Rust owns no inference logic — it is thin glue over the C engine via `extern C`.
#
# Runs on the HOST. One-time setup installs the generator from GitHub:
#   make build_codegen_tool
# Needs: a Rust toolchain, protoc, and (like the Docker image) libvulkan-dev for
# native/vulkan_engine.c.
# --------------------------------------------------------------------------
SYNURANG_GIT            ?= https://github.com/ivere27/synurang
SYNURANG_REV            ?= 6b76e9687be5469f68e0ac8eb221aa69b104d5a1
PROTOC                  ?= protoc
PROTOC_GEN_SYNURANG_FFI ?= $(shell cargo_home=$${CARGO_HOME:-$$HOME/.cargo}; printf '%s/bin/protoc-gen-synurang-ffi-rs' "$$cargo_home")
PROTO_DIR               := proto
GEN_DIR                 := runtime/src/gen
PROTO3_OPT              := --experimental_allow_proto3_optional
CXX                     ?= c++
CXXFLAGS                ?= -std=c++17 -O2 -Wall -Wextra
CPP_FFI_EXAMPLE         := runtime/target/examples/cpp_ffi_client

# Install the Synurang Rust code generator from GitHub (one-time).
build_codegen_tool:
	cargo install --git $(SYNURANG_GIT) --rev $(SYNURANG_REV) protoc-gen-synurang-ffi-rs
	@test -x "$(PROTOC_GEN_SYNURANG_FFI)" || { echo "protoc-gen-synurang-ffi-rs not found at $(PROTOC_GEN_SYNURANG_FFI)"; exit 1; }

# Generate the Rust plugin trait + Synurang_* FFI glue into runtime/src/gen/.
proto_rust:
	@mkdir -p $(GEN_DIR)
	$(PROTOC) -I$(PROTO_DIR) $(PROTO3_OPT) \
		--plugin=protoc-gen-synurang-ffi=$(PROTOC_GEN_SYNURANG_FFI) \
		--synurang-ffi_out=$(GEN_DIR) --synurang-ffi_opt=lang=rust,mode=plugin_server \
		$(PROTO_DIR)/volvoxai.proto
	# The generated file leads with inner attributes, so imports can't precede
	# its `include!`. Inject the message-type imports right after them.
	@sed -i '/^#!\[allow(dead_code)\]/a use super::pb::*;' $(GEN_DIR)/volvoxai_ffi_plugin.rs
	# The proto defines `message Box`; the glob above would shadow std Box, which
	# the generated code needs for trait objects. Explicit import wins over glob.
	@sed -i '/^#!\[allow(dead_code)\]/a use std::boxed::Box;' $(GEN_DIR)/volvoxai_ffi_plugin.rs
	@echo "Generated Rust bindings in $(GEN_DIR)/"

# Build the single cdylib. runtime/build.rs cc-compiles native/*.c into the .so.
build_server: proto_rust
	cd runtime && cargo build --release
	@echo "Built runtime/target/release/libvolvoxai.so"

clean_server:
	rm -rf $(GEN_DIR) runtime/target

# Standalone C++ FFI client example. It dlopen's libvolvoxai.so and calls the
# generated Synurang_Invoke_* ABI without linking protobuf.
cpp_ffi_example:
	@mkdir -p runtime/target/examples
	$(CXX) $(CXXFLAGS) runtime/examples/cpp_ffi_client.cpp -o $(CPP_FFI_EXAMPLE) -ldl

run_cpp_ffi_example:
	cd runtime && cargo build --release
	$(MAKE) cpp_ffi_example
	$(CPP_FFI_EXAMPLE) runtime/target/release/libvolvoxai.so .
