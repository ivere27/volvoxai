.PHONY: all clean cpp_ffi_example run_cpp_ffi_example \
	native_receipt_split_inference_example \
	test_native_receipt_split_inference run_native_receipt_split_inference \
	native_task_cli test_native_task_cli build_ffi

EXAMPLES_DIR           := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT                   := $(abspath $(EXAMPLES_DIR)/..)
TARGET_DIR             := $(EXAMPLES_DIR)/target
BIN_DIR                := $(TARGET_DIR)/bin
OBJ_DIR                := $(TARGET_DIR)/obj
CPP_FFI_LITE_DIR       := $(ROOT)/runtime/generated/c
CPP_FFI_LITE_HEADER    := $(CPP_FFI_LITE_DIR)/volvoxai_lite.h
CPP_FFI_LITE_SOURCE    := $(CPP_FFI_LITE_DIR)/volvoxai_lite.c
CPP_FFI_LITE_OBJECT    := $(OBJ_DIR)/volvoxai_lite.o
CPP_FFI_EXAMPLE        := $(BIN_DIR)/cpp_ffi_client
NATIVE_RECEIPT_SPLIT_EXAMPLE := $(BIN_DIR)/tiny_receipt_split_w8a8
NATIVE_TASK_CLI              := $(BIN_DIR)/volvoxai-tasks

CXX                     ?= c++
CXXFLAGS                ?= -std=c++17 -O2 -Wall -Wextra
CPP_FFI_CFLAGS          ?= -std=c11 -O2 -Wall -Wextra -pedantic
RUNTIME_SHARED_LIBRARY := $(if $(filter Darwin,$(shell uname -s)),libvolvoxai.dylib,libvolvoxai.so)
RUNTIME_LIBRARY        := $(ROOT)/runtime/target/release/$(RUNTIME_SHARED_LIBRARY)
DLOPEN_LIB             := $(if $(filter Darwin,$(shell uname -s)),,-ldl)

all: cpp_ffi_example

build_ffi:
	$(MAKE) -C $(ROOT) build_ffi

cpp_ffi_example: $(CPP_FFI_EXAMPLE)

$(CPP_FFI_LITE_OBJECT): $(CPP_FFI_LITE_SOURCE) $(CPP_FFI_LITE_HEADER)
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CPP_FFI_CFLAGS) -I$(CPP_FFI_LITE_DIR) -c $(CPP_FFI_LITE_SOURCE) -o $@

$(CPP_FFI_EXAMPLE): $(EXAMPLES_DIR)/cpp_ffi_client.cpp $(CPP_FFI_LITE_OBJECT) \
		$(CPP_FFI_LITE_HEADER)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -I$(CPP_FFI_LITE_DIR) $< $(CPP_FFI_LITE_OBJECT) \
		-o $@ $(DLOPEN_LIB)

run_cpp_ffi_example: build_ffi cpp_ffi_example
	$(CPP_FFI_EXAMPLE) $(RUNTIME_LIBRARY) \
		$(ROOT)/models/efficientdet_lite0_fp32/graph.json \
		$(ROOT)/models/efficientdet_lite0_fp32/model.safetensors

native_receipt_split_inference_example:
	$(MAKE) -C $(ROOT) build_tiny_receipt_split_native_example

test_native_receipt_split_inference:
	$(MAKE) -C $(ROOT) test_tiny_receipt_split_native_example

run_native_receipt_split_inference: native_receipt_split_inference_example
	@test -n "$(PACKAGE)" || { echo "Error: pass PACKAGE=/path/to/package"; exit 1; }
	@test -n "$(IMAGE)" || { echo "Error: pass IMAGE=/path/to/receipt.png"; exit 1; }
	@test -n "$(PROMPT)" || { echo "Error: pass PROMPT='question'"; exit 1; }
	$(NATIVE_RECEIPT_SPLIT_EXAMPLE) "$(PACKAGE)" --image "$(IMAGE)" --prompt "$(PROMPT)"

native_task_cli:
	$(MAKE) -C $(ROOT) build_native_task_cli

test_native_task_cli:
	$(MAKE) -C $(ROOT) test_native_task_cli

clean:
	rm -rf $(TARGET_DIR)
