.PHONY: all clean c_api_example run_c_api_example native_task_cli

EXAMPLES_DIR           := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT                   := $(abspath $(EXAMPLES_DIR)/..)
TARGET_DIR             := $(EXAMPLES_DIR)/target
BIN_DIR                := $(TARGET_DIR)/bin
OBJ_DIR                := $(TARGET_DIR)/obj
# Generated schema and the same pinned module ABI as the native engine.
API_GEN_DIR            := $(ROOT)/runtime/generated/c/inference
SYNURANG_INCLUDE_DIR    := $(ROOT)/native/third_party/synurang/include
API_LITE_HEADER        := $(API_GEN_DIR)/volvoxai_lite.h
API_METHODS_HEADER     := $(ROOT)/native/src/generated/proto_methods.h
CALL_CLIENT_HEADER     := $(ROOT)/native/cli/call_client.h
SYNURANG_RUNTIME_HEADER := $(SYNURANG_INCLUDE_DIR)/synurang/call.h
C_API_EXAMPLE          := $(BIN_DIR)/c_api_client_raw
NATIVE_TASK_CLI              := $(BIN_DIR)/volvoxai-tasks

C_API_CFLAGS           ?= -std=c11 -O2 -Wall -Wextra
DL_LIBS                ?= $(if $(filter Linux,$(shell uname -s)),-ldl,)
# The profile archive is self-contained. Override this for an out-of-tree
# native build or select libvolvoxai.a for full-profile examples.
NATIVE_LIB             ?= $(ROOT)/native/libvolvoxai-lite.a

all: c_api_example

c_api_example: $(C_API_EXAMPLE)

$(C_API_EXAMPLE): $(EXAMPLES_DIR)/c_api_client_raw.c $(API_LITE_HEADER) \
		$(API_METHODS_HEADER) $(CALL_CLIENT_HEADER) $(SYNURANG_RUNTIME_HEADER) \
		$(SYNURANG_INCLUDE_DIR)/synurang/c_runtime.h $(NATIVE_LIB)
	@mkdir -p $(BIN_DIR)
	$(CC) $(C_API_CFLAGS) -I$(API_GEN_DIR) -I$(SYNURANG_INCLUDE_DIR) \
		$(EXAMPLES_DIR)/c_api_client_raw.c $(NATIVE_LIB) -o $@ -lm -lpthread $(DL_LIBS)

run_c_api_example: c_api_example
	$(C_API_EXAMPLE) $(ROOT)/models/tinystories_1m

native_task_cli:
	$(MAKE) -C $(ROOT) build_native_task_cli

clean:
	rm -rf $(TARGET_DIR)
