#!/bin/bash
# Assert the CPU-only binary links no device backend (was
# test_native_backend_composition in the root Makefile).
set -e
bin="$1"
"$bin" --help >/dev/null
if nm "$bin" | grep -Eq ' (vk_init|opengl_init|metal_init|cuda_init)$'; then
  echo 'CPU-only binary unexpectedly contains a device backend.'
  exit 1
fi
echo 'Verified CPU-only binary contains no device backend.'
