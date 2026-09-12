#!/bin/bash
# Assert the inference/full release binaries keep their capability boundaries.
#   $1 = inference binary (native/volvoxai)
#   $2 = full binary      (native/volvoxai-full)
set -e
inf="$1"
full="$2"

elf_build_id() {
  readelf -n "$1" 2>/dev/null | awk '/Build ID:/ { print $3; exit }'
}

# Linux release executables are stripped after their optimized debug image is
# saved in the standard sibling .debug directory. Use the shipped binary when
# it still has a symbol table (for example on macOS), otherwise require a
# build-id-matched sidecar. This keeps physical implementation checks strong
# without making private symbols part of the distributed executable ABI.
native_symbol_file() {
  local binary="$1"
  if [ "$(uname -s)" = "Darwin" ] ||
     nm --defined-only "$binary" 2>/dev/null | grep -q .; then
    printf '%s\n' "$binary"
    return
  fi
  local debug_dir="${VOLVOXAI_NATIVE_DEBUG_DIR:-$(dirname "$binary")/.debug}"
  local debug_file="$debug_dir/$(basename "$binary").debug"
  if [ ! -f "$debug_file" ]; then
    echo "$binary is stripped and its debug artifact is missing: $debug_file" >&2
    exit 1
  fi
  local binary_build_id="$(elf_build_id "$binary")"
  local debug_build_id="$(elf_build_id "$debug_file")"
  if [ -z "$binary_build_id" ] || [ "$binary_build_id" != "$debug_build_id" ]; then
    echo "$binary and $debug_file do not have the same non-empty build ID" >&2
    exit 1
  fi
  printf '%s\n' "$debug_file"
}

inf_symbols="$(native_symbol_file "$inf")"
full_symbols="$(native_symbol_file "$full")"

# Apple nm prefixes C symbols with an underscore and uses -U for
# "defined-only". Normalize both GNU/ELF and Apple/Mach-O output to one symbol
# per line so this release-boundary check is identical on Linux and macOS.
defined_symbol_names() {
  if [ "$(uname -s)" = "Darwin" ]; then
    nm -U "$1" 2>/dev/null | awk '{ symbol = $NF; sub(/^_/, "", symbol); print symbol }'
  else
    nm --defined-only "$1" 2>/dev/null | awk '{ print $NF }'
  fi
}

global_symbol_names() {
  if [ "$(uname -s)" = "Darwin" ]; then
    nm -gU "$1" 2>/dev/null | awk '{ symbol = $NF; sub(/^_/, "", symbol); print symbol }'
  else
    nm -g --defined-only "$1" 2>/dev/null | awk '{ print $NF }'
  fi
}

default_global_symbol_names() {
  if [ "$(uname -s)" = "Darwin" ]; then
    nm -g -m -U "$1" 2>/dev/null | awk '
      $0 !~ /private external/ {
        for (field = 1; field < NF; field++) {
          if ($field == "external") {
            symbol = $(field + 1)
            sub(/^_/, "", symbol)
            print symbol
            break
          }
        }
      }
    '
  else
    readelf -Ws "$1" | awk '$5 == "GLOBAL" && $6 == "DEFAULT" { print $8 }'
  fi
}

defined_dynamic_symbol_names() {
  if [ "$(uname -s)" != "Darwin" ]; then
    readelf --dyn-syms -W "$1" | awk '
      $5 ~ /^(GLOBAL|WEAK)$/ && $7 != "UND" { print $8 }
    '
  fi
}

if "$inf" --help | grep -q "^  train "; then
  echo "$inf unexpectedly exposes the full-profile train command."; exit 1
fi
if ! "$full" --help | grep -q "^  train "; then
  echo "$full is missing the full-profile train command."; exit 1
fi
if "$inf" train >/dev/null 2>&1; then
  echo "$inf unexpectedly accepts the full-profile train command."; exit 1
fi
if ! "$full" train --help >/dev/null 2>&1; then
  echo "$full does not provide train command help."; exit 1
fi

if defined_symbol_names "$inf_symbols" | grep -Eq \
    "^(volvoxai_engine_train_|volvoxai_training_|volvoxai_autograd_|vx_dynamic_autograd|vx_training_control_|optimizer_state_for|volvoxai_(engine_)?ptq_)"; then
  echo "Inference binary unexpectedly contains training implementation."; exit 1
fi
if global_symbol_names "$inf_symbols" | grep -Eq \
    "^(vx_model_create_trainer|vx_trainer_[A-Za-z0-9_]*|vx_model_create_ptq_plan|vx_ptq_plan_[A-Za-z0-9_]*)$"; then
  echo "Inference binary unexpectedly exposes a full-profile authoring API."; exit 1
fi
for symbol in vx_training_control_core_abi_version \
              vx_training_control_state_init_v1 \
              vx_training_control_state_validate_v1 \
              vx_training_control_step_begin_v1 \
              vx_training_control_step_finish_v1 \
              vx_training_control_prepare_commit_v1 \
              vx_training_control_prepare_rollback_v1 \
              vx_training_control_prepare_reset_accumulation_v1; do
  # These are portable implementation authorities, not public ABI. A linker
  # may localize helpers used within one translation unit, so require their
  # code to be present without accidentally making global visibility part of
  # the release contract.
  if ! defined_symbol_names "$full_symbols" | grep -Fxq "$symbol"; then
    echo "Full binary is missing portable training-control core ${symbol}."; exit 1
  fi
done
if defined_symbol_names "$inf_symbols" | grep -Eq \
    "^(volvoxai_cuda_training_ptx(_size)?|cuda_profile_route_tracking_active|cuda_training_(available|supports|preflight|begin|dispatch|sync|mark_failed|end))$"; then
  echo "Inference binary unexpectedly contains CUDA training code."; exit 1
fi
# CUDA-off profiles have neither PTX module. When the inference executable
# proves CUDA is enabled by retaining the referenced forward PTX, the matching
# full executable must also retain its separately embedded training module.
if global_symbol_names "$inf_symbols" | grep -Fxq "volvoxai_cuda_ptx"; then
  if ! global_symbol_names "$full_symbols" | grep -Fxq "volvoxai_cuda_training_ptx"; then
    echo "CUDA-enabled full binary is missing its training PTX module."; exit 1
  fi
fi
for binary in "$inf" "$full"; do
  if default_global_symbol_names "$binary" | grep -Eq \
      '^(volvoxai_engine_|volvoxai_training_|volvoxai_autograd_|volvoxai_ptq_)'; then
    echo "$binary exposes a private engine or training symbol."; exit 1
  fi
  if defined_dynamic_symbol_names "$binary" | grep -q .; then
    echo "$binary unexpectedly exposes a defined dynamic symbol."; exit 1
  fi
done
for profile in "inference|$inf|$inf_symbols" "full|$full|$full_symbols"; do
  profile_name="${profile%%|*}"
  remainder="${profile#*|}"
  binary="${remainder%%|*}"
  symbols="${remainder#*|}"
  if "$binary" --help | grep -Eq "^  (generate|classify|detect|ctc|seq2seq|chat|tinyreceipt) "; then
    echo "$binary unexpectedly exposes a model-specific command."; exit 1
  fi
  if global_symbol_names "$symbols" | grep -Eq "^(tiny_receipt_split_w8a8_run|kie_chat|kie_config_is_tiny_receipt|volvoxai_tokenizer_[A-Za-z0-9_]*)$"; then
    echo "$profile_name profile unexpectedly contains a model-specific session symbol."; exit 1
  fi
  if strings "$binary" | grep -Eq "volvoxai-tiny-receipt-vqa-split-kv-onnx-package-v2|tiny_receipt_kie|vocab\.bin|merges\.txt"; then
    echo "$binary unexpectedly contains model or vocabulary policy."; exit 1
  fi
done
echo 'Verified inference/full profile boundaries.'
