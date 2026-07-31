#!/bin/bash
# Assert the inference/full release binaries keep their capability boundaries.
#   $1 = inference binary (native/volvoxai)
#   $2 = full binary      (native/volvoxai-full)
set -e
inf="$1"
full="$2"

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

if nm --defined-only "$inf" | grep -Eq \
    "(volvoxai_engine_train_|volvoxai_training_|volvoxai_autograd_|vx_dynamic_autograd|optimizer_state_for|volvoxai_(engine_)?ptq_)"; then
  echo "Inference binary unexpectedly contains training implementation."; exit 1
fi
if nm -g --defined-only "$inf" | grep -Eq \
    " (vx_model_create_trainer|vx_trainer_[A-Za-z0-9_]*|vx_model_create_ptq_plan|vx_ptq_plan_[A-Za-z0-9_]*)$"; then
  echo "Inference binary unexpectedly exposes a full-profile authoring API."; exit 1
fi
if ! nm -g --defined-only "$full" | grep -q " vx_model_create_trainer$"; then
  echo "Full binary is missing the public trainer lifecycle."; exit 1
fi
for symbol in vx_trainer_set_input vx_trainer_train_step vx_trainer_commit vx_trainer_rollback; do
  if ! nm -g --defined-only "$full" | grep -q " ${symbol}$"; then
    echo "Full binary is missing ${symbol}."; exit 1
  fi
done
for symbol in vx_model_create_ptq_plan vx_ptq_plan_retain vx_ptq_plan_release \
              vx_ptq_plan_close vx_ptq_plan_input_count vx_ptq_plan_input_info \
              vx_ptq_plan_calibrate vx_ptq_plan_info \
              vx_ptq_plan_tensor_parameters vx_ptq_plan_write_package; do
  if ! nm -g --defined-only "$full" | grep -q " ${symbol}$"; then
    echo "Full binary is missing ${symbol}."; exit 1
  fi
done
if nm --defined-only "$inf" | grep -Eq \
    "(volvoxai_cuda_training_ptx(_size)?|cuda_profile_route_tracking_active|cuda_training_(available|supports|preflight|begin|dispatch|sync|mark_failed|end))$"; then
  echo "Inference binary unexpectedly contains CUDA training code."; exit 1
fi
# CUDA-off profiles have neither PTX module. When the inference executable
# proves CUDA is enabled by retaining the referenced forward PTX, the matching
# full executable must also retain its separately embedded training module.
if nm -g --defined-only "$inf" | grep -q "volvoxai_cuda_ptx$"; then
  if ! nm -g --defined-only "$full" | grep -q "volvoxai_cuda_training_ptx$"; then
    echo "CUDA-enabled full binary is missing its training PTX module."; exit 1
  fi
fi
for binary in "$inf" "$full"; do
  if readelf -Ws "$binary" | awk '
      $5 == "GLOBAL" && $6 == "DEFAULT" &&
      $8 ~ /^(volvoxai_engine_|volvoxai_training_|volvoxai_autograd_|volvoxai_ptq_)/ {
        found = 1
      }
      END { exit found ? 0 : 1 }
    '; then
    echo "$binary exposes a private engine or training symbol."; exit 1
  fi
done
for binary in "$inf" "$full"; do
  if "$binary" --help | grep -Eq "^  (generate|classify|detect|ctc|seq2seq|chat|tinyreceipt) "; then
    echo "$binary unexpectedly exposes a model-specific command."; exit 1
  fi
  if nm -g --defined-only "$binary" | grep -Eq " (tiny_receipt_w8a8_run|kie_chat|kie_config_is_tiny_receipt|volvoxai_tokenizer_[A-Za-z0-9_]*)$"; then
    echo "$binary unexpectedly contains a model-specific session symbol."; exit 1
  fi
  if strings "$binary" | grep -Eq "volvoxai-tiny-receipt-vqa-w8a8-materialized-package-v1|tiny_receipt_kie|vocab\.bin|merges\.txt"; then
    echo "$binary unexpectedly contains model or vocabulary policy."; exit 1
  fi
done
echo 'Verified inference/full profile boundaries.'
