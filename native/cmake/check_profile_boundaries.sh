#!/bin/bash
# Assert the inference/full release binaries keep their capability boundaries
# (ported verbatim from the `test_native` recipe in the root Makefile).
#   $1 = inference binary (native/volvoxai)
#   $2 = full binary      (native/volvoxai-full)
set -e
inf="$1"
full="$2"

if "$inf" --help | grep -q "^  train "; then
  echo "Inference CLI unexpectedly exposes training."; exit 1
fi
"$full" train --help | grep -q "^Usage: .* train "
if nm -g --defined-only "$inf" | grep -q "volvoxai_engine_train_step"; then
  echo "Inference binary unexpectedly exports training."; exit 1
fi
nm -g --defined-only "$full" | grep -q "volvoxai_engine_train_step"
if nm -g --defined-only "$inf" | grep -Eq "volvoxai_(engine_)?ptq_"; then
  echo "Inference binary unexpectedly exports PTQ authoring."; exit 1
fi
nm -g --defined-only "$full" | grep -q "volvoxai_ptq_observer_observe_f32"
nm -g --defined-only "$full" | grep -q "volvoxai_engine_ptq_materialize_weight_i8"
nm -g --defined-only "$full" | grep -q "volvoxai_ptq_plan_write_package"
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
