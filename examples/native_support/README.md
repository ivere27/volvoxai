# Native example support

This directory contains neutral helpers shared by native example applications.
It is not part of `native/volvoxai-lite` or `native/volvoxai`.

`image_io.[ch]` decodes PNG/JPEG files and writes an explicitly normalized
F32 tensor. Each consuming example remains responsible for choosing the input
shape and normalization policy.
