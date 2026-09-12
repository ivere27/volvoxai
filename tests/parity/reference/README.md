`Tokenizer.ts` is the unchanged test oracle from
`99dfd8f4bcc66b30da6872852b978a5ff10d22d7:ts/core/Tokenizer.ts`.
It is only imported by migration tests; no release entry imports this directory.
Changes to the C implementation must not change this oracle.

`Initializers.ts` is likewise unchanged from that commit's
`ts/training/Initializers.ts`. Migration tests compare its F32 values with the
generated `VxTrainingService.InitializeTensor` API, including numeric and UTF-16
text seeds. Its TypeScript type-only import is erased by the test bundler.
