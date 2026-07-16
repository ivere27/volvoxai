#include "tiny_receipt_w8a8.h"

/* The native kernels use this buffer as their standalone bump-allocator base. */
int main(int argc, char** argv) {
    return tiny_receipt_w8a8_run(argc, argv);
}
