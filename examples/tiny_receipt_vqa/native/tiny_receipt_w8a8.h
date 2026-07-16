#ifndef VOLVOXAI_TINY_RECEIPT_W8A8_H
#define VOLVOXAI_TINY_RECEIPT_W8A8_H

/*
 * Dedicated host session for a materialized TinyReceiptVQA W8A8 package.
 *
 * The package has an I32-token / QArgMax ABI and runs a separate hard-router
 * graph before an explicit adapter-family graph.
 */
int tiny_receipt_w8a8_run(int argc, char** argv);

#endif
