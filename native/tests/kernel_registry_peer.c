#include "generated/kernel_registry.h"

/* Lookup from a separate translation unit must return the same shared row. */
const VxKernelRegistration* kernel_registry_peer_find(
        VxBackendKind backend, VxOperatorKind operator_kind) {
    return vx_kernel_registry_find(backend, operator_kind);
}

const VxGeneratedShapeContractRoute* kernel_registry_peer_shape(
        VxOperatorKind operator_kind) {
    return vx_kernel_shape_contract_find(operator_kind);
}
