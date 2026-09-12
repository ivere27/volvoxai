#include "generated/kernel_registry.h"

#include <stdio.h>

const VxKernelRegistration* kernel_registry_peer_find(
    VxBackendKind backend, VxOperatorKind operator_kind);
const VxGeneratedShapeContractRoute* kernel_registry_peer_shape(
    VxOperatorKind operator_kind);

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        return 1; \
    } \
} while (0)

int main(void) {
    CHECK(vx_kernel_registration_count > 0);
    CHECK(vx_generated_operator_kind_count == vx_generated_shape_contract_route_count);
    for (size_t i = 0; i < vx_kernel_registration_count; ++i) {
        const VxKernelRegistration* row = &vx_kernel_registrations[i];
        CHECK(vx_generated_operator_kind_valid(row->operator_kind));
        CHECK(vx_kernel_registry_find(row->backend, row->operator_kind) == row);
        CHECK(kernel_registry_peer_find(row->backend, row->operator_kind) == row);
        const VxGeneratedShapeContractRoute* shape =
            vx_kernel_shape_contract_find(row->operator_kind);
        CHECK(shape && shape->operator_kind == row->operator_kind);
        CHECK(kernel_registry_peer_shape(row->operator_kind) == shape);
    }
    for (size_t i = 0; i < vx_generated_shape_contract_route_count; ++i) {
        const VxGeneratedShapeContractRoute* row = &vx_generated_shape_contract_routes[i];
        CHECK(row->operator_kind == vx_generated_operator_kinds[i]);
        CHECK(kernel_registry_peer_shape(row->operator_kind) == row);
        CHECK(vx_operator_kind_from_name(row->operator_name) == row->operator_kind);
        CHECK(!strcmp(vx_operator_kind_name(row->operator_kind), row->operator_name));
    }
    CHECK(!kernel_registry_peer_find(VX_BACKEND_KIND_UNSPECIFIED, VX_OP_MATMUL));
    CHECK(!kernel_registry_peer_shape(VX_OP_UNSPECIFIED));
    CHECK(!vx_generated_operator_kind_valid(-1));
    CHECK(vx_operator_kind_from_name(NULL) == VX_OP_UNSPECIFIED);
    CHECK(vx_operator_kind_from_name("") == VX_OP_UNSPECIFIED);
    CHECK(vx_operator_kind_from_name("linear") == VX_OP_UNSPECIFIED);
    CHECK(vx_operator_kind_from_name("Linear ") == VX_OP_UNSPECIFIED);
    CHECK(vx_operator_kind_from_name("Swish") == VX_OP_UNSPECIFIED);
    CHECK(vx_operator_kind_from_name("LinearWithAnUnknownSuffix") == VX_OP_UNSPECIFIED);
    CHECK(!vx_operator_kind_name(VX_OP_UNSPECIFIED));
    CHECK(!vx_operator_kind_name(-1));
    CHECK(!vx_operator_kind_name(1008)); /* gap in the proto enum */
    /* The wire boundary consumes exact byte views, including embedded NULs.
     * A length-limited name is valid; its suffix must never be ignored. */
    CHECK(vx_operator_kind_from_bytes("Linear!", 6) == VX_OP_LINEAR);
    CHECK(vx_operator_kind_from_bytes("Linear\0suffix", 13) == VX_OP_UNSPECIFIED);
    CHECK(vx_port_kind_from_bytes("input!", 5) == VX_PORT_INPUT);
    CHECK(vx_port_kind_from_bytes("input\0suffix", 12) == VX_PORT_UNSPECIFIED);
    CHECK(vx_port_kind_from_name("A") == VX_PORT_SSM_A);
    CHECK(vx_port_kind_from_name("a") == VX_PORT_A);
    CHECK(vx_port_kind_from_name("input42") == VX_PORT_UNSPECIFIED);
    CHECK(vx_backend_kind_from_bytes("webgpu!", 6) == VX_BACKEND_KIND_WEBGPU);
    CHECK(vx_backend_kind_from_bytes("webgpu\0suffix", 13) == VX_BACKEND_KIND_UNSPECIFIED);
    CHECK(vx_backend_kind_from_name("custom-provider") == VX_BACKEND_KIND_UNSPECIFIED);
    puts("Shared kernel registry identity and lookup invariants passed");
    return 0;
}
