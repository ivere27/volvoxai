#include "runtime_lifecycle.h"

#include <limits.h>

static int vx_runtime_lifecycle_owner_kind_valid(uint32_t owner_kind) {
    return owner_kind == VX_RUNTIME_LIFECYCLE_OWNER_RUNTIME ||
        owner_kind == VX_RUNTIME_LIFECYCLE_OWNER_COMPILED_MODEL ||
        owner_kind == VX_RUNTIME_LIFECYCLE_OWNER_EXECUTION_CONTEXT;
}

static int vx_runtime_lifecycle_domain_empty(
        const VxRuntimeLifecycleDomainV1* domain) {
    return domain && domain->magic == 0u && domain->abi_version == 0u &&
        domain->identity_exhausted == 0u && domain->reserved == 0u &&
        domain->next_owner_identity == 0u &&
        domain->issued_owner_identities == 0u;
}

/* Spell out transactional record copies so a freestanding WASM build never
 * acquires an implicit libc memcpy import from aggregate assignment. */
static void vx_runtime_lifecycle_domain_copy(
        VxRuntimeLifecycleDomainV1* destination,
        const VxRuntimeLifecycleDomainV1* source) {
    destination->magic = source->magic;
    destination->abi_version = source->abi_version;
    destination->identity_exhausted = source->identity_exhausted;
    destination->reserved = source->reserved;
    destination->next_owner_identity = source->next_owner_identity;
    destination->issued_owner_identities = source->issued_owner_identities;
}

static uint32_t vx_runtime_lifecycle_owner_mode(uint32_t owner_kind) {
    return owner_kind == VX_RUNTIME_LIFECYCLE_OWNER_EXECUTION_CONTEXT
        ? VX_RUNTIME_LIFECYCLE_ADMISSION_FIFO
        : VX_RUNTIME_LIFECYCLE_ADMISSION_PARALLEL;
}

static int vx_runtime_lifecycle_state_empty(
        const VxRuntimeLifecycleStateV1* state) {
    return state && state->magic == 0u && state->abi_version == 0u &&
        state->owner_kind == 0u && state->admission_mode == 0u &&
        state->phase == 0u && state->close_state == 0u &&
        state->close_result == 0u && state->ticket_exhausted == 0u &&
        state->identity == 0u && state->parent_identity == 0u &&
        state->transition_id == 0u && state->next_ticket == 0u &&
        state->serving_ticket == 0u && state->active_ticket == 0u &&
        state->close_ticket == 0u && state->admitted_operations == 0u &&
        state->active_operations == 0u && state->child_count == 0u &&
        state->parent_pin_held == 0u && state->reserved == 0u;
}

static void vx_runtime_lifecycle_state_copy(
        VxRuntimeLifecycleStateV1* destination,
        const VxRuntimeLifecycleStateV1* source) {
    destination->magic = source->magic;
    destination->abi_version = source->abi_version;
    destination->owner_kind = source->owner_kind;
    destination->admission_mode = source->admission_mode;
    destination->phase = source->phase;
    destination->close_state = source->close_state;
    destination->close_result = source->close_result;
    destination->ticket_exhausted = source->ticket_exhausted;
    destination->identity = source->identity;
    destination->parent_identity = source->parent_identity;
    destination->transition_id = source->transition_id;
    destination->next_ticket = source->next_ticket;
    destination->serving_ticket = source->serving_ticket;
    destination->active_ticket = source->active_ticket;
    destination->close_ticket = source->close_ticket;
    destination->admitted_operations = source->admitted_operations;
    destination->active_operations = source->active_operations;
    destination->child_count = source->child_count;
    destination->parent_pin_held = source->parent_pin_held;
    destination->reserved = source->reserved;
}

static int vx_runtime_lifecycle_ticket_empty(
        const VxRuntimeLifecycleTicketV1* ticket) {
    return ticket && ticket->magic == 0u && ticket->abi_version == 0u &&
        ticket->operation == 0u && ticket->phase == 0u &&
        ticket->owner_identity == 0u && ticket->ticket == 0u &&
        ticket->integrity_tag == 0u;
}

/* This is an accidental-corruption tag, not an authentication primitive. */
static uint64_t vx_runtime_lifecycle_mix(uint64_t value) {
    value ^= value >> 30u;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27u;
    value *= UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31u;
    return value;
}

static uint64_t vx_runtime_lifecycle_ticket_tag(
        const VxRuntimeLifecycleTicketV1* ticket) {
    uint64_t value = UINT64_C(0x315452564C434658);
    value ^= vx_runtime_lifecycle_mix(ticket->owner_identity);
    value ^= vx_runtime_lifecycle_mix(ticket->ticket);
    value ^= (uint64_t)ticket->operation << 32u;
    value ^= (uint64_t)ticket->phase;
    return vx_runtime_lifecycle_mix(value);
}

static void vx_runtime_lifecycle_ticket_write(
        VxRuntimeLifecycleTicketV1* ticket,
        uint64_t owner_identity,
        uint64_t ticket_number,
        uint32_t operation,
        uint32_t phase) {
    ticket->magic = VX_RUNTIME_LIFECYCLE_TICKET_MAGIC;
    ticket->abi_version = VX_RUNTIME_LIFECYCLE_ABI_VERSION;
    ticket->operation = operation;
    ticket->phase = phase;
    ticket->owner_identity = owner_identity;
    ticket->ticket = ticket_number;
    ticket->integrity_tag = vx_runtime_lifecycle_ticket_tag(ticket);
}

static void vx_runtime_lifecycle_transition_advance(
        VxRuntimeLifecycleStateV1* state) {
    /* Diagnostic only.  Exhaustion must never strand an accepted operation,
     * child pin, or close transition. */
    if (state->transition_id != UINT64_MAX) state->transition_id++;
}

static int vx_runtime_lifecycle_parent_edge_valid(uint32_t parent_kind,
                                                   uint32_t child_kind) {
    return (parent_kind == VX_RUNTIME_LIFECYCLE_OWNER_RUNTIME &&
            child_kind == VX_RUNTIME_LIFECYCLE_OWNER_COMPILED_MODEL) ||
        (parent_kind == VX_RUNTIME_LIFECYCLE_OWNER_COMPILED_MODEL &&
         child_kind == VX_RUNTIME_LIFECYCLE_OWNER_EXECUTION_CONTEXT);
}

uint32_t vx_runtime_lifecycle_core_abi_version(void) {
    return VX_RUNTIME_LIFECYCLE_ABI_VERSION;
}

VxRuntimeLifecycleStatusV1 vx_runtime_lifecycle_domain_validate_v1(
        const VxRuntimeLifecycleDomainV1* domain) {
    if (!domain) return VX_RUNTIME_LIFECYCLE_INVALID_ARGUMENT;
    if (domain->magic != VX_RUNTIME_LIFECYCLE_DOMAIN_MAGIC ||
        domain->abi_version != VX_RUNTIME_LIFECYCLE_ABI_VERSION ||
        domain->identity_exhausted > 1u || domain->reserved != 0u ||
        domain->next_owner_identity == 0u)
        return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
    if (domain->identity_exhausted) {
        if (domain->next_owner_identity != UINT64_MAX ||
            domain->issued_owner_identities != UINT64_MAX)
            return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
    } else if (domain->issued_owner_identities == UINT64_MAX ||
               domain->next_owner_identity !=
                   domain->issued_owner_identities + 1u) {
        return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
    }
    return VX_RUNTIME_LIFECYCLE_OK;
}

VxRuntimeLifecycleStatusV1 vx_runtime_lifecycle_domain_init_v1(
        VxRuntimeLifecycleDomainV1* domain) {
    if (!domain || !vx_runtime_lifecycle_domain_empty(domain))
        return VX_RUNTIME_LIFECYCLE_INVALID_ARGUMENT;
    domain->magic = VX_RUNTIME_LIFECYCLE_DOMAIN_MAGIC;
    domain->abi_version = VX_RUNTIME_LIFECYCLE_ABI_VERSION;
    domain->identity_exhausted = 0u;
    domain->reserved = 0u;
    domain->next_owner_identity = 1u;
    domain->issued_owner_identities = 0u;
    return VX_RUNTIME_LIFECYCLE_OK;
}

static VxRuntimeLifecycleStatusV1 vx_runtime_lifecycle_domain_allocate(
        VxRuntimeLifecycleDomainV1* domain,
        uint64_t* identity) {
    uint64_t issued;
    if (!domain || !identity)
        return VX_RUNTIME_LIFECYCLE_INVALID_ARGUMENT;
    if (domain->identity_exhausted)
        return VX_RUNTIME_LIFECYCLE_OVERFLOW;
    issued = domain->next_owner_identity;
    if (issued == UINT64_MAX) {
        domain->identity_exhausted = 1u;
        domain->issued_owner_identities = UINT64_MAX;
    } else {
        domain->next_owner_identity++;
        domain->issued_owner_identities++;
    }
    *identity = issued;
    return VX_RUNTIME_LIFECYCLE_OK;
}

VxRuntimeLifecycleStatusV1 vx_runtime_lifecycle_state_validate_v1(
        const VxRuntimeLifecycleStateV1* state) {
    uint64_t fifo_outstanding;
    if (!state) return VX_RUNTIME_LIFECYCLE_INVALID_ARGUMENT;
    if (state->magic != VX_RUNTIME_LIFECYCLE_STATE_MAGIC ||
        state->abi_version != VX_RUNTIME_LIFECYCLE_ABI_VERSION ||
        !vx_runtime_lifecycle_owner_kind_valid(state->owner_kind) ||
        state->admission_mode !=
            vx_runtime_lifecycle_owner_mode(state->owner_kind) ||
        state->identity == 0u || state->transition_id == 0u ||
        state->next_ticket == 0u || state->ticket_exhausted > 1u ||
        state->parent_pin_held > 1u || state->reserved != 0u ||
        state->child_count > UINT64_MAX - state->identity ||
        (state->owner_kind == VX_RUNTIME_LIFECYCLE_OWNER_EXECUTION_CONTEXT &&
         state->child_count != 0u) ||
        state->active_operations > state->admitted_operations)
        return VX_RUNTIME_LIFECYCLE_INVALID_STATE;

    if (state->phase != VX_RUNTIME_LIFECYCLE_PHASE_OPEN &&
        state->phase != VX_RUNTIME_LIFECYCLE_PHASE_CLOSING &&
        state->phase != VX_RUNTIME_LIFECYCLE_PHASE_CLOSED)
        return VX_RUNTIME_LIFECYCLE_INVALID_STATE;

    if (state->owner_kind == VX_RUNTIME_LIFECYCLE_OWNER_RUNTIME) {
        if (state->parent_identity != 0u || state->parent_pin_held != 0u)
            return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
    } else if (state->parent_identity == 0u ||
               state->identity <= state->parent_identity ||
               (state->phase != VX_RUNTIME_LIFECYCLE_PHASE_CLOSED &&
                state->parent_pin_held != 1u)) {
        return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
    }

    if (state->phase == VX_RUNTIME_LIFECYCLE_PHASE_OPEN) {
        if (state->close_state != VX_RUNTIME_LIFECYCLE_CLOSE_NONE ||
            state->close_result != VX_RUNTIME_LIFECYCLE_CLOSE_RESULT_NONE ||
            state->close_ticket != 0u || state->ticket_exhausted)
            return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
    } else if (state->phase == VX_RUNTIME_LIFECYCLE_PHASE_CLOSING) {
        if ((state->close_state != VX_RUNTIME_LIFECYCLE_CLOSE_ADMITTED &&
             state->close_state != VX_RUNTIME_LIFECYCLE_CLOSE_ACTIVE) ||
            state->close_result != VX_RUNTIME_LIFECYCLE_CLOSE_RESULT_NONE ||
            state->close_ticket == 0u)
            return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
    } else {
        if (state->close_state != VX_RUNTIME_LIFECYCLE_CLOSE_COMPLETED ||
            (state->close_result !=
                 VX_RUNTIME_LIFECYCLE_CLOSE_RESULT_SUCCEEDED &&
             state->close_result != VX_RUNTIME_LIFECYCLE_CLOSE_RESULT_FAILED) ||
            state->close_ticket == 0u || state->admitted_operations != 0u ||
            state->active_operations != 0u || state->child_count != 0u ||
            state->active_ticket != 0u)
            return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
    }

    if (state->ticket_exhausted &&
        (state->next_ticket != UINT64_MAX ||
         state->close_ticket != UINT64_MAX ||
         state->phase == VX_RUNTIME_LIFECYCLE_PHASE_OPEN))
        return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
    if (!state->ticket_exhausted && state->close_ticket != 0u &&
        (state->close_ticket == UINT64_MAX ||
         state->next_ticket != state->close_ticket + 1u))
        return VX_RUNTIME_LIFECYCLE_INVALID_STATE;

    if (state->admission_mode == VX_RUNTIME_LIFECYCLE_ADMISSION_PARALLEL) {
        if (state->serving_ticket != 0u || state->active_ticket != 0u)
            return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
        if ((state->phase == VX_RUNTIME_LIFECYCLE_PHASE_OPEN &&
             state->admitted_operations > state->next_ticket - 1u) ||
            (state->phase == VX_RUNTIME_LIFECYCLE_PHASE_CLOSING &&
             state->admitted_operations > state->close_ticket - 1u))
            return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
        if (state->close_state == VX_RUNTIME_LIFECYCLE_CLOSE_ACTIVE &&
            (state->admitted_operations != 0u ||
             state->active_operations != 0u || state->child_count != 0u))
            return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
        return VX_RUNTIME_LIFECYCLE_OK;
    }

    if (state->serving_ticket == 0u || state->active_operations > 1u)
        return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
    if (state->phase == VX_RUNTIME_LIFECYCLE_PHASE_OPEN) {
        if (state->serving_ticket > state->next_ticket)
            return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
        fifo_outstanding = state->next_ticket - state->serving_ticket;
    } else if (state->phase == VX_RUNTIME_LIFECYCLE_PHASE_CLOSING) {
        if (state->serving_ticket > state->close_ticket)
            return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
        fifo_outstanding = state->close_ticket - state->serving_ticket;
    } else {
        fifo_outstanding = 0u;
        if ((state->ticket_exhausted &&
             state->serving_ticket != UINT64_MAX) ||
            (!state->ticket_exhausted &&
             state->serving_ticket != state->close_ticket + 1u))
            return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
    }
    if (fifo_outstanding != state->admitted_operations)
        return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
    if (state->close_state == VX_RUNTIME_LIFECYCLE_CLOSE_ACTIVE) {
        if (state->active_ticket != state->close_ticket ||
            state->admitted_operations != 0u ||
            state->active_operations != 0u || state->child_count != 0u)
            return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
    } else if (state->active_operations == 1u) {
        if (state->active_ticket != state->serving_ticket)
            return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
    } else if (state->active_ticket != 0u) {
        return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
    }
    return VX_RUNTIME_LIFECYCLE_OK;
}

VxRuntimeLifecycleStatusV1 vx_runtime_lifecycle_ticket_validate_v1(
        const VxRuntimeLifecycleTicketV1* ticket) {
    if (!ticket) return VX_RUNTIME_LIFECYCLE_INVALID_ARGUMENT;
    if (ticket->magic != VX_RUNTIME_LIFECYCLE_TICKET_MAGIC ||
        ticket->abi_version != VX_RUNTIME_LIFECYCLE_ABI_VERSION ||
        ticket->operation == VX_RUNTIME_LIFECYCLE_OPERATION_NONE ||
        ticket->phase < VX_RUNTIME_LIFECYCLE_TICKET_ADMITTED ||
        ticket->phase > VX_RUNTIME_LIFECYCLE_TICKET_COMPLETED ||
        ticket->owner_identity == 0u || ticket->ticket == 0u ||
        ticket->integrity_tag != vx_runtime_lifecycle_ticket_tag(ticket))
        return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
    return VX_RUNTIME_LIFECYCLE_OK;
}

static VxRuntimeLifecycleStatusV1 vx_runtime_lifecycle_state_init(
        VxRuntimeLifecycleStateV1* state,
        uint64_t identity,
        uint64_t parent_identity,
        uint32_t parent_pin_held,
        uint32_t owner_kind) {
    if (!state || identity == 0u ||
        !vx_runtime_lifecycle_owner_kind_valid(owner_kind) ||
        !vx_runtime_lifecycle_state_empty(state) ||
        parent_pin_held > 1u ||
        (parent_pin_held && parent_identity == 0u) ||
        (!parent_pin_held && parent_identity != 0u))
        return VX_RUNTIME_LIFECYCLE_INVALID_ARGUMENT;
    state->magic = VX_RUNTIME_LIFECYCLE_STATE_MAGIC;
    state->abi_version = VX_RUNTIME_LIFECYCLE_ABI_VERSION;
    state->owner_kind = owner_kind;
    state->admission_mode = vx_runtime_lifecycle_owner_mode(owner_kind);
    state->phase = VX_RUNTIME_LIFECYCLE_PHASE_OPEN;
    state->close_state = VX_RUNTIME_LIFECYCLE_CLOSE_NONE;
    state->close_result = VX_RUNTIME_LIFECYCLE_CLOSE_RESULT_NONE;
    state->ticket_exhausted = 0u;
    state->identity = identity;
    state->parent_identity = parent_identity;
    state->transition_id = 1u;
    state->next_ticket = 1u;
    state->serving_ticket = state->admission_mode ==
        VX_RUNTIME_LIFECYCLE_ADMISSION_FIFO ? 1u : 0u;
    state->active_ticket = 0u;
    state->close_ticket = 0u;
    state->admitted_operations = 0u;
    state->active_operations = 0u;
    state->child_count = 0u;
    state->parent_pin_held = parent_pin_held;
    state->reserved = 0u;
    return VX_RUNTIME_LIFECYCLE_OK;
}

VxRuntimeLifecycleStatusV1 vx_runtime_lifecycle_state_init_root_v1(
        VxRuntimeLifecycleDomainV1* domain,
        VxRuntimeLifecycleStateV1* state) {
    VxRuntimeLifecycleDomainV1 candidate_domain;
    VxRuntimeLifecycleStateV1 candidate_state;
    VxRuntimeLifecycleStatusV1 status =
        vx_runtime_lifecycle_domain_validate_v1(domain);
    uint64_t identity = 0u;
    if (status != VX_RUNTIME_LIFECYCLE_OK) return status;
    if (!state || !vx_runtime_lifecycle_state_empty(state))
        return VX_RUNTIME_LIFECYCLE_INVALID_ARGUMENT;
    vx_runtime_lifecycle_domain_copy(&candidate_domain, domain);
    vx_runtime_lifecycle_state_copy(&candidate_state, state);
    status = vx_runtime_lifecycle_domain_allocate(
        &candidate_domain, &identity);
    if (status != VX_RUNTIME_LIFECYCLE_OK) return status;
    status = vx_runtime_lifecycle_state_init(
        &candidate_state, identity, 0u, 0u,
        VX_RUNTIME_LIFECYCLE_OWNER_RUNTIME);
    if (status != VX_RUNTIME_LIFECYCLE_OK) return status;
    if (vx_runtime_lifecycle_domain_validate_v1(&candidate_domain) !=
            VX_RUNTIME_LIFECYCLE_OK ||
        vx_runtime_lifecycle_state_validate_v1(&candidate_state) !=
            VX_RUNTIME_LIFECYCLE_OK)
        return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
    vx_runtime_lifecycle_domain_copy(domain, &candidate_domain);
    vx_runtime_lifecycle_state_copy(state, &candidate_state);
    return VX_RUNTIME_LIFECYCLE_OK;
}

VxRuntimeLifecycleStatusV1 vx_runtime_lifecycle_state_init_child_v1(
        VxRuntimeLifecycleDomainV1* domain,
        VxRuntimeLifecycleStateV1* parent,
        VxRuntimeLifecycleStateV1* child,
        VxRuntimeLifecycleOwnerKindV1 child_kind) {
    VxRuntimeLifecycleDomainV1 candidate_domain;
    VxRuntimeLifecycleStateV1 candidate_parent;
    VxRuntimeLifecycleStateV1 candidate_child;
    VxRuntimeLifecycleStatusV1 status =
        vx_runtime_lifecycle_domain_validate_v1(domain);
    uint64_t child_identity = 0u;
    if (status != VX_RUNTIME_LIFECYCLE_OK) return status;
    status =
        vx_runtime_lifecycle_state_validate_v1(parent);
    if (status != VX_RUNTIME_LIFECYCLE_OK) return status;
    /* A parent created by this domain must already be in its issued prefix.
     * This catches future/corrupted identities before allocating or pinning. */
    if (parent->identity > domain->issued_owner_identities)
        return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
    if (!child || !vx_runtime_lifecycle_state_empty(child) ||
        !vx_runtime_lifecycle_parent_edge_valid(parent->owner_kind,
                                                child_kind))
        return VX_RUNTIME_LIFECYCLE_INVALID_ARGUMENT;
    if (parent->phase != VX_RUNTIME_LIFECYCLE_PHASE_OPEN)
        return VX_RUNTIME_LIFECYCLE_HANDLE_DISPOSED;
    if (parent->child_count == UINT64_MAX)
        return VX_RUNTIME_LIFECYCLE_OVERFLOW;
    vx_runtime_lifecycle_domain_copy(&candidate_domain, domain);
    vx_runtime_lifecycle_state_copy(&candidate_parent, parent);
    vx_runtime_lifecycle_state_copy(&candidate_child, child);
    status = vx_runtime_lifecycle_domain_allocate(
        &candidate_domain, &child_identity);
    if (status != VX_RUNTIME_LIFECYCLE_OK) return status;
    if (child_identity <= candidate_parent.identity)
        return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
    status = vx_runtime_lifecycle_state_init(
        &candidate_child, child_identity, candidate_parent.identity, 1u,
        child_kind);
    if (status != VX_RUNTIME_LIFECYCLE_OK) return status;
    candidate_parent.child_count++;
    vx_runtime_lifecycle_transition_advance(&candidate_parent);
    if (vx_runtime_lifecycle_domain_validate_v1(&candidate_domain) !=
            VX_RUNTIME_LIFECYCLE_OK ||
        vx_runtime_lifecycle_state_validate_v1(&candidate_parent) !=
            VX_RUNTIME_LIFECYCLE_OK ||
        vx_runtime_lifecycle_state_validate_v1(&candidate_child) !=
            VX_RUNTIME_LIFECYCLE_OK)
        return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
    vx_runtime_lifecycle_domain_copy(domain, &candidate_domain);
    vx_runtime_lifecycle_state_copy(parent, &candidate_parent);
    vx_runtime_lifecycle_state_copy(child, &candidate_child);
    return VX_RUNTIME_LIFECYCLE_OK;
}

VxRuntimeLifecycleStatusV1 vx_runtime_lifecycle_operation_admit_v1(
        VxRuntimeLifecycleStateV1* state,
        VxRuntimeLifecycleOperationV1 operation,
        VxRuntimeLifecycleTicketV1* ticket) {
    VxRuntimeLifecycleStatusV1 status =
        vx_runtime_lifecycle_state_validate_v1(state);
    uint64_t number;
    if (status != VX_RUNTIME_LIFECYCLE_OK) return status;
    if (!ticket || !vx_runtime_lifecycle_ticket_empty(ticket) ||
        operation == VX_RUNTIME_LIFECYCLE_OPERATION_NONE ||
        operation == VX_RUNTIME_LIFECYCLE_OPERATION_CLOSE)
        return VX_RUNTIME_LIFECYCLE_INVALID_ARGUMENT;
    if (state->phase != VX_RUNTIME_LIFECYCLE_PHASE_OPEN)
        return VX_RUNTIME_LIFECYCLE_HANDLE_DISPOSED;
    /* The final numeric ticket is reserved for close, so overflow can never
     * make an otherwise valid owner impossible to close. */
    if (state->next_ticket == UINT64_MAX ||
        state->admitted_operations == UINT64_MAX)
        return VX_RUNTIME_LIFECYCLE_OVERFLOW;
    number = state->next_ticket;
    state->next_ticket++;
    state->admitted_operations++;
    vx_runtime_lifecycle_transition_advance(state);
    vx_runtime_lifecycle_ticket_write(
        ticket, state->identity, number, operation,
        VX_RUNTIME_LIFECYCLE_TICKET_ADMITTED);
    return VX_RUNTIME_LIFECYCLE_OK;
}

static VxRuntimeLifecycleStatusV1 vx_runtime_lifecycle_ticket_matches(
        const VxRuntimeLifecycleStateV1* state,
        const VxRuntimeLifecycleTicketV1* ticket,
        int close_ticket) {
    VxRuntimeLifecycleStatusV1 status =
        vx_runtime_lifecycle_ticket_validate_v1(ticket);
    if (status != VX_RUNTIME_LIFECYCLE_OK) return status;
    if (ticket->owner_identity != state->identity)
        return VX_RUNTIME_LIFECYCLE_STALE;
    if (close_ticket) {
        if (ticket->operation != VX_RUNTIME_LIFECYCLE_OPERATION_CLOSE ||
            ticket->ticket != state->close_ticket)
            return VX_RUNTIME_LIFECYCLE_STALE;
    } else {
        if (ticket->operation == VX_RUNTIME_LIFECYCLE_OPERATION_CLOSE ||
            (state->close_ticket != 0u &&
             ticket->ticket >= state->close_ticket) ||
            ticket->ticket >= state->next_ticket)
            return VX_RUNTIME_LIFECYCLE_STALE;
    }
    return VX_RUNTIME_LIFECYCLE_OK;
}

VxRuntimeLifecycleStatusV1 vx_runtime_lifecycle_operation_begin_v1(
        VxRuntimeLifecycleStateV1* state,
        VxRuntimeLifecycleTicketV1* ticket) {
    VxRuntimeLifecycleStatusV1 status =
        vx_runtime_lifecycle_state_validate_v1(state);
    if (status != VX_RUNTIME_LIFECYCLE_OK) return status;
    status = vx_runtime_lifecycle_ticket_matches(state, ticket, 0);
    if (status != VX_RUNTIME_LIFECYCLE_OK) return status;
    if (ticket->phase == VX_RUNTIME_LIFECYCLE_TICKET_COMPLETED ||
        ticket->phase == VX_RUNTIME_LIFECYCLE_TICKET_ACTIVE)
        return VX_RUNTIME_LIFECYCLE_DUPLICATE;
    if (state->phase == VX_RUNTIME_LIFECYCLE_PHASE_CLOSED ||
        state->admitted_operations == 0u)
        return VX_RUNTIME_LIFECYCLE_STALE;
    if (state->admission_mode == VX_RUNTIME_LIFECYCLE_ADMISSION_FIFO) {
        if (ticket->ticket < state->serving_ticket)
            return VX_RUNTIME_LIFECYCLE_STALE;
        if (ticket->ticket != state->serving_ticket ||
            state->active_ticket != 0u)
            return VX_RUNTIME_LIFECYCLE_WAIT;
    }
    if (state->active_operations >= state->admitted_operations)
        return VX_RUNTIME_LIFECYCLE_STALE;
    state->active_operations++;
    if (state->admission_mode == VX_RUNTIME_LIFECYCLE_ADMISSION_FIFO)
        state->active_ticket = ticket->ticket;
    vx_runtime_lifecycle_transition_advance(state);
    vx_runtime_lifecycle_ticket_write(
        ticket, ticket->owner_identity, ticket->ticket, ticket->operation,
        VX_RUNTIME_LIFECYCLE_TICKET_ACTIVE);
    return VX_RUNTIME_LIFECYCLE_OK;
}

VxRuntimeLifecycleStatusV1 vx_runtime_lifecycle_operation_complete_v1(
        VxRuntimeLifecycleStateV1* state,
        VxRuntimeLifecycleTicketV1* ticket) {
    VxRuntimeLifecycleStatusV1 status =
        vx_runtime_lifecycle_state_validate_v1(state);
    if (status != VX_RUNTIME_LIFECYCLE_OK) return status;
    status = vx_runtime_lifecycle_ticket_matches(state, ticket, 0);
    if (status != VX_RUNTIME_LIFECYCLE_OK) return status;
    if (ticket->phase == VX_RUNTIME_LIFECYCLE_TICKET_COMPLETED)
        return VX_RUNTIME_LIFECYCLE_DUPLICATE;
    if (ticket->phase != VX_RUNTIME_LIFECYCLE_TICKET_ACTIVE)
        return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
    if (state->phase == VX_RUNTIME_LIFECYCLE_PHASE_CLOSED ||
        state->active_operations == 0u ||
        state->admitted_operations == 0u)
        return VX_RUNTIME_LIFECYCLE_STALE;
    if (state->admission_mode == VX_RUNTIME_LIFECYCLE_ADMISSION_FIFO &&
        (state->active_ticket != ticket->ticket ||
         state->serving_ticket != ticket->ticket))
        return VX_RUNTIME_LIFECYCLE_STALE;
    state->active_operations--;
    state->admitted_operations--;
    if (state->admission_mode == VX_RUNTIME_LIFECYCLE_ADMISSION_FIFO) {
        state->active_ticket = 0u;
        state->serving_ticket++;
    }
    vx_runtime_lifecycle_transition_advance(state);
    vx_runtime_lifecycle_ticket_write(
        ticket, ticket->owner_identity, ticket->ticket, ticket->operation,
        VX_RUNTIME_LIFECYCLE_TICKET_COMPLETED);
    return VX_RUNTIME_LIFECYCLE_OK;
}

static void vx_runtime_lifecycle_close_ticket_project(
        const VxRuntimeLifecycleStateV1* state,
        VxRuntimeLifecycleTicketV1* ticket) {
    uint32_t phase = VX_RUNTIME_LIFECYCLE_TICKET_ADMITTED;
    if (state->close_state == VX_RUNTIME_LIFECYCLE_CLOSE_ACTIVE)
        phase = VX_RUNTIME_LIFECYCLE_TICKET_ACTIVE;
    else if (state->close_state == VX_RUNTIME_LIFECYCLE_CLOSE_COMPLETED)
        phase = VX_RUNTIME_LIFECYCLE_TICKET_COMPLETED;
    vx_runtime_lifecycle_ticket_write(
        ticket, state->identity, state->close_ticket,
        VX_RUNTIME_LIFECYCLE_OPERATION_CLOSE, phase);
}

VxRuntimeLifecycleStatusV1 vx_runtime_lifecycle_close_request_v1(
        VxRuntimeLifecycleStateV1* state,
        VxRuntimeLifecycleTicketV1* close_ticket) {
    VxRuntimeLifecycleStatusV1 status =
        vx_runtime_lifecycle_state_validate_v1(state);
    if (status != VX_RUNTIME_LIFECYCLE_OK) return status;
    if (!close_ticket) return VX_RUNTIME_LIFECYCLE_INVALID_ARGUMENT;
    if (!vx_runtime_lifecycle_ticket_empty(close_ticket)) {
        status = vx_runtime_lifecycle_ticket_matches(state, close_ticket, 1);
        if (status != VX_RUNTIME_LIFECYCLE_OK) return status;
    }
    if (state->phase == VX_RUNTIME_LIFECYCLE_PHASE_CLOSING) {
        vx_runtime_lifecycle_close_ticket_project(state, close_ticket);
        return VX_RUNTIME_LIFECYCLE_ALREADY_CLOSING;
    }
    if (state->phase == VX_RUNTIME_LIFECYCLE_PHASE_CLOSED) {
        vx_runtime_lifecycle_close_ticket_project(state, close_ticket);
        return VX_RUNTIME_LIFECYCLE_ALREADY_CLOSED;
    }
    if (!vx_runtime_lifecycle_ticket_empty(close_ticket))
        return VX_RUNTIME_LIFECYCLE_STALE;
    state->phase = VX_RUNTIME_LIFECYCLE_PHASE_CLOSING;
    state->close_state = VX_RUNTIME_LIFECYCLE_CLOSE_ADMITTED;
    state->close_ticket = state->next_ticket;
    if (state->next_ticket == UINT64_MAX) {
        state->ticket_exhausted = 1u;
    } else {
        state->next_ticket++;
    }
    vx_runtime_lifecycle_transition_advance(state);
    vx_runtime_lifecycle_close_ticket_project(state, close_ticket);
    return VX_RUNTIME_LIFECYCLE_OK;
}

VxRuntimeLifecycleStatusV1 vx_runtime_lifecycle_close_readiness_v1(
        const VxRuntimeLifecycleStateV1* state,
        VxRuntimeLifecycleCloseReadinessV1* readiness) {
    VxRuntimeLifecycleStatusV1 status =
        vx_runtime_lifecycle_state_validate_v1(state);
    if (status != VX_RUNTIME_LIFECYCLE_OK) return status;
    if (!readiness) return VX_RUNTIME_LIFECYCLE_INVALID_ARGUMENT;
    if (state->phase == VX_RUNTIME_LIFECYCLE_PHASE_OPEN) {
        *readiness = VX_RUNTIME_LIFECYCLE_CLOSE_NOT_REQUESTED;
    } else if (state->phase == VX_RUNTIME_LIFECYCLE_PHASE_CLOSED) {
        *readiness = VX_RUNTIME_LIFECYCLE_CLOSE_FINISHED;
    } else if (state->admitted_operations != 0u ||
               state->active_operations != 0u) {
        *readiness = VX_RUNTIME_LIFECYCLE_CLOSE_DRAINING_WORK;
    } else if (state->child_count != 0u) {
        *readiness = VX_RUNTIME_LIFECYCLE_CLOSE_DRAINING_CHILDREN;
    } else if (state->close_state == VX_RUNTIME_LIFECYCLE_CLOSE_ACTIVE) {
        *readiness = VX_RUNTIME_LIFECYCLE_CLOSE_HOST_ACTIVE;
    } else {
        *readiness = VX_RUNTIME_LIFECYCLE_CLOSE_READY;
    }
    return VX_RUNTIME_LIFECYCLE_OK;
}

VxRuntimeLifecycleStatusV1 vx_runtime_lifecycle_close_begin_v1(
        VxRuntimeLifecycleStateV1* state,
        VxRuntimeLifecycleTicketV1* close_ticket) {
    VxRuntimeLifecycleCloseReadinessV1 readiness;
    VxRuntimeLifecycleStatusV1 status =
        vx_runtime_lifecycle_state_validate_v1(state);
    if (status != VX_RUNTIME_LIFECYCLE_OK) return status;
    status = vx_runtime_lifecycle_ticket_matches(state, close_ticket, 1);
    if (status != VX_RUNTIME_LIFECYCLE_OK) return status;
    if (state->phase == VX_RUNTIME_LIFECYCLE_PHASE_CLOSED)
        return VX_RUNTIME_LIFECYCLE_ALREADY_CLOSED;
    if (state->close_state == VX_RUNTIME_LIFECYCLE_CLOSE_ACTIVE ||
        close_ticket->phase == VX_RUNTIME_LIFECYCLE_TICKET_ACTIVE ||
        close_ticket->phase == VX_RUNTIME_LIFECYCLE_TICKET_COMPLETED)
        return VX_RUNTIME_LIFECYCLE_DUPLICATE;
    status = vx_runtime_lifecycle_close_readiness_v1(state, &readiness);
    if (status != VX_RUNTIME_LIFECYCLE_OK) return status;
    if (readiness != VX_RUNTIME_LIFECYCLE_CLOSE_READY)
        return VX_RUNTIME_LIFECYCLE_WAIT;
    if (state->admission_mode == VX_RUNTIME_LIFECYCLE_ADMISSION_FIFO &&
        (state->serving_ticket != state->close_ticket ||
         state->active_ticket != 0u))
        return VX_RUNTIME_LIFECYCLE_WAIT;
    state->close_state = VX_RUNTIME_LIFECYCLE_CLOSE_ACTIVE;
    if (state->admission_mode == VX_RUNTIME_LIFECYCLE_ADMISSION_FIFO)
        state->active_ticket = state->close_ticket;
    vx_runtime_lifecycle_transition_advance(state);
    vx_runtime_lifecycle_ticket_write(
        close_ticket, close_ticket->owner_identity, close_ticket->ticket,
        VX_RUNTIME_LIFECYCLE_OPERATION_CLOSE,
        VX_RUNTIME_LIFECYCLE_TICKET_ACTIVE);
    return VX_RUNTIME_LIFECYCLE_OK;
}

VxRuntimeLifecycleStatusV1 vx_runtime_lifecycle_close_complete_v1(
        VxRuntimeLifecycleStateV1* state,
        VxRuntimeLifecycleTicketV1* close_ticket,
        VxRuntimeLifecycleCloseResultV1 result) {
    VxRuntimeLifecycleStatusV1 status =
        vx_runtime_lifecycle_state_validate_v1(state);
    if (status != VX_RUNTIME_LIFECYCLE_OK) return status;
    status = vx_runtime_lifecycle_ticket_matches(state, close_ticket, 1);
    if (status != VX_RUNTIME_LIFECYCLE_OK) return status;
    if (state->phase == VX_RUNTIME_LIFECYCLE_PHASE_CLOSED)
        return VX_RUNTIME_LIFECYCLE_ALREADY_CLOSED;
    if (result != VX_RUNTIME_LIFECYCLE_CLOSE_RESULT_SUCCEEDED &&
        result != VX_RUNTIME_LIFECYCLE_CLOSE_RESULT_FAILED)
        return VX_RUNTIME_LIFECYCLE_INVALID_ARGUMENT;
    if (state->close_state != VX_RUNTIME_LIFECYCLE_CLOSE_ACTIVE ||
        close_ticket->phase != VX_RUNTIME_LIFECYCLE_TICKET_ACTIVE)
        return state->close_state == VX_RUNTIME_LIFECYCLE_CLOSE_COMPLETED
            ? VX_RUNTIME_LIFECYCLE_DUPLICATE
            : VX_RUNTIME_LIFECYCLE_INVALID_STATE;
    if (state->admitted_operations != 0u ||
        state->active_operations != 0u || state->child_count != 0u ||
        (state->admission_mode == VX_RUNTIME_LIFECYCLE_ADMISSION_FIFO &&
         (state->serving_ticket != state->close_ticket ||
          state->active_ticket != state->close_ticket)))
        return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
    state->phase = VX_RUNTIME_LIFECYCLE_PHASE_CLOSED;
    state->close_state = VX_RUNTIME_LIFECYCLE_CLOSE_COMPLETED;
    state->close_result = result;
    if (state->admission_mode == VX_RUNTIME_LIFECYCLE_ADMISSION_FIFO) {
        state->active_ticket = 0u;
        if (state->close_ticket != UINT64_MAX)
            state->serving_ticket++;
    }
    vx_runtime_lifecycle_transition_advance(state);
    vx_runtime_lifecycle_ticket_write(
        close_ticket, close_ticket->owner_identity, close_ticket->ticket,
        VX_RUNTIME_LIFECYCLE_OPERATION_CLOSE,
        VX_RUNTIME_LIFECYCLE_TICKET_COMPLETED);
    return VX_RUNTIME_LIFECYCLE_OK;
}

VxRuntimeLifecycleStatusV1 vx_runtime_lifecycle_parent_pin_release_v1(
        VxRuntimeLifecycleStateV1* parent,
        VxRuntimeLifecycleStateV1* child) {
    VxRuntimeLifecycleStatusV1 status =
        vx_runtime_lifecycle_state_validate_v1(parent);
    if (status != VX_RUNTIME_LIFECYCLE_OK) return status;
    status = vx_runtime_lifecycle_state_validate_v1(child);
    if (status != VX_RUNTIME_LIFECYCLE_OK) return status;
    if (child->parent_identity != parent->identity)
        return VX_RUNTIME_LIFECYCLE_PARENT_MISMATCH;
    if (!child->parent_pin_held)
        return VX_RUNTIME_LIFECYCLE_DUPLICATE;
    if (!vx_runtime_lifecycle_parent_edge_valid(parent->owner_kind,
                                                child->owner_kind))
        return VX_RUNTIME_LIFECYCLE_PARENT_MISMATCH;
    if (child->phase != VX_RUNTIME_LIFECYCLE_PHASE_CLOSED)
        return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
    if (parent->phase == VX_RUNTIME_LIFECYCLE_PHASE_CLOSED ||
        parent->child_count == 0u)
        return VX_RUNTIME_LIFECYCLE_INVALID_STATE;
    parent->child_count--;
    child->parent_pin_held = 0u;
    vx_runtime_lifecycle_transition_advance(parent);
    vx_runtime_lifecycle_transition_advance(child);
    return VX_RUNTIME_LIFECYCLE_OK;
}
