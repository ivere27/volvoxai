#include "scheduler_policy.h"

#include <limits.h>

#define VX_SCHEDULER_AGING_QUANTUM_MICROS UINT64_C(10000)

int64_t vx_scheduler_policy_effective_priority_v1(
    int32_t priority,
    uint64_t submitted_micros,
    uint64_t now_micros) {
    uint64_t age_points = 0;
    if (now_micros > submitted_micros)
        age_points = (now_micros - submitted_micros) /
                     VX_SCHEDULER_AGING_QUANTUM_MICROS;
    if (age_points > (uint64_t)INT64_MAX -
                         (uint64_t)VX_SCHEDULER_POLICY_PRIORITY_MAX)
        return INT64_MAX;
    return (int64_t)priority + (int64_t)age_points;
}

uint32_t vx_scheduler_policy_abi_version(void) {
    return VX_SCHEDULER_POLICY_ABI_VERSION;
}

int32_t vx_scheduler_policy_precedes_v1(
    int32_t candidate_priority,
    uint64_t candidate_submitted_micros,
    uint64_t candidate_deadline_micros,
    uint64_t candidate_request_id,
    int32_t current_priority,
    uint64_t current_submitted_micros,
    uint64_t current_deadline_micros,
    uint64_t current_request_id,
    uint64_t now_micros) {
    int64_t candidate_effective = vx_scheduler_policy_effective_priority_v1(
        candidate_priority, candidate_submitted_micros, now_micros);
    int64_t current_effective = vx_scheduler_policy_effective_priority_v1(
        current_priority, current_submitted_micros, now_micros);
    uint64_t candidate_deadline = candidate_deadline_micros
        ? candidate_deadline_micros : UINT64_MAX;
    uint64_t current_deadline = current_deadline_micros
        ? current_deadline_micros : UINT64_MAX;
    if (candidate_effective != current_effective)
        return candidate_effective > current_effective;
    if (candidate_deadline != current_deadline)
        return candidate_deadline < current_deadline;
    return candidate_request_id < current_request_id;
}

int32_t vx_scheduler_policy_deadline_expired_v1(
    uint64_t deadline_micros,
    uint64_t now_micros) {
    return deadline_micros && now_micros && now_micros >= deadline_micros;
}

int32_t vx_scheduler_policy_latest_matches_v1(
    uint64_t queued_route_id,
    uint32_t queued_freshness,
    uint64_t queued_stream_key,
    uint64_t incoming_route_id,
    uint32_t incoming_freshness,
    uint64_t incoming_stream_key) {
    return queued_freshness == VX_SCHEDULER_POLICY_FRESHNESS_LATEST &&
           incoming_freshness == VX_SCHEDULER_POLICY_FRESHNESS_LATEST &&
           queued_route_id == incoming_route_id &&
           queued_stream_key == incoming_stream_key;
}

int32_t vx_scheduler_policy_budget_admits_v1(
    uint64_t active_requests,
    uint64_t maximum_requests,
    uint64_t active_input_bytes,
    uint64_t maximum_input_bytes,
    uint64_t incoming_input_bytes,
    uint32_t reuse_request_slot) {
    if (!reuse_request_slot && active_requests >= maximum_requests) return 0;
    if (active_input_bytes > maximum_input_bytes) return 0;
    return incoming_input_bytes <= maximum_input_bytes - active_input_bytes;
}
