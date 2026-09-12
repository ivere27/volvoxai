#ifndef VOLVOXAI_SCHEDULER_POLICY_H
#define VOLVOXAI_SCHEDULER_POLICY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Dependency-free scheduler policy shared by native and future WASM control
 * compositions. Every cross-call value is fixed width; clocks, queue storage,
 * locking, and request ownership remain host adapters. */
#define VX_SCHEDULER_POLICY_ABI_VERSION 1u
#define VX_SCHEDULER_POLICY_PRIORITY_MAX 1000
#define VX_SCHEDULER_POLICY_FRESHNESS_LATEST 1u

uint32_t vx_scheduler_policy_abi_version(void);

/* Saturating priority after one point per 10 ms of queue age. */
int64_t vx_scheduler_policy_effective_priority_v1(
    int32_t priority,
    uint64_t submitted_micros,
    uint64_t now_micros);

/* Higher aged priority wins. Equal priorities use earliest deadline first
 * (zero means no deadline), then the lower request identity. Priorities must
 * be in [-VX_SCHEDULER_POLICY_PRIORITY_MAX,
 * VX_SCHEDULER_POLICY_PRIORITY_MAX]. */
int32_t vx_scheduler_policy_precedes_v1(
    int32_t candidate_priority,
    uint64_t candidate_submitted_micros,
    uint64_t candidate_deadline_micros,
    uint64_t candidate_request_id,
    int32_t current_priority,
    uint64_t current_submitted_micros,
    uint64_t current_deadline_micros,
    uint64_t current_request_id,
    uint64_t now_micros);

/* A zero deadline or unavailable zero clock never expires a request. */
int32_t vx_scheduler_policy_deadline_expired_v1(
    uint64_t deadline_micros,
    uint64_t now_micros);

/* LATEST matches only within one exact compiled route and stream. Freshness
 * numbers are the generated public vocabulary projected by the host adapter. */
int32_t vx_scheduler_policy_latest_matches_v1(
    uint64_t queued_route_id,
    uint32_t queued_freshness,
    uint64_t queued_stream_key,
    uint64_t incoming_route_id,
    uint32_t incoming_freshness,
    uint64_t incoming_stream_key);

/* Admission is atomic with the host-owned reservation. Reusing a superseded
 * request slot bypasses only the count increment; byte headroom is unchanged. */
int32_t vx_scheduler_policy_budget_admits_v1(
    uint64_t active_requests,
    uint64_t maximum_requests,
    uint64_t active_input_bytes,
    uint64_t maximum_input_bytes,
    uint64_t incoming_input_bytes,
    uint32_t reuse_request_slot);

#ifdef __cplusplus
}
#endif

#endif
