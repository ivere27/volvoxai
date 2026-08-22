//! Runtime-scoped adapter from the versioned native process sampler to typed
//! memory evidence.
//!
//! The adapter intentionally emits no resource records and never interprets
//! `VxReport::allocated_bytes`. Native resource accounting and periodic
//! sampling require separate versioned collector surfaces.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use crate::abi::{
    vx_process_memory_sample_v1, VxProcessMemorySampleV1, VX_PROCESS_MEMORY_AVAILABLE_CURRENT_RSS,
    VX_PROCESS_MEMORY_AVAILABLE_MONOTONIC_TIME, VX_PROCESS_MEMORY_AVAILABLE_PEAK_RSS,
};
use crate::memory_evidence::MEMORY_EVIDENCE_FORMAT;
use crate::pb::{
    MemoryByteSize, MemoryCaptureOptions, MemoryEnvelopeEvidence, MemoryEnvelopeKind,
    MemoryEvidence, MemoryEvidenceSource, MemoryInventoryKind, MemoryMonotonicTime,
    MemoryOwnerKind, MemoryOwnerRef, MemorySnapshot, MemorySnapshotPoint, MemoryTemporalCoverage,
    MemoryValueRelation, OperationReport, OperationStage,
};

pub(crate) const MEMORY_CAPTURE_PROTOCOL: &str = "volvoxai-memory-capture/v1";
const MAX_PERIODIC_SNAPSHOTS: u32 = 4096;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MemoryCapturePolicy {
    _include_resource_inventory: bool,
    _include_domain_attestation: bool,
    requested_envelopes: Vec<MemoryEnvelopeKind>,
}

struct RuntimeCaptureState {
    native_runtime_id: u64,
    policy: Option<MemoryCapturePolicy>,
    next_capture: Mutex<u64>,
    registry: Weak<MemoryCaptureRegistry>,
}

#[derive(Default)]
pub(crate) struct MemoryCaptureRegistry {
    entries: Mutex<HashMap<u64, Vec<Weak<RuntimeCaptureState>>>>,
}

#[derive(Clone)]
pub(crate) struct MemoryCaptureToken {
    state: Arc<RuntimeCaptureState>,
    subject: MemoryOwnerRef,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl Drop for RuntimeCaptureState {
    fn drop(&mut self) {
        let Some(registry) = self.registry.upgrade() else {
            return;
        };
        let mut entries = lock(&registry.entries);
        let mut remove_key = false;
        if let Some(states) = entries.get_mut(&self.native_runtime_id) {
            states.retain(|state| !std::ptr::eq(state.as_ptr(), self as *const Self));
            remove_key = states.is_empty();
        }
        if remove_key {
            entries.remove(&self.native_runtime_id);
        }
    }
}

pub(crate) fn validate_options(
    options: &MemoryCaptureOptions,
) -> Result<MemoryCapturePolicy, String> {
    if options.protocol != MEMORY_CAPTURE_PROTOCOL {
        return Err(format!(
            "memory_capture.protocol must be exactly {MEMORY_CAPTURE_PROTOCOL}"
        ));
    }

    let mut requested_envelopes = Vec::with_capacity(options.requested_envelopes.len());
    let mut seen = HashSet::new();
    for (index, raw_kind) in options.requested_envelopes.iter().copied().enumerate() {
        let kind = MemoryEnvelopeKind::try_from(raw_kind).map_err(|_| {
            format!("memory_capture.requested_envelopes[{index}] has unknown value {raw_kind}")
        })?;
        if kind == MemoryEnvelopeKind::Unspecified {
            return Err(format!(
                "memory_capture.requested_envelopes[{index}] must not be UNSPECIFIED"
            ));
        }
        if !seen.insert(raw_kind) {
            return Err(format!(
                "memory_capture.requested_envelopes[{index}] is duplicated"
            ));
        }
        requested_envelopes.push(kind);
    }

    if !options.include_resource_inventory
        && !options.include_domain_attestation
        && requested_envelopes.is_empty()
    {
        return Err(
            "memory_capture must select a resource inventory, domain attestation, or envelope"
                .to_owned(),
        );
    }

    match (
        options.sampling_interval_nanoseconds,
        options.max_periodic_snapshots,
    ) {
        (None, None) => {}
        (Some(_), None) | (None, Some(_)) => {
            return Err(
                "memory_capture periodic interval and snapshot limit must be present together"
                    .to_owned(),
            )
        }
        (Some(0), Some(_)) => {
            return Err("memory_capture.sampling_interval_nanoseconds must be positive".to_owned())
        }
        (Some(_), Some(0)) => {
            return Err("memory_capture.max_periodic_snapshots must be in [1, 4096]".to_owned())
        }
        (Some(_), Some(max)) if max > MAX_PERIODIC_SNAPSHOTS => {
            return Err("memory_capture.max_periodic_snapshots must be in [1, 4096]".to_owned())
        }
        (Some(_), Some(_)) => {
            return Err(
                "periodic memory sampling is not supported by the native FFI adapter".to_owned(),
            )
        }
    }

    Ok(MemoryCapturePolicy {
        _include_resource_inventory: options.include_resource_inventory,
        _include_domain_attestation: options.include_domain_attestation,
        requested_envelopes,
    })
}

impl MemoryCaptureRegistry {
    pub(crate) fn register(
        self: &Arc<Self>,
        native_runtime_id: u64,
        policy: Option<MemoryCapturePolicy>,
    ) -> Result<MemoryCaptureToken, String> {
        if policy.is_none() {
            return Ok(MemoryCaptureToken {
                state: Arc::new(RuntimeCaptureState {
                    native_runtime_id,
                    policy: None,
                    next_capture: Mutex::new(1),
                    registry: Weak::new(),
                }),
                subject: MemoryOwnerRef {
                    kind: MemoryOwnerKind::Runtime as i32,
                    owner_id: native_runtime_id.to_string(),
                },
            });
        }
        if native_runtime_id == 0 {
            return Err("native runtime report has no runtime_id".to_owned());
        }
        let state = Arc::new(RuntimeCaptureState {
            native_runtime_id,
            policy,
            next_capture: Mutex::new(1),
            registry: Arc::downgrade(self),
        });
        lock(&self.entries)
            .entry(native_runtime_id)
            .or_default()
            .push(Arc::downgrade(&state));
        Ok(MemoryCaptureToken {
            state,
            subject: MemoryOwnerRef {
                kind: MemoryOwnerKind::Runtime as i32,
                owner_id: native_runtime_id.to_string(),
            },
        })
    }

    #[cfg(test)]
    fn active_entries(&self) -> usize {
        lock(&self.entries).values().map(Vec::len).sum()
    }
}

impl MemoryCaptureToken {
    pub(crate) fn child(
        &self,
        kind: MemoryOwnerKind,
        owner_id: impl Into<String>,
    ) -> Result<Self, String> {
        let owner_id = owner_id.into();
        if kind == MemoryOwnerKind::Unspecified {
            return Err("memory capture subject kind must not be UNSPECIFIED".to_owned());
        }
        if !owner_id
            .bytes()
            .any(|byte| !matches!(byte, b'\t' | b'\n' | 0x0b | 0x0c | b'\r' | b' '))
        {
            return Err("memory capture subject owner_id must not be empty".to_owned());
        }
        Ok(Self {
            state: Arc::clone(&self.state),
            subject: MemoryOwnerRef {
                kind: kind as i32,
                owner_id,
            },
        })
    }

    pub(crate) fn enabled(&self) -> bool {
        self.state.policy.is_some()
    }

    pub(crate) fn attach(&self, mut report: OperationReport) -> Result<OperationReport, String> {
        if report.memory_evidence.is_some() {
            return Err("operation report already contains memory evidence".to_owned());
        }
        if !matches!(
            OperationStage::try_from(report.stage),
            Ok(stage) if stage != OperationStage::None
        ) {
            // Capture is best effort. Do not turn a successful native call
            // into an FFI failure when its report cannot identify a valid
            // lifecycle phase for a conforming snapshot.
            return Ok(report);
        }
        if report.runtime_id != self.state.native_runtime_id {
            // The token and native lineage must describe the same runtime.
            // Omit best-effort evidence rather than turning an otherwise
            // successful lifecycle operation into a collector failure.
            return Ok(report);
        }
        let Some(policy) = self.state.policy.as_ref() else {
            return Ok(report);
        };
        let capture_sequence = {
            let mut next_capture = lock(&self.state.next_capture);
            let capture_sequence = *next_capture;
            *next_capture = next_capture
                .checked_add(1)
                .ok_or_else(|| "memory capture sequence exhausted".to_owned())?;
            capture_sequence
        };

        let mut sample = VxProcessMemorySampleV1::new();
        let needs_process_sampler = policy.requested_envelopes.iter().any(|kind| {
            matches!(
                kind,
                MemoryEnvelopeKind::ProcessRss | MemoryEnvelopeKind::ProcessPeakRss
            )
        });
        let sampled =
            needs_process_sampler && unsafe { vx_process_memory_sample_v1(&mut sample) } != 0;
        report.memory_evidence = Some(build_evidence(
            policy,
            &report,
            capture_sequence,
            sampled,
            &sample,
            &self.subject,
        ));
        Ok(report)
    }
}

fn build_evidence(
    policy: &MemoryCapturePolicy,
    report: &OperationReport,
    capture_sequence: u64,
    sampled: bool,
    sample: &VxProcessMemorySampleV1,
    subject: &MemoryOwnerRef,
) -> MemoryEvidence {
    let native_runtime_id = report.runtime_id;
    let capture_id = format!("native-{native_runtime_id}-capture-{capture_sequence}");
    let phase_occurrence_id = format!("native-{native_runtime_id}-phase-{capture_sequence}");
    let envelopes = policy
        .requested_envelopes
        .iter()
        .copied()
        .map(|kind| envelope(kind, sampled, sample))
        .collect();

    MemoryEvidence {
        format: MEMORY_EVIDENCE_FORMAT.to_owned(),
        capture_id,
        required_features: Vec::new(),
        snapshots: vec![MemorySnapshot {
            sequence: 1,
            monotonic_time: (sampled
                && sample.available_mask & VX_PROCESS_MEMORY_AVAILABLE_MONOTONIC_TIME != 0)
                .then_some(MemoryMonotonicTime {
                    nanoseconds: sample.monotonic_nanoseconds,
                }),
            stage: report.stage,
            point: MemorySnapshotPoint::After as i32,
            phase_occurrence_id,
            subject: Some(subject.clone()),
            backend: report.backend.clone(),
            device: report.device.clone(),
            domain_attestation: None,
            resources: Vec::new(),
            envelopes,
            resource_inventory: MemoryInventoryKind::Partial as i32,
        }],
    }
}

fn envelope(
    kind: MemoryEnvelopeKind,
    sampled: bool,
    sample: &VxProcessMemorySampleV1,
) -> MemoryEnvelopeEvidence {
    let (available, bytes, source, coverage, sampler) = match kind {
        MemoryEnvelopeKind::ProcessRss => (
            sampled && sample.available_mask & VX_PROCESS_MEMORY_AVAILABLE_CURRENT_RSS != 0,
            sample.rss_bytes,
            MemoryEvidenceSource::OsSampler,
            MemoryTemporalCoverage::Instant,
            "vx_process_memory_sample_v1",
        ),
        MemoryEnvelopeKind::ProcessPeakRss => (
            sampled && sample.available_mask & VX_PROCESS_MEMORY_AVAILABLE_PEAK_RSS != 0,
            sample.peak_rss_bytes,
            MemoryEvidenceSource::OsSampler,
            MemoryTemporalCoverage::ProcessLifetime,
            "vx_process_memory_sample_v1",
        ),
        MemoryEnvelopeKind::ProcessPss | MemoryEnvelopeKind::ProcessPrivateBytes => (
            false,
            0,
            MemoryEvidenceSource::OsSampler,
            MemoryTemporalCoverage::Instant,
            "vx_process_memory_sample_v1",
        ),
        MemoryEnvelopeKind::ProcessManagedHeapUsed
        | MemoryEnvelopeKind::ProcessExternalBytes
        | MemoryEnvelopeKind::ProcessArrayBufferBytes => (
            false,
            0,
            MemoryEvidenceSource::RuntimeCounter,
            MemoryTemporalCoverage::Instant,
            "native-runtime-memory-counters/v1",
        ),
        MemoryEnvelopeKind::DeviceProcessUsed
        | MemoryEnvelopeKind::DeviceTotalUsed
        | MemoryEnvelopeKind::DeviceTotalCapacity => (
            false,
            0,
            MemoryEvidenceSource::DriverSampler,
            MemoryTemporalCoverage::Instant,
            "native-driver-memory-sampler/v1",
        ),
        MemoryEnvelopeKind::Unspecified => unreachable!("capture policy validation rejects zero"),
    };
    MemoryEnvelopeEvidence {
        kind: kind as i32,
        bytes: available.then_some(MemoryByteSize { bytes }),
        source: source as i32,
        value_relation: if available {
            MemoryValueRelation::Exact as i32
        } else {
            MemoryValueRelation::Unavailable as i32
        },
        temporal_coverage: coverage as i32,
        sampler: sampler.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_evidence::validate_memory_evidence;
    use crate::pb::NativeStatus;

    fn options(envelopes: Vec<MemoryEnvelopeKind>) -> MemoryCaptureOptions {
        MemoryCaptureOptions {
            protocol: MEMORY_CAPTURE_PROTOCOL.to_owned(),
            include_resource_inventory: false,
            include_domain_attestation: false,
            requested_envelopes: envelopes.into_iter().map(|kind| kind as i32).collect(),
            sampling_interval_nanoseconds: None,
            max_periodic_snapshots: None,
        }
    }

    fn report() -> OperationReport {
        OperationReport {
            native_status: NativeStatus::Ok as i32,
            stage: OperationStage::Execute as i32,
            runtime_id: 41,
            context_id: 7,
            backend: "cpu".to_owned(),
            device: "host".to_owned(),
            ..Default::default()
        }
    }

    fn context_subject() -> MemoryOwnerRef {
        MemoryOwnerRef {
            kind: MemoryOwnerKind::ExecutionContext as i32,
            owner_id: "7".to_owned(),
        }
    }

    #[test]
    fn capture_options_are_fail_closed() {
        let mut requested = options(vec![MemoryEnvelopeKind::ProcessRss]);
        requested.protocol = "volvoxai-memory-capture/v2".to_owned();
        assert!(validate_options(&requested)
            .unwrap_err()
            .contains("exactly"));

        let empty = options(Vec::new());
        assert!(validate_options(&empty)
            .unwrap_err()
            .contains("must select"));

        let duplicate = options(vec![
            MemoryEnvelopeKind::ProcessRss,
            MemoryEnvelopeKind::ProcessRss,
        ]);
        assert!(validate_options(&duplicate)
            .unwrap_err()
            .contains("duplicated"));

        let mut unspecified = options(vec![MemoryEnvelopeKind::ProcessRss]);
        unspecified.requested_envelopes[0] = MemoryEnvelopeKind::Unspecified as i32;
        assert!(validate_options(&unspecified)
            .unwrap_err()
            .contains("UNSPECIFIED"));

        let mut unknown = options(vec![MemoryEnvelopeKind::ProcessRss]);
        unknown.requested_envelopes[0] = i32::MAX;
        assert!(validate_options(&unknown).unwrap_err().contains("unknown"));

        let mut unpaired = options(vec![MemoryEnvelopeKind::ProcessRss]);
        unpaired.sampling_interval_nanoseconds = Some(1);
        assert!(validate_options(&unpaired)
            .unwrap_err()
            .contains("present together"));

        let mut excessive = options(vec![MemoryEnvelopeKind::ProcessRss]);
        excessive.sampling_interval_nanoseconds = Some(1);
        excessive.max_periodic_snapshots = Some(4097);
        assert!(validate_options(&excessive)
            .unwrap_err()
            .contains("[1, 4096]"));

        let mut periodic = options(vec![MemoryEnvelopeKind::ProcessRss]);
        periodic.sampling_interval_nanoseconds = Some(1);
        periodic.max_periodic_snapshots = Some(1);
        assert!(validate_options(&periodic)
            .unwrap_err()
            .contains("not supported"));
    }

    #[test]
    fn available_and_unavailable_envelopes_remain_typed() {
        let policy = validate_options(&options(vec![
            MemoryEnvelopeKind::ProcessRss,
            MemoryEnvelopeKind::ProcessPeakRss,
            MemoryEnvelopeKind::ProcessPss,
            MemoryEnvelopeKind::DeviceProcessUsed,
        ]))
        .unwrap();
        let mut sample = VxProcessMemorySampleV1::new();
        sample.available_mask = VX_PROCESS_MEMORY_AVAILABLE_CURRENT_RSS
            | VX_PROCESS_MEMORY_AVAILABLE_PEAK_RSS
            | VX_PROCESS_MEMORY_AVAILABLE_MONOTONIC_TIME;
        sample.rss_bytes = 123;
        sample.peak_rss_bytes = 456;
        sample.monotonic_nanoseconds = 789;
        let evidence = build_evidence(&policy, &report(), 3, true, &sample, &context_subject());

        validate_memory_evidence(&evidence).unwrap();
        assert_eq!(evidence.capture_id, "native-41-capture-3");
        let snapshot = &evidence.snapshots[0];
        assert_eq!(snapshot.point, MemorySnapshotPoint::After as i32);
        assert_eq!(
            snapshot.resource_inventory,
            MemoryInventoryKind::Partial as i32
        );
        assert!(snapshot.resources.is_empty());
        assert_eq!(snapshot.subject.as_ref().unwrap().owner_id, "7");
        assert_eq!(snapshot.envelopes[0].bytes.as_ref().unwrap().bytes, 123);
        assert_eq!(snapshot.envelopes[1].bytes.as_ref().unwrap().bytes, 456);
        for unavailable in &snapshot.envelopes[2..] {
            assert_eq!(
                unavailable.value_relation,
                MemoryValueRelation::Unavailable as i32
            );
            assert!(unavailable.bytes.is_none());
        }
    }

    #[test]
    fn sampler_failure_marks_even_supported_process_signals_unavailable() {
        let policy = validate_options(&options(vec![
            MemoryEnvelopeKind::ProcessRss,
            MemoryEnvelopeKind::ProcessPeakRss,
        ]))
        .unwrap();
        let evidence = build_evidence(
            &policy,
            &report(),
            1,
            false,
            &VxProcessMemorySampleV1::new(),
            &context_subject(),
        );
        validate_memory_evidence(&evidence).unwrap();
        assert!(evidence.snapshots[0]
            .envelopes
            .iter()
            .all(
                |item| item.value_relation == MemoryValueRelation::Unavailable as i32
                    && item.bytes.is_none()
            ));
    }

    #[test]
    fn same_native_id_tokens_keep_their_policies_isolated() {
        let registry = Arc::new(MemoryCaptureRegistry::default());
        let rss = registry
            .register(
                41,
                Some(validate_options(&options(vec![MemoryEnvelopeKind::ProcessRss])).unwrap()),
            )
            .unwrap();
        let peak = registry
            .register(
                41,
                Some(validate_options(&options(vec![MemoryEnvelopeKind::ProcessPeakRss])).unwrap()),
            )
            .unwrap();

        let rss_report = rss.attach(report()).unwrap();
        let peak_report = peak.attach(report()).unwrap();
        assert_eq!(
            rss_report.memory_evidence.unwrap().snapshots[0].envelopes[0].kind,
            MemoryEnvelopeKind::ProcessRss as i32
        );
        assert_eq!(
            peak_report.memory_evidence.unwrap().snapshots[0].envelopes[0].kind,
            MemoryEnvelopeKind::ProcessPeakRss as i32
        );
        assert_eq!(registry.active_entries(), 2);
        drop(rss);
        drop(peak);
        assert_eq!(registry.active_entries(), 0);
    }

    #[test]
    fn invalid_native_stage_omits_best_effort_capture_without_consuming_an_id() {
        let registry = Arc::new(MemoryCaptureRegistry::default());
        let token = registry
            .register(
                41,
                Some(validate_options(&options(vec![MemoryEnvelopeKind::ProcessRss])).unwrap()),
            )
            .unwrap();

        for stage in [OperationStage::None as i32, i32::MAX] {
            let mut invalid = report();
            invalid.stage = stage;
            assert!(token.attach(invalid).unwrap().memory_evidence.is_none());
        }
        let captured = token.attach(report()).unwrap();
        assert_eq!(
            captured.memory_evidence.unwrap().capture_id,
            "native-41-capture-1"
        );
    }

    #[test]
    fn disabled_runtime_state_never_inherits_another_policy() {
        let registry = Arc::new(MemoryCaptureRegistry::default());
        let disabled = registry.register(41, None).unwrap();
        let enabled = registry
            .register(
                41,
                Some(validate_options(&options(vec![MemoryEnvelopeKind::ProcessRss])).unwrap()),
            )
            .unwrap();
        assert!(disabled.attach(report()).unwrap().memory_evidence.is_none());
        assert!(enabled.attach(report()).unwrap().memory_evidence.is_some());
        assert_eq!(registry.active_entries(), 1);
        drop(disabled);
        drop(enabled);
        assert_eq!(registry.active_entries(), 0);
    }

    #[test]
    fn descendant_token_keeps_capture_alive_after_root_drop() {
        let registry = Arc::new(MemoryCaptureRegistry::default());
        let root = registry
            .register(
                41,
                Some(validate_options(&options(vec![MemoryEnvelopeKind::ProcessRss])).unwrap()),
            )
            .unwrap();
        let model = root.child(MemoryOwnerKind::Model, "9").unwrap();
        drop(root);

        assert_eq!(registry.active_entries(), 1);
        let captured = model.attach(report()).unwrap();
        let evidence = captured.memory_evidence.unwrap();
        let subject = evidence.snapshots[0].subject.as_ref().unwrap();
        assert_eq!(subject.kind, MemoryOwnerKind::Model as i32);
        assert_eq!(subject.owner_id, "9");
        drop(model);
        assert_eq!(registry.active_entries(), 0);
    }

    #[test]
    fn concurrent_final_drop_and_same_id_registration_do_not_deadlock() {
        for _ in 0..64 {
            let registry = Arc::new(MemoryCaptureRegistry::default());
            let token = registry
                .register(
                    41,
                    Some(validate_options(&options(vec![MemoryEnvelopeKind::ProcessRss])).unwrap()),
                )
                .unwrap();
            let barrier = Arc::new(std::sync::Barrier::new(3));
            let drop_barrier = Arc::clone(&barrier);
            let dropper = std::thread::spawn(move || {
                drop_barrier.wait();
                drop(token);
            });
            let register_registry = Arc::clone(&registry);
            let register_barrier = Arc::clone(&barrier);
            let registrar = std::thread::spawn(move || {
                register_barrier.wait();
                register_registry
                    .register(
                        41,
                        Some(
                            validate_options(&options(vec![MemoryEnvelopeKind::ProcessPeakRss]))
                                .unwrap(),
                        ),
                    )
                    .unwrap()
            });
            barrier.wait();
            dropper.join().unwrap();
            let replacement = registrar.join().unwrap();
            assert!(replacement
                .attach(report())
                .unwrap()
                .memory_evidence
                .is_some());
            drop(replacement);
            assert_eq!(registry.active_entries(), 0);
        }
    }
}
