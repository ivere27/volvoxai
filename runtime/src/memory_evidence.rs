//! Semantic validation for protobuf memory evidence.
//!
//! This module deliberately depends only on the generated protobuf model. The
//! native runtime validates its own collector/provider records before crossing
//! the C ABI; this validator is the final fail-closed check before typed
//! evidence is returned by the Synurang boundary.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

use crate::pb::{
    MemoryBackingRelation, MemoryBoundKind, MemoryDomainAttestation, MemoryEnvelopeEvidence,
    MemoryEnvelopeKind, MemoryEvidence, MemoryEvidenceSource, MemoryInventoryKind,
    MemoryMeasurement, MemoryMetric, MemoryOwnerKind, MemoryOwnerRef, MemoryResourceEvidence,
    MemoryResourceRole, MemorySnapshot, MemorySnapshotPoint, MemorySpace, MemoryTemporalCoverage,
    MemoryValueRelation, OperationStage,
};

pub const MEMORY_EVIDENCE_FORMAT: &str = "volvoxai-memory-evidence/v1";
pub const MEMORY_EVIDENCE_PROOF_PROTOCOL: &str = "canonical-symbolic-domain-proof/v1";
pub const MEMORY_EVIDENCE_RESOURCE_PROTOCOL: &str = "bounded-resource-maxima/v1";

/// Features whose additional validation semantics this implementation knows.
///
/// Version 1 currently defines no opt-in feature extensions. Keeping the set
/// explicit makes adding a feature a deliberate validator change rather than a
/// silently accepted producer convention.
pub const SUPPORTED_MEMORY_EVIDENCE_FEATURES: &[&str] = &[];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryEvidenceValidationError {
    path: String,
    detail: String,
}

impl MemoryEvidenceValidationError {
    fn new(path: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            detail: detail.into(),
        }
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for MemoryEvidenceValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.path, self.detail)
    }
}

impl Error for MemoryEvidenceValidationError {}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct OwnerKey {
    kind: i32,
    owner_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ResourceDescriptor {
    backing_resource_id: String,
    backing_relation: i32,
    backing_offset_bytes: Option<u64>,
    backing_length_bytes: Option<u64>,
    addressable_bytes: u64,
    owner: OwnerKey,
    role: i32,
    space: i32,
    allocator: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PhaseOccurrenceDescriptor {
    stage: i32,
    subject: OwnerKey,
    backend: String,
    device: String,
}

fn invalid(path: impl Into<String>, detail: impl Into<String>) -> MemoryEvidenceValidationError {
    MemoryEvidenceValidationError::new(path, detail)
}

fn known_nonzero_enum<E>(value: i32, path: &str) -> Result<E, MemoryEvidenceValidationError>
where
    E: TryFrom<i32>,
{
    if value == 0 {
        return Err(invalid(path, "must not be UNSPECIFIED"));
    }
    E::try_from(value).map_err(|_| invalid(path, format!("unknown enum value {value}")))
}

fn valid_feature_token(feature: &str) -> bool {
    let mut bytes = feature.bytes();
    matches!(bytes.next(), Some(b'a'..=b'z'))
        && bytes.all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'_' | b'-')
        })
}

fn nonblank(value: &str) -> bool {
    value
        .bytes()
        .any(|byte| !matches!(byte, b'\t' | b'\n' | 0x0b | 0x0c | b'\r' | b' '))
}

fn required_owner(
    owner: Option<&MemoryOwnerRef>,
    path: &str,
) -> Result<OwnerKey, MemoryEvidenceValidationError> {
    let owner = owner.ok_or_else(|| invalid(path, "is required"))?;
    validate_owner(owner, path)
}

fn validate_owner(
    owner: &MemoryOwnerRef,
    path: &str,
) -> Result<OwnerKey, MemoryEvidenceValidationError> {
    known_nonzero_enum::<MemoryOwnerKind>(owner.kind, &format!("{path}.kind"))?;
    if !nonblank(&owner.owner_id) {
        return Err(invalid(format!("{path}.owner_id"), "must not be empty"));
    }
    Ok(OwnerKey {
        kind: owner.kind,
        owner_id: owner.owner_id.clone(),
    })
}

fn validate_domain_attestation(
    attestation: &MemoryDomainAttestation,
    path: &str,
) -> Result<(), MemoryEvidenceValidationError> {
    if attestation.proof_protocol != MEMORY_EVIDENCE_PROOF_PROTOCOL {
        return Err(invalid(
            format!("{path}.proof_protocol"),
            format!("must be exactly {MEMORY_EVIDENCE_PROOF_PROTOCOL}"),
        ));
    }
    if attestation.resource_protocol != MEMORY_EVIDENCE_RESOURCE_PROTOCOL {
        return Err(invalid(
            format!("{path}.resource_protocol"),
            format!("must be exactly {MEMORY_EVIDENCE_RESOURCE_PROTOCOL}"),
        ));
    }
    for (field, value) in [
        ("graph_fingerprint", attestation.graph_fingerprint.as_str()),
        (
            "shape_domain_proof_identity",
            attestation.shape_domain_proof_identity.as_str(),
        ),
    ] {
        if !nonblank(value) {
            return Err(invalid(format!("{path}.{field}"), "must not be empty"));
        }
    }
    if attestation.bounds.is_empty() {
        return Err(invalid(format!("{path}.bounds"), "must not be empty"));
    }

    let mut bound_keys = HashSet::new();
    for (bound_index, bound) in attestation.bounds.iter().enumerate() {
        let bound_path = format!("{path}.bounds[{bound_index}]");
        if !nonblank(&bound.budget_domain_id) {
            return Err(invalid(
                format!("{bound_path}.budget_domain_id"),
                "must not be empty",
            ));
        }
        known_nonzero_enum::<MemoryBoundKind>(bound.kind, &format!("{bound_path}.kind"))?;
        if !bound_keys.insert((bound.budget_domain_id.as_str(), bound.kind)) {
            return Err(invalid(&bound_path, "duplicate (budget_domain_id, kind)"));
        }

        let maximum = bound
            .maximum_bytes
            .as_ref()
            .ok_or_else(|| invalid(format!("{bound_path}.maximum_bytes"), "is required"))?
            .bytes;
        let mut case_ids = HashSet::new();
        let mut computed_maximum = 0u64;
        for (case_index, case) in bound.cases.iter().enumerate() {
            let case_path = format!("{bound_path}.cases[{case_index}]");
            if !nonblank(&case.case_id) {
                return Err(invalid(format!("{case_path}.case_id"), "must not be empty"));
            }
            if !case_ids.insert(case.case_id.as_str()) {
                return Err(invalid(&case_path, "duplicate case_id"));
            }
            if case.terms.is_empty() {
                return Err(invalid(format!("{case_path}.terms"), "must not be empty"));
            }

            let mut term_ids = HashSet::new();
            let mut computed_total = 0u64;
            for (term_index, term) in case.terms.iter().enumerate() {
                let term_path = format!("{case_path}.terms[{term_index}]");
                if !nonblank(&term.term_id) {
                    return Err(invalid(format!("{term_path}.term_id"), "must not be empty"));
                }
                if !term_ids.insert(term.term_id.as_str()) {
                    return Err(invalid(&term_path, "duplicate term_id"));
                }
                known_nonzero_enum::<MemoryOwnerKind>(
                    term.owner_kind,
                    &format!("{term_path}.owner_kind"),
                )?;
                known_nonzero_enum::<MemoryResourceRole>(term.role, &format!("{term_path}.role"))?;
                known_nonzero_enum::<MemorySpace>(term.space, &format!("{term_path}.space"))?;
                let bytes = term
                    .upper_bound_bytes
                    .as_ref()
                    .ok_or_else(|| {
                        invalid(format!("{term_path}.upper_bound_bytes"), "is required")
                    })?
                    .bytes;
                computed_total = computed_total.checked_add(bytes).ok_or_else(|| {
                    invalid(
                        format!("{case_path}.terms"),
                        "upper-bound byte sum overflows u64",
                    )
                })?;
            }

            let declared_total = case
                .total_bytes
                .as_ref()
                .ok_or_else(|| invalid(format!("{case_path}.total_bytes"), "is required"))?
                .bytes;
            if declared_total != computed_total {
                return Err(invalid(
                    format!("{case_path}.total_bytes"),
                    format!("declared {declared_total}, computed {computed_total}"),
                ));
            }
            computed_maximum = computed_maximum.max(declared_total);
        }

        if !bound.cases.is_empty() && maximum != computed_maximum {
            return Err(invalid(
                format!("{bound_path}.maximum_bytes"),
                format!("declared {maximum}, computed {computed_maximum}"),
            ));
        }
        if let Some(limit) = bound.limit_bytes.as_ref() {
            if maximum > limit.bytes {
                return Err(invalid(
                    format!("{bound_path}.maximum_bytes"),
                    format!("maximum {maximum} exceeds limit {}", limit.bytes),
                ));
            }
        }
    }
    Ok(())
}

fn validate_value_presence(
    bytes_present: bool,
    relation: MemoryValueRelation,
    path: &str,
) -> Result<(), MemoryEvidenceValidationError> {
    if (relation == MemoryValueRelation::Unavailable) != !bytes_present {
        return Err(invalid(
            format!("{path}.bytes"),
            "must be absent exactly when value_relation is UNAVAILABLE",
        ));
    }
    Ok(())
}

fn validate_sampled_window_relation(
    relation: MemoryValueRelation,
    temporal_coverage: MemoryTemporalCoverage,
    path: &str,
) -> Result<(), MemoryEvidenceValidationError> {
    if temporal_coverage == MemoryTemporalCoverage::SampledWindow
        && relation != MemoryValueRelation::LowerBound
        && relation != MemoryValueRelation::Unavailable
    {
        return Err(invalid(
            format!("{path}.value_relation"),
            "SAMPLED_WINDOW values must be LOWER_BOUND or UNAVAILABLE",
        ));
    }
    Ok(())
}

fn is_high_water_metric(metric: MemoryMetric) -> bool {
    matches!(
        metric,
        MemoryMetric::LogicalHighWater
            | MemoryMetric::LiveHighWater
            | MemoryMetric::ReservedHighWater
            | MemoryMetric::CapacityHighWater
    )
}

fn validate_measurement(
    measurement: &MemoryMeasurement,
    path: &str,
) -> Result<(), MemoryEvidenceValidationError> {
    let metric = known_nonzero_enum::<MemoryMetric>(measurement.metric, &format!("{path}.metric"))?;
    let source =
        known_nonzero_enum::<MemoryEvidenceSource>(measurement.source, &format!("{path}.source"))?;
    let relation = known_nonzero_enum::<MemoryValueRelation>(
        measurement.value_relation,
        &format!("{path}.value_relation"),
    )?;
    let temporal_coverage = known_nonzero_enum::<MemoryTemporalCoverage>(
        measurement.temporal_coverage,
        &format!("{path}.temporal_coverage"),
    )?;

    if !matches!(
        source,
        MemoryEvidenceSource::AllocatorCounter
            | MemoryEvidenceSource::RuntimeCounter
            | MemoryEvidenceSource::ApiRequest
    ) {
        return Err(invalid(
            format!("{path}.source"),
            "resource measurements require ALLOCATOR_COUNTER, RUNTIME_COUNTER, or API_REQUEST",
        ));
    }
    if (source == MemoryEvidenceSource::ApiRequest) != (relation == MemoryValueRelation::Requested)
    {
        return Err(invalid(
            path,
            "API_REQUEST source and REQUESTED value relation must occur together",
        ));
    }
    validate_value_presence(measurement.bytes.is_some(), relation, path)?;

    if (is_high_water_metric(metric) || metric == MemoryMetric::CumulativeAllocated)
        && temporal_coverage == MemoryTemporalCoverage::Instant
    {
        return Err(invalid(
            format!("{path}.temporal_coverage"),
            "high-water and cumulative metrics cannot have INSTANT coverage",
        ));
    }
    if temporal_coverage == MemoryTemporalCoverage::SampledWindow {
        if !is_high_water_metric(metric) {
            return Err(invalid(
                format!("{path}.metric"),
                "must be a high-water metric for SAMPLED_WINDOW",
            ));
        }
        validate_sampled_window_relation(relation, temporal_coverage, path)?;
    }
    Ok(())
}

fn expected_envelope_source(kind: MemoryEnvelopeKind) -> MemoryEvidenceSource {
    match kind {
        MemoryEnvelopeKind::ProcessRss
        | MemoryEnvelopeKind::ProcessPeakRss
        | MemoryEnvelopeKind::ProcessPss
        | MemoryEnvelopeKind::ProcessPrivateBytes => MemoryEvidenceSource::OsSampler,
        MemoryEnvelopeKind::ProcessManagedHeapUsed
        | MemoryEnvelopeKind::ProcessExternalBytes
        | MemoryEnvelopeKind::ProcessArrayBufferBytes => MemoryEvidenceSource::RuntimeCounter,
        MemoryEnvelopeKind::DeviceProcessUsed
        | MemoryEnvelopeKind::DeviceTotalUsed
        | MemoryEnvelopeKind::DeviceTotalCapacity => MemoryEvidenceSource::DriverSampler,
        MemoryEnvelopeKind::Unspecified => unreachable!("validated before source selection"),
    }
}

fn validate_envelope(
    envelope: &MemoryEnvelopeEvidence,
    path: &str,
) -> Result<(), MemoryEvidenceValidationError> {
    let kind = known_nonzero_enum::<MemoryEnvelopeKind>(envelope.kind, &format!("{path}.kind"))?;
    let source =
        known_nonzero_enum::<MemoryEvidenceSource>(envelope.source, &format!("{path}.source"))?;
    let relation = known_nonzero_enum::<MemoryValueRelation>(
        envelope.value_relation,
        &format!("{path}.value_relation"),
    )?;
    let temporal_coverage = known_nonzero_enum::<MemoryTemporalCoverage>(
        envelope.temporal_coverage,
        &format!("{path}.temporal_coverage"),
    )?;

    let expected_source = expected_envelope_source(kind);
    if source != expected_source {
        return Err(invalid(
            format!("{path}.source"),
            format!("is incompatible with envelope kind {kind:?}"),
        ));
    }
    if relation == MemoryValueRelation::Requested {
        return Err(invalid(
            format!("{path}.value_relation"),
            "envelopes cannot use REQUESTED",
        ));
    }
    if !nonblank(&envelope.sampler) {
        return Err(invalid(format!("{path}.sampler"), "must not be empty"));
    }
    validate_value_presence(envelope.bytes.is_some(), relation, path)?;
    validate_sampled_window_relation(relation, temporal_coverage, path)?;
    if kind == MemoryEnvelopeKind::ProcessPeakRss
        && (temporal_coverage != MemoryTemporalCoverage::ProcessLifetime
            || (relation != MemoryValueRelation::Exact
                && relation != MemoryValueRelation::Unavailable))
    {
        return Err(invalid(
            path,
            "PROCESS_PEAK_RSS must be EXACT over PROCESS_LIFETIME, or UNAVAILABLE",
        ));
    }
    Ok(())
}

fn validate_resource(
    resource: &MemoryResourceEvidence,
    path: &str,
) -> Result<ResourceDescriptor, MemoryEvidenceValidationError> {
    if !nonblank(&resource.resource_id) {
        return Err(invalid(format!("{path}.resource_id"), "must not be empty"));
    }
    let backing_relation = known_nonzero_enum::<MemoryBackingRelation>(
        resource.backing_relation,
        &format!("{path}.backing_relation"),
    )?;
    let owner = required_owner(resource.owner.as_ref(), &format!("{path}.owner"))?;
    known_nonzero_enum::<MemoryResourceRole>(resource.role, &format!("{path}.role"))?;
    known_nonzero_enum::<MemorySpace>(resource.space, &format!("{path}.space"))?;
    if !nonblank(&resource.allocator) {
        return Err(invalid(format!("{path}.allocator"), "must not be empty"));
    }
    if resource.measurements.is_empty() {
        return Err(invalid(format!("{path}.measurements"), "must not be empty"));
    }
    let addressable_bytes = resource
        .addressable_bytes
        .as_ref()
        .ok_or_else(|| invalid(format!("{path}.addressable_bytes"), "is required"))?
        .bytes;

    let (backing_offset_bytes, backing_length_bytes) = match backing_relation {
        MemoryBackingRelation::Independent => {
            if !resource.backing_resource_id.is_empty()
                || resource.backing_offset_bytes.is_some()
                || resource.backing_length_bytes.is_some()
            {
                return Err(invalid(
                    path,
                    "INDEPENDENT resources cannot name backing storage or a backing range",
                ));
            }
            (None, None)
        }
        MemoryBackingRelation::Alias | MemoryBackingRelation::Suballocation => {
            if !nonblank(&resource.backing_resource_id) {
                return Err(invalid(
                    format!("{path}.backing_resource_id"),
                    "is required for ALIAS/SUBALLOCATION",
                ));
            }
            let offset = resource
                .backing_offset_bytes
                .as_ref()
                .ok_or_else(|| {
                    invalid(
                        format!("{path}.backing_offset_bytes"),
                        "is required for ALIAS/SUBALLOCATION",
                    )
                })?
                .bytes;
            let length = resource
                .backing_length_bytes
                .as_ref()
                .ok_or_else(|| {
                    invalid(
                        format!("{path}.backing_length_bytes"),
                        "is required for ALIAS/SUBALLOCATION",
                    )
                })?
                .bytes;
            if addressable_bytes != length {
                return Err(invalid(
                    format!("{path}.addressable_bytes"),
                    format!("must equal backing_length_bytes ({length})"),
                ));
            }
            offset.checked_add(length).ok_or_else(|| {
                invalid(
                    format!("{path}.backing_length_bytes"),
                    "backing range end overflows u64",
                )
            })?;
            if resource.backing_resource_id == resource.resource_id {
                return Err(invalid(
                    format!("{path}.backing_resource_id"),
                    "resource cannot back itself",
                ));
            }
            (Some(offset), Some(length))
        }
        MemoryBackingRelation::Unspecified => unreachable!("validated above"),
    };

    let mut consumer_ids = HashSet::new();
    for (consumer_index, consumer) in resource.consumers.iter().enumerate() {
        let consumer_path = format!("{path}.consumers[{consumer_index}]");
        let key = validate_owner(consumer, &consumer_path)?;
        if !consumer_ids.insert(key) {
            return Err(invalid(&consumer_path, "duplicate consumer owner"));
        }
    }
    let mut measurement_axes = HashSet::new();
    for (measurement_index, measurement) in resource.measurements.iter().enumerate() {
        let measurement_path = format!("{path}.measurements[{measurement_index}]");
        validate_measurement(measurement, &measurement_path)?;
        if !measurement_axes.insert((
            measurement.metric,
            measurement.source,
            measurement.value_relation,
            measurement.temporal_coverage,
        )) {
            return Err(invalid(&measurement_path, "duplicate measurement axis"));
        }
    }

    Ok(ResourceDescriptor {
        backing_resource_id: resource.backing_resource_id.clone(),
        backing_relation: resource.backing_relation,
        backing_offset_bytes,
        backing_length_bytes,
        addressable_bytes,
        owner,
        role: resource.role,
        space: resource.space,
        allocator: resource.allocator.clone(),
    })
}

fn validate_backing_graph(
    resources: &HashMap<String, (ResourceDescriptor, String)>,
) -> Result<(), MemoryEvidenceValidationError> {
    for (resource_id, (descriptor, first_path)) in resources {
        match MemoryBackingRelation::try_from(descriptor.backing_relation)
            .expect("resource relation was validated")
        {
            MemoryBackingRelation::Independent => {}
            MemoryBackingRelation::Alias | MemoryBackingRelation::Suballocation => {
                let (backing, _) =
                    resources
                        .get(&descriptor.backing_resource_id)
                        .ok_or_else(|| {
                            invalid(
                                format!("{first_path}.backing_resource_id"),
                                format!(
                                    "resource {} does not resolve in this capture",
                                    descriptor.backing_resource_id
                                ),
                            )
                        })?;
                let offset = descriptor
                    .backing_offset_bytes
                    .expect("backing offset was validated");
                let length = descriptor
                    .backing_length_bytes
                    .expect("backing length was validated");
                let end = offset.checked_add(length).ok_or_else(|| {
                    invalid(
                        format!("{first_path}.backing_length_bytes"),
                        "backing range end overflows u64",
                    )
                })?;
                if end > backing.addressable_bytes {
                    return Err(invalid(
                        format!("{first_path}.backing_length_bytes"),
                        format!(
                            "range end {end} exceeds backing resource extent {}",
                            backing.addressable_bytes
                        ),
                    ));
                }
                if resource_id == &descriptor.backing_resource_id {
                    return Err(invalid(
                        format!("{first_path}.backing_resource_id"),
                        "resource cannot back itself",
                    ));
                }
            }
            MemoryBackingRelation::Unspecified => unreachable!("resource relation was validated"),
        }
    }

    // Each resource has at most one backing edge. Resolve every chain without
    // recursion so adversarially deep evidence cannot overflow the Rust stack.
    // Keeping the absolute root offset also catches overflow across a deep
    // chain even when each individual backing range is locally valid.
    let mut root_offsets: HashMap<&str, u64> = HashMap::new();
    for resource_id in resources.keys() {
        if root_offsets.contains_key(resource_id.as_str()) {
            continue;
        }
        let mut path = Vec::new();
        let mut active = HashSet::new();
        let mut current = resource_id.as_str();
        let mut base_offset = 0u64;
        loop {
            if let Some(cached_offset) = root_offsets.get(current) {
                base_offset = *cached_offset;
                break;
            }
            if !active.insert(current) {
                return Err(invalid(
                    "memory_evidence.snapshots[].resources",
                    format!("backing cycle contains resource {current}"),
                ));
            }
            path.push(current);
            let descriptor = &resources
                .get(current)
                .expect("backing references were resolved")
                .0;
            if descriptor.backing_relation == MemoryBackingRelation::Independent as i32 {
                break;
            }
            current = descriptor.backing_resource_id.as_str();
        }
        for current in path.into_iter().rev() {
            let (descriptor, first_path) = resources
                .get(current)
                .expect("backing references were resolved");
            if descriptor.backing_relation != MemoryBackingRelation::Independent as i32 {
                base_offset = base_offset
                    .checked_add(
                        descriptor
                            .backing_offset_bytes
                            .expect("backing offset was validated"),
                    )
                    .ok_or_else(|| {
                        invalid(
                            format!("{first_path}.backing_offset_bytes"),
                            "absolute backing offset overflows u64",
                        )
                    })?;
            }
            root_offsets.insert(current, base_offset);
        }
    }
    Ok(())
}

fn validate_complete_suballocations(
    snapshot: &MemorySnapshot,
    snapshot_index: usize,
) -> Result<(), MemoryEvidenceValidationError> {
    let mut siblings: HashMap<&str, Vec<(u64, u64, &str)>> = HashMap::new();
    for (resource_index, resource) in snapshot.resources.iter().enumerate() {
        if resource.backing_relation != MemoryBackingRelation::Suballocation as i32 {
            continue;
        }
        let offset = resource
            .backing_offset_bytes
            .as_ref()
            .expect("suballocation offset was validated")
            .bytes;
        let length = resource
            .backing_length_bytes
            .as_ref()
            .expect("suballocation length was validated")
            .bytes;
        let end = offset.checked_add(length).ok_or_else(|| {
            invalid(
                format!(
                    "memory_evidence.snapshots[{snapshot_index}].resources[{resource_index}].backing_length_bytes"
                ),
                "backing range end overflows u64",
            )
        })?;
        if offset != end {
            siblings
                .entry(resource.backing_resource_id.as_str())
                .or_default()
                .push((offset, end, resource.resource_id.as_str()));
        }
    }
    for (backing_id, ranges) in &mut siblings {
        ranges.sort_unstable_by_key(|(start, end, _)| (*start, *end));
        for pair in ranges.windows(2) {
            let (_, previous_end, previous_id) = pair[0];
            let (current_start, _, current_id) = pair[1];
            if current_start < previous_end {
                return Err(invalid(
                    format!("memory_evidence.snapshots[{snapshot_index}].resources"),
                    format!(
                        "COMPLETE inventory has overlapping sibling suballocations {previous_id} and {current_id} in {backing_id}"
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// Validates the complete semantic contract of one memory-evidence capture.
pub fn validate_memory_evidence(
    evidence: &MemoryEvidence,
) -> Result<(), MemoryEvidenceValidationError> {
    if evidence.format != MEMORY_EVIDENCE_FORMAT {
        return Err(invalid(
            "memory_evidence.format",
            format!("must be exactly {MEMORY_EVIDENCE_FORMAT}"),
        ));
    }
    if !nonblank(&evidence.capture_id) {
        return Err(invalid("memory_evidence.capture_id", "must not be empty"));
    }

    let mut required_features = HashSet::new();
    for (feature_index, feature) in evidence.required_features.iter().enumerate() {
        let path = format!("memory_evidence.required_features[{feature_index}]");
        if !valid_feature_token(feature) {
            return Err(invalid(path, "must be a canonical lowercase feature token"));
        }
        if !required_features.insert(feature.as_str()) {
            return Err(invalid(path, "duplicate required feature"));
        }
        if !SUPPORTED_MEMORY_EVIDENCE_FEATURES.contains(&feature.as_str()) {
            return Err(invalid(
                path,
                format!("unsupported required feature {feature}"),
            ));
        }
    }
    if evidence.snapshots.is_empty() {
        return Err(invalid("memory_evidence.snapshots", "must not be empty"));
    }

    let mut previous_sequence = None;
    let mut previous_time = None;
    let mut resource_descriptors: HashMap<String, (ResourceDescriptor, String)> = HashMap::new();
    let mut occurrences: HashMap<String, PhaseOccurrenceDescriptor> = HashMap::new();
    let mut singleton_occurrence_points = HashSet::new();

    for (snapshot_index, snapshot) in evidence.snapshots.iter().enumerate() {
        let snapshot_path = format!("memory_evidence.snapshots[{snapshot_index}]");
        if previous_sequence.is_some_and(|previous| snapshot.sequence <= previous) {
            return Err(invalid(
                format!("{snapshot_path}.sequence"),
                "must be strictly increasing",
            ));
        }
        previous_sequence = Some(snapshot.sequence);
        if let Some(time) = snapshot.monotonic_time.as_ref() {
            if previous_time.is_some_and(|previous| time.nanoseconds < previous) {
                return Err(invalid(
                    format!("{snapshot_path}.monotonic_time.nanoseconds"),
                    "must be non-decreasing",
                ));
            }
            previous_time = Some(time.nanoseconds);
        }

        known_nonzero_enum::<OperationStage>(snapshot.stage, &format!("{snapshot_path}.stage"))?;
        let point = known_nonzero_enum::<MemorySnapshotPoint>(
            snapshot.point,
            &format!("{snapshot_path}.point"),
        )?;
        let inventory = known_nonzero_enum::<MemoryInventoryKind>(
            snapshot.resource_inventory,
            &format!("{snapshot_path}.resource_inventory"),
        )?;
        if !nonblank(&snapshot.phase_occurrence_id) {
            return Err(invalid(
                format!("{snapshot_path}.phase_occurrence_id"),
                "must not be empty",
            ));
        }
        let subject = required_owner(
            snapshot.subject.as_ref(),
            &format!("{snapshot_path}.subject"),
        )?;
        let occurrence = PhaseOccurrenceDescriptor {
            stage: snapshot.stage,
            subject,
            backend: snapshot.backend.clone(),
            device: snapshot.device.clone(),
        };
        if let Some(previous) = occurrences.get(&snapshot.phase_occurrence_id) {
            if previous != &occurrence {
                return Err(invalid(
                    format!("{snapshot_path}.phase_occurrence_id"),
                    "reused occurrence has different stage, subject, backend, or device",
                ));
            }
        } else {
            occurrences.insert(snapshot.phase_occurrence_id.clone(), occurrence);
        }
        if point != MemorySnapshotPoint::Periodic
            && !singleton_occurrence_points
                .insert((snapshot.phase_occurrence_id.as_str(), snapshot.point))
        {
            return Err(invalid(
                format!("{snapshot_path}.point"),
                "duplicate singleton point for one phase occurrence",
            ));
        }

        if let Some(attestation) = snapshot.domain_attestation.as_ref() {
            validate_domain_attestation(
                attestation,
                &format!("{snapshot_path}.domain_attestation"),
            )?;
        }

        let mut snapshot_resource_ids = HashSet::new();
        for (resource_index, resource) in snapshot.resources.iter().enumerate() {
            let resource_path = format!("{snapshot_path}.resources[{resource_index}]");
            if !snapshot_resource_ids.insert(resource.resource_id.as_str()) {
                return Err(invalid(&resource_path, "duplicate resource_id in snapshot"));
            }
            let descriptor = validate_resource(resource, &resource_path)?;
            if let Some((previous, first_path)) =
                resource_descriptors.get(resource.resource_id.as_str())
            {
                if previous != &descriptor {
                    return Err(invalid(
                        &resource_path,
                        format!("immutable descriptor differs from {first_path}"),
                    ));
                }
            } else {
                resource_descriptors
                    .insert(resource.resource_id.clone(), (descriptor, resource_path));
            }
        }
        let mut envelope_axes = HashSet::new();
        for (envelope_index, envelope) in snapshot.envelopes.iter().enumerate() {
            let envelope_path = format!("{snapshot_path}.envelopes[{envelope_index}]");
            validate_envelope(envelope, &envelope_path)?;
            if !envelope_axes.insert((envelope.kind, envelope.sampler.as_str())) {
                return Err(invalid(&envelope_path, "duplicate envelope kind/sampler"));
            }
        }
        if inventory == MemoryInventoryKind::Complete {
            for (resource_index, resource) in snapshot.resources.iter().enumerate() {
                if resource.backing_relation != MemoryBackingRelation::Independent as i32
                    && !snapshot_resource_ids.contains(resource.backing_resource_id.as_str())
                {
                    return Err(invalid(
                        format!("{snapshot_path}.resources[{resource_index}].backing_resource_id"),
                        "COMPLETE inventory must contain the immediate backing resource",
                    ));
                }
            }
            validate_complete_suballocations(snapshot, snapshot_index)?;
        }
    }

    validate_backing_graph(&resource_descriptors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pb::{
        MemoryBoundProof, MemoryBoundTerm, MemoryByteSize, MemoryMonotonicTime, MemoryPeakCase,
    };

    fn bytes(value: u64) -> Option<MemoryByteSize> {
        Some(MemoryByteSize { bytes: value })
    }

    fn owner(kind: MemoryOwnerKind, owner_id: &str) -> MemoryOwnerRef {
        MemoryOwnerRef {
            kind: kind as i32,
            owner_id: owner_id.to_owned(),
        }
    }

    fn measurement(
        metric: MemoryMetric,
        value: u64,
        source: MemoryEvidenceSource,
        relation: MemoryValueRelation,
        coverage: MemoryTemporalCoverage,
    ) -> MemoryMeasurement {
        MemoryMeasurement {
            metric: metric as i32,
            bytes: bytes(value),
            source: source as i32,
            value_relation: relation as i32,
            temporal_coverage: coverage as i32,
        }
    }

    fn root_resource() -> MemoryResourceEvidence {
        MemoryResourceEvidence {
            resource_id: "arena-1".to_owned(),
            backing_resource_id: String::new(),
            backing_relation: MemoryBackingRelation::Independent as i32,
            backing_offset_bytes: None,
            backing_length_bytes: None,
            owner: Some(owner(MemoryOwnerKind::CompiledModel, "41")),
            role: MemoryResourceRole::Arena as i32,
            space: MemorySpace::NativeHeap as i32,
            allocator: "malloc".to_owned(),
            consumers: vec![owner(MemoryOwnerKind::ExecutionContext, "52")],
            measurements: vec![measurement(
                MemoryMetric::Live,
                100,
                MemoryEvidenceSource::AllocatorCounter,
                MemoryValueRelation::Exact,
                MemoryTemporalCoverage::Instant,
            )],
            addressable_bytes: bytes(100),
        }
    }

    fn child_resource(id: &str, offset: u64, length: u64) -> MemoryResourceEvidence {
        MemoryResourceEvidence {
            resource_id: id.to_owned(),
            backing_resource_id: "arena-1".to_owned(),
            backing_relation: MemoryBackingRelation::Suballocation as i32,
            backing_offset_bytes: bytes(offset),
            backing_length_bytes: bytes(length),
            owner: Some(owner(MemoryOwnerKind::ExecutionContext, "52")),
            role: MemoryResourceRole::Activation as i32,
            space: MemorySpace::NativeHeap as i32,
            allocator: "arena".to_owned(),
            consumers: Vec::new(),
            measurements: vec![measurement(
                MemoryMetric::Logical,
                length,
                MemoryEvidenceSource::ApiRequest,
                MemoryValueRelation::Requested,
                MemoryTemporalCoverage::Instant,
            )],
            addressable_bytes: bytes(length),
        }
    }

    fn attestation() -> MemoryDomainAttestation {
        MemoryDomainAttestation {
            proof_protocol: MEMORY_EVIDENCE_PROOF_PROTOCOL.to_owned(),
            resource_protocol: MEMORY_EVIDENCE_RESOURCE_PROTOCOL.to_owned(),
            graph_fingerprint: "graph-1".to_owned(),
            shape_domain_proof_identity: "shape-1".to_owned(),
            bounds: vec![MemoryBoundProof {
                budget_domain_id: "provider-resident".to_owned(),
                kind: MemoryBoundKind::OrdinaryResident as i32,
                cases: vec![MemoryPeakCase {
                    case_id: "ordinary".to_owned(),
                    terms: vec![
                        MemoryBoundTerm {
                            term_id: "weights".to_owned(),
                            owner_kind: MemoryOwnerKind::CompiledModel as i32,
                            role: MemoryResourceRole::Weights as i32,
                            space: MemorySpace::NativeHeap as i32,
                            upper_bound_bytes: bytes(60),
                        },
                        MemoryBoundTerm {
                            term_id: "activation".to_owned(),
                            owner_kind: MemoryOwnerKind::ExecutionContext as i32,
                            role: MemoryResourceRole::Activation as i32,
                            space: MemorySpace::NativeHeap as i32,
                            upper_bound_bytes: bytes(40),
                        },
                    ],
                    total_bytes: bytes(100),
                }],
                maximum_bytes: bytes(100),
                limit_bytes: bytes(120),
            }],
        }
    }

    fn snapshot(sequence: u64, nanoseconds: u64, point: MemorySnapshotPoint) -> MemorySnapshot {
        MemorySnapshot {
            sequence,
            monotonic_time: Some(MemoryMonotonicTime { nanoseconds }),
            stage: OperationStage::Execute as i32,
            point: point as i32,
            phase_occurrence_id: "execute-1".to_owned(),
            subject: Some(owner(MemoryOwnerKind::ExecutionContext, "52")),
            backend: "cpu".to_owned(),
            device: "host".to_owned(),
            domain_attestation: (point == MemorySnapshotPoint::After).then(attestation),
            resources: vec![root_resource(), child_resource("activation-1", 10, 20)],
            envelopes: vec![MemoryEnvelopeEvidence {
                kind: MemoryEnvelopeKind::ProcessRss as i32,
                bytes: bytes(1_000),
                source: MemoryEvidenceSource::OsSampler as i32,
                value_relation: MemoryValueRelation::Exact as i32,
                temporal_coverage: MemoryTemporalCoverage::Instant as i32,
                sampler: "procfs-statm".to_owned(),
            }],
            resource_inventory: MemoryInventoryKind::Complete as i32,
        }
    }

    fn valid_evidence() -> MemoryEvidence {
        MemoryEvidence {
            format: MEMORY_EVIDENCE_FORMAT.to_owned(),
            capture_id: "capture-1".to_owned(),
            required_features: Vec::new(),
            snapshots: vec![
                snapshot(1, 100, MemorySnapshotPoint::Before),
                snapshot(2, 200, MemorySnapshotPoint::After),
            ],
        }
    }

    fn assert_invalid(evidence: &MemoryEvidence, needle: &str) {
        let error = validate_memory_evidence(evidence).expect_err("evidence must be rejected");
        assert!(
            error.to_string().contains(needle),
            "expected {needle:?} in {error}"
        );
    }

    #[test]
    fn valid_evidence_passes() {
        validate_memory_evidence(&valid_evidence()).unwrap();
    }

    #[test]
    fn format_capture_and_feature_contract_is_fail_closed() {
        let mut evidence = valid_evidence();
        evidence.format = "volvoxai-memory-evidence/v2".to_owned();
        assert_invalid(&evidence, "format");

        let mut evidence = valid_evidence();
        evidence.capture_id.clear();
        assert_invalid(&evidence, "capture_id");

        let mut evidence = valid_evidence();
        evidence.capture_id = " \t\n\u{000b}\u{000c}\r".to_owned();
        assert_invalid(&evidence, "capture_id");

        for capture_id in ["\u{0085}", "\u{feff}"] {
            let mut evidence = valid_evidence();
            evidence.capture_id = capture_id.to_owned();
            validate_memory_evidence(&evidence).unwrap();
        }

        let mut evidence = valid_evidence();
        evidence
            .required_features
            .push("future-meaning-v1".to_owned());
        assert_invalid(&evidence, "unsupported required feature");

        let mut evidence = valid_evidence();
        evidence.required_features.push("Bad Feature".to_owned());
        assert_invalid(&evidence, "canonical lowercase feature token");

        let mut evidence = valid_evidence();
        evidence.snapshots.clear();
        assert_invalid(&evidence, "snapshots: must not be empty");
    }

    #[test]
    fn attestation_requires_canonical_protocols_and_nonempty_proof_parts() {
        let mut evidence = valid_evidence();
        evidence.snapshots[1]
            .domain_attestation
            .as_mut()
            .unwrap()
            .proof_protocol = "other-proof/v1".to_owned();
        assert_invalid(&evidence, MEMORY_EVIDENCE_PROOF_PROTOCOL);

        let mut evidence = valid_evidence();
        evidence.snapshots[1]
            .domain_attestation
            .as_mut()
            .unwrap()
            .resource_protocol = "other-resource/v1".to_owned();
        assert_invalid(&evidence, MEMORY_EVIDENCE_RESOURCE_PROTOCOL);

        let mut evidence = valid_evidence();
        evidence.snapshots[1]
            .domain_attestation
            .as_mut()
            .unwrap()
            .bounds
            .clear();
        assert_invalid(&evidence, "bounds: must not be empty");

        let mut evidence = valid_evidence();
        evidence.snapshots[1]
            .domain_attestation
            .as_mut()
            .unwrap()
            .bounds[0]
            .cases[0]
            .terms
            .clear();
        assert_invalid(&evidence, "terms: must not be empty");
    }

    #[test]
    fn sequence_and_present_times_are_monotonic() {
        let mut evidence = valid_evidence();
        evidence.snapshots[1].sequence = 1;
        assert_invalid(&evidence, "strictly increasing");

        let mut evidence = valid_evidence();
        evidence.snapshots[1].monotonic_time = Some(MemoryMonotonicTime { nanoseconds: 99 });
        assert_invalid(&evidence, "non-decreasing");

        let mut evidence = valid_evidence();
        evidence.snapshots[1].subject = Some(owner(MemoryOwnerKind::ExecutionContext, "different"));
        assert_invalid(&evidence, "reused occurrence has different");

        let mut evidence = valid_evidence();
        evidence.snapshots[1].backend = "different".to_owned();
        assert_invalid(&evidence, "reused occurrence has different");

        let mut evidence = valid_evidence();
        evidence
            .snapshots
            .push(snapshot(3, 300, MemorySnapshotPoint::Before));
        assert_invalid(&evidence, "duplicate singleton point");

        let mut evidence = MemoryEvidence {
            format: MEMORY_EVIDENCE_FORMAT.to_owned(),
            capture_id: "capture-periodic".to_owned(),
            required_features: Vec::new(),
            snapshots: vec![
                snapshot(1, 100, MemorySnapshotPoint::Periodic),
                snapshot(2, 200, MemorySnapshotPoint::Periodic),
            ],
        };
        validate_memory_evidence(&evidence).unwrap();
        evidence.snapshots[1].device = "different".to_owned();
        assert_invalid(&evidence, "reused occurrence has different");
    }

    #[test]
    fn unknown_and_unspecified_enums_are_rejected() {
        let mut evidence = valid_evidence();
        evidence.snapshots[0].resource_inventory = MemoryInventoryKind::Unspecified as i32;
        assert_invalid(&evidence, "resource_inventory");

        let mut evidence = valid_evidence();
        evidence.snapshots[0].resources[0].space = 99_999;
        assert_invalid(&evidence, "unknown enum value");
    }

    #[test]
    fn bound_proof_checks_ids_sums_maxima_overflow_and_limits() {
        let mut evidence = valid_evidence();
        let bound = &mut evidence.snapshots[1]
            .domain_attestation
            .as_mut()
            .unwrap()
            .bounds[0];
        bound.cases[0].total_bytes = bytes(99);
        assert_invalid(&evidence, "computed 100");

        let mut evidence = valid_evidence();
        let bound = &mut evidence.snapshots[1]
            .domain_attestation
            .as_mut()
            .unwrap()
            .bounds[0];
        bound.maximum_bytes = bytes(99);
        assert_invalid(&evidence, "computed 100");

        let mut evidence = valid_evidence();
        let bound = &mut evidence.snapshots[1]
            .domain_attestation
            .as_mut()
            .unwrap()
            .bounds[0];
        bound.limit_bytes = bytes(99);
        assert_invalid(&evidence, "exceeds limit");

        let mut evidence = valid_evidence();
        let bound = &mut evidence.snapshots[1]
            .domain_attestation
            .as_mut()
            .unwrap()
            .bounds[0];
        bound.cases[0].terms[0].upper_bound_bytes = bytes(u64::MAX);
        bound.cases[0].terms[1].upper_bound_bytes = bytes(1);
        assert_invalid(&evidence, "overflows u64");

        let mut evidence = valid_evidence();
        let bound = &mut evidence.snapshots[1]
            .domain_attestation
            .as_mut()
            .unwrap()
            .bounds[0];
        bound.cases[0].terms[1].term_id = "weights".to_owned();
        assert_invalid(&evidence, "duplicate term_id");
    }

    #[test]
    fn backing_graph_requires_resolved_acyclic_contained_ranges() {
        let mut evidence = valid_evidence();
        for snapshot in &mut evidence.snapshots {
            snapshot.resource_inventory = MemoryInventoryKind::Partial as i32;
        }
        evidence.snapshots[0].resources[1].backing_resource_id = "missing".to_owned();
        evidence.snapshots[1].resources[1].backing_resource_id = "missing".to_owned();
        assert_invalid(&evidence, "does not resolve");

        let mut evidence = valid_evidence();
        for snapshot in &mut evidence.snapshots {
            snapshot.resources[1].backing_offset_bytes = bytes(90);
        }
        assert_invalid(&evidence, "exceeds backing resource extent");

        let mut evidence = valid_evidence();
        for snapshot in &mut evidence.snapshots {
            snapshot.resources[1].backing_offset_bytes = bytes(u64::MAX);
            snapshot.resources[1].backing_length_bytes = bytes(1);
            snapshot.resources[1].addressable_bytes = bytes(1);
        }
        assert_invalid(&evidence, "range end overflows u64");

        let mut evidence = valid_evidence();
        for snapshot in &mut evidence.snapshots {
            let root = &mut snapshot.resources[0];
            root.backing_relation = MemoryBackingRelation::Alias as i32;
            root.backing_resource_id = "activation-1".to_owned();
            root.backing_offset_bytes = bytes(0);
            root.backing_length_bytes = bytes(20);
            root.addressable_bytes = bytes(20);
            let child = &mut snapshot.resources[1];
            child.backing_offset_bytes = bytes(0);
        }
        assert_invalid(&evidence, "backing cycle");
    }

    #[test]
    fn addressable_geometry_and_descriptors_are_immutable() {
        let mut evidence = valid_evidence();
        evidence.snapshots[0].resources[0].addressable_bytes = None;
        assert_invalid(&evidence, "addressable_bytes");

        let mut evidence = valid_evidence();
        evidence.snapshots[0].resources[1].addressable_bytes = bytes(19);
        assert_invalid(&evidence, "must equal backing_length_bytes");

        let mut evidence = valid_evidence();
        evidence.snapshots[1].resources[0].allocator = "different".to_owned();
        assert_invalid(&evidence, "immutable descriptor differs");

        let mut evidence = valid_evidence();
        for snapshot in &mut evidence.snapshots {
            snapshot.resources.truncate(1);
            snapshot.resources[0].addressable_bytes = bytes(0);
            snapshot.resources[0].measurements[0].bytes = bytes(0);
        }
        validate_memory_evidence(&evidence).unwrap();
    }

    #[test]
    fn resources_require_identity_measurements_and_complete_backing_membership() {
        let mut evidence = valid_evidence();
        evidence.snapshots[0].resources[0].allocator.clear();
        assert_invalid(&evidence, "allocator: must not be empty");

        let mut evidence = valid_evidence();
        evidence.snapshots[0].resources[0].measurements.clear();
        assert_invalid(&evidence, "measurements: must not be empty");

        let mut evidence = valid_evidence();
        evidence.snapshots[0].resources.remove(0);
        assert_invalid(&evidence, "COMPLETE inventory must contain");

        let mut evidence = valid_evidence();
        let duplicate = evidence.snapshots[0].resources[0].measurements[0].clone();
        evidence.snapshots[0].resources[0]
            .measurements
            .push(duplicate);
        assert_invalid(&evidence, "duplicate measurement axis");
    }

    #[test]
    fn only_complete_inventory_requires_disjoint_sibling_suballocations() {
        let mut evidence = valid_evidence();
        for snapshot in &mut evidence.snapshots {
            snapshot
                .resources
                .push(child_resource("activation-2", 15, 20));
        }
        assert_invalid(&evidence, "overlapping sibling suballocations");

        for snapshot in &mut evidence.snapshots {
            snapshot.resource_inventory = MemoryInventoryKind::Partial as i32;
        }
        validate_memory_evidence(&evidence).unwrap();
    }

    #[test]
    fn unavailable_is_exactly_the_absence_of_bytes() {
        let mut evidence = valid_evidence();
        let measurement = &mut evidence.snapshots[0].resources[0].measurements[0];
        measurement.value_relation = MemoryValueRelation::Unavailable as i32;
        assert_invalid(&evidence, "absent exactly");

        let mut evidence = valid_evidence();
        let envelope = &mut evidence.snapshots[0].envelopes[0];
        envelope.bytes = None;
        assert_invalid(&evidence, "absent exactly");
    }

    #[test]
    fn resource_measurements_enforce_source_and_requested_pairing() {
        let mut evidence = valid_evidence();
        let measurement = &mut evidence.snapshots[0].resources[0].measurements[0];
        measurement.source = MemoryEvidenceSource::OsSampler as i32;
        assert_invalid(&evidence, "resource measurements require");

        let mut evidence = valid_evidence();
        let measurement = &mut evidence.snapshots[0].resources[0].measurements[0];
        measurement.value_relation = MemoryValueRelation::Requested as i32;
        assert_invalid(&evidence, "must occur together");
    }

    #[test]
    fn envelope_kind_controls_source_and_requested_is_forbidden() {
        let mut evidence = valid_evidence();
        evidence.snapshots[0].envelopes[0].source = MemoryEvidenceSource::DriverSampler as i32;
        assert_invalid(&evidence, "incompatible with envelope kind");

        let mut evidence = valid_evidence();
        evidence.snapshots[0].envelopes[0].value_relation = MemoryValueRelation::Requested as i32;
        assert_invalid(&evidence, "cannot use REQUESTED");

        let mut evidence = valid_evidence();
        evidence.snapshots[0].envelopes[0].sampler.clear();
        assert_invalid(&evidence, "sampler: must not be empty");

        let mut evidence = valid_evidence();
        let duplicate = evidence.snapshots[0].envelopes[0].clone();
        evidence.snapshots[0].envelopes.push(duplicate);
        assert_invalid(&evidence, "duplicate envelope kind/sampler");
    }

    #[test]
    fn temporal_aggregates_cannot_claim_instant_and_samples_are_lower_bounds() {
        let mut evidence = valid_evidence();
        let measurement = &mut evidence.snapshots[0].resources[0].measurements[0];
        measurement.metric = MemoryMetric::LiveHighWater as i32;
        assert_invalid(&evidence, "cannot have INSTANT");

        let mut evidence = valid_evidence();
        let envelope = &mut evidence.snapshots[0].envelopes[0];
        envelope.temporal_coverage = MemoryTemporalCoverage::SampledWindow as i32;
        assert_invalid(&evidence, "must be LOWER_BOUND");

        let mut evidence = valid_evidence();
        let envelope = &mut evidence.snapshots[0].envelopes[0];
        envelope.temporal_coverage = MemoryTemporalCoverage::SampledWindow as i32;
        envelope.value_relation = MemoryValueRelation::LowerBound as i32;
        validate_memory_evidence(&evidence).unwrap();

        let mut evidence = valid_evidence();
        let measurement = &mut evidence.snapshots[0].resources[0].measurements[0];
        measurement.temporal_coverage = MemoryTemporalCoverage::SampledWindow as i32;
        measurement.value_relation = MemoryValueRelation::LowerBound as i32;
        assert_invalid(&evidence, "must be a high-water metric");

        let mut evidence = valid_evidence();
        let measurement = &mut evidence.snapshots[0].resources[0].measurements[0];
        measurement.metric = MemoryMetric::LiveHighWater as i32;
        measurement.temporal_coverage = MemoryTemporalCoverage::SampledWindow as i32;
        measurement.value_relation = MemoryValueRelation::LowerBound as i32;
        validate_memory_evidence(&evidence).unwrap();
    }

    #[test]
    fn process_peak_rss_is_lifetime_exact_or_unavailable() {
        let mut evidence = valid_evidence();
        evidence.snapshots[0].envelopes[0].kind = MemoryEnvelopeKind::ProcessPeakRss as i32;
        assert_invalid(&evidence, "PROCESS_PEAK_RSS");

        let mut evidence = valid_evidence();
        let envelope = &mut evidence.snapshots[0].envelopes[0];
        envelope.kind = MemoryEnvelopeKind::ProcessPeakRss as i32;
        envelope.temporal_coverage = MemoryTemporalCoverage::ProcessLifetime as i32;
        validate_memory_evidence(&evidence).unwrap();

        let mut evidence = valid_evidence();
        let envelope = &mut evidence.snapshots[0].envelopes[0];
        envelope.kind = MemoryEnvelopeKind::ProcessPeakRss as i32;
        envelope.bytes = None;
        envelope.value_relation = MemoryValueRelation::Unavailable as i32;
        envelope.temporal_coverage = MemoryTemporalCoverage::ProcessLifetime as i32;
        validate_memory_evidence(&evidence).unwrap();
    }
}
