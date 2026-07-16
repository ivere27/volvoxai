//! Durable model persistence: atomic staged file and directory commits,
//! safetensors writing, and training-checkpoint parsing/validation. All writes
//! stage to a sibling path and fsync before renaming into place so a crash
//! never leaves a half-written checkpoint.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use serde_json::Value;

use crate::ffi::FfiError;
use crate::pb::*;
use crate::{
    err, optimizer_values, read_safetensors_info, strict_metadata_map, string_entry_value,
    DATA_TYPE_F32, INTERNAL, INVALID_ARGUMENT, NATIVE_MAX_TENSORS, NOT_FOUND, PERSIST_SEQUENCE,
};

pub(crate) fn unique_sibling_path(target: &Path, label: &str) -> Result<PathBuf, FfiError> {
    let parent = target
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = target
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| err("persistence path has no valid filename", INVALID_ARGUMENT))?;
    for _ in 0..16 {
        let sequence = PERSIST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(
            ".{name}.volvox.{label}.{}.{}",
            std::process::id(),
            sequence
        ));
        if !candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(err(
        "could not allocate a unique persistence staging path",
        INTERNAL,
    ))
}

pub(crate) fn sync_file(path: &Path) -> Result<(), FfiError> {
    fs::File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|e| err(format!("fsync {} failed: {e}", path.display()), INTERNAL))
}

pub(crate) fn sync_parent(path: &Path) -> Result<(), FfiError> {
    let parent = path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|e| {
            err(
                format!("fsync directory {} failed: {e}", parent.display()),
                INTERNAL,
            )
        })
}

pub(crate) fn commit_staged_files(staged: Vec<(PathBuf, PathBuf)>) -> Result<(), FfiError> {
    struct CommitEntry {
        stage: PathBuf,
        target: PathBuf,
        backup: Option<PathBuf>,
        committed: bool,
    }

    let mut targets = std::collections::HashSet::new();
    let mut entries: Vec<CommitEntry> = Vec::with_capacity(staged.len());
    for (stage, target) in staged {
        if !targets.insert(target.clone()) {
            for entry in &entries {
                let _ = fs::remove_file(&entry.stage);
            }
            let _ = fs::remove_file(&stage);
            return Err(err(
                format!("duplicate persistence destination {}", target.display()),
                INVALID_ARGUMENT,
            ));
        }
        if let Err(error) = sync_file(&stage) {
            for entry in &entries {
                let _ = fs::remove_file(&entry.stage);
            }
            let _ = fs::remove_file(&stage);
            return Err(error);
        }
        entries.push(CommitEntry {
            stage,
            target,
            backup: None,
            committed: false,
        });
    }

    let rollback = |entries: &mut [CommitEntry]| {
        let mut failures = Vec::new();
        for entry in entries.iter_mut().rev() {
            if entry.committed && entry.target.exists() {
                if let Err(error) = fs::remove_file(&entry.target) {
                    failures.push(format!("remove {}: {error}", entry.target.display()));
                }
            }
            if let Some(backup) = entry.backup.take() {
                if let Err(error) = fs::rename(&backup, &entry.target) {
                    failures.push(format!(
                        "restore {} from {}: {error}",
                        entry.target.display(),
                        backup.display()
                    ));
                }
            }
            let _ = fs::remove_file(&entry.stage);
            let _ = sync_parent(&entry.target);
        }
        failures
    };

    for index in 0..entries.len() {
        if entries[index].target.exists() {
            let backup = match unique_sibling_path(&entries[index].target, "backup") {
                Ok(backup) => backup,
                Err(error) => {
                    let failures = rollback(&mut entries);
                    return Err(err(
                        format!(
                            "prepare atomic persistence failed: {error}; rollback: {failures:?}"
                        ),
                        INTERNAL,
                    ));
                }
            };
            if let Err(error) = fs::rename(&entries[index].target, &backup) {
                let failures = rollback(&mut entries);
                return Err(err(
                    format!(
                        "prepare atomic persistence for {} failed: {error}; rollback: {:?}",
                        entries[index].target.display(),
                        failures
                    ),
                    INTERNAL,
                ));
            }
            entries[index].backup = Some(backup);
        }
    }

    for index in 0..entries.len() {
        if let Err(error) = fs::rename(&entries[index].stage, &entries[index].target) {
            let target = entries[index].target.display().to_string();
            let failures = rollback(&mut entries);
            return Err(err(
                format!("commit atomic persistence for {target} failed: {error}; rollback: {failures:?}"),
                INTERNAL,
            ));
        }
        entries[index].committed = true;
    }
    for index in 0..entries.len() {
        if let Err(error) = sync_parent(&entries[index].target) {
            let failures = rollback(&mut entries);
            return Err(err(
                format!(
                    "atomic persistence directory sync failed: {error}; rollback: {failures:?}"
                ),
                INTERNAL,
            ));
        }
    }
    for entry in &mut entries {
        if let Some(backup) = entry.backup.take() {
            fs::remove_file(&backup).map_err(|error| {
                err(
                    format!(
                        "remove persistence backup {} failed: {error}",
                        backup.display()
                    ),
                    INTERNAL,
                )
            })?;
        }
        sync_parent(&entry.target)?;
    }
    Ok(())
}

pub(crate) const TRAINING_CHECKPOINT_FORMAT: &str = "volvox.training_checkpoint.v1";

pub(crate) struct ParsedTrainingCheckpoint {
    pub config: PathBuf,
    pub weights: Vec<PathBuf>,
    pub optimizer: PathBuf,
    pub tokenizer: Option<PathBuf>,
    pub training_step: i64,
    pub training_optimizer: Option<OptimizerOptions>,
    pub training_update_mode: Option<i32>,
    pub metadata: std::collections::HashMap<String, String>,
}

pub(crate) fn write_synced_bytes(path: &Path, data: &[u8], context: &str) -> Result<(), FfiError> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| err(format!("create {context} failed: {error}"), INTERNAL))?;
    file.write_all(data)
        .map_err(|error| err(format!("write {context} failed: {error}"), INTERNAL))?;
    file.sync_all()
        .map_err(|error| err(format!("fsync {context} failed: {error}"), INTERNAL))?;
    Ok(())
}

pub(crate) fn checkpoint_member(directory: &Path, name: &str, context: &str) -> Result<PathBuf, FfiError> {
    use std::path::Component;
    let path = Path::new(name);
    let mut components = path.components();
    match (components.next(), components.next()) {
        (Some(Component::Normal(_)), None) => Ok(directory.join(path)),
        _ => Err(err(
            format!("{context} must be a single relative filename"),
            INVALID_ARGUMENT,
        )),
    }
}

pub(crate) fn parse_training_checkpoint(directory: &Path) -> Result<ParsedTrainingCheckpoint, FfiError> {
    let manifest_path = directory.join("manifest.json");
    let manifest: Value = serde_json::from_slice(
        &fs::read(&manifest_path)
            .map_err(|error| err(format!("read training checkpoint manifest failed: {error}"), NOT_FOUND))?,
    )
    .map_err(|error| err(format!("invalid training checkpoint manifest: {error}"), INVALID_ARGUMENT))?;
    let object = manifest
        .as_object()
        .ok_or_else(|| err("training checkpoint manifest must be an object", INVALID_ARGUMENT))?;
    if object.get("format").and_then(Value::as_str) != Some(TRAINING_CHECKPOINT_FORMAT) {
        return Err(err("unsupported training checkpoint format", INVALID_ARGUMENT));
    }
    let training_step = object
        .get("training_step")
        .and_then(Value::as_i64)
        .filter(|step| *step >= 0)
        .ok_or_else(|| err("training checkpoint has invalid training_step", INVALID_ARGUMENT))?;
    let config = checkpoint_member(
        directory,
        object
            .get("config")
            .and_then(Value::as_str)
            .ok_or_else(|| err("training checkpoint is missing config", INVALID_ARGUMENT))?,
        "training checkpoint config",
    )?;
    let weights = object
        .get("weights")
        .and_then(Value::as_array)
        .ok_or_else(|| err("training checkpoint is missing weights", INVALID_ARGUMENT))?
        .iter()
        .map(|value| {
            checkpoint_member(
                directory,
                value
                    .as_str()
                    .ok_or_else(|| err("training checkpoint weight name must be a string", INVALID_ARGUMENT))?,
                "training checkpoint weight",
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    if weights.is_empty() {
        return Err(err("training checkpoint requires at least one weight shard", INVALID_ARGUMENT));
    }
    let optimizer = checkpoint_member(
        directory,
        object
            .get("optimizer")
            .and_then(Value::as_str)
            .ok_or_else(|| err("training checkpoint is missing optimizer state", INVALID_ARGUMENT))?,
        "training checkpoint optimizer",
    )?;
    let tokenizer = object
        .get("tokenizer")
        .filter(|value| !value.is_null())
        .map(|value| {
            checkpoint_member(
                directory,
                value
                    .as_str()
                    .ok_or_else(|| err("training checkpoint tokenizer must be a string", INVALID_ARGUMENT))?,
                "training checkpoint tokenizer",
            )
        })
        .transpose()?;
    let metadata = strict_metadata_map(object.get("metadata"), "training checkpoint")?;
    let training_optimizer = object
        .get("training_optimizer")
        .filter(|value| !value.is_null())
        .map(|value| -> Result<OptimizerOptions, FfiError> {
            let options = value.as_object().ok_or_else(|| {
                err("training checkpoint optimizer must be an object", INVALID_ARGUMENT)
            })?;
            let number = |name: &str| -> Result<Option<f32>, FfiError> {
                options
                    .get(name)
                    .filter(|value| !value.is_null())
                    .map(|value| {
                        value.as_f64().map(|value| value as f32).ok_or_else(|| {
                            err(
                                format!("training checkpoint optimizer {name} must be numeric"),
                                INVALID_ARGUMENT,
                            )
                        })
                    })
                    .transpose()
            };
            Ok(OptimizerOptions {
                learning_rate: number("learning_rate")?.ok_or_else(|| {
                    err("training checkpoint optimizer is missing learning_rate", INVALID_ARGUMENT)
                })?,
                beta1: number("beta1")?,
                beta2: number("beta2")?,
                epsilon: number("epsilon")?,
                weight_decay: number("weight_decay")?,
                max_grad_norm: number("max_grad_norm")?,
                step: None,
            })
        })
        .transpose()?;
    if let Some(options) = training_optimizer.as_ref() {
        optimizer_values(None, Some(options), training_step.max(1))?;
    }
    let training_update_mode = object
        .get("training_update_mode")
        .filter(|value| !value.is_null())
        .map(|value| {
            let mode = value.as_i64().ok_or_else(|| {
                err("training checkpoint update mode must be an integer", INVALID_ARGUMENT)
            })?;
            i32::try_from(mode)
                .ok()
                .and_then(|mode| TensorUpdateMode::try_from(mode).ok())
                .filter(|mode| {
                    matches!(
                        mode,
                        TensorUpdateMode::TensorUpdateSgd | TensorUpdateMode::TensorUpdateAdamw
                    )
                })
                .map(|mode| mode as i32)
                .ok_or_else(|| {
                    err(
                        "training checkpoint update mode must be SGD or AdamW",
                        INVALID_ARGUMENT,
                    )
                })
        })
        .transpose()?;
    if training_optimizer.is_some() != training_update_mode.is_some() {
        return Err(err(
            "training checkpoint optimizer and update mode must be present together",
            INVALID_ARGUMENT,
        ));
    }
    fs::read(&config)
        .map_err(|error| err(format!("read checkpoint config failed: {error}"), NOT_FOUND))?;
    for weight in &weights {
        read_safetensors_info(&weight.to_string_lossy(), false)?;
    }
    read_safetensors_info(&optimizer.to_string_lossy(), true)?;
    if let Some(path) = tokenizer.as_ref() {
        fs::metadata(path)
            .map_err(|error| err(format!("checkpoint tokenizer is unavailable: {error}"), NOT_FOUND))?;
    }
    Ok(ParsedTrainingCheckpoint {
        config,
        weights,
        optimizer,
        tokenizer,
        training_step,
        training_optimizer,
        training_update_mode,
        metadata,
    })
}

pub(crate) fn validate_training_optimizer_checkpoint(
    parsed: &ParsedTrainingCheckpoint,
    weight_files: &[SafetensorsInfo],
) -> Result<SafetensorsInfo, FfiError> {
    let optimizer = read_safetensors_info(&parsed.optimizer.to_string_lossy(), true)?;
    if string_entry_value(&optimizer.metadata, "volvox.optimizer_format")
        != Some("volvox.optimizer.v1")
    {
        return Err(err("optimizer checkpoint has unsupported format", INVALID_ARGUMENT));
    }
    let optimizer_step = string_entry_value(&optimizer.metadata, "volvox.training_step")
        .and_then(|value| value.parse::<i64>().ok())
        .ok_or_else(|| err("optimizer checkpoint has invalid training step", INVALID_ARGUMENT))?;
    if optimizer_step != parsed.training_step {
        return Err(err(
            "optimizer checkpoint training step does not match manifest",
            INVALID_ARGUMENT,
        ));
    }
    let state_count = string_entry_value(&optimizer.metadata, "volvox.optimizer_state_count")
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|count| *count <= NATIVE_MAX_TENSORS)
        .ok_or_else(|| err("optimizer checkpoint has invalid state count", INVALID_ARGUMENT))?;
    if optimizer.tensors.len() != state_count.saturating_mul(2) {
        return Err(err(
            "optimizer checkpoint tensor count does not match its metadata",
            INVALID_ARGUMENT,
        ));
    }

    let config: Value = serde_json::from_slice(
        &fs::read(&parsed.config)
            .map_err(|error| err(format!("read checkpoint config failed: {error}"), NOT_FOUND))?,
    )
    .map_err(|error| err(format!("invalid checkpoint config: {error}"), INVALID_ARGUMENT))?;
    let graph_references: std::collections::HashSet<String> = config
        .get("nodes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|node| node.get("inputs").and_then(Value::as_object))
        .flat_map(|inputs| inputs.values())
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect();
    let mut active_weights = std::collections::HashMap::new();
    for file in weight_files {
        for tensor in &file.tensors {
            active_weights.insert(tensor.name.as_str(), tensor);
        }
    }
    let state_tensors: std::collections::HashMap<_, _> = optimizer
        .tensors
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect();
    let bytes = fs::read(&parsed.optimizer)
        .map_err(|error| err(format!("read optimizer checkpoint failed: {error}"), NOT_FOUND))?;
    if bytes.len() < 8 {
        return Err(err("optimizer checkpoint is truncated", INVALID_ARGUMENT));
    }
    let header_len = usize::try_from(u64::from_le_bytes(bytes[..8].try_into().unwrap()))
        .map_err(|_| err("optimizer checkpoint header is too large", INVALID_ARGUMENT))?;
    let data_base = 8usize
        .checked_add(header_len)
        .filter(|base| *base <= bytes.len())
        .ok_or_else(|| err("optimizer checkpoint header is out of range", INVALID_ARGUMENT))?;
    let mut target_names = std::collections::HashSet::new();
    for index in 0..state_count {
        let target_key = format!("volvox.optimizer_state.{index}.name");
        let target_name = string_entry_value(&optimizer.metadata, &target_key)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| err("optimizer checkpoint is missing a state target", INVALID_ARGUMENT))?;
        if !target_names.insert(target_name) {
            return Err(err("optimizer checkpoint repeats a state target", INVALID_ARGUMENT));
        }
        let target = active_weights.get(target_name).ok_or_else(|| {
            err(
                format!("optimizer state target {target_name} is not present in weight shards"),
                INVALID_ARGUMENT,
            )
        })?;
        if target.dtype != DATA_TYPE_F32 || !graph_references.contains(target_name) {
            return Err(err(
                format!("optimizer state target {target_name} is not an F32 graph parameter"),
                INVALID_ARGUMENT,
            ));
        }
        let elements = target.shape.iter().try_fold(1i64, |product, dimension| {
            product.checked_mul(*dimension).filter(|value| *value > 0)
        }).ok_or_else(|| err("optimizer target shape is invalid", INVALID_ARGUMENT))?;
        for suffix in ["m", "v"] {
            let name = format!("state.{index}.{suffix}");
            let state = state_tensors.get(name.as_str()).ok_or_else(|| {
                err(format!("optimizer checkpoint is missing {name}"), INVALID_ARGUMENT)
            })?;
            if state.dtype != DATA_TYPE_F32 || state.shape != [elements] {
                return Err(err(
                    format!("optimizer tensor {name} does not match target {target_name}"),
                    INVALID_ARGUMENT,
                ));
            }
            let start = data_base
                .checked_add(state.data_start as usize)
                .ok_or_else(|| err("optimizer tensor offset overflow", INVALID_ARGUMENT))?;
            let end = data_base
                .checked_add(state.data_end as usize)
                .filter(|end| *end <= bytes.len())
                .ok_or_else(|| err("optimizer tensor offset is out of range", INVALID_ARGUMENT))?;
            if bytes[start..end].chunks_exact(4).any(|chunk| {
                let value = f32::from_le_bytes(chunk.try_into().unwrap());
                !value.is_finite() || (suffix == "v" && value < 0.0)
            }) {
                return Err(err(
                    format!("optimizer tensor {name} contains an invalid moment value"),
                    INVALID_ARGUMENT,
                ));
            }
        }
    }
    Ok(optimizer)
}

pub(crate) fn commit_staged_directory(stage: &Path, target: &Path, overwrite: bool) -> Result<(), FfiError> {
    let parent = target.parent().filter(|path| !path.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .map_err(|error| err(format!("create checkpoint parent failed: {error}"), INTERNAL))?;
    let backup = if target.exists() {
        if !overwrite {
            return Err(err(
                format!("refusing to overwrite {}", target.display()),
                INVALID_ARGUMENT,
            ));
        }
        if !target.is_dir() {
            return Err(err("checkpoint target exists and is not a directory", INVALID_ARGUMENT));
        }
        let backup = unique_sibling_path(target, "backup")?;
        fs::rename(target, &backup)
            .map_err(|error| err(format!("stage checkpoint replacement failed: {error}"), INTERNAL))?;
        Some(backup)
    } else {
        None
    };
    if let Err(error) = fs::rename(stage, target) {
        if let Some(backup) = backup.as_ref() {
            let _ = fs::rename(backup, target);
        }
        return Err(err(format!("commit training checkpoint failed: {error}"), INTERNAL));
    }
    if let Err(error) = sync_parent(target) {
        let _ = fs::remove_dir_all(target);
        if let Some(backup) = backup.as_ref() {
            let _ = fs::rename(backup, target);
        }
        return Err(error);
    }
    if let Some(backup) = backup {
        fs::remove_dir_all(&backup)
            .map_err(|error| err(format!("remove checkpoint backup failed: {error}"), INTERNAL))?;
        sync_parent(target)?;
    }
    Ok(())
}
