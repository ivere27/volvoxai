use naga::back::glsl;
use naga::back::msl::{
    self, BindTarget, EntryPointResourceMap, EntryPointResources,
    PipelineOptions as MslPipelineOptions,
};
use naga::valid::{Capabilities, ModuleInfo, ValidationFlags, Validator};
use naga::{AddressSpace, ResourceBinding, ShaderStage, StorageAccess};
use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn parse_module(source: &str, label: &Path) -> Result<naga::Module, Box<dyn Error>> {
    naga::front::wgsl::parse_str(source)
        .map_err(|error| {
            invalid(format!(
                "{}:\n{}",
                label.display(),
                error.emit_to_string(source)
            ))
        })
        .map_err(Into::into)
}

fn validate_module(
    module: &naga::Module,
    label: &Path,
    capabilities: Capabilities,
) -> Result<ModuleInfo, Box<dyn Error>> {
    Validator::new(ValidationFlags::all(), capabilities)
        .validate(module)
        .map_err(|error| invalid(format!("{}: {error}", label.display())))
        .map_err(Into::into)
}

fn logical_binding_slots(
    module: &naga::Module,
) -> Result<BTreeMap<ResourceBinding, u8>, Box<dyn Error>> {
    let mut bindings = BTreeMap::new();
    for (_, global) in module.global_variables.iter() {
        let Some(binding) = global.binding else {
            continue;
        };
        if binding.group != 0 {
            return Err(invalid(format!(
                "native shader generator only supports bind group 0, found group {} binding {}",
                binding.group, binding.binding
            ))
            .into());
        }
        let slot = u8::try_from(binding.binding).map_err(|_| {
            invalid(format!(
                "native buffer binding {} exceeds u8",
                binding.binding
            ))
        })?;
        bindings.insert(binding, slot);
    }
    if bindings.is_empty() {
        return Err(invalid("shader has no bound resources").into());
    }
    Ok(bindings)
}

fn metal_binding_map(
    module: &naga::Module,
) -> Result<(BTreeMap<ResourceBinding, BindTarget>, u8), Box<dyn Error>> {
    let logical_slots = logical_binding_slots(module)?;
    let mut resources = BTreeMap::new();

    for (_, global) in module.global_variables.iter() {
        let Some(binding) = global.binding else {
            continue;
        };
        let slot = logical_slots[&binding];
        let mutable = matches!(
            global.space,
            AddressSpace::Storage { access } if access.contains(StorageAccess::STORE)
        );
        resources.insert(
            binding,
            BindTarget {
                buffer: Some(slot),
                mutable,
                ..BindTarget::default()
            },
        );
    }

    let maximum_binding = logical_slots
        .values()
        .copied()
        .max()
        .ok_or_else(|| invalid("shader has no bound resources"))?;
    let sizes_buffer = maximum_binding
        .checked_add(1)
        .ok_or_else(|| invalid("Metal sizes-buffer binding overflow"))?;
    Ok((resources, sizes_buffer))
}

fn translate_msl_shader(source: &str, label: &Path) -> Result<String, Box<dyn Error>> {
    let module = parse_module(source, label)?;
    let info = validate_module(&module, label, msl::supported_capabilities())?;
    let (resources, sizes_buffer) = metal_binding_map(&module)?;

    let mut per_entry_point_map = EntryPointResourceMap::new();
    for entry in &module.entry_points {
        per_entry_point_map.insert(
            entry.name.clone(),
            EntryPointResources {
                resources: resources.clone(),
                sizes_buffer: Some(sizes_buffer),
                ..EntryPointResources::default()
            },
        );
    }
    let options = msl::Options {
        per_entry_point_map,
        fake_missing_bindings: false,
        ..msl::Options::default()
    };
    let (metal, translation) =
        msl::write_string(&module, &info, &options, &MslPipelineOptions::default())
            .map_err(|error| invalid(format!("{}: {error}", label.display())))?;
    for (index, name) in translation.entry_point_names.iter().enumerate() {
        if let Err(error) = name {
            return Err(invalid(format!(
                "{}: entry point {}: {error}",
                label.display(),
                module.entry_points[index].name
            ))
            .into());
        }
    }
    Ok(metal)
}

fn translate_glsl_entry(
    module: &naga::Module,
    info: &ModuleInfo,
    label: &Path,
    entry_name: &str,
    version: glsl::Version,
    binding_map: &glsl::BindingMap,
) -> Result<String, Box<dyn Error>> {
    let options = glsl::Options {
        version,
        binding_map: binding_map.clone(),
        zero_initialize_workgroup_memory: glsl_zero_initialize_workgroup_memory(
            label, entry_name, version,
        ),
        ..glsl::Options::default()
    };
    let pipeline_options = glsl::PipelineOptions {
        shader_stage: ShaderStage::Compute,
        entry_point: entry_name.to_owned(),
        multiview: None,
    };
    let mut output = String::new();
    let mut writer = glsl::Writer::new(
        &mut output,
        module,
        info,
        &options,
        &pipeline_options,
        naga::proc::BoundsCheckPolicies::default(),
    )
    .map_err(|error| {
        invalid(format!(
            "{}: entry point {entry_name}, GLSL {version}: {error}",
            label.display()
        ))
    })?;
    writer.write().map_err(|error| {
        invalid(format!(
            "{}: entry point {entry_name}, GLSL {version}: {error}",
            label.display()
        ))
    })?;
    drop(writer);
    Ok(output)
}

fn glsl_zero_initialize_workgroup_memory(
    label: &Path,
    entry_name: &str,
    version: glsl::Version,
) -> bool {
    if !version.is_es() || entry_name != "main" {
        return true;
    }
    let Some(stem) = label.file_stem().and_then(|value| value.to_str()) else {
        return true;
    };

    /* Naga's portable default emits a lane-zero clear plus a workgroup
     * barrier for every workgroup variable. These inference kernels already
     * overwrite every element they can read before their first barrier. The
     * tiled linear and convolution kernels cover each complete tile (or its
     * exact live prefix), the normalization reductions cover every lane, and
     * qSDPA covers every lane/channel in its live head_dim range. Keep the
     * conservative default for every shader not audited here. */
    !matches!(
        stem,
        "linearF32Tiled"
            | "linearF32RowMajorTiled"
            | "linearInt8Tiled"
            | "qLinearInt8Tiled"
            | "qConv2DInt8Tiled"
            | "conv2DPointwise16Tile"
            | "qGroupNormStats"
            | "qLayerNormStats"
            | "qSDPAInt8"
    )
}

fn glsl_output_name(stem: &str, entry_name: &str, entry_count: usize) -> String {
    if entry_count == 1 {
        format!("{stem}.comp")
    } else {
        format!("{stem}_{entry_name}.comp")
    }
}

fn compile_msl_shader(input: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    let source = fs::read_to_string(input)?;
    let metal = translate_msl_shader(&source, input)?;
    fs::write(output, metal)?;
    Ok(())
}

fn compile_glsl_shader(
    input: &Path,
    core_output_dir: &Path,
    es_output_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let source = fs::read_to_string(input)?;
    let module = parse_module(&source, input)?;
    let info = validate_module(&module, input, glsl::supported_capabilities())?;
    let binding_map = logical_binding_slots(&module)?;
    if module.entry_points.is_empty() {
        return Err(invalid(format!("{}: shader has no entry points", input.display())).into());
    }
    for entry in &module.entry_points {
        if entry.stage != ShaderStage::Compute {
            return Err(invalid(format!(
                "{}: entry point {} is not a compute shader",
                input.display(),
                entry.name
            ))
            .into());
        }
    }

    let stem = input
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| invalid(format!("invalid shader path {}", input.display())))?;
    for entry in &module.entry_points {
        let output_name = glsl_output_name(stem, &entry.name, module.entry_points.len());
        let core = translate_glsl_entry(
            &module,
            &info,
            input,
            &entry.name,
            glsl::Version::Desktop(430),
            &binding_map,
        )?;
        fs::write(core_output_dir.join(&output_name), core)?;

        let es = translate_glsl_entry(
            &module,
            &info,
            input,
            &entry.name,
            glsl::Version::new_gles(310),
            &binding_map,
        )?;
        fs::write(es_output_dir.join(output_name), es)?;
    }
    Ok(())
}

fn shader_paths(
    input_dir: &Path,
    requested: Vec<OsString>,
) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    if !requested.is_empty() {
        let mut paths = Vec::with_capacity(requested.len());
        for name in requested {
            let name = PathBuf::from(name);
            if name.components().count() != 1 {
                return Err(invalid(format!(
                    "shader name must not contain a path: {}",
                    name.display()
                ))
                .into());
            }
            let mut path = input_dir.join(name);
            if path.extension().is_none() {
                path.set_extension("wgsl");
            }
            paths.push(path);
        }
        return Ok(paths);
    }
    let mut paths = Vec::new();
    for entry in fs::read_dir(input_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) == Some("wgsl") {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn required_arg(args: &mut impl Iterator<Item = OsString>) -> Result<PathBuf, Box<dyn Error>> {
    args.next()
        .map(PathBuf::from)
        .ok_or_else(|| invalid(usage()).into())
}

fn usage() -> &'static str {
    "usage:\n  volvoxai-native-shader-compiler glsl <wgsl-dir> <core430-dir> <es310-dir> [shader ...]\n  volvoxai-native-shader-compiler msl <wgsl-dir> <metal-dir> [shader ...]"
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os().skip(1);
    let command = args.next().ok_or_else(|| invalid(usage()))?;
    match command.to_str() {
        Some("glsl") => {
            let input_dir = required_arg(&mut args)?;
            let core_output_dir = required_arg(&mut args)?;
            let es_output_dir = required_arg(&mut args)?;
            let requested = args.collect::<Vec<_>>();
            fs::create_dir_all(&core_output_dir)?;
            fs::create_dir_all(&es_output_dir)?;
            for input in shader_paths(&input_dir, requested)? {
                compile_glsl_shader(&input, &core_output_dir, &es_output_dir)?;
            }
        }
        Some("msl") => {
            let input_dir = required_arg(&mut args)?;
            let output_dir = required_arg(&mut args)?;
            let requested = args.collect::<Vec<_>>();
            fs::create_dir_all(&output_dir)?;
            for input in shader_paths(&input_dir, requested)? {
                let stem = input
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .ok_or_else(|| invalid(format!("invalid shader path {}", input.display())))?;
                compile_msl_shader(&input, &output_dir.join(format!("{stem}.metal")))?;
            }
        }
        _ => return Err(invalid(usage()).into()),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msl_preserves_logical_bindings_for_each_entry_point() {
        let source = r#"
            @group(0) @binding(0) var<storage, read> a : array<f32>;
            @group(0) @binding(1) var<storage, read> b : array<f32>;
            @group(0) @binding(2) var<storage, read_write> output : array<f32>;

            @compute @workgroup_size(1)
            fn from_a(@builtin(global_invocation_id) id : vec3<u32>) {
                output[id.x] = a[id.x];
            }

            @compute @workgroup_size(1)
            fn from_b(@builtin(global_invocation_id) id : vec3<u32>) {
                output[id.x] = b[id.x];
            }
        "#;
        let metal = translate_msl_shader(source, Path::new("multi-entry.wgsl")).unwrap();
        assert!(metal.contains("a [[buffer(0)]]"));
        assert!(metal.contains("b [[buffer(1)]]"));
        assert!(metal.contains("output [[buffer(2)]]"));
        assert!(metal.contains("_buffer_sizes [[buffer(3)]]"));
        assert!(!metal.contains("user(fake"));
    }

    #[test]
    fn glsl_profiles_preserve_sparse_bindings_for_selected_entry() {
        let source = r#"
            struct Params { length : u32 }
            @group(0) @binding(0) var<storage, read> a : array<f32>;
            @group(0) @binding(3) var<storage, read> b : array<f32>;
            @group(0) @binding(5) var<storage, read_write> output : array<f32>;
            @group(0) @binding(7) var<uniform> params : Params;

            @compute @workgroup_size(1)
            fn from_a(@builtin(global_invocation_id) id : vec3<u32>) {
                let index = id.x % params.length;
                output[index] = a[index];
            }

            @compute @workgroup_size(1)
            fn from_b(@builtin(global_invocation_id) id : vec3<u32>) {
                let index = id.x % params.length;
                output[index] = b[index];
            }
        "#;
        let label = Path::new("sparse-multi-entry.wgsl");
        let module = parse_module(source, label).unwrap();
        let info = validate_module(&module, label, glsl::supported_capabilities()).unwrap();
        let bindings = logical_binding_slots(&module).unwrap();

        for version in [glsl::Version::Desktop(430), glsl::Version::new_gles(310)] {
            let output =
                translate_glsl_entry(&module, &info, label, "from_b", version, &bindings).unwrap();
            assert!(output.contains(&format!("#version {version}")));
            assert!(output.contains("layout(std430, binding = 3)"));
            assert!(output.contains("layout(std430, binding = 5)"));
            assert!(output.contains("layout(std140, binding = 7)"));
            assert!(!output.contains("layout(std430, binding = 0)"));
        }
    }

    #[test]
    fn gles_omits_redundant_workgroup_clear_only_for_audited_inference_shaders() {
        for stem in [
            "linearF32Tiled",
            "linearF32RowMajorTiled",
            "linearInt8Tiled",
            "qLinearInt8Tiled",
            "qConv2DInt8Tiled",
            "conv2DPointwise16Tile",
            "qGroupNormStats",
            "qLayerNormStats",
            "qSDPAInt8",
        ] {
            let label = PathBuf::from(format!("{stem}.wgsl"));
            assert!(!glsl_zero_initialize_workgroup_memory(
                &label,
                "main",
                glsl::Version::new_gles(310),
            ));
        }
        assert!(glsl_zero_initialize_workgroup_memory(
            Path::new("matMulBackward.wgsl"),
            "main",
            glsl::Version::new_gles(310),
        ));

        let source = r#"
            @group(0) @binding(0) var<storage, read_write> output : array<u32>;
            var<workgroup> values : array<u32, 64>;

            @compute @workgroup_size(64)
            fn main(@builtin(local_invocation_index) lane : u32) {
                values[lane] = lane;
                workgroupBarrier();
                output[lane] = values[lane];
            }
        "#;
        let audited_label = Path::new("qGroupNormStats.wgsl");
        let module = parse_module(source, audited_label).unwrap();
        let info = validate_module(&module, audited_label, glsl::supported_capabilities()).unwrap();
        let bindings = logical_binding_slots(&module).unwrap();
        let gles = translate_glsl_entry(
            &module,
            &info,
            audited_label,
            "main",
            glsl::Version::new_gles(310),
            &bindings,
        )
        .unwrap();
        assert!(!gles.contains("gl_LocalInvocationID == uvec3(0u)"));

        let conservative_label = Path::new("unreviewedSharedMemory.wgsl");
        let conservative = translate_glsl_entry(
            &module,
            &info,
            conservative_label,
            "main",
            glsl::Version::new_gles(310),
            &bindings,
        )
        .unwrap();
        assert!(conservative.contains("gl_LocalInvocationID == uvec3(0u)"));

        let desktop = translate_glsl_entry(
            &module,
            &info,
            audited_label,
            "main",
            glsl::Version::Desktop(430),
            &bindings,
        )
        .unwrap();
        assert!(desktop.contains("gl_LocalInvocationID == uvec3(0u)"));
    }

    #[test]
    fn multi_entry_glsl_names_are_deterministic() {
        assert_eq!(
            glsl_output_name("matMulBackward", "weight_main", 3),
            "matMulBackward_weight_main.comp"
        );
        assert_eq!(glsl_output_name("copy", "main", 1), "copy.comp");
    }

    #[test]
    fn rejects_nonzero_bind_groups_instead_of_colliding_slots() {
        let source = r#"
            @group(1) @binding(0) var<storage, read> input : array<f32>;
            @group(0) @binding(0) var<storage, read_write> output : array<f32>;
            @compute @workgroup_size(1)
            fn main(@builtin(global_invocation_id) id : vec3<u32>) {
                output[id.x] = input[id.x];
            }
        "#;
        let module = parse_module(source, Path::new("bad-group.wgsl")).unwrap();
        let error = logical_binding_slots(&module).unwrap_err();
        assert!(error.to_string().contains("only supports bind group 0"));
    }
}
