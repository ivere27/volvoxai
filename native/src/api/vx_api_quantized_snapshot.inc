/* Full-only immutable export. Every binding reads one coherent Trainer
 * snapshot; edits apply to private template bytes, never to model/master state. */
#include "training_kernels.h"
#include "vx_training_lifecycle.h"

typedef struct {
    VolvoxaiV1SafetensorsContent* shards;
    size_t count;
    VxStatus status;
} VxQuantizedSnapshotSource;

static void vx_quantized_source_free(VxQuantizedSnapshotSource* source) {
    for (size_t i = 0; i < source->count; ++i)
        volvoxai_v1_safetensors_content_free(&source->shards[i]);
    free(source->shards);
}

static int vx_quantized_source_allocate(VxQuantizedSnapshotSource* source, size_t count) {
    if (!count || count > SIZE_MAX / sizeof(*source->shards)) return 0;
    source->shards = calloc(count, sizeof(*source->shards));
    if (!source->shards) return 0;
    source->count = count;
    for (size_t i = 0; i < count; ++i) volvoxai_v1_safetensors_content_init(&source->shards[i]);
    return 1;
}

static int vx_quantized_source_read(VxQuantizedSnapshotSource* source, size_t index,
        const unsigned char* bytes, size_t size) {
    VolvoxaiV1ReadSafetensorsRequest read;
    volvoxai_v1_read_safetensors_request_init(&read);
    read.field_source.data = (uint8_t*)bytes; read.field_source.len = size;
    VolvoxaiV1SafetensorsContent* shard = &source->shards[index];
    if (vx_api_read_safetensors(&read, shard, NULL) || !shard->field_report)
        source->status = VX_STATUS_OUT_OF_MEMORY;
    else source->status = shard->field_report->field_status;
    return source->status == VX_STATUS_OK;
}

static int vx_quantized_source_sink(void* user, size_t index, size_t count,
        const unsigned char* bytes, size_t size) {
    VxQuantizedSnapshotSource* source = user;
    if (!index && !vx_quantized_source_allocate(source, count)) {
        source->status = VX_STATUS_OUT_OF_MEMORY; return 0;
    }
    return vx_quantized_source_read(source, index, bytes, size);
}

static int vx_quantized_name_equal(const SynurangLiteBytes* left, const SynurangLiteBytes* right) {
    return left->len == right->len && (!left->len || !memcmp(left->data, right->data, left->len));
}

static VolvoxaiV1Tensor* vx_quantized_find(VxQuantizedSnapshotSource* source, const SynurangLiteBytes* name) {
    VolvoxaiV1Tensor* result = NULL;
    for (size_t f = 0; f < source->count; ++f)
        for (size_t i = 0; i < source->shards[f].field_tensors.len; ++i) {
            VolvoxaiV1Tensor* tensor = &source->shards[f].field_tensors.data[i];
            if (vx_quantized_name_equal(&tensor->field_name, name)) {
                if (result) return NULL;
                result = tensor;
            }
        }
    return result;
}

static int vx_quantized_binding_names(VxGraphWriter* writer,
        const VolvoxaiV1ExportQuantizedTrainerWeightsRequest* request, size_t index) {
    const VolvoxaiV1TrainerQuantizedWeightBinding* current = &request->field_bindings.data[index];
    const SynurangLiteBytes* names[] = {&current->field_weight_name, &current->field_scale_name, &current->field_zero_point_name};
    if (!vx_graph_string(writer, &current->field_master_name, 0)) return 0;
    for (size_t i = 0; i < 3; ++i) {
        if (!vx_graph_string(writer, names[i], 0)) return 0;
        for (size_t j = 0; j < i; ++j) if (vx_quantized_name_equal(names[i], names[j])) return 0;
    }
    for (size_t j = 0; j < index; ++j) {
        const VolvoxaiV1TrainerQuantizedWeightBinding* prior = &request->field_bindings.data[j];
        if (vx_quantized_name_equal(&current->field_master_name, &prior->field_master_name)) return 0;
        const SynurangLiteBytes* used[] = {&prior->field_weight_name, &prior->field_scale_name, &prior->field_zero_point_name};
        for (size_t i = 0; i < 3; ++i) for (size_t k = 0; k < 3; ++k)
            if (vx_quantized_name_equal(names[i], used[k])) return 0;
    }
    return 1;
}

static VxStatus vx_quantized_snapshot_binding(VxQuantizedSnapshotSource* masters,
        VxQuantizedSnapshotSource* targets, const VolvoxaiV1TrainerQuantizedWeightBinding* binding,
        uint64_t* saturation) {
    VolvoxaiV1Tensor* master = vx_quantized_find(masters, &binding->field_master_name);
    VolvoxaiV1Tensor* weight = vx_quantized_find(targets, &binding->field_weight_name);
    VolvoxaiV1Tensor* scale = vx_quantized_find(targets, &binding->field_scale_name);
    VolvoxaiV1Tensor* zero = vx_quantized_find(targets, &binding->field_zero_point_name);
    if (!master || !weight || !scale || !zero || master->field_shape.len != 2u || weight->field_shape.len != 2u ||
        master->field_dtype != VOLVOXAI_V1_DATA_TYPE_F32 || weight->field_dtype != VOLVOXAI_V1_DATA_TYPE_I8 ||
        scale->field_dtype != VOLVOXAI_V1_DATA_TYPE_F32 || zero->field_dtype != VOLVOXAI_V1_DATA_TYPE_I8 ||
        scale->field_shape.len != 1u || zero->field_shape.len != 1u) return VX_STATUS_INVALID_ARGUMENT;
    int64_t rows = weight->field_shape.data[0], columns = weight->field_shape.data[1];
    if (rows <= 0 || columns <= 0 || rows > INT_MAX || columns > INT_MAX ||
        (uint64_t)rows * columns > UINT32_MAX || (uint64_t)rows * columns > SIZE_MAX / sizeof(float) ||
        master->field_shape.data[binding->field_transpose_master ? 1 : 0] != rows ||
        master->field_shape.data[binding->field_transpose_master ? 0 : 1] != columns ||
        scale->field_shape.data[0] != rows || zero->field_shape.data[0] != rows) return VX_STATUS_INVALID_ARGUMENT;
    size_t count = (size_t)rows * (size_t)columns;
    if (master->field_inline.len != count * sizeof(float) || weight->field_inline.len != count ||
        scale->field_inline.len != (size_t)rows * sizeof(float) || zero->field_inline.len != (size_t)rows)
        return VX_STATUS_INVALID_ARGUMENT;
    for (size_t i = 0; i < (size_t)rows; ++i) if (zero->field_inline.data[i]) return VX_STATUS_INVALID_ARGUMENT;
    float* input = malloc(count * sizeof(float));
    float* scales = malloc((size_t)rows * sizeof(float));
    if (!input || !scales) { free(input); free(scales); return VX_STATUS_OUT_OF_MEMORY; }
    memcpy(input, master->field_inline.data, count * sizeof(float));
    memcpy(scales, scale->field_inline.data, (size_t)rows * sizeof(float));
    uint64_t clipped = 0;
    int ok = volvoxai_training_quantize_weight_f32_to_i8(input, (int8_t*)weight->field_inline.data,
        scales, (uint32_t)rows, (uint32_t)columns, !!binding->field_transpose_master,
        binding->field_preserve_scales ? VOLVOXAI_TRAINING_WEIGHT_SCALES_PRESERVE :
            VOLVOXAI_TRAINING_WEIGHT_SCALES_RECOMPUTE, &clipped);
    if (ok && clipped <= UINT64_MAX - *saturation) {
        memcpy(scale->field_inline.data, scales, (size_t)rows * sizeof(float));
        *saturation += clipped;
    } else ok = 0;
    free(input); free(scales);
    return ok ? VX_STATUS_OK : VX_STATUS_INVALID_ARGUMENT;
}

int vx_api_export_quantized_trainer_weights(const VolvoxaiV1ExportQuantizedTrainerWeightsRequest* request,
        VolvoxaiV1QuantizedTrainerWeights* response, void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* a = response->_allocator;
    VxApiScratch scratch = VX_API_SCRATCH_INIT;
    VxGraphWriter writer = {&scratch, VX_STATUS_OK};
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxQuantizedSnapshotSource masters = {0}, targets = {0};
    VxReport report = VX_REPORT_INIT;
    VxStatus status = VX_STATUS_INVALID_ARGUMENT;
    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_TRAINER, request->field_trainer_id, &lease)) {
        status = VX_STATUS_HANDLE_DISPOSED; goto done;
    }
    if (!request->field_template_shards.len || !request->field_bindings.len) goto done;
    for (size_t i = 0; i < request->field_bindings.len; ++i)
        if (!vx_quantized_binding_names(&writer, request, i)) {
            if (writer.status != VX_STATUS_OK) status = writer.status;
            goto done;
        }
    if (!vx_quantized_source_allocate(&targets, request->field_template_shards.len)) {
        status = VX_STATUS_OUT_OF_MEMORY; goto done;
    }
    for (size_t i = 0; i < targets.count; ++i) {
        const SynurangLiteBytes* bytes = &request->field_template_shards.data[i];
        if (!vx_quantized_source_read(&targets, i, bytes->data, bytes->len)) { status = targets.status; goto done; }
    }
    /* Duplicate names across shards are ambiguous even when not edited. */
    for (size_t f = 0; f < targets.count; ++f)
        for (size_t i = 0; i < targets.shards[f].field_tensors.len; ++i)
            if (!vx_quantized_find(&targets, &targets.shards[f].field_tensors.data[i].field_name)) goto done;
    status = vx_trainer_export_weight_bytes(lease.pointer, vx_quantized_source_sink, &masters, &report);
    if (status != VX_STATUS_OK) { if (masters.status != VX_STATUS_OK) status = masters.status; goto done; }
    for (size_t i = 0; i < request->field_bindings.len; ++i) {
        status = vx_quantized_snapshot_binding(&masters, &targets, &request->field_bindings.data[i], &response->field_saturation_count);
        if (status != VX_STATUS_OK) goto done;
    }
    for (size_t f = 0; f < targets.count; ++f) {
        VolvoxaiV1WriteSafetensorsRequest write;
        VolvoxaiV1SafetensorsArtifact artifact;
        volvoxai_v1_write_safetensors_request_init(&write);
        volvoxai_v1_safetensors_artifact_init_with_allocator(&artifact, a);
        write.field_source = request->field_template_shards.data[f];
        write.field_edits.len = targets.shards[f].field_tensors.len;
        write.field_edits.data = calloc(write.field_edits.len ? write.field_edits.len : 1u, sizeof(*write.field_edits.data));
        if (!write.field_edits.data) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
        for (size_t i = 0; i < write.field_edits.len; ++i) {
            volvoxai_v1_safetensors_edit_init(&write.field_edits.data[i]);
            write.field_edits.data[i].which_operation = 1;
            write.field_edits.data[i].field_set_tensor = &targets.shards[f].field_tensors.data[i];
        }
        int result = vx_api_write_safetensors(&write, &artifact, NULL);
        free(write.field_edits.data); /* every tensor above is borrowed */
        status = result || !artifact.field_report ? VX_STATUS_OUT_OF_MEMORY : artifact.field_report->field_status;
        if (status == VX_STATUS_OK) {
            SynurangLiteBytes* shard = volvoxai_v1_quantized_trainer_weights_add_shards(response);
            if (!shard) status = VX_STATUS_OUT_OF_MEMORY;
            else { *shard = artifact.field_data; memset(&artifact.field_data, 0, sizeof(artifact.field_data)); }
        }
        volvoxai_v1_safetensors_artifact_free(&artifact);
        if (status != VX_STATUS_OK) goto done;
    }
done:
    vx_quantized_source_free(&masters); vx_quantized_source_free(&targets);
    vx_api_scratch_release(&scratch);
    if (status != VX_STATUS_OK) {
        volvoxai_v1_quantized_trainer_weights_free(response);
        volvoxai_v1_quantized_trainer_weights_init_with_allocator(response, a);
    }
    report.status = status; report.stage = VX_STAGE_TRAINER_EXPORT;
    report.code = status == VX_STATUS_OK ? VX_CODE_NONE : status == VX_STATUS_BUSY ? VX_CODE_BUSY :
        status == VX_STATUS_HANDLE_DISPOSED ? VX_CODE_HANDLE_DISPOSED :
        status == VX_STATUS_OUT_OF_MEMORY ? VX_CODE_OUT_OF_MEMORY : VX_CODE_INVALID_ARGUMENT;
    snprintf(report.message, sizeof(report.message), "%s",
        status == VX_STATUS_OK ? "quantized snapshot exported; Trainer and template remain unchanged" :
        "quantized export requires an idle Trainer, unique compatible bindings and symmetric template weights");
    int result = vx_api_report_attach(a, &response->field_report, &report) ? 0 : -1;
    if (!result) vx_api_report_set_public_lineage(response->field_report, lease.lineage.runtime_id,
        lease.lineage.model_id, lease.lineage.compiled_model_id, lease.lineage.context_id);
    vx_api_handle_lease_release(&lease);
    return result;
}
