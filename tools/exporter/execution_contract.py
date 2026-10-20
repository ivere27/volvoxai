"""Execution metadata authored by importers, using the public proto vocabulary."""
from __future__ import annotations

from .generated.volvox_enums import ExecutionEffect, ExecutionRangeKind


def stateless_execution(inputs, outputs):
    """Declare one complete forward evaluation for an imported ordinary graph."""
    return {"entrypoints": [{"name": "evaluate", "effect": int(ExecutionEffect.EVALUATE),
        "allowed_ranges": [int(ExecutionRangeKind.FULL)],
        "inputs": list(inputs), "outputs": list(outputs)}], "stateless": {}}


def tensor_feedback_execution(inputs, outputs, *, feedback, position_input, invariant_inputs=()):
    """Declare the importer-verified external tensor state transition."""
    return {"entrypoints": [{"name": name, "effect": int(effect),
        "allowed_ranges": [int(ExecutionRangeKind.FULL)],
        "inputs": list(inputs), "outputs": list(outputs)} for name, effect in [
            ("initialize", ExecutionEffect.INITIALIZE), ("transition", ExecutionEffect.TRANSITION)]],
        "invariant_inputs": list(invariant_inputs),
        "tensor_feedback": {"position_input": position_input,
            "bindings": [{"input_name": name, "output_name": output} for name, output in feedback.items()]}}


def incremental_decoder_execution(inputs, outputs, *, token_input, token_output,
                                  keep_input, position_input, logits_output,
                                  id_domain_size, invariant_inputs=(),
                                  sequence_input=None, chunk_mask_input=None,
                                  chunk_size=None, zero_append_inputs=(),
                                  multimodal_position_input=None, incremental_controls=False):
    """Declare batch-one categorical feedback over an explicit token row state."""
    invariants = list(dict.fromkeys(["active_lengths", *invariant_inputs]))
    execution = {"entrypoints": [{"name": name, "effect": int(effect),
        "allowed_ranges": [int(value) for value in ranges],
        "inputs": list(inputs), "outputs": list(outputs)} for name, effect, ranges in (
            ("evaluate", ExecutionEffect.EVALUATE, (ExecutionRangeKind.FULL, ExecutionRangeKind.PREFIX_ROWS)),
            ("initialize", ExecutionEffect.INITIALIZE, (ExecutionRangeKind.FULL,)),
            ("transition", ExecutionEffect.TRANSITION, (ExecutionRangeKind.SLOT_ROWS, ExecutionRangeKind.DEPENDENCIES)))],
        "invariant_inputs": invariants,
        "incremental": {"active_lengths_input": "active_lengths", "position_input": position_input,
            "sequence_axis": 1, "update_inputs": [name for name in inputs if name not in invariants]},
        "generation": {"initialize_entrypoint": "initialize", "transition_entrypoint": "transition",
            "token_input": token_input, "token_output": token_output, "keep_input": keep_input,
            "logits_output": logits_output, "logits_axis": 2, "id_domain_size": id_domain_size,
            "output_name": "generated_ids", "sequence_axis": 1}}
    if (incremental_controls or sequence_input is not None or chunk_mask_input is not None or chunk_size is not None or
            zero_append_inputs or multimodal_position_input is not None):
        controls = {"position_input": position_input}
        if sequence_input is not None:
            controls["sequence_input"] = sequence_input
        if chunk_mask_input is not None:
            if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
                raise ValueError("incremental chunk mask requires a positive chunk_size")
            controls.update(chunk_mask_input=chunk_mask_input, chunk_size=chunk_size)
        elif chunk_size is not None:
            raise ValueError("incremental chunk_size requires a chunk mask")
        if zero_append_inputs:
            controls["zero_append_inputs"] = list(zero_append_inputs)
        if multimodal_position_input is not None:
            controls["multimodal_position_input"] = multimodal_position_input
        execution["generation"]["incremental_controls"] = controls
    return execution


def full_prefix_decoder_execution(inputs,outputs,*,token_input,token_output,keep_input,
        position_input,logits_output,id_domain_size,sequence_input=None,
        chunk_mask_input=None,chunk_size=None,zero_append_inputs=(),
        multimodal_position_input=None,invariant_inputs=()):
    """Declare exact whole-prefix feedback, including fixed image conditioning.

    The engine owns categorical feedback and extends each named position or
    routing control according to the public FullPrefixGenerationContract.
    Recurrent and attention nodes re-evaluate the prefix from initial state.
    """
    execution=stateless_execution(inputs,outputs)
    execution["invariant_inputs"]=list(invariant_inputs)
    controls={"position_input":position_input}
    if sequence_input is not None:controls["sequence_input"]=sequence_input
    if chunk_mask_input is not None:
        controls.update(chunk_mask_input=chunk_mask_input,chunk_size=chunk_size)
    if zero_append_inputs:controls["zero_append_inputs"]=list(zero_append_inputs)
    if multimodal_position_input is not None:controls["multimodal_position_input"]=multimodal_position_input
    execution["generation"]={"initialize_entrypoint":"evaluate","transition_entrypoint":"evaluate",
        "token_input":token_input,"token_output":token_output,"keep_input":keep_input,
        "logits_output":logits_output,"logits_axis":2,"id_domain_size":id_domain_size,
        "output_name":"generated_ids","sequence_axis":1,"full_prefix":controls}
    return execution
