"""Explicit execution declarations for test-authored graph packages."""
from copy import deepcopy


def stateless_graph(document):
    document = deepcopy(document)
    existing = document.get("execution", {})
    if "tensor_feedback" in existing or "incremental" in existing:
        return document
    document["execution"] = {
        "entrypoints": [{"name": "evaluate", "effect": 1, "allowed_ranges": [1],
                         "inputs": list(document["inputs"]), "outputs": list(document["outputs"])}],
        "stateless": {},
    }
    return document


def feedback_graph(document, feedback, *, invariants=()):
    document = deepcopy(document)
    document["inputs"]["position"] = {"shape": [1], "dtype": "int32"}
    document["execution"] = {
        "entrypoints": [{"name": name, "effect": effect, "allowed_ranges": [1],
                         "inputs": list(document["inputs"]), "outputs": list(document["outputs"])}
                        for name, effect in (("initialize", 2), ("transition", 3))],
        "invariant_inputs": list(invariants),
        "tensor_feedback": {"position_input": "position", "bindings": [
            {"input_name": name, "output_name": output} for name, output in feedback.items()]},
    }
    return document
