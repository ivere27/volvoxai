"""Current volvox-graph/v1 opType vocabulary.

The graph spellings and numeric ``OperatorKind`` values are generated together
from the protobuf enum. There are no aliases, compatibility spellings, or
optimizer-only serialized operators: every registered name is runnable.
"""

from __future__ import annotations

from ..generated.volvox_enums import (
    OPERATOR_GRAPH_NAMES,
    RUNTIME_OPERATOR_NAMES,
    OperatorKind,
)


OP_STRING_TO_KIND: dict[str, OperatorKind] = {
    name: kind for kind, name in OPERATOR_GRAPH_NAMES.items()
}

if set(OP_STRING_TO_KIND) != set(RUNTIME_OPERATOR_NAMES):
    raise RuntimeError("generated operator graph vocabulary is internally inconsistent")

# A generated enum entry without a current graph spelling is a release error,
# so there is no best-effort reconciliation list.
UNRECONCILED_RUNTIME_OPS: tuple[str, ...] = ()
