"""Current runtime graph-name contract shared with the JavaScript ``Graph``.

Source IR is lossless and may preserve arbitrary source names.  Runnable
``volvox-graph/v1`` packages are narrower because JavaScript graph descriptors
use ordinary object properties for named ports and safetensors reserves
``__metadata__``.  Keep this rule at runtime/publication boundaries only.
"""

from __future__ import annotations

from typing import Any


# ``Object.getOwnPropertyNames(Object.prototype)`` in the supported JavaScript
# runtime.  Graph.validName rejects these exact strings with an own-property
# check, including the legacy accessor spellings and ``__proto__``.
OBJECT_PROTOTYPE_OWN_NAMES = frozenset({
    "constructor",
    "__defineGetter__",
    "__defineSetter__",
    "hasOwnProperty",
    "__lookupGetter__",
    "__lookupSetter__",
    "isPrototypeOf",
    "propertyIsEnumerable",
    "toString",
    "valueOf",
    "__proto__",
    "toLocaleString",
})

RUNTIME_RESERVED_NAMES = OBJECT_PROTOTYPE_OWN_NAMES | {"__metadata__"}

# ECMAScript WhiteSpace + LineTerminator code points consumed by String.trim().
# Python's default str.strip() additionally removes characters such as U+0085,
# so use the exact set instead of accidentally narrowing the cross-language ABI.
_ECMASCRIPT_TRIM = (
    "\u0009\u000a\u000b\u000c\u000d\u0020\u00a0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000\ufeff"
)


def is_runtime_graph_name(value: Any) -> bool:
    """Return whether ``value`` satisfies TypeScript ``Graph.validName``."""

    return (
        isinstance(value, str)
        and bool(value.strip(_ECMASCRIPT_TRIM))
        and value not in RUNTIME_RESERVED_NAMES
    )
