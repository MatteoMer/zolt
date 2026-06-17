#!/usr/bin/env python3
"""Static guard for lookup-table support added by the Catallaxy campaign.

The Zig tests exercise the new table semantics. This script checks the failure
surface requested by the campaign: newly supported tables must no longer be
left as "not implemented" panics, while known unsupported table paths stay loud.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "src" / "zkvm" / "lookup_table" / "mod.zig"
TEXT = MOD.read_text()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


SUPPORTED_TABLES = {
    22: "Pow2W",
    29: "VirtualChangeDivisor",
    30: "VirtualChangeDivisorW",
    31: "MulUNoOverflow",
    32: "VirtualXORROT32",
    33: "VirtualXORROT24",
    34: "VirtualXORROT16",
    35: "VirtualXORROT63",
    36: "VirtualXORROTW16",
    37: "VirtualXORROTW12",
    38: "VirtualXORROTW8",
    39: "VirtualXORROTW7",
}

PLACEHOLDER_WORDS = ("not implemented", "not_implemented", "unimplemented", "todo", "placeholder")


def switch_arm(body: str, table: int) -> str | None:
    """Return the full text of a numeric switch arm, including multi-line blocks."""
    match = re.search(
        rf"(?m)^\s*{table}\s*=>\s*(?P<arm>.*?)(?=^\s*(?:\d+|else)\s*=>|\Z)",
        body,
        flags=re.S,
    )
    return None if match is None else match.group("arm").strip()


def has_placeholder_panic(arm: str) -> bool:
    lowered = arm.lower()
    has_panic = "@panic" in lowered or "std.debug.panic" in lowered or "unreachable" in lowered
    return has_panic and any(word in lowered for word in PLACEHOLDER_WORDS)


def strip_line_comments(text: str) -> str:
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines()).strip()


def is_zero_only_arm(arm: str) -> bool:
    expression = strip_line_comments(arm).rstrip(",;").strip()
    return re.fullmatch(r"(?:0|@as\([^,]+,\s*0\)|@intCast\(0\)|(?:F\.)?zero\(\))", expression) is not None


# The requested supported tables must not remain as placeholder panics in the dispatcher.
evaluate_match = re.search(
    r"pub fn evaluateTableMLE\(table_index: usize, r: \[\]const F\) F \{(?P<body>.*?)\n        \}\n\n        /// Materialize",
    TEXT,
    flags=re.S,
)
require(evaluate_match is not None, "could not locate evaluateTableMLE body")
evaluate_body = evaluate_match.group("body")
for table, name in SUPPORTED_TABLES.items():
    arm_body = switch_arm(evaluate_body, table)
    require(arm_body is not None, f"table {table} ({name}) is missing from evaluateTableMLE")
    require(
        not has_placeholder_panic(arm_body),
        f"table {table} ({name}) is still an evaluateTableMLE placeholder/not-implemented panic",
    )

require(
    re.search(r"else\s*=>\s*std\.debug\.panic\(\"unsupported lookup table index:", evaluate_body),
    "evaluateTableMLE must keep a loud unsupported-table panic fallback",
)
require(
    not re.search(r"else\s*=>\s*(?:F\.)?zero\s*\(", evaluate_body) and "else => 0" not in evaluate_body,
    "evaluateTableMLE must not silently return zero for unsupported tables",
)

# materializeTableEntry must keep an unsupported-table panic fallback, not a zero fallback.
match = re.search(
    r"pub fn materializeTableEntry\(table_index: usize, index: u128\) u64 \{(?P<body>.*?)\n        \}\n\n        pub const ValidSignedRemainder",
    TEXT,
    flags=re.S,
)
require(match is not None, "could not locate materializeTableEntry body")
body = match.group("body")
for table, name in SUPPORTED_TABLES.items():
    arm_body = switch_arm(body, table)
    require(
        arm_body is not None,
        f"materializeTableEntry lacks explicit support for table {table} ({name})",
    )
    require(
        not is_zero_only_arm(arm_body),
        f"materializeTableEntry table {table} ({name}) silently returns zero instead of explicit support",
    )
    require(
        not has_placeholder_panic(arm_body),
        f"materializeTableEntry table {table} ({name}) is a placeholder/not-implemented panic",
    )
require(
    re.search(r"else\s*=>\s*std\.debug\.panic\(\"unsupported lookup table index for materializeTableEntry", body),
    "materializeTableEntry must keep a loud unsupported-table panic fallback",
)
require(
    not re.search(r"else\s*=>\s*(?:F\.)?zero\s*\(", body) and "else => 0" not in body,
    "materializeTableEntry must not silently return zero for unsupported tables",
)

print("lookup table unsupported-path guard passed")
