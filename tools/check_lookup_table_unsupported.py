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

# The requested supported tables must not remain as placeholder panics in the dispatcher.
evaluate_match = re.search(
    r"pub fn evaluateTableMLE\(table_index: usize, r: \[\]const F\) F \{(?P<body>.*?)\n        \}\n\n        /// Materialize",
    TEXT,
    flags=re.S,
)
require(evaluate_match is not None, "could not locate evaluateTableMLE body")
evaluate_body = evaluate_match.group("body")
for table, name in SUPPORTED_TABLES.items():
    arm = re.search(rf"\b{table}\s*=>\s*(?P<body>[^,;\n]*(?:\([^\n]*\))?)", evaluate_body)
    require(arm is not None, f"table {table} ({name}) is missing from evaluateTableMLE")
    arm_body = arm.group("body").lower()
    require(
        not ("@panic" in arm_body and any(word in arm_body for word in ("not implemented", "todo", "placeholder"))),
        f"table {table} ({name}) is still an evaluateTableMLE placeholder panic",
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
    require(
        re.search(rf"\b{table}\s*=>", body),
        f"materializeTableEntry lacks explicit support for table {table} ({name})",
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
