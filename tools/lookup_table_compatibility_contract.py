#!/usr/bin/env python3
"""Acceptance contract for maintained lookup-table compatibility checks.

This script is intentionally black-box where possible: it verifies the public
commands and repository artifacts promised by the campaign rather than a single
implementation shape.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED = [22, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
FIXTURE = ROOT / "testdata" / "lookup-table-mle" / "nonboolean_bn254.txt"


def fail(message: str) -> None:
    raise SystemExit(f"lookup-table compatibility contract failed: {message}")


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        fail(f"command failed: {' '.join(cmd)}")
    return proc


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def main() -> None:
    mod = (ROOT / "src" / "zkvm" / "lookup_table" / "mod.zig").read_text()
    guard = (ROOT / "tools" / "check_lookup_table_unsupported.py").read_text()
    build_zig = (ROOT / "build.zig").read_text()
    cargo_toml_path = ROOT / "tools" / "jolt-lookup-mle-fixtures" / "Cargo.toml"
    cargo_toml = cargo_toml_path.read_text()
    readme_path = ROOT / "tools" / "jolt-lookup-mle-fixtures" / "README.md"
    readme = readme_path.read_text()

    # The guard itself is part of the maintained interface and must pass.
    run(["python3", "tools/check_lookup_table_unsupported.py"])

    # It must no longer encode 29/30 as intentionally unsupported tables.
    require(
        "must still panic loudly until implemented" not in guard,
        "unsupported-table guard still treats some campaign tables as intentionally unsupported",
    )
    for table in SUPPORTED:
        require(re.search(rf"(?<!\d){table}(?!\d)", guard), f"guard does not mention supported table {table}")

    # Supported evaluateTableMLE arms must not be placeholder/not-implemented panics.
    for table in SUPPORTED:
        arm = re.search(rf"(?<!\d){table}\s*=>\s*(?P<body>[^,;\n]*(?:\([^\n]*\))?)", mod)
        require(arm is not None, f"evaluateTableMLE has no visible arm for supported table {table}")
        body = arm.group("body").lower()
        require(
            not ("@panic" in body and ("not implemented" in body or "todo" in body or "placeholder" in body)),
            f"evaluateTableMLE table {table} still looks like a placeholder panic",
        )

    # materializeTableEntry must explicitly dispatch every supported table and keep a loud fallback.
    materialize = re.search(
        r"pub fn materializeTableEntry\(table_index: usize, index: u128\) u64 \{(?P<body>.*?)\n\s*\}",
        mod,
        flags=re.S,
    )
    require(materialize is not None, "could not locate materializeTableEntry body")
    materialize_body = materialize.group("body")
    for table in SUPPORTED:
        require(
            re.search(rf"(?<!\d){table}(?!\d)", materialize_body),
            f"materializeTableEntry lacks explicit support for table {table}",
        )
    require(
        re.search(r"else\s*=>\s*(?:std\.)?debug\.panic\(", materialize_body) or re.search(r"@panic\(", materialize_body),
        "materializeTableEntry lacks a loud unsupported-table fallback",
    )
    require(
        not re.search(r"else\s*=>\s*(?:0|(?:F\.)?zero\s*\()", materialize_body),
        "materializeTableEntry fallback may silently return zero",
    )

    # The supported Zig build step is the public repository check.
    require('b.step("check-lookup-tables"' in build_zig, "build.zig does not define zig build check-lookup-tables")
    before_hash = sha256(FIXTURE)
    before_mtime = FIXTURE.stat().st_mtime_ns
    run(["zig", "build", "check-lookup-tables"])
    require(sha256(FIXTURE) == before_hash, "check-lookup-tables modified the committed fixture contents")
    require(FIXTURE.stat().st_mtime_ns == before_mtime, "check-lookup-tables touched the committed fixture file")

    # Ordinary Zig tests must not depend on compiling/running the Rust fixture generator.
    test_step_match = re.search(r'const test_step = b\.step\("test",.*?\n\s*// Example', build_zig, flags=re.S)
    require(test_step_match is not None, "could not locate build.zig test step")
    require(
        "jolt-lookup-mle-fixtures" not in test_step_match.group(0) and "cargo" not in test_step_match.group(0),
        "zig build test appears to run the Rust/Jolt fixture generator",
    )

    # Generator reproducibility: applications with external dependencies should commit a lockfile;
    # git dependencies should not float on branches without a rev/tag pin.
    require(
        (cargo_toml_path.parent / "Cargo.lock").exists(),
        "tools/jolt-lookup-mle-fixtures/Cargo.lock is missing",
    )
    require(
        not re.search(r"git\s*=.*\n(?:(?!\n\[).)*branch\s*=", cargo_toml, flags=re.S) or re.search(r"rev\s*=|tag\s*=", cargo_toml),
        "Cargo.toml has a git branch dependency without a rev/tag pin",
    )

    # Docs near the generator must tell maintainers how to check and regenerate fixtures.
    require("check-lookup-tables" in readme, "fixture generator README does not mention zig build check-lookup-tables")
    require("cargo run --release" in readme and "--out" in readme, "fixture generator README lacks regeneration command")
    require("nonboolean_bn254.txt" in readme, "fixture generator README does not name the committed fixture")

    print("lookup-table compatibility contract passed")


if __name__ == "__main__":
    main()
