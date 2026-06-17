#!/usr/bin/env python3
"""Verify the committed upstream Jolt lookup-table MLE fixture is reproducible.

The check regenerates the fixture into a temporary file and compares it to the
checked-in copy. It intentionally does not write to testdata by default.
"""

from __future__ import annotations

import difflib
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "testdata" / "lookup-table-mle" / "nonboolean_bn254.txt"
MANIFEST = ROOT / "tools" / "jolt-lookup-mle-fixtures" / "Cargo.toml"


def fail(message: str) -> None:
    raise SystemExit(f"lookup table fixture reproducibility check failed: {message}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="zolt-lookup-fixture-") as tmpdir:
        generated = Path(tmpdir) / "nonboolean_bn254.txt"
        cmd = [
            "cargo",
            "run",
            "--locked",
            "--release",
            "--manifest-path",
            str(MANIFEST),
            "--",
            "--out",
            str(generated),
        ]
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if proc.returncode != 0:
            sys.stdout.write(proc.stdout)
            fail(f"generator command failed: {' '.join(cmd)}")

        expected = FIXTURE.read_text().splitlines(keepends=True)
        actual = generated.read_text().splitlines(keepends=True)
        if expected != actual:
            diff = "".join(
                difflib.unified_diff(
                    expected,
                    actual,
                    fromfile=str(FIXTURE),
                    tofile="regenerated lookup-table MLE fixture",
                )
            )
            sys.stdout.write(diff)
            fail(
                "committed fixture is stale; regenerate with `cargo run --locked --release "
                "--manifest-path tools/jolt-lookup-mle-fixtures/Cargo.toml -- "
                "--out testdata/lookup-table-mle/nonboolean_bn254.txt`"
            )

    print("lookup table fixture reproducibility check passed")


if __name__ == "__main__":
    main()
