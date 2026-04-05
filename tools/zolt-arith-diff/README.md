# Zolt-Arith Differential Workflow

This is an optional, repo-level workflow for `zolt-arith` differential verification.

It is intentionally kept outside `packages/zolt-arith/` so:

- `cd packages/zolt-arith && zig build test` stays lightweight
- Rust tooling is not required for normal package builds
- CI can run differential verification in dedicated jobs only

## What It Does

- `arkworks-fixtures/`
  Generates deterministic BN254 differential fixtures for:
  - field ops (add, sub, mul, inverse for Fr and Fp)
  - accumulator ops (sumOfProducts, batchInverse, mulU64, mulU128)
  - pairing on generator-derived points
  - MSM on deterministic doubled bases (G1 Fr, G1 i128, G2 Fr)
  - Blake2b transcript state and challenge vectors (independent Rust oracle)

- `check.zig`
  Reads the generated fixtures and verifies Zolt against them.

## Commands

Generate fixtures:

```bash
zig build gen-zolt-arith-diff-fixtures
```

Run optional differential tests:

```bash
zig build test-zolt-arith-diff
```

Run both in one shot:

```bash
tools/zolt-arith-diff/run.sh
```

## CI Shape

Recommended dedicated jobs:

- one job that regenerates fixtures and fails on drift
- one job that runs `zig build test-zolt-arith-diff`
- one job that runs `zig build bench-zolt-arith-field` and publishes results
