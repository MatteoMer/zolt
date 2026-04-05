# Zolt-Arith Test And Benchmark Matrix

**Date:** 2026-04-05
**Scope:** `packages/zolt-arith/`
**Purpose:** Turn the package's already-large inline-test baseline into a fixture-backed, differential-tested, benchmarked verification target

## Executive Summary

`zolt-arith` is not starting from zero.

Today it already has broad inline coverage across field arithmetic, pairings, transcripts, MSM, Dory, polynomial helpers, and GPU code. The package currently contains 203 `test` blocks under `packages/zolt-arith/src/`.

The real gaps are elsewhere:

- no broad checked-in external vector corpus yet
- no differential harness against `arkworks` or `gnark-crypto`
- no package-local microbenchmark entrypoints in `packages/zolt-arith/build.zig`
- no CI story for fixture freshness or benchmark regression tracking

So the job is not "add tests" in the abstract. The job is:

1. preserve the existing inline tests
2. add stable fixtures where exact bytes matter
3. add differential checks where BN254 behavior needs an oracle
4. standardize a benchmark workflow around the hot kernels that must not regress

## Why This Exists

The formal verification plan depends on a stable behavioral baseline and a stable performance baseline.

That means `zolt-arith` needs two kinds of evidence before deeper proof work:

- correctness evidence stronger than local unit/property tests alone
- performance evidence strong enough to reject proof-driven rewrites that slow down the field core

This document is the concrete matrix for that work.

## Current Baseline

### Build and workflow reality

Inside `packages/zolt-arith/`, [build.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/build.zig) currently exposes only:

- `zig build test`

There is currently:

- an initial `src/testdata/` for field and transcript vectors
- no package-local
- `bench/`
- `tools/`
- benchmark step
- fixture generation step

### Existing coverage that should be kept

The package already has substantial inline tests. Highest-density areas include:

| Area | Representative files | Current signal |
|---|---|---|
| Field core | [field/mod.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/field/mod.zig), [field/accumulators.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/field/accumulators.zig) | arithmetic identities, inverse/power, sum-of-products, reduction paths, signed accumulation behavior |
| Pairings and extensions | [field/extensions.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/field/extensions.zig), [field/pairing.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/field/pairing.zig), [field/g2.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/field/g2.zig) | tower arithmetic, Miller-loop helpers, bilinearity, subgroup-related behavior |
| MSM | [msm/mod.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/msm/mod.zig), [msm/glv.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/msm/glv.zig) | scalar multiplication, windowing, sequential/parallel behavior, GLV decomposition |
| Transcripts | [transcripts/blake2b.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/transcripts/blake2b.zig), [transcripts/mod.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/transcripts/mod.zig) | determinism, challenge derivation, Jolt compatibility checks |
| Polynomial and subprotocol code | [poly/mod.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/poly/mod.zig), [poly/split_eq.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/poly/split_eq.zig), [subprotocols/mod.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/subprotocols/mod.zig) | eq-polynomial identities, binding behavior, sumcheck round logic |
| Commitment and serialization-adjacent code | [poly/commitment/dory.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/poly/commitment/dory.zig), [poly/commitment/point_compression.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/poly/commitment/point_compression.zig) | commitment consistency, serialization roundtrips, compressed-point behavior |
| GPU | [gpu/mod.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/gpu/mod.zig), [gpu/field_ops.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/gpu/field_ops.zig), [gpu/poly_ops.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/gpu/poly_ops.zig), [gpu/msm_ops.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/gpu/msm_ops.zig) | CPU-vs-GPU correctness tests, with non-Metal targets compiling to stubs |

### Existing benchmark machinery outside the package

The repo already contains useful benchmark entrypoints that should be reused instead of replaced:

- [bench/msm/main.zig](/Users/matteo/projects/zolt-code-review/bench/msm/main.zig)
- [bench/msm/bench_arkworks.rs](/Users/matteo/projects/zolt-code-review/bench/msm/bench_arkworks.rs)
- [bench/msm/compare.sh](/Users/matteo/projects/zolt-code-review/bench/msm/compare.sh)
- [bench/threadpool_vs_rayon/main.zig](/Users/matteo/projects/zolt-code-review/bench/threadpool_vs_rayon/main.zig)
- [bench/run-bench.sh](/Users/matteo/projects/zolt-code-review/bench/run-bench.sh)

At the repo root, [build.zig](/Users/matteo/projects/zolt-code-review/build.zig) already exposes `bench-msm`, `bench-tp`, and `bench-scaling`.

The package-local benchmark plan should standardize and narrow this existing machinery around `zolt-arith`, not duplicate it blindly.

## Principles

1. Keep the current inline tests and treat them as the first layer, not as throwaway coverage.
2. Add checked-in fixtures only where exact outputs matter across time and across implementations.
3. Use differential tests, not standards suites, for BN254-specific arithmetic and pairing behavior.
4. Benchmark the exact kernels that make the package fast, especially the field layer.
5. Keep GPU testing configuration-aware.
   Metal execution is only real on macOS + Apple Silicon; on other targets the package intentionally compiles GPU stubs.
6. Do not pre-create a giant empty fixture tree.
   Add directories and generator scripts only when they are tied to live tests.

## External Sources Matrix

Use external sources selectively:

| Source | Use It For | Do Not Use It For |
|---|---|---|
| RFC 7693 BLAKE2 | exact transcript hashing fixtures for [transcripts/blake2b.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/transcripts/blake2b.zig) | BN254 arithmetic or pairing |
| Ethereum BN254 behavior | G1/G2/pairing compatibility and compressed-point behavior where Ethereum semantics are the target | complete field-kernel coverage |
| `arkworks` | primary differential oracle for field, extension field, G1/G2, MSM, pairings, and point compression | GPU-specific behavior |
| `gnark-crypto` | second independent oracle for field, extension field, and pairing behavior | GPU-specific behavior |
| Upstream Jolt fixtures or transcript behavior | transcript compatibility and protocol-level fixture capture | low-level limb arithmetic |
| Wycheproof | only if the package later adds standardized crypto APIs that actually map to it | BN254 field, pairing, MSM, Dory |
| NIST CAVP / ACVP | only if NIST-approved primitives are added later | BN254 field, pairing, MSM, Dory |

## Coverage Gap Matrix

The table below separates the current baseline from the missing work that matters for verification and performance preservation.

### P0: must land first

| Module | Current baseline | Missing addition | Benchmark target | Priority |
|---|---|---|---|---|
| [field/mod.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/field/mod.zig) | inline algebraic tests, inverse/power, byte roundtrips, sum-of-products equivalence | checked-in BN254 scalar/base-field vectors plus `arkworks` differential corpus for exact arithmetic cases and Montgomery conversions | add/sub/mul/square/inverse, `toMontgomery`, `fromMontgomery` | P0 |
| [field/accumulators.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/field/accumulators.zig) | strong inline coverage for reductions, batch inversion, signed accumulators, stress cases | checked-in accumulator fixtures plus offline differential outputs for reducer paths | `reduceMulU64`, `reduceMulU128`, `sumOfProducts`, batch inversion | P0 |
| [transcripts/blake2b.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/transcripts/blake2b.zig) | determinism and Jolt-compat inline tests | RFC 7693 vectors plus checked-in transcript byte-for-byte fixtures captured from the compatibility target | append scalar/bytes, derive challenge | P0 |
| [msm/mod.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/msm/mod.zig) | inline sequential/parallel and scalar-mul tests | checked-in small/medium vectors plus `arkworks` differential outputs for G1 Fr and G1 i128 cases | reuse [bench/msm/main.zig](/Users/matteo/projects/zolt-code-review/bench/msm/main.zig) and root `bench-msm` step | P0 |
| [field/pairing.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/field/pairing.zig) | bilinearity, identity, prepared/unprepared equivalence | Ethereum-compatible fixtures plus `arkworks` and `gnark-crypto` differential outputs for G1/G2/pairing tuples | Miller loop, final exponentiation, full pairing | P0 |

### P1: next wave after the baseline is stable

| Module | Current baseline | Missing addition | Benchmark target | Priority |
|---|---|---|---|---|
| [field/extensions.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/field/extensions.zig) | inline `Fp2/Fp6/Fp12` arithmetic and serialization tests | checked-in extension-field fixtures plus differential outputs from `arkworks` and `gnark-crypto` | `Fp2`, `Fp6`, `Fp12` mul/square/inverse | P1 |
| [field/g2.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/field/g2.zig) | inline G2 ops and scalar-mul consistency tests | generator/scalar-mul fixtures plus subgroup-focused differential cases | scalar multiplication and precompute helpers | P1 |
| [poly/commitment/point_compression.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/poly/commitment/point_compression.zig) | inline roundtrip and identity tests | checked-in valid and invalid compressed encodings, ideally Ethereum-compatible where relevant | compression throughput and batch decompression | P1 |
| [msm/glv.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/msm/glv.zig) | inline decomposition and endomorphism tests | checked-in decomposition vectors from an offline oracle | decomposition overhead and end-to-end MSM win | P1 |
| [poly/commitment/g2_msm.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/poly/commitment/g2_msm.zig) | indirect coverage via Dory paths | explicit small G2 MSM fixtures plus `arkworks` differential outputs | G2 MSM by Dory-relevant sizes | P1 |
| [poly/commitment/dory.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/poly/commitment/dory.zig) | extensive inline behavior and serialization tests | stable checked-in small fixtures for commit/open/verify traces, especially where Jolt compatibility matters | commit/open by shape and pool mode | P1 |
| [gpu/field_ops.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/gpu/field_ops.zig), [gpu/poly_ops.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/gpu/poly_ops.zig) | inline CPU-vs-GPU correctness on Metal, stub behavior elsewhere | explicit crossover fixtures and benchmark reports by size; CI should treat non-Metal as compile-only for GPU paths | CPU vs GPU crossover points | P1 |

### P2: useful, but lower leverage

| Module | Current baseline | Missing addition | Benchmark target | Priority |
|---|---|---|---|---|
| [poly/mod.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/poly/mod.zig) | strong inline eq/binding coverage | checked-in tiny exact-value fixtures for deterministic regression coverage | `bindLow`, table builders, dense evaluate | P2 |
| [poly/interpolation.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/poly/interpolation.zig) | a few inline interpolation tests | small exact interpolation fixtures and slow-scalar differential cases | interpolation by degree and size | P2 |
| [poly/product_tree.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/poly/product_tree.zig) | indirect coverage | tiny exact fixtures | product tree scaling | P2 |
| [poly/split_eq.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/poly/split_eq.zig), [poly/lt_poly.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/poly/lt_poly.zig), [poly/multiquadratic.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/poly/multiquadratic.zig), [expanding_table.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/expanding_table.zig) | inline property tests already carry most of the value | optional exact fixtures if regressions appear repeatedly | helper microbenches only if they hit real profiles | P2 |
| [bits.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/bits.zig) | boundary and roundtrip inline tests | optional edge-case fixture file only if serialization consumers need byte stability | low priority | P2 |
| [transcripts/mod.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/transcripts/mod.zig), [subprotocols/mod.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/src/subprotocols/mod.zig) | inline protocol-state checks | small deterministic fixture traces where protocol composition is fragile | protocol microbench only after field/MSM/pairing | P2 |

## Test Types To Add

For each P0 and P1 target, add only the test classes that buy unique signal:

### 1. Golden vector tests

Use checked-in fixtures for:

- exact field input/output tuples
- exact transcript bytes and challenges
- exact compressed-point encodings
- exact small pairing tuples
- exact small MSM cases

### 2. Differential tests

Run the same fixture inputs against:

- Zolt
- `arkworks`
- `gnark-crypto` where applicable

Then compare exact outputs after normalization.

This is the main oracle for BN254 arithmetic behavior.

### 3. Property tests

Keep and expand the existing property-style tests where they already fit well:

- field algebraic laws
- serialization roundtrips
- sequential vs parallel equivalence
- CPU vs GPU equivalence
- prepared vs unprepared pairing equivalence

### 4. Negative tests

Add or strengthen negative cases for:

- invalid compressed points
- malformed transcript labels or oversized packing inputs
- subgroup failures and invalid pairing inputs
- GPU unavailable-path behavior on non-Metal targets
- malformed or stale fixture decoding

## Benchmark Matrix

The benchmark plan should start from the hottest kernels and the already-existing repo entrypoints.

### Immediate benchmark entrypoints to keep using

- `zig build bench-msm` at the repo root
- [bench/msm/compare.sh](/Users/matteo/projects/zolt-code-review/bench/msm/compare.sh) for Zig-vs-arkworks MSM comparison
- `zig build bench-tp`
- `zig build bench-scaling`

These are useful today even before package-local benchmark files exist.

### Package-local benchmarks to add later

Add package-local benches only once they have a clear owner and CI story:

| Planned file | What it should measure | Why it matters |
|---|---|---|
| `packages/zolt-arith/bench/field_micro.zig` | field add/sub/mul/square/inverse, Montgomery conversions, reducers | hottest correctness-preserving perf gate |
| `packages/zolt-arith/bench/pairing_micro.zig` | Miller loop, final exponentiation, full pairing, prepared vs unprepared | protects expensive cryptographic core |
| `packages/zolt-arith/bench/poly_micro.zig` | `bindLow`, table builders, interpolation, product-tree helpers | useful only after field baseline is stable |
| `packages/zolt-arith/bench/transcript_micro.zig` | append scalar/bytes, derive challenge | low-cost guard for transcript-heavy call sites |
| `packages/zolt-arith/bench/gpu_compare.zig` | CPU vs GPU crossover by size | turns GPU claims into measured thresholds |

### Metrics to capture

Record at least:

- ns/op
- ops/sec
- input size
- backend/configuration
- relative delta vs baseline

Do not gate on exact absolute numbers. Gate on tolerance bands and regression percentages.

## Proposed Minimal Repo Layout

Do not create the full idealized tree up front. Start with the minimum directories that correspond to live tests and scripts. For fixtures that are embedded directly into Zig tests, the buildable location is under `src/testdata/`:

```text
packages/zolt-arith/
├── src/
│   └── testdata/
│       ├── field/
│       ├── transcripts/
│       ├── msm/
│       └── pairing/
├── bench/
│   ├── field_micro.zig
│   └── pairing_micro.zig
└── tools/
    ├── gen_arkworks_vectors.rs
    └── gen_gnark_vectors.go
```

Only after those are real should the package grow optional directories for:

- extension-field vectors
- point-compression vectors
- Dory fixtures
- GPU crossover tools

The scripts in `tools/` are offline generators. They are not trusted proof artifacts. Their job is only to emit stable checked-in fixtures.

## Build And CI Rollout

### Phase 1: no package build changes yet

Land first:

- `src/testdata/field/`
- `src/testdata/transcripts/`
- `src/testdata/msm/`
- `src/testdata/pairing/`
- test loaders and vector-driven tests inside the existing source files

Keep running:

- `cd packages/zolt-arith && zig build test`
- repo-root `zig build bench-msm`

### Phase 2: add package-local benchmark steps

Extend [packages/zolt-arith/build.zig](/Users/matteo/projects/zolt-code-review/packages/zolt-arith/build.zig) with:

- `bench-field`
- `bench-pairing`

Do not add `bench-poly`, `bench-transcript`, or fixture-regeneration steps until the first two are stable and clearly useful.

### Phase 3: fixture freshness and differential verification

Add CI jobs that:

- run `zig build test` for the package
- verify checked-in fixtures still match the offline generators
- run differential fixture generation in a reproducible way
- publish benchmark deltas for the hot kernels

### Blocking gates

Start with non-blocking benchmark reporting.

After the baseline is stable, promote only these to blocking perf checks:

- field mul
- field reduction
- `sumOfProducts`
- Dory-relevant MSM sizes
- full pairing

## Immediate Action Plan

Implement in this order:

1. Add checked-in fixtures for BN254 field ops, Blake2b RFC cases, Jolt transcript compatibility, small MSM cases, and small pairing cases.
2. Add vector-driven tests in the existing source files instead of creating a separate parallel test hierarchy.
3. Reuse root `bench-msm` as the first benchmark gate while adding package-local `field_micro.zig`.
4. Add offline vector generators for `arkworks` and `gnark-crypto`.
5. Extend the package build with `bench-field` and `bench-pairing` only after the first fixtures are in place.

## Bottom Line

The package already has meaningful inline coverage. The missing foundation is fixture-backed differential testing and a package-local benchmark workflow centered on the hot arithmetic path.

For `zolt-arith`, the right baseline is:

- keep the current inline tests
- add exact fixtures where bytes matter
- use `arkworks` and `gnark-crypto` as BN254 oracles
- reuse existing MSM benchmarking first
- add package-local field and pairing benches before lower-value benchmark work
