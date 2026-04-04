# Zolt Code Quality Report

**Date:** 2026-04-04  
**Repository:** `zolt-code-review`  
**Scope:** Whole project, with emphasis on build hygiene, maintainability, correctness safeguards, testing strategy, and long-term developer ergonomics.

## Executive Summary

Zolt is technically ambitious and already stronger than many experimental systems projects in one important area: the core implementation is not a toy. The repository contains a real package split, a large body of low-level tests, meaningful documentation, and a prover/verifier compatibility story that is at least partially automated in practice.

The main code quality problem is not “the code is sloppy.” The problem is **quality drift around the edges of a large, fast-moving codebase**:

- The core `zig build test` path is healthy.
- Several documented or registered auxiliary paths are not healthy.
- The codebase has a strong inner loop and a weak outer shell.

Today, the project looks like an advanced research implementation that can be productive for a small expert team, but it does **not** yet enforce a consistently high engineering bar across all supported workflows. The gap to “best possible in terms of code quality” is mostly about:

1. Making every advertised path executable and continuously verified.
2. Reducing the size and cognitive load of the largest prover modules.
3. Removing silent fallback behavior and placeholder logic in correctness-critical code.
4. Converting documentation and compatibility claims into automated checks.
5. Establishing a real release/CI/formatting discipline.

If those five things are done well, the project’s quality level can move from “promising experimental system” to “serious, durable, high-trust systems codebase.”

## Review Method

This report is based on:

- Repository structure inspection.
- Static scan of Zig and Rust sources.
- LOC and test-count measurements.
- Targeted reading of build scripts, CLI code, compatibility layers, and large prover modules.
- Verification runs of the main build/test commands.

### Commands Run

```bash
zig build test
zig build bench-cycle
zig build metal-shaders
zig fmt --check $(find src packages examples bench -name '*.zig' -type f)
cargo test --manifest-path jolt-verifier/Cargo.toml
cargo test --manifest-path jolt-bench/Cargo.toml
```

### Observed Outcomes

| Command | Result | Notes |
|---|---|---|
| `zig build test` | Pass | Main Zig test path is healthy |
| `zig build bench-cycle` | Fail | `bench/cycle_compute/main.zig` missing |
| `zig build metal-shaders` | Fail | Build script points at non-existent shader path |
| `zig fmt --check ...` | Fail | 45 Zig files are not formatted according to `zig fmt` |
| `cargo test --manifest-path jolt-verifier/Cargo.toml` | Pass | 0 tests executed |
| `cargo test --manifest-path jolt-bench/Cargo.toml` | Fail | Upstream API drift in the Rust benchmark crate |

## Repository Snapshot

| Metric | Value |
|---|---|
| Zig source files under `src/` and `packages/` | 119 |
| Total Zig LOC under `src/` and `packages/` | 111,398 |
| Mean Zig file size | 936 LOC |
| Median Zig file size | 451 LOC |
| Files >= 1,000 LOC | 34 |
| Files >= 2,000 LOC | 16 |
| Files >= 3,000 LOC | 7 |
| Zig `test` blocks | 566 |
| Files failing `zig fmt --check` | 45 |
| Root CI/workflow files found | 0 |
| Root style/automation files found (`.editorconfig`, pre-commit, markdown lint, etc.) | 0 |

### Largest Hotspots

| LOC | File |
|---:|---|
| 7,065 | `src/zkvm/spartan/stage5_prover.zig` |
| 6,704 | `src/zkvm/spartan/stage6_prover.zig` |
| 5,163 | `src/tracer/mod.zig` |
| 4,075 | `packages/zolt-arith/src/poly/commitment/dory.zig` |
| 3,944 | `src/zkvm/instruction/lookups.zig` |
| 3,286 | `src/zkvm/spartan/bytecode_entries.zig` |
| 3,123 | `src/zkvm/spartan/stage3_prover.zig` |

## What Is Already Good

### 1. There is real modular intent

The split into `zolt-pool`, `zolt-arith`, and the main `zolt` package is a legitimate quality improvement. It creates a foundation for ownership boundaries, focused testing, and cleaner dependency reasoning.

### 2. The project has substantial low-level test coverage

566 Zig test blocks is a meaningful number. The arithmetic, polynomial, transcript, RAM, lookup, thread-pool, and prover-support layers are not untested.

### 3. The repository is honest about its maturity

The `README.md` explicitly warns that the project is experimental and unaudited. That is a good quality signal because it sets correct expectations instead of overselling.

### 4. There is evidence of active architectural self-correction

`docs/research/refactoring-plan.md` shows the team is aware of file-size and modularity problems and has already acted on them. That matters. The project is trying to get better, not just accumulate code.

### 5. Dependency pinning exists

The Rust support crates pin upstream revisions explicitly. That is better than floating dependencies in a compatibility-sensitive project.

## Highest-Priority Findings

## 1. Build Graph Drift Is Real

This is the most immediate quality problem because it breaks trust in the repository’s declared interfaces.

### Evidence

- `build.zig` defines `bench-thresh` and `bench-cycle` from `bench/cycle_compute/...`, but that directory does not exist.
- See `build.zig:183-213`.
- Running `zig build bench-cycle` fails with `file_hash FileNotFound`.
- `build.zig` defines `metal-shaders` using `src/gpu/shaders`, but the actual shader files live under `packages/zolt-arith/src/gpu/shaders`.
- See `build.zig:344-386`.
- Running `zig build metal-shaders` fails with missing `src/gpu/shaders/*.metal`.

### Why It Matters

If a build step is registered, it becomes part of the project contract. Broken optional steps create three forms of damage:

- They waste contributor time.
- They make documentation less trustworthy.
- They hide drift until it becomes expensive to repair.

### Recommendation

Create a policy that every step registered in `build.zig` must be in one of two states:

1. Fully working and covered by CI.
2. Removed from the build graph.

Do not keep dead-but-advertised steps.

### Concrete Actions

- Remove or restore `bench-thresh` and `bench-cycle`.
- Fix `metal-shaders` to use `packages/zolt-arith/src/gpu/shaders`.
- Add a `zig build ci` step that includes:
  - `zig build test`
  - `zig build -Doptimize=ReleaseFast`
  - `zig fmt --check`
  - platform-specific optional steps on appropriate runners

## 2. The Public Build Contract Is Under-Tested

The exported dependency module in `build.zig` is not exercised the same way as internal code paths.

### Evidence

- The exported `zolt` module is added in `build.zig:47-55`.
- That module only declares `zolt_pool` as an import.
- `src/root.zig` imports both `zolt_pool` and `zolt_arith`.
- Internal examples and executables use `lib.root_module`, not the exported dependency contract.

### Why It Matters

This means the project’s “how another Zig project consumes `zolt` as a dependency” path is not being validated by normal repository builds. That is exactly the kind of issue that only appears for external users.

### Recommendation

Add a tiny smoke-consumer integration test:

- Create a minimal fixture project under `testdata/consumer/`.
- Build it in CI.
- Ensure it imports the public `zolt` module exactly the way downstream users will.

## 3. Placeholder Logic Remains in a Correctness-Critical Lookup Path

This is the most serious code-level quality concern.

### Evidence

`src/zkvm/lookup_table/mod.zig:1364-1391` contains 16 `TODO`-tagged lookup-table cases that currently return `F.zero()`:

- `UpperWord`
- `Pow2W`
- `VirtualRev8W`
- `VirtualROTR`
- `VirtualROTRW`
- `VirtualChangeDivisor`
- `VirtualChangeDivisorW`
- `MulUNoOverflow`
- `VirtualXORROT32`
- `VirtualXORROT24`
- `VirtualXORROT16`
- `VirtualXORROT63`
- `VirtualXORROTW16`
- `VirtualXORROTW12`
- `VirtualXORROTW8`
- `VirtualXORROTW7`

### Why It Matters

Returning zero is a dangerous placeholder in proof-related code because it looks like valid behavior. If a path is accidentally exercised, the failure mode is semantic corruption, not an immediate loud error.

### Recommendation

Replace placeholder behavior with one of:

- `@compileError` for unsupported table families.
- An explicit `error.UnsupportedLookupTable`.
- A feature flag that excludes unsupported tables from the build.

The one thing not to do is silently return a value that can masquerade as real computation.

### Required Follow-Up

- Add tests that enumerate all supported table IDs.
- Add negative tests that assert unsupported IDs fail loudly.
- Document the currently supported instruction/lookup surface in one machine-checked place.

## 4. Documentation and Compatibility Claims Have Drifted

The repo currently says different things in different places.

### Evidence

- `README.md:92` says the verifier is pinned to upstream commit `2e05fe88`.
- `jolt-verifier/Cargo.toml:7-9` pins upstream to `997c1543`.
- `jolt-bench/Cargo.toml:7-8` also pins `997c1543`.
- `docs/research/refactoring-plan.md:100-102` says `zig build` is clean and verified, but specialized build steps currently fail.

### Why It Matters

In a compatibility-sensitive zkVM project, documentation drift is not a cosmetic problem. It undermines:

- reproducibility,
- upstream compatibility debugging,
- onboarding,
- release confidence.

### Recommendation

Create a single source of truth for upstream compatibility:

- One file containing:
  - current upstream Jolt revision,
  - current arkworks branch/revision,
  - last verified proof-compatibility date,
  - supported proof fixtures.

Then generate or validate README claims from that source in CI.

## 5. Auxiliary Rust Tooling Is Not Held to the Same Standard as the Main Path

### Evidence

- `cargo test --manifest-path jolt-verifier/Cargo.toml` passes, but it runs 0 tests.
- `cargo test --manifest-path jolt-bench/Cargo.toml` fails.
- The failure in `jolt-bench/src/main.rs:72` shows a tuple arity mismatch with `guest::program::decode`.
- The failure in `jolt-bench/src/main.rs:118-124` shows an API mismatch for `JoltSharedPreprocessing::new` and `ProverPreproc::new`.

### Why It Matters

The repository’s compatibility story is split across Zig and Rust. If the auxiliary crates drift, then “we are compatible with upstream” becomes only partially true.

### Recommendation

Choose one of these two strategies and enforce it consistently:

1. Treat `jolt-verifier` and `jolt-bench` as first-class supported tooling and keep them green in CI.
2. Mark `jolt-bench` as experimental/non-supported and remove it from normal quality expectations until maintained again.

Half-supported tooling is worse than clearly unsupported tooling.

## 6. The Biggest Prover Modules Are Still Too Large

This is the main long-term maintainability problem.

### Evidence

Seven files are above 3,000 LOC and sixteen are above 2,000 LOC. The biggest ones include:

- `src/zkvm/spartan/stage5_prover.zig` at 7,065 LOC
- `src/zkvm/spartan/stage6_prover.zig` at 6,704 LOC
- `src/tracer/mod.zig` at 5,163 LOC
- `packages/zolt-arith/src/poly/commitment/dory.zig` at 4,075 LOC

### Why It Matters

Very large files degrade quality in predictable ways:

- Harder code review
- Harder local reasoning
- Higher merge-conflict rate
- Higher formatting drift
- Debugging pressure on a few “expert-only” files

### Recommendation

Do not split these files arbitrarily. Split them by responsibility.

### Suggested Boundaries

For `stage5_prover.zig`:

- round orchestration
- polynomial/materialization kernels
- transcript interaction
- state structs and lifecycle
- worker-thread orchestration

For `stage6_prover.zig`:

- phase transitions
- round polynomial generation
- transcript compression/binding logic
- debug/diagnostic helpers
- proof object construction

For `tracer/mod.zig`:

- instruction decode/dispatch
- execution handlers by opcode family
- trace recording
- memory/register side effects

For `dory.zig`:

- commitment building
- opening/proof generation
- serialization
- hint/precomputation logic
- test-only helpers

## 7. Duplicate Domain Models Increase Divergence Risk

### Evidence

Two separate `JoltDevice` / `MemoryLayout` implementations exist:

- `src/common/jolt_device.zig`
- `src/zkvm/jolt_device.zig`

The structures are conceptually similar but not identical:

- `panic_addr` vs `panic`
- one side is emulator/device oriented
- one side is verifier/transcript compatibility oriented
- constants and layout logic are duplicated

### Why It Matters

In protocol code, duplicated domain models are a drift trap. You eventually get:

- subtly different layouts,
- mismatched serialization semantics,
- bug fixes applied to only one side,
- documentation that describes neither one accurately.

### Recommendation

Define one canonical memory-layout model and add thin adapters if necessary.

If the transcript-compatible representation must differ, make that difference explicit:

- canonical runtime model
- canonical serialized compatibility model
- tested conversion layer between them

## 8. Silent Failure Patterns Exist in User-Facing and Debug Paths

These patterns are understandable in prototype code, but they should be removed if the goal is top-tier quality.

### Evidence

- `src/cli/args.zig:81-94` parses invalid hex nibbles as `0` via `catch 0`.
- `src/commands/run.zig:49` swallows `emulator.step()` errors with `catch break`.
- `src/commands/run.zig:152` swallows register-read errors with `catch 0`.
- `src/main.zig` uses several `std.process.exit(1)` branches instead of structured error returns.
- `src/zkvm/spartan/stage6_prover.zig:4048-4090` writes diagnostics directly to `/tmp/...` when `debug_verbose` is enabled.

### Why It Matters

Silent fallback is the enemy of trustworthy tooling:

- Invalid input can look valid.
- Runtime errors become truncated output.
- Debugging side effects become environment-dependent.

### Recommendation

Adopt a hard rule:

- Parsing code must return typed errors.
- CLI command code must report errors, not silently coerce them.
- Debug artifacts must go through an explicit debug sink, not fixed `/tmp` paths.

## 9. Formatting and Style Enforcement Are Not Operational

### Evidence

- `zig fmt --check` fails on 45 files.
- No root `.editorconfig` or similar formatting/automation files were found.
- No root CI/workflow directory was found.

### Why It Matters

Formatting discipline is not about aesthetics in a codebase this large. It is about:

- reducing review noise,
- minimizing accidental diffs,
- making huge files less hostile,
- enabling mass refactors safely.

### Recommendation

Make formatting non-negotiable:

- Run `zig fmt` in CI.
- Add a contributor-facing `make fmt` or `zig build fmt` step.
- Add pre-commit hooks if the team wants fast feedback.

## 10. The Test Suite Is Large but Unbalanced

The current suite has good depth in libraries and internals, but weak assurance at the repository contract level.

### Evidence

- 566 Zig test blocks exist.
- Most tests are concentrated in `src/zkvm`, `packages/zolt-arith`, and `packages/zolt-pool`.
- `src/commands/` contains no tests.
- `jolt-verifier` has 0 Rust tests.
- The main compatibility claims rely more on manual or ad-hoc flows than on repository-enforced smoke tests.

### Why It Matters

A high-quality zkVM project needs both:

- deep math/kernel tests,
- and thin end-to-end contract tests.

Right now the first category is much stronger than the second.

### Recommendation

Add a tiered test model.

### Minimum Required Test Layers

| Layer | Purpose | Status |
|---|---|---|
| Unit tests | arithmetic, poly, transcripts, RAM, tables | Strong |
| Module integration tests | prover-stage interactions | Moderate |
| CLI smoke tests | `zolt run`, `zolt prove` | Weak |
| Cross-language compatibility tests | Zig prover -> Rust verifier | Weak |
| Build graph tests | optional steps and package-consumer smoke tests | Weak |
| Negative/error-path tests | malformed input, unsupported ops, compatibility mismatch | Weak |

## Detailed Recommendations By Area

## A. Build, CI, and Release Engineering

### Goal

Make the repository self-verifying.

### Recommended Standard

Every PR should answer these questions automatically:

- Does the main Zig code build?
- Do all supported build steps exist?
- Does formatting pass?
- Does the Rust verifier still build?
- Does at least one real proof still verify against upstream?
- Is the downstream dependency consumption path still valid?

### Recommended CI Matrix

| Runner | Checks |
|---|---|
| macOS arm64 | `zig build test`, `zig fmt --check`, `zig build metal-shaders`, GPU smoke tests if available |
| Linux x86_64 | `zig build test`, `zig build -Doptimize=ReleaseFast`, Rust verifier build/test |
| Compatibility job | `zolt prove examples/fibonacci.elf` then verify with `jolt-verifier` |
| Consumer job | build a minimal external Zig consumer against exported `zolt` module |

### Release Hygiene Improvements

- Add a `QUALITY.md` or `DEVELOPMENT.md` file defining the supported commands.
- Add a changelog or release notes policy for compatibility-sensitive changes.
- Treat benchmark scripts as supported only if they are green in CI.

## B. Testing and Verification Strategy

### Goal

Raise confidence without making iteration unbearably slow.

### Recommended Test Pyramid

1. Fast unit tests on every PR.
2. Medium integration tests on every PR.
3. One end-to-end prove/verify smoke test on every PR.
4. Broader proof fixtures on nightly or scheduled jobs.

### High-Value Tests To Add First

- CLI parse failures for malformed `--input-hex`.
- `zolt run` golden output for one tiny ELF.
- `zolt prove` smoke test producing a proof file.
- Rust verifier integration test that checks a known-good proof fixture.
- Unsupported lookup-table IDs fail loudly.
- Single-threaded vs thread-pooled prover equivalence.
- CPU vs GPU equivalence for selected kernels on Apple Silicon.
- Serialization round-trip goldens for proof and preprocessing files.
- Fuzzing targets for ELF parsing, serialization, and CLI input handling.

### Quality Rule For Cryptographic/Proof Optimizations

Every optimized path should have:

- a slower reference implementation,
- equivalence tests,
- determinism checks,
- serialization/golden-vector coverage.

## C. Architecture and Modularity

### Goal

Reduce the number of “hero files” that only one or two people can safely edit.

### Recommended Policies

- No file should grow past 2,000 LOC without an explicit exception.
- No protocol stage file should own state, transcript logic, worker orchestration, serialization, and debug utilities in the same module.
- Shared domain models must have one canonical definition.

### Concrete Refactoring Priorities

1. `stage5_prover.zig`
2. `stage6_prover.zig`
3. `tracer/mod.zig`
4. `dory.zig`
5. `instruction/lookups.zig`

### Refactoring Pattern To Prefer

Split by data-flow boundary, not by arbitrary line count:

- state types
- preparation/materialization
- transcript round logic
- final proof assembly
- diagnostics

That kind of split produces reusable internal APIs instead of just smaller files.

## D. Correctness and Safety Patterns

### Goal

Make invalid states loud.

### Rules Worth Adopting

- No `catch 0` or `catch null` in parsing or user input handling.
- No silent `catch break` in command code unless the error is printed and classified.
- No placeholder zero-return behavior in proof logic.
- Every `unreachable` and `@panic` in non-test code should either:
  - have an invariant comment, or
  - be replaced with a typed error.

### Memory and Resource Management

The project already does a lot of explicit `defer`/`errdefer` cleanup correctly. That is good. The next step is standardization:

- define allocator ownership conventions in one document,
- prefer explicit “caller owns/callee owns” comments on returned buffers,
- add stress tests around high-allocation prover paths,
- consider leak-checking or allocator-scope diagnostics in dedicated test builds.

## E. Documentation and Knowledge Management

### Goal

Keep docs accurate enough to be used as operational guides.

### Problems To Fix

- README compatibility revision drift.
- Refactoring-plan verification claims no longer match all build steps.
- Auxiliary tooling support status is unclear.

### Better Documentation Model

Keep these documents separate:

- `README.md` for user-facing quick start and supported workflows.
- `docs/compatibility.md` for upstream Jolt revision, fixtures, and last verified date.
- `docs/architecture.md` for package/module relationships.
- `docs/development.md` for contributor workflow and quality gates.

Do not mix historical refactoring notes with current operational truth.

## F. Developer Experience

### Goal

Make the right thing easy.

### Recommended Additions

- `zig build fmt`
- `zig build ci`
- `zig build smoke-compat`
- `zig build smoke-cli`
- `scripts/update-compat-docs.sh` or equivalent
- contributor guide for building examples, verifier, and optional GPU tooling

### Why This Matters

Large codebases decay when contributors need to remember tribal knowledge instead of relying on executable workflows.

## G. Performance Code Quality

This project is performance-sensitive, so code quality cannot mean “abstract everything away.” The right target is **performance with auditability**.

### Good Direction

- Keep hot kernels specialized.
- Keep low-level arithmetic explicit.
- Keep thread-pool and GPU code close to the metal.

### But Add These Safeguards

- Pair every optimization with a reference-path equivalence test.
- Separate benchmarking code from supported build graph unless maintained.
- Keep instrumentation structured instead of ad hoc.
- Record performance baselines in machine-readable form where possible.

## Recommended 30/60/90 Day Roadmap

## First 2 Weeks

- Fix or remove broken build steps in `build.zig`.
- Fix `metal-shaders` paths.
- Add `zig fmt --check` to the normal workflow and reformat the repo.
- Update README compatibility revision to match manifests.
- Decide whether `jolt-bench` is supported; either fix it or clearly demote it.
- Replace placeholder zero-return lookup cases with loud failure modes.

## First 30 Days

- Add one prove/verify smoke test using `examples/fibonacci.elf`.
- Add a downstream consumer smoke test for the exported `zolt` module.
- Add tests for CLI error paths and malformed hex input.
- Remove or encapsulate `/tmp` diagnostic writes.
- Start splitting `stage5_prover.zig` and `stage6_prover.zig` by responsibility.

## First 60 Days

- Unify `JoltDevice` / `MemoryLayout` modeling.
- Add compatibility metadata as a single source of truth.
- Add nightly matrix jobs for optional GPU/benchmark paths.
- Add reference-vs-optimized equivalence tests for more kernels.

## First 90 Days

- Bring hotspot files below a 2,000-3,000 LOC band wherever practical.
- Establish a documented quality bar for merges.
- Add reproducible release smoke tests.
- Turn historical refactoring docs into current architecture docs and ADRs.

## Target Quality Bar

If the goal is “best possible in terms of code quality,” this is the standard I would aim for:

### Required For Every Merge

- `zig build test` passes
- `zig fmt --check` passes
- Rust verifier build/test passes
- Supported optional build steps are green on their supported platform
- No docs/manifests disagreement on upstream revision
- No placeholder behavior in correctness-critical code

### Required For Every Release

- At least one proof generated by Zig verifies with the pinned upstream Rust verifier
- One CLI smoke test passes in a clean environment
- Compatibility document updated with exact revision and verification date
- Release notes list any protocol/serialization/compatibility changes

## Final Assessment

Zolt has a strong technical core and a serious amount of real implementation work behind it. That is the good news.

The bad news is that the project currently relies too much on human memory to keep the outer shell aligned with the inner core. Broken optional steps, documentation drift, placeholder lookup behavior, giant hotspot files, and uneven workflow enforcement all point to the same root issue: **the repository does not yet continuously enforce the quality level that its core code deserves**.

The fastest path to a high-trust codebase is not a full rewrite and not “more abstraction.” It is a disciplined tightening of contracts:

- every documented workflow must be executable,
- every supported compatibility claim must be testable,
- every critical unsupported path must fail loudly,
- every giant file must be split by responsibility,
- every contributor must have a paved path for build, format, test, and compatibility checks.

If that work is done, this repository can move from “excellent experimental prover implementation” to “excellent engineering artifact.”
