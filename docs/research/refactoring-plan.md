# Zolt Refactoring Plan

**Date:** 2026-04-03
**Scope:** Full codebase restructuring — package split + internal modularization

---

## Table of Contents

1. [Work Completed](#1-work-completed)
2. [Current Codebase State](#2-current-codebase-state)
3. [Remaining Work: Deduplication](#3-remaining-work-deduplication)
4. [Remaining Work: File Splits](#4-remaining-work-file-splits)
5. [Projected Final Outcome](#5-projected-final-outcome)
6. [Appendix A: Sumcheck Implementations](#appendix-a-all-sumcheck-implementations)
7. [Appendix B: Parallelism Call Sites](#appendix-b-parallelism-call-sites-by-file)

---

## 1. Work Completed

### Phase 1: Circular Dependency Fix (DONE)

Broke the `utils/ → zkvm/lookup_table/` cycle that blocked package extraction.

| File | Change |
|------|--------|
| `src/utils/bits.zig` | NEW — canonical home for `uninterleaveBits`, `interleaveBits`, `LookupBits` |
| `src/utils/proof_serializer.zig` | NEW — extracted `ProofSerializer(F)` (was pulling in zkvm/spartan, poly, subprotocols) |
| `src/utils/mod.zig` | Removed `@import("../zkvm/lookup_table/mod.zig")`, replaced inline code with re-exports |
| `src/zkvm/lookup_table/mod.zig` | Replaced inline bit functions with re-exports from `utils/bits.zig` |

### Phase 2: Extract zolt-pool Package (DONE)

Created standalone thread pool package with zero dependencies.

```
packages/zolt-pool/
├── build.zig
├── build.zig.zon
└── src/
    ├── root.zig
    ├── thread_pool.zig    (2,006 LOC — Chase-Lev work-stealing deque)
    └── parallel_sort.zig  (225 LOC — parallel sample sort)
```

- Moved `thread_pool.zig` and `parallel_sort.zig` from `src/utils/`
- Updated 21 import sites across the codebase
- Wired into root `build.zig` / `build.zig.zon` as dependency

### Phase 3: Extract zolt-arith Package (DONE)

Created arithmetic library package (arkworks equivalent), depends on zolt-pool.

```
packages/zolt-arith/
├── build.zig
├── build.zig.zon
└── src/
    ├── root.zig
    ├── bits.zig             (from src/utils/)
    ├── expanding_table.zig  (from src/utils/)
    ├── field/               (from src/field/)
    ├── poly/                (from src/poly/)
    ├── msm/                 (from src/msm/)
    ├── gpu/                 (from src/gpu/)
    ├── transcripts/         (from src/transcripts/)
    └── subprotocols/        (from src/subprotocols/)
```

- Moved 6 directories + 2 utility files
- Updated ~210 import sites across 48 files (`@import("../field/mod.zig")` → `@import("zolt_arith").field`)
- Fixed `integration_tests.zig`, `lookup_table/mod.zig` broken paths
- Added `lt_poly` export to `poly/mod.zig` (was missing)

### Phase 4: File Splits (DONE)

Split 5 oversized files into 12 smaller modules within their packages.

#### In zolt-arith:

| Original | Split Into | LOC |
|----------|-----------|-----|
| `field/mod.zig` (3,920) | `mod.zig` (2,453) + `accumulators.zig` (1,374) + `simd_ops.zig` (179) | Largest: 2,453 |
| `field/pairing.zig` (3,534) | `pairing.zig` (1,481) + `extensions.zig` (1,276) + `g2.zig` (881) | Largest: 1,481 |
| `poly/mod.zig` (2,107) | `mod.zig` (1,273) + `interpolation.zig` (645) + `product_tree.zig` (284) | Largest: 1,273 |
| `dory.zig` (4,766) | `dory.zig` (4,075) + `point_compression.zig` (471) + `g2_msm.zig` (278) | Largest: 4,075 |

#### In zolt:

| Original | Split Into | LOC |
|----------|-----------|-----|
| `tracer/mod.zig` (5,479) | `mod.zig` (5,166) + `witness.zig` (333) | Largest: 5,166 |

Note: `tracer/rv_handlers.zig` extraction was skipped — instruction handlers are deeply
embedded as methods of the `Emulator` struct, accessing private state throughout.

### Verification

All work verified at each phase:
- `zig build test` — all tests pass (exit 0)
- `zig build` — clean in Debug and ReleaseFast
- `zolt run examples/fibonacci.elf` — 78 cycles, correct execution
- `zolt prove examples/fibonacci.elf` — 61KB proof, 186ms
- `zolt prove examples/sha256.elf` — 83KB proof, 6.4s
- **Rust Jolt verifier confirms: VERIFIED: proof is valid**

---

## 2. Current Codebase State (after Phase 5)

### Package Structure

```
zolt/
├── packages/
│   ├── zolt-pool/    (2,244 LOC)   Thread pool + parallel sort + helpers
│   └── zolt-arith/   (24,046 LOC)  Field, poly, MSM, GPU, transcripts, sumcheck
├── src/              (84,550 LOC)   zkVM, tracer, host, guest, CLI
└── total:            110,928 LOC
```

### Dependency Graph

```
zolt-pool           (leaf — zero deps)
    │
    ▼
zolt-arith          (depends on: zolt-pool)
    │
    ▼
zolt                (depends on: zolt-arith, zolt-pool)
```

### Current Largest Files

| File | LOC | Status |
|------|-----|--------|
| `stage6_prover.zig` | 12,285 | Needs split (bytecode entries + instances) |
| `stage5_prover.zig` | 8,551 | Needs split (instances) |
| `tracer/mod.zig` | 5,166 | Partially split (witness extracted, rv_handlers too embedded) |
| `jolt_prover.zig` | 4,462 | Needs proveWithTranscript breakup |
| `dory.zig` | 4,075 | Partially split (compression + G2 MSM extracted) |
| `instruction/lookups.zig` | 3,944 | Phase 5: BinaryLookup generic applied (-797 LOC) |
| `stage3_prover.zig` | 3,157 | Round loop can use orchestrator |
| `preprocessing.zig` | 2,157 | Phase 5: decoder + pc_mapper + dory_setup extracted (-957 LOC) |

---

## 3. Remaining Work: Deduplication

### Lessons Learned from Phase 5

Initial estimates were too optimistic. Key findings from implementation:

- **BinaryLookup generic** (Phase 5): estimated ~3,000, achieved **797 LOC**. Many lookups
  have subtle differences (division interleaves divisor with quotient, W-extension masks to
  32 bits, virtual instructions need self-dependent flags). ~19 lookups couldn't be converted.
- **fromBinaryLookup factory** (Phase 5): estimated ~1,500, achieved **653 LOC**. Load/store/
  virtual/jump factories have different signatures and can't use the generic.
- **File splits** add ~50-100 LOC overhead each (new imports, re-exports, build boilerplate).
  Phase 2-4 package/file splits added ~900 LOC of scaffolding that offset dedup savings.
- **Instruction lookup boilerplate was real** but the original "70% boilerplate" was overestimated.
  After conversion, lookups.zig went from 4,741 → 3,944 (17% reduction, not 70%).

**Rule of thumb:** Actual savings ≈ 30-50% of initial estimate. Always prototype on one
instance before committing to mass conversion.

### 3A. Comptime Sumcheck Orchestrator (~400 LOC savings)

> **REVISED from original ~3,600 estimate.** Deep analysis revealed the three batched
> stages use different compression formats (finite differences vs Toom-Cook vs monomial),
> different evaluation methods, and different binding patterns. A full generic must
> abstract all three axes.

**18 sumcheck implementations** share the same skeleton:

```
for round in 0..num_rounds:
    1. COMPUTE:   evals = instance.computeRoundPoly()
    2. COMPRESS:  strip c1 → [c0, c2, c3, ...]
    3. APPEND:    transcript.appendScalars("sumcheck_poly", compressed)
    4. CHALLENGE: r = transcript.challengeScalar()
    5. BIND:      instance.bind(r)
    6. UPDATE:    claim = evaluate(compressed, r)
```

**What actually varies (critical — this killed the original estimate):**

| Axis | Stage 3 | Stage 5 | Stage 6 |
|------|---------|---------|---------|
| Compression | Finite differences | Toom-Cook | Monomial extraction |
| Evaluation | `evalFromHint` | `evaluateToomCookAt` | `evalFromHintGeneral` |
| Binding | Explicit per-instance | None (constant polys) | Phase-aware per-instance |
| Max degree | 3 (fixed) | 3 (fixed) | Variable (up to ~10) |
| Instance count | 3 | 3 | 6 |

**Zig comptime solution:** Use `inline for` over `@typeInfo(InstanceTuple).Struct.fields`
to iterate heterogeneous instance types at comptime. Each instance duck-types
`computeRoundEvals()` and `bind()`. Compression strategy is a comptime enum parameter.

```zig
pub fn BatchedSumcheckOrchestrator(
    comptime F: type,
    comptime InstanceTuple: type,  // e.g., struct { shift: ShiftProver, instr: InstrProver, reg: RegProver }
    comptime config: struct {
        max_degree: usize,
        compression: enum { finite_differences, toom_cook, monomial },
    },
) type {
    const N = @typeInfo(InstanceTuple).Struct.fields.len;
    return struct {
        instances: InstanceTuple,
        batch_coeffs: [N]F,
        instance_claims: [N]F,
        num_rounds: [N]usize,
        max_num_rounds: usize,

        pub fn prove(self: *Self, transcript: anytype) !ProofResult {
            for (0..self.max_num_rounds) |round| {
                var combined: [config.max_degree + 1]F = ...;
                inline for (@typeInfo(InstanceTuple).Struct.fields, 0..) |field, i| {
                    if (inactive) { combined += scaled_constant; }
                    else {
                        const evals = @field(self.instances, field.name).computeRoundEvals(...);
                        combined += batch_coeffs[i] * extrapolate(evals);
                    }
                }
                // compress → transcript → challenge → bind (all generic)
                inline for (...) |field, i| { @field(self.instances, field.name).bind(r); }
            }
        }
    };
}
```

**Realistic savings:** ~130 LOC per stage × 3 stages = ~400 LOC, minus ~150 LOC for
the orchestrator itself = **~250 LOC net**. The value is more in correctness (single
implementation of the round loop) than LOC reduction.

**Risk:** Medium. Stage 6 has per-instance coefficient caching between compute and bind
that may not fit the generic cleanly. Start with Stage 3, verify proof, then adapt.

### 3B. Instruction Lookup (DONE in Phase 5)

- ~~instruction/lookups.zig (4,741 → ~1,500)~~ → **Achieved: 4,741 → 3,944 (-797)**
- ~~instruction/lookup_trace.zig (2,256 → ~800)~~ → **Achieved: 2,256 → 1,604 (-653)**
- ~~instruction_lookups/mod.zig DELETE~~ → **Done (-125)**
- **Total achieved: -1,575 LOC**

### 3C. Parallelism Helpers (infrastructure DONE, adoption pending)

`parallelReduceOptional` and `parallelForOptional` created in `zolt-pool/src/helpers.zig`.
The 61 call site replacements are mechanical but deferred for incremental adoption.
Each replacement saves ~3 lines. **Potential: ~180 LOC when fully adopted.**

### 3D. Shared small helpers (~200 LOC)

| Helper | Sites | Savings |
|--------|-------|---------|
| `deriveGammaPowers(F, allocator, gamma, n)` | 11 | ~44 |
| `inactiveInstanceContribution(F, claim, remaining, num_rounds)` | 8 | ~40 |
| `deriveBatchingCoeffs(F, N, transcript)` | 3 | ~12 |
| `extrapolateDeg2(F, evals)` | 2 | ~8 |
| `finiteDifferencesCompress(F, evals)` | 1 | ~12 |
| **Total** | | **~116** |

### 3E. Other Dedup (NOT YET VALIDATED — estimates may be optimistic)

| Target | Original Est. | Reality Check |
|--------|--------------|---------------|
| Shared `ValueEvaluationProver` | ~400 | RAM and Registers differ in witness count and eq array construction — may be ~150 |
| Shared EQ utilities | ~300 | Need to verify the signatures actually match across 5+ modules |
| Multi-pairing consolidation | ~200 | 4 variants with 5-10% differences — may save ~80 after refactor overhead |
| millerLoop dedup | ~100 | ~20 lines differ — easy ~80 |

---

## 4. Remaining Work: File Splits

These reorganize code for navigability but don't reduce LOC.

### 4A. Spartan Stage Provers (after dedup)

| Current | Target after dedup + split |
|---------|---------------------------|
| `stage6_prover.zig` (12,285) | `stage6_prover.zig` (~6,000) + `bytecode_entries.zig` (~1,900) + `inc_claim_reduction.zig` (~2,400) |
| `stage5_prover.zig` (8,551) | ~5,000 after dedup |
| `stage3_prover.zig` (3,157) | ~1,500 after TwoPhaseProver dedup |

### 4B. Core zkVM

| Current | Target |
|---------|--------|
| `jolt_prover.zig` (4,462) | ~2,000 — break up `proveWithTranscript` (2,225 LOC), extract constraint_evaluator + opening_claim_builder |
| `preprocessing.zig` (3,114) | ~600 — extract `instruction_decoder.zig` (1,645), `pc_mapper.zig` (130), `dory_setup.zig` (200) |
| `zkvm/mod.zig` (1,806) | ~300 — move JoltProver impl to jolt_prover.zig |

### 4C. CLI

| Current | Target |
|---------|--------|
| `main.zig` (741) | ~150 — extract `cli/argument_parser.zig`, `commands/run.zig`, `commands/prove.zig` |

### 4D. R1CS

- Extract immediate decoders from `constraints.zig` → `instruction_decoding.zig`
- Extract witness types (`CompactWitness`, `RawR1CSInputs`) → `witness_types.zig`

---

## 5. Projected Final Outcome

### Revised Estimates (grounded in Phase 5 actuals)

| Abstraction | Original Est. | Actual/Revised | Status |
|---|---|---|---|
| Generic InstructionLookup (BinaryLookup) | ~3,000 | **797** | DONE |
| Generic fromXxx factory (fromBinaryLookup) | ~1,500 | **653** | DONE |
| Dead stub removal | ~125 | **125** | DONE |
| BatchedSumcheck orchestrator (comptime) | ~2,100 | **~400** | Planned — use `inline for` over InstanceTuple |
| TwoPhaseProver generic (comptime) | ~1,500 | **~500** | Planned — comptime config for P/Q counts, witness counts |
| Shared sumcheck helpers | included above | **~116** | Planned |
| Parallel helper adoption (61 sites × 3 LOC) | ~270 | **~180** | Infrastructure done, adoption pending |
| Shared ValueEvaluationProver (comptime) | ~400 | **~200** | Use comptime config for RAM vs Register differences |
| Shared EQ utilities | ~300 | **~200** | Not started |
| Multi-pairing consolidation | ~200 | **~100** | Comptime strategy enum for prepared/unprepared/affine |
| millerLoop dedup | ~100 | **~80** | Not started |
| Prefix comptime compression | ~500 | **~300** | Comptime dispatch table for 46 prefix types |
| Debug print gating | ~350 | **~350** | Wrap behind `comptime debug_verbose` flag |
| **Total original** | **~10,400** | | |
| **Already achieved** | | **-1,575** | Phase 5 |
| **Total remaining** | | **~2,400** | Achievable with Zig comptime generics |

**Key lesson from Phase 5:** Actual LOC savings per-item run lower than initial estimates
because edge cases (division interleave, W-extension masking, virtual self-dependent flags)
require `customIndex` or config overrides. But Zig comptime handles these cleanly via
optional function pointers and config enums — the generics ARE feasible, they just need
to account for the variations as comptime parameters rather than assuming uniformity.

### Package-Level Projection

| Package | Start | Current | After All Remaining | Notes |
|---------|-------|---------|---------------------|-------|
| zolt-pool | 2,200 | 2,244 | 2,244 | Helpers done |
| zolt-arith | 22,500 | 24,046 | 23,500 | -500 from pairing/EQ/millerLoop dedup |
| zolt (source) | 98,000 | 84,550 | 82,600 | -2,000 from sumcheck/TwoPhase/helpers dedup |
| **Total** | **112,000** | **110,928** | **~108,300** | |

The real value of the refactoring is twofold:
1. **Navigability** — 3 packages, no file > 8K, clean dependencies
2. **Shared comptime infrastructure** — `BinaryLookup`, `BatchedSumcheckOrchestrator`,
   `TwoPhaseProver`, `parallelReduceOptional` — reusable primitives that make future
   protocol changes easier and less error-prone

### File-Level Targets

| Metric | Before Refactor | Current | After Phase 6 |
|--------|----------------|---------|----------------|
| Largest file | 12,285 (stage6) | 12,285 (stage6) | ~8,000 (after split) |
| Files > 3,000 LOC | 10 | 7 | 4-5 |
| Files > 2,000 LOC | 16 | 9 | 8-9 |
| Total files | 89 | ~105 | ~110 |
| Shared abstractions | 0 | 3 (BinaryLookup, fromBinaryLookup, parallel helpers) | 5+ |
| Packages | 1 | 3 | 3 |

### What Can't Shrink Further

Irreducible domain complexity — the math is unique per item:

| Area | LOC | Why |
|------|-----|-----|
| Lookup table MLEs (evaluateMLE) | 1,500 | 42 unique polynomial formulas |
| Suffix MLE implementations | 350 | Unique per-operation math |
| Identity polynomial | 420 | Already generic, lean |
| BN254 field arithmetic | 2,000 | Montgomery mul, Barrett reduce |
| Extension fields (Fp2/6/12) | 600 | Irreducible algebra |
| Miller loop + final exp | 700 | Core pairing math |
| RISC-V instruction handlers | 4,000 | 100+ opcodes, each unique |
| Dory commitment scheme | 2,500 | Sub-protocol logic |
| Per-stage sumcheck instances | 8,000 | Instance-specific witness computation (each has unique P/Q pairs, witness MLEs, binding logic) |
| R1CS constraints | 2,400 | 19 constraints + witness indices |
| Non-convertible instruction lookups | ~1,500 | Load/store/virtual/jump lookups with unique signatures |

### Zig Comptime Patterns for Generics

Use these Zig-specific patterns to handle heterogeneous types without runtime dispatch:

1. **`inline for` over struct fields** — iterate different instance types at comptime:
   ```zig
   inline for (@typeInfo(InstanceTuple).Struct.fields, 0..) |field, i| {
       const evals = @field(self.instances, field.name).computeRoundEvals(claims[i]);
       // Each iteration is a different type — resolved at compile time
   }
   ```

2. **Duck-typed interfaces** — instances just need matching method names:
   ```zig
   // Any type with computeRoundEvals() and bind() works as an instance
   pub fn computeRoundEvals(self: *Self, claim: F) [D + 1]F { ... }
   pub fn bind(self: *Self, r: F) void { ... }
   ```

3. **Comptime config structs** — parameterize behavior without runtime cost:
   ```zig
   pub fn BinaryLookup(comptime XLEN: comptime_int, comptime cfg: struct {
       table: LookupTables(XLEN),
       interleave: bool = true,
       computeResult: *const fn (u64, u64) u64,
       customIndex: ?*const fn (u64, u64) u128 = null,
   }) type { ... }
   ```

4. **Comptime enum for strategy selection** — zero-cost polymorphism:
   ```zig
   comptime compression: enum { finite_differences, toom_cook, monomial },
   ```

5. **`anytype` for transcript/allocator parameters** — avoids threading generic types everywhere.

These patterns let you build a single `BatchedSumcheckOrchestrator` that handles
stages with different compression formats, different instance counts, and different
polynomial degrees — all resolved at compile time with no runtime overhead.

### Codebase Health Audit

| Metric | Finding |
|--------|---------|
| Dead code / TODOs | Clean — 0 TODO/FIXME markers |
| Commented-out code | 2 minor instances (excellent) |
| Test coverage | All tests pass (exit 0) |
| Circular deps | None — eliminated in Phase 1 |
| Packages | 3 clean packages with verified dependency graph |
| Proof verification | Cross-verified with Rust Jolt verifier (fibonacci + SHA256) |
| Shared generics | BinaryLookup, fromBinaryLookup, parallelism helpers |
| Unimplemented stubs | 16 lookup table evaluators return F.zero() (tracked) |
| Debug output | 337 prints — gate behind comptime debug flag |

---

## Appendix A: All Sumcheck Implementations

| # | File | Instances | Degree | Batched | Phases |
|---|------|-----------|--------|---------|--------|
| 1 | subprotocols/mod.zig | 1 | 1 | No | 1 |
| 2 | spartan/stage3_prover.zig | 3 | 2-3 | Yes | 2 (prefix-suffix) |
| 3 | spartan/stage4_gruen_prover.zig | 1 | 3 | No | 2 (cycle→address) |
| 4 | spartan/stage5_prover.zig | 3 | 2-3 | Yes | 1 |
| 5 | spartan/stage6_prover.zig | 6 | 2-3 | Yes | 2 (some instances) |
| 6 | spartan/streaming_outer.zig | 1 | 3+28 | No | 1 (special first round) |
| 7 | spartan/product_remainder.zig | 1 | 2 | No | 1 |
| 8 | ram/read_write_checking.zig | 1 | 3 | No | 3 (cycle→addr→remaining) |
| 9 | ram/val_evaluation.zig | 1 | 3 | No | 1 |
| 10 | ram/val_final.zig | 1 | 3 | No | 1 |
| 11 | ram/raf_checking.zig | 1 | 3 | No | 1 |
| 12 | ram/output_check.zig | 1 | 2 | No | 1 |
| 13 | registers/val_evaluation.zig | 1 | 3 | No | 1 |
| 14 | claim_reductions/instruction_lookups.zig | 1 | 3 | No | 2 (prefix-suffix) |
| 15 | shout/prover.zig | 1 | 2 | No | 2 (address→cycle) |
| 16 | lookup_table/prefix_suffix_prover.zig | 1 | 1 | No | multi-phase |
| 17 | r1cs/jolt_r1cs.zig | 1 | 2 | No | 1 |
| 18 | jolt_prover.zig | orchestrator | varies | Yes | all stages |

## Appendix B: Parallelism Call Sites by File

| File | parallelReduce | parallelFor | parallelForForce | Total |
|------|---------------|-------------|------------------|-------|
| stage6_prover.zig | 9 | 3 | 8 | 20+ |
| dory.zig | 4 | 0 | 10+ | 14+ |
| stage3_prover.zig | 3 | 0 | 14 | 17 |
| stage5_prover.zig | 4 | 3 | 5 | 12 |
| read_write_checking.zig | 4 | 2 | 5 | 11 |
| raf_checking.zig | 3 | 1 | 6 | 10 |
| val_evaluation.zig (ram) | 2 | 1 | 4 | 7 |
| preprocessing.zig | 2 | 0 | 5 | 7 |
| streaming_outer.zig | 2 | 0 | 4 | 6 |
| msm/mod.zig | 1 | 0 | 5 | 6 |
| poly/mod.zig | 1 | 0 | 4 | 5 |
| Others (10 files) | ~22 | ~28 | ~29 | ~79 |
| **Total** | **57** | **38** | **99** | **194** |
