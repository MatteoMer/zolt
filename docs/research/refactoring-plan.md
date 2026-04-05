# Zolt Refactoring Plan

**Date:** 2026-04-03 (updated 2026-04-06)
**Scope:** Full codebase restructuring — package split + internal modularization

---

## Table of Contents

1. [Work Completed](#1-work-completed)
2. [Current Codebase State](#2-current-codebase-state)
3. [Remaining Work](#3-remaining-work)
4. [Projected Final Outcome](#4-projected-final-outcome)
5. [Appendix A: Sumcheck Implementations](#appendix-a-all-sumcheck-implementations)
6. [Appendix B: Parallelism Call Sites](#appendix-b-parallelism-call-sites-by-file)

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

### Phase 4: zolt-arith File Splits (DONE)

Split 4 oversized files into 11 smaller modules within zolt-arith.

| Original | Split Into | LOC |
|----------|-----------|-----|
| `field/mod.zig` (3,920) | `mod.zig` (2,453) + `accumulators.zig` (1,374) + `simd_ops.zig` (179) | Largest: 2,453 |
| `field/pairing.zig` (3,534) | `pairing.zig` (1,481) + `extensions.zig` (1,276) + `g2.zig` (881) | Largest: 1,481 |
| `poly/mod.zig` (2,107) | `mod.zig` (1,273) + `interpolation.zig` (645) + `product_tree.zig` (284) | Largest: 1,273 |
| `dory.zig` (4,766) | `dory.zig` (4,075) + `point_compression.zig` (471) + `g2_msm.zig` (278) | Largest: 4,075 |

### Phase 5: Instruction Lookup Dedup (DONE)

Applied `BinaryLookup` comptime generic and `fromBinaryLookup` factory.

- `instruction/lookups.zig`: 4,741 → 3,944 (-797 LOC)
- `instruction/lookup_trace.zig`: 2,256 → 1,604 (-653 LOC)
- `instruction_lookups/mod.zig` deleted (-125 LOC)
- **Total: -1,575 LOC**

### Phase 6: Spartan Stage Splits (DONE)

Split all three large stage provers into focused modules.

| Original | Split Into |
|----------|-----------|
| `stage6_prover.zig` (12,285) | `stage6_prover.zig` (1,798) + `stage6_instances.zig` (2,971) + `stage6_bytecode_raf.zig` (1,587) + `stage6_helpers.zig` (2,466) + `stage6_debug.zig` (445) |
| `stage5_prover.zig` (8,551) | `stage5_prover.zig` (2,600) + `stage5_instances.zig` (1,516) + `stage5_lookups.zig` (1,122) + `stage5_ram_ra.zig` (926) |
| `stage3_prover.zig` (3,157) | `stage3_prover.zig` (1,245) + `stage3_instances.zig` (1,538) + `stage3_instruction_input.zig` (372) |

### Phase 7: Core zkVM Splits (DONE)

| Original | Result |
|----------|--------|
| `jolt_prover.zig` (4,462) | `jolt_prover.zig` (2,507) + `proving_pipeline.zig` (1,447) + `stage4_prover.zig` + `stage7_prover.zig` extracted |
| `preprocessing.zig` (3,114) | `preprocessing.zig` (205) + `instruction_decoder.zig` (317) + `bytecode_pc_mapper.zig` (149) + `dory_verifier_setup.zig` (430) + `bytecode_preprocessing.zig` (1,852) |
| `zkvm/mod.zig` (1,806) | `mod.zig` (397) — JoltProver impl moved out |
| `main.zig` (741) | `main.zig` (157) + `cli/args.zig` (244) |

### Phase 8: Helpers & Dedup (DONE)

- Extracted `sumcheck_helpers.zig` (124 LOC) — shared helpers across stages
- Extracted `eq_utils.zig` (58 LOC) — shared EQ polynomial utilities
- Extracted `commitment_types.zig` (119 LOC)
- Extracted `debug.zig` (30 LOC)
- Extracted R1CS `witness_types.zig` (213 LOC)
- Unified `MemoryLayout` into single canonical definition
- Removed dead pairing code
- Deduplicated helpers across stages

### Verification

All work verified at each phase:
- `zig build test` — all tests pass (exit 0)
- `zig build` — clean in Debug and ReleaseFast
- `zolt run examples/fibonacci.elf` — 78 cycles, correct execution
- `zolt prove examples/fibonacci.elf` — 61KB proof, 186ms
- `zolt prove examples/sha256.elf` — 83KB proof, 6.4s
- **Rust Jolt verifier confirms: VERIFIED: proof is valid**

---

## 2. Current Codebase State

### Package Structure

```
zolt/
├── packages/
│   ├── zolt-pool/    (2,269 LOC)   Thread pool + parallel sort + helpers
│   └── zolt-arith/   (23,836 LOC)  Field, poly, MSM, GPU, transcripts, sumcheck
├── src/              (84,192 LOC)   zkVM, tracer, host, guest, CLI
└── total:            110,297 LOC
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

| File | LOC | Notes |
|------|-----|-------|
| `tracer/mod.zig` | 6,313 | rv_handlers too embedded to extract |
| `dory.zig` | 4,101 | Core pairing math — partially split |
| `instruction/lookups.zig` | 3,944 | After BinaryLookup dedup; remaining are non-convertible |
| `bytecode_entries.zig` | 3,306 | Extracted from stage6 |
| `stage6_instances.zig` | 2,971 | Extracted from stage6 |
| `streaming_outer.zig` | 2,956 | Special first-round sumcheck |
| `stage5_prover.zig` | 2,600 | Orchestrator after split |
| `prefixes.zig` | 2,556 | 46 prefix type implementations |
| `jolt_prover.zig` | 2,507 | After stage4/7/pipeline extraction |
| `stage6_helpers.zig` | 2,466 | Extracted from stage6 |
| `prefix_suffix_prover.zig` | 2,446 | Multi-phase sumcheck |
| `field/mod.zig` | 2,440 | After accumulators + SIMD extraction |

### Files > 3,000 LOC: 4

### Files > 2,000 LOC: 14

### Total Zig files: 131

---

## 3. Remaining Work

All file splits are **done**. The remaining work is deduplication and cleanup — optional
improvements that reduce LOC and improve correctness but are not structural.

### Lessons Learned

Initial estimates were too optimistic. Key findings from implementation:

- **BinaryLookup generic**: estimated ~3,000, achieved **797 LOC**. Many lookups
  have subtle differences (division interleaves divisor with quotient, W-extension masks to
  32 bits, virtual instructions need self-dependent flags). ~19 lookups couldn't be converted.
- **fromBinaryLookup factory**: estimated ~1,500, achieved **653 LOC**. Load/store/
  virtual/jump factories have different signatures and can't use the generic.
- **File splits** add ~50-100 LOC overhead each (new imports, re-exports, build boilerplate).
- **Rule of thumb:** Actual savings ≈ 30-50% of initial estimate. Always prototype on one
  instance before committing to mass conversion.

### 3A. Comptime Sumcheck Orchestrator (~250 LOC net)

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
    comptime InstanceTuple: type,
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

### 3B. Parallelism Helper Adoption (~180 LOC)

`parallelReduceOptional` and `parallelForOptional` created in `zolt-pool/src/helpers.zig`.
The 61 call site replacements are mechanical but deferred for incremental adoption.
Each replacement saves ~3 lines. **Potential: ~180 LOC when fully adopted.**

### 3C. Shared Small Helpers (~116 LOC)

| Helper | Sites | Savings |
|--------|-------|---------|
| `deriveGammaPowers(F, allocator, gamma, n)` | 11 | ~44 |
| `inactiveInstanceContribution(F, claim, remaining, num_rounds)` | 8 | ~40 |
| `deriveBatchingCoeffs(F, N, transcript)` | 3 | ~12 |
| `extrapolateDeg2(F, evals)` | 2 | ~8 |
| `finiteDifferencesCompress(F, evals)` | 1 | ~12 |
| **Total** | | **~116** |

### 3D. Other Dedup (NOT YET VALIDATED — estimates may be optimistic)

| Target | Original Est. | Reality Check |
|--------|--------------|---------------|
| Shared `ValueEvaluationProver` | ~400 | RAM and Registers differ in witness count and eq array construction — may be ~150 |
| Multi-pairing consolidation | ~200 | 4 variants with 5-10% differences — may save ~80 after refactor overhead |
| millerLoop dedup | ~100 | ~20 lines differ — easy ~80 |
| Prefix comptime compression | ~500 | Comptime dispatch table for 46 prefix types — may be ~300 |

### 3E. Debug Print Gating (~120 debug prints remaining in zkvm/)

Wrap behind `comptime debug_verbose` flag. Mechanical but touches many files.

---

## 4. Projected Final Outcome

### Dedup Scorecard

| Abstraction | Original Est. | Actual/Revised | Status |
|---|---|---|---|
| Generic InstructionLookup (BinaryLookup) | ~3,000 | **797** | DONE |
| Generic fromXxx factory (fromBinaryLookup) | ~1,500 | **653** | DONE |
| Dead stub removal | ~125 | **125** | DONE |
| Shared helpers dedup (cross-stage) | — | **done** | DONE |
| MemoryLayout unification | — | **done** | DONE |
| Dead pairing code removal | — | **done** | DONE |
| R1CS witness type extraction | — | **done** | DONE |
| EQ utilities extraction | — | **58** | DONE |
| Sumcheck helpers extraction | — | **124** | DONE |
| BatchedSumcheck orchestrator (comptime) | ~2,100 | **~250 net** | Not started |
| Shared sumcheck helpers (remaining) | — | **~116** | Not started |
| Parallel helper adoption (61 sites × 3 LOC) | ~270 | **~180** | Infrastructure done, adoption pending |
| Shared ValueEvaluationProver (comptime) | ~400 | **~150** | Not started |
| Multi-pairing consolidation | ~200 | **~80** | Not started |
| millerLoop dedup | ~100 | **~80** | Not started |
| Prefix comptime compression | ~500 | **~300** | Not started |
| Debug print gating | ~350 | **~200** | Not started |

**Total remaining potential: ~1,350 LOC** (conservative estimates after Phase 5 calibration)

### Package-Level Summary

| Package | Before Refactor | Current | After All Remaining |
|---------|----------------|---------|---------------------|
| zolt-pool | 2,200 | 2,269 | 2,269 |
| zolt-arith | 22,500 | 23,836 | ~23,600 |
| zolt (source) | 98,000 | 84,192 | ~83,000 |
| **Total** | **~112,000** | **110,297** | **~109,000** |

### File-Level Metrics

| Metric | Before Refactor | Current |
|--------|----------------|---------|
| Largest file | 12,285 (stage6_prover) | 6,313 (tracer/mod.zig) |
| Files > 3,000 LOC | 10 | 4 |
| Files > 2,000 LOC | 16 | 14 |
| Total Zig files | 89 | 131 |
| Shared abstractions | 0 | 5+ (BinaryLookup, fromBinaryLookup, parallel helpers, sumcheck_helpers, eq_utils) |
| Packages | 1 | 3 |

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

### Codebase Health Audit

| Metric | Finding |
|--------|---------|
| Dead code / TODOs | Clean — 0 TODO/FIXME markers |
| Commented-out code | 2 minor instances (excellent) |
| Test coverage | All tests pass (exit 0) |
| Circular deps | None — eliminated in Phase 1 |
| Packages | 3 clean packages with verified dependency graph |
| Proof verification | Cross-verified with Rust Jolt verifier (fibonacci + SHA256) |
| Shared generics | BinaryLookup, fromBinaryLookup, parallelism helpers, sumcheck_helpers, eq_utils |
| Unimplemented stubs | 16 lookup table evaluators return F.zero() (tracked) |
| Debug output | ~122 prints in zkvm/ — gate behind comptime debug flag |

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
