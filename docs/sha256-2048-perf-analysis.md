# SHA256-2048 Performance Analysis: Zolt vs Jolt

**Date:** 2026-03-25
**Program:** SHA256 with 2048-byte input (497,102 cycles, padded to 524,288)
**Benchmark:** `ZOLT_BENCH=1 ./zig-out/bin/zolt prove examples/sha256_2048.elf ...`

## Executive Summary

Zolt is **1.63x slower** wall-clock (18.8s vs 11.6s). The prove phase is **1.74x slower**
(16.0s vs 9.2s). The root cause is **missing inner-loop parallelism** in sumcheck stages 2, 3,
and 4, plus missing delayed Montgomery reduction in stages 2 and 4.

Recoverable gap: **~4-5 seconds** from parallelizing sumcheck inner loops.

---

## Stage-by-Stage Comparison

```
Stage              Jolt (ms)  Zolt (ms)   Ratio   Gap (ms)
─────────────────  ─────────  ─────────  ──────  ─────────
Commit + Stage 1     3311.3     5184.6   1.57x     +1873
Stage 2               287.4     1827.9   6.36x     +1540   ← no parallelism
Stage 3               138.1      823.8   5.97x      +686   ← partial parallelism
Stage 4               102.9     1470.4  14.29x     +1367   ← no parallelism
Stage 5               288.3     1184.6   4.11x      +896
Stage 6               683.2     2455.8   3.59x     +1773
Stage 7              1053.0      156.0   0.15x      -897   ← Zolt faster
Stage 8              3312.9     2898.5   0.87x      -414   ← Zolt faster
─────────────────  ─────────  ─────────  ──────  ─────────
PROVE TOTAL          9177.1    16001.6   1.74x     +6825

Overhead:
Trace + witness       143.8      607.6   4.23x      +464
SRS / preprocessing  2134.3     1807.7   0.85x      -327   ← Zolt faster
Prover gen            127.6        0.4   0.00x      -127   ← Zolt faster
─────────────────  ─────────  ─────────  ──────  ─────────
WALL CLOCK          11583.8    18840.0   1.63x     +7256
```

### Zolt Sub-Stage Breakdown (from [BENCH] output)

```
Stage 1: total=1912.9  init=201.9   sumcheck=1147.9  claims=563.1
Stage 2: total=1827.9  init=390.2   sumcheck=1437.7  claims=0.0
Stage 3: total= 823.8  init=  0.0   sumcheck= 823.8  claims=0.0
Stage 4: total=1470.4  init=179.6   sumcheck=1248.6  claims=42.2
Stage 5: total=1184.6  init=  0.0   sumcheck=1184.6  claims=0.1
Stage 6: total=2455.8  init=  0.1   sumcheck=2455.6  claims=0.1
Stage 7: total= 156.0  init=154.1   sumcheck=  0.4   claims=1.5
Stage 8: total=2898.5  commit=3271.7  joint_poly=403.1  opening=2495.4
```

---

## Root Cause: Inner-Loop Parallelism Map

### Verified parallelism status per stage

| Stage | Compute parallelism | Bind parallelism | UnreducedProductAccum | Ratio |
|-------|--------------------|-----------------|-----------------------|-------|
| **1 (Outer)** | ThreadPool (streaming_outer) | ThreadPool | Yes | 1.57x* |
| **2 (Product)** | **NONE** | **NONE** | **NONE** | 6.36x |
| **3 (Shift+Instr+Regs)** | 1/3 sub-provers (InstrInput only) | 1/3 sub-provers | Partial (Shift yes, InstrInput yes) | 5.97x |
| **4 (Regs RW + ValEval)** | **NONE** | **NONE** | **NONE** | 14.3x |
| **5 (Lookups)** | ThreadPool | ThreadPool | Yes | 4.11x |
| **6 (Bytecode)** | ThreadPool | ThreadPool | Yes | 3.59x |
| **7 (Hamming)** | N/A (4 rounds) | N/A | N/A | 0.15x |

*Stage 1 ratio includes commit time bundled in Jolt's measurement.

**Pattern:** Stages with zero parallelism (2, 4) have the worst ratios.
Stage 3 has partial parallelism (1/3 sub-provers) and a mid-range ratio.
Stages 5, 6 have full parallelism and are the closest to Jolt.

---

## Detailed Findings Per Stage

### Stage 2: ProductVirtualRemainder (6.36x, +1540ms)

**File:** `src/zkvm/spartan/product_remainder.zig`, `src/zkvm/jolt_prover.zig:3404`

**5 batched instances, all fully sequential:**

| Instance | Compute | Bind | Accum type |
|----------|---------|------|------------|
| ProductVirtualRemainder | Sequential loop over E_out×E_in | Sequential `bindLow()` on left+right polys | Immediate F.mul()+F.add() |
| RamRafEvaluation | Sequential for loop | Sequential `bind()` | Immediate |
| RamReadWriteChecking | Sequential phase1/phase2 | Sequential | Immediate |
| OutputSumcheck | Sequential | Sequential | Immediate |
| InstructionLookupsClaimReduction | Sequential for loop | Sequential for loop | Immediate |

**What Jolt does differently:**
- `ProductVirtualRemainderProver::compute_message` uses `split_eq_poly.par_fold_out_in_unreduced()`:
  parallel over E_out blocks (√T ≈ 724 tasks), with `UnreducedProductAccum` inner accumulation
- `ingest_challenge` uses `rayon::join(left.bind_parallel, right.bind_parallel)`: concurrent
  binding of two T/2-element polynomials, each internally parallelized with `par_iter_mut`
- All RAM/RAF/RWC instances similarly use `into_par_iter()` in compute + bind

**Key insight:** Zolt's `DensePolynomial` already has `bindLowParallel()` — it's just not called.

### Stage 3: Shift + InstructionInput + RegistersClaimReduction (5.97x, +686ms)

**File:** `src/zkvm/spartan/stage3_prover.zig`

**3 sub-provers with mixed parallelism:**

| Sub-prover | Compute | Bind | ThreadPool field |
|------------|---------|------|-----------------|
| ShiftPrefixSuffix | Sequential (4 P×Q pairs) | Sequential | **No** |
| **InstructionInput** | **parallelReduce** | **parallelFor (9 arrays)** | **Yes** |
| RegistersPrefixSuffix | Sequential (1 P×Q pair) | Sequential | **No** |

**What Jolt does differently:**
- All three sub-provers use `par_fold_out_in_unreduced` for compute
- All use `rayon::join` or `bind_parallel` for binding
- The `SpartanShiftProver` compute_message uses Gruen split-eq with parallel fold

**Opportunities:**
- Add ThreadPool to ShiftPrefixSuffix and RegistersPrefixSuffix
- Parallelize Phase1 P×Q pair computation (4 independent pairs for Shift)
- Parallelize Phase2 witness MLE binding (multiple independent arrays)

### Stage 4: RegistersReadWriteChecking + RamValCheck (14.29x, +1367ms)

**File:** `src/zkvm/spartan/stage4_gruen_prover.zig`, `src/zkvm/ram/val_evaluation.zig`

**ZERO parallelism anywhere in hot path:**

| Component | Function | Parallelism |
|-----------|----------|-------------|
| Cycle-major sparse compute | `computeMessage` → sequential merge | **None** |
| Cycle-major sparse bind | 2-pass sequential count + fill | **None** |
| Address-major sparse compute | `computeRoundEvals` → sequential merge | **None** |
| Address-major sparse bind | 2-pass sequential count + fill | **None** |
| ValEvaluation compute | `computeRoundPolynomialCombined` → for loop | **None** |
| ValEvaluation bind | 3 sequential fold loops (inc, wa, lt) | **None** |

**What Jolt does differently:**
- Phase 1: nested `par_chunk_by` (E_out blocks × row-pairs) + recursive `rayon::join`
  for large row-pair merges (threshold 32K entries)
- Phase 2: `par_chunk_by` over column-pairs
- Phase 3: `(0..T_prime/2).into_par_iter()` for dense materialized arrays
- Binding: two-pass with parallel dry-run counts + parallel fill
- ValEvaluation: `(0..inc.len()/2).into_par_iter()` with `UnreducedProductAccum` reduce
- All binding uses `bind_parallel` (par_iter_mut with min_len=4096)

**The Stage4GruenProver struct has no thread_pool field at all.**

**ThreadPool is only used in `fromTrace` (matrix construction from execution trace) —
initialization only, not in the hot sumcheck loop.**

### Stage 5 and 6: Already Parallelized

Stages 5 and 6 already use ThreadPool extensively for both compute and bind.
Their ratios (4.1x, 3.6x) reflect remaining differences:
- Jolt's Rayon has better work-stealing than Zolt's ThreadPool for irregular workloads
- Possible algorithmic differences in sub-protocols (not investigated here)
- Jolt may have better cache locality patterns

---

## Jolt Optimizations Missing in Zolt

### 1. UnreducedProductAccum in Inner Loops (Stages 2, 4)

Jolt delays Montgomery reduction:
```rust
// Jolt inner loop — accumulate unreduced products
inner[k] += e_in.mul_to_product_accum(vals[k]);  // no reduction!
// ... loop over all x_in ...
let inner_red = F::reduce_product_accum(inner[k]); // reduce ONCE
```

Zolt's equivalent:
```zig
// Zolt inner loop — full reduction every iteration
acc[0] = acc[0].add(evals[0]);  // F.add does Montgomery reduction
```

Zolt already has `UnreducedProductAccum` (field/mod.zig:2179) and uses it in stages 1, 3, 5, 6.
**Stages 2 and 4 simply don't use it yet.**

### 2. Parallel Split-Eq Fold (Stages 2, 4)

Jolt's `par_fold_out_in_unreduced`:
```rust
(0..out_len).into_par_iter()  // √T parallel tasks
    .map(|x_out| {
        for x_in in 0..in_len { ... }  // sequential inner
        scale by e_out
    })
    .reduce_with(sum)
```

Zolt's equivalent: single sequential loop over all (E_out × E_in) entries.

### 3. Parallel Polynomial Binding

Jolt:
```rust
// Bind multiple polys concurrently
rayon::join(|| left.bind_parallel(r), || right.bind_parallel(r));

// Each bind_parallel is itself parallel
left.par_iter_mut().zip(right.par_iter()).with_min_len(4096).for_each(...)
```

Zolt: Sequential `bindLow()` calls, one polynomial at a time.
**Note:** `bindLowParallel()` exists in Zolt's DensePolynomial — it's just not called.

### 4. Sparse Matrix Parallel Merge (Stage 4)

Jolt: Two-pass with `par_chunk_by` + parallel fill + recursive `rayon::join` for large merges.
Zolt: Functions literally named `seqMergeComputeEvals`, `seqBindRows`, `seqBindCols`.

### 5. OneHotCoeffLookupTable Parallel Bind (Stage 4)

Jolt: `par_iter().flat_map(|a| table.par_iter().map(...))` — parallel table expansion.
Zolt: Sequential loop in `table.bind()`.

---

## Priority-Ordered Fix Plan

### P0 — Stage 4 Parallelism (estimated: -1200ms)

1. Add `thread_pool` field to `Stage4GruenProver`
2. Parallelize `phase1ComputeMessage`: parallel over row-pair groups (like Jolt's `par_chunk_by`)
3. Parallelize `phase2ComputeMessage`: parallel over column-pair groups
4. Parallelize `phase1Bind`/`phase2Bind`: parallel count pass + parallel fill pass
5. Add `thread_pool` to `ValEvaluationProver`:
   - Parallelize `computeRoundPolynomialCombined`: parallel over i=0..half
   - Parallelize `bindChallengeWithPoly`: bind inc/wa/lt arrays concurrently
6. Use `UnreducedProductAccum` in sparse matrix compute inner loops

### P1 — Stage 2 Parallelism (estimated: -1200ms)

1. Add `thread_pool` field to `ProductVirtualRemainderProver`
2. Parallelize `computeRoundPolynomial`: parallel fold over E_out blocks
   (like Jolt's `par_fold_out_in_unreduced`)
3. Use `bindLowParallel()` instead of `bindLow()` for left/right polynomials
4. Bind left+right concurrently (independent arrays)
5. Use `UnreducedProductAccum` in the E_out×E_in inner loop
6. Propagate thread_pool to RamRWC, RafEval, InstrLookups, OutputSumcheck

### P2 — Stage 3 Parallelism (estimated: -500ms)

1. Add `thread_pool` to `ShiftPrefixSuffixProver` and `RegistersPrefixSuffixProver`
2. Parallelize ShiftPrefixSuffix Phase1 compute (4 independent P×Q pairs)
3. Parallelize ShiftPrefixSuffix Phase2 bind (5 independent witness MLEs)
4. Parallelize RegistersPrefixSuffix similarly

### P3 — UnreducedProductAccum in Stages 2+4 (estimated: -200ms)

Already covered in P0/P1 above but called out separately because it's also valuable
without parallelism. Stages 2 and 4 do many F.mul()+F.add() in inner loops that could
use `mulToProductAccum` + single `reduce()` instead.

---

## How to Reproduce Benchmarks

```bash
# Build
zig build -Doptimize=ReleaseFast

# Run Zolt with bench output
ZOLT_BENCH=1 ./zig-out/bin/zolt prove examples/sha256_2048.elf \
    --jolt-format -o /tmp/sha256_bench.bin \
    --export-preprocessing /tmp/sha256_bench_preproc.bin 2>&1 | grep '\[BENCH\]'

# Run Jolt
JOLT_BENCH_DETAIL=1 jolt-bench/target/release/jolt-bench examples/sha256_2048.elf

# Verify proof
cargo run --release --manifest-path jolt-verifier/Cargo.toml -- \
    --proof /tmp/sha256_bench.bin --preprocessing /tmp/sha256_bench_preproc.bin

# Run side-by-side comparison (needs both binaries)
ZOLT_BENCH=1 ./bench/compare-detailed.sh sha256_2048
```

---

## Files Reference

| File | Role |
|------|------|
| `src/zkvm/jolt_prover.zig` | Stage orchestration, batched sumcheck loops |
| `src/zkvm/spartan/product_remainder.zig` | Stage 2 ProductVirtualRemainder prover |
| `src/zkvm/spartan/stage3_prover.zig` | Stage 3 (Shift + InstrInput + Registers) |
| `src/zkvm/spartan/stage4_gruen_prover.zig` | Stage 4 RegistersRWC (sparse Gruen) |
| `src/zkvm/spartan/sparse_registers.zig` | Sparse register matrix (cycle/address major) |
| `src/zkvm/ram/val_evaluation.zig` | Stage 4 RamValCheck (inc×wa×lt) |
| `src/zkvm/spartan/stage5_prover.zig` | Stage 5 (already parallelized) |
| `src/zkvm/spartan/stage6_prover.zig` | Stage 6 (already parallelized) |
| `src/field/mod.zig` | UnreducedProductAccum, mulToProductAccum |
| `src/poly/mod.zig` | DensePolynomial with bindLow/bindLowParallel |
| `src/utils/thread_pool.zig` | Chase-Lev work-stealing ThreadPool |
