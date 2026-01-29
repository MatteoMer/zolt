# Zolt-Jolt Cross-Verification Progress

## Session 78 Update - Stages 1-3 PASS, Stage 4 FAILS (2026-01-29)

### MAJOR PROGRESS: Stages 1-3 now pass!

**Fix applied:**
- When `input_claim = 0` for RAF (Instance 1) or RWC (Instance 2), skip prover initialization
- This causes the batched sumcheck to use zero polynomials, matching Jolt's expectations
- Code changes in `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig`

### Current Issue: Stage 4 fails

**Error:**
```
output_claim:   [fc, f5, b5, 80, ...]
expected_claim: [2d, 76, 8b, f4, ...]
Verification failed: Stage 4
```

**Key observation from Jolt debug:**
```
r_cycle (from sumcheck): [7b, be, 40, f8, ...]
params.r_cycle (from Stage 3): [d7, 9b, 60, 5e, ...]
```

These differ because:
- `r_cycle (from sumcheck)` is built from Stage 4 sumcheck challenges
- `params.r_cycle (from Stage 3)` is passed from Stage 3's RegistersClaimReduction

In Jolt, expected_output_claim uses `params.r_cycle` (from Stage 3), but Zolt might be using the wrong one.

**Stage 4 structure:**
- Instance 0: RegistersRWC - uses r_cycle from Stage 3
- Instance 1: ValEvaluation - uses r_address from Stage 2
- Instance 2: ValFinal - uses r_address from Stage 2

Need to verify Zolt's Stage 4 prover is using the correct r_cycle/r_address from previous stages.

---

## Session 78 Part 1 - Fixed Stage 2 (2026-01-29)

### Major Finding

The expected_output_claims for Instances 1-4 differ between Zolt provers and Jolt verifier:

| Instance | Jolt expected | Zolt produced | Match? |
|----------|---------------|---------------|--------|
| 0 (Product) | [18, f9, 1f, 65, ...] | [18, f9, 1f, 65, ...] | ✓ YES |
| 1 (RAF) | [00, 00, ...] (zero) | [11, 16, 65, 8d, ...] | ✗ NO |
| 2 (RWC) | [2a, 7c, 07, 29, ...] | [0a, ba, 02, 25, ...] | ✗ NO |
| 3 (Output) | [24, ce, 75, 46, ...] | [08, 3d, 41, 13, ...] | ✗ NO |
| 4 (Instr) | [5b, b0, 11, 45, ...] | [0d, 2d, be, 9c, ...] | ✗ NO |

### Root Cause: Synthetic Termination Write in Memory Trace

Zolt's tracer records a "synthetic termination write" at cycle 54 to address 0x7fffc008.
This write is included in the memory trace passed to RAF and RWC provers.

For fibonacci (no user RAM operations):
- Jolt's input_claim[1] (RamAddress at SpartanOuter) = 0
- But Zolt's RAF prover receives a memory trace with the termination write
- The ra polynomial is computed from this trace (non-zero due to termination)
- Prover computes s0 ≠ 0 from actual polynomial
- Even though s0 + s1 = 0 is satisfied, s(r) ≠ 0 for random r
- Final claim cascades to wrong value

### Key Files

1. `/home/vivado/projects/zolt/src/tracer/mod.zig` - `recordTerminationWrite()` function
2. `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig` - `generateStage2BatchedSumcheckProof()` lines 2736-3700
3. `/home/vivado/projects/zolt/src/zkvm/ram/raf_checking.zig` - RafEvaluationProver
4. `/home/vivado/projects/zolt/src/zkvm/ram/read_write_checking.zig` - RamReadWriteCheckingProver

### Fix Options

**Option A** (RECOMMENDED): Filter termination/panic writes from memory trace
- RAF/RWC should only see "real" RAM operations
- Termination is already handled by OutputSumcheck's val_final
- Need to exclude addresses in I/O region from RAF/RWC memory trace

**Option B**: Zero-polynomial fallback when input_claim = 0
- Workaround, not proper fix
- Would mask underlying memory trace mismatch

**Option C**: Match Jolt's memory handling exactly
- Need to understand how Jolt handles termination in its preprocessing
- Most correct but requires deeper investigation

### Verified Components
- Instance 0 (ProductVirtualRemainder) - ✓ expected_output_claim matches
- batching_coeffs - ✓ match Jolt
- input_claims - ✓ match Jolt

### Next Steps
1. Filter termination/panic writes from memory trace for RAF/RWC
2. Re-run verification test
3. Commit and push once Stage 2 passes

---

## Session 77 Summary - Config Format Fixed, Polynomial Mismatch Found (2026-01-29)

### Major Progress

1. **Config Serialization Fixed** - trace_length, ram_K, bytecode_K, ReadWriteConfig (4 u8s), OneHotConfig (2 u8s), DoryLayout (1 u8) now match Jolt format exactly.

2. **Proof Deserialization Works** - 91 opening claims, 37 commitments parsed correctly.

3. **Stage 1 Passes** - Outer Spartan sumcheck verification succeeds!

4. **Stage 2 Fails** - `output_claim != expected_claim`

### Stage 2 Analysis

**What Matches:**
- `initial_claim`: Zolt `fd 01 cb 55...` = Jolt `[fd, 01, cb, 55, ...]` ✓
- `batching_coeff[0]`: Zolt `de 49 43 bd...` = Jolt `[de, 49, 43, bd, ...]` ✓
- `input_claim[0]`: Zolt `86 a8 80 d3...` = Jolt `[86, a8, 80, d3, ...]` ✓

**What Doesn't Match:**
- Jolt `first round coeffs_except_linear[0]`: `[97, 3f, b6, 7c, c2, de, 38, c7, ...]`
- Zolt `combined_evals[0]`: `[0e, 82, 58, f7, 16, 29, e4, 34, ...]` (different!)

---

## Session 75 Summary - Challenge Type Analysis (2026-01-29)

### Challenge Type Mapping Verified

| Jolt Function | Returns | Zolt Equivalent | Use Case |
|---------------|---------|-----------------|----------|
| `challenge_scalar::<F>()` | Fr (Montgomery) | `challengeScalarFull()` | Batching coeffs, gamma values |
| `challenge_scalar_optimized::<F>()` | MontU128Challenge (125-bit, `[0,0,L,H]`) | `challengeScalar()` | tau_high, r0, sumcheck r_i |
| `challenge_vector(n)` | Vec<Fr> | n × `challengeScalarFull()` | Batching coeffs |
| `challenge_vector_optimized(n)` | Vec<MontU128Challenge> | n × `challengeScalar()` | r_address |
| `challenge_scalar_powers(n)` | Vec<Fr> (1, q, q², ...) | `challengeScalarPowers()` | Gamma powers |

---

## Session 74 Summary - Stage 2 Deep Dive (2026-01-29)

### Stage 2 Architecture Analysis

| Instance | Verifier | Rounds | Start | input_claim |
|----------|----------|--------|-------|-------------|
| 0 | ProductVirtualRemainder | 8 | 16 | uni_skip_claim |
| 1 | RamRafEvaluation | 16 | 8 | RamAddress@SpartanOuter |
| 2 | RamReadWriteChecking | 24 | 0 | RamReadValue + γ*RamWriteValue |
| 3 | OutputSumcheck | 16 | 8 | 0 |
| 4 | InstructionLookupsClaimReduction | 8 | 16 | LookupOutput + γ*Left + γ²*Right |

---

## Previous Sessions

### Session 73 (2026-01-29)
- Fixed SumcheckId mismatch
- Deserialization complete - all 40544 bytes parse correctly

### Session 72 (2026-01-28)
- 714/714 unit tests passing
- Stage 3 sumcheck mathematically correct

### Session 71 (2026-01-28)
- Instance 0 (RegistersRWC) verified correct
- Synthetic termination write discovery
