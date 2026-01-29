# Zolt-Jolt Compatibility: Current Status

## Status: Stage 2 Sumcheck Verification Failure 🔴

## Session 74 Summary (2026-01-29)

### Analysis: Stage 2 Output Claim Mismatch

**Problem:**
Zolt generates proof where Stage 2 sumcheck fails:
- `output_claim` (sumcheck evaluation) ≠ `expected_output_claim` (verifier formula)
- Values: 21381532... vs 7589737... (different by factor ~3x)

**Key Insight: Zolt's Internal Consistency is CORRECT**
- `STAGE2_FINAL: output_claim` = `{ 181, 30, 249, 122, ... }` (LE bytes)
- `expected_batched (from provers)` = `{ 35, 43, 4, 85, ... }` (BE bytes)
- These ARE the same value (LE reversed = BE) ✓

**Therefore:** The issue is Jolt's verifier computing a DIFFERENT expected_output_claim.

### Stage 2 Architecture (5 Instances)

| Idx | Instance | Rounds | Start Round | input_claim |
|-----|----------|--------|-------------|-------------|
| 0 | ProductVirtualRemainder | 8 | 16 | uni_skip_claim |
| 1 | RamRafEvaluation | 16 | 8 | RamAddress@SpartanOuter |
| 2 | RamReadWriteChecking | 24 | 0 | RamReadValue + γ*RamWriteValue |
| 3 | OutputSumcheck | 16 | 8 | 0 |
| 4 | InstructionLookupsClaimReduction | 8 | 16 | LookupOutput + γ*Left + γ²*Right |

max_rounds = 24, n_cycle_vars = 8, log_ram_k = 16

### Expected Output Claim Formulas (Jolt Verifier)

**Instance 0 (ProductVirtualRemainder):**
```
tau_high_bound_r0 * eq(tau_low, r_tail_reversed) * fused_left * fused_right
```
- Reads claims from SpartanProductVirtualization

**Instance 4 (InstructionLookupsClaimReduction):**
```
eq(opening_point, r_spartan) * (lookup_output + γ*left + γ²*right)
```
- opening_point = normalized Stage 2 challenges[16..23]
- r_spartan = LookupOutput's opening point from SpartanOuter
- Claims from InstructionClaimReduction

### Suspected Root Cause

The expected_output_claim computation requires opening claims stored at specific (VirtualPolynomial, SumcheckId) pairs. If Zolt stores these claims differently from what Jolt expects:
- Different values
- Different sumcheck_ids
- Different ordering/endianness

Then Jolt's verifier formula produces wrong result.

### Current Opening Claims in Zolt Proof (sample)

From log:
- `claim[13]` = RamReadValue@SpartanOuter = **ZERO**
- `claim[14]` = RamWriteValue@SpartanOuter = **ZERO**
- These are correct for fibonacci (no load/store ops)

### Next Steps

1. **HIGH PRIORITY**: Run Jolt with `zolt-debug` feature to see per-instance expected values
   - Requires: `sudo apt-get install pkg-config libssl-dev`

2. Compare batching_coeffs between Zolt and Jolt
   - Zolt Stage 2 coeffs: `82,247,239,154...`, `198,106,87,42...`, etc.

3. Check Instance 4's r_spartan handling
   - r_spartan[0] = `{ 11, 189, 190, 138, ... }` (from Stage 1)
   - Stage 2 challenge[16] = `[6a, 48, b2, db, ...]`

### Blocking Issue

Cannot run Jolt verification test - missing system dependencies:
```
pkg-config: command not found
openssl-sys build failed
```

Workaround: Analyze from Zolt logs and Jolt source code.

## Previous Sessions

### Session 73 (2026-01-29)
- Fixed SumcheckId mismatch (22 values, not 24)
- Fixed proof serialization (5 advice options, 5 usize config)
- Proof deserializes completely ✓

### Session 72 (2026-01-28)
- 714/714 unit tests passing
- Stage 3 mathematically verified

### Session 71 (2026-01-28)
- Instance 0 verified correct
- Synthetic termination write
