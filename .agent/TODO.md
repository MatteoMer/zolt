# Zolt-Jolt Compatibility TODO

## Current Progress

### ✅ Stage 1 - PASSES
- Outer Spartan sumcheck with univariate skip
- All round polynomials verified

### ✅ Stage 2 - PASSES
- Batched sumcheck with 5 instances
- ProductVirtualRemainder, RamRAF, RamRWC, OutputSumcheck, InstructionLookupsClaimReduction

### ✅ Stage 3 - PASSES
- Shift, InstructionInput, RegistersClaimReduction sumchecks
- Fixed InstructionInput witness consistency
- Fixed RegistersClaimReduction prefix challenge ordering

### 🔄 Stage 4 - PROVER IMPLEMENTED BUT INCORRECT
**Status:** Stage 4 prover is integrated but produces incorrect polynomial evaluations.

**What's Implemented:**
- `src/zkvm/spartan/stage4_prover.zig` - Full Stage 4 prover module
- Integrates with proof_converter via execution_trace in ConversionConfig
- Generates round polynomials and opening claims

**Root Cause of Failures:**
The sumcheck polynomial evaluations don't satisfy `p(0) + p(1) = claim`. This indicates:
1. The polynomial construction from trace is incorrect
2. The indexing scheme (k * T + j) may not match Jolt's MLE ordering
3. The register value tracking may have bugs

**Jolt's Implementation Notes:**
- Uses `ReadWriteMatrixCycleMajor` - a sparse representation
- Binds **cycle variables first** (log_T rounds), then address variables (LOG_K rounds)
- Uses Gruen optimization for sparse matrix sumcheck
- The MLE evaluation order matters critically

**Key Difference:**
Jolt's RegistersReadWriteChecking prover uses:
1. Sparse matrix representation for efficiency
2. Complex entry types (`RegistersCycleMajorEntry`) that track ra, wa, val together
3. Special handling for "implicit" entries (zeros)

Our implementation uses dense arrays which should be correct in principle but may have ordering issues.

**To Debug Further:**
1. Compare the input claim computation against what Jolt computes
2. Verify the eq polynomial evaluation matches Jolt's `EqPolynomial::mle_endian`
3. Check if r_cycle from Stage 3 is in correct format (big-endian vs little-endian)

### ❌ Stages 5-7 - BLOCKED

## Key Technical Findings

### Variable Ordering (IMPORTANT!)
Jolt's RegistersReadWriteChecking:
- **First log_T rounds:** Bind cycle variables (j)
- **Next LOG_K rounds:** Bind address variables (k)

This is documented in `cycle_major.rs`:
> "Entries are sorted by `(row, col)` - optimal for binding cycle variables first."

### Opening Claims Flow for Stage 4
1. Stage 3 produces: RdWriteValue, Rs1Value, Rs2Value @ RegistersClaimReduction
2. Stage 4 RegistersReadWriteChecking:
   - gamma sampled from transcript (challengeScalarFull)
   - r_cycle = Stage 3 challenges
   - input_claim = rd_wv + γ*(rs1_rv + γ*rs2_rv)
   - Runs sumcheck over log_T + LOG_K = 8 + 7 = 15 rounds
   - Output claims: RegistersVal, Rs1Ra, Rs2Ra, RdWa, RdInc

## Files Modified
- `src/zkvm/spartan/stage4_prover.zig` - NEW: Stage 4 prover implementation
- `src/zkvm/spartan/mod.zig` - Added stage4_prover export
- `src/zkvm/proof_converter.zig` - Integrated Stage 4 prover, added execution_trace to config
- `src/zkvm/mod.zig` - Pass execution_trace in ConversionConfig

## Testing
```bash
bash scripts/build_verify.sh
```
