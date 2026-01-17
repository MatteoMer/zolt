# Zolt-Jolt Compatibility TODO

## Current Progress

| Stage | Status | Notes |
|-------|--------|-------|
| 1 | ✅ PASS | Univariate skip sumcheck |
| 2 | ❌ FAIL | RamReadWriteChecking round polynomial mismatch |
| 3 | ⏳ Blocked | Waiting for Stage 2 |
| 4 | ⏳ Blocked | Waiting for Stage 3 |
| 5-7 | ⏳ Blocked | Waiting for Stage 4 |

## Stage 2 Root Cause Analysis (Session 37)

### Key Discovery: Fibonacci Has 1 Memory Operation!
- Entry at cycle=54, addr=2049 (RAM address 0x80004008)
- This is for output/termination mechanism
- The RWC sumcheck is NOT a zero polynomial!

### Current Implementation Status
- Added GruenSplitEqPolynomial to RWC prover
- Modified computePhase1Polynomial to use Gruen formula
- Round polynomials are being computed but still not matching Jolt

### Root Cause: Complex Sparse Matrix Structure

Jolt's RWC prover uses a sophisticated sparse matrix structure:
1. **Cycle-major matrix** for Phase 1 (binding cycle variables)
2. **Address-major matrix** for Phase 2 (binding address variables)
3. **Even/odd row pairing** for computing quadratic coefficients

The key function `ReadWriteMatrixCycleMajor::prover_message_contribution`:
- Takes even_row and odd_row entries (entries at rows 2j and 2j+1)
- Computes `[q(0), q(∞)]` coefficients for the row pair
- Uses `inc_evals = [inc_0, inc_infty]` where inc_infty = inc_1 - inc_0

The formula for each entry pair involves:
```rust
// For each (even, odd) entry pair at address k:
ra_0 * (val_0 + gamma * (val_0 + inc_0))     // X=0 contribution
ra_infty * (val_infty + gamma * (val_infty + inc_infty))  // slope-of-slope
```

Where:
- `ra_0`, `ra_infty` are the ra polynomial values at even/odd rows
- `val_0`, `val_infty` are the val polynomial values at even/odd rows
- `inc_0`, `inc_infty` are the inc values (inc_infty = inc_1 - inc_0)

### Implementation Required

To match Jolt exactly, Zolt's RWC prover needs:

1. **Sparse matrix reorganization**:
   - Group entries by row pairs (rows 2j and 2j+1)
   - Handle cases where even or odd entry may be missing

2. **Proper inc computation**:
   - `inc_0 = inc.get_bound_coeff(j_prime)`
   - `inc_1 = inc.get_bound_coeff(j_prime + 1)`
   - `inc_infty = inc_1 - inc_0`

3. **Entry contribution formula**:
   - Implement `compute_evals(even, odd, inc_evals, gamma) -> [q_0, q_infty]`
   - Sum contributions weighted by E_out * E_in

4. **Gruen formula application**:
   - `gruen_eq.gruen_poly_deg_3(q_constant, q_quadratic, previous_claim)`

### Files to Reference

- `jolt-core/src/subprotocols/read_write_matrix/cycle_major.rs` - Entry struct and compute_evals
- `jolt-core/src/subprotocols/read_write_matrix/address_major.rs` - Phase 2 computation
- `jolt-core/src/zkvm/ram/read_write_checking.rs` - Phase 1/2 orchestration

### Debug Output Added
```
[RWC PHASE1] round=X, q_constant=..., q_quadratic=..., current_claim=...
[RWC PHASE1] result: s0=..., s1=...
[GRUEN ROUND X] q(0)=..., q(1)=..., previous_claim=...
```

## Testing Commands

```bash
# Build and run prover
zig build && ./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin --srs /tmp/jolt_dory_srs.bin

# Run Jolt verifier
cd /Users/matteo/projects/jolt && ZOLT_LOGS_DIR=/Users/matteo/projects/zolt/logs cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Next Steps

1. Implement proper even/odd row pairing in RWC
2. Compute inc_evals correctly for each row pair
3. Implement entry contribution formula matching Jolt
4. Test and verify round polynomials match
