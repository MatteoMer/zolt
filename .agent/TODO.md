# Zolt-Jolt Compatibility - Current Status

## Summary

**Stage 1**: PASSES ✓
**Stage 2**: Not tested yet (blocked on SpartanShift)
**SpartanShift**: MISSING - Needs implementation

## Progress This Session

### 1. Fixed Dory Opening Proof (DONE)

Root cause was that the reduce-and-fold loop assumed symmetric g1_vec/g2_vec sizes.
When sigma > nu, g2_vec has fewer elements than the loop tried to access.

Fixed by:
- Tracking col_len (2^sigma) and row_len (2^nu) separately
- Using minimum of available sizes when accessing g2_vec in the loop
- Properly updating both dimensions in each round

### 2. Larger SRS Generation (DONE)

Modified Jolt's test_export_dory_srs to generate 16-variable SRS (256 G1/G2 points)
instead of 3-variable SRS (4 G1/G2 points).

### 3. Proof Now Generates Successfully (DONE)

- Proof size: 30374 bytes
- Preprocessing size: 22225 bytes
- Stage 1 sumcheck passes in Jolt verifier

### 4. Missing SpartanShift Claims (CURRENT ISSUE)

The Jolt verifier now fails with:
```
Tried to populate opening point for non-existent key: Virtual(UnexpandedPC, SpartanShift)
```

The SpartanShift sumcheck stage needs the following claims:
1. `Virtual(UnexpandedPC, SpartanShift)`
2. `Virtual(PC, SpartanShift)`
3. `Virtual(OpFlags(VirtualInstruction), SpartanShift)`
4. `Virtual(OpFlags(IsFirstInSequence), SpartanShift)`
5. `Virtual(InstructionFlags(IsNoop), SpartanShift)`

These are "shift" polynomials where f_shift(j) = f(j+1). Zolt needs to:
1. Implement the SpartanShift sumcheck prover
2. Compute the shift polynomial evaluations
3. Add these claims to the opening_claims

## Next Steps

1. [ ] Study Jolt's ShiftSumcheckProver implementation in shift.rs
2. [ ] Add VirtualPolynomial entries for shift polynomials if missing
3. [ ] Implement SpartanShift sumcheck prover in Zolt
4. [ ] Generate the shift polynomial claims
5. [ ] Test Stage 2 verification

## Testing Commands

```bash
# Build Zolt
zig build

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin --srs /tmp/jolt_dory_srs.bin

# Run Jolt verification
cd /Users/matteo/projects/jolt/jolt-core && cargo test test_verify_zolt_proof_with_zolt_preprocessing --release -- --ignored --nocapture
```

## Commits Made

1. 60c1ad1 - debug: Found OutputSumcheck zero-check failure root cause
2. b74cb76 - fix: Set panic/termination bits in val_final + correct ELF base address
3. fc2c8cc - docs: Update TODO with progress on OutputSumcheck fix
4. (pending) - fix: Dory opening proof for asymmetric matrix sizes
