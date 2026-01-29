# Zolt-Jolt Compatibility: Current Status

## Status: Ready for Jolt Verification ⏳

## Session 72 Summary (2026-01-29)

### Verified ✓

1. **714/714 Unit Tests Pass** - All Zolt internal tests passing
2. **Stage 3 Sumcheck Mathematically Correct** - All 8 rounds verify p(0)+p(1)=claim
3. **Individual Instance Claims Correct** - shift, instr, reg claims match at each round
4. **Opening Claims Storage/Retrieval** - Claims correctly stored by Stage 1/2 and read by Stage 3
5. **Transcript Flow** - Matches Jolt's design (gamma derivation, input_claims append, batching coeffs)

### Current Blocker

Cannot run Jolt verifier due to missing system dependencies:
- `pkg-config` package
- `libssl-dev` package

These require sudo access which is not available.

### Next Steps (Requires Dependencies)

1. **Install dependencies**:
   ```bash
   sudo apt-get install pkg-config libssl-dev
   ```

2. **Run Jolt verification**:
   ```bash
   cd /home/vivado/projects/zolt/jolt
   cargo test zolt_compat -- --ignored --nocapture
   ```

3. **If verification fails**, the error message will indicate:
   - Which stage fails
   - What claim doesn't match
   - This will pinpoint exactly where Zolt differs from Jolt

### Alternative Approach

Generate a reference proof with Jolt and compare byte-by-byte:
```bash
cd /home/vivado/projects/zolt/jolt/examples/fibonacci
cargo run -- --save
# Creates /tmp/fib_proof.bin

# Compare with Zolt proof
xxd /tmp/fib_proof.bin > /tmp/jolt_proof.hex
xxd /tmp/zolt_test.bin > /tmp/zolt_proof.hex
diff /tmp/jolt_proof.hex /tmp/zolt_proof.hex
```

## Technical Details

### Stage 3 Input Claims Read From Opening Accumulator

**ShiftSumcheck**:
- NextUnexpandedPC @ SpartanOuter: `{ 127, 220, 98, 84, ... }`
- NextPC @ SpartanOuter: `{ 127, 220, 98, 84, ... }` (same as unexpanded)
- NextIsVirtual @ SpartanOuter: `{ 0, 0, 0, ... }`
- NextIsFirstInSequence @ SpartanOuter: `{ 0, 0, 0, ... }`
- NextIsNoop @ SpartanProductVirtualization: `{ 152, 69, 221, 72, ... }`

**InstructionInputSumcheck**:
- LeftInstructionInput @ SpartanOuter
- RightInstructionInput @ SpartanOuter
- LeftInstructionInput @ SpartanProductVirtualization
- RightInstructionInput @ SpartanProductVirtualization

**RegistersClaimReduction**:
- Rs1Value @ SpartanOuter
- Rs2Value @ SpartanOuter
- RdWriteValue @ SpartanOuter

### Proof File Generated

- Path: `/tmp/zolt_test.bin`
- Size: ~40KB
- Contains: All sumcheck proofs, opening claims, commitments

## Completed This Session

- [x] Verified Stage 3 sumcheck mechanics are correct
- [x] Confirmed opening claims are stored/retrieved correctly
- [x] Traced full Stage 3 flow through all 8 rounds
- [x] Updated documentation with detailed findings

## Remaining Work

- [ ] Install dependencies and run Jolt verifier
- [ ] Compare proof with Jolt reference
- [ ] Fix any discrepancies identified by verifier
