# Zolt → Jolt Verification Progress

## Current Status
**🔧 IN PROGRESS — Stage 6 sumcheck fails despite matching Val polynomials**

Stages 1-5 pass. Stage 6 BytecodeReadRaf+Booleanity+IncClaimReduction batched sumcheck fails: `output_claim != expected_claim`.

### What's been fixed this session:
1. **Val poly imm encoding**: Format-aware encoding matching Jolt conventions:
   - I-type, J-type: unsigned u64 (zero-extended via `F.fromU64(@bitCast(imm_signed))`)
   - S-type, B-type: signed (via `fieldFromI128`)
   - U-type: unsigned u64
2. **Val poly rd=0 handling**: rd=0 is NOT treated as "no rd" for I/R/U/J-type instructions
3. **Termination store bytecode entries disabled**: Phase 2 removed for vanilla Jolt compatibility
4. **R1CS witness imm encoding**: `computeImmediate` updated to match Val poly format-aware encoding

### Current investigation:
- All 320 Val polynomial values match exactly (0 diff between Zolt and Jolt)
- Stage 6 sumcheck output_claim (from proof round polys) doesn't match expected_claim (from Val poly evaluation + other instance claims)
- The mismatch is NOT in the Val polys — it's in the sumcheck round polynomials

### Hypotheses for remaining Stage 6 failure:
1. **Booleanity instance**: The Booleanity sumcheck might have wrong polynomials due to termination store trace cycles referencing NoOp bytecode entries
2. **IncClaimReduction instance**: Ram/Rd increment claims might not match
3. **Transcript/batching divergence**: The batching coefficients for the 3 instances might not match between prover and verifier
4. **Opening claims**: The opening claims from Stages 1-5 feed into Stage 6's expected output. If these are inconsistent with the proof, Stage 6 fails.

### Key file: `computeImmediate` encoding convention
Jolt's NormalizedOperands.imm encoding depends on instruction format:
- FormatI.imm: u64 → `u64 as i128` (zero-extension, always positive)
- FormatB.imm: i128 → signed
- FormatS.imm: i64 → `i64 as i128` (sign-extension, signed)
- FormatJ.imm: u64 → `u64 as i128` (zero-extension, always positive)
- FormatU.imm: u64 → `u64 as i128` (zero-extension, always positive)

## Test Commands
```bash
# Build Zolt
cd /home/vivado/projects/zolt && zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --trace-length 64 -o /tmp/zolt_proof.bin --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin --srs /tmp/jolt_dory_srs.bin

# Verify with Jolt (debug mode)
cd /home/vivado/projects/jolt && RAYON_NUM_THREADS=1 cargo run --release --features zolt-debug --manifest-path examples/fibonacci/Cargo.toml -- --verify-zolt-proof /tmp/zolt_proof.bin --zolt-preprocessing /tmp/zolt_preprocessing.bin.ram

# Run Zig tests
cd /home/vivado/projects/zolt && zig build test
```
