# Zolt-Jolt Cross-Verification Progress

## Session 82 - Stage 4 Sumcheck Investigation (2026-01-30)

### Summary
Stage 4 (RegistersReadWriteChecking) verification fails with sumcheck output mismatch. The root cause is that Zolt's Stage 4 sumcheck proof produces different round polynomials than what Jolt expects.

### Key Findings

1. **Stages 1-3 Pass:**
   - Stage 1: Outer Spartan sumcheck ✅
   - Stage 2: Batched sumcheck (RAF, RWC, Output, Instruction) ✅
   - Stage 3: Registers claim reduction ✅
   - Stage 4: **FAILS** ❌

2. **Stage 4 Mismatch Details:**
   - `output_claim` (from sumcheck proof verification): `2794768927403232170685203001712134750206965869554042859404932801547924672323`
   - `expected_output_claim` (computed by verifier): `19036722498929976088547735251378923562016308482664214076291639064331774676064`
   - Difference is orders of magnitude - not a small numerical error

3. **Stage 4 Architecture:**
   - 3 batched instances: RegistersRWC, ValEvaluation, ValFinal
   - Instance 1 and 2 have zero claims (expected for fibonacci example)
   - Instance 0 (RegistersRWC) carries all the weight

4. **eq_val Computation:**
   - `eq_val = eq(r_cycle_stage4, params.r_cycle_from_stage3)`
   - r_cycle_stage4: Challenges from Stage 4 sumcheck rounds
   - params.r_cycle: Stage 3's opening point (retrieved from accumulator)
   - The eq polynomial evaluates how "equal" these two points are

5. **Likely Issues:**
   - Round polynomial computation in `stage4_gruen_prover.zig`
   - Gruen optimization (`gruenPolyDeg3`) may have bugs
   - Variable ordering in 3-phase structure may not match Jolt

### Debug Data from Test Run

**Jolt Verifier Stage 4 Output:**
```
[JOLT STAGE4 DEBUG]   eq_val = 14447182824539522361174030945338492588349944286339905486292912866657917375174
[JOLT STAGE4 DEBUG]   combined = 20992970233233921422641107357873519633929136172040123830888410534632773245456
[JOLT STAGE4 DEBUG]   expected = 17266308235761105456215483608498978797273351746775005780030669392838509233139
```

**r_cycle Values:**
- Stage 4 sumcheck r_cycle[0]: `6709444460737048432665932647077461968217451116529630129102448257410915106816`
- params.r_cycle[0] (from Stage 3): `11210511683772200605067092683842474276111331544071549587645049051967513427968`

---

## Session 81 - All Stages Pass (2026-01-29)

### NOTE: This was with internal verification, NOT Jolt cross-verification

All 6 verification stages pass with Zolt's internal verifier:
- Stage 1: Outer Spartan sumcheck ✅
- Stage 2: Batched sumcheck (RAF, RWC, Output, Instruction) ✅
- Stage 3: Registers claim reduction ✅
- Stage 4: Batched sumcheck (Registers, ValEval, ValFinal) ✅
- Stage 5: Bytecode claim reduction ✅
- Stage 6: Instruction claim reduction ✅

---

## Technical Architecture

### Proof Format (Jolt-compatible)
```
[Claims: 91 entries]
[Commitments: 37 Dory G1 points]
[Stage 1: UniSkip + Sumcheck]
[Stage 2: UniSkip + Batched Sumcheck]
[Stage 3: Sumcheck]
[Stage 4: Batched Sumcheck]
[Stage 5: Sumcheck]
[Stage 6: Sumcheck]
[Stage 7: Sumcheck (Dory opening)]
```

### Stage 4 Batched Sumcheck Structure
| Instance | Verifier | Rounds | Start |
|----------|----------|--------|-------|
| 0 | RegistersReadWriteChecking | 15 (LOG_K=7 + log_T=8) | 0 |
| 1 | ValEvaluation | 24 (log_K=16 + log_T=8) | 0 |
| 2 | ValFinal | 24 (log_K=16 + log_T=8) | 0 |

Note: For fibonacci example, instances 1 and 2 have zero input_claims.

### Stage 4 RegistersRWC 3-Phase Structure
- **Phase 1** (rounds 0 to phase1-1): Bind first `phase1_num_rounds` cycle vars using Gruen
- **Phase 2** (rounds phase1 to phase1+phase2-1): Bind address vars (eq NOT bound)
- **Phase 3** (rounds phase1+phase2 to end): Bind remaining cycle vars via merged dense eq

---

## File Modifications Summary

### Core Proof Generation
- `src/zkvm/proof_converter.zig` - Main proof generation logic
- `src/zkvm/spartan/stage4_gruen_prover.zig` - Stage 4 with Gruen optimization
- `src/zkvm/ram/val_evaluation.zig` - ValEvaluation prover
- `src/zkvm/ram/val_final.zig` - ValFinal prover

### Serialization
- Jolt-compatible format using arkworks conventions
- Little-endian field elements
- Uncompressed curve points

### Transcript
- Blake2b-based Fiat-Shamir transform
- Challenge generation matches Jolt exactly (for Stages 1-3)
- Stage 4 may have transcript divergence

---

## How to Test

### Internal Verification
```bash
cd /home/vivado/projects/zolt
zig build example-pipeline
```

### Generate Jolt-compatible Proof
```bash
./zig-out/bin/zolt prove examples/fibonacci.elf \
  --jolt-format \
  --export-preprocessing logs/zolt_preprocessing.bin \
  -o logs/zolt_proof_dory.bin \
  --srs /tmp/jolt_dory_srs.bin
```

### Verify with Jolt
```bash
cd /home/vivado/projects/zolt/jolt
cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
