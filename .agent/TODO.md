# Zolt→Jolt Cross-Verification Progress

## COMPLETED
- [x] All Stage 1-5 fixes (R1CS, operands, serialization, preprocessing)
- [x] Stages 1-4 sumcheck PASS with Jolt verifier (confirmed Feb 14)
- [x] Stage 6 fixes: rd=0 sentinel, termination address, SD flags, val_poly gamma[0]
- [x] ALL 5 BCRAF stages now match (Stage 6 internal diagnostics pass)
- [x] Booleanity gamma fix
- [x] Transcript states and challenges CONFIRMED matching
- [x] **Fix SB anchor bytecode entry: VirtualInstruction=false, DoNotUpdateUnexpandedPC=true**
  - All BCRAF stages match (raf_match=1, val_only==ext=1)
- [x] Fix opening_point double-free in JoltProofBundle.deinit()

## CURRENT STATUS: Stage 5 (InstructionReadRaf) FAILS

This was a PRE-EXISTING issue (also failed in old logs/jolt_verify.log).
Stages 1-4 pass. Stage 5 fails with sumcheck output_claim ≠ expected_claim.

### What Stage 5 consists of (3 batched instances):
1. **Instance 0**: RegistersValEvaluation (8 rounds)
2. **Instance 1**: RamRaClaimReduction (24 rounds)
3. **Instance 2**: InstructionReadRaf (137 rounds) - instruction lookup RAF

### Failure details:
- All 137 sumcheck rounds pass internally (no round-level failures)
- The final output_claim from sumcheck ≠ expected_claim from verifier
- This means the sumcheck polynomial evaluations are self-consistent,
  but the verifier's expected_output_claim (computed from opening claims) differs
- Same pattern as the BCRAF Stage 6 issue we just fixed

### Key diagnostic output:
```
output_claim:   [b8, d1, 95, 50, c0, 53, 29, f4, ...]
expected_claim: [d1, 5a, 76, 4e, 54, 17, 4e, 44, ...]
```

Per-instance claims from verifier:
- Instance 0: [0b, eb, ae, 28, ...] * coeff = [ee, 6c, 5d, ...]
- Instance 1: [40, 66, e7, 2d, ...] * coeff = [65, 80, 3e, ...]
- Instance 2: [3d, 55, 53, 90, ...] * coeff = [80, 6d, da, ...]

`manual f0+f1+f2` ≠ `expected_output_claim` - suspicious, may be debug issue.

## INVESTIGATION PLAN
1. The expected_output_claim is computed from opening claims for each instance
2. The output_claim comes from the prover's sumcheck polynomial
3. Since the sumcheck rounds all pass, the issue is in the FINAL evaluation
4. Need to check what the verifier computes as expected_output_claim for each instance
5. Compare with what the prover's polynomial evaluates to at the final point
6. Likely another field/flag mismatch similar to the BCRAF issue

### Possible causes:
- InstructionReadRaf val polynomial construction mismatch (instruction flags, lookup tables)
- RamRaClaimReduction claim components wrong
- RegistersValEvaluation claim computation differs
- Gamma/batching coefficient mismatch between prover and verifier

## NEXT STEPS
1. Add diagnostics to Stage 5 prover to compare per-instance expected_output_claim
2. Check if the InstructionReadRaf val polynomial uses correct table entries
3. Check if RamRaClaimReduction's 4 sub-claims are correct
4. After fixing: verify Stage 6 passes (our BCRAF fix should make it work)
5. Regression test all 8 programs

## KEY FILES
- Proof: /home/vivado/projects/zolt/logs/zolt_proof_dory.bin (70145 bytes)
- Preprocessing: /home/vivado/projects/zolt/logs/zolt_preprocessing.bin (26880 bytes)
- Prover log: /tmp/zolt_sbfix3_stderr.log
- Jolt verifier: /home/vivado/projects/jolt/jolt-core/src/zolt_compat_test.rs
