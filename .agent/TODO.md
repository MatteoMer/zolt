# Zolt-Jolt Compatibility - Status Update

## Current Status: Session 13

### Stage 2 Sumcheck Failure - ROOT CAUSE IDENTIFIED!

- **Stage 1: PASSING ✅** - Sumcheck output_claim matches expected
- **Stage 2: FAILING ❌** - output_claim != expected_output_claim

```
output_claim:          15813746968267243297450630513407769417288591023625754132394390395019523654383
expected_output_claim: 21370344117342657988577911810586668133317596586881852281711504041258248730449
```

### ROOT CAUSE

The bug is in `generateStage2BatchedSumcheckProof` in `proof_converter.zig`:

For instances 1 (RamRafEvaluation), 2 (RamReadWriteChecking), and 4 (InstructionLookupsClaimReduction):
- Zolt generates **constant polynomials** instead of actual sumcheck round polynomials
- These instances have **non-zero input claims**, so they MUST compute actual polynomials
- The sumcheck invariant `s(0) + s(1) = claim` is violated because Zolt's constant polynomial approach doesn't satisfy this relationship

### Evidence

At round 0:
- Initial batched claim = 712179532459811457325748625852270562576712947602781856366289191649144450642
- Zolt's combined polynomial has s(0) + s(1) = 19476553769496810873727250934421201282694881988612493455483545179070188702623
- These don't match! The sumcheck verifier expects s(0) + s(1) = claim

### Input Claims (all non-zero except OutputSumcheck)
- input_claim[0] (ProductVirtualRemainder): 12851287012901887478695803449826071783573256819376888577491820047958512367456
- input_claim[1] (RamRafEvaluation): 2650649598265260989182717046219406276180311178055872518194047386832825074409
- input_claim[2] (RamReadWriteChecking): 21887007497497759329777797652503861135248043541615729717104019635173577208018
- input_claim[3] (OutputSumcheck): 0 (only this one is zero!)
- input_claim[4] (InstructionLookupsClaimReduction): 10135704891399761968224133532100949168728422535041076985926780318168130954991

### What's Verified to Work
1. ✅ Initial batched claim matches between Zolt and Jolt
2. ✅ All 26 round challenges match exactly
3. ✅ All c0, c2, c3 coefficients are written correctly (byte-level match)
4. ✅ ProductVirtualRemainder prover works
5. ✅ OutputSumcheck prover works

### What's Broken
For instances 1, 2, 4 - when they are ACTIVE (round_idx >= start_round):
- Zolt produces constant polynomials instead of actual sumcheck polynomials
- The polynomial contribution doesn't satisfy the sumcheck invariant

### FIX OPTIONS

**Option A: Implement Missing Provers (Hard but Correct)**
Implement actual provers for:
- RamRafEvaluation
- RamReadWriteChecking
- InstructionLookupsClaimReduction

Each needs to maintain state and compute proper round polynomials.

**Option B: Zero-Instance Detection (Medium)**
If these instances have zero input claims (which they should for fibonacci with no RAM), the constant polynomial approach is correct. The bug might be that Zolt is computing non-zero claims incorrectly.

Need to verify:
- Are the R1CS input evaluations for RAM fields correct?
- Should ram_address_claim, ram_read_value_claim, etc. be 0 for fibonacci?

**Option C: Simple Fix for Zero-Sum Instances**
For instances where the "output" should be zero (read-write consistency, etc.), the round polynomial should be `s(x) = 0` for all x, which is what the constant polynomial approach produces when input_claim = 0.

### Next Steps

1. Check if RAM-related R1CS input evaluations should be 0 for fibonacci
2. If not, fix the evaluation computation
3. If yes (and they're not), fix why they're non-zero
4. If everything's correct and claims ARE non-zero, implement proper provers

## Previous Sessions

### Session 12
- Found that all polynomial coefficients and challenges match
- Identified that s(0)+s(1) != claim at round 0

### Session 11
- Verified extensive formula matching between Zolt and Jolt
- Identified that polynomial indexing/storage must match exactly

### Session 10
- Fixed output-sumcheck r_address_prime reversal
- Stage 1 started passing
