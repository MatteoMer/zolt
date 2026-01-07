# Zolt-Jolt Compatibility - Stage 2 Progress

## Current Status
- Stage 1: PASSING ✅
- Stage 2: FAILING ❌ (input_claims mismatch in batched sumcheck)
- Stage 3+: Not reached yet

## ROOT CAUSE IDENTIFIED! (Session 7)

### The Problem

Stage 2 batched sumcheck has 5 instances:
1. ProductVirtualRemainder: input_claim = uni_skip_claim ✅
2. RamRafEvaluation: input_claim = RamAddress from SpartanOuter ❌ (Zolt uses 0)
3. RamReadWriteChecking: input_claim = RamReadValue + gamma * RamWriteValue ❌ (Zolt uses 0)
4. OutputSumcheck: input_claim = 0 ✅
5. InstructionLookupsClaimReduction: input_claim = LookupOutput + gamma * LeftOperand + gamma^2 * RightOperand ❌ (Zolt uses 0)

Zolt is using F.zero() for instances 1, 2, 4 but Jolt expects:
- input_claim[1] = 2650649598265260989182717046219406276180311178055872518194047386832825074409
- input_claim[2] = 21887007497497759329777797652503861135248043541615729717104019635173577208018
- input_claim[4] = 10135704891399761968224133532100949168728422535041076985926780318168130954991

### The Fix Required

1. Get opening claims from the proof:
   - RamAddress from SpartanOuter
   - RamReadValue from SpartanOuter
   - RamWriteValue from SpartanOuter
   - LookupOutput from SpartanOuter
   - LeftLookupOperand from SpartanOuter
   - RightLookupOperand from SpartanOuter

2. Sample gamma challenges from transcript:
   - gamma_ram_rwc (for RamReadWriteChecking)
   - gamma_instr (for InstructionLookupsClaimReduction)

3. Compute input_claims:
   - input_claim[1] = RamAddress
   - input_claim[2] = RamReadValue + gamma_ram_rwc * RamWriteValue
   - input_claim[4] = LookupOutput + gamma_instr * LeftOperand + gamma_instr^2 * RightOperand

## Verified Components
1. ✅ Field element serialization (LE bytes match arkworks)
2. ✅ Factor evaluations (l_inst, r_inst, etc. match)
3. ✅ Transcript synchronization through Stage 1
4. ✅ tau_high sampling matches
5. ✅ Opening claims loaded correctly from proof
6. ✅ uni_skip_claim for ProductVirtualRemainder
7. ✅ Stage 2 sumcheck polynomial serialization

## Next Steps
1. Fix Stage 2 input_claims computation in proof_converter.zig
2. Sample gamma challenges at the right points in the transcript
3. Compute input_claims using the correct formulas
4. Re-test Stage 2 verification
