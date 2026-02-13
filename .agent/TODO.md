# Zolt→Jolt Cross-Verification Progress

## COMPLETED
- [x] All Stage 1-5 fixes (R1CS, operands, serialization, preprocessing)
- [x] Stages 1-5 sumcheck PASS with Jolt verifier
- [x] Stage 6 rd=0 sentinel fix (commit 015a76d)
- [x] Stage 6 termination address fix (commit 0a6080d)
- [x] Stage 6 termination SD flags fix (commit 7f615e8)
- [x] All 28 bytecode entries now match between prover and verifier
  - circuit_flags: ✓ all match
  - instruction_flags: ✓ all match
  - imm values: ✓ all match
  - rd/rs1/rs2: ✓ all match (including rd=0→Some(0) for JAL/JALR)

## CURRENT ISSUE
Stage 6 batched sumcheck still FAILS despite all bytecode entries matching.
```
output_claim     = 2421321862140450572589135325982982518639318250339047471642946425026197275291
expected_output  = 17917843374658554570928839023959203648978462696942486780182230761311923996214
```

The batched sumcheck has 6 instances with 16 rounds (max degree 5):
1. BytecodeReadRaf (instance 0, num_rounds=16)
2. HammingBooleanity (instance 1, num_rounds=11)
3. Booleanity (instance 2, num_rounds=15)
4. RamRaVirtual (instance 3, num_rounds=11)
5. LookupsRaVirtual (instance 4, num_rounds=11)
6. IncClaimReduction (instance 5, num_rounds=11)

## NEXT STEPS FOR DEBUGGING
1. The bytecode entries match perfectly → the val_poly should be identical
2. Need to check if there's an issue with:
   - How the imm field is interpreted as a field element (signed vs unsigned)
   - How the address field is interpreted (Zolt uses hex addresses, Jolt uses u64)
   - The int_poly (identity polynomial/RAF) computation
   - The eq_r_cycle evaluation
   - The RA (random access) claims from the proof
   - The OTHER 5 instances (not just BytecodeReadRaf)
3. Possible next investigation:
   - Print Val poly evaluations from both prover and verifier and compare
   - Check if the mismatch is in BytecodeReadRaf specifically or in another instance
   - Verify that the opening_claims in the proof match what the verifier expects

## FILES
- Proof: /tmp/collatz_jolt_proof.bin (70,145 bytes)
- Preprocessing: /tmp/collatz_preprocessing.bin (26,880 bytes)
- Verifier output: /tmp/verifier_output3.txt
- Prover log: /tmp/zolt_prove_collatz3.log
