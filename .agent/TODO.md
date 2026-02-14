# Zolt→Jolt Cross-Verification Progress

## COMPLETED
- [x] All Stage 1-5 fixes (R1CS, operands, serialization, preprocessing)
- [x] Stages 1-5 sumcheck PASS with Jolt verifier
- [x] Stage 6 rd=0 sentinel fix
- [x] Stage 6 termination address fix
- [x] Stage 6 termination SD flags fix
- [x] Stage 6 val_poly gamma[0] fix
- [x] Stage 6 SB termination entry fix
- [x] rd=0 write inclusion fix: tracer + Stage 4/5 provers
- [x] **ALL 5 BCRAF stages now match**
- [x] Booleanity gamma fix: sample total_d=38 independent challenges (not 1)
- [x] Booleanity gamma fix: use directly as γ_i (not γ^{2i} powers)
- [x] **Transcript states and challenges CONFIRMED matching between prover and verifier**
- [x] Fix entry k=27 SB termination circuit flags
- [x] Confirmed all 6 individual instance expected_output_claims match between prover/verifier
- [x] **Fix SB anchor bytecode entry: VirtualInstruction=false, DoNotUpdateUnexpandedPC=true**
  - Root cause: bytecode entry for SB anchor had VirtualInstruction=true (from vsr.is_some())
    and DoNotUpdateUnexpandedPC=false (from vsr==0), but R1CS witness had VirtualInstruction=false
    and DoNotUpdateUnexpandedPC=true (from createTerminationStoreWitness override)
  - Fix: make bytecode entry match R1CS witness AND Jolt verifier's termination_entry_anchor()
  - Result: ALL BCRAF stages now match (raf_match=1, val_only==ext=1 for all 5 stages)
- [x] Fix opening_point double-free in JoltProofBundle.deinit()
  - opening_point was freed both by proof.deinit() and bundle.deinit()

## CURRENT STATUS
Stage 6 BCRAF match confirmed. All internal diagnostics pass:
- All BF_CHECK Phase2 rounds match=1 (rounds 0-15)
- match_val_ra=1 for Instance 0
- S6P_BCRAF_COMPARE match=1
- All field-level comparisons match (address, imm, all 13 circuit flags, RAF)

Proof generated successfully (70145 bytes). Now testing with Jolt verifier.

## NEXT STEPS
1. Wait for prover to complete (including preprocessing export)
2. Run Jolt verifier test_verify_zolt_proof_with_zolt_preprocessing
3. If Stage 6 passes, regression test all 8 programs
4. Clean up diagnostic prints

## FILES
- Proof: /tmp/collatz_jolt_proof_sbfix.bin (70145 bytes)
- Prover log: /tmp/zolt_sbfix_stderr.log
- New proof (with preprocessing): pending from sbfix2 run
