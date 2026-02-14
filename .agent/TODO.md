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
  - R0 state before: e98e2657...1cb84d (both sides)
  - R0 state after:  df9c12ef...30c36c (both sides)
  - R0 n_rounds: 1552 (both sides)
  - R0 challenge values are SAME (just displayed differently: BigInt vs canonical)

## CURRENT STATUS
Stage 6 batched sumcheck still fails, but the transcript/challenges are in sync.
The sumcheck ROUNDS all pass (p(0)+p(1)=claim for all 16 rounds).
But the FINAL OUTPUT CLAIM doesn't match the EXPECTED OUTPUT CLAIM:
- output_claim = 10068428028361103562999431687109727156273651794029258343283738030662297904304
- expected_output_claim = 18661973779153731974364406210909728062712157382685357377790163200435919635328

The expected_output_claim is computed from instance evaluations at the final r_sumcheck point.
The output_claim is the evaluation of the last round polynomial at the last challenge.

This means one or more INSTANCE polynomials have incorrect coefficients.

## NEXT STEPS
1. Identify which instance(s) contribute incorrect expected_output_claim
   - Add per-instance expected_output_claim vs actual comparison
   - The verifier already prints per-instance expected_output_claim (see logs)
2. Debug the specific instance polynomial computation
3. Fix and verify
4. Regression test all 8 programs

## KEY INSIGHT
The challenges are the SAME between prover and verifier - confirmed via:
- Transcript state comparison (32 bytes match)
- n_rounds comparison (both = 1552 at R0)
- Coefficient comparison (both see same 5 coefficients per round)
The issue is purely in the POLYNOMIAL CONTENT, not the Fiat-Shamir transcript.

## DEBUG NOTES
- Must use `--jolt-format` flag with `-o` flag to trigger the correct code path
- std.debug.print output appears in stderr when using --jolt-format
- The prover crashes (exit=134) during deinit but proof IS saved correctly
- Previous runs without --jolt-format use a DIFFERENT code path that doesn't call generateStage6Proof
- MontU128Challenge Debug trait shows BigInt([0,0,L,H]) = L*2^128 + H*2^192 (NOT field value)
  This caused confusion when comparing challenges - they looked different but were the same

## FILES
- Proof: /tmp/collatz_jolt_proof_fix.bin
- Preprocessing: /tmp/collatz_preprocessing_fix.bin
- Verifier log: /tmp/verifier_output_new.log
- Prover log: /tmp/zolt_jolt_stderr.log
- Diagnostics: /tmp/s6p_diag.bin, /tmp/s6p_r0_challenge.bin, /tmp/s6p_state_after_r0.bin
