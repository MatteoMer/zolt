# Zolt→Jolt Cross-Verification Progress

## COMPLETED
- [x] All Stage 1-5 fixes (R1CS, operands, serialization, preprocessing)
- [x] Stages 1-4 sumcheck PASS with Jolt verifier (confirmed Feb 14)
- [x] Stage 6 fixes: rd=0 sentinel, termination address, SD flags, val_poly gamma[0]
- [x] ALL 5 BCRAF stages now match (Stage 6 internal diagnostics pass)
- [x] Booleanity gamma fix
- [x] Transcript states and challenges CONFIRMED matching
- [x] **Fix SB anchor bytecode entry: VirtualInstruction=false, DoNotUpdateUnexpandedPC=true**
- [x] Fix opening_point double-free in JoltProofBundle.deinit()

## CURRENT STATUS: Stage 5 (InstructionReadRaf) FAILS

### Root Cause Analysis (Feb 14)
The InstructionRa opening claims from the prover don't match what the verifier
independently computes. Comparing:

**Zolt prover ra_chunks[0] (LE bytes):**
`6a 42 0a ec 4d d7 07 60 46 e9 51 fd 49 59 4d e5 84 12 17 a7 5f f4 7f 60 18 90 9a 35 31 05 15 2e`

**Jolt verifier ra_claims[0] (LE bytes):**
`90 71 1c ac b9 19 ef e7 09 e5 71 65 55 11 ea 59 9f 34 61 15 45 1f d3 33 3b 8b ad c7 38 1e b8 2e`

These should be the same. The prover sends ra_chunks as opening claims, and the
verifier uses them to compute expected_output_claim. But the verifier's values
come from the proof serialization, so either:

1. The prover computes ra_chunk opening claims incorrectly (wrong binding)
2. The serialization is wrong (values get corrupted in transit)
3. The verifier reads them with wrong byte order or parsing

### Key finding: Prover's own consistency check PASSES
- `scalar*ra_product*combined == lookups_claim: true`
- This means the prover's polynomial (round messages) is consistent with its OWN ra_product
- But the ra_product it sends as opening claims differs from what the verifier reads

### Additional finding: ra_product != lookups_ra_weights[0]
- The prover warns: "ra_product and lookups_ra_weights[0] don't match after binding"
- Comment says "binding the product != product of bindings"
- This is actually EXPECTED because Π_i(bind(ra_chunk_i)) ≠ bind(Π_i(ra_chunk_i))
- The prover uses ra_chunks[i] = ra_chunk_weights[i][0] (product of bindings)

### NEXT STEPS
1. **Compare ra_chunk serialization** - Check if the Jolt verifier reads the
   InstructionRa opening claims from the correct location in the proof bytes
2. **Check byte ordering** - Verify LE/BE consistency between Zolt serialization
   and Jolt deserialization for opening claims
3. **Compare table_flag opening claims** - Check if LookupTableFlag values also mismatch
4. **Compare raf_flag opening claim** - Check InstructionRafFlag
5. If opening claims are correct in the proof but wrong after deserialization,
   the issue is in Jolt's proof parsing code
6. If opening claims are wrong in the proof, the issue is in Zolt's Stage 5 prover

## KEY FILES
- Proof: /home/vivado/projects/zolt/logs/zolt_proof_dory.bin (70145 bytes)
- Preprocessing: /home/vivado/projects/zolt/logs/zolt_preprocessing.bin (26880 bytes)
- Prover log: /tmp/zolt_sbfix3_stderr.log
- Jolt verifier: /home/vivado/projects/jolt/jolt-core/src/zolt_compat_test.rs
- Jolt verifier log: /tmp/jolt_collatz_stderr.log
