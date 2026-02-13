# Zolt→Jolt Cross-Verification Progress

## COMPLETED
- [x] stepREMWDIVW in tracer/mod.zig (21-step virtual sequence)
- [x] Lookup trace recording functions for new instructions
- [x] interleaveBits fix in lookup_trace.zig
- [x] getLookupTableIndex updated in stage5 and stage6 for new funct3 values
- [x] funct3 values updated in stage6 bytecode entry helpers
- [x] Jolt verifier funct3 values updated to match
- [x] materializeTableEntry for VirtualSRA and VirtualChangeDivisorW fixed (swapped x/y)
- [x] Stage 5 prover sumcheck passes (self-check)
- [x] InstructionReadRaf (Instance 2) opening claims match between prover and verifier
- [x] R1CS: XOR/AND/OR/SLT/SLTU/SRL/SRA incorrectly set AddOperands flag - FIXED
- [x] Stage 5 operand handling matched to corrected R1CS
- [x] VirtualChangeDivisorW materializeTableEntry implemented
- [x] Stage 5 lookup_output for VirtualAssertEQ/VirtualAssertValidUnsignedRemainder → output=1
- [x] Stage 5 VirtualSRLI bitmask computation for right_input
- [x] Serialization: SumcheckId COUNT 24→22, removed Advice variants
- [x] Serialization: CommittedPolynomial removed TrustedAdvice/UntrustedAdvice (5 variants not 7)
- [x] Serialization: Added 4 advice proof Option<None> fields to JoltProof
- [x] Serialization: Fixed config fields (log_k_chunk, lookups_ra_virtual_log_k_chunk as usize)
- [x] Preprocessing: Changed termination store from SB→SD (SB not in Jolt macro list)
- [x] Stage 6 VirtualSRLI bitmask: @intCast→@bitCast to avoid overflow
- [x] Stage 5 r_cycle_reduced_be buffer overflow fix
- [x] Stage 8 progress prints (std.debug.print for Dory opening proof)
- [x] All previous fixes committed and pushed (commit a5f7595, 976a162)
- [x] First proof generation complete: 70,145 bytes, Time: 3674 seconds (~61 min)
- [x] Stages 1-5 sumcheck PASS with Jolt verifier
- [x] **Stage 6 rd=0 sentinel mismatch FIX (commit 015a76d)**
  - Root cause: Zolt prover mapped rd=0 to sentinel 255 (zero contribution)
  - But Jolt stores rd=0 as Some(0) → eq_r_register[0] (non-zero contribution)
  - Fix: Remove `decoded.rd == 0` condition from sentinel mapping
  - Only S-format (0x23) and B-format (0x63) use sentinel 255 for rd (rd=None in Jolt)

## IN PROGRESS
- [ ] Proof regeneration after rd=0 fix (PID 2849680, started 17:41 UTC Feb 13)
  - Proof file generated: /tmp/collatz_jolt_proof.bin (70,145 bytes) at 18:43 UTC
  - Preprocessing export in progress (DoryVerifierSetup pairings)
  - Expected completion: ~19:30-20:00 UTC

## NEXT STEPS
1. Wait for preprocessing export to complete
2. Run Jolt verifier:
   ```
   cd /home/vivado/projects/zolt/jolt && cargo run --release -p zolt-verifier -- --proof /tmp/collatz_jolt_proof.bin --preprocessing /tmp/collatz_preprocessing.bin
   ```
3. If collatz passes, run regression tests for all 8 programs
4. If Stage 6 still fails, investigate other potential mismatches

## KEY FINDINGS
- Pure Zig BN254 pairing is ~2s each (no assembly optimization)
- Dory opening proof: 8 rounds, ~50 min total (768 pairings in round 0 alone)
- Preprocessing export: DoryVerifierSetup.fromSRS does ~3069 pairings = ~100 min
- Total end-to-end time: ~2.5 hours for a single small program
- rd=0 handling: Jolt's NormalizedOperands uses Some(0), contributing eq_r_register[0]
  to Stages 4 and 5 val polynomials. Only FormatB and FormatS have rd=None.
