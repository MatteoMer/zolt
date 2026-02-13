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
- [x] All fixes committed and pushed (commit a5f7595, 976a162)
- [x] Proof generation complete: 70,145 bytes, Time: 3674 seconds (~61 min)
  - Proof saved to /tmp/collatz_jolt_proof.bin

## IN PROGRESS
- [ ] Preprocessing export for collatz.elf (RUNNING - PID 2838766)
  - Proof file exists: /tmp/collatz_jolt_proof.bin (70,145 bytes)
  - Preprocessing NOT yet written: /tmp/collatz_preprocessing.bin
  - Process is in DoryVerifierSetup.fromSRS() which does ~3069 pairings
  - Estimated remaining time: ~80-90 minutes (started at ~72 min mark)
  - Process started: Feb 13 ~15:32 UTC
  - Command: zig-out/bin/zolt prove --jolt-format -o /tmp/collatz_jolt_proof.bin --export-preprocessing /tmp/collatz_preprocessing.bin examples/collatz.elf

## NEXT STEPS
1. Wait for preprocessing export to complete
   - File: /tmp/collatz_preprocessing.bin
2. Run Jolt verifier:
   ```
   jolt/target/release/zolt-verifier --proof /tmp/collatz_jolt_proof.bin --preprocessing /tmp/collatz_preprocessing.bin
   ```
3. If collatz passes, run regression tests for all 8 programs
   - Script ready: .agent/regression_test.sh
4. If verification fails, debug the error

## KEY FINDINGS
- Pure Zig BN254 pairing is ~2s each (no assembly optimization)
- Dory opening proof: 8 rounds, ~50 min total (768 pairings in round 0 alone)
- Preprocessing export: DoryVerifierSetup.fromSRS does ~3069 pairings = ~100 min
- No SRS caching between proof generation and preprocessing export
- Total end-to-end time: ~2.5 hours for a single small program (collatz)
- Future optimization: multi-Miller loop, SRS caching, parallelization
