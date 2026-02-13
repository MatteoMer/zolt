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

## IN PROGRESS
- [ ] Test Jolt verifier with new proof+preprocessing (generating now)
- [ ] Regression test all 8 programs with Jolt verifier

## NEXT STEPS
- Commit and push all fixes
- Test with all 8 example programs

## KEY FINDINGS
- Jolt's `define_rv32im_trait_impls!` macro at instruction/mod.rs:273-289 defines which
  instructions have circuit_flags()/lookup_table() implementations
- SB, SH, SW are NOT in this list (only SD is)
- SLL, SLLI, SRA, SRL, etc. are NOT in this list (only their Virtual* equivalents)
- Store instructions must be decomposed into inline sequences before entering bytecode
- For our test programs, no raw SB/SH/SW appear in the ELF bytecode
- The only SB was in the termination store virtual sequence → changed to SD
- Proof deserialization now works: "Deserialized OK (compressed format)"
- Stages 1-4 pass; Stage 5 was panicking on SB → should be fixed now
