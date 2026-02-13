# Zolt → Jolt Verification Progress

## Current Status
**7/8 programs pass all 8 stages. gcd needs REMW/DIVW 21-step decomposition.**

### Verified Programs (ALL 8 stages pass):
- ✅ fibonacci.elf (trace-length 128)
- ✅ factorial.elf (trace-length 128)
- ✅ sum.elf (trace-length 128)
- ✅ signed.elf (trace-length 128)
- ✅ primes.elf (trace-length 128)
- ✅ collatz.elf (trace-length 128)
- ✅ bitwise.elf (trace-length 128)

### Programs not yet working:
- ❌ gcd.elf - Needs REMW/DIVW 21-step decomposition

### Recent Fixes (This Session):
1. **VirtualSRLI handler in Jolt verifier** - `from_raw_words` was missing an explicit
   handler for `Instruction::VirtualSRLI`, causing it to fall through to the generic
   W-extension opcode mapping which produced wrong flags/imm. Fix: added explicit
   VirtualSRLI case that calls `virtual_srli_entry` with correct bitmask from bytecode.

2. **Preprocessing termination entries** - `BytecodePreprocessing.preprocess` didn't
   include the 3 termination store entries (LUI+ADDI+SB) before power-of-2 padding.
   This caused bytecode_K mismatch between prover (which adds +3 in computeBytecodeCodeSize)
   and the serialized preprocessing. Fix: added termination entries to preprocessing.

3. **VirtualAdvice JSON serialization** - VirtualAdvice instruction has an extra `advice`
   field that other instructions don't have. The JSON serializer wasn't including it,
   causing deserialization failure on programs with REMUW (like primes). Fix: added
   `advice:0` field when variant == VirtualAdvice.

### Next Steps:
1. Implement REMW/DIVW 21-step decomposition for gcd.elf
2. Clean up debug prints from stage6_prover.zig
3. Consider adding more test programs
