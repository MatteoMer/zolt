# Zolt → Jolt Cross-Verification Progress

## STATUS: ALL 8 PROGRAMS VERIFIED ✅

### Verified Programs (Mar 7 2026)
All 8 test programs produce valid proofs verified by upstream a16z/jolt verifier:
- fibonacci, factorial, bitwise, collatz, primes, sum, gcd, signed

## COMPLETED

### Full Upstream Alignment
- All transcript labels, proof format, R1CS constraints match upstream
- JAL/JALR rd=x0 remapped to virtual register 40 (upstream inline_sequence)
- Stage 6 BytecodeRa(0) aliasing fix (skip Booleanity flush when bytecode_log_k % log_k_chunk == 0)
- debug_verbose flags disabled in all prover files
- Diagnostic prints removed from upstream Rust checkout

## REMAINING TODO

### Nice to Have
- [ ] Remove the jolt/ fork directory (replace fully with jolt-verifier/)
