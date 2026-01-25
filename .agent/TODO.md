# Zolt-Jolt Compatibility TODO

## 🎯 Current Task: Implement Ra Polynomial Generation

**Status:** Deserialization ✅ | Ra Polynomials ✅ (generated) | Stage 4 Verification ❌ (values incorrect)

### Problem

Ra polynomials are now generated with actual trace data (no longer zeros), but Stage 4 sumcheck still fails with different values:

```
[JOLT BATCHED] output_claim = 13790373438827639882557683572286534321489361070389115930961142260387674941556
[JOLT BATCHED] expected_output_claim (sum) = 12640480056023150955589545284889516342512199511163763258096899648096280534264
=== SUMCHECK VERIFICATION FAILED ===
```

**Root cause**: Pre/post value tracking for RdInc/RamInc may not match Jolt's approach. Need to investigate how Jolt's tracer stores register pre/post values vs Zolt's approach.

### What Needs to Be Implemented

**Location**: `src/zkvm/mod.zig:919-956`

1. **RdInc** - Register destination increment polynomial
   - Formula: `rd_inc[i] = post_value[rd] - pre_value[rd]`
   - Track per-register pre/post values across trace
   - Size: `reg_poly_size` (padded to power of 2)

2. **RamInc** - RAM increment polynomial
   - Formula: `ram_inc[i] = post_value[addr] - pre_value[addr]`
   - Track per-address pre/post values across memory trace
   - Size: `memory_poly_size` (padded to power of 2)

3. **InstructionRa[0..31]** - Instruction read address chunks
   - Extract instruction address chunks using OneHot decomposition
   - Each chunk: `(addr >> shift[i]) & ((1 << log_k_chunk) - 1)`
   - 32 polynomials for LOG_K=128, log_k_chunk=4

4. **RamRa[0..ram_d-1]** - RAM read address chunks
   - Extract RAM address chunks using OneHot decomposition
   - Number of chunks depends on log2(ram_k)

5. **BytecodeRa[0..bytecode_d-1]** - Bytecode read address chunks
   - Extract bytecode address chunks using OneHot decomposition
   - Number of chunks depends on log2(bytecode_k)

### Reference Implementation

**Jolt**: `/Users/matteo/projects/jolt/jolt-core/src/zkvm/witness.rs:56-210`
- `CommittedPolynomial::stream_witness_and_commit_rows()`
- Shows how to build each polynomial type from execution trace

**Zolt execution trace data**:
- Register trace: `emulator.trace.steps.items[i].rd_value`
- Memory trace: `emulator.ram.trace.accesses.items[i].value`
- Instruction addresses: `emulator.trace.steps.items[i].pc`
- RAM addresses: `emulator.ram.trace.accesses.items[i].address`

### Action Plan

1. **Study Jolt's witness generation** to understand Ra polynomial construction
2. **Implement RdInc/RamInc** using execution trace to track increments
3. **Implement address chunk extraction** for InstructionRa/RamRa/BytecodeRa
4. **Test** with fibonacci example - verify Stage 4 passes
5. **Verify** all 578+ Zolt tests still pass

---

## Current Progress

| Stage | Internal (Zolt) | Cross-verify (Jolt) | Notes |
|-------|-----------------|---------------------|-------|
| 1 | ✅ PASS | ✅ PASS | - |
| 2 | ✅ PASS | ✅ PASS | - |
| 3 | ✅ PASS | ✅ PASS | - |
| 4 | ✅ PASS | ❌ FAIL | **ACTIVE**: Ra polynomials are zeros |
| 5 | ✅ PASS | - | Blocked by Stage 4 |
| 6 | ✅ PASS | - | Blocked by Stage 4 |

**Recent Progress**:
- Session 57: ✅ Commitment serialization - all 37 commitments generated and serialized
- Session 58: ✅ Ra polynomial implementation - all polynomials use real trace data (no longer zeros)

---

## Commands

```bash
# Generate proof (without SRS for faster iteration)
zig build run -- prove examples/fibonacci.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Test cross-verification
cd /Users/matteo/projects/jolt/jolt-core && cargo test test_verify_zolt_proof_with_zolt_preprocessing --release -- --nocapture --ignored

# Run Zolt tests
zig build test

# Quick build check
zig build
```

## Success Criteria

- ✅ Deserialization works (all 37 commitments read correctly)
- ❌ Stage 4 sumcheck passes with Jolt verifier
- ❌ All 578+ Zolt tests pass
- ❌ Zolt proof fully verifies with Jolt for Fibonacci example

---

## Recent Session History

**Session 58 (2026-01-25)**: ⚡ Implemented Ra polynomial generation - RdInc, RamInc, InstructionRa, RamRa, BytecodeRa all generate real values from trace. Stage 4 still fails but with different output_claim, indicating progress. Need to investigate pre/post value tracking.

**Session 57 (2026-01-25)**: ✅ Fixed serialization - implemented all 37 commitments (RdInc, RamInc, InstructionRa[], RamRa[], BytecodeRa[])

**Session 56 (2026-01-25)**: 🔥 Identified root cause - commitment count mismatch (5 vs 37)

**Session 55 (2026-01-24)**: Fixed double-batching in Stage4GruenProver

**Session 52-54**: Cross-verification setup, deep code audit

See `.agent/SERIALIZATION_BUG_FOUND.md` and `.agent/BUG_FOUND.md` for detailed analysis