# Zolt-Jolt Compatibility Implementation

## Status: Session 67 - R1CS vs Memory Trace Inconsistency (Analysis Complete)

## Root Cause Analysis COMPLETE

Stage 5 sumcheck fails due to fundamental inconsistency between R1CS trace and memory trace.

### The Core Issue

1. **R1CS RamAddress polynomial**: Only includes addresses from Load/Store instructions
   - For fibonacci (no Load/Store), this is ALL ZEROS
   - Stage 1 (SpartanOuter) produces RamAddress claim = 0

2. **Memory trace**: Includes ALL memory accesses
   - Includes init/terminate writes (e.g., termination write at cycle 54)
   - Stage 5 uses this to compute RA claims

3. **RAF Evaluation (Stage 2 Instance 1)**:
   - Input claim = RamAddress from SpartanOuter = 0
   - Should prove: Σ ra(k) * unmap(k) = 0
   - But ra(k) computed from memory trace is non-zero!
   - INCONSISTENCY between contract and computation

### Current Behavior
```
[ZOLT] STAGE2: raf_final_claim = { 0, 0, 0, 0, 0, 0, ... }
[STAGE5] Computed RamRa claims from trace:
  computed_claim_raf = 8b69890fb81b05ffe27e42b69e38b121
```

Stage 2 correctly outputs zero (matching R1CS contract), but Stage 5 computes non-zero from memory trace.

### Investigation Needed

1. **Check Jolt's termination write handling**: Does Jolt exclude termination writes from RAF evaluation?
2. **Verify the contract**: What should the RAF polynomial contain vs R1CS RamAddress?
3. **Option A**: Exclude termination writes from Stage 5's RA computation
4. **Option B**: Include termination writes in R1CS RamAddress (if that's what Jolt does)

## Progress This Session

1. ✅ Fixed batched claim tracking for cycle rounds 128-135
2. ✅ All cycle rounds: `current_batched_claim matches expected: true`
3. ✅ Identified root cause: R1CS vs memory trace inconsistency
4. ❌ Stage 5 verification still fails
5. 🔄 Need to investigate Jolt's termination write handling

## Key Debug Output

```
[TRACE] Detected infinite loop at PC 0x80000010, cycle 54
[TRACE] Recorded termination write: addr=0x000000007fffc008, cycle=54
[STAGE5 RAM_RA] Initializing with 1 RAM accesses
[STAGE5 RAM_RA] Access 0: raw_addr=0x7fffc008, remapped_addr=2049, cycle=54
```

The fibonacci program has NO Load/Store instructions but HAS termination writes in memory trace.

## Test Commands

```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Files Modified This Session

- `src/zkvm/spartan/stage5_prover.zig`:
  - Fixed scaled claim initialization
  - Added batched claim recomputation after RA materialization

## SESSION_ENDING

Context is getting long. Key findings saved. Next session should:
1. Investigate how Jolt handles termination writes vs RAF evaluation
2. Check if Stage 5's RA computation should exclude non-Load/Store accesses
