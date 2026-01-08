# Zolt-Jolt Compatibility - Stage 2 tau_high Mismatch

## Current Status: Stage 2 Verification Fails (tau_high divergence)

### What Works (After evalFromHint fix)
- [x] Stage 1 (SpartanOuter) verification passes
- [x] Stage 2 polynomial coefficients (c0, c2, c3) match Jolt at each round
- [x] Stage 2 challenges match between Zolt and Jolt
- [x] Stage 2 output_claim evolution matches (fixed with evalFromHint)
- [x] fused_left/fused_right match exactly
- [x] All 712 tests pass

### Current Problem
- Stage 2 sumcheck verification fails because expected_output_claim differs
- Root cause: tau_high for Stage 2 differs due to transcript state divergence

**Values:**
- `output_claim = 11948928263400051798463901278432764058724926493141863520413443728531572654384` (MATCHES!)
- `expected_output_claim = 14998460073388315545242452814285195471990034347995786920854240537701021643062` (DIFFERS)
- Zolt tau_high: 55597861199438361161714452967226452302444674035205491421209262082033450074888
- Jolt tau_high: 3964043112274501458186604711136127524303697198496731069976411879372059241338

### Root Cause: Transcript Append Order Mismatch

The transcript state before tau_high sampling differs because opening claims are appended in different order.

**Jolt's order (after Stage 1 completes):**
1. OuterUniSkip verifier calls cache_openings → appends UnivariateSkip@SpartanOuter
2. OuterRemainingSumcheck verifier calls cache_openings → appends 36 R1CS input claims
3. Then samples tau_high for Stage 2

**Zolt's current order:**
1. During Stage 1: appends uni_skip_claim twice (cache_openings + BatchedSumcheck input)
2. Processes sumcheck rounds
3. After Stage 1: appends 36 R1CS input claims (addSpartanOuterOpeningClaimsWithEvaluations)
4. Samples tau_high

The exact sequence and ordering must match Jolt's verification flow precisely.

### Fix Required

1. **Trace Jolt's exact transcript append order:**
   - When does OuterUniSkip.cache_openings run relative to sumcheck?
   - When does OuterRemainingSumcheck.cache_openings run?
   - What's the exact sequence before tau_high sampling?

2. **Fix Zolt's append order to match:**
   - Adjust where/when UnivariateSkip claim is appended
   - Ensure R1CS claims are appended at correct position

3. **Verify tau_high matches after fix**

### Files to Modify
- `src/zkvm/proof_converter.zig` - Fix transcript append order

### Key Insight from Session 21

The evalFromHint fix resolved the claim evolution mismatch:
- Before: Zolt used Lagrange interpolation from combined_evals → wrong next_claim
- After: Zolt uses eval_from_hint formula → next_claim matches Jolt byte-for-byte

The remaining issue is purely about transcript state before tau_high sampling.

## Verification Commands

```bash
# Build and test Zolt
zig build test --summary all

# Generate Jolt-compatible proof
./zig-out/bin/zolt prove --jolt-format -o /tmp/zolt_proof_dory.bin examples/fibonacci.elf

# Verify with Jolt
cd /Users/matteo/projects/jolt/jolt-core
cargo test test_verify_zolt_proof -- --ignored --nocapture
```

## Code Locations
- Proof converter: `/Users/matteo/projects/zolt/src/zkvm/proof_converter.zig`
- ProductVirtual prover: `/Users/matteo/projects/zolt/src/zkvm/spartan/product_remainder.zig`
- Blake2b transcript: `/Users/matteo/projects/zolt/src/transcripts/blake2b.zig`
- Jolt outer.rs (cache_openings): `/Users/matteo/projects/jolt/jolt-core/src/zkvm/spartan/outer.rs`

## Session History
- Session 21: Fixed evalFromHint for claim update, output_claim now matches, identified tau_high divergence
- Session 20: Identified Stage 1 tau mismatch root cause
- Session 19: Fixed Instance 4 endianness bug, all component values match
