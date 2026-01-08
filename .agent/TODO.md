# Zolt-Jolt Compatibility - Stage 2 tau_high Mismatch

## Current Status: Deep Investigation of Transcript Divergence

### What Works (Verified)
- [x] Stage 1 sumcheck proof generation
- [x] Stage 2 polynomial coefficients match Jolt
- [x] Stage 2 challenges match Jolt
- [x] Stage 2 output_claim evolution matches (fixed with evalFromHint)
- [x] fused_left/fused_right match exactly
- [x] uni_skip_claim@SpartanOuter value matches Jolt (15578270688667249954...)
- [x] All 36 R1CS input claim values match Jolt
- [x] All 712 tests pass

### Current Problem: tau_high Sampling Diverges

Despite all claim VALUES matching, the tau_high sampled for Stage 2 differs:
- Zolt tau_high: 55597861199438361161714452967226452302444674035205491421209262082033450074888
- Jolt tau_high: 3964043112274501458186604711136127524303697198496731069976411879372059241338

The transcript state at tau_high sampling differs, causing expected_output_claim mismatch.

### Verified Transcript Sequence

**Jolt Stage 1 verification order:**
1. [UniSkip verify] uni_poly appended
2. [UniSkip verify] r0 sampled
3. [UniSkip verify] cache_openings → append UnivariateSkip claim
4. [BatchedSumcheck] append input_claim (uni_skip_claim)
5. [BatchedSumcheck] batching_coeffs sampled
6. [BatchedSumcheck] process 11 rounds
7. [BatchedSumcheck] cache_openings → append 36 R1CS claims
8. Sample tau_high for Stage 2

**Zolt Stage 1 proof generation order:**
1. uni_poly appended
2. r0 sampled
3. Line 425: append uni_skip_claim (cache_openings equivalent)
4. Line 438: append uni_skip_claim (BatchedSumcheck input)
5. batching_coeff sampled
6. process 11 rounds
7. addSpartanOuterOpeningClaimsWithEvaluations: append 36 R1CS claims
8. Sample tau_high for Stage 2

The sequences appear to match, but the transcript state still diverges.

### Investigation Needed

1. **Compare exact bytes appended** at each step between Zolt and Jolt
2. **Check if there's an extra/missing append** somewhere
3. **Verify round polynomial encoding** is byte-identical
4. **Check if batching_coeff sampling** produces the same value

### Possible Root Causes

1. Different byte encoding of claims (big endian vs little endian)
2. Extra or missing transcript appends
3. Different round polynomial coefficient encoding
4. Different batching coefficient computation

### Files to Investigate
- `src/zkvm/proof_converter.zig` - Stage 1 proof generation
- `src/transcripts/blake2b.zig` - Transcript implementation
- Jolt's `verifier.rs` - Stage 1 verification

### Test Commands
```bash
# Build and test Zolt
zig build test --summary all

# Generate proof with debug output
./zig-out/bin/zolt prove --jolt-format -o /tmp/zolt_proof_dory.bin examples/fibonacci.elf 2>&1 | grep -E "appendBytes|STAGE1|OPENING_CLAIMS"

# Verify with Jolt
cd /Users/matteo/projects/jolt/jolt-core
cargo test test_verify_zolt_proof -- --ignored --nocapture 2>&1 | grep -E "append_virtual|STAGE"
```

## Technical Summary

### Key Values That Match
- uni_skip_claim@SpartanOuter: 15578270688667249954692364540555337347090181127244999411735720905215105446756
- LeftInstructionInput: 6149008884082944649395010520634152975755517950284410925647529691383783473665
- RightInstructionInput: 5305691460212091458894976248163812456721981669912385268553172458466588619558
- Product: 18665406647812718617781198690494953139719373408826706814854637365498608866768
- (all 36 R1CS claims match)

### Values That Differ
- Transcript state before tau_high sampling
- tau_high value itself
- expected_output_claim (computed from tau_high)

## Session History
- Session 21: Fixed evalFromHint, identified tau_high divergence, verified all claim values match
- Session 20: Identified Stage 1 tau mismatch root cause
- Session 19: Fixed Instance 4 endianness bug
