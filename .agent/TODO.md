# Zolt-Jolt Compatibility - Current Progress

## Current Status: Stage 3 Prefix-Suffix Optimization Required

### Problem Identified (Session 32)

Stage 3 verification fails because **Zolt and Jolt produce different (but both valid) round polynomials**:

- Both satisfy `p(0) + p(1) = previous_claim` ✓
- But coefficients differ due to different computation approaches
- Different coefficients → different Fiat-Shamir challenges → verification failure

### Root Cause

Jolt uses **prefix-suffix sumcheck optimization** in Stage 3:
1. Split eq+1(r, x) into prefix and suffix components
2. Use P/Q buffers to efficiently compute round polynomials
3. Transition from Phase1 (prefix-suffix) to Phase2 (direct) after n/2 rounds

Zolt uses **naive direct computation**:
- Iterates over all indices computing sumcheck directly
- Produces valid but different round polynomials

### Evidence

Round 0 coefficient comparison:
- Zolt c0: 15162749667655265946555954462559066615162111224393573091137614644230810640633
- Jolt c0: 7091024619817638108434831413024896929049773929476085946006737149609972313435
- **Completely different!**

But both satisfy sumcheck property:
- Zolt: shift_p0 + shift_p1 = shift_claim ✓
- Zolt: combined_eval0 + combined_eval1 = current_claim ✓

### Implementation Plan

#### Completed ✓
1. Added `EqPlusOnePrefixSuffixPoly` struct to poly/mod.zig
   - Decomposes eq+1 into prefix_0, suffix_0, prefix_1, suffix_1
   - Computes all evaluations at initialization

#### In Progress
2. Refactor Stage 3 prover to use prefix-suffix optimization
   - Need to implement P/Q buffer initialization
   - Need to implement Phase1 round polynomial computation
   - Need to implement Phase1 binding (halve P and Q buffers)
   - Need to implement Phase2 transition when prefix_size == 2

#### Remaining Work
3. Implement Phase2 materialization
   - Evaluate prefix polynomials at accumulated challenges
   - Construct full eq+1 polynomial for remaining rounds
   - Run standard sumcheck for remaining variables

4. Test round polynomial compatibility
   - Verify c0, c2, c3 coefficients match Jolt's output
   - Verify challenges match
   - Verify final output claim matches

### Technical Details

#### Phase1 Round Polynomial Formula

For each pair (P, Q):
```
g(0) = Σ_i P[2*i] * Q[2*i]
g(1) = Σ_i P[2*i+1] * Q[2*i+1]
```

Then compress using `from_evals_and_hint(previous_claim, [g(0), g(1)])`.

#### Phase1 Binding

After receiving challenge r_j:
```
P_new[i] = P[2*i] + r_j * (P[2*i+1] - P[2*i])
Q_new[i] = Q[2*i] + r_j * (Q[2*i+1] - Q[2*i])
```

#### Phase2 Transition

Occurs when `prefix_size == 2` (log2 of size == 1).
Evaluate prefix polynomials at sumcheck challenges, then materialize full eq+1.

### Files to Modify

- `src/poly/mod.zig` - EqPlusOnePrefixSuffixPoly ✓
- `src/zkvm/spartan/stage3_prover.zig` - Main prover refactor (major)

### Testing Commands

```bash
# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format \
  --export-preprocessing /tmp/zolt_preprocessing.bin \
  -o /tmp/zolt_proof_dory.bin --srs /tmp/jolt_dory_srs.bin

# Verify with Jolt
cd /Users/matteo/projects/jolt && cargo test -p jolt-core \
  test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture

# Run Zolt tests
zig build test
```

### Success Criteria

1. Stage 3 round 0 c0, c2, c3 coefficients match Jolt's
2. All 10 rounds produce matching polynomials
3. Final output_claim matches expected_output_claim
4. Full proof verification passes

---

## Overall Progress

| Stage | Status | Notes |
|-------|--------|-------|
| 1 | ✓ PASSES | Outer Spartan sumcheck works |
| 2 | ✓ PASSES | Product virtualization works |
| 3 | ✗ FAILS | Prefix-suffix optimization needed |
| 4-7 | Blocked | Waiting on Stage 3 |

## Tests Status

- `zig build test`: All 578+ tests pass
- Jolt verification with Zolt preprocessing: Stage 3 fails
