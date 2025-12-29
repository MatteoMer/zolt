# Zolt-Jolt Compatibility Notes

## Current Status (December 29, 2024)

### Session 15 - Investigating 1.23x Discrepancy (Continued)

**Status**: UniSkip passes. Stage 1 output_claim is ~1.23x off from expected.

**Current Values:**
- output_claim = 15155108253109715956971809974428807981154511443156768969051245367813784134214
- expected = 18643585735450861043207165215350408775243828862234148101070816349947522058550
- Ratio: 0.813 (about 1.23x)
- Missing fraction: ~18.7% ≈ 3/16

**Deep Analysis Performed:**

1. **ExpandingTable**: Verified update logic matches Jolt exactly
   - LowToHigh: `values[i] = (1-r)*old`, `values[i+len] = r*old`

2. **GruenSplitEqPolynomial**:
   - initWithScaling matches Jolt's split: m=len/2, E_out for first half, E_in for second
   - bind() formula matches: `eq(τ, r) = 1 - τ - r + 2*τ*r`

3. **Index Structure in Streaming Round**:
   - Jolt: `full_idx = offset + j*klen + k`, `step_idx = full_idx >> 1`, `selector = full_idx & 1`
   - For first streaming window: `klen=1`, `jlen=2`, so `step_idx = i`, `selector = j`
   - Our iteration: directly over cycles `i`, computing both groups

4. **E_active_for_window**: For window_size=1, returns [1] (no-op)

5. **Eq Table Factorization**:
   - For tau_low.len=11: E_out has 32 entries (5 bits), E_in has 32 entries (5 bits)
   - Total: 32*32 = 1024 = padded_trace_len ✓

6. **project_to_first_variable**: For window_size=1, just returns evals[0] or evals[2]

**What's Been Ruled Out:**
- ExpandingTable update formula
- eq table construction (E_out, E_in split)
- bind() formula for current_scalar
- Iteration ranges (0..1023 for 1024 cycles)
- Index mapping (out_idx = i/32, in_idx = i%32)
- switch_over calculation

**Remaining Mystery:**
The ratio 0.813 ≈ 13/16 suggests we're computing 13/16 of the expected sum.
Could indicate:
- Some cycles being skipped?
- Some terms being computed with wrong weights?
- Off-by-one in a subtle place?

**Next Investigation:**
Add debug output to compare per-round values between Zolt and Jolt.

---

## Working Components

| Component | Status | Notes |
|-----------|--------|-------|
| Blake2b Transcript | Working | State and n_rounds match |
| Challenge Derivation | Working | MontU128Challenge-compatible |
| Dory Commitment | Working | GT elements match, MSM correct |
| Proof Structure | Working | 7 stages, claims, all parse |
| Serialization | Working | Byte-level compatible |
| UniSkip Algorithm | Working | Domain sum = 0, claims match |
| Preprocessing Export | Working | Full JoltVerifierPreprocessing |
| DoryVerifierSetup | Working | Precomputed pairings |

---

## Commands

```bash
# Test Zolt (all 632 tests)
zig build test --summary all

# Generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/sum.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Run Jolt debug test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_debug_stage1_verification -- --ignored --nocapture
```
