# Zolt-Jolt Compatibility Implementation

## Status: Stage 6 BytecodeReadRaf Phase 1 polynomial structure mismatch

### Recent Fixes (this session)
- [x] Fixed NUM_LOOKUP_TABLES 42→41 (transcript sync issue)
- [x] Fixed challengeScalarPowers to use challengeScalarFull (128-bit) instead of challengeScalar (125-bit)
- [x] Fixed inc_gamma to use challengeScalarFull
- [x] Fixed challengeVector to use challengeScalarFull
- Input claims and batching coefficients now match perfectly between Zolt and Jolt

### Current Issue: BytecodeReadRaf Phase 1 Polynomial Degree Mismatch

The Stage 6 sumcheck fails because the round polynomials diverge from Round 0.

**Root Cause**: BytecodeReadRaf has a two-phase structure:
- Phase 1: Address binding (log_K rounds)
- Phase 2: Cycle binding (log_T rounds)

In **Jolt**, Phase 1 produces **degree 2** polynomials because:
- H(k,j) = ra(k,j) · [Σ_s γ^s · Val_s(k) · eq_s(j)]
- Both `ra(k)` (frequency) and `Val(k)` are linear in the bound address variable
- Their product gives degree 2

In **Zolt**, Phase 1 produces **degree 1** (linear) polynomials because:
- combined[k] = Σ_s γ^s · (Val_s(k) + RAF_s(k)) · F_s[k]
- F_s[k] = Σ_{c:PC(c)=k} eq(r_cycle_s, c) — cycle already pre-summed
- The RA frequency factor is NOT included in Phase 1
- This means Phase 1 is just linear: eval_0 + eval_1 = input_claim

**Fix Required**: Need to restructure BytecodeReadRaf Phase 1 to match Jolt's approach:
1. Keep Val and RA (F_s) as SEPARATE arrays over address domain
2. In Phase 1, compute degree-2 round poly: product of two linear polys
3. After Phase 1 transition, bind Val and RA to get bound_vals and RA chunks
4. Phase 2 then uses bound_vals and product of RA chunk polynomials

### Jolt Phase 1 Details (for reference)
In Jolt's compute_message for Phase 1:
```
for each pair (k_even, k_odd):
    ra_evals[s] = [F_s[k_even], F_s[k_odd]]  // frequency at even/odd
    val_evals[s] = [Val_s(k_even) + RAF_s(k_even), Val_s(k_odd) + RAF_s(k_odd)]
    product_evals[s] = [ra_evals * val_evals pointwise]

eval_at_0 = sum over k_even of product[0]
eval_at_2 = sum over k_even of product[2]  (extrapolated)
eval_at_1 = claim_s - eval_at_0

round_poly = from_evals([eval_at_0, eval_at_1, eval_at_2])
agg_round_poly += gamma^s * round_poly
```

### What's Working
- Stages 1-5 PASS
- Stage 6 input claims and batching coefficients match
- Stage 6 transcript state synchronized

### What Needs Fixing
1. **[IN PROGRESS]** BytecodeReadRaf Phase 1 degree 2 polynomial
2. Stage 7 (HammingWeightClaimReduction) not yet implemented
3. End-to-end verification test

### Test Commands
```bash
cd /home/vivado/projects/zolt && zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --trace-length 64 -o /tmp/zolt_proof.bin --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin
cd /home/vivado/projects/jolt && RAYON_NUM_THREADS=1 cargo run --release --features zolt-debug --manifest-path examples/fibonacci/Cargo.toml -- --verify-zolt-proof /tmp/zolt_proof.bin --zolt-preprocessing /tmp/zolt_preprocessing.bin.ram
```
