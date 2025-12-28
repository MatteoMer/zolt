# Zolt-Jolt Compatibility TODO

## Completed ✅

1. **Phase 1: Transcript Compatibility** - Blake2b transcript matches Jolt
2. **Phase 2: Proof Structure Refactoring** - 7-stage proof with UniSkip
3. **Phase 3: Serialization Alignment** - Arkworks-compatible serialization
4. **Phase 4: Commitment Scheme** - Dory with Jolt-compatible SRS
5. **Phase 5: Verifier Preprocessing Export** - DoryVerifierSetup exports correctly
6. **Fix Lagrange Interpolation Bug** - Dead code was corrupting basis array
7. **Stage 1 UniSkip Verification** - Domain sum check passes
8. **UnivariateSkip Claim** - Now correctly set to uni_poly.evaluate(r0)
9. **Montgomery Form Fix** - appendScalar now converts from Montgomery form
10. **MontU128Challenge Compatibility** - Challenge scalars now match Jolt's format
11. **Symmetric Lagrange Domain** - Fixed to use {-4,...,5} matching Jolt
12. **Streaming Round Logic** - Separate handling for constraint group selection
13. **MultiquadraticPolynomial** - Already implemented in src/poly/multiquadratic.zig

---

## ROOT CAUSE IDENTIFIED ❌

### Problem: Az*Bz Product Computation

The streaming outer sumcheck needs to compute `t'(0)` and `t'(∞)` where:
- `t'(0)` = sum over boolean points with current var = 0
- `t'(∞)` = the **quadratic coefficient** (not linear slope!)

Jolt computes this correctly by:
1. Computing `grid_a` = Az evaluations separately
2. Computing `grid_b` = Bz evaluations separately
3. Expanding each to multiquadratic (`expand_linear_grid_to_multiquadratic`)
4. Multiplying pointwise: `buff_a[i] * buff_b[i]`
5. Summing with eq weights to get `t'(0)` and `t'(∞)`

Zolt's current approach:
- Computes `az_bz = Az * Bz` directly as a single product
- This loses the structure needed for multiquadratic expansion
- Results in `t_infinity = t_one - t_zero` (wrong - this is linear slope!)

### Required Changes

To fix `StreamingOuterProver::computeRemainingRoundPoly()`:

```zig
// Current (WRONG):
const az_bz = self.computeCycleAzBzProduct(...);
t_zero += eq_val * az_bz;
// ...
t_infinity = t_one - t_zero;  // LINEAR slope, not quadratic coeff!

// Should be:
// 1. Compute grid_az and grid_bz separately
// 2. Expand each to multiquadratic
// 3. Multiply pointwise
// 4. Project to get (t_zero, t_infinity)
```

This is a significant refactoring because:
- Need to store Az and Bz grids separately (2 * 3^window_size elements)
- Need to implement the window-based streaming approach
- The `extrapolate_from_binary_grid_to_tertiary_grid` logic is complex

### Current Values

```
output_claim (from sumcheck):     11612374852220731197013232400393975162132149637091984341606359412226379830051
expected_output_claim (from R1CS): 18745955558119577451624936825732812500259023178565119442902704003954906526404
```

---

## Progress Indicators

- [x] UniSkip verification passes
- [x] Stage 1 sumcheck equations pass (p(0)+p(1)=claim)
- [x] R1CS input evaluations computed
- [x] MultiquadraticPolynomial implemented
- [ ] **Az/Bz grid separation** ← BLOCKING
- [ ] **Multiquadratic product computation** ← BLOCKING
- [ ] Stage 1 expected_output_claim matches
- [ ] Stages 2-7 verify
- [ ] Full proof verification passes

---

## Files to Modify

1. `src/zkvm/spartan/streaming_outer.zig`
   - Add `computeCycleAzGrid()` and `computeCycleBzGrid()`
   - Use `MultiquadraticPolynomial` for expansion
   - Compute product pointwise
   - Use `projectToFirstVariable()` for correct t'(∞)

2. `src/poly/multiquadratic.zig`
   - May need additional helper functions

---

## Test Commands

```bash
zig build test --summary all
./zig-out/bin/zolt prove examples/sum.elf --jolt-format -o /tmp/zolt_proof_dory.bin
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_debug_stage1_verification -- --ignored --nocapture
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```
