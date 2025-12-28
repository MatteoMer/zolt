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

---

## ROOT CAUSE IDENTIFIED ❌

### The Problem: Multiquadratic Polynomial Representation

Jolt's streaming outer sumcheck uses a **MultiquadraticPolynomial** representation:
- Stores evaluations at `{0, 1, INFINITY}` for each variable
- Uses `project_to_first_variable(E_active, 0)` for t'(0)
- Uses `project_to_first_variable(E_active, INFINITY)` for t'(∞)

Zolt's current implementation:
- Sums over cycles with binary 0/1 partitioning
- Computes `t_zero` (first half) and `t_one` (second half)
- `t_infinity = t_one - t_zero` (LINEAR slope, NOT quadratic coefficient!)

### Key Insight

The Gruen method's `gruen_poly_deg_3(q_constant, q_quadratic_coeff, ...)` expects:
- `q_constant = q(0)` - the constant term
- `q_quadratic_coeff = e` - the coefficient of X² in q(X) = c + dX + eX²

But I was passing:
- `q_constant = t_zero` ✓ (correct)
- `q_quadratic_coeff = t_one - t_zero` ✗ (this is the linear slope, not quadratic coeff!)

### What Jolt Does

Jolt uses a `MultiquadraticPolynomial` that:
1. Expands each variable from binary {0,1} to ternary {0,1,∞}
2. The ∞ (INFINITY=2) index stores the quadratic coefficient
3. `project_to_first_variable` sums over the prefix eq table scaled by the polynomial evaluations

This is implemented in:
- `jolt-core/src/poly/multiquadratic_poly.rs`
- Used by `OuterSharedState::compute_evaluation_grid_from_trace()`

### Required Fix

To make Zolt produce compatible proofs, I need to:

1. **Implement MultiquadraticPolynomial in Zolt**
   - Store evaluations at {0, 1, ∞} for each variable
   - Implement `expand_linear_grid_to_multiquadratic()`
   - Implement `project_to_first_variable()`

2. **Update StreamingOuterProver**
   - Use multiquadratic representation for Az*Bz products
   - Compute proper t'(∞) as quadratic coefficient extraction

### Complexity

This is a significant algorithmic change. The multiquadratic expansion involves:
- Converting binary evaluations to ternary
- Polynomial interpolation at {0, 1, ∞}
- Efficient folding during binding

---

## Current Values (for debugging)

From latest test run:
```
output_claim (from sumcheck):    11612374852220731197013232400393975162132149637091984341606359412226379830051
expected_output_claim (from R1CS): 18745955558119577451624936825732812500259023178565119442902704003954906526404

expected = tau_high_bound_r0 * tau_bound_r_tail * inner_sum_prod
         = 5082598541187396806031046967159366823594641154848460582581091712291471884094
         * 16153740132551411969570181916217515401545621836647993763662759575768152882318
         * 8911191101246844644469009006381839599717758671761069483099516096600528609494
```

---

## Progress Indicators

- [x] UniSkip verification passes
- [x] Stage 1 sumcheck equations pass (p(0)+p(1)=claim)
- [x] R1CS input evaluations computed
- [ ] **MultiquadraticPolynomial implementation** ← BLOCKING
- [ ] Stage 1 expected_output_claim matches
- [ ] Stages 2-7 verify
- [ ] Full proof verification passes

---

## Test Commands

```bash
# Zolt tests
zig build test --summary all

# Generate proof
./zig-out/bin/zolt prove examples/sum.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Jolt verification
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
cargo test --package jolt-core test_debug_stage1_verification -- --ignored --nocapture
```

---

## Next Steps

1. Study Jolt's `MultiquadraticPolynomial` implementation
2. Port `expand_linear_grid_to_multiquadratic()` to Zig
3. Update `StreamingOuterProver::computeRemainingRoundPoly()` to use multiquadratic
4. Re-test verification
