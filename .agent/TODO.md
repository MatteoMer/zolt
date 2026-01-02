# Zolt-Jolt Compatibility TODO

## Current Status: Session 35 - January 2, 2026

**All 710 Zolt tests pass**

### Completed This Session

1. **MultiquadraticPolynomial.bind()** - Added Jolt-compatible quadratic interpolation
   - Formula: `f(r) = f(0)*(1-r) + f(1)*r + f(∞)*r(r-1)`
   - Reference: `jolt-core/src/poly/multiquadratic_poly.rs:bind_first_variable`

2. **GruenSplitEqPolynomial.getEActiveForWindow()** - Added E_active projection
   - Computes eq table over active window bits (window_size - 1 bits)
   - Reference: `jolt-core/src/poly/split_eq_poly.rs:E_active_for_window`

3. **t_prime_poly integration** - Added to StreamingOuterProver
   - `buildTPrimePoly()` - Creates multiquadratic from bound Az/Bz
   - `rebuildTPrimePoly()` - nextWindow equivalent for rebuilding
   - `computeTEvals()` - Projects t_prime using E_active weights
   - **Bind t_prime_poly after each linear round** (CRITICAL fix!)

---

## Verified Architecture Match

### Jolt's ingest_challenge (OuterLinearStage)
```rust
fn ingest_challenge(&mut self, shared: &mut Self::Shared, r_j: F::Challenge, _round: usize) {
    shared.split_eq_poly.bind(r_j);
    if let Some(t_prime_poly) = shared.t_prime_poly.as_mut() {
        t_prime_poly.bind(r_j, BindingOrder::LowToHigh);  // <-- CRITICAL
    }
    rayon::join(
        || self.az.bind_parallel(r_j, BindingOrder::LowToHigh),
        || self.bz.bind_parallel(r_j, BindingOrder::LowToHigh),
    );
}
```

### Zolt's bindRemainingRoundChallenge (now matches!)
```zig
pub fn bindRemainingRoundChallenge(self: *Self, r: F) !void {
    // ... streaming/linear phase handling ...

    try self.challenges.append(self.allocator, r);
    self.split_eq.bind(r);

    // Bind t_prime_poly (CRITICAL: matches Jolt's ingest_challenge)
    if (self.t_prime_poly) |*t_prime| {
        t_prime.bind(r);
    }

    self.current_round += 1;
}
```

---

## Next Steps

### Phase 3: End-to-End Verification

1. **Test proof generation with t_prime binding**
   ```bash
   zig build -Doptimize=ReleaseFast && ./zig-out/bin/zolt prove examples/fibonacci.elf \
     --jolt-format -o /tmp/zolt_proof_dory.bin
   ```

2. **Run Jolt verification**
   ```bash
   cd /Users/matteo/projects/jolt
   cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
   ```

3. **Debug sumcheck intermediate values** (if needed)
   - Compare t_prime_poly values after each bind
   - Compare (t_zero, t_infinity) projections
   - Compare round polynomial coefficients

---

## Implementation Checklist

### Phase 1: Primitives [COMPLETE]
- [x] Add `MultiquadraticPolynomial.bind()` method
- [x] Add unit tests for bind()
- [x] Add `GruenSplitEqPolynomial.getEActiveForWindow()`
- [x] Add unit tests for getEActiveForWindow()

### Phase 2: t_prime Integration [COMPLETE]
- [x] Add t_prime_poly field to StreamingOuterProver
- [x] Add buildTPrimePoly() for initial materialization
- [x] Add rebuildTPrimePoly() (nextWindow equivalent)
- [x] Add computeTEvals() for projection
- [x] Bind t_prime_poly in bindRemainingRoundChallenge
- [x] Update computeRemainingRoundPoly to use t_prime in linear phase

### Phase 3: Verification [IN PROGRESS]
- [ ] Generate proof with new t_prime binding
- [ ] Run Jolt verification test
- [ ] Compare intermediate values if verification fails

---

## Verified Correct Components

### Transcript
- [x] Blake2b transcript format matches Jolt
- [x] Challenge scalar computation (128-bit, no masking)
- [x] Field serialization (Arkworks LE format)

### Polynomial Computation
- [x] Gruen cubic polynomial formula
- [x] Split eq polynomial factorization (E_out/E_in)
- [x] bind() operation (eq factor computation)
- [x] Lagrange interpolation
- [x] evalsToCompressed format
- [x] DensePolynomial.bindLow() matches Jolt's bound_poly_var_bot
- [x] MultiquadraticPolynomial.bind() matches Jolt
- [x] GruenSplitEqPolynomial.getEActiveForWindow() matches Jolt

### RISC-V & R1CS
- [x] R1CS constraint definitions (19 constraints, 2 groups)
- [x] Constraint 8 (RightLookupSub) has 2^64 constant
- [x] UniSkip polynomial generation
- [x] Memory layout constants match Jolt
- [x] R1CS input ordering matches Jolt's ALL_R1CS_INPUTS

### Supporting Structures
- [x] ExpandingTable implementation
- [x] GruenSplitEqPolynomial (complete with E_active)
- [x] MultiquadraticPolynomial (complete with bind)
- [x] t_prime_poly binding in linear phase

### All Tests Pass
- [x] 710/710 Zolt tests pass

---

## Key Files Modified This Session

| File | Changes |
|------|---------|
| `src/poly/multiquadratic.zig` | Added `bind()`, `isBound()`, `finalSumcheckClaim()` |
| `src/poly/split_eq.zig` | Added `getEActiveForWindow()` |
| `src/zkvm/spartan/streaming_outer.zig` | Added t_prime_poly field, buildTPrimePoly, rebuildTPrimePoly, computeTEvals, bind in ingestChallenge |

---

## Test Commands

```bash
# Run Zolt tests
zig build test --summary all

# Generate proof
zig build -Doptimize=ReleaseFast && ./zig-out/bin/zolt prove examples/fibonacci.elf \
  --jolt-format -o /tmp/zolt_proof_dory.bin

# Jolt verification tests
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```
