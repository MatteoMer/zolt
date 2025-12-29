# Zolt-Jolt Compatibility Notes

## Current Status (December 29, 2024)

### Session 17 - Big-Endian Eq Table Fix

**Status**: Fixed E_out/E_in tables to use big-endian indexing. Ratio changed from 0.8129 to 1.1987.

**Current Values:**
- output_claim = 17544243885955816008056628262847401707989885215135853123958675606975515887014
- expected = 14636075186748817511857284373650752059613754347411376791236690874143105070933
- Ratio: 1.1987 (close to 6/5 = 1.2)

**Key Fix:**
The eq polynomial tables (E_out_vec and E_in_vec) were using little-endian indexing, but Jolt uses big-endian indexing where the MSB corresponds to the first tau element.

Changes:
1. Fixed `initWithScaling` in split_eq.zig to use big-endian table construction
2. Fixed `getFullEqTable` to use big-endian ordering
3. Updated tests to verify big-endian correctness

### Session 16 - Investigation of Previous 0.8129 Ratio

**Previous Values:**
- output_claim = 15155108253109715956971809974428807981154511443156768969051245367813784134214
- expected = 18643585735450861043207165215350408775243828862234148101070816349947522058550
- Ratio: 0.8129 (close to 13/16 = 0.8125)

---

## Verified Components (All Match Jolt)

1. **Constraint Group Ordering**
   - First group: indices {1,2,3,4,5,6,11,14,17,18} (10 constraints)
   - Second group: indices {0,7,8,9,10,12,13,15,16} (9 constraints)
   - Both use same Lagrange weight array `w[0..N]`

2. **Streaming Round Index Structure**
   - Jolt: for each (out_idx, in_idx), for j in 0..2:
     - full_idx = i * 2 + j
     - step_idx = full_idx >> 1 = i (cycle index)
     - selector = full_idx & 1 = j (group selector)
   - Zolt: same logic in streaming iteration

3. **Eq Table Factorization**
   - E_out: 5 bits (32 entries)
   - E_in: 5 bits (32 entries)
   - Total: 32 × 32 = 1024 for 1024 cycles
   - **NOW CORRECT**: Big-endian indexing

4. **Gruen Polynomial Construction**
   - l(X) = eq(0) + (eq(1) - eq(0)) * X
   - q(0) = t'(0), q(∞) = t'(∞)
   - q(1) = (claim - l(0)*q(0)) / l(1)
   - s(X) = l(X) * q(X)

5. **ExpandingTable Update**
   - `values[i] = (1-r)*old`, `values[i+len] = r*old`

---

## Remaining Investigation Areas

The ~1.2 ratio (close to 6/5) suggests a remaining systematic issue:

1. **Gruen polynomial q(X) computation**
   - Verify l(X) is computed correctly
   - Verify q(1) derivation from claim

2. **current_scalar application**
   - Should only be applied in l(X), not in t'(0)/t'(∞)

3. **Linear phase cycle rounds**
   - Verify selector = full_idx & 1 is applied correctly

---

## Big-Endian Eq Polynomial Convention

From jolt-core/src/poly/eq_poly.rs:60-70:
```
evals(r)[i] = eq(r, b₀…b_{n-1})
where i has MSB b₀ and LSB b_{n-1}
```

The index i's bit pattern in BIG-ENDIAN corresponds to the variable values:
- MSB of i → r[0] (first variable)
- LSB of i → r[n-1] (last variable)

### Cycle Index Factorization (Verified Correct)

For cycle i in 0..N:
- out_idx = i >> head_in_bits  (high bits)
- in_idx = i & ((1 << head_in_bits) - 1)  (low bits)
- eq(tau, i) = E_out[out_idx] * E_in[in_idx] * l(w_last, r_stream)

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
| E_out/E_in Tables | Fixed | Now using big-endian indexing |

---

## Commands

```bash
# Test Zolt (all 656 tests)
zig build test --summary all

# Generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/sum.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Run Jolt debug test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_debug_stage1_verification -- --ignored --nocapture
```
