# Deep Code Audit: Zolt vs Jolt Stage 4

## Summary of Audit

Performed line-by-line comparison of Zolt's `stage4_gruen_prover.zig` and Jolt's `read_write_checking.rs` to identify the source of coefficient mismatch.

## Key Findings

### 1. Formula Correctness ✅

**Jolt (RegistersCycleMajorEntry::compute_evals, lines 306-314 in registers.rs):**
```rust
let ra_evals = [even.ra_coeff, odd.ra_coeff - even.ra_coeff];
let wa_evals = [even.wa_coeff, odd.wa_coeff - even.wa_coeff];
let val_evals = [even.val_coeff, odd.val_coeff - even.val_coeff];
[
    ra_evals[0] * val_evals[0] + wa_evals[0] * (val_evals[0] + inc_evals[0]),
    ra_evals[1] * val_evals[1] + wa_evals[1] * (val_evals[1] + inc_evals[1]),
]
```

**Zolt (stage4_gruen_prover.zig, lines 542-548):**
```zig
const ra_slope = ra_odd.sub(ra_even);
const wa_slope = wa_odd.sub(wa_even);
const val_slope = val_odd.sub(val_even);
const c_0 = ra_even.mul(val_even).add(wa_even.mul(val_even.add(inc_0)));
const c_X2 = ra_slope.mul(val_slope).add(wa_slope.mul(val_slope.add(inc_slope)));
```

**Result**: Formulas are IDENTICAL. ✅

### 2. val_poly Semantics ✅

Both implementations store the value BEFORE each cycle:

**Jolt (registers.rs, line 146-147):**
```rust
// val_coeff stores the value *before* any access at this cycle.
val_coeff: F::from_u64(rd_pre_val),
```

**Zolt (stage4_gruen_prover.zig, lines 166-168):**
```zig
// Set val(k, j) for all registers - value BEFORE this cycle
for (0..32) |k| {
    val_poly[k * T + cycle] = F.fromU64(register_values[k]);
}
```

**Result**: Semantics match. ✅

### 3. x_in/x_out Computation ✅

**Jolt (read_write_checking.rs, lines 294, 302, 308):**
```rust
let x_out = (entries[0].row / 2) >> num_x_in_bits;
let j_prime = 2 * (entries[0].row / 2);
let x_in = (j_prime / 2) & x_bitmask;
```

**Zolt (stage4_gruen_prover.zig, lines 507, 511-512):**
```zig
const j_prime = 2 * i; // i iterates from 0 to half_T-1
const x_in = if (num_x_in_bits > 0) (i & x_bitmask) else 0;
const x_out = if (num_x_in_bits < 64) (i >> @as(u6, @intCast(num_x_in_bits))) else 0;
```

For cycle pair (j_even, j_odd) where j_even = 2*i:
- Jolt: `x_in = (row/2) & x_bitmask = i & x_bitmask`
- Zolt: `x_in = i & x_bitmask`

**Result**: Computations match. ✅

### 4. evalsCached Implementation ✅

**Jolt (eq_poly.rs, lines 174-181):**
```rust
for j in 0..r.len() {
    size *= 2;
    for i in (0..size).rev().step_by(2) {
        let scalar = evals[j][i / 2];
        evals[j + 1][i] = scalar * r[j];
        evals[j + 1][i - 1] = scalar - evals[j + 1][i];
    }
}
```

**Zolt (gruen_eq.zig, lines 245-256):**
```zig
var i: usize = new_len - 1;
while (i > 0) : (i -= 2) {
    const scalar = prev[i / 2];
    curr[i] = scalar.mul(w[k]);
    curr[i - 1] = scalar.sub(curr[i]);
    if (i < 2) break;
}
```

**Result**: Implementations are identical. ✅

### 5. Sparse vs Dense Representation

**THIS IS A KEY DIFFERENCE:**

**Jolt:**
- Uses sparse matrix representation
- Only stores entries for registers that are actually accessed
- Handles missing entries specially in `compute_evals` (lines 316-341 in registers.rs)
- When even exists but odd doesn't: uses `ra_evals = [even.ra_coeff, -even.ra_coeff]`
- When odd exists but even doesn't: uses `ra_evals = [0, odd.ra_coeff]`

**Zolt:**
- Uses dense representation
- Iterates over ALL registers (line 525: `for (0..self.current_K)`)
- Registers not accessed have ra=0, wa=0, val=<last known value>
- When register accessed at even but not odd: `ra_slope = 0 - gamma = -gamma` (correct!)
- When register not accessed at all: `ra_slope = 0 - 0 = 0` (correct!)

**Analysis**: The dense approach should be mathematically equivalent IF all polynomial values are correctly maintained. ✅

### 6. Binding Order - **CRITICAL DIFFERENCE!** ⚠️

**Jolt (read_write_checking.rs, line 226-228):**
```rust
Some(GruenSplitEqPolynomial::new(
    &r_prime.r,
    BindingOrder::LowToHigh,  // ← Binds from LSB to MSB
))
```

**Zolt (stage4_gruen_prover.zig, lines 244-255):**
```zig
// Convert r_cycle from LE (round order) to BE (MSB first) for GruenSplitEqPolynomial
// Stage 3 challenges are in round order: r_cycle[0] = round 0 challenge (binds LSB)
// GruenSplitEqPolynomial expects big-endian: w[0] = MSB
const r_cycle_be = try allocator.alloc(F, r_cycle.len);
for (0..r_cycle.len) |i| {
    r_cycle_be[i] = r_cycle[r_cycle.len - 1 - i];  // ← REVERSES the array!
}
const gruen_eq_poly = try GruenSplitEqPolynomial(F).init(allocator, r_cycle_be);
```

**Jolt's BindingOrder::LowToHigh** (split_eq_poly.rs, lines 89-98):
- `w = [w_out, w_in, w_last]` where `w_out = w[0..m]`, `w_in = w[m..n-1]`, `w_last = w[n-1]`
- Binds starting from `w[n-1]` (w_last) down to `w[0]`
- current_index starts at n and decrements to 0

**Zolt's Implementation:**
- Reverses r_cycle array to big-endian
- GruenSplitEqPolynomial.bind() decrements current_index from n to 0
- Binds from `w[n-1]` down to `w[0]`

**Question**: Are these equivalent given the array reversal?

### 7. OpeningPoint Endianness

**Jolt (opening_proof.rs, line 29-30):**
```rust
pub const BIG_ENDIAN: Endianness = false;
pub const LITTLE_ENDIAN: Endianness = true;
```

**Jolt's r_cycle type** (read_write_checking.rs, line 66):
```rust
pub r_cycle: OpeningPoint<BIG_ENDIAN, F>,
```

All openings in Jolt use `OpeningPoint<BIG_ENDIAN, F>`.

### 8. Missing Entry Handling - **POTENTIAL BUG?** ⚠️

When Jolt has an entry at even cycle but NOT at odd cycle (registers.rs, lines 316-327):
```rust
(Some(even), None) => {
    let odd_val_coeff = F::from_u64(even.next_val);
    let ra_evals = [even.ra_coeff, -even.ra_coeff];
    let wa_evals = [even.wa_coeff, -even.wa_coeff];
    let val_evals = [even.val_coeff, odd_val_coeff - even.val_coeff];
    // Returns [q(0), q_X2_coeff]
}
```

The key is `odd_val_coeff = even.next_val` - this is the value AFTER the even cycle.

In Zolt's dense representation, when register k is accessed at even cycle j but not at odd cycle j+1:
- `val_poly[k*T + j]` = value BEFORE cycle j
- `val_poly[k*T + (j+1)]` = value BEFORE cycle j+1 = value AFTER cycle j = `even.next_val`

So Zolt's `val_odd` should equal Jolt's `even.next_val`. ✅

## Conclusion

All core computations appear to match between Zolt and Jolt:
- ✅ Polynomial formulas
- ✅ val_poly semantics
- ✅ x_in/x_out indexing
- ✅ evalsCached table building
- ✅ Sparse vs dense equivalence
- ✅ Missing entry handling

**REMAINING QUESTION:** Binding order and array reversal
- Zolt reverses r_cycle to big-endian before passing to GruenSplitEqPolynomial
- Jolt uses BindingOrder::LowToHigh with non-reversed r_cycle
- Are these equivalent? Need to trace through binding logic carefully.

### 9. gruenPolyDeg3 Implementation ✅

**Jolt (split_eq_poly.rs, lines 422-450):**
```rust
let eq_eval_1 = self.current_scalar * self.w[self.current_index - 1]; // for LowToHigh
let eq_eval_0 = self.current_scalar - eq_eval_1;
let eq_m = eq_eval_1 - eq_eval_0;
let eq_eval_2 = eq_eval_1 + eq_m;
let eq_eval_3 = eq_eval_2 + eq_m;
...
let quadratic_eval_1 = cubic_eval_1 / eq_eval_1;
let quadratic_eval_2 = quadratic_eval_1 + quadratic_eval_1 - quadratic_eval_0 + e_times_2;
let quadratic_eval_3 = quadratic_eval_2 + quadratic_eval_1 - quadratic_eval_0 + e_times_2 + e_times_2;
UniPoly::from_evals(&[cubic_eval_0, cubic_eval_1, eq_eval_2 * quadratic_eval_2, eq_eval_3 * quadratic_eval_3])
```

**Zolt (gruen_eq.zig, lines 158-188):**
```zig
const eq_eval_1 = self.current_scalar.mul(self.w[self.current_index - 1]);
const eq_eval_0 = self.current_scalar.sub(eq_eval_1);
const eq_m = eq_eval_1.sub(eq_eval_0);
const eq_eval_2 = eq_eval_1.add(eq_m);
const eq_eval_3 = eq_eval_2.add(eq_m);
...
const quadratic_eval_1 = cubic_eval_1.mul(eq_eval_1.inverse() orelse F.one());
const quadratic_eval_2 = quadratic_eval_1.add(quadratic_eval_1).sub(quadratic_eval_0).add(e_times_2);
const quadratic_eval_3 = quadratic_eval_2.add(quadratic_eval_1).sub(quadratic_eval_0).add(e_times_2).add(e_times_2);
return fromEvals(.{ cubic_eval_0, cubic_eval_1, cubic_eval_2, cubic_eval_3 });
```

**Result**: Implementations are IDENTICAL. ✅

## Summary of Audit Results

✅ **All core implementations match:**
1. Polynomial formulas (c_0, c_X2 computation)
2. val_poly semantics (value before cycle)
3. x_in/x_out indexing
4. evalsCached table building
5. Sparse vs dense equivalence
6. Missing entry handling
7. gruenPolyDeg3 conversion
8. Coefficient to evaluation conversion (fromEvals)

⚠️ **Potential issues identified:**
1. Binding order and array reversal - needs verification
2. Cannot directly compare E_out/E_in tables (different programs)

## Recommendation

**Since all implementations match, the bug is likely very subtle.**

**Option A**: Add extensive logging to both implementations
- Log q_0, q_X2 for first 3 contributions
- Log E_combined for each contribution
- Compare accumulation step-by-step

**Option B**: Create a minimal test case
- 2-3 cycle trace with known register values
- Manually compute expected q_0, q_X2
- Verify both implementations produce correct results

**Option C**: Binary search for divergence point
- Check if Round 0 polynomial is correct (p(0) + p(1) = claim)
- Check if coefficients are just in wrong format/order
- Check if there's a sign error or field arithmetic bug

**Most Likely Issue**: Given that all formulas match, the issue is probably:
1. A subtle bug in how contributions are accumulated (order, doubling, etc.)
2. An error in coordinate computation that only affects certain cases
3. A field arithmetic precision issue (inverse, multiplication order)
4. The dense iteration including extra zero contributions that shouldn't be there
