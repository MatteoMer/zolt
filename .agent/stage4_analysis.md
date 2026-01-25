# Stage 4 Coefficient Mismatch - Analysis

## Current Status

**Cross-Verification Result:** Stage 4 FAILS
- Stages 1-3: ✅ PASS
- Stage 4: ❌ FAIL (Sumcheck verification fails)

## Root Cause

Zolt's Stage 4 Round 0 polynomial coefficients differ from what Jolt expects:

**Zolt Round 0 coefficients (from proof):**
```
coeffs[0] = [167, 2, 138, 133, 197, 33, 242, 168, ...]  (c0 in LE format)
coeffs[1] = [213, 211, 179, 236, 114, 215, 98, 219, ...]  (c2 in LE format)
coeffs[2] = [28, 7, 243, 236, 36, 154, 133, 63, ...]      (c3 in LE format)
```

**Effect:**
- Different coefficients → different Fiat-Shamir challenges → different r_cycle
- Jolt verification derives r_cycle from coefficients, gets different value than params.r_cycle
- This causes eq(r_cycle_expected, r_cycle_actual) ≠ 1 → verification failure

## Zolt Stage 4 Computation Details

### Inputs (verified to match):
- `gamma = { 165, 220, 86, 216, 147, 169, 75, 108 }`
- `T = 256` (trace length)
- `K = 128` (register count)
- `previous_claim` (batched claim from Stages 1-3)

### Intermediate Values (Round 0):
```
q_0  = { 142, 134, 24, 23, 198, 184, 119, 182, ... }
q_X2 = { 142, 219, 181, 75, 210, 72, 66, 115, ... }
```

### E_out/E_in Tables:
```
E_out.len = 16, E_in.len = 8
E_out[0] = { 71, 160, 211, 72, 66, 155, 28, 51, ... }
E_in[0]  = { 212, 155, 232, 122, 38, 129, 172, 11, ... }
```

### Sample Contribution (k=2, j_pair=(0,1)):
```
EVEN: ra=0, wa=1, val=0
ODD:  ra=gamma, wa=1, val=32768
inc_0 = 32768
c_0 = 32768
c_X2 = { 0, 128, 82, 110, 43, 236, 201, 212 }
E_combined = { 208, 239, 137, 209, 201, 127, 209, 238 }
```

## Comparison Challenge

We cannot directly compare Jolt's fibonacci-guest coefficients because:
- Jolt's `fib_e2e_dory` test uses `fibonacci-guest` with input `100u32`
- Zolt uses bare-metal `fibonacci.elf` that computes `fib(10)`
- Different programs → different traces → different coefficients

**Jolt fibonacci-guest Round 0 coefficients (for reference only):**
```
coeffs[0] = [31, ed, 7b, 49, 20, fc, 6c, b6, ...]
coeffs[1] = [54, e9, 89, d3, bc, 46, 77, 0f, ...]
coeffs[2] = [4a, 84, 37, 71, 84, 25, c3, e4, ...]
```

## Hypotheses for Mismatch

1. **E_out/E_in table computation**: Values might differ from Jolt's GruenSplitEqPolynomial
2. **Sparse vs Dense iteration**: Jolt uses sparse matrix, Zolt uses dense representation
3. **Value polynomial timing**: val_poly should have value BEFORE cycle, might be off
4. **Inc polynomial computation**: inc_poly might not match Jolt's expectations
5. **Accumulation order**: Order of accumulation might affect final q_0/q_X2

## Next Steps to Debug

### Option 1: Add Matching Jolt Test
Create a Jolt test that proves Zolt's `fibonacci.elf` using Jolt's prover:
- This would give us direct coefficient comparison
- Requires adapting Jolt's framework to load bare-metal ELF

### Option 2: Manual Verification
Manually compute expected q_0/q_X2 for first few contributions:
- Take E_out[0], E_in[0] values from Zolt
- Take polynomial values (ra, wa, val) from Zolt
- Manually compute what Jolt's formulas would produce
- Compare with Zolt's c_0, c_X2 for first contribution

### Option 3: Deep Code Audit
Compare Zolt's `stage4_gruen_prover.zig` line-by-line with Jolt's `read_write_checking.rs`:
- Focus on `phase1ComputeMessage` (Zolt) vs `phase1_compute_message` (Jolt)
- Check GruenSplitEqPolynomial implementation
- Verify polynomial formulas match exactly

### Option 4: Minimal Test Case
Create a 2-cycle fibonacci test with known expected values:
- Manually compute all expected polynomials
- Run through both Zolt and Jolt
- Identify exact point of divergence

## Recommended Approach

**PRIORITY: Option 2 - Manual Verification**

For the first contribution (k=2, j_pair=(0,1)):
1. Take known inputs: ra_even=0, wa_even=1, val_even=0, ra_odd=gamma, wa_odd=1, val_odd=32768
2. Compute: c_0 = ra_even*val_even + wa_even*(val_even+inc_0) = 0*0 + 1*(0+32768) = 32768 ✓
3. Compute: c_X2 = ra_slope*val_slope + wa_slope*(val_slope+inc_slope)
4. Verify E_combined = E_out[x_out] * E_in[x_in] matches expected
5. Check if accumulated q_0 += E_combined * c_0 matches Zolt's output

If manual verification shows Zolt's computation is correct for individual contributions,
the issue might be in:
- E_out/E_in table initialization
- GruenSplitEqPolynomial.bind() updates
- Coefficient to evaluation conversion (gruenPolyDeg3)

## Files of Interest

**Zolt:**
- `src/zkvm/spartan/stage4_gruen_prover.zig:phase1ComputeMessage` - Main computation
- `src/zkvm/gruen.zig:gruenPolyDeg3` - Coefficient conversion
- `src/poly/split_eq.zig:GruenSplitEqPolynomial` - E_out/E_in tables

**Jolt:**
- `jolt-core/src/zkvm/registers/read_write_checking.rs:phase1_compute_message`
- `jolt-core/src/poly/split_eq_poly.rs:GruenSplitEqPolynomial`
- `jolt-core/src/subprotocols/read_write_matrix/*` - Sparse matrix

## Success Criteria

- [ ] Understand why q_0/q_X2 differ from Jolt's expectations
- [ ] Identify specific line(s) in Zolt causing divergence
- [ ] Fix computation to match Jolt
- [ ] Cross-verification test passes Stage 4
