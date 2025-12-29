# Zolt-Jolt Compatibility Notes

## Current Status (Session 19 - December 29, 2024)

### Session 19 - EqPolynomial Big-Endian Fix

**Status**: Fixed EqPolynomial.evals() to use big-endian indexing. Ratio changed from ~1.2 to ~1.34.

**Current Values:**
- output_claim = 21176670064311113248327121399637823341669491654917035040693110982193526510099
- expected = 15830891598945306629010829910964994017594280764528826029442912827815044293203
- Ratio: ~1.338

**Key Finding:**
The EqPolynomial.evals() was using little-endian indexing (first variable controls LSB of index), but Jolt uses big-endian (first variable controls MSB of index).

The fix in `src/poly/mod.zig` changed the table construction to match Jolt's algorithm.

**Remaining Issue:**
Despite fixing EqPolynomial, the expected output claim still doesn't match. The ratio changed, indicating progress, but there's still a mismatch.

### Verified Components

1. **EqPolynomial.evals()** - NOW matches Jolt's big-endian convention ✓
2. **split_eq E tables** - Big-endian indexing ✓
3. **computeCubicRoundPoly** - Matches Jolt's gruen_poly_deg_3 ✓
4. **Constraint definitions** - Match Jolt exactly ✓
5. **First/Second group indices** - Correct ✓
6. **Lagrange domain** - {-4,...,5} matching Jolt ✓
7. **r_cycle computation** - challenges[1..] reversed ✓
8. **tau_bound_r_tail** - Uses all 11 challenges reversed ✓

### R1CS Input Evaluations from Debug Test

The Jolt verifier reads these from Zolt's proof:
```
[0] LeftInstructionInput => 1891106565568279723...
[1] RightInstructionInput => 7815611673650657012...
...
[26] OpFlags(Load) => 3011325360017154853...
[27] OpFlags(Store) => 236388320696034965...
...
```

These are the MLE evaluations at r_cycle, computed by Zolt's `R1CSInputEvaluator.computeClaimedInputs`.

### Expected Output Claim Formula

```
expected = tau_high_bound_r0 * tau_bound_r_tail * inner_sum_prod

tau_high_bound_r0 = 12826223823521221946558236994983949818417555624042883646429867758366229902653
tau_bound_r_tail = 8603170193156393692230406904477425345246858118990710025063879121918491155552
inner_sum_prod = 18034351451926217603377778125390129356175428155321402285998220639456984589925

inner_sum_prod = az_final * bz_final
az_final = az_g0 + r_stream * (az_g1 - az_g0)
bz_final = bz_g0 + r_stream * (bz_g1 - bz_g0)

az_g0 = 11307336203138615556774249738056012003110982357404568850780192491166454123623
bz_g0 = 15871205286623107670421364844665791558158034368595592735643934424121987926930
```

### Potential Remaining Issues

1. **Witness extraction mismatch**
   - Zolt's `R1CSCycleInputs.fromTraceStep` may produce different values than Jolt's `R1CSCycleInputs::from_trace`
   - The witness values are summed with eq weights to produce the sumcheck output

2. **MLE evaluation mismatch**
   - The `computeClaimedInputs` function may have a subtle indexing issue
   - Check if cycle index t is mapped correctly to eq_evals[t]

3. **Constraint evaluation during sumcheck**
   - The streaming round computes az_g0, bz_g0, etc. for each cycle
   - If these differ from what the verifier expects, the output claim will mismatch

### Next Steps

1. Add debug output to Zolt's streaming round to print az_g0, bz_g0 for first few cycles
2. Compare with what the Jolt verifier computes from the opening claims
3. Track down which component differs

---

## Previous Sessions

### Session 17-18 - Big-Endian Eq Table Fix

Fixed E_out/E_in tables to use big-endian indexing. Changed ratio from 0.8129 to 1.1987.

### Session 16 - Initial Investigation

Investigated 0.8129 ratio (close to 13/16).

---

## Big-Endian Convention

From jolt-core/src/poly/eq_poly.rs:
```
evals(r)[i] = eq(r, b₀…b_{n-1})
where i has MSB b₀ and LSB b_{n-1}
```

Index i's bit pattern in BIG-ENDIAN corresponds to variable values:
- MSB of i → r[0] (first variable)
- LSB of i → r[n-1] (last variable)
