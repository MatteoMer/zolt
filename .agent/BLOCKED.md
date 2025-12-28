# Zolt-Jolt Compatibility: Current Blockers

## Achieved

1. **Format Compatibility** ✅
   - Jolt successfully deserializes Zolt proofs
   - All 48 opening claims preserved
   - UniSkip polynomial degrees correct (28/13 coefficients)
   - Dory commitment GT serialization correct
   - Blake2b transcript produces matching challenges

2. **Proof Structure** ✅
   - 7-stage format with UniSkipFirstRoundProof
   - OpeningClaims with proper VirtualPolynomial ordering
   - All 36 R1CS inputs for SpartanOuter
   - All 13 OpFlags variants preserved

## Current Blocker: R1CS Constraint Mismatch

### Why Verification Fails

The Jolt verifier computes `expected_output_claim` from R1CS constraint evaluations:
```rust
let claim = sumcheck.expected_output_claim(opening_accumulator, r_slice);
if output_claim != expected_output_claim {
    return Err(SumcheckVerificationError);  // Line 247-248
}
```

Our "zero proofs" produce `output_claim = 0`, but `expected_output_claim` is computed
from actual R1CS constraint evaluations at the challenge point. These don't match.

### Fundamental Issue

Zolt and Jolt have different R1CS constraint structures:
- **Jolt**: 19 R1CS constraints, 7 verification stages, specific constraint polynomials
- **Zolt**: Different constraint layout, 6 internal stages

For Jolt to verify a Zolt proof, Zolt would need to:
1. Implement Jolt's exact 19 R1CS constraints
2. Compute Az(x,y) · Bz(x,y) at extended domain points
3. Generate proper univariate skip polynomials from constraint evaluations
4. Ensure sumcheck round polynomials satisfy p(0) + p(1) = claim with matching values

### Scope Analysis

This is not a serialization issue - it's an architectural incompatibility. To achieve full
cross-verification, Zolt would need to adopt Jolt's R1CS constraint system exactly, which
is a fundamental redesign of Zolt's proving logic.

## Options

### Option A: Port Jolt's R1CS to Zolt (Major)
- Implement all 19 R1CS constraints from Jolt's `constraints.rs`
- Match constraint evaluation order and batching
- Estimated effort: Weeks of work

### Option B: Proof Translation Layer (Complex)
- Generate intermediate format that can be translated
- Would require deep understanding of both constraint systems
- May not be feasible if constraint semantics differ

### Option C: Accept Format-Only Compatibility (Current)
- Zolt proofs can be serialized in Jolt format
- Jolt can deserialize and inspect Zolt proofs
- Full verification not possible without constraint match

## Conclusion

The serialization/format compatibility goal has been achieved. Full verification
compatibility would require Zolt to adopt Jolt's R1CS constraint system, which
is beyond the scope of format alignment work.
