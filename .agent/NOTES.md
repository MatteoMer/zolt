# Zolt Implementation Notes

## Current Status (2024-12-28)

### What Works
- **Proof Serialization**: Byte-perfect Arkworks compatibility
- **Transcript**: Blake2b matches Jolt exactly
- **Opening Claims**: Non-zero MLE evaluations computed correctly
- **Proof Structure**: All 7 stages, correct round counts

### What's Failing
- **Stage 1 Sumcheck Verification**: The round polynomials don't satisfy the sumcheck relation

## Latest Progress (Iteration 17)

### Opening Claims Now Have Non-Zero Values

Successfully implemented R1CS input MLE evaluation at the challenge point:
- PC values: 470923325918454702788286590928955227900599927949267948307234034664185460615
- OpFlags(AddOperands): 14116661703799451320060418720194240191430100414874762526722692778591556927761
- OpFlags(WriteLookupOutputToRD): Same non-zero value

### Round Polynomial Count Fixed

Stage 1 sumcheck proof now has 4 round polynomials (was 3):
- 1 streaming round
- 3 cycle variable rounds
- For trace length 8 (log₂(8) = 3 cycle vars)

### Sumcheck Verification Still Fails

The verification fails because our round polynomials don't satisfy:
```
s_j(0) + s_j(1) == previous_claim
```

## Root Cause Analysis

### The Sumcheck Round Polynomial Formula (from Jolt)

For each round `j`, the round polynomial is:
```
s_j(X) = l_j(X) · q_j(X)
```

Where:
- `l_j(X)` = linear eq polynomial factor
  - `l_j(0) = current_scalar · (1 - τ_j)`
  - `l_j(1) = current_scalar · τ_j`

- `q_j(X)` = quadratic from multiquadratic grid
  - `q_j(0) = t'_j(0)` (projection at z₀=0)
  - `q_j(∞) = t'_j(∞)` (coefficient of X²)
  - `d_j` derived from sumcheck constraint

### Initial Claim Issue

The initial claim for remaining rounds should be `s₁(r₀)` from evaluating the
UniSkip polynomial at `r₀`. Currently we initialize `current_claim = F.zero()`.

### Multiquadratic Grid Not Implemented

We need to maintain a {0, 1, ∞}^w grid that tracks the constraint products:
```
t'(z₀, ..., z_{w-1}) = Σ_{x_out} E_out(x_out) · Σ_{x_in} E_in(x_in) · Az(x_out, x_in, z₀) · Bz(x_out, x_in, z₀)
```

This requires implementing `project_to_first_variable` for projections.

## Files to Fix

### High Priority
1. `src/zkvm/spartan/streaming_outer.zig`
   - Initialize `current_claim` from UniSkip evaluation
   - Implement proper multiquadratic grid
   - Fix `computeRemainingRoundPoly`

2. `src/zkvm/spartan/outer.zig`
   - Fix UniSkip polynomial computation
   - Return proper claim `s₁(r₀)`

### Medium Priority
3. `src/poly/multiquadratic.zig`
   - Ensure ternary grid {0, 1, ∞}^w is correct
   - Implement `project_to_first_variable`

## Mathematical Reference

### UniSkip First Round
```
s₁(Y) = L(τ_high, Y) · Σ_{x_out, x_in} eq(τ, (x_out, x_in)) · Az(x_out, x_in, Y) · Bz(x_out, x_in, Y)
```

### Remaining Rounds
```
s_j(X) = eq(τ_j, X) · current_scalar · [t'(0) + d·X + t'(∞)·X²]
```

### What We Prove
```
Σ_{x_out, x_in, z} eq(τ, (x_out, x_in, z)) · Az(x_out, x_in, z) · Bz(x_out, x_in, z) = 0
```

This is zero for a valid R1CS-satisfying witness.

## Test Commands

```bash
# Build Zolt
cd /Users/matteo/projects/zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove -o /tmp/zolt_proof_dory.bin --jolt-format examples/fibonacci.elf

# Test with Jolt
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```

## Progress Log

### Iteration 17
- Added R1CS input MLE evaluation (`evaluation.zig`)
- Fixed round polynomial count (4 for trace_len=8)
- Opening claims now show non-zero values
- Sumcheck still fails due to incorrect polynomial values

### Next Steps
1. Fix UniSkip polynomial evaluation
2. Initialize current_claim from UniSkip
3. Implement proper multiquadratic grid
4. Test with simple trace

## References

- Jolt Paper: Section on Spartan/Lasso
- Gruen's Method: Efficient multilinear sumcheck via split-eq tables
- Jolt Code: `jolt-core/src/zkvm/spartan/outer.rs`

---

## Previous Implementation Notes

### Univariate Skip Implementation (Iteration 11) - SUCCESS

Successfully implemented Jolt's univariate skip optimization for stages 1-2:

1. **univariate_skip.zig** - Core module with:
   - Constants matching Jolt (NUM_R1CS_CONSTRAINTS=19, DEGREE=9, NUM_COEFFS=28)
   - `buildUniskipFirstRoundPoly()` - Produces degree-27 polynomial from extended evals
   - `LagrangePolynomial` - Interpolation on extended symmetric domain
   - `uniskipTargets()` - Compute extended evaluation points

2. **spartan/outer.zig** - Spartan outer prover:
   - `SpartanOuterProver` with univariate skip support
   - `computeUniskipFirstRoundPoly()` - Generates proper first-round polynomial

3. **proof_converter.zig** - Updated to generate proper-degree polynomials:
   - Stage 1: `createUniSkipProofStage1()` - 28 coefficients (degree 27)
   - Stage 2: `createUniSkipProofStage2()` - 13 coefficients (degree 12)

### Blake2b Transcript Compatibility (Complete)

Successfully implemented Blake2b transcript matching Jolt's implementation:
- 32-byte state with round counter
- Messages right-padded to 32 bytes
- Scalars serialized LE then reversed to BE (EVM format)
- 128-bit challenges
- Vector operations with begin/end markers

All 7 test vectors from Jolt verified to match.

### Dory Commitment Implementation (Complete)

**Location**: `src/poly/commitment/dory.zig`

1. **DoryCommitmentScheme** - Matches Jolt's DoryCommitmentScheme
   - `setup(allocator, max_num_vars)` - Generate SRS using "Jolt Dory URS seed"
   - `commit(params, evals)` - Commit polynomial to GT element
   - DorySRS with G1/G2 generators
   - DoryCommitment = GT = Fp12

2. **GT (Fp12) Serialization** - Added to `src/field/pairing.zig`
   - `Fp12.toBytes()` - 384 bytes arkworks format (12 × 32 bytes)
   - `Fp12.fromBytes()` - Deserialize from arkworks format
   - Serialization order: c0.c0.c0, c0.c0.c1, ..., c1.c2.c1

### Cross-Verification Status

**Jolt successfully deserializes Zolt proofs!**

```
cargo test --package jolt-core test_deserialize_zolt_proof -- --ignored --nocapture

Successfully deserialized Zolt proof!
  Trace length: 8
  RAM K: 65536
  Bytecode K: 65536
  Commitments: 5
```

### Test Status

All 608 Zolt tests pass:
```
zig build test --summary all
Build Summary: 5/5 steps succeeded; 608/608 tests passed
```

Cross-verification tests (Jolt):
- `test_deserialize_zolt_proof`: PASS
- `test_debug_zolt_format`: PASS
- `test_verify_zolt_proof`: FAIL (Stage 1 sumcheck verification failed)
