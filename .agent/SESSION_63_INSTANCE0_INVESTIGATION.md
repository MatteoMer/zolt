# Session 63: Instance 0 Investigation - The Real Problem

## Date
2026-01-26

## Critical Discovery

After the user correctly challenged my lazy assessment, we discovered that **Instance 0 does NOT match**, not just "both non-zero":

```
Zolt computes:  1,930,212,391,506,207,668,262,592,570,261,649,377,277,154,075,731,561,955,806,344,171,090,306,014,790
Jolt expects:  17,266,308,235,761,105,456,215,483,608,498,978,797,273,351,746,775,005,780,030,669,392,838,509,233,139
Difference:    15,336,095,844,254,897,787,952,891,038,237,329,419,996,197,671,043,443,824,224,325,221,748,203,218,349
```

This is a **HUGE** difference (~8x), not just a small arithmetic error.

## What We Verified

### ✅ Stage 3 Claims Are Correct

**Zolt wrote:**
```
RdWriteValue (RegistersClaimReduction): [26, 44, 126, 91, 14, 228, 117, 109...]
Rs1Value (RegistersClaimReduction):     [188, 86, 19, 38, 123, 29, 129, 231...]
Rs2Value (RegistersClaimReduction):     [21, 100, 102, 153, 207, 166, 176, 21...]
```

**Jolt reads:**
```
rd_wv_claim (Stage 3):  [26, 44, 126, 91, 14, 228, 117, 109...] ✅ EXACT MATCH
rs1_rv_claim (Stage 3): [188, 86, 19, 38, 123, 29, 129, 231...] ✅ EXACT MATCH
rs2_rv_claim (Stage 3): [21, 100, 102, 153, 207, 166, 176, 21...] ✅ EXACT MATCH
```

The serialization is perfect. Stages 1-3 pass verification.

### ✅ Gamma Matches

Both Zolt and Jolt use the same gamma value:
```
gamma: [206, 210, 137, 221, 10, 103, 12, 206, 81, 87, 154, 23, 68, 152, 9, 144...]
     = 191,458,650,385,834,247,327,713,611,961,596,498,638 (as integer)
```

### ✅ Formula Is Correct

The formula used is:
```
input_claim = rd_wv_claim + gamma * (rs1_rv_claim + gamma * rs2_rv_claim)
```

We verified this in Jolt's code at `src/zkvm/registers/read_write_checking.rs:101-148`.

### ✅ Zolt's Arithmetic Is Correct

Python verification:
```python
rd = int.from_bytes(rd_bytes, 'little') % p
rs1 = int.from_bytes(rs1_bytes, 'little') % p
rs2 = int.from_bytes(rs2_bytes, 'little') % p
gamma = int.from_bytes(gamma_bytes, 'little') % p

result = (rd + gamma * (rs1 + gamma * rs2)) % p
# result = 1,930,212,391,506,207,668,262,592,570,261,649,377,277,154,075,731,561,955,806,344,171,090,306,014,790

zolt_computed = 1930212391506207668262592570261649377277154075731561955806344171090306014790
assert result == zolt_computed  # ✅ MATCH
```

Zolt's computation is mathematically correct given the Stage 3 claims.

## The Paradox

**User's Key Insight:** "I don't understand how the claims could be wrong since Stage 3 passes"

This is **absolutely correct**. If Stage 3 verification passes, then:
1. The Stage 3 sumcheck verified correctly
2. The opening claims (RdWriteValue, Rs1Value, Rs2Value) were validated
3. These values are cryptographically bound by the proof

So the Stage 3 claims **cannot be wrong** - Jolt's verifier would have rejected them.

Yet when Jolt computes Instance 0's `expected_claim` in Stage 4, it gets `17,266...` instead of `1,930...`.

## What Could Be Wrong?

### Hypothesis 1: Jolt Computes expected_claim Differently

Maybe Jolt's Stage 4 verifier doesn't use the formula we think it does. Perhaps:
- It reads from a different sumcheck stage (not RegistersClaimReduction)
- It applies a transformation we're not aware of
- It uses a different gamma value for Stage 4

**Evidence against:** We saw Jolt's code clearly states:
```rust
let (_, rd_wv_claim) = accumulator.get_virtual_polynomial_opening(
    VirtualPolynomial::RdWriteValue,
    SumcheckId::RegistersClaimReduction,  // ← Explicitly RegistersClaimReduction
);
```

### Hypothesis 2: Stage 3 Claims Are Stored/Retrieved Incorrectly

Even though the bytes match when printed, perhaps:
- Jolt stores them differently in the opening_accumulator
- The retrieval process applies some transformation
- There's a field element representation issue (Montgomery form, etc.)

**To test:** Add debug output inside `get_virtual_polynomial_opening()` to see the actual field element values, not just bytes.

### Hypothesis 3: Multiple RegistersClaimReduction Claims

There are 3 Rs1Value claims in the proof:
1. Rs1Value (SpartanOuter)
2. Rs1Value (InstructionInputVirtualization)
3. Rs1Value (RegistersClaimReduction)

Maybe Jolt is retrieving from the wrong one?

**Evidence against:** The code explicitly checks that RegistersClaimReduction matches InstructionInputVirtualization:
```rust
assert_eq!(rs1_rv_claim, rs1_rv_claim_instruction_input);
```

### Hypothesis 4: Endianness Issue in Field Element Construction

The bytes are correct, but maybe:
- Zolt stores field elements as LE limbs internally
- Jolt expects BE limbs
- The serialized bytes match, but the field element values differ

**To test:** Print the actual field element in both Zolt and Jolt, not just the bytes.

### Hypothesis 5: Stage 3 Doesn't Actually Pass

Maybe Jolt's verifier silently continues even when Stage 3 fails?

**Evidence against:** The test code would fail earlier if Stage 3 verification threw an error. Rust's Result type would propagate the error.

## Current Status - Instance Validity

❌ **Instance 0 (RegistersRWC):**
- Zolt computes: `1,930...`
- Jolt expects: `17,266...`
- **NOT VALID**

✅ **Instance 1 (RamValEvaluation):**
- Both expect: `0`
- **VALID**

✅ **Instance 2 (RamValFinalEvaluation):**
- Both expect: `0`
- **VALID**

## Next Debugging Steps

### Step 1: Add Field Element Debug Output to Jolt

Modify `src/zkvm/registers/read_write_checking.rs` around line 101-148:

```rust
fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
    let (_, rd_wv_claim) = accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RdWriteValue,
        SumcheckId::RegistersClaimReduction,
    );

    // ADD THIS:
    eprintln!("[JOLT STAGE4 DEBUG] rd_wv_claim as field element: {:?}", rd_wv_claim);
    eprintln!("[JOLT STAGE4 DEBUG] rd_wv_claim.into_bigint(): {:?}", rd_wv_claim.into_bigint());

    let (_, rs1_rv_claim) = accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::Rs1Value,
        SumcheckId::RegistersClaimReduction,
    );

    eprintln!("[JOLT STAGE4 DEBUG] rs1_rv_claim as field element: {:?}", rs1_rv_claim);

    // ... etc for rs2 and gamma

    let input_claim = rd_wv_claim + self.gamma * (rs1_rv_claim + self.gamma * rs2_rv_claim);
    eprintln!("[JOLT STAGE4 DEBUG] Computed input_claim: {:?}", input_claim);
    eprintln!("[JOLT STAGE4 DEBUG] Computed input_claim.into_bigint(): {:?}", input_claim.into_bigint());

    input_claim
}
```

Compare these with Zolt's values.

### Step 2: Verify Stage 3 Actually Passes

Add assertion to the test:

```rust
match verifier.verify() {
    Ok(()) => println!("SUCCESS..."),
    Err(e) => {
        println!("Verification failed: {:?}", e);
        panic!("Verification should not fail!");  // ← ADD THIS
    }
}
```

Run test. If it panics before Stage 4, then Stage 3 is failing.

### Step 3: Check Field Element Representation

In Zolt's proof_converter.zig, when computing input_claim_registers:

```zig
const input_claim_registers = stage3_claims.rd_write_value
    .add(gamma_stage4.mul(stage3_claims.rs1_value))
    .add(gamma_stage4.mul(gamma_stage4).mul(stage3_claims.rs2_value));

// ADD DEBUG:
std.debug.print("[ZOLT STAGE4] rd_write_value.limbs = [{}, {}]\n",
    .{stage3_claims.rd_write_value.limbs[0], stage3_claims.rd_write_value.limbs[1]});
std.debug.print("[ZOLT STAGE4] rs1_value.limbs = [{}, {}]\n",
    .{stage3_claims.rs1_value.limbs[0], stage3_claims.rs1_value.limbs[1]});
// ... etc
```

### Step 4: Trace Opening Accumulator

Add debug to `opening_proof.rs` in Jolt:

```rust
fn get_virtual_polynomial_opening(
    &self,
    polynomial: VirtualPolynomial,
    sumcheck: SumcheckId,
) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
    eprintln!("[OPENING_ACC] Getting: {:?} from {:?}", polynomial, sumcheck);

    let (point, claim) = self.openings.get(&OpeningId::Virtual(polynomial, sumcheck))
        .unwrap_or_else(|| panic!("No opening found"));

    eprintln!("[OPENING_ACC] Found claim: {:?}", claim);
    eprintln!("[OPENING_ACC] Claim bytes: {:?}", claim.to_bytes());

    (point.clone(), *claim)
}
```

This will show if the accumulator is returning different values than expected.

## Key Files

**Zolt:**
- `src/zkvm/proof_converter.zig:1625-1635` - Instance 0 computation
- `src/field/bn254.zig` - Field element implementation

**Jolt:**
- `jolt-core/src/zkvm/registers/read_write_checking.rs:101-148` - Instance 0 expected_claim
- `jolt-core/src/poly/opening_proof.rs:468-479` - Opening accumulator retrieval
- `jolt-core/src/zkvm/claim_reductions/registers.rs:148-187` - Stage 3 cache_openings

## Conclusion

The termination bit fixes for Instances 1 & 2 are working perfectly. The Instance 0 mismatch is a **separate, pre-existing bug** that appears to be related to how Stage 3 claims are interpreted as field elements when computing Stage 4's expected_claim.

The bug is subtle because:
- ✅ Bytes serialize/deserialize correctly
- ✅ Stage 3 passes verification
- ✅ Arithmetic is mathematically correct
- ❌ Final result is wildly different

This suggests a field element representation or retrieval issue, not a serialization or arithmetic bug.
