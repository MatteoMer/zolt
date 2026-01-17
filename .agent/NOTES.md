# Zolt-Jolt Cross-Verification Progress

## Session 40 Summary - Stage 4 Investigation (2026-01-17)

### Current Status
- **Stage 1: PASSES** ✓
- **Stage 2: PASSES** ✓ (Fixed by removing synthetic termination write)
- **Stage 3: PASSES** ✓
- **Stage 4: FAILS** - sumcheck output_claim != expected_output_claim

### Stage 2 Fix
Removed synthetic termination write from memory trace. In Jolt, the termination bit
is set directly in val_final during OutputSumcheck, NOT in the execution/memory trace.
The RWC sumcheck only includes actual LOAD/STORE instructions.

### Stage 4 Deep Investigation

#### Verified Matches
1. **Transcript state**: IDENTICAL between Zolt and Jolt at all checkpoints
2. **Challenge bytes**: IDENTICAL (f5 ce c4 8c b0 64 ba b5 ce 4d a4 2a db 38 f8 ac)
3. **Input claims**: ALL THREE match exactly
4. **Batching coefficients**: MATCH
5. **Polynomial coefficients in proof**: MATCH

#### The Mystery
Despite all values matching, the sumcheck verification fails:
- output_claim = 19271728596168755243423895321875251085487803860811927729070795448153376555895
- expected_output_claim = 5465056395000139767713092380206826725893519464559027111920075372240160609265

The transcript state matches after UniPoly_end (round 438):
- Jolt: state = [a1, 26, 18, ca, 99, 40, f2, f2]
- Zolt: state = [a1, 26, 18, ca, 99, 40, f2, f2]

But the derived challenges differ:
- Zolt: { 28, 74, 106, 25, 220, 50, 233, 243, ... }
- Jolt: [ac, f8, 38, db, 2a, a4, 4d, ce, ...]

### Root Cause Analysis

The challenge in Zolt is stored as:
```zig
result = F{ .limbs = .{ 0, 0, masked_low, masked_high } };
```

But this is NOT proper Montgomery form. When toBytesBE() is called (which calls
fromMontgomery()), it converts incorrectly.

**Key insight from challenge derivation debug:**
- mont_limbs = [0, 0, 0xb5ba64b08cc4cef5, 0x0cf838db2aa44dce]
- But challenge.toBytesBE() shows { 28, 74, 106, 25, ... }

The [0, 0, low, high] format is Jolt's MontU128Challenge optimization, but Zolt's
field type expects proper Montgomery form. When Zolt multiplies by this challenge,
it may produce wrong results.

### Hypothesis
Jolt's MontU128Challenge has special multiplication code that handles the [0, 0, low, high]
format. Zolt's F type doesn't have this specialization, so field operations with the
challenge produce incorrect results.

### Potential Fix
Convert the 125-bit challenge to proper Montgomery form:
```zig
const standard_value = F{ .limbs = .{ low, high, 0, 0 } };
const montgomery = standard_value.toMontgomery();
```

But the code has a comment saying this was tried and didn't work - need to investigate why.

---

## Session 39 Summary - Stage 2 Fix (2026-01-17)

Found that Fibonacci has NO actual STORE/LOAD instructions. The synthetic termination
write was causing a mismatch between R1CS claims (0) and RWC polynomial (non-zero).

Fixed by removing recordTerminationWrite() calls from the tracer.

---

## Previous Sessions

### Stage 3 Fix (Session 35)
- Fixed prefix-suffix decomposition convention (r_hi/r_lo)

### Stage 1 Fix
- Fixed NextPC = 0 issue for NoOp padding

---

## Technical References

- Jolt MontU128Challenge: `jolt-core/src/field/challenge/mont_ark_u128.rs`
- Jolt BatchedSumcheck verify: `jolt-core/src/subprotocols/sumcheck.rs:180`
- Zolt Blake2b transcript: `src/transcripts/blake2b.zig`
- Zolt Stage 4 proof: `src/zkvm/proof_converter.zig` line ~1700
