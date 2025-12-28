# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: NEED TRANSCRIPT PREAMBLE MATCHING**

Key achievements:
1. **R1CS Input MLE Evaluations** - Opening claims now contain actual computed values
2. **Correct Round Polynomial Count** - Stage 1 has proper 1 + num_cycle_vars polynomials
3. **Cross-Deserialization** - Jolt successfully deserializes Zolt proofs
4. **Non-zero Round Polynomials** - Stage 1 sumcheck produces non-trivial polynomials
5. **UniSkip Polynomial Structure** - 28 coefficients as expected

Current issue: **Transcript state mismatch**
- Jolt's verifier derives `tau` from transcript after Fiat-Shamir preamble
- Zolt uses placeholder `tau` values
- Different tau → different expected_output_claim → verification fails

---

## Root Cause Analysis

### Jolt's Fiat-Shamir Preamble Flow

1. Transcript initialized with `b"Jolt"` label
2. `fiat_shamir_preamble` adds:
   - `memory_layout.max_input_size` (u64)
   - `memory_layout.max_output_size` (u64)
   - `memory_layout.memory_size` (u64)
   - `program_io.inputs` (bytes)
   - `program_io.outputs` (bytes)
   - `program_io.panic` (u64)
   - `ram_K` (u64)
   - `trace_length` (u64)
3. Commitments are appended (GT elements for Dory)
4. `tau = transcript.challenge_vector_optimized(num_rows_bits)`

### What Zolt Needs to Do

1. Initialize transcript with `b"Jolt"` (currently uses `"jolt_v1"`)
2. Implement `fiat_shamir_preamble` to append same data
3. Get memory layout from preprocessing
4. Append commitments in same format as Jolt
5. Derive tau from transcript

---

## Major Milestones Achieved

1. ✅ **Blake2b Transcript** - Identical Fiat-Shamir challenges as Jolt
2. ✅ **Proof Types** - JoltProof with 7-stage structure
3. ✅ **Arkworks Serialization** - Byte-perfect format compatibility
4. ✅ **Dory Commitment** - GT (Fp12) serialization
5. ✅ **Cross-Deserialization** - Jolt successfully deserializes Zolt proofs!
6. ✅ **Univariate Skip Infrastructure** - Degree-27/12 polynomials
7. ✅ **All 48 Opening Claims** - Including all R1CS inputs + OpFlags
8. ✅ **19 R1CS Constraints** - Matching Jolt's exact structure
9. ✅ **Constraint Evaluators** - Az/Bz for first and second groups
10. ✅ **GruenSplitEqPolynomial** - Prefix eq tables
11. ✅ **MultiquadraticPolynomial** - Ternary grid expansion
12. ✅ **StreamingOuterProver** - Framework with round poly generation
13. ✅ **R1CS Input MLE Evaluation** - Compute actual evaluations

---

## Immediate Tasks

### 1. Fix Transcript Label

Change Zolt's transcript from `"jolt_v1"` to `"Jolt"` to match Jolt exactly.

Location: `src/zkvm/mod.zig` line 449

### 2. Implement Fiat-Shamir Preamble

Create a function that matches Jolt's `fiat_shamir_preamble`:

```zig
fn fiatShamirPreamble(
    transcript: *Blake2bTranscript,
    program_io: *const JoltDevice,
    ram_K: usize,
    trace_length: usize,
) void {
    transcript.appendU64(program_io.memory_layout.max_input_size);
    transcript.appendU64(program_io.memory_layout.max_output_size);
    transcript.appendU64(program_io.memory_layout.memory_size);
    transcript.appendBytes(program_io.inputs);
    transcript.appendBytes(program_io.outputs);
    transcript.appendU64(if (program_io.panic) 1 else 0);
    transcript.appendU64(ram_K);
    transcript.appendU64(trace_length);
}
```

### 3. Read Program IO from Jolt-Generated File

The test uses `/tmp/fib_io_device.bin` which contains the JoltDevice.
Zolt needs to either:
- Read this file during proof generation, OR
- Generate compatible IO from its own execution

### 4. Derive tau from Transcript

After preamble and commitments, derive tau:

```zig
const num_rows_bits = spartan_key.num_rows_bits();
var tau = try allocator.alloc(F, num_rows_bits);
for (0..num_rows_bits) |i| {
    tau[i] = transcript.challengeScalar();
}
```

---

## Test Status

### All 608 Tests Passing (Zolt)

```
zig build test --summary all
Build Summary: 5/5 steps succeeded; 608/608 tests passed
```

### Cross-Verification Tests (Jolt)

| Test | Status | Details |
|------|--------|---------|
| `test_deserialize_zolt_proof` | ✅ PASS | 26558 bytes, 48 claims |
| `test_debug_zolt_format` | ✅ PASS | All claims valid |
| `test_verify_zolt_proof` | ❌ FAIL | Stage 1 sumcheck fails |

**Debug output from verification:**
- UniSkip poly has 28 coefficients (correct)
- UniSkip coeffs are all 0 (should not be for real witness)
- Stage 1 sumcheck has 4 round polys (correct for trace_length=8)
- Round polys have non-zero coefficients (progress!)

---

## Key Files

### Core Implementation
| File | Status | Purpose |
|------|--------|---------|
| `src/transcripts/blake2b.zig` | ✅ Done | Blake2bTranscript |
| `src/zkvm/mod.zig` | 🔄 Need preamble | Jolt proof generation |
| `src/zkvm/proof_converter.zig` | 🔄 Need tau fix | Stage conversion |
| `src/zkvm/spartan/outer.zig` | ✅ Done | UniSkip computation |
| `src/zkvm/spartan/streaming_outer.zig` | ✅ Done | Remaining rounds |

---

## Summary

**Serialization Compatibility: COMPLETE**
**Transcript Integration: NEEDS PREAMBLE**
**Verification: BLOCKED ON PREAMBLE**

The core cryptographic components are in place. The remaining issue is
matching Jolt's Fiat-Shamir transcript state so that tau values are
derived identically.

Next steps:
1. Fix transcript label
2. Implement fiat_shamir_preamble
3. Load program_io from file or generate compatible
4. Derive tau from transcript after preamble + commitments
5. Generate UniSkip polynomial with correct tau
6. Re-test verification
