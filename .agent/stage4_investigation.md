# Stage 4 Investigation Log

**Status:** Stage 4 fails with `output_claim != expected_output_claim`

## Validated Facts (All Match!)

| Component | Zolt | Jolt | Status |
|-----------|------|------|--------|
| Gamma | `0xBC6DB96DA7E28854F05610B16FA99513` | `250464584498748727615350532275610162451` | ✅ Match |
| Batching coeff[0] | `0x28744516F8AE2E4B09CC0A8C1FB4BD24` | `53772827573231756341512990294283697444` | ✅ Match |
| Input claim (regs) | `27 83 ab 70 1b 95 87 df` | Same (reversed) | ✅ Match |
| MLE sums (rd_wv, rs1_v, rs2_v) | Match Stage 3 claims | | ✅ Match |

## Stage 4 Architecture

Stage 4 has **3 sumcheck instances**:
- **Instance 0**: RegistersReadWriteChecking (main instance)
- **Instance 1**: RamValEvaluation (claim = 0 for fibonacci - no RAM ops)
- **Instance 2**: ValFinal (claim = 0 for fibonacci - no RAM ops)

For fibonacci, only Instance 0 contributes since instances 1 & 2 have zero polynomials.

## How Batched Sumcheck Works

**Jolt's approach** (per `sumcheck.rs`):
1. Each instance computes its own polynomial
2. Polynomials are weighted by batching coeffs: `combined = Σ(poly_i * coeff_i)`
3. Combined polynomial is compressed to [c0, c2, c3]
4. Compressed poly is appended to transcript
5. Challenge is derived from transcript

**Zolt's approach** (per `proof_converter.zig` lines 1780-1839):
1. Same - computes polynomials per instance
2. Same - combines with batching coeffs at lines 1783-1821
3. Same - compresses to [c0, c2, c3] at line 1825
4. Same - appends to transcript at lines 1835-1839
5. Same - derives challenge at line 1841

## The Divergence

**Symptom**: Round 0 challenge differs
- Zolt derives: `2b 6e f8 0c e5 44 d9 18 ...`
- Jolt derives: `9a 95 69 5e 07 04 10 de`

**What's confirmed working**:
- Transcript state matches through gamma derivation
- Batching coefficients match
- Input claims match
- Polynomial combination logic is correct

## Current Hypothesis

~~The issue is likely in how compressed polynomial bytes are serialized~~ - RULED OUT (see "Things Tried")

**New focus**: The issue is in Stage 4 polynomial **computation**, not serialization.

### Key observation from Jolt verifier debug output:
```
[JOLT STAGE4 DEBUG]   rd_write_value_claim = 686954849608142544842930683117740565008420479238252739345477500438178256621
[JOLT STAGE4 DEBUG]   rs1_value_claim = 18552476043348766945413274177395855269555180862918779162105506136388477924969
[JOLT STAGE4 DEBUG]   rs2_value_claim = 5244799667053811342398110744926160704297957708774102577445106913951104137390
[JOLT STAGE4 DEBUG]   eq_val = 1027502176794887237869389874331266800663355178654009808303778100003264603590
[JOLT STAGE4 DEBUG]   combined = 4717642735211727391904767171677409252837369730741723836681285559673432741383
[JOLT STAGE4 DEBUG]   expected = 4801511371718514973618621525633690462841892265614240235189123313842141855751

Instance 0 expected_claim = 4801511371718514973618621525633690462841892265614240235189123313842141855751
After batching: weighted = 12140057478814186156378889252409437120722846392760694384171609527722202919821
After sumcheck: output_claim = 4025718365397880246610377225086562173672770992931618085272964253447434290014
```

### What Jolt computes for expected_output_claim:
```
rd_write_value_claim = rd_wa_claim * (inc_claim + val_claim)
rs1_value_claim = rs1_ra_claim * val_claim
rs2_value_claim = rs2_ra_claim * val_claim
combined = rd_write_value_claim + gamma * (rs1_value_claim + gamma * rs2_value_claim)
expected = eq_val * combined
expected_output_claim = expected * batching_coeff[0]  // (scaled by 2^(max_rounds - instance_rounds) first)
```

### Possible causes:
1. **eq_val computation differs** - Zolt and Jolt might compute EqPolynomial differently
2. **r_cycle point differs** - The evaluation point passed to Stage 4 might be wrong
3. **Stage 4 prover computes wrong polynomial** - The actual sumcheck polynomial evaluations are wrong

## Things Tried

- [x] Reversed r_cycle ordering (LE→BE) - not the issue
- [x] Increased Dory SRS size - not the issue
- [x] Montgomery conversion fix in proof serialization - **NOT THE ISSUE**
  - Hypothesis: `CompressedUniPoly.serialize()` writes Montgomery form bytes, Jolt expects standard form
  - Tested: Added `fromMontgomery()` in `jolt_types.zig:405` and `jolt_types.zig:494`
  - Result: No change - same error with same values
  - Why it was wrong: Stages 1-3 pass with same serialization code and non-zero values
  - Conclusion: Proof serialization format is correct; issue is elsewhere
- [x] Verified batching coefficient application - correct in proof_converter.zig

## ROOT CAUSE FOUND

### Stage 3 challenges differ between Zolt and Jolt!

**Zolt's Stage 3 challenges (LE bytes):**
```
[0] = { 7d, 1a, 1d, a6, 1d, 61, fc, d7, ... }
[1] = { fb, d4, 3f, e1, e1, e3, 67, 40, ... }
...
[7] = { 6b, 34, 63, 76, 9d, e0, 0e, 7a, ... }
```

**Jolt's Stage 3 challenges (LE bytes from verifier):**
```
[0] = [21, 7a, 93, 9f, 1e, bf, 00, 7b, ...]
[1] = [d4, fd, 61, d7, 3e, a5, 2d, 65, ...]
...
[7] = [e5, 44, 91, 1d, 07, be, 76, a7, ...]
```

These are COMPLETELY DIFFERENT values. This is why Stage 4 fails - Zolt passes different r_cycle values to Stage 4 than what Jolt computes from its transcript.

### Why this happens:
- Challenges are derived from the transcript
- Transcript state = f(all polynomials appended so far)
- If Stage 3 polynomials differ, challenges differ
- Different challenges → different r_cycle → different eq_val → Stage 4 fails

### Investigation needed:
1. Why does Stage 3 "pass" if the challenges differ?
2. Where does the transcript diverge during Stage 3?
3. Are Stage 3 polynomial coefficients correct?

## Next Steps

1. **Add debug to Stage 3** - Compare transcript state before/after each round polynomial append
2. **Compare Stage 3 Round 0 polynomial** between Zolt and Jolt
3. **Trace transcript divergence** - Find first round where challenges differ

## Key Files

- `src/zkvm/proof_converter.zig:1780-1860` - Stage 4 polynomial combination and transcript append
- `src/zkvm/jolt_types.zig` - Proof serialization types
- `jolt-core/src/subprotocols/sumcheck.rs:80-115` - Jolt's batched sumcheck
- `jolt-core/src/poly/unipoly.rs:478-486` - Jolt's CompressedUniPoly transcript append
