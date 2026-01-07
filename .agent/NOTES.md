# Zolt-Jolt Cross-Verification Progress

## Session 15 Summary

### Root Cause Analysis: Stage 2 Failure

The Stage 2 batched sumcheck fails at Jolt verification. After extensive debugging:

#### 1. r0 Mismatch
- Zolt r0: `8768758914789955585787902790032491769856779696899125603611137465800193155946`
- Jolt r0: `16176819525807790011525369806787798080841445107270164702191186390206256879646`

r0 is derived from the transcript after the Stage 2 UniSkip first round polynomial is appended.

#### 2. Sumcheck Claim Errors
Starting at round 23, the sumcheck constraint `s(0)+s(1) != old_claim` fails.

#### 3. Missing Provers
The batched sumcheck has 5 instances:
- Instance 0: ProductVirtualRemainder - **implemented**
- Instance 1: RamRafEvaluation - **uses zero fallback**
- Instance 2: RamReadWriteChecking - **uses zero fallback**
- Instance 3: OutputSumcheck - **implemented**
- Instance 4: InstructionLookupsClaimReduction - **uses zero fallback**

### What MATCHES Between Zolt and Jolt

1. **fused_left**: `3680814111042145831100417079225278919431426777627349458700618452903652360804`
2. **fused_right**: `5628401284835057616148875782341094898402011560234054472864896388346845354264`
3. **tau_high**: `1724079782492782403949918631195347939403999634829548103697761600182229454970`
4. **Stage 2 UniSkip domain_sum = input_claim** (polynomial passes sum check)
5. **Stage 2 UniSkip coeffs[0]**: Both are zero
6. **All 26 sumcheck challenges** (rounds 0-25)

### Serialization Format

The Stage 2 UniSkip polynomial serialization is correct:
- Writes length (usize = 13 coefficients)
- Writes all 13 coefficients as field elements (32 bytes each, LE)

### Transcript Protocol

UniSkip appends to transcript:
1. `"UncompressedUniPoly_begin"`
2. All coefficients (c0, c1, c2, ..., c12)
3. `"UncompressedUniPoly_end"`
4. Sample r0 challenge

The issue may be in how coefficients c1-c12 are computed or serialized.

### Key Insight

Despite the polynomial passing its internal sum check:
```
domain_sum = { 43, 113, 211, 223, 28, 171, 74, 188, 15, 63, 128, 23, 194, 28, 198, 221, 28, 243, 107, 60, 31, 26, 41, 160, 149, 101, 217, 37, 27, 218, 236, 52 } (Zolt BE)
input_claim = [52, 236, 218, 27, 37, 217, 101, 149, 160, 41, 26, 31, 60, 107, 243, 28, 221, 198, 28, 194, 23, 128, 63, 15, 188, 74, 171, 28, 223, 211, 113, 43] (Jolt LE)
```

These match (just different endianness in display). So the polynomial construction is correct, but the transcript state must differ somewhere.

### Next Steps

1. Compare ALL 13 coefficients between Zolt and Jolt (not just c0)
2. Add more debug output to Jolt's UniSkip verification
3. Trace transcript state byte-by-byte between Zolt and Jolt

### Files Modified
- `src/zkvm/proof_converter.zig`
- `src/zkvm/r1cs/univariate_skip.zig`
- `.agent/TODO.md`

### Test Status
- All 712 internal tests pass
- Stage 1 passes Jolt verification
- Stage 2 fails at batched sumcheck output_claim check
