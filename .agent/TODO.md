# Zolt-Jolt Compatibility: Current Status

## Status: Stage 2 Verification Failure 🔴

## Session 76 Summary (2026-01-29)

### Major Progress: Proof Deserialization Fixed!

**Working:**
- ✅ Proof deserialization (all 7 stages, 91 opening claims, 37 commitments)
- ✅ Stage 1 (OuterRemainingSumcheck) verification passes
- ✅ Preprocessing loading from both compressed and uncompressed formats

**Failing:**
- ❌ Stage 2 sumcheck verification - expected_output_claim mismatch

### Changes Made This Session

1. Added `deserialize_from_bytes_uncompressed` method to `Serializable` trait in Jolt
2. Fixed test to use correct deserialization format (compressed for both preprocessing and proof)
3. Identified that Stage 2's `expected_output_claim` doesn't match the proof's `output_claim`

### Commits

- `db0e57e3` - feat: add uncompressed deserialization support to Serializable trait
- `de20eda` - docs: update TODO.md with Stage 2 failure analysis

### Stage 2 Error Details

```
output_claim:          15906954023365202249122192714132265766544458757312739318826275235085359324853
expected_output_claim: 11386433087960536582639845443917888291615956842149860534020066572649924103188
```

Stage 2 is a batched sumcheck with 5 instances:
1. ProductVirtualRemainderVerifier (n_cycle_vars rounds)
2. RamRafEvaluationSumcheckVerifier (log_ram_k rounds)
3. RamReadWriteCheckingVerifier (log_ram_k + n_cycle_vars rounds - max!)
4. OutputSumcheckVerifier (log_ram_k rounds)
5. InstructionLookupsClaimReductionSumcheckVerifier (n_cycle_vars rounds)

### Key Discovery: Factor Claims Source

The `expected_output_claim` for each Stage 2 instance depends on factor claims retrieved from the proof's opening claims at `SumcheckId::SpartanProductVirtualization`:

**ProductVirtualRemainderVerifier needs these 8 factors:**
1. VirtualPolynomial::LeftInstructionInput
2. VirtualPolynomial::RightInstructionInput
3. VirtualPolynomial::InstructionFlags(IsRdNotZero = 6)
4. VirtualPolynomial::OpFlags(WriteLookupOutputToRD = 6)
5. VirtualPolynomial::OpFlags(Jump = 5)
6. VirtualPolynomial::LookupOutput
7. VirtualPolynomial::InstructionFlags(Branch = 4)
8. VirtualPolynomial::NextIsNoop

These are stored in Zolt at `proof_converter.zig` lines 1329-1364 in `stage2_result.factor_evals`.

### Root Cause Analysis

The expected_output_claim is computed from:
1. Opening claims stored in proof (factor evaluations at r_cycle point)
2. Batching coefficients derived from transcript
3. Weighted sum of all instance claims

The mismatch indicates Zolt's Stage 2 sumcheck proof has:
- Incorrect round polynomials, OR
- Incorrect opening claims for the verifier's computation, OR
- Transcript divergence causing different batching coefficients

### Next Steps (Priority Order)

1. [ ] Add debug prints in Jolt to show the 8 factor claims it retrieves from the proof
2. [ ] Compare with what Zolt stores in `factor_evals[0..7]`
3. [ ] Verify the MLE evaluations at r_cycle are correct in Zolt
4. [ ] If factor claims match, investigate transcript state divergence

### How to Run Tests

```bash
# Run Jolt verification test
ZOLT_LOGS_DIR=/home/vivado/projects/zolt/logs cargo test --features "minimal" --no-default-features -p jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --nocapture --ignored

# Run Zolt proof generation
cd /home/vivado/projects/zolt
zig build && ./zig-out/bin/zolt prove examples/fibonacci.elf --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin
```

### Technical Details
- trace_length: 256 (padded from 54 actual cycles)
- n_cycle_vars: 8
- log_ram_k: 16
- Stage 2: 24 rounds
- Proof file: `/home/vivado/projects/zolt/logs/zolt_proof_dory.bin` (40544 bytes)
- Preprocessing file: `/home/vivado/projects/zolt/logs/zolt_preprocessing.bin` (26356 bytes)

---

## Test Status

- ✅ 714/714 unit tests passing
- ✅ Proof serialization/deserialization working
- ✅ Stage 1 verification passing
- ❌ Stage 2 verification failing

---

## Previous Sessions

### Session 75 (2026-01-29)
- Verified challenge type mapping correct
- Verified factor polynomial order matches `PRODUCT_UNIQUE_FACTOR_VIRTUALS`

### Session 74 (2026-01-29)
- Verified Zolt's prover is internally consistent
- SumcheckId enum has 22 values matching Jolt

### Session 73 (2026-01-29)
- Fixed SumcheckId mismatch
- Fixed proof serialization format
- Proof deserializes completely
