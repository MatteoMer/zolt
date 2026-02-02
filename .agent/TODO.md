# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Polynomial Degree Mismatch

## Session 126 Summary (Final)

### ROOT CAUSE IDENTIFIED: Polynomial Degree Mismatch

**Critical Finding:** Zolt's Stage 5 produces degree-2 polynomials for address rounds, but Jolt expects degree-3.

**Evidence:**
1. Jolt's Stage 5 Round 0 debug shows: `Round 0 (degree 3):`
2. Zolt's code explicitly uses degree-2: `// This produces [c0, c2, 0] for degree-2 polynomial`
3. The polynomial coefficients don't match:
   - Jolt coeff[0]: `d5d3057cdadc59f4...`
   - Zolt c0: `8693777e9c24e8b0...`

**Why degree matters:**
- The sumcheck protocol requires prover and verifier to agree on polynomial degrees
- Polynomial coefficients are appended to transcript to derive challenges
- Different coefficients → different transcript state → different challenges
- Different challenges → verification failure

### Stage 5 Architecture

The batched sumcheck in Stage 5 combines 3 instances:
1. **Instance 0: RegistersValEvaluation** - degree depends on polynomial structure
2. **Instance 1: RamRaClaimReduction** - degree depends on phase
3. **Instance 2: InstructionReadRaf** - degree varies (2 for address, 10 for cycle)

The batched polynomial degree = `max(instance_degrees)` for each round.

**For address rounds (0-127):**
- Instance 0: degree 3 (product of 3 linear polynomials: inc, wa, lt)
- Instance 1: degree 2-3 (depends on phase)
- Instance 2: degree 2 (prefix-suffix decomposition)
- **Batched degree: 3**

**For cycle rounds (128-135):**
- Instance 0: degree 3
- Instance 1: degree 2-3
- Instance 2: degree 10 (product of 10 factors: 8 ra_chunks + eq + combined_val)
- **Batched degree: 10**

### What Verified Working

1. **Initial claim matches:** ✓
   ```
   Jolt: 990578d0e96c66a0a2c80e472d900a1cb2dac0db537eaae82e1b48bc1760fb00
   Zolt: 990578d0e96c66a0a2c80e472d900a1cb2dac0db537eaae82e1b48bc1760fb00
   ```

2. **Gamma matches:** ✓
   ```
   Both: 5ab9a012f2c4742080476c8d0fc0accb
   ```

3. **Transcript state at Stage 5 start:** ✓

4. **Polynomial coefficients DON'T match:** ✗
   - This causes transcript divergence after round 0
   - All subsequent challenges are wrong, including those used for ra_chunk

### Fix Required

Zolt's Stage 5 address rounds must use degree-3 polynomials:
1. Compute eval_0, eval_1, eval_2, eval_3 (or eval_inf) instead of just eval_0, eval_2
2. Use the correct Toom-Cook format `[p(0), p(1), p(2), p_inf]`
3. Convert to compressed format `[c0, c2, c3]` for degree-3

### Key Code Location

**Zolt:** `src/zkvm/spartan/stage5_prover.zig` lines 2080-2095

```zig
// Current (WRONG):
// This produces [c0, c2, 0] for degree-2 polynomial
const uni_poly = UniPoly(F).fromEvalsAndHint(current_batched_claim, eval_0, eval_2);

// Need to change to degree-3 by using all four Toom-Cook evals
// and computing c3 properly
```

### Next Steps

1. **Fix degree-3 polynomial computation for address rounds**
   - Modify `stage5_prover.zig` to compute full Toom-Cook evals [p(0), p(1), p(2), p_inf]
   - Update polynomial combination logic for all three instances
   - Ensure Instance 0 (RegistersValEvaluation) contributes degree-3 terms

2. **Verify each instance's degree computation**
   - Instance 0: Must produce degree-3 (product of inc, wa, lt)
   - Instance 1: Verify RamRaClaimReduction produces correct degree
   - Instance 2: Keep degree-2 for prefix-suffix

3. **Test incrementally**
   - First fix round 0 coefficients to match Jolt
   - Verify transcript state matches after round 0
   - Continue until all 136 rounds pass

### Key Files

**Zolt:**
- `src/zkvm/spartan/stage5_prover.zig` - Stage 5 batched sumcheck prover

**Jolt:**
- `jolt-core/src/subprotocols/sumcheck.rs` - Batched sumcheck implementation
- `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` - InstructionReadRaf

### Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof with debug
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin 2>&1 | grep "STAGE5 COEFF"

# Copy and verify
cp logs/zolt_*.bin /tmp/
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

SESSION_ENDING: Context is getting long. The root cause is identified - polynomial degree mismatch in Stage 5 address rounds. Next session should fix the degree computation in `stage5_prover.zig` to use degree-3 polynomials for address rounds.
