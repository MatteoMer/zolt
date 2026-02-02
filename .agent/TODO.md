# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 ra_chunk Mismatch

## Session 126 Summary

### Progress Made

1. **Verified transcript produces correct challenges**
   - Transcript outputs `masked_value=0x15b2ebced7ca0d488e1f5913aabdd05a` which matches Jolt's r[0]
   - The challenge stored has limbs `[0, 0, 0x8e1f5913aabdd05a, 0x15b2ebced7ca0d48]`
   - Debug output shows `toBytesBE()` converts from Montgomery form correctly
   - **The arithmetic representation is correct - both use MontU128Challenge format**

2. **Understood ra_claim computation flow**
   - Jolt: `ra_polys` initialized with `eq_evals[x] = eq(x, r_address_chunk)`, then bound over cycle rounds
   - Zolt: `ra_chunk_weights` computed bit-by-bit during address rounds, then bound over cycle rounds
   - Both should produce the same result mathematically

3. **Found key difference: challenge ordering in Jolt**
   - Jolt's `normalize_opening_point()` REVERSES the cycle challenges:
     ```rust
     let r_cycle_prime = r_cycle_prime.iter().copied().rev().collect::<Vec<_>>();
     ```
   - This is for converting from LowToHigh binding order to BIG_ENDIAN opening point
   - Need to verify Zolt handles this correctly

### Current Issue

**ra_chunk values don't match between Zolt and Jolt**

```
Zolt ra_chunks[0] = 72c54fffc84783cff5628ecc74b37775
Jolt ra_claims[0] = 12109e5de8bae83db5b8fca2612309a8 (LE)
```

### Key Insight About ra_claim

The `ra_claim[i]` written by the prover is:
```
Σ_j eq(j, r_cycle_prime) × eq(lookup_index_chunk_i[j], r_address_chunk_i)
```

Where:
- `r_cycle_prime` = reversed cycle challenges (from sumcheck rounds 128-135)
- `r_address_chunk_i` = challenges for chunk i (rounds i*16 to (i+1)*16-1)
- `lookup_index_chunk_i[j]` = bits [128-16*i-1 : 128-16*(i+1)] of `lookup_index[j]`

### Possible Root Causes

1. **Challenge ordering mismatch during cycle binding**
   - Jolt binds ra_polys with LowToHigh order but stores opening with reversed cycle challenges
   - Zolt may not be reversing or may be reversing in wrong place

2. **Bit extraction order for lookup_index chunks**
   - Zolt extracts MSB first during address rounds
   - Need to verify this matches how Jolt computes `lookup_index_chunk`

3. **eq polynomial evaluation order**
   - Jolt uses `EqPolynomial::evals(&r_address_chunk)` which uses specific bit ordering
   - Zolt computes eq bit-by-bit which should be equivalent but may differ

### Next Steps

1. **Add detailed debug to compare Zolt and Jolt challenge sequences**
   - Print all 128 address challenges from Zolt
   - Compare with Jolt's r_address_prime

2. **Verify lookup_index_chunk extraction**
   - Check that Zolt's bit extraction matches Jolt's `lookup_index_chunk()`
   - For chunk 0: should be bits [127:112] = high 16 bits
   - For chunk 7: should be bits [15:0] = low 16 bits

3. **Test eq polynomial equivalence**
   - Compute eq(index_chunk, r_chunk) directly in Zolt
   - Compare with bit-by-bit accumulated value

### Key Files

**Zolt:**
- `src/zkvm/spartan/stage5_prover.zig` - Stage 5 batched sumcheck prover
- `src/zkvm/proof_converter.zig` - Proof generation orchestration

**Jolt:**
- `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` - InstructionReadRaf sumcheck
- `jolt-core/src/zkvm/instruction_lookups/ra_virtual.rs` - RaPolynomial and virtualization
- `jolt-core/src/zkvm/config.rs` - OneHotParams::compute_r_address_chunks

### Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin

# Copy to /tmp for Jolt test
cp logs/zolt_*.bin /tmp/

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Architecture Understanding

### Stage 5 (LookupsReadRaf) Sumcheck

The sumcheck proves:
```
Σ_j eq(j, r_cycle) × ra(j) × (combined_val(j) + γ×raf_val(j))
```

Where:
- j ranges over cycles [0, T)
- `ra(j) = Π_{chunk} eq(lookup_index_chunk[j], r_address_chunk)`
- `combined_val(j)` = table value + RAF contribution
- `raf_val(j)` = left + γ×right or γ×identity

### Address Rounds (0-127)

During address rounds, the prover uses prefix-suffix decomposition to compute:
- read_checking: Σ_j eq × ra × table_value
- raf: Σ_j eq × ra × raf_value

The `ra` is implicitly computed through the decomposition.

### Cycle Rounds (128-135)

During cycle rounds, the `ra_polys` (or `ra_chunk_weights`) are explicitly bound with cycle challenges.

After all rounds, the final claims are:
- `ra_chunks[i]`: The bound ra polynomial value
- `table_flag[t]`: Which table was used
- `raf_flag`: RAF path indicator
