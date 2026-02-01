# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Prefix-Suffix Integration Needed

## Session 96 Summary

### Infrastructure Complete

Created the prefix-suffix decomposition infrastructure required for Jolt compatibility:

1. **prefixes.zig** - Implements all 45+ prefix types from Jolt:
   - `Prefixes` enum with all prefix variants
   - `PrefixCheckpoint` and `PrefixCheckpoints` types
   - `LookupBits` for bit manipulation
   - `prefixMle()` and `updatePrefixCheckpoint()` functions
   - Implemented: Eq, LowerWord, UpperWord, And, Or, Xor, LessThan,
     LeftOperandIsZero, RightOperandIsZero, LeftOperandMsb, RightOperandMsb
   - Remaining prefixes return zero (placeholder)

2. **identity_poly.zig** - Implements RAF operand polynomials:
   - `IdentityPolynomial`: Evaluates to binary index `Σ r_i * 2^(n-1-i)`
   - `OperandPolynomial`: Left/Right operand from interleaved bits
   - `BindingOrder` enum (LowToHigh, HighToLow)
   - `sumcheckEvals()` for both polynomial types

3. **mod.zig** - Exports new modules and types

### Key Technical Insight

The critical difference between Jolt and Zolt's current approach:

**Jolt (address rounds):**
```rust
// Compute [eval_0, eval_2] via prefix-suffix decomposition
let eval_at_0 = read_checking[0] + raf[0];
let eval_at_2 = read_checking[1] + raf[1];
// Create degree-2 polynomial
UniPoly::from_evals_and_hint(previous_claim, &[eval_at_0, eval_at_2])
```

**Zolt (current, address rounds):**
```zig
// Simple bit-splitting produces degree-1
if (bit == 0) p0 += contrib;
else p1 += contrib;
// Results in linear polynomial p(X) = p0 + X*(p1-p0)
```

The fundamental change needed: Use prefix-suffix decomposition to compute
`[eval_0, eval_2]` and then use `fromEvalsAndHint(previous_claim, eval_0, eval_2)`
to produce a degree-2 polynomial matching Jolt's format.

### Remaining Implementation

To complete Stage 5 prefix-suffix integration:

1. **Initialize Suffix Polynomials** (per phase):
   - For each lookup table, build suffix accumulators
   - Suffix polynomial size = 2^(log_m) where log_m = LOG_K / phases
   - Accumulate: suffix_poly[idx] += u_eval[j] * suffix_mle(suffix_bits)

2. **Initialize RAF Decompositions**:
   - `identity_ps`: PrefixSuffixDecomposition for identity polynomial
   - `left_operand_ps`: PrefixSuffixDecomposition for left operand
   - `right_operand_ps`: PrefixSuffixDecomposition for right operand
   - Each needs Q accumulators initialized from u_evals and lookup indices

3. **Compute Address Round Message**:
   ```zig
   fn computePrefixSuffixProverMessage(round: usize, previous_claim: F) UniPoly {
       // Read-checking: Σ over tables of P(c) * Q
       const read_checking = proverMsgReadChecking(round);
       // RAF: γ*left + γ²*(right + identity)
       const raf = proverMsgRaf();

       const eval_0 = read_checking[0] + raf[0];
       const eval_2 = read_checking[1] + raf[1];

       return UniPoly.fromEvalsAndHint(previous_claim, eval_0, eval_2);
   }
   ```

4. **Bind and Update State After Each Round**:
   - Bind challenge in suffix polynomials
   - Bind challenge in RAF decompositions (identity_ps, left/right_operand_ps)
   - Update prefix checkpoints every 2 rounds
   - Initialize next phase when current phase completes

5. **Transition to Cycle Rounds (after round 127)**:
   - Call `init_log_t_rounds(gamma, gamma_sqr)` to materialize:
     - `ra_polys`: Product of expanding table values
     - `combined_val_polynomial`: table_mle(r_addr) + raf_val
   - These are then used for the final 8 cycle rounds (existing code)

### Files to Modify

- `src/zkvm/spartan/stage5_prover.zig`:
  - Import prefixes and identity_poly
  - Add state for prefix_checkpoints, suffix_polys, v (expanding tables)
  - Add identity_ps, left_operand_ps, right_operand_ps
  - Replace address round bit-splitting with prefix-suffix computation
  - Add phase management (init_phase, checkpoint updates)

### Estimated Remaining Effort

- Suffix polynomial initialization: 2-3 hours
- RAF decomposition integration: 2-3 hours
- Prover message computation: 2-3 hours
- Phase management and binding: 2-3 hours
- Testing and debugging: 4-6 hours
- **Total: ~2-3 days of focused work**

### Test Commands

```bash
# Build
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Current Verification Status

```
Sumcheck verification failed!
  output_claim:   [d9, 50, 6a, 6e, 69, 84, 32, f8, ...]
  expected_claim: [bb, 2a, d3, 8c, 2c, 8c, 44, d3, ...]
```

- Stages 1-4: PASS
- Stage 5: FAIL (polynomial degree mismatch - need prefix-suffix for degree-2)

### Commits This Session

1. `cef73b6` - feat: implement Jolt-compatible prefix-suffix decomposition
   - Created prefixes.zig with prefix MLE implementations
   - Created identity_poly.zig with Identity/Operand polynomials
   - Updated mod.zig with exports

---

## Session 97 Progress

### Deep Analysis Complete

Investigated the fundamental difference between Jolt's prefix-suffix decomposition and Zolt's bit-splitting approach.

**Key Finding:** The mismatch is NOT just about polynomial degree format, but about how values are computed:

1. **Jolt computes**: `eval_0 = Σ_tables Σ_b P(c=0, b) * Q[b]` where Q = Σ u_eval[j] * suffix_mle(suffix_bits[j])
2. **Zolt computes**: `eval_0 = Σ_{j: bit=0} eq[j] * ra[j] * combined[j]` where combined = concrete output + γ*operands

The suffix_mle is the critical missing piece - it evaluates the **table-specific MLE** on the suffix bits, not the concrete value.

### Implementation Strategy

**Phase 1: Suffix MLE Implementation**
- Implement `suffix_mle()` for each suffix type in Jolt
- Map cycles to tables using Jolt's table index ordering (0-40)

**Phase 2: Q Polynomial Initialization**
- For each table and suffix type: `Q[table][suffix][prefix] = Σ u[j] * suffix_mle(suffix_bits[j])`
- u_evals[j] = eq(r_reduction, j) already computed

**Phase 3: Address Round Computation**
- Compute eval_0 and eval_2 via prefix-suffix decomposition
- Use table.combine(prefixes, suffixes) for each table

**Phase 4: RAF Integration**
- Add RAF contribution via identity/operand prefix-suffix decompositions

### Immediate Next Steps

1. Implement suffix_mle for common suffix types (One, And, Or, Xor, LessThan)
2. Create suffix enum matching Jolt's Suffixes
3. Define which suffixes each table uses
4. Build Q polynomial initialization loop
5. Test single table (AND) end-to-end

### Files Reference

Key Jolt files for suffix implementation:
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/lookup_table/suffixes/mod.rs` - Suffixes enum
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/lookup_table/suffixes/and.rs` - AND suffix_mle
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/lookup_table/and.rs` - AND table suffixes() and combine()

### Notes

Updated .agent/NOTES.md with detailed analysis of the mathematical difference between approaches.

### Progress This Session

1. ✅ Created `suffixes.zig` with all 43 suffix types from Jolt
2. ✅ Implemented suffix_mle for key suffixes (One, And, Or, Xor, LessThan, etc.)
3. ✅ Added tableSuffixes() mapping tables to their suffix configurations
4. ✅ Exported suffixes module from mod.zig
5. ✅ Committed and pushed: `b8239f1`

### Next Immediate Steps

1. Add suffix polynomial Q structures to Stage 5 prover state
2. Initialize Q polynomials from u_evals and lookup_indices
3. Implement prover_msg_read_checking using P*Q products
4. Test with a simple trace to verify polynomial values

### Technical Note on Q Initialization

The Q polynomial for each table and suffix is initialized as:
```zig
// For each cycle j with lookup index k[j]:
//   (prefix_bits, suffix_bits) = split(k[j], suffix_len)
//   Q[table][suffix][prefix_bits] += u_eval[j] * suffix_mle(suffix_bits)
```

Where:
- `suffix_len = LOG_K - phase * log_m` (varies by phase)
- `log_m = LOG_K / phases` (typically phases=8, so log_m=16)
- `u_eval[j] = eq(r_reduction, j)` (already computed as lookups_eq_evals)

### Session 97 Final Status

**Completed:**
1. ✅ suffixes.zig - All 43 suffix types with suffix_mle implementations
2. ✅ prefix_suffix_prover.zig - Q polynomial structures and proverMsgReadChecking
3. ✅ Updated mod.zig with exports
4. ✅ Added imports to stage5_prover.zig
5. ✅ Initialized Q polynomials in Stage 5 prover (lookup_indices_u128, suffix_polys.initPhase)

**Commits:**
- `b8239f1` - feat: implement suffix MLE functions for prefix-suffix decomposition
- `7cdfc90` - docs: update TODO with Session 97 progress
- `222a5f4` - feat: implement prefix-suffix prover state and Q polynomial structures
- `43acde6` - chore: add prefix-suffix prover imports and integration plan
- `639eb1f` - docs: update TODO with Session 97 final status
- `fcc19e8` - feat: initialize suffix polynomials in Stage 5 prover

**Next Session:**
1. Replace address round bit-splitting with proverMsgReadChecking calls (~lines 1270-1285)
2. Add suffix_polys.bindAll(challenge) and prefix checkpoint updates after each round
3. Add phase transitions (every 16 rounds for 8 phases)
4. Add RAF contribution via identity/operand prefix-suffix decomposition
5. Test end-to-end with Jolt verification

### Integration Plan (for next session)

To integrate prefix-suffix into Stage 5 `generateStage5ProofWithTrace`:

**Step 1: Initialize (after line ~945, where cycle_table_indices is populated):**
```zig
// Import is already added at top of file
// Initialize suffix polynomials and prefix checkpoints
var suffix_polys = AllSuffixPolys(F).init(self.allocator);
defer suffix_polys.deinit();

var prefix_checkpoints = PrefixCheckpointsState(F).init();

// Build lookup_indices as u128 array
var lookup_indices_u128 = try self.allocator.alloc(u128, T);
defer self.allocator.free(lookup_indices_u128);
for (0..T) |j| {
    lookup_indices_u128[j] = (@as(u128, lookups_indices_hi[j]) << 64) | lookups_indices_lo[j];
}

// Initialize Q polynomials for phase 0
const num_phases = 8;
try suffix_polys.initPhase(0, num_phases, lookups_eq_evals, lookup_indices_u128, cycle_table_indices);
```

**Step 2: Replace address round computation (lines 1240-1255):**
```zig
// Compute r_x for odd rounds (previous challenge)
const r_x: ?F = if (round % 2 == 1) challenges[round - 1] else null;

// Use prefix-suffix decomposition
const read_checking = proverMsgReadChecking(F, round, &suffix_polys, &prefix_checkpoints, r_x);

// TODO: Add RAF contribution (identity/operand prefix-suffix)
const eval_0_inst2 = read_checking[0];
const eval_2_inst2 = read_checking[1];
```

**Step 3: After address round update (around line 1320, before `continue`):**
```zig
// Bind challenge in suffix polynomials
suffix_polys.bindAll(challenge);

// Update prefix checkpoints every 2 rounds
if (round % 2 == 1) {
    const suffix_len = LOOKUPS_LOG_K - (round / 2 + 1);
    prefix_checkpoints.update(challenges[round - 1], challenge, round, suffix_len);
}

// Phase transition every log_m rounds (log_m = 16 for 8 phases)
const log_m = LOOKUPS_LOG_K / num_phases; // = 16
const current_phase = round / log_m;
const next_phase = (round + 1) / log_m;
if (next_phase > current_phase and next_phase < num_phases) {
    try suffix_polys.initPhase(next_phase, num_phases, lookups_eq_evals, lookup_indices_u128, cycle_table_indices);
}
```

**Note:** The RAF contribution (γ*left + γ²*right) also needs prefix-suffix decomposition via OperandPolynomial. This is a separate integration step.
