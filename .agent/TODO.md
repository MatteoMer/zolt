# Zolt-Jolt Compatibility TODO

## Current Status: Session 58 - January 6, 2026

**STATUS: SECOND_GROUP implemented, still debugging verification mismatch**

### Changes Made

1. Added `evaluateAzBzAtDomainPointForGroup` with group parameter
2. Added `buildEqTable` helper for factored E_out * E_in computation
3. Updated `computeFirstRoundPoly` to iterate over both groups
4. Added `full_tau` field to store tau for UniSkip
5. Fixed eq_table structure to use factored E_out * E_in (dropping tau_high)

### Still Failing

Sumcheck verification still fails:
- output_claim: 3156099394088378331739429618582031493604140997965859776862374574205175751175
- expected_output_claim: 6520563849248945342410334176740245598125896542821607373002483479060307387386

### Previous Root Cause (Session 57)

**Zolt's UniSkip (`computeFirstRoundPoly`) only evaluated FIRST_GROUP constraints, but Jolt evaluates BOTH groups.**

### The Inconsistency

1. **UniSkip** (`computeFirstRoundPoly`):
   - Loops over `cycle` from 0 to padded_trace_len (1024)
   - Calls `evaluateAzBzAtDomainPoint(witness, domain_idx)`
   - This function **only uses FIRST_GROUP_INDICES**
   - Missing the SECOND_GROUP contribution entirely

2. **Remaining rounds** (`az_poly` construction):
   - Lines 285-326 compute BOTH groups:
     - `az_evals[base_idx + j] = az0` (FIRST_GROUP)
     - `az_evals[base_idx + j + 1] = az1` (SECOND_GROUP)
   - `t_prime` is built from `az_poly` with both groups interleaved

3. **Result**:
   - `uni_skip_claim` = sum over FIRST_GROUP only
   - `t_prime[0]` = sum with group=0, `t_prime[1]` = sum with group=1
   - The Gruen constraint cannot be satisfied because they represent different sums

### Jolt's UniSkip (for reference)

From `jolt-core/src/zkvm/spartan/outer.rs:196-208`:
```rust
let is_group1 = (x_in & 1) == 1;
for j in 0..OUTER_UNIVARIATE_SKIP_DEGREE {
    let prod_s192 = if !is_group1 {
        eval.extended_azbz_product_first_group(j)
    } else {
        eval.extended_azbz_product_second_group(j)
    };
    inner[j].fmadd(&e_in, &prod_s192);
}
```

Jolt iterates over all `x_in` values where:
- Even `x_in`: evaluates FIRST_GROUP
- Odd `x_in`: evaluates SECOND_GROUP

### The Fix

Update `computeFirstRoundPoly` to match Jolt:

```zig
// Current (WRONG):
for (0..padded_trace_len) |cycle| {
    const eq_val = eq_table[cycle];
    const az_bz = evaluateAzBzAtDomainPoint(witness, domain_idx);  // FIRST_GROUP only
    sum += eq_val * az_bz;
}

// Fixed (sum over both groups):
for (0..padded_trace_len) |cycle| {
    for ([_]u1{0, 1}) |group| {
        const g = (cycle << 1) | group;  // 11-bit index
        const eq_val = eq_table[g];
        const az_bz = evaluateAzBzAtDomainPoint(witness, domain_idx, group);
        sum += eq_val * az_bz;
    }
}
```

And update `evaluateAzBzAtDomainPoint` to take a `group` parameter:
- `group=0`: use `FIRST_GROUP_INDICES` (10 constraints)
- `group=1`: use `SECOND_GROUP_INDICES` (9 constraints)

---

## Implementation Tasks

- [ ] Update `evaluateAzBzAtDomainPoint` to accept `group: u1` parameter
- [ ] Handle FIRST_GROUP (10 constraints at Y ∈ {-4,...,5}) and SECOND_GROUP (9 constraints at Y ∈ {-4,...,4})
- [ ] Update `computeFirstRoundPoly` to loop over (cycle, group) pairs
- [ ] Ensure eq_table indexing uses 11-bit index: `g = (cycle << 1) | group`
- [ ] Test and verify q(1) matches t_prime[1] at first Gruen round
- [ ] Verify final output_claim matches expected_output_claim

---

## Completed Tasks

- [x] Compare R1CS input ordering between Zolt and Jolt (they match!)
- [x] Compare R1CS constraint ordering between Zolt and Jolt (they match!)
- [x] Verify transcript challenges match (they do!)
- [x] Verify r1cs_input_evals match between prover and verifier (they do!)
- [x] Fix batching coefficient Montgomery form bug (Session 53)
- [x] Add debug output showing t_prime values and q(1) mismatch
- [x] Trace Gruen constraint failure to UniSkip vs t_prime inconsistency
- [x] Identify root cause: UniSkip missing SECOND_GROUP evaluation

---

## Test Commands

```bash
# Build and generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove /tmp/jolt-guest-targets/fibonacci-guest-fib/riscv64imac-unknown-none-elf/release/fibonacci-guest --jolt-format --input-hex 32 --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```

---

## Debug Values from Jolt Verifier

```
inner_sum_prod = 3428564744352329898278095955238265070037131657307455691194697055242544749299
az_g0 = 7784823065365355150329898612995661848060669911039813026488804049763027733277
az_g1 = 1819540214425536184764541679660107604875800049998949590309674377395141916961
bz_g0 = 17295306602198664491678763456607120003277220072842593579631007088433375750847
bz_g1 = 7283832083962387142906994059148710843912079143868948223318847447205362507259
```

The verifier blends groups: `az_final = az_g0 + r_stream * (az_g1 - az_g0)`
