# Zolt-Jolt Compatibility TODO

## Current Status: Stage 1 Sumcheck Verification (Session 30)

### Latest Findings
Fixed tau_low extraction - now passing tau_low (not full tau) to split_eq:
- tau_high = tau[tau.len - 1] (stored separately for Lagrange kernel)
- tau_low = tau[0..tau.len - 1] (passed to split_eq)
- m = tau_low.len / 2 = 5 (for tau_low.len = 11)
- E_out has 32 entries (2^5), E_in has 32 entries (2^5)

Output claim ratio changed from ~3.14 to ~1.52:
```
output_claim:          9012353217108547355521203237420571194773588078319073875630688947404723196510
expected_output_claim: 13679333034475978057044998736602529104372847349794408879776350094438916822983
ratio: ~1.52x
```

### What Was Fixed This Session
- [x] Pass tau_low to split_eq (not full tau)
- [x] Store tau_high separately for first-round Lagrange kernel
- [x] Update proof_converter comment to reflect internal extraction

### Verified Components
- Opening claims are read correctly (LeftInstructionInput, RightInstructionInput, etc.)
- Challenge derivation using Blake2b transcript matches Jolt
- Gruen polynomial construction formula matches Jolt

### Possible Remaining Issues
1. **Eq polynomial evaluation ordering** - current_index decrements from 11 to 0, binding tau_low[10] first
2. **Streaming round accumulation** - group selector and cycle index mapping
3. **r_grid integration** - weights for streaming phase
4. **Split eq table indexing** - big-endian vs little-endian

### Debug Data Analysis
```
tau_high_bound_r0:          10811398959691251178446374398729567517345474364208367822721217673367518413943
tau_bound_r_tail_reversed:  14125962160514979480084655056025740513842614775637540494241851278366681977590
inner_sum_prod:             2309286487192234790818585784375390705895687692935567051482424615198743651405
r0:                         11697823044030274153413191308021134249784459050027548723260376992474506918271
rx_constr[0] (r_stream):    2116934244015962401899708433894043262098824377802461496614220883524162579815
sumcheck_challenges.len:    11
```

### Next Investigation Steps
1. Add debug to print t_zero/t_infinity at each round
2. Compare current_scalar progression with Jolt
3. Verify r_grid values during streaming phase
4. Check if eq polynomial tables have same indexing as Jolt

## Test Commands
```bash
# Zolt tests
zig build test --summary all

# Generate proof
zig build && ./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```

## Completed Milestones
- [x] Blake2b transcript implementation
- [x] Field serialization (Arkworks format)
- [x] UniSkip polynomial generation
- [x] Stage 1 remaining rounds sumcheck
- [x] R1CS constraint definitions
- [x] Split eq polynomial factorization (tau_low)
- [x] Lagrange kernel computation
- [x] Opening claims with MLE evaluation
- [x] Challenge Montgomery form conversion
- [x] Store tau_high separately
