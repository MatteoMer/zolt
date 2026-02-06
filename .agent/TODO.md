# Zolt-Jolt Compatibility Implementation

## Status: Session 91 - Stage 5 RAF prefix-suffix drift investigation

## Current Issue: lookups_claim drifts from materialized_sum over 128 address rounds

### Key Finding (Session 91)
The RAF polynomial evaluations are **internally consistent** (eval_0 + eval_1 = claim at each round), but
they don't match what Jolt expects. After 128 address rounds:
- `materialized_sum` (computed directly from trace) = `e728cffa1af93851e97fbac6cb36aca0`
- `lookups_claim` (evolved through polynomial chain) = `f21f2ce546c92b0f7c9ad5e065cec05a`

### Verified Correct
1. Round 0 RAF evaluation matches brute force ✅
2. Prefix polynomial computation is correct (verified `8 * r[0]` formula) ✅
3. Q array binding formula is correct: `new[j] = Q[j] + r*(Q[j+half] - Q[j])` ✅
4. `proverMsgRaf` result matches explicit sum over (prefix, Q) pairs ✅
5. Claim chain is self-consistent: `eval_0 + eval_1 = before_claim` at each round ✅

### Suspected Issues
1. The **eq weighting** from prior challenges may not be correctly incorporated
2. The brute force doesn't account for eq(k[bound_bits], r[bound_challenges])
3. Something in the **phase transition logic** (`condenseUEvals`) may be wrong
4. The **suffix_len** calculation might affect Q initialization

### Architecture Reminder
- 16 phases, each handling 8 address bits (chunk_len = 8, total_len = 128)
- Q arrays: size 256 at phase start, bound to 128, 64, 32, 16, 8, 4, 2, 1 over 8 rounds
- Prefix polynomial: analytically computed from bound_value and remaining bits
- Suffix: fixed for each phase (suffix_len = 128 - (phase+1)*8)

### Next Steps
1. Add debug at phase transitions to verify `condenseUEvals` is correct
2. Check if the issue is in RAF polynomial only or also in read_checking
3. Compare Jolt's exact Q values at round 0 and round 1 against Zolt's
4. Verify that the `is_interleaved_operands` flag is set correctly

### Results After Fix
- Stage 1: PASSES ✅
- Stage 2: PASSES ✅
- Stage 3: PASSES ✅
- Stage 4: PASSES ✅
- Stage 5: FAILS ❌ (RAF drift over 128 rounds)
- Stages 6-7: Not yet reached

### Build/Test Commands
```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
