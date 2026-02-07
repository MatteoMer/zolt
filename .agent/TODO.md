# Zolt-Jolt Compatibility Implementation

## Status: PREFIX SHIFT FIX APPLIED - DEBUGGING POLYNOMIAL CHAIN DIVERGENCE

### Root Cause Analysis Progress

#### CONFIRMED FACTS:
1. Proof coefficients match between Zolt prover and Jolt verifier (S5P==S5V at every round) ✅
2. Challenges match between prover and verifier ✅
3. Initial claim matches ✅
4. p(0)+p(1) = claim at every round ✅
5. CONSISTENCY (batch0*inst0 + batch1*inst1 + batch2*inst2 == batched) at all rounds ✅
6. mulHiBigIntU128 is equivalent to full montgomeryMul when operand has zero low limbs ✅
7. All code formulas (tableCombine, prefixMle, suffixMle, bind, uninterleave) match Jolt ✅
8. Expanding tables match direct EQ computation at phase boundaries ✅
9. AND, XOR, OR table values from prefix-suffix NOW MATCH direct MLE ✅ (after shift fix)
10. Polynomial IS degree 2 for prefix-suffix decomposition (product of linear prefix × linear Q) ✅

#### PREFIX SHIFT FIX (APPLIED):
- Bug: And/Or/Xor prefix MLE functions used `XLEN - (j/2)` on odd rounds instead of `XLEN - 1 - (j/2)`
- Same bug in andUpdateCheckpoint, orUpdateCheckpoint, xorUpdateCheckpoint
- Fix: Changed all 6 instances to use `XLEN - 1 - (j/2)` matching Jolt exactly
- Result: AND/OR/XOR table MLE values now match, but doesn't affect fibonacci (those tables unused)

#### CURRENT BUG:
- The polynomial chain still diverges from brute-force sum at phase 0→1 transition
- Expected condensed sum: `8f7186063fc35b718dbaa749c92ff73a`
- Poly chain gives: `53c2cdc63981b1fad8e79cc4ee122673`
- At materialization (round 128): materialized sum ≠ lookups_claim (poly chain)
- The polynomial IS degree 2 (correct for prefix-suffix), but chain evolves differently

#### KEY INSIGHT:
The polynomial p(c) is degree 2 because it's a sum of products: prefix(c)*Q(c) where both are linear.
This means p(2) ≠ 2*p(1) - p(0) in general. The "MULTILINEAR BUG" diagnostic was WRONG.
The degree-2 polynomial is handled correctly by from_evals_and_hint.

The REAL issue is that the degree-2 extrapolation (eval_2) computed by the prefix-suffix decomposition
doesn't match what the Jolt prover would compute. This causes the polynomial chain to evolve
differently, leading to a mismatch between the chain's output_claim and the expected_claim from openings.

#### WHAT TO INVESTIGATE NEXT:
1. The prefix-suffix computes p(2) = sum_b prefix(2,b) * (2*Q_right(b) - Q_left(b))
2. This should equal the true p(2) for a multilinear polynomial
3. But p(c) = sum_b prefix(c,b) * (Q_left(b)*(1-c) + Q_right(b)*c) is degree 2 in c
   because prefix is linear and Q interpolation is linear → product = degree 2
4. For the sum, p(2) = 2*p(1) - p(0) + 2*sum_b (P1-P0)*(Qr-Ql)
5. The extra term is the degree-2 contribution

Possible causes of divergence:
1. The Q polynomial initialization may be wrong (Q values don't match Jolt's Q values)
2. The RAF decomposition may compute wrong evaluations
3. There may be a subtle issue with the GruenSplitEq interaction in Jolt that we're missing
4. The condensation at phase boundaries may not correctly track the polynomial chain

### Next Steps
1. [IN PROGRESS] Compare Q array initialization with Jolt's initialization for specific cycles
2. [PENDING] Add RangeCheck/NotEqual table verification (tables actually used by fibonacci)
3. [PENDING] Compare RAF evaluations between Jolt and Zolt for round 0
4. [PENDING] Fix whatever data/algorithm difference is found

### Test Commands
```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --trace-length 64 -o /tmp/zolt_proof.bin --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin
cd /home/vivado/projects/jolt && cargo run --release --features zolt-debug --manifest-path examples/fibonacci/Cargo.toml -- --verify-zolt-proof /tmp/zolt_proof.bin --zolt-preprocessing /tmp/zolt_preprocessing.bin.ram
```

### Remaining Tasks
1. [IN PROGRESS] Fix polynomial chain divergence in stage 5
2. [PENDING] Verify end-to-end proof generation (all stages pass)
3. [PENDING] Run full test suite (578+ tests)
