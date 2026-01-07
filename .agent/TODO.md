# Zolt-Jolt Compatibility - Stage 2 Progress

## Current Status
- Stage 1: PASSING ✅
- Stage 2: FAILING ❌ (Claim ordering mismatch causing val_final_claim read as 0)
- Stage 3+: Not reached yet
- All 712 Zolt tests pass

## Session 8 Progress

### Memory Layout Fixed ✅
- Fixed `zkvm/jolt_device.zig` to place I/O region BEFORE RAM_START_ADDRESS
- termination address now at 0x7FFFE008 (correct) instead of RAM region
- remapAddress() returns index 1025 (correct) instead of 268M

### OpeningId Ordering Fixed ✅
- Changed order() to compare poly FIRST, then sumcheck_id
- Added VirtualPolynomial.order() and CommittedPolynomial.order()
- Added test for Virtual claim ordering - passes

### val_final_claim Computation ✅
- Passed memory_layout through ConversionConfig
- val_final_claim IS being computed (non-zero value)
- val_final_claim IS being serialized to proof file correctly

### CURRENT ISSUE: Claim Alignment Mismatch

When Jolt deserializes the proof:
- Jolt reads `RamValFinal@RamOutputCheck` claim as 0
- But Zolt serialized a non-zero value at that position

Root cause: Claims appear at different positions in Zolt vs Jolt ordering.

Example:
- Zolt claim 5: `Virtual(NextIsNoop, SpartanProductVirtualization)`
- Jolt expects claim 5: `Virtual(NextIsVirtual, SpartanOuter)`

Both order by `(poly, sumcheck_id)` tuple where poly comes first.
Since NextIsNoop(4) < NextIsVirtual(5), NextIsNoop should come first.
But Jolt's file shows NextIsVirtual at position 5.

This could mean:
1. Jolt generates a different set of claims
2. Jolt has a different ordering for some reason
3. There's additional criteria in Jolt's ordering

### Next Steps

1. Compare FULL claim lists from Jolt-generated proof vs Zolt-generated proof
2. Check if Jolt preprocessing generates a different set of claims
3. Verify BTreeMap serialization order in arkworks

## Previous Session Progress

### Verified Components (ALL MATCH!)
1. ✅ Stage 2 input_claims (all 5 match Jolt)
2. ✅ Stage 2 gamma_rwc and gamma_instr
3. ✅ Batching coefficients
4. ✅ Polynomial coefficients (c0, c2, c3) for all 26 rounds
5. ✅ Sumcheck challenges for all 26 rounds
6. ✅ 8 factor evaluations
7. ✅ Factor claims inserted into proof correctly

## Commits
- `964ec0f`: Fixed OpeningId ordering and memory layout
- `abe09a4`: Fixed input_claims and gamma sampling
- `5033064`: Debug - polynomial coefficients match
- `78a09cf`: Deep dive - termination bit is the issue
- `68db1c2`: WIP structure improvements for OutputSumcheck
