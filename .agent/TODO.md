# Zolt-Jolt Compatibility - Current Status

## Summary

**Stage 1**: PASSES ✓
**Stage 2**: OutputSumcheck now passes (panic/termination bits fixed)
**Proof Deserialization**: FAILS - "invalid data" error

## Progress This Session

### 1. Fixed OutputSumcheck Zero-Check (DONE)

Root cause was that the fibonacci.elf was linked at wrong address (0x7FFFF000 in IO region instead of 0x80000000 in RAM).

Fixed by:
1. Rebuilding fibonacci.elf with correct linker script (`-T jolt.ld` with text at 0x80000000)
2. Setting panic/termination bits in val_final to match Jolt's gen_ram_final_memory_state behavior

### 2. Proof Deserialization Error (CURRENT ISSUE)

The Jolt verifier test now fails with:
```
Failed to deserialize proof: the input buffer contained invalid data
```

This is an arkworks serialization error, suggesting one of the Dory group elements or field elements is malformed in the proof file.

### Next Investigation Steps

1. [ ] Compare proof structure between old working proof and new proof
2. [ ] Check if Dory commitment points are valid BN254 curve points
3. [ ] Verify field elements are correctly serialized (LE Montgomery form)
4. [ ] Check if the proof size changed unexpectedly

### Potential Causes

1. **Different trace length** - old ELF ran much longer (wrong address), new ELF is smaller
2. **Proof structure mismatch** - different number of commitments/claims
3. **Serialization bug** - Dory points might be malformed

## Testing Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Run Jolt verification
cd /Users/matteo/projects/jolt/jolt-core && cargo test test_verify_zolt_proof_with_zolt_preprocessing --release -- --ignored --nocapture
```

## Commits Made

1. 60c1ad1 - debug: Found OutputSumcheck zero-check failure root cause
2. b74cb76 - fix: Set panic/termination bits in val_final + correct ELF base address
