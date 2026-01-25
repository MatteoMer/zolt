# Serialization Bug: Root Cause Identified

## Date: 2026-01-25

## Summary

The Stage 4 cross-verification failure is caused by **two critical serialization mismatches** in the Dory commitments:

1. ✅ **Wrong commitment size**: Writing 384 bytes (uncompressed) instead of ~288 bytes (compressed)
2. ✅ **Wrong number of commitments**: Writing 5 instead of ~37

## Evidence

### Issue #1: Uncompressed GT Elements

**File**: `src/zkvm/mod.zig:1360-1363`
```zig
// Write the pre-computed Dory commitments (GT elements, 384 bytes each)
try serializer.writeUsize(5);
for (bundle.dory_commitments) |comm| {
    try serializer.writeGT(comm);  // ← Writes 384 bytes per commitment
}
```

**File**: `src/zkvm/jolt_serialization.zig:109-112`
```zig
pub fn writeGT(self: *Self, gt: GT) !void {
    const gt_bytes = gt.toBytes();  // ← Returns [384]u8
    try self.buffer.appendSlice(self.allocator, &gt_bytes);
}
```

**File**: `src/field/pairing.zig:635`
```zig
pub fn toBytes(self: Fp12) [384]u8 {
    // 12 Fp elements × 32 bytes = 384 bytes (UNCOMPRESSED)
}
```

**Jolt expectation**: Compressed GT elements (~288 bytes each) per arkworks `CanonicalSerialize` compressed format.

**Impact**:
- Zolt writes: 8 + (5 × 384) = 1928 bytes
- Jolt expects: 8 + (5 × 288) = 1448 bytes
- **Offset mismatch: 480 bytes!**

### Issue #2: Wrong Number of Commitments

**Jolt expectation**: `2 + instruction_d + ram_d + bytecode_d` commitments

For typical parameters:
- `instruction_d = 32` (from LOG_K=128, log_k_chunk=4)
- `ram_d = 2-5` (varies by program)
- `bytecode_d = 1-2` (varies by program)
- **Total: ~37 commitments**

**Zolt is writing**: 5 commitments

**Impact**:
- Even if we fix the compression, we're writing 32 fewer commitments!
- This causes massive deserialization offset errors

### Combined Impact

When Jolt tries to deserialize Stage 4 sumcheck proof:
1. It skips past opening_claims (correct)
2. It tries to read the commitments Vec:
   - Reads length: expects ~37, gets 5
   - Reads 37 GT elements of ~288 bytes each: gets only 5 × 384 bytes
   - **Total mismatch: Missing ~9,000 bytes of commitment data!**
3. When it tries to read Stage 1 UniSkip, it's reading from the wrong offset
4. When it tries to read Stage 4 sumcheck, it's actually reading Stage 1 data

## The Fix

### Step 1: Implement GT Compression

Add a `toBytesCompressed()` method to Fp12/GT that produces ~288-byte compressed format matching arkworks.

### Step 2: Fix Commitment Count

The `bundle.dory_commitments` array should contain **all** committed polynomials:
- RdInc
- RamInc
- InstructionRa[0..31]
- RamRa[0..ram_d-1]
- BytecodeRa[0..bytecode_d-1]

### Step 3: Update Serialization

```zig
// Write the correct number of commitments in compressed format
try serializer.writeUsize(all_commitments.len);  // Should be ~37
for (all_commitments) |comm| {
    try serializer.writeGTCompressed(comm);  // Use compressed format
}
```

## Files to Modify

1. `src/field/pairing.zig`:
   - Add `toBytesCompressed()` to Fp12
   - Implement arkworks-compatible GT compression

2. `src/zkvm/jolt_serialization.zig`:
   - Add `writeGTCompressed()` method
   - Update commitment serialization to use compressed format

3. `src/zkvm/proof_converter.zig`:
   - Ensure all committed polynomials are being tracked
   - Generate commitments for all: RdInc, RamInc, InstructionRa[], RamRa[], BytecodeRa[]

4. `src/zkvm/mod.zig`:
   - Update `serializeJoltProofWithDory()` to write all commitments
   - Ensure bundle.dory_commitments contains all ~37 commitments

## References

- Jolt commitment structure: `/Users/matteo/projects/jolt/jolt-core/src/zkvm/witness.rs:41-53`
- Jolt proof structure: `/Users/matteo/projects/jolt/jolt-core/src/zkvm/proof_serialization.rs:28-54`
- Arkworks serialization: `/Users/matteo/projects/jolt/jolt-core/src/zkvm/proof_serialization.rs:470` (uses compressed format)
