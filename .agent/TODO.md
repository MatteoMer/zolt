# Zolt → Jolt Verification Progress

## Current Status
**🔧 IN PROGRESS — Goal: verify Zolt proofs against vanilla (unmodified) Jolt**

The current proof passes against a modified Jolt fork. The project is NOT complete until verification works against an unmodified upstream Jolt.

## Goal
Zolt-generated proofs must verify against a **vanilla Jolt verifier** with zero modifications to Jolt's code. Any incompatibility must be fixed on the Zolt side.

## Resources
- Upstream Jolt: `./jolt` (git submodule)
- Arkworks: `./arkworks` (local copy for reference)

## Current Jolt Modifications That Must Be Eliminated
These are changes currently in our Jolt fork that we need to remove by fixing Zolt instead:

1. **Debug prints**: ~35 files with `eprintln!` behind `#[cfg(feature = "zolt-debug")]`
2. **Test harness**: `examples/fibonacci/src/main.rs` with `--verify-zolt-proof` CLI mode
3. **Test module**: `zolt_compat_test.rs` (behind `#[cfg(test)]`)
4. **New instructions**: ADDIW, ADDW, SLLI - ISA extensions added for RV64I programs

### Analysis Needed
- Which of these modifications are load-bearing for verification vs. just debug/test infrastructure?
- Can the test harness be written as a standalone Rust binary that links against vanilla Jolt as a library?
- Do the new ISA instructions (ADDIW, ADDW, SLLI) affect verification, or only proof generation?

## TODO
- [ ] Inventory all Jolt-side changes and classify as: verification-affecting vs. debug/test-only
- [ ] Build a standalone verifier binary that links against vanilla Jolt (no modifications)
- [ ] Fix any Zolt proof generation issues that arise from removing Jolt modifications
- [ ] Verify Zolt proof against vanilla Jolt — all 8 stages must pass
- [ ] Ensure `zig build test` still passes all tests

## Historical Fixes (for reference)
These are all the fixes that were made to Zolt to get verification working against the modified Jolt fork. They remain relevant context.

### Stage 1: Transcript & Challenge Alignment
- Blake2b transcript initialization matching Jolt's format
- Challenge scalar byte ordering (LE representation)
- Field element serialization in arkworks-compatible format

### Stage 2: Spartan/R1CS Sumcheck
- R1CS witness generation for all instruction types
- Proper handling of instructions without lookup tables (Load, Store, SLL, SLLI)

### Stage 3: Instruction Lookup Sumcheck
- And/Or/Xor prefix MLE shift off-by-one fix
- RAF prefix MLE materialization (tables instead of formula-based evaluation)

### Stage 4: Read-Address-Flag (RAF) Decomposition
- UpperWord prefix formula fix (XLEN-j instead of 2*XLEN-j)

### Stage 5: Bytecode Verification
- BuildBytecodeEntries populated from static ELF bytecode
- Termination store bytecode entry flags alignment
- NoOp bytecode entry is_interleaved flag matching

### Stage 6: Claim Reduction (Registers/RAM)
- Real sumcheck provers for ALL instances (IncClaimReduction, HammingBooleanity)
- Claim tracking with cached round polynomials
- Booleanity Phase 1→Phase 2 transition using consistent eq_cycle table
- NUM_LOOKUP_TABLES count alignment

### Stage 7: Dory Commitment Verification
- Bytecode_K computation from decoded instruction count

### Stage 8: Dory Opening Proof
- Dense polynomial padding to k_chunk*trace_length for DoryGlobals matrix layout
- Dory transcript challenge type: full 128-bit (challengeScalarFull) not 125-bit masked
- DoryVerifierSetup.fromSRS using correct SRS parameters (h2, max_num_vars=20)
- Joint polynomial gamma power ordering: RamInc, RdInc, InstructionRa, BytecodeRa, RamRa

## Test Commands
```bash
# Build Zolt
cd /home/vivado/projects/zolt && zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --trace-length 64 -o /tmp/zolt_proof.bin --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin --srs /tmp/jolt_dory_srs.bin

# Verify with vanilla Jolt (TARGET — not yet working)
# TODO: build standalone verifier against unmodified Jolt

# Verify with modified Jolt (current — to be replaced)
cd /home/vivado/projects/jolt && RAYON_NUM_THREADS=1 cargo run --release --features zolt-debug --manifest-path examples/fibonacci/Cargo.toml -- --verify-zolt-proof /tmp/zolt_proof.bin --zolt-preprocessing /tmp/zolt_preprocessing.bin.ram

# Run Zig tests
cd /home/vivado/projects/zolt && zig build test
```
