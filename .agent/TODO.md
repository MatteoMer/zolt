# Zolt zkVM Implementation TODO

## Completed (This Session - Iteration 46)

### CLI Stats Command
- [x] Add `zolt stats <proof>` command for detailed proof analysis
  - Shows file format (JSON/Binary) and size
  - Displays commitment status (bytecode, memory, register)
  - Shows R1CS proof information
  - Displays sumcheck stage breakdown (all 6 stages)
  - Shows per-stage statistics (rounds, coefficients, claims)
  - Estimates size breakdown (commitments, stages, overhead)
- [x] Update help message with new command
- [x] Add test for stats command parsing
- [x] Update README with stats command documentation

## Completed (Previous Session - Iteration 45)

### JSON Deserialization for Proof Loading
- [x] Add `JsonProofReader(F)` type for parsing JSON proofs
  - Helper functions for extracting JSON fields (getString, getInt, getObject, getArray)
  - `parseFieldElement()` to convert hex strings to field elements
  - `parseCommitment()` to parse G1 point commitments
  - `parseStageProof()` to parse individual stage proofs
  - `parseJoltStageProofs()` to parse all 6 stages
- [x] Add `deserializeProofFromJson()` function
  - Validates ZOLT-JSON magic and version
  - Parses bytecode, memory, register proofs
  - Parses R1CS proof metadata
  - Parses stage proofs (if present)
- [x] Add `readProofFromJsonFile()` for file I/O
- [x] Add `readProofAutoDetect()` - auto-detects binary vs JSON format
- [x] Add `isJsonProof()` helper to detect format from content
- [x] Update CLI verify command to auto-detect proof format
- [x] Display proof format (JSON/Binary) in verifier output
- [x] Add `JsonDeserializationError` type
- [x] Export new functions from zkvm module
- [x] Add comprehensive tests for JSON deserialization
  - Basic deserialization test
  - Full roundtrip test with stage proofs
  - Format detection test

### Bug Fixes
- [x] Fix commitment field types (use base field Fp, not scalar field)
- [x] Add `writeBaseFieldElement()` for base field serialization (big-endian)
- [x] Add `hexToBaseField()` for base field deserialization (big-endian)
- [x] Fix double-free in JSON proof deserialization (separate allocations)
- [x] Fix Zig 0.15 API compatibility (bufPrint instead of formatIntBuf)

## Completed (Previous Session - Iteration 44)

### JSON Serialization Format
- [x] Add JSON proof writer in `src/zkvm/serialization.zig`
  - Human-readable JSON format with "ZOLT-JSON" magic
  - Field elements serialized as 64-character hex strings
  - Pretty-printed with indentation
  - Includes all proof components: bytecode, memory, register, R1CS, stages
- [x] Add `fieldToHex()` and `hexToField()` conversion functions
- [x] Add `JsonProofWriter` type for building JSON output
- [x] Add `serializeProofToJson()` and `writeProofToJsonFile()` exports
- [x] Add `--json` option to CLI prove command
- [x] Add comprehensive tests for JSON serialization

## Completed (Previous Sessions)

### Iteration 43 - Proof Serialization/Deserialization
- [x] Create serialization module (`src/zkvm/serialization.zig`)
- [x] Add `-o/--output` option to `prove` command
- [x] Add `verify` command to load and verify saved proofs
- [x] Add comprehensive tests including full proof roundtrip

### Iteration 42 - CLI Enhancements
- [x] Add `zolt info` command showing zkVM capabilities
- [x] Add `--max-cycles N` option to run command
- [x] Add `--regs` option to run command
- [x] Add `--max-cycles N` option to prove command

### Iteration 41 - Verifier Benchmarks
- [x] Add benchVerifier() function to measure verification performance
- [x] Results: Verify is ~130-165x faster than prove

### Iteration 40 - Complex Program Tests & Benchmarks
- [x] Fix branch target calculation for high PC addresses
- [x] Add complex program tests (loops, memory, shifts, comparisons)
- [x] Add emulator and prover benchmarks

### Iteration 39 - Strict Sumcheck Verification
- [x] Full pipeline now passes with strict_sumcheck = true!

### Iterations 1-38 - Core Implementation
- [x] Complete zkVM implementation
- [x] All 6 sumcheck stages working correctly

## Next Steps (Future Iterations)

### High Priority
- [ ] Add proof compression (gzip/zstd) - requires Zig 0.15 flate API stabilization
- [ ] Add more example programs

### Medium Priority
- [ ] Performance optimization with SIMD
- [ ] Parallel sumcheck round computation

### Low Priority
- [ ] Complete HyperKZG pairing verification
- [ ] More comprehensive benchmarking
- [ ] Documentation improvements

## Test Status
- All tests pass (578 tests)
- Full pipeline with strict verification: PASSED
- All 6 stages verify with p(0) + p(1) = claim check
- Binary serialization roundtrip tests: PASSED
- JSON serialization tests: PASSED
- JSON deserialization tests: PASSED
- Auto-detect format tests: PASSED

## CLI Commands
```
zolt help              # Show help message
zolt version           # Show version
zolt info              # Show zkVM capabilities
zolt run [opts] <elf>  # Run RISC-V ELF binary
  --max-cycles N       # Limit execution cycles
  --regs               # Show final register state
zolt prove [opts] <elf> # Generate ZK proof
  --max-cycles N       # Limit proving cycles
  -o, --output F       # Save proof to file
  --json               # Output in JSON format
zolt verify <proof>    # Verify a saved proof (auto-detects format)
zolt stats <proof>     # Show detailed proof statistics
zolt decode <hex>      # Decode RISC-V instruction
zolt srs <ptau>        # Inspect PTAU file
zolt bench             # Run benchmarks
```

## Performance Summary (from benchmarks)
- Field addition: 4.0 ns/op
- Field multiplication: 54.4 ns/op
- Field inversion: 13.8 us/op
- MSM (256 points): 0.49 ms/op
- HyperKZG commit (1024): 1.5 ms/op
- Emulator (sum 1-100): 72 us/op
- Prover (2 steps): ~97 ms/op
- Prover (14 steps): ~98 ms/op
- Verifier (2 steps): ~593 us/op
- Verifier (14 steps): ~753 us/op
- Proof size (2 steps): 6.38 KB
- Proof size (14 steps): 7.56 KB
