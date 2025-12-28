# Zolt zkVM Implementation TODO

## Completed (This Session - Iteration 44)

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

## Completed (Previous Session - Iteration 43)

### Proof Serialization/Deserialization
- [x] Create serialization module (`src/zkvm/serialization.zig`)
  - Binary format with magic header "ZOLT" and version
  - Serialize JoltProof, StageProof, JoltStageProofs
  - Field element and commitment serialization
  - File I/O helpers (writeProofToFile, readProofFromFile)
- [x] Add `-o/--output` option to `prove` command
- [x] Add `verify` command to load and verify saved proofs
- [x] Add comprehensive tests including full proof roundtrip
- [x] Update README with new CLI commands
- [x] Fix Zig 0.15 API changes (ArrayListUnmanaged)
- [x] Add toBytes() method to BN254Scalar for serialization
- [x] Update R1CS proof serialization to match struct fields
- [x] Add toBytes/fromBytes roundtrip test

## Completed (Previous Sessions)

### Iteration 42 - CLI Enhancements
- [x] Add `zolt info` command showing zkVM capabilities
- [x] Display proof system details (HyperKZG, Spartan, Lasso)
- [x] Show 6-stage sumcheck overview
- [x] List RISC-V ISA support (60+ instructions, 24 lookup tables)
- [x] Include performance metrics
- [x] Add tests for info command
- [x] Add `--max-cycles N` option to run command (limit execution cycles)
- [x] Add `--regs` option to run command (show final register state)
- [x] Update README with new commands and options
- [x] Add `--max-cycles N` option to prove command (limit proving cycles)

### Iteration 41 - Verifier Benchmarks
- [x] Add benchVerifier() function to measure verification performance
- [x] Results: Verify is ~130-165x faster than prove

### Iteration 40 - Complex Program Tests & Benchmarks
- [x] Fix branch target calculation for high PC addresses (0x80000000+)
- [x] Add complex program tests (loops, memory, shifts, comparisons)
- [x] Add emulator and prover benchmarks

### Iteration 39 - Strict Sumcheck Verification
- [x] Fix Lasso prover eq_evals padding for proper cycle phase folding
- [x] Fix Val prover to use 4-point Lagrange interpolation for degree-3 sumcheck
- [x] Full pipeline now passes with strict_sumcheck = true!

### Iterations 1-38 - Core Implementation
- [x] Complete zkVM implementation
- [x] All 6 sumcheck stages working correctly

## Next Steps (Future Iterations)

### High Priority
- [ ] Add proof compression (gzip/zstd) - Zig 0.15 has new I/O API
- [ ] Add JSON deserialization for proof loading

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
- Serialization roundtrip tests: PASSED
- Field element toBytes/fromBytes: PASSED
- JSON serialization tests: PASSED

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
zolt verify <proof>    # Verify a saved proof
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
