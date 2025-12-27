# Zolt zkVM Implementation TODO

## Completed
- [x] Lookup table infrastructure (14 tables)
- [x] LookupBits utility for bit manipulation
- [x] Fixed interleave/uninterleave to match Jolt conventions
- [x] Fixed MLE tests with MSB-first ordering

## In Progress
- [ ] Commit and push current changes

## Next Up
- [ ] Create instruction flags (CircuitFlags, InstructionFlags)
- [ ] Create Lasso prover infrastructure
- [ ] Create Lasso verifier
- [ ] Integrate with sumcheck protocol

## Later
- [ ] Memory RAF checking
- [ ] Multi-stage sumcheck orchestration
- [ ] Fix HyperKZG verification
- [ ] Implement execute(), prove(), verify()
