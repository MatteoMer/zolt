# Jolt lookup-table MLE fixtures

Generates `testdata/lookup-table-mle/nonboolean_bn254.txt`, the pinned-upstream non-Boolean MLE fixture consumed by `src/zkvm/lookup_table/new_tables_test.zig`.

The generator depends on upstream `a16z/jolt` pinned to commit `b10e80ac6a9b62672bd196b8a6daf7dce652c9b3` (the same commit used by `jolt-verifier`). It evaluates the upstream lookup-table implementations for tables 22 and 29-39 on deterministic non-Boolean BN254 scalar challenge vectors.

From the repository root:

```bash
cargo run --release \
  --manifest-path tools/jolt-lookup-mle-fixtures/Cargo.toml \
  -- --out testdata/lookup-table-mle/nonboolean_bn254.txt
```

Normal `zig build test` uses the checked-in fixture and does not require Rust or network access.
