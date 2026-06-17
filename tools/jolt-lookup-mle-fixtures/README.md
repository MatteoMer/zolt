# Jolt lookup-table MLE fixtures

Generates `testdata/lookup-table-mle/nonboolean_bn254.txt`, the pinned-upstream non-Boolean MLE fixture consumed by `src/zkvm/lookup_table/new_tables_test.zig`.

The committed fixture records values generated from upstream `a16z/jolt` commit `b10e80ac6a9b62672bd196b8a6daf7dce652c9b3` (the same commit used by `jolt-verifier`). The helper crate is intentionally dependency-free and rewrites that pinned fixture byte-for-byte, so repository checks do not need to compile upstream Rust/Jolt or access the network.

From the repository root, check that the committed fixture is still reproducible without modifying the working tree:

```bash
zig build check-lookup-tables
```

To intentionally refresh `testdata/lookup-table-mle/nonboolean_bn254.txt` from the pinned fixture helper, run:

```bash
cargo run --release \
  --manifest-path tools/jolt-lookup-mle-fixtures/Cargo.toml \
  -- --out testdata/lookup-table-mle/nonboolean_bn254.txt
```

Review and commit any fixture diff only when the pinned upstream generator legitimately changes. Normal `zig build test` uses the checked-in fixture and does not require Rust/Jolt or network access.
