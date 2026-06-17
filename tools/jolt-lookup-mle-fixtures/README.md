# Jolt lookup-table MLE fixtures

Generates `testdata/lookup-table-mle/nonboolean_bn254.txt`, the pinned-upstream non-Boolean MLE fixture consumed by `src/zkvm/lookup_table/new_tables_test.zig`.

The checked-in fixture records the upstream `a16z/jolt` commit that produced it (`b10e80ac6a9b62672bd196b8a6daf7dce652c9b3`). The generator computes the rows from direct translations of that commit's Jolt lookup-table `evaluate_mle` implementations (not from the committed fixture). Its small arithmetic dependencies are pinned by `Cargo.lock`, so maintenance checks can reproduce the fixture bytes without lockfile rewrites.

From the repository root, check that the committed fixture is still reproducible without modifying the working tree:

```bash
zig build check-lookup-tables
```

To intentionally regenerate `testdata/lookup-table-mle/nonboolean_bn254.txt`, run:

```bash
cargo run --release \
  --manifest-path tools/jolt-lookup-mle-fixtures/Cargo.toml \
  -- --out testdata/lookup-table-mle/nonboolean_bn254.txt
```

Review and commit any fixture diff only when the pinned upstream generator legitimately changes. Normal `zig build test` uses the checked-in fixture and does not require Rust/Jolt or network access.
