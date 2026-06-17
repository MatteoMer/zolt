# Jolt lookup-table MLE fixtures

Generates `testdata/lookup-table-mle/nonboolean_bn254.txt`, the pinned-upstream non-Boolean MLE fixture consumed by `src/zkvm/lookup_table/new_tables_test.zig`.

The generator depends on upstream `a16z/jolt` pinned to commit `b10e80ac6a9b62672bd196b8a6daf7dce652c9b3` (the same commit used by `jolt-verifier`) and arkworks pinned to commit `76bb3a4518928f1ff7f15875f940d614bb9845e6` so `jolt-core` and this fixture crate use one compatible arkworks instance. `Cargo.lock` is committed and the check runs Cargo with `--locked` so dependency resolution failures are reported instead of rewriting the lockfile. It evaluates the upstream lookup-table implementations for tables 22 and 29-39 on deterministic non-Boolean BN254 scalar challenge vectors.

From the repository root, check that the committed fixture is still reproducible without modifying the working tree:

```bash
zig build check-lookup-tables
```

To intentionally regenerate `testdata/lookup-table-mle/nonboolean_bn254.txt`, run:

```bash
cargo run --locked --release \
  --manifest-path tools/jolt-lookup-mle-fixtures/Cargo.toml \
  -- --out testdata/lookup-table-mle/nonboolean_bn254.txt
```

Review and commit any fixture diff only when the pinned upstream generator legitimately changes. Normal `zig build test` uses the checked-in fixture and does not require Rust/Jolt or network access.
