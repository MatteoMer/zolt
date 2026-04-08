# Inline SHA-256 guest workspace

Five minimal Jolt SDK guest crates that compute `sha256(zero-buf-of-N-bytes)`
using `jolt_inlines_sha2::Sha256::digest`, one per size (64, 128, 512, 1024,
2048 bytes). The resulting ELFs trigger Jolt's inline SHA-256 expansion
(`VirtualRev8W`, `VirtualROTRW`, `Andn`, etc.), which produces traces roughly
8× smaller than a pure-software SHA-256 implementation at the same input size.

## Layout

- `Cargo.toml` — workspace root pinning Jolt `997c1543` and the matching
  `arkworks-algebra dev/twist-shout` branch (same as `jolt-bench/Cargo.toml`,
  so cargo reuses the cached git checkouts).
- `linker.ld` — verbatim port of `jolt-core/src/linker.ld.template` with the
  template variables substituted for the `jolt build --mode no-std --backtrace
  off` defaults (memory origin `0x80000000`, 128 MiB RAM, 64 MiB heap, 8 MiB
  stack, no unwind tables).
- `.cargo/config.toml` — matches the rustflags that `zeroos-build` passes when
  `jolt build` runs (see `src/main.rs::build_command` in jolt-core 997c1543).
- `rust-toolchain.toml` — pins to Rust 1.94 with the RISC-V targets.
- `guest-N/` — one crate per input size, each with `src/lib.rs` (the
  `#[jolt::provable]` function) and `src/main.rs` (a `#![no_main]` shim so
  cargo produces a binary target).

## Building

```sh
cd examples/sha2-inline-guests
cargo build --release --bin sha256-inline-guest-64 --features sha256-inline-guest-64/guest
```

Replace `64` with any of `64, 128, 512, 1024, 2048`. The resulting ELF lands at
`target/riscv64imac-unknown-none-elf/release/sha256-inline-guest-<N>`.

To rebuild all five and copy into `examples/` (replacing the committed binaries):

```sh
for n in 64 128 512 1024 2048; do
  cargo build --release --bin sha256-inline-guest-${n} --features sha256-inline-guest-${n}/guest
done

OUT=target/riscv64imac-unknown-none-elf/release
cp "$OUT/sha256-inline-guest-64"   ../sha256.elf
cp "$OUT/sha256-inline-guest-128"  ../sha256_128.elf
cp "$OUT/sha256-inline-guest-512"  ../sha256_512.elf
cp "$OUT/sha256-inline-guest-1024" ../sha256_1024.elf
cp "$OUT/sha256-inline-guest-2048" ../sha256_2048.elf
```

## Why not use `jolt build`?

Building the guests via direct `cargo build` keeps the toolchain anchored to
whatever Jolt revision is pinned in this workspace's `Cargo.toml`, rather than
whatever version happens to be installed as `~/.cargo/bin/jolt`. The two can
drift (linker template changes, rustflag tweaks), which would produce ELFs
that Zolt's verifier expectations no longer match.

If you need to regenerate the linker script or the rustflag set, mirror them
from `jolt-core/src/main.rs::build_command` and
`jolt-core/src/linker.ld.template` at the same rev (`997c1543`).
