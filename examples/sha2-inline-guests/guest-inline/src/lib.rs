#![cfg_attr(feature = "guest", no_std)]

/// Minimal SHA-256 inline example: digest an empty buffer.
///
/// The previous `sha256_inline.elf` in this repo was a ~4096-cycle binary
/// that exercised exactly one inline compression + padding block. This crate
/// reproduces that smallest-possible inline SHA-256 trace so the pinned
/// `sha256_inline` bench entry stays comparable across rebuilds.
#[jolt::provable(heap_size = 65536, max_trace_length = 65536)]
fn sha256_inline() -> [u8; 32] {
    jolt_inlines_sha2::Sha256::digest(&[])
}
