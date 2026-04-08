#![cfg_attr(feature = "guest", no_std)]

/// SHA-256 of a fixed 512-byte input using the Jolt inline implementation.
///
/// Uses the jolt-sdk macro's default sizes for stack/io/advice (see
/// `common/src/constants.rs` in jolt-core `997c1543`), overriding only
/// `heap_size` down to 64 KiB since a fixed-size SHA-256 digest doesn't
/// allocate. The compile-time `output_start` constant the macro emits for
/// these defaults is `0x7FFFB000`; Zolt's prover and jolt-bench must both
/// construct a `MemoryConfig` with matching io/advice sizes or the guest
/// will write its output into a region the host prover considers unmapped.
#[jolt::provable(heap_size = 65536, max_trace_length = 131072)]
fn sha256_inline_512() -> [u8; 32] {
    let buf = [0u8; 512];
    jolt_inlines_sha2::Sha256::digest(&buf)
}
