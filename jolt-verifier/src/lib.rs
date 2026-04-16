use ark_serialize::CanonicalDeserialize;
use jolt_core::curve::Bn254Curve;
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::zkvm::verifier::JoltVerifierPreprocessing;
use jolt_core::zkvm::{RV64IMACProof, RV64IMACVerifier, Serializable};
use std::io::Cursor;

type PreprocessingType = JoltVerifierPreprocessing<ark_bn254::Fr, Bn254Curve, DoryCommitmentScheme>;

/// Verify a Jolt proof via C FFI.
///
/// Returns:
///   0  — proof is valid
///   1  — proof is invalid (verification failed)
///   2  — deserialization or other error
///
/// `proof_ptr` / `proof_len`           — serialized RV64IMACProof bytes
/// `preprocessing_ptr` / `preprocessing_len` — serialized JoltVerifierPreprocessing bytes
/// `io_ptr` / `io_len`                 — serialized JoltDevice bytes (program I/O sidecar);
///                                       pass null / 0 for empty I/O
#[no_mangle]
pub extern "C" fn jolt_verify(
    proof_ptr: *const u8,
    proof_len: usize,
    preprocessing_ptr: *const u8,
    preprocessing_len: usize,
    io_ptr: *const u8,
    io_len: usize,
) -> i32 {
    // Safety: caller guarantees valid pointers and lengths
    let proof_bytes = unsafe { std::slice::from_raw_parts(proof_ptr, proof_len) };
    let preprocessing_bytes =
        unsafe { std::slice::from_raw_parts(preprocessing_ptr, preprocessing_len) };

    // Deserialize preprocessing
    let mut pp_cursor = Cursor::new(preprocessing_bytes);
    let preprocessing = match PreprocessingType::deserialize_compressed(&mut pp_cursor) {
        Ok(pp) => pp,
        Err(_) => return 2,
    };

    // Deserialize proof
    let proof = match RV64IMACProof::deserialize_from_bytes(proof_bytes) {
        Ok(p) => p,
        Err(_) => return 2,
    };

    // Deserialize program I/O (or fall back to empty JoltDevice)
    let program_io = if !io_ptr.is_null() && io_len > 0 {
        let io_bytes = unsafe { std::slice::from_raw_parts(io_ptr, io_len) };
        let mut io_cursor = Cursor::new(io_bytes);
        match common::jolt_device::JoltDevice::deserialize_compressed(&mut io_cursor) {
            Ok(dev) => dev,
            Err(_) => return 2,
        }
    } else {
        common::jolt_device::JoltDevice {
            memory_layout: preprocessing.shared.memory_layout.clone(),
            ..Default::default()
        }
    };

    // Build verifier and run verification
    let verifier = match RV64IMACVerifier::new(&preprocessing, proof, program_io, None, None) {
        Ok(v) => v,
        Err(_) => return 2,
    };

    match verifier.verify() {
        Ok(()) => 0,
        Err(_) => 1,
    }
}
