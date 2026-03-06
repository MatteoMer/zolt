use ark_serialize::CanonicalDeserialize;
use clap::Parser;
use jolt_core::curve::Bn254Curve;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::zkvm::verifier::JoltVerifierPreprocessing;
use jolt_core::zkvm::{RV64IMACProof, RV64IMACVerifier, Serializable};
use std::fs;
use std::io::Cursor;
use std::process;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

type PreprocessingType = JoltVerifierPreprocessing<ark_bn254::Fr, Bn254Curve, DoryCommitmentScheme>;
type DoryComm = <DoryCommitmentScheme as CommitmentScheme>::Commitment;
type DoryPrf = <DoryCommitmentScheme as CommitmentScheme>::Proof;

#[derive(Parser)]
#[command(name = "jolt-verifier", about = "Verify Zolt proofs using Jolt verifier")]
struct Cli {
    /// Path to the proof file (e.g. zolt_proof_dory.bin)
    #[arg(short, long)]
    proof: String,

    /// Path to the preprocessing file (e.g. zolt_preprocessing.bin)
    #[arg(short = 'P', long)]
    preprocessing: String,

    /// Diagnostic mode: try to deserialize proof fields one at a time
    #[arg(long)]
    diagnose: bool,
}

fn main() {
    let cli = Cli::parse();

    // Load preprocessing
    eprintln!("Loading preprocessing from {}...", cli.preprocessing);
    let preprocessing_bytes =
        fs::read(&cli.preprocessing).unwrap_or_else(|e| fatal(&format!("read preprocessing: {e}")));

    let mut pp_cursor = Cursor::new(&preprocessing_bytes);
    let preprocessing = PreprocessingType::deserialize_compressed(&mut pp_cursor)
        .unwrap_or_else(|e| fatal(&format!("deserialize preprocessing: {e}")));

    let pp_consumed = pp_cursor.position() as usize;
    eprintln!("Preprocessing loaded ({} bytes consumed out of {})", pp_consumed, preprocessing_bytes.len());

    // Load proof
    eprintln!("Loading proof from {}...", cli.proof);
    let proof_bytes = fs::read(&cli.proof).unwrap_or_else(|e| fatal(&format!("read proof: {e}")));

    if cli.diagnose {
        diagnose_proof(&proof_bytes);
        return;
    }

    let proof = RV64IMACProof::deserialize_from_bytes(&proof_bytes)
        .unwrap_or_else(|e| fatal(&format!("deserialize proof: {e}")));

    eprintln!(
        "Proof loaded (trace_length={}, commitments={})",
        proof.trace_length,
        proof.commitments.len()
    );

    // Create minimal program I/O
    let program_io = common::jolt_device::JoltDevice {
        memory_layout: preprocessing.shared.memory_layout.clone(),
        ..Default::default()
    };

    // Verify
    eprintln!("Verifying...");
    let verifier = RV64IMACVerifier::new(&preprocessing, proof, program_io, None, None)
        .unwrap_or_else(|e| fatal(&format!("create verifier: {e}")));

    match verifier.verify() {
        Ok(()) => {
            eprintln!("VERIFIED: proof is valid");
            process::exit(0);
        }
        Err(e) => {
            eprintln!("FAILED: {e:?}");
            process::exit(1);
        }
    }
}

fn diagnose_proof(bytes: &[u8]) {
    use ark_serialize::Compress;
    use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
    use jolt_core::subprotocols::univariate_skip::UniSkipFirstRoundProofVariant;
    use jolt_core::transcripts::Blake2bTranscript;

    type F = ark_bn254::Fr;
    type C = Bn254Curve;
    type FS = Blake2bTranscript;

    let mut cursor = Cursor::new(bytes);

    eprintln!("\n--- Diagnosing proof deserialization ---");
    eprintln!("Total proof size: {} bytes", bytes.len());

    // 1. commitments: Vec<DoryComm>
    match Vec::<DoryComm>::deserialize_with_mode(&mut cursor, Compress::Yes, ark_serialize::Validate::No) {
        Ok(comms) => eprintln!("[OK] commitments: {} items, cursor at {}", comms.len(), cursor.position()),
        Err(e) => { eprintln!("[FAIL] commitments at offset {}: {e}", cursor.position()); return; }
    }

    // 2. stage1_uni_skip_first_round_proof
    match UniSkipFirstRoundProofVariant::<F, C, FS>::deserialize_with_mode(&mut cursor, Compress::Yes, ark_serialize::Validate::No) {
        Ok(_) => eprintln!("[OK] stage1_uni_skip, cursor at {}", cursor.position()),
        Err(e) => { eprintln!("[FAIL] stage1_uni_skip at offset {}: {e}", cursor.position()); return; }
    }

    // 3. stage1_sumcheck_proof
    match SumcheckInstanceProof::<F, C, FS>::deserialize_with_mode(&mut cursor, Compress::Yes, ark_serialize::Validate::No) {
        Ok(_) => eprintln!("[OK] stage1_sumcheck, cursor at {}", cursor.position()),
        Err(e) => { eprintln!("[FAIL] stage1_sumcheck at offset {}: {e}", cursor.position()); return; }
    }

    // 4. stage2_uni_skip_first_round_proof
    match UniSkipFirstRoundProofVariant::<F, C, FS>::deserialize_with_mode(&mut cursor, Compress::Yes, ark_serialize::Validate::No) {
        Ok(_) => eprintln!("[OK] stage2_uni_skip, cursor at {}", cursor.position()),
        Err(e) => { eprintln!("[FAIL] stage2_uni_skip at offset {}: {e}", cursor.position()); return; }
    }

    // 5. stage2_sumcheck_proof
    match SumcheckInstanceProof::<F, C, FS>::deserialize_with_mode(&mut cursor, Compress::Yes, ark_serialize::Validate::No) {
        Ok(_) => eprintln!("[OK] stage2_sumcheck, cursor at {}", cursor.position()),
        Err(e) => { eprintln!("[FAIL] stage2_sumcheck at offset {}: {e}", cursor.position()); return; }
    }

    // 6-10. stages 3-7
    for stage in 3..=7u32 {
        match SumcheckInstanceProof::<F, C, FS>::deserialize_with_mode(&mut cursor, Compress::Yes, ark_serialize::Validate::No) {
            Ok(_) => eprintln!("[OK] stage{}_sumcheck, cursor at {}", stage, cursor.position()),
            Err(e) => { eprintln!("[FAIL] stage{}_sumcheck at offset {}: {e}", stage, cursor.position()); return; }
        }
    }

    // 11. joint_opening_proof
    match DoryPrf::deserialize_with_mode(&mut cursor, Compress::Yes, ark_serialize::Validate::No) {
        Ok(_) => eprintln!("[OK] joint_opening_proof, cursor at {}", cursor.position()),
        Err(e) => { eprintln!("[FAIL] joint_opening_proof at offset {}: {e}", cursor.position()); return; }
    }

    // 12. untrusted_advice_commitment: Option<DoryComm>
    match Option::<DoryComm>::deserialize_with_mode(&mut cursor, Compress::Yes, ark_serialize::Validate::No) {
        Ok(v) => eprintln!("[OK] untrusted_advice_commitment: is_some={}, cursor at {}", v.is_some(), cursor.position()),
        Err(e) => { eprintln!("[FAIL] untrusted_advice_commitment at offset {}: {e}", cursor.position()); return; }
    }

    let pos = cursor.position() as usize;
    let remaining = bytes.len() - pos;
    eprintln!("\nRemaining bytes: {}", remaining);
    let end = std::cmp::min(pos + 32, bytes.len());
    eprintln!("Next bytes: {:02x?}", &bytes[pos..end]);

    // 13. opening_claims (Claims)
    use jolt_core::zkvm::proof_serialization::Claims;
    type ClaimsF = Claims<F>;
    match ClaimsF::deserialize_with_mode(&mut cursor, Compress::Yes, ark_serialize::Validate::No) {
        Ok(claims) => eprintln!("[OK] opening_claims: {} entries, cursor at {}", claims.0.len(), cursor.position()),
        Err(e) => { eprintln!("[FAIL] opening_claims at offset {}: {e}", cursor.position()); return; }
    }

    // 14. remaining config fields
    let pos2 = cursor.position() as usize;
    let remaining2 = bytes.len() - pos2;
    eprintln!("After claims: remaining={} bytes, next: {:02x?}", remaining2, &bytes[pos2..std::cmp::min(pos2+32, bytes.len())]);
}

fn fatal(msg: &str) -> ! {
    eprintln!("Error: {msg}");
    process::exit(2);
}
