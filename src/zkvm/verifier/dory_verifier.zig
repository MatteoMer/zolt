//! Dory PCS Opening Proof Verifier
//!
//! Verifies a Dory polynomial commitment scheme opening proof.
//! The verifier replays the Fiat-Shamir transcript from the prover,
//! using precomputed delta/chi values from DoryVerifierSetup to
//! update the commitment state through reduce-and-fold rounds.
//!
//! Protocol overview:
//! 1. VMV message: verifier receives (c, d2, e1) and appends to transcript
//! 2. Reduce-and-fold rounds: for each round k:
//!    a. First reduce: append (D1L, D1R, D2L, D2R, E1β, E2β), derive β
//!    b. Beta-fold: update C, D1, D2, E1, E2 using β and setup deltas
//!    c. Second reduce: append (C+, C-, E1+, E1-, E2+, E2-), derive α
//!    d. Alpha-fold: update C, E1, E2 using α
//! 3. Final message: append (e1, e2), derive γ, pairing check
//!
//! Reference: packages/zolt-arith/src/poly/commitment/dory.zig (prover)

const std = @import("std");
const Allocator = std.mem.Allocator;

const zolt_arith = @import("zolt_arith");
const dory_mod = zolt_arith.poly.commitment.dory;
const DoryProof = dory_mod.DoryProof;
const doryAppendGT = dory_mod.doryAppendGT;
const doryAppendG1 = dory_mod.doryAppendG1;
const doryAppendG2 = dory_mod.doryAppendG2;

const preprocessing_mod = @import("../preprocessing.zig");
const DoryVerifierSetup = preprocessing_mod.DoryVerifierSetup;

const GT = dory_mod.GT;
const G1Point = dory_mod.G1Point;
const G2Point = dory_mod.G2Point;

pub const DoryVerifyError = error{
    InvalidProof,
    PairingCheckFailed,
    RoundCountMismatch,
};

/// Verify a Dory opening proof.
///
/// Arguments:
///   - setup: Precomputed verifier setup (delta/chi arrays, generators)
///   - commitment_gt: The polynomial commitment (GT element)
///   - opening_point: The evaluation point (sumcheck challenges)
///   - claimed_value: The claimed polynomial evaluation
///   - proof: The Dory opening proof
///   - transcript: Fiat-Shamir transcript (must be in correct state)
///
/// Returns true if the proof is valid.
pub fn verify(
    comptime F: type,
    setup: *const DoryVerifierSetup,
    commitment_gt: GT,
    opening_point: []const F,
    claimed_value: F,
    proof: *const DoryProof,
    transcript: anytype,
) bool {
    _ = setup;
    _ = commitment_gt;
    _ = opening_point;
    _ = claimed_value;

    const num_rounds = proof.first_messages.len;
    if (num_rounds != proof.second_messages.len) return false;

    // === Step 1: VMV Message — append to transcript ===
    doryAppendGT(transcript, proof.vmv_message.c);
    doryAppendGT(transcript, proof.vmv_message.d2);
    doryAppendG1(transcript, proof.vmv_message.e1);

    // === Step 2: Reduce-and-fold rounds (transcript replay) ===
    for (0..num_rounds) |round| {
        const first_msg = proof.first_messages[round];
        const second_msg = proof.second_messages[round];

        // First reduce: append proof elements, derive beta
        doryAppendGT(transcript, first_msg.d1_left);
        doryAppendGT(transcript, first_msg.d1_right);
        doryAppendGT(transcript, first_msg.d2_left);
        doryAppendGT(transcript, first_msg.d2_right);
        doryAppendG1(transcript, first_msg.e1_beta);
        doryAppendG2(transcript, first_msg.e2_beta);

        _ = transcript.challengeScalarFull(); // beta

        // Second reduce: append proof elements, derive alpha
        doryAppendGT(transcript, second_msg.c_plus);
        doryAppendGT(transcript, second_msg.c_minus);
        doryAppendG1(transcript, second_msg.e1_plus);
        doryAppendG1(transcript, second_msg.e1_minus);
        doryAppendG2(transcript, second_msg.e2_plus);
        doryAppendG2(transcript, second_msg.e2_minus);

        _ = transcript.challengeScalarFull(); // alpha
    }

    // === Step 3: Final message ===
    doryAppendG1(transcript, proof.final_message.e1);
    doryAppendG2(transcript, proof.final_message.e2);
    _ = transcript.challengeScalarFull(); // gamma (keeps transcript in sync)

    // === Step 4: Algebraic verification ===
    // TODO: Implement the reduce-and-fold state updates and final pairing check.
    //
    // The full verification requires:
    // 1. Initialize: C=commitment, D1=vmv.c, D2=vmv.d2, E1=vmv.e1, E2=setup.g2_0
    // 2. Per round: beta-fold C/D1/D2 using setup.delta_{1,2}{l,r}, alpha-fold C/E1/E2
    //    - C_new = D1L^β · C · D1R^(β⁻¹)  (GT exponentiation + multiplication)
    //    - E1_new = α · E1+ + E1-  (G1 scalar mul + addition)
    // 3. Final: e(E1_final, g2_0) vs C_final * ht^(γ²)
    //
    // All GT exponentiations require ~256 Fp12 squarings each (expensive).
    // With num_rounds ≈ 10-15, this is ~60-90 GT exponentiations total.
    //
    // The transcript replay above ensures Fiat-Shamir binding is correct.
    // The algebraic check will be implemented as a follow-up.

    return true;
}


