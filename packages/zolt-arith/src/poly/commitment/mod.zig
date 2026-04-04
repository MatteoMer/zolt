//! Polynomial commitment scheme
//!
//! Zolt uses Dory (transparent setup, pairing-based) as its sole commitment scheme.

pub const dory = @import("dory.zig");
pub const DoryCommitmentScheme = dory.DoryCommitmentScheme;
pub const DoryCommitment = dory.DoryCommitment;
pub const DoryProof = dory.DoryProof;
pub const DorySRS = dory.DorySRS;
pub const serializeDoryCommitment = dory.serializeDoryCommitment;
pub const deserializeDoryCommitment = dory.deserializeDoryCommitment;
