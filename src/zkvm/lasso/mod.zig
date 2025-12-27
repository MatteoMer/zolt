//! Lasso Lookup Arguments for Jolt
//!
//! This module implements the Lasso lookup argument protocol, which is the core
//! technique that makes Jolt efficient. The Lasso protocol proves that a set of
//! values are all valid entries in a lookup table, using a logarithmic number
//! of commitments.
//!
//! Key components:
//! - ExpandingTable: Incrementally builds EQ polynomial evaluations
//! - SplitEqPolynomial: Gruen's optimization for EQ polynomial evaluation
//! - PrefixSuffixDecomposition: Decomposes table evaluations into prefix/suffix products
//! - LassoProver/LassoVerifier: Main protocol implementations
//!
//! The protocol consists of two main phases:
//! 1. Address binding (first LOG_K rounds): Uses prefix-suffix decomposition
//! 2. Cycle binding (last log_T rounds): Uses Gruen split EQ
//!
//! Reference: "Jolt: SNARKs for Virtual Machines via Lookups" and "Lasso: A Lookup Argument with O(C log N) Prover Cost"

const std = @import("std");

pub const expanding_table = @import("expanding_table.zig");
pub const prefix_suffix = @import("prefix_suffix.zig");
pub const prover = @import("prover.zig");
pub const split_eq = @import("split_eq.zig");
pub const verifier = @import("verifier.zig");

pub const ExpandingTable = expanding_table.ExpandingTable;
pub const PrefixSuffixDecomposition = prefix_suffix.PrefixSuffixDecomposition;
pub const PrefixPolynomial = prefix_suffix.PrefixPolynomial;
pub const SuffixPolynomial = prefix_suffix.SuffixPolynomial;
pub const PrefixRegistry = prefix_suffix.PrefixRegistry;
pub const SuffixType = prefix_suffix.SuffixType;
pub const PrefixType = prefix_suffix.PrefixType;
pub const SplitEqPolynomial = split_eq.SplitEqPolynomial;
pub const LassoProver = prover.LassoProver;
pub const LassoParams = prover.LassoParams;
pub const LassoProof = prover.LassoProof;
pub const runLassoProver = prover.runLassoProver;
pub const LassoVerifier = verifier.LassoVerifier;
pub const verifyLassoProof = verifier.verifyLassoProof;
pub const batchVerifyLassoProofs = verifier.batchVerifyLassoProofs;

test {
    std.testing.refAllDecls(@This());
}
