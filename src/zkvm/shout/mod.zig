//! Shout Lookup Arguments for Jolt
//!
//! This module implements the Shout lookup argument protocol, which is the core
//! technique that makes Jolt efficient. The Shout protocol proves that a set of
//! values are all valid entries in a lookup table, using one-hot address polynomials
//! and prefix-suffix decomposition without committing to chunk polynomials.
//!
//! Key components:
//! - ExpandingTable: Incrementally builds EQ polynomial evaluations
//! - SplitEqPolynomial: Gruen's optimization for EQ polynomial evaluation
//! - PrefixSuffixDecomposition: Decomposes table evaluations into prefix/suffix products
//! - ShoutProver/ShoutVerifier: Main protocol implementations
//!
//! The protocol consists of two main phases:
//! 1. Address binding (first LOG_K rounds): Uses prefix-suffix decomposition
//! 2. Cycle binding (last log_T rounds): Uses Gruen split EQ
//!
//! Reference: "Twist and Shout" (eprint 2025/105) and "Proving CPU Executions in Small Space" (eprint 2025/611)

const std = @import("std");

pub const expanding_table = @import("expanding_table.zig");
pub const integration_test = @import("integration_test.zig");
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
pub const ShoutProver = prover.ShoutProver;
pub const ShoutParams = prover.ShoutParams;
pub const ShoutProof = prover.ShoutProof;
pub const runShoutProver = prover.runShoutProver;
pub const ShoutVerifier = verifier.ShoutVerifier;
pub const verifyShoutProof = verifier.verifyShoutProof;
pub const batchVerifyShoutProofs = verifier.batchVerifyShoutProofs;

test {
    std.testing.refAllDecls(@This());
}
