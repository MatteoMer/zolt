//! Zolt: A Zig port of the Jolt zkVM
//!
//! Jolt is a zero-knowledge virtual machine (zkVM) that proves the correct execution
//! of RISC-V programs using lookup arguments and sumcheck protocols.
//!
//! This is a port of the original Rust implementation from a16z/jolt.

const std = @import("std");

// Core modules
pub const common = @import("common/mod.zig");
pub const field = @import("field/mod.zig");
pub const poly = @import("poly/mod.zig");
pub const subprotocols = @import("subprotocols/mod.zig");
pub const utils = @import("utils/mod.zig");
pub const zkvm = @import("zkvm/mod.zig");
pub const msm = @import("msm/mod.zig");
pub const host = @import("host/mod.zig");
pub const transcripts = @import("transcripts/mod.zig");
pub const guest = @import("guest/mod.zig");

// Re-export commonly used types
pub const JoltField = field.JoltField;
pub const BN254Scalar = field.BN254Scalar;

/// Library version
pub const version = "0.1.0";

test {
    // Run all module tests
    std.testing.refAllDecls(@This());
}
