//! Zolt: A Zig port of the Jolt zkVM
//!
//! Jolt is a zero-knowledge virtual machine (zkVM) that proves the correct execution
//! of RISC-V programs using lookup arguments and sumcheck protocols.
//!
//! This is a port of the original Rust implementation from a16z/jolt.

const std = @import("std");

// Package re-exports
const zolt_arith = @import("zolt_arith");
const zolt_pool = @import("zolt_pool");

// Arithmetic modules (from zolt-arith package)
pub const field = zolt_arith.field;
pub const poly = zolt_arith.poly;
pub const msm = zolt_arith.msm;
pub const gpu = zolt_arith.gpu;
pub const transcripts = zolt_arith.transcripts;
pub const subprotocols = zolt_arith.subprotocols;

// Re-export commonly used types
pub const JoltField = field.JoltField;
pub const BN254Scalar = field.BN254Scalar;

// zkVM modules
pub const common = @import("common/mod.zig");
pub const zkvm = @import("zkvm/mod.zig");
pub const host = @import("host/mod.zig");
pub const guest = @import("guest/mod.zig");
pub const tracer = @import("tracer/mod.zig");
pub const utils = @import("utils/mod.zig");

/// Library version
pub const version = "0.1.0";

test {
    // Run all module tests
    std.testing.refAllDecls(@This());

    // Run integration tests
    _ = @import("integration_tests.zig");
}
