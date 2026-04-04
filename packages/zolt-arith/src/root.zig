//! zolt-arith: Arithmetic primitives for Zolt.
//!
//! BN254 field arithmetic, polynomial operations, MSM, commitments,
//! transcripts, and sumcheck protocol.

pub const field = @import("field/mod.zig");
pub const poly = @import("poly/mod.zig");
pub const msm = @import("msm/mod.zig");
pub const gpu = @import("gpu/mod.zig");
pub const transcripts = @import("transcripts/mod.zig");
pub const subprotocols = @import("subprotocols/mod.zig");

// Re-export commonly used types
pub const JoltField = field.JoltField;
pub const BN254Scalar = field.BN254Scalar;

// Utility types
pub const bits = @import("bits.zig");
pub const LookupBits = bits.LookupBits;
pub const uninterleaveBits = bits.uninterleaveBits;
pub const interleaveBits = bits.interleaveBits;
pub const expanding_table = @import("expanding_table.zig");
pub const ExpandingTable = expanding_table.ExpandingTable;

test {
    _ = @import("field/mod.zig");
    _ = @import("poly/mod.zig");
    _ = @import("msm/mod.zig");
    _ = @import("transcripts/mod.zig");
    _ = @import("subprotocols/mod.zig");
    _ = @import("bits.zig");
}
