//! Configuration attributes for Jolt programs.
//!
//! These attributes control VM configuration like memory size, stack size,
//! and maximum sizes for various buffers.

const constants = @import("constants.zig");

/// Configuration attributes for a Jolt program
pub const Attributes = struct {
    /// Enable WebAssembly target
    wasm: bool = false,

    /// Use nightly features
    nightly: bool = false,

    /// Guest-only mode (no host interaction)
    guest_only: bool = false,

    /// VM memory size in bytes
    memory_size: u64 = constants.DEFAULT_MEMORY_SIZE,

    /// Stack size in bytes
    stack_size: u64 = constants.DEFAULT_STACK_SIZE,

    /// Maximum input size in bytes
    max_input_size: u64 = constants.DEFAULT_MAX_INPUT_SIZE,

    /// Maximum output size in bytes
    max_output_size: u64 = constants.DEFAULT_MAX_OUTPUT_SIZE,

    /// Maximum trusted advice size in bytes
    max_trusted_advice_size: u64 = constants.DEFAULT_MAX_TRUSTED_ADVICE_SIZE,

    /// Maximum untrusted advice size in bytes
    max_untrusted_advice_size: u64 = constants.DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE,

    /// Maximum trace length (number of VM steps)
    max_trace_length: u64 = constants.DEFAULT_MAX_TRACE_LENGTH,

    /// Create default attributes
    pub fn default() Attributes {
        return .{};
    }

    /// Create attributes with custom memory and stack size
    pub fn withMemory(memory_size: u64, stack_size: u64) Attributes {
        return .{
            .memory_size = memory_size,
            .stack_size = stack_size,
        };
    }
};

test "default attributes" {
    const attrs = Attributes.default();
    const std = @import("std");

    std.testing.expect(!attrs.wasm) catch unreachable;
    std.testing.expect(!attrs.nightly) catch unreachable;
    std.testing.expect(!attrs.guest_only) catch unreachable;
    std.testing.expectEqual(constants.DEFAULT_MEMORY_SIZE, attrs.memory_size) catch unreachable;
    std.testing.expectEqual(constants.DEFAULT_STACK_SIZE, attrs.stack_size) catch unreachable;
}
