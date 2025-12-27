//! Common types and constants for Jolt zkVM
//!
//! This module contains shared definitions used across the entire codebase.

pub const constants = @import("constants.zig");
pub const attributes = @import("attributes.zig");
pub const jolt_device = @import("jolt_device.zig");

// Re-export commonly used types
pub const JoltDevice = jolt_device.JoltDevice;
pub const MemoryLayout = jolt_device.MemoryLayout;
pub const MemoryConfig = jolt_device.MemoryConfig;
pub const Attributes = attributes.Attributes;

// Re-export constants
pub const XLEN = constants.XLEN;
pub const REGISTER_COUNT = constants.REGISTER_COUNT;
pub const RAM_START_ADDRESS = constants.RAM_START_ADDRESS;

test {
    // Run all submodule tests
    _ = constants;
    _ = attributes;
    _ = jolt_device;
}
