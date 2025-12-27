//! Jolt constants defining the zkVM configuration.
//!
//! These constants define the RISC-V architecture parameters, memory layout,
//! and default sizes for various components.

/// Register width in bits (64-bit RISC-V)
pub const XLEN: usize = 64;

/// Number of RISC-V general purpose registers
pub const RISCV_REGISTER_COUNT: u8 = 32;

/// Number of virtual registers used by Jolt (see Section 6.1 of Jolt paper)
pub const VIRTUAL_REGISTER_COUNT: u8 = 96;

/// Reserved virtual registers for virtual instructions
pub const VIRTUAL_INSTRUCTION_RESERVED_REGISTER_COUNT: u8 = 7;

/// Total register count (must be a power of 2)
pub const REGISTER_COUNT: u8 = RISCV_REGISTER_COUNT + VIRTUAL_REGISTER_COUNT;

/// Size of a RISC-V instruction in bytes
pub const BYTES_PER_INSTRUCTION: usize = 4;

/// Alignment factor for bytecode
pub const ALIGNMENT_FACTOR_BYTECODE: usize = 2;

/// Start address of RAM in the VM memory space
pub const RAM_START_ADDRESS: u64 = 0x80000000;

/// Emulator memory capacity (128 MB - big enough for Linux and xv6)
pub const EMULATOR_MEMORY_CAPACITY: u64 = 1024 * 1024 * 128;

/// Default memory size for the VM
pub const DEFAULT_MEMORY_SIZE: u64 = EMULATOR_MEMORY_CAPACITY;

/// Default stack size in bytes
pub const DEFAULT_STACK_SIZE: u64 = 4096;

/// Stack canary size (64 bytes - 4 word protection for 32-bit, 2 word for 64-bit)
pub const STACK_CANARY_SIZE: u64 = 128;

/// Stack start address (end of RAM region, grows downward)
pub const STACK_START_ADDRESS: u64 = RAM_START_ADDRESS + EMULATOR_MEMORY_CAPACITY - DEFAULT_STACK_SIZE;

/// Stack end address (bottom of stack region)
pub const STACK_END_ADDRESS: u64 = RAM_START_ADDRESS + EMULATOR_MEMORY_CAPACITY;

/// Maximum size for trusted advice data
pub const DEFAULT_MAX_TRUSTED_ADVICE_SIZE: u64 = 4096;

/// Maximum size for untrusted advice data
pub const DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE: u64 = 4096;

/// Maximum input size
pub const DEFAULT_MAX_INPUT_SIZE: u64 = 4096;

/// Maximum output size
pub const DEFAULT_MAX_OUTPUT_SIZE: u64 = 4096;

/// Maximum trace length (2^24 = ~16 million steps)
pub const DEFAULT_MAX_TRACE_LENGTH: u64 = 1 << 24;

// Layout of the witness (where || denotes concatenation):
//     advice || inputs || outputs || panic || termination || padding || RAM
//
// Layout of VM memory:
//     peripheral devices || advice || inputs || outputs || panic || termination || padding || RAM
//
// Notably, we want to be able to map the VM memory address space to witness indices
// using a constant shift, namely (RAM_WITNESS_OFFSET + RAM_START_ADDRESS)

test "constants are valid" {
    const std = @import("std");

    // REGISTER_COUNT must be power of 2
    std.debug.assert(REGISTER_COUNT == 128); // 32 + 96
    std.debug.assert(@popCount(REGISTER_COUNT) == 1);

    // Sanity checks
    std.debug.assert(BYTES_PER_INSTRUCTION == 4);
    std.debug.assert(RAM_START_ADDRESS > 0);
    std.debug.assert(XLEN == 64);
}
