# Research Report: Jolt-Inline Support for Zolt

## Executive Summary

Jolt-inline is an optimization framework that replaces high-level operations (hashing, bigint, ECC) with optimized sequences of virtual RISC-V instructions inside the Jolt zkVM trace. Guest programs compiled with Jolt's SDK emit custom RISC-V opcodes (`0x0B` / `0x2B`) in the ELF. At trace time, the Jolt tracer expands each custom instruction into a virtual instruction sequence using extended registers (48-63) and custom lookup-friendly instructions (e.g., `VirtualROTRI`). The result: 3-6x cycle reduction for crypto operations, all proved within the standard Jolt lookup system with zero separate constraint systems.

**To support jolt-inline ELFs in Zolt without modifying Jolt source**, Zolt must:
1. Decode opcodes `0x0B`/`0x2B` in the instruction decoder
2. Implement each inline's sequence builder natively in Zig
3. Add missing virtual instructions (`VirtualROTRI`, `VirtualROTRIW`, `VirtualAdvice`, `VirtualXorRot`)
4. Add their corresponding lookup tables to the prover
5. Expand the virtual register allocator to support inline allocation (registers 48-63)

---

## Part 1: What is Jolt-Inline?

### 1.1 Architecture

Jolt-inline is **not** a traditional precompile with a separate circuit. Instead, it's a virtual instruction expansion system:

```
Guest code (Rust, no_std)
    |
    v
SDK API (e.g., Sha256::digest())
    |
    v
Inline assembly: .insn r 0x0B, funct3, funct7, x0, rs1, rs2
    |
    v
RISC-V ELF binary (contains 0x0B custom instructions in .text)
    |
    v
Tracer encounters 0x0B instruction
    |
    v
Registry lookup: (opcode=0x0B, funct3, funct7) -> sequence_builder_fn
    |
    v
Sequence builder generates Vec<VirtualInstruction> (100-1000+ instructions)
    |
    v
Virtual instructions execute in-trace using registers 48-63
    |
    v
Standard Jolt lookup-based proving (no special circuits)
```

### 1.2 ELF Encoding

Custom inline instructions are encoded in the standard RISC-V R-type format, but parsed as `FormatInline`:

```
Bits [31:25]  funct7   - Operation family (SHA2=0x00, Keccak=0x01, Blake2=0x02, ...)
Bits [24:20]  rs2      - Pointer register (input data)
Bits [19:15]  rs1      - Pointer register (state/output)
Bits [14:12]  funct3   - Sub-operation variant
Bits [11:7]   rd/rs3   - Output pointer register (always x0 in practice, remapped to virtual)
Bits [6:0]    opcode   - 0x0B (core) or 0x2B (user-defined)
```

**Critical difference from FormatR**: `FormatInline` treats `rd` as `rs3` (a third source/pointer register). The inline never writes to real RISC-V registers -- it only reads memory pointers from rs1/rs2/rs3 and operates on memory + virtual registers.

Source: `/Users/matteo/projects/jolt/tracer/src/instruction/format/format_inline.rs`

### 1.3 Available Inlines

| Inline          | Opcode | funct7 | funct3 | Description                              |
|-----------------|--------|--------|--------|------------------------------------------|
| SHA256          | 0x0B   | 0x00   | 0x00   | SHA-256 compression with existing state  |
| SHA256INIT      | 0x0B   | 0x00   | 0x01   | SHA-256 compression with IV constants    |
| KECCAK256       | 0x0B   | 0x01   | 0x00   | Keccak-256 permutation                   |
| BLAKE2B         | 0x0B   | 0x02   | 0x00   | BLAKE2b compression                      |
| BLAKE3          | 0x0B   | 0x03   | 0x00   | BLAKE3 compression                       |
| BLAKE3KEYED64   | 0x0B   | 0x03   | 0x01   | BLAKE3 keyed compression                 |
| BIGINT256_MUL   | 0x0B   | 0x04   | 0x00   | 256-bit bigint multiplication            |
| SECP256K1_MULQ  | 0x0B   | 0x05   | 0x00   | secp256k1 base field multiplication      |
| SECP256K1_SQUAREQ | 0x0B | 0x05   | 0x01   | secp256k1 base field squaring            |
| SECP256K1_DIVQ  | 0x0B   | 0x05   | 0x02   | secp256k1 base field division            |
| SECP256K1_MULR  | 0x0B   | 0x05   | 0x04   | secp256k1 scalar field multiplication    |
| SECP256K1_SQUARER | 0x0B | 0x05   | 0x05   | secp256k1 scalar field squaring          |
| SECP256K1_DIVR  | 0x0B   | 0x05   | 0x06   | secp256k1 scalar field division          |
| SECP256K1_GLVR  | 0x0B   | 0x05   | 0x07   | secp256k1 GLV decomposition              |
| GRUMPKIN_DIVQ   | 0x0B   | 0x06   | 0x00   | Grumpkin base field division             |
| GRUMPKIN_DIVR   | 0x0B   | 0x06   | 0x01   | Grumpkin scalar field division           |

Source: `jolt-inlines/*/src/lib.rs` and `book/src/how/optimizations/inlines.md`

### 1.4 Performance

| Hash Function | Without Inline | With Inline  | Speedup  |
|---------------|---------------|--------------|----------|
| SHA-256       | 10,414,653 cy | 1,765,207 cy | **5.9x** |
| Keccak-256    | 2,556,519 cy  | 848,224 cy   | **3.01x**|
| Blake2B       | 968,562 cy    | 340,787 cy   | **2.85x**|

---

## Part 2: How Inlines Work Internally

### 2.1 Virtual Register Layout (Jolt)

Total: 128 registers (32 real + 96 virtual). Constants from `common/src/constants.rs`:

```
RISCV_REGISTER_COUNT = 32
VIRTUAL_REGISTER_COUNT = 96
REGISTER_COUNT = 128  (must be power of 2)
```

Layout:
```
 0-31:  Standard RISC-V x0-x31
32-39:  Reserved (LR/SC reservation, CSRs: mtvec, mscratch, mepc, mcause, mtval, mstatus)
40-47:  Instruction virtual sequences (allocate()) -- used by SLL, SRL, SRA decompositions
48-63:  Inline virtual sequences (allocate_for_inline()) -- used by SHA256, etc.
64-127: Unused (available for future expansion)
```

Source: `/Users/matteo/projects/jolt/tracer/src/utils/virtual_registers.rs`

**Critical rule**: All inline registers (48-63) must be zeroed at the end of each inline sequence via ADDI rd, x0, 0 instructions. This is enforced by `finalize_inline()`.

### 2.2 Inline Registration (Rust side)

```rust
// Global registry: (opcode, funct3, funct7) -> (name, sequence_fn, advice_fn)
static INLINE_REGISTRY: LazyLock<RwLock<HashMap<InlineKey, InlineRegistryValue>>>

pub fn register_inline(
    opcode: u32,        // 0x0B or 0x2B only
    funct3: u32,        // 0-7
    funct7: u32,        // 0-127
    name: &str,
    inline_sequence_fn: InlineSequenceFunction,  // builds Vec<Instruction>
    advice_fn: Option<AdviceFunction>,           // non-deterministic hints
) -> Result<(), String>
```

Each inline crate auto-registers via `#[ctor::ctor]` constructor functions.

Source: `/Users/matteo/projects/jolt/tracer/src/instruction/inline.rs:57-94`

### 2.3 Trace Execution Flow

When the tracer encounters opcode 0x0B/0x2B:

1. Parse as `INLINE` struct with `FormatInline` operands
2. If `rs3 == 0` (rd field is x0), remap to a virtual register (allocate())
3. Look up `(opcode, funct3, funct7)` in registry
4. If advice_fn exists: call it first to get `VecDeque<u64>` of non-deterministic values
5. Call `sequence_builder_fn(InstrAssembler, FormatInline)` -> `Vec<Instruction>`
6. Execute each virtual instruction in sequence, setting `virtual_sequence_remaining` counts
7. `finalize_inline()` appends ADDI x0 instructions to zero registers 48-63

Source: `/Users/matteo/projects/jolt/tracer/src/instruction/inline.rs:202-269`

### 2.4 InstrAssembler

The `InstrAssembler` is the tool used by sequence builders to emit virtual instructions:

```rust
pub struct InstrAssembler {
    address: u64,
    is_compressed: bool,
    xlen: Xlen,
    sequence: Vec<Instruction>,
    has_inline_instr_format: bool,  // true for inlines, false for instruction sequences
    allocator: VirtualRegisterAllocator,
}
```

Key methods:
- `emit_r::<ADD>(rd, rs1, rs2)` - R-type
- `emit_i::<ADDI>(rd, rs1, imm)` - I-type (supports 64-bit immediates!)
- `emit_s::<SW>(rs1, rs2, offset)` - Store
- `emit_ld::<LD>(rd, rs1, offset)` - Load
- `bin::<XOR, XORI>(a, b, rd, fold)` - Generic binary with constant folding
- `rotri32(rs1, shamt, rd)` - Rotate right (emits `VirtualROTRI` or `VirtualROTRIW`)
- `finalize_inline()` - Zero all allocated inline registers, set sequence metadata

**Validation**: When `has_inline_instr_format == true`, the assembler panics if any emitted instruction writes to a register < 48 (except x0). This enforces that inlines never corrupt real or reserved virtual registers.

Source: `/Users/matteo/projects/jolt/tracer/src/utils/inline_helpers.rs`

### 2.5 R1CS Circuit Flags

Each instruction in the trace has circuit flags that drive R1CS constraints. The flags relevant to inlines:

```
VirtualInstruction (7):       true if virtual_sequence_remaining is Some(_)
DoNotUpdateUnexpandedPC (9):  true if not first in sequence (keeps unexpanded PC constant)
IsFirstInSequence (12):       true for first instruction in a virtual sequence
IsLastInSequence (13):        true for last instruction in a virtual sequence
```

R1CS constraint for PC update:
```
NextUnexpandedPC = UnexpandedPC + 4 - 4*DoNotUpdateUnexpandedPC - 2*IsCompressed
```

Within an inline virtual sequence, `DoNotUpdateUnexpandedPC=1` for all instructions except the first, so the unexpanded PC advances only once for the entire inline.

Source: `/Users/matteo/projects/jolt/jolt-core/src/zkvm/r1cs/constraints.rs`

### 2.6 Custom Virtual Instructions Used by Inlines

Inlines use standard RISC-V instructions (ADD, XOR, AND, ADDI, LW, SW, etc.) plus custom virtual instructions:

| Instruction      | Purpose                                        | MLE Structure |
|-----------------|------------------------------------------------|---------------|
| VirtualROTRI    | Rotate right immediate (64-bit)                | Structured    |
| VirtualROTRIW   | Rotate right immediate (32-bit word)           | Structured    |
| VirtualAdvice   | Non-deterministic hint value from prover       | Identity      |
| VirtualXorRot   | XOR followed by rotate (combined)              | Structured    |
| VirtualXorRotW  | XOR + rotate for 32-bit words                  | Structured    |

These are **not** standard RISC-V -- they're Jolt-specific instructions that have structured MLEs for efficient lookup verification.

### 2.7 Advice Functions

Some inlines need non-deterministic advice (e.g., field division requires the quotient as a hint, then verifies `a * result == b`):

```rust
pub type AdviceFunction =
    Box<dyn Fn(InstrAssembler, FormatInline, &mut Cpu) -> VecDeque<u64> + Send + Sync>;
```

The advice function reads the current CPU state (memory via pointers in rs1/rs2), computes the non-deterministic result, and returns advice values. The sequence builder then places these into `VirtualAdvice` instructions, which are verified via assertions in the sequence.

Used by: `SECP256K1_DIVQ`, `SECP256K1_DIVR`, `SECP256K1_GLVR_ADV`, `GRUMPKIN_DIVQ_ADV`, `GRUMPKIN_DIVR_ADV`

---

## Part 3: Zolt Current State

### 3.1 Instruction Decoder

Zolt's decoder at `src/zkvm/instruction_decoder.zig` handles standard RISC-V opcodes:
- 0b0110111 (LUI), 0b0010111 (AUIPC), 0b1101111 (JAL), 0b1100111 (JALR)
- 0b1100011 (Branch), 0b0000011 (Load), 0b0100011 (Store)
- 0b0010011 (OP-IMM), 0b0110011 (OP), 0b0011011 (OP-IMM-32), 0b0111011 (OP-32)
- 0b0001111 (FENCE), 0b1110011 (SYSTEM/ECALL)

**Not handled**: Opcodes `0x0B` (0b0001011) and `0x2B` (0b0101011) fall through to `else => .UNIMPL`.

### 3.2 Virtual Instruction Support

Zolt already supports virtual instruction sequences for:
- W-extension decompositions (ADDW -> ADD + VirtualSignExtendWord)
- Shift decompositions (SLL -> VirtualPow2 + MUL)
- Division decompositions (REMUW -> 12-step sequence, DIVW/REMW -> 21-step)
- Sub-word loads/stores (LB, SB -> multi-step)

The trace step struct already has fields for:
- `virtual_sequence_remaining: u16`
- `is_first_in_sequence: bool`
- `is_last_in_sequence: bool`

### 3.3 Virtual Register Count

Zolt matches Jolt: `VIRTUAL_REGISTER_COUNT = 96`, `REGISTER_COUNT = 128`.

But Zolt's emulator register file and the virtual register allocator logic need verification for inline-specific allocation (registers 48-63).

### 3.4 Missing Virtual Instructions

Zolt does NOT have:
- `VirtualROTRI` / `VirtualROTRIW` (rotate right immediate) -- used extensively by SHA256, Blake2/3
- `VirtualAdvice` for non-deterministic hints -- used by Secp256k1 division operations
- `VirtualXorRot` / `VirtualXorRotW` -- combined XOR+rotate used by some inlines

### 3.5 Circuit Flags

Zolt already has all the needed circuit flags in `src/zkvm/instruction/mod.zig`:
```zig
pub const CircuitFlags = enum(u8) {
    DoNotUpdateUnexpandedPC = 9,
    IsFirstInSequence = 12,
    IsLastInSequence = 13,
    VirtualInstruction = 7,
    // ... all 14 flags present
};
```

The R1CS constraints in `src/zkvm/r1cs/constraints.zig` already handle `DoNotUpdateUnexpandedPC` correctly.

---

## Part 4: What Needs to Change in Zolt

### 4.1 Instruction Decoder (src/zkvm/instruction_decoder.zig)

Add cases for opcodes `0x0B` and `0x2B`:

```zig
0b0001011, 0b0101011 => {  // Custom-0 (0x0B) and Custom-1 (0x2B)
    variant = .INLINE;
    operands = .{ .FormatInline = .{
        .rs1 = rs1,
        .rs2 = rs2,
        .rs3 = rd,     // rd field maps to rs3 in FormatInline
        .funct3 = funct3,
        .funct7 = funct7,
        .opcode = opcode,
    }};
}
```

This requires adding:
- `InstructionVariant.INLINE` to the variant enum
- `Operands.FormatInline` to the operands union
- The `FormatInline` struct definition

### 4.2 Inline Registry (new module)

Create an inline registry that maps `(opcode, funct3, funct7)` to a Zig sequence builder function:

```zig
pub const InlineKey = struct { opcode: u32, funct3: u32, funct7: u32 };
pub const SequenceBuilderFn = *const fn (FormatInline, *Emulator) []TraceStep;
pub const AdviceFn = ?*const fn (FormatInline, *Emulator) []u64;

pub const InlineRegistry = struct {
    entries: std.AutoHashMap(InlineKey, RegistryEntry),
    // ...
};
```

### 4.3 Sequence Builders (new modules)

Each inline needs a Zig implementation of its sequence builder. The builders generate trace steps (not actual RISC-V execution) using virtual registers 48-63.

**Priority order** (by usage/impact):
1. SHA256 + SHA256INIT (most commonly used, 5.9x speedup)
2. KECCAK256 (3.01x speedup)
3. BLAKE2B (2.85x speedup)
4. BIGINT256_MUL (foundation for ECC operations)
5. SECP256K1 operations (ECDSA verification)
6. BLAKE3 variants
7. GRUMPKIN operations

Each builder must:
- Read input from memory via pointers in rs1/rs2
- Allocate virtual registers 48-63
- Emit virtual instruction trace steps (ADD, XOR, AND, ADDI, LW/LD, SW/SD, VirtualROTRI, etc.)
- Zero all used virtual registers at the end
- Set `virtual_sequence_remaining`, `is_first_in_sequence`, `is_last_in_sequence` correctly

### 4.4 New Virtual Instructions

Add to `InstructionVariant` enum and implement lookup tables:

**VirtualROTRI**: Rotate right immediate
- Input: (value, bitmask) where bitmask encodes rotation amount via trailing zeros
- Output: value rotated right by trailing_zeros(bitmask) positions
- MLE: Structured (decomposable as shift + OR)

**VirtualROTRIW**: Same as ROTRI but for 32-bit words (sign-extends result to 64-bit)

**VirtualAdvice**: Non-deterministic prover hint
- The trace step carries an `advice` value that the prover supplies
- No lookup needed (identity function)

**VirtualXorRot / VirtualXorRotW**: Combined XOR + rotate
- These may or may not be strictly needed -- need to verify which inlines actually use them vs. decomposing into separate XOR + ROTRI

### 4.5 Emulator Changes (src/tracer/mod.zig)

The emulator's `step()` function needs a new code path for INLINE instructions:

```zig
fn stepInline(self: *Emulator, instruction: u32) !void {
    const decoded = FormatInline.parse(instruction);
    const opcode = instruction & 0x7f;
    const funct3 = (instruction >> 12) & 0x7;
    const funct7 = (instruction >> 25) & 0x7f;

    // Look up sequence builder
    const builder = self.inline_registry.get(.{ .opcode = opcode, .funct3 = funct3, .funct7 = funct7 })
        orelse return error.UnknownInline;

    // If advice function exists, compute advice values first
    var advice = if (builder.advice_fn) |afn| afn(decoded, self) else null;

    // Generate and execute virtual instruction sequence
    const sequence = builder.sequence_fn(decoded, self, advice);

    // Each step in the sequence becomes a trace entry with proper metadata
    for (sequence, 0..) |step, i| {
        step.virtual_sequence_remaining = sequence.len - i - 1;
        step.is_first_in_sequence = (i == 0);
        step.is_last_in_sequence = (i == sequence.len - 1);
        self.trace.append(step);
    }
}
```

### 4.6 Lookup Table Extensions

Each new virtual instruction needs a corresponding lookup implementation in `src/zkvm/instruction/lookups.zig`:

- `VirtualRotriLookup`: rotate right by encoded immediate
- `VirtualRotriwLookup`: 32-bit variant
- `VirtualAdviceLookup`: identity (output = advice value)
- Potentially `VirtualXorRotLookup` if used

### 4.7 Bytecode Preprocessing

The bytecode preprocessor at `src/zkvm/bytecode_preprocessing.zig` needs to handle the new INLINE variant. The preprocessor computes per-instruction metadata used by the prover, including circuit flags.

### 4.8 Witness Generation

The witness generator at `src/tracer/witness.zig` needs to handle INLINE-expanded trace steps. Since inline steps expand to standard instructions (ADD, XOR, LW, etc.) plus the new virtual instructions, most of the witness logic should work -- only the new virtual instructions need new witness code paths.

---

## Part 5: Compatibility Constraints

### 5.1 What We Cannot Change

**Jolt source is read-only.** This means:
- The ELF format is fixed -- we must decode exactly the instructions Jolt's compiler emits
- The sequence builder logic must produce **identical traces** to Jolt's Rust implementation
- The circuit flags, R1CS constraints, and proof format must match exactly
- Virtual register layout (48-63 for inlines) is fixed
- The `finalize_inline()` zeroing behavior is required

### 5.2 What Must Match Exactly

For proofs to verify against Jolt's verifier:
1. **Expanded trace**: Each inline must expand to the exact same number and type of virtual instructions as Jolt
2. **Register allocation**: Must use the same virtual registers in the same order
3. **Memory access pattern**: Loads and stores must happen at the same addresses in the same order
4. **Circuit flags per step**: VirtualInstruction, DoNotUpdateUnexpandedPC, IsFirstInSequence, IsLastInSequence must match
5. **Lookup entries**: Each instruction's lookup table query must produce identical values

### 5.3 Testing Strategy

1. **Trace comparison**: Compile a Jolt guest with inline (e.g., sha2-chain), run through both Jolt's tracer and Zolt's tracer, compare traces step-by-step
2. **Proof verification**: Generate proof with Zolt for inline-using ELF, verify with Jolt's verifier
3. **Per-inline unit tests**: For each inline, test that the Zig sequence builder produces the same virtual instruction sequence as the Rust version

### 5.4 Incremental Approach

Phase 1: **Decode + Emulate** (no proving)
- Add INLINE opcode decoding
- Implement SHA256 sequence builder
- Run sha2-chain example and verify correct output

Phase 2: **Add Virtual Instructions**
- Implement VirtualROTRI/VirtualROTRIW with lookup tables
- Implement VirtualAdvice
- Verify trace matches Jolt's trace exactly

Phase 3: **Prove**
- Wire inline trace steps through the full 7-stage prover
- Verify proof against Jolt's verifier
- Add remaining inlines (Keccak, Blake2/3, BigInt, Secp256k1)

---

## Part 6: Key Source Files Reference

### Jolt (read-only)

| File | Purpose |
|------|---------|
| `tracer/src/instruction/inline.rs` | INLINE struct, registry, trace() method |
| `tracer/src/instruction/format/format_inline.rs` | FormatInline parsing |
| `tracer/src/utils/inline_helpers.rs` | InstrAssembler for building sequences |
| `tracer/src/utils/virtual_registers.rs` | Virtual register allocator (48-63 for inlines) |
| `common/src/constants.rs` | VIRTUAL_REGISTER_COUNT=96, REGISTER_COUNT=128 |
| `jolt-inlines/sha2/src/sequence_builder.rs` | SHA256 virtual instruction sequence |
| `jolt-inlines/sha2/src/host.rs` | SHA256 registration + advice |
| `jolt-inlines/sha2/src/sdk.rs` | SHA256 guest SDK (inline assembly) |
| `jolt-inlines/keccak256/src/` | Keccak256 implementation |
| `jolt-inlines/bigint/src/multiplication/` | BigInt256 multiplication |
| `jolt-inlines/secp256k1/src/` | Secp256k1 field operations |
| `jolt-core/src/zkvm/instruction/virtual_rotri.rs` | VirtualROTRI lookup table |
| `jolt-core/src/zkvm/r1cs/constraints.rs` | R1CS constraints for virtual sequences |
| `book/src/how/optimizations/inlines.md` | Official documentation |

### Zolt (to modify)

| File | Change Needed |
|------|---------------|
| `src/zkvm/instruction_decoder.zig` | Add 0x0B/0x2B opcode decoding |
| `src/zkvm/jolt_instruction.zig` | Add INLINE variant, FormatInline operands |
| `src/zkvm/instruction/mod.zig` | Add INLINE to CircuitFlags handling |
| `src/zkvm/instruction/lookups.zig` | Add VirtualROTRI lookup |
| `src/tracer/mod.zig` | Add inline expansion in step() |
| `src/zkvm/r1cs/constraints.zig` | Already handles DoNotUpdateUnexpandedPC (verify) |
| `src/zkvm/bytecode_preprocessing.zig` | Handle INLINE in preprocessing |
| `src/tracer/witness.zig` | Handle new virtual instructions |
| `src/common/constants.zig` | Already correct (VIRTUAL_REGISTER_COUNT=96) |
| NEW: `src/tracer/inlines/` | Sequence builders for each inline |
| NEW: `src/tracer/inlines/sha256.zig` | SHA256 sequence builder |
| NEW: `src/tracer/inlines/keccak256.zig` | Keccak256 sequence builder |
| NEW: `src/tracer/inlines/registry.zig` | Inline dispatch table |
