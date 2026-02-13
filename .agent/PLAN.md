# REMW/DIVW 21-Step Implementation Plan

## Root Cause Found
The tracer (`src/tracer/mod.zig`) falls through to `stepNormal` for REMW/DIVW,
creating only 1 trace step instead of the 21 needed. The preprocessing and bytecode
entry code was already done, but the trace emission was missing entirely.

## Opcode Assignments (for distinguishing virtual instructions)
- VirtualSRAI: opcode=0x5B, funct3=5 (VirtualSRLI keeps funct3=0)
- VirtualAssertValidDiv0: opcode=0x22, funct3=1 (VirtualAssertEQ keeps funct3=0)
- VirtualChangeDivisorW: opcode=0x3b, funct3=6, funct7=0x01

## Files to modify:
1. src/tracer/mod.zig - stepREMWDIVW, instruction builders, dispatch
2. src/zkvm/instruction/lookup_trace.zig - recording functions
3. src/zkvm/spartan/stage5_prover.zig - getLookupTableIndex
4. src/zkvm/spartan/stage6_prover.zig - funct3 updates
5. jolt-core/src/zkvm/bytecode/read_raf_checking.rs - opcode updates
