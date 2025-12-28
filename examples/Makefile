# Makefile for Zolt example programs
#
# Prerequisites:
#   - RISC-V toolchain (riscv32-unknown-elf-gcc)
#
# Usage:
#   make all       - Build all examples
#   make clean     - Remove built files
#   make fibonacci - Build fibonacci example

# Use the SP1 toolchain if available
RISCV_CC ?= $(shell which riscv32-unknown-elf-gcc 2>/dev/null || echo riscv32-unknown-elf-gcc)

# Compiler flags
CFLAGS = -march=rv32im -mabi=ilp32 -nostdlib -O2
LDFLAGS = -Wl,-Ttext=0x80000000

# All targets
TARGETS = fibonacci.elf sum.elf

.PHONY: all clean

all: $(TARGETS)

fibonacci.elf: fibonacci.c
	$(RISCV_CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

sum.elf: sum.c
	$(RISCV_CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

# Show disassembly
disasm-fibonacci: fibonacci.elf
	riscv32-unknown-elf-objdump -d $<

disasm-sum: sum.elf
	riscv32-unknown-elf-objdump -d $<
