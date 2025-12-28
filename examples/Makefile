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
RISCV_OBJDUMP ?= $(shell which riscv32-unknown-elf-objdump 2>/dev/null || echo riscv32-unknown-elf-objdump)

# Compiler flags
CFLAGS = -march=rv32im -mabi=ilp32 -nostdlib -O2
LDFLAGS = -Wl,-Ttext=0x80000000

# All targets
TARGETS = fibonacci.elf sum.elf factorial.elf bitwise.elf array.elf gcd.elf

.PHONY: all clean help

all: $(TARGETS)
	@echo "Built all examples:"
	@for t in $(TARGETS); do echo "  $$t"; done

fibonacci.elf: fibonacci.c
	$(RISCV_CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

sum.elf: sum.c
	$(RISCV_CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

factorial.elf: factorial.c
	$(RISCV_CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

bitwise.elf: bitwise.c
	$(RISCV_CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

array.elf: array.c
	$(RISCV_CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

gcd.elf: gcd.c
	$(RISCV_CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

# Show disassembly of each example
disasm-%: %.elf
	$(RISCV_OBJDUMP) -d $<

# Help target
help:
	@echo "Zolt Example Programs"
	@echo "====================="
	@echo ""
	@echo "Examples:"
	@echo "  fibonacci.elf - Compute Fibonacci(10) = 55"
	@echo "  sum.elf       - Compute sum of 1..100 = 5050"
	@echo "  factorial.elf - Compute 10! = 3628800"
	@echo "  bitwise.elf   - Bitwise AND, OR, XOR, shift ops"
	@echo "  array.elf     - Array load/store operations"
	@echo "  gcd.elf       - GCD using M extension (div, rem)"
	@echo ""
	@echo "Targets:"
	@echo "  make all       - Build all examples"
	@echo "  make clean     - Remove built ELF files"
	@echo "  make disasm-X  - Show disassembly of X.elf"
	@echo ""
	@echo "Usage with Zolt:"
	@echo "  zolt run examples/fibonacci.elf"
	@echo "  zolt prove examples/sum.elf -o proof.bin"
	@echo "  zolt verify proof.bin"
