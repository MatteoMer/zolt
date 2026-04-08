/* Linker script for Jolt zkVM guest ELFs.
 *
 * Ported from jolt-core 997c1543 `src/linker.ld.template` with the template
 * variables substituted for the defaults used by `jolt build --mode no-std`:
 *   MEMORY_ORIGIN = 0x80000000
 *   MEMORY_SIZE   = 128 MiB
 *   HEAP_SIZE     = 64 KiB (matches guest lib.rs `heap_size` attribute)
 *   STACK_SIZE    = 64 KiB (comfortable for SHA-256 of a few KiB input)
 *   EMIT_UNWIND_TABLES = false (matches `--backtrace off`)
 *
 * The linker script's heap/stack sizes determine the `__stack_top` /
 * `__heap_end` symbols the guest's `_start` uses to initialise sp and the
 * allocator. The **runtime** `MemoryLayout` computed by Zolt / jolt-bench
 * from `MemoryConfig::stack_size` / `heap_size` is separate — it only
 * bounds-checks memory accesses. Keeping both sides small shrinks the RAM
 * polynomial (which Zolt sizes from `max_remapped_address`), so proving is
 * substantially faster than when the guest spins up a 32 MiB stack region.
 *
 * We discard .eh_frame — keeping them triggers R_RISCV_SET32 relocation
 * overflows at the 0x80000000 base address.
 */
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 128M
}

__heap_size = 64K;
__stack_size = 64K;
__stack_canary_size = 128;

PHDRS
{
    text PT_LOAD FLAGS(5);        /* R-X */
    rodata PT_LOAD FLAGS(4);      /* R-- */
    data PT_LOAD FLAGS(6);        /* RW- (TLS sections will be in this LOAD) */
    tls PT_TLS;                   /* TLS metadata (references sections in data LOAD) */
}

SECTIONS
{
    . = 0x80000000;
    PROVIDE_HIDDEN(__ehdr_start = .);

    .text : {
        *(.text.boot)
        *(.text .text.*)
        . = ALIGN(4);
    } > RAM : text

    .rodata : {
        *(.rodata .rodata.*)
        *(.srodata .srodata.*)
        . = ALIGN(8);
    } > RAM : rodata

    /DISCARD/ : {
        *(.eh_frame)
        *(.eh_frame.*)
        *(.eh_frame_hdr)
    }

    .init_array : {
        PROVIDE_HIDDEN(__init_array_start = .);
        KEEP(*(SORT_BY_INIT_PRIORITY(.init_array.*)))
        KEEP(*(.init_array))
        PROVIDE_HIDDEN(__init_array_end = .);
    } > RAM : data

    .fini_array : {
        PROVIDE_HIDDEN(__fini_array_start = .);
        KEEP(*(SORT_BY_INIT_PRIORITY(.fini_array.*)))
        KEEP(*(.fini_array))
        PROVIDE_HIDDEN(__fini_array_end = .);
    } > RAM : data

    .data : {
        *(.data .data.*)
        PROVIDE(__sdata_start = .);
        *(.sdata .sdata.*)
        . = ALIGN(8);
    } > RAM : data

    PROVIDE(__global_pointer$ = __sdata_start + 0x800);

    .tdata : ALIGN(16) {
        PROVIDE_HIDDEN(__tdata_start = .);
        *(.tdata .tdata.*)
        . = . + 8;
        PROVIDE_HIDDEN(__tdata_end = .);
    } > RAM : data : tls

    .tbss : ALIGN(16) {
        PROVIDE_HIDDEN(__tbss_start = .);
        *(.tbss .tbss.*)
        *(.tcommon)
        . = . + 8;
        PROVIDE_HIDDEN(__tbss_end = .);
    } > RAM : data : tls

    .bss : {
        PROVIDE(__bss_start = .);
        *(.bss .bss.*)
        *(COMMON)
        *(.sbss .sbss.*)
        . = ALIGN(8);
        PROVIDE(__bss_end = .);
    } > RAM : data

    __memory_start = ORIGIN(RAM);
    __ram_start = __memory_start;
    __program_end = __bss_end;

    ASSERT(__program_end <= ORIGIN(RAM) + LENGTH(RAM), "image does not fit in RAM")

    . = ALIGN(8);
    __stack_bottom = .;
    . = . + __stack_canary_size;
    . = . + __stack_size;
    __stack_top = .;

    . = ALIGN(8);
    __heap_start = .;
    . = . + __heap_size;
    __heap_end = .;

    __memory_end = .;
    __ram_end = .;
}
