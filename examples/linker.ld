/* Linker script for Zolt guest programs - matching Jolt's memory layout */

MEMORY
{
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 128M
}

SECTIONS
{
    . = 0x80000000;

    .text : {
        *(.text.boot)
        *(.text._start)
        *(.text)
        *(.text.*)
    } > RAM

    .rodata : {
        *(.rodata)
        *(.rodata.*)
    } > RAM

    .data : {
        *(.data)
        *(.data.*)
        *(.sdata)
        *(.sdata.*)
    } > RAM

    .bss : {
        __bss_start = .;
        *(.bss)
        *(.bss.*)
        *(.sbss)
        *(.sbss.*)
        __bss_end = .;
    } > RAM

    /DISCARD/ : {
        *(.comment)
        *(.note.*)
        *(.riscv.attributes)
    }
}

ENTRY(_start)
