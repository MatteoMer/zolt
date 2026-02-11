#!/usr/bin/env python3
"""
Gate debug prints behind a compile-time constant.
Replaces std.debug.print with dbg_print throughout the file,
and adds a dbg_print function that's a no-op when debug is disabled.

This is a safe transformation that won't break any code logic.
"""

import sys
import os
import re


def process_file(filepath):
    """Add debug gating to a file."""
    with open(filepath, 'r') as f:
        content = f.read()

    original_count = content.count('std.debug.print(')
    if original_count == 0:
        print(f"  {filepath}: no prints found, skipping")
        return 0

    # Check if already processed
    if 'const debug_verbose' in content:
        print(f"  {filepath}: already processed, skipping")
        return 0

    # Replace all std.debug.print( with dbg( but NOT in the definition we'll add
    new_content = content.replace('std.debug.print(', 'dbg(')

    # Find where to insert the debug function
    # Look for the first `const std = @import("std");` line
    std_import_match = re.search(r'^const std = @import\("std"\);', new_content, re.MULTILINE)
    if not std_import_match:
        # Try to find any const at the top
        first_const = re.search(r'^const ', new_content, re.MULTILINE)
        insert_pos = first_const.start() if first_const else 0
    else:
        insert_pos = std_import_match.end()

    # Insert the debug function right after std import
    debug_fn = """

// Debug output control - set to true to enable verbose debug prints
const debug_verbose = false;
fn dbg(comptime fmt: []const u8, args: anytype) void {
    if (debug_verbose) std.debug.print(fmt, args);
}
"""
    new_content = new_content[:insert_pos] + debug_fn + new_content[insert_pos:]

    with open(filepath, 'w') as f:
        f.write(new_content)

    print(f"  {filepath}: gated {original_count} prints")
    return original_count


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 gate_debug_prints.py <file1.zig> ...")
        sys.exit(1)

    total = 0
    for filepath in sys.argv[1:]:
        if os.path.exists(filepath):
            total += process_file(filepath)
        else:
            print(f"  SKIP: {filepath}")

    print(f"\nTotal gated: {total}")


if __name__ == '__main__':
    main()
