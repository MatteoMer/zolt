#!/usr/bin/env python3
"""
Remove unused local constants/variables flagged by the Zig compiler.
Reads compiler errors from stdin and removes the offending lines.
"""

import sys
import re
import os
from collections import defaultdict


def parse_errors(error_output):
    """Parse Zig compiler unused variable errors.
    Returns dict: filepath -> set of line numbers (1-based)."""
    errors = defaultdict(set)
    for line in error_output.split('\n'):
        m = re.match(r'^(.+\.zig):(\d+):\d+: error: unused local (constant|variable)', line)
        if m:
            filepath = m.group(1)
            lineno = int(m.group(2))
            errors[filepath].add(lineno)
    return errors


def remove_lines(filepath, line_numbers):
    """Remove specified lines from a file.
    Also removes continuation lines for multi-line const declarations."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    to_remove = set()
    for ln in line_numbers:
        idx = ln - 1  # Convert to 0-based
        if idx >= len(lines):
            continue

        # Check if this is a multi-line declaration
        # e.g., "const foo = bar.method(\n    arg1,\n    arg2,\n);"
        line = lines[idx]

        # Add this line for removal
        to_remove.add(idx)

        # Check if the statement continues (no semicolon at end)
        full_line = line.rstrip()
        if not full_line.endswith(';') and not full_line.endswith('}'):
            # Look forward for the rest of the statement
            j = idx + 1
            while j < len(lines):
                full_line = lines[j].rstrip()
                to_remove.add(j)
                if full_line.endswith(';') or full_line.endswith('}'):
                    break
                j += 1

    # Also check for lines that only use removed variables
    # (This is a second pass - won't catch everything but helps)

    result = []
    for i, line in enumerate(lines):
        if i not in to_remove:
            result.append(line)

    with open(filepath, 'w') as f:
        f.writelines(result)

    return len(to_remove)


def main():
    if len(sys.argv) > 1:
        # Read from file
        with open(sys.argv[1], 'r') as f:
            error_output = f.read()
    else:
        # Read from stdin
        error_output = sys.stdin.read()

    errors = parse_errors(error_output)

    total = 0
    for filepath, line_numbers in errors.items():
        if os.path.exists(filepath):
            removed = remove_lines(filepath, line_numbers)
            print(f"  {filepath}: removed {removed} lines ({len(line_numbers)} unused vars)")
            total += removed

    print(f"\nTotal lines removed: {total}")


if __name__ == '__main__':
    main()
