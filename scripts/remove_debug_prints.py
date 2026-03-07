#!/usr/bin/env python3
"""
Remove std.debug.print statements from Zig source files.
Handles single-line and multi-line print calls, including cases where
the print is the body of a for/if/while statement.
"""

import sys
import os
import re


def find_matching_paren(content, start):
    """Find the matching closing paren starting from the opening paren at 'start'.
    Returns the index one past the closing paren."""
    n = len(content)
    depth = 1
    j = start + 1
    in_string = False
    in_char = False

    while j < n and depth > 0:
        c = content[j]

        if in_string:
            if c == '\\' and j + 1 < n:
                j += 2
                continue
            if c == '"':
                in_string = False
            j += 1
            continue

        if in_char:
            if c == '\\' and j + 1 < n:
                j += 2
                continue
            if c == "'":
                in_char = False
            j += 1
            continue

        if c == '"':
            in_string = True
        elif c == "'":
            in_char = True
        elif c == '(':
            depth += 1
        elif c == ')':
            depth -= 1

        j += 1

    return j


def process_content(content):
    """Remove all std.debug.print calls, handling various patterns."""
    lines = content.split('\n')
    result = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check if this line contains std.debug.print
        if 'std.debug.print(' not in stripped:
            result.append(line)
            i += 1
            continue

        # Get the full statement (may span multiple lines)
        # Reconstruct the multi-line statement
        full_stmt = line
        stmt_lines = [i]
        # Check if parens are balanced
        open_parens = full_stmt.count('(') - full_stmt.count(')')
        j = i + 1
        while open_parens > 0 and j < len(lines):
            full_stmt += '\n' + lines[j]
            open_parens += lines[j].count('(') - lines[j].count(')')
            stmt_lines.append(j)
            j += 1

        full_stripped = full_stmt.strip()

        # Case 1: Line is ONLY a std.debug.print(...); call
        # e.g., "    std.debug.print("hello", .{});"
        if re.match(r'^std\.debug\.print\(.*\);$', full_stripped, re.DOTALL):
            # Skip this line entirely
            i = j
            continue

        # Case 2: for (...) |var| std.debug.print(...);
        # The entire for + print should be removed
        m = re.match(r'^for\s*\(.*?\)\s*\|[^|]*\|\s*std\.debug\.print\(', full_stripped)
        if m:
            # Skip the entire for+print
            i = j
            continue

        # Case 3: if (...) std.debug.print(...);  (no braces)
        m = re.match(r'^if\s*\(.*?\)\s*std\.debug\.print\(', full_stripped)
        if m:
            i = j
            continue

        # Case 4: while (...) std.debug.print(...);
        m = re.match(r'^while\s*\(.*?\)\s*std\.debug\.print\(', full_stripped)
        if m:
            i = j
            continue

        # Case 5: Mixed line - there's code before/after the print
        # e.g., "const x = foo; std.debug.print(...);"
        # or "} else std.debug.print(...);"
        # Try to remove just the print call from the line
        # Find the print call and remove it
        idx = line.find('std.debug.print(')
        if idx >= 0:
            before = line[:idx]
            # Find the end of the print call (matching paren + semicolon)
            remaining = full_stmt[idx:]
            paren_start = remaining.index('(')
            after_paren = find_matching_paren(remaining, paren_start)
            # Skip optional semicolon
            rest = remaining[after_paren:]
            if rest.lstrip().startswith(';'):
                semi_pos = rest.index(';')
                rest = rest[semi_pos + 1:]

            cleaned = before.rstrip()
            if rest.strip():
                cleaned += ' ' + rest.strip()

            if cleaned.strip():
                result.append(cleaned)
            i = j
            continue

        # Fallback: keep the line
        result.append(line)
        i += 1

    return '\n'.join(result)


def cleanup_empty_blocks(content):
    """Remove empty if/for blocks left after print removal."""
    lines = content.split('\n')
    changed = True
    while changed:
        changed = False
        new_lines = []
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()

            # Check for empty if blocks: "if (...) {" followed by "}"
            if re.match(r'^if\s*\(.*\)\s*\{$', stripped):
                j = i + 1
                while j < len(lines) and lines[j].strip() == '':
                    j += 1
                if j < len(lines) and lines[j].strip() == '}':
                    # Empty if block, remove it
                    i = j + 1
                    changed = True
                    continue

            # Check for empty for blocks: "for (...) |...| {" followed by "}"
            if re.match(r'^for\s*\(.*\)\s*\|.*\|\s*\{$', stripped):
                j = i + 1
                while j < len(lines) and lines[j].strip() == '':
                    j += 1
                if j < len(lines) and lines[j].strip() == '}':
                    i = j + 1
                    changed = True
                    continue

            # Check for standalone empty blocks: "{" followed by "}"
            # Only remove if previous non-empty line ends with ; or }
            if stripped == '{':
                j = i + 1
                while j < len(lines) and lines[j].strip() == '':
                    j += 1
                if j < len(lines) and lines[j].strip() == '}':
                    # Check previous content
                    prev = ''
                    for k in range(len(new_lines) - 1, -1, -1):
                        if new_lines[k].strip():
                            prev = new_lines[k].strip()
                            break
                    if prev.endswith(';') or prev.endswith('}') or prev == '':
                        i = j + 1
                        changed = True
                        continue

            new_lines.append(lines[i])
            i += 1

        lines = new_lines

    return '\n'.join(lines)


def remove_orphan_comments(content):
    """Remove comments that only describe debug prints."""
    lines = content.split('\n')
    result = []
    for line in lines:
        stripped = line.strip()
        # Remove comments that are clearly about debug prints
        if stripped.startswith('// DEBUG:') or \
           stripped.startswith('// Debug:') or \
           stripped.startswith('// Print ') or \
           stripped.startswith('// Also print') or \
           stripped == '// Print gamma as decimal for comparison with Jolt' or \
           stripped.startswith('// Debug first few claims') or \
           stripped.startswith('// Debug: print transcript state') or \
           stripped.startswith('// DEBUG: Print transcript state'):
            continue
        result.append(line)
    return '\n'.join(result)


def reduce_blank_lines(content):
    """Reduce 3+ consecutive blank lines to 1."""
    lines = content.split('\n')
    result = []
    blank_count = 0
    for line in lines:
        if line.strip() == '':
            blank_count += 1
            if blank_count <= 1:
                result.append(line)
        else:
            blank_count = 0
            result.append(line)
    return '\n'.join(result)


def process_file(filepath):
    """Process a single file."""
    with open(filepath, 'r') as f:
        content = f.read()

    original_count = content.count('std.debug.print(')

    new_content = process_content(content)
    new_content = cleanup_empty_blocks(new_content)
    new_content = remove_orphan_comments(new_content)
    new_content = reduce_blank_lines(new_content)

    remaining_count = new_content.count('std.debug.print(')

    with open(filepath, 'w') as f:
        f.write(new_content)

    removed = original_count - remaining_count
    print(f"  {filepath}: {removed}/{original_count} removed ({remaining_count} remaining)")
    return removed


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 remove_debug_prints.py <file1.zig> ...")
        sys.exit(1)

    total = 0
    for filepath in sys.argv[1:]:
        if os.path.exists(filepath):
            total += process_file(filepath)
        else:
            print(f"  SKIP: {filepath}")

    print(f"\nTotal removed: {total}")
