#!/usr/bin/env python3
"""
Remove all std.debug.print(...) calls from Zig source files and fix resulting compiler errors.

Handles:
  - Single-line and multi-line std.debug.print calls
  - Inline for loops: for (...) |b| std.debug.print(...);
  - Unused local constants left behind (removes the line)
  - Unused captures (replaces with _)
  - Discard of unbounded counter (removes the counter capture)
  - Discard of error capture (removes the error capture)
  - Empty blocks cleanup
"""

import re
import sys
import subprocess


def remove_debug_prints(content: str) -> str:
    """Remove all std.debug.print calls."""
    lines = content.split('\n')
    output_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if 'std.debug.print(' not in line:
            output_lines.append(line)
            i += 1
            continue

        idx = line.find('std.debug.print(')
        before_print = line[:idx]

        # Check if there's real code before the print
        before_stripped = before_print.strip()
        
        if before_stripped == '' or re.match(r'^\s*for\s*\(', line):
            # Pure debug print line or for-loop debug print - remove it
            end_line = find_print_end(lines, i, idx + len('std.debug.print'))
            i = end_line + 1
            continue
        
        # There's other code before the print - keep the line
        output_lines.append(line)
        i += 1

    return '\n'.join(output_lines)


def find_print_end(lines, start_line, paren_start_col):
    """Find the line where the print statement ends (matching parens)."""
    paren_depth = 0
    j = start_line

    col = paren_start_col
    while j < len(lines):
        scan_line = lines[j]
        start_col = col if j == start_line else 0

        for c in range(start_col, len(scan_line)):
            ch = scan_line[c]
            if ch == '(':
                paren_depth += 1
            elif ch == ')':
                paren_depth -= 1
                if paren_depth == 0:
                    return j
        j += 1

    return start_line


def clean_empty_blocks(content: str) -> str:
    """Remove empty if(true/false){} blocks and compact blank lines."""
    lines = content.split('\n')

    changed = True
    while changed:
        changed = False
        new_lines = []
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if re.match(r'^if\s*\((true|false)\)\s*\{\s*$', stripped):
                j = i + 1
                while j < len(lines) and lines[j].strip() == '':
                    j += 1
                if j < len(lines) and lines[j].strip() == '}':
                    i = j + 1
                    changed = True
                    continue
            new_lines.append(lines[i])
            i += 1
        lines = new_lines

    # Compact consecutive blank lines (max 2)
    final = []
    blank_count = 0
    for line in lines:
        if line.strip() == '':
            blank_count += 1
            if blank_count <= 2:
                final.append(line)
        else:
            blank_count = 0
            final.append(line)

    return '\n'.join(final)


def fix_compiler_errors(filepath: str, max_iterations: int = 10) -> int:
    """Iteratively fix unused variable/capture errors until compilation succeeds."""
    total_fixes = 0
    
    for iteration in range(max_iterations):
        result = subprocess.run(
            ['zig', 'build'],
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        output = result.stdout + '\n' + result.stderr
        
        # Parse errors for this file only
        errors = []
        seen = set()
        for line in output.split('\n'):
            if filepath not in line or 'error:' not in line:
                continue
            
            m = re.match(r'^(.+?):(\d+):(\d+): error: (.+)$', line)
            if not m:
                continue
            
            f, lineno, col, msg = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
            key = (f, lineno, col, msg)
            if key not in seen:
                seen.add(key)
                errors.append({
                    'file': f,
                    'line': lineno,
                    'col': col,
                    'msg': msg,
                })
        
        if not errors:
            print(f"  Iteration {iteration + 1}: No errors remain. Success!")
            break
        
        print(f"  Iteration {iteration + 1}: {len(errors)} errors to fix")
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Track lines to remove (set of 0-based indices)
        lines_to_remove = set()
        # Track line modifications (line_idx -> new_content)
        line_mods = {}
        
        for err in errors:
            line_idx = err['line'] - 1
            col = err['col'] - 1
            msg = err['msg']
            
            if line_idx >= len(lines):
                continue
            
            line = lines[line_idx]
            
            if 'unused local constant' in msg or 'unused local variable' in msg:
                # Remove the entire const/var line
                # But we need to handle multi-line const declarations
                # Check if this line ends with ';' (single line) or continues
                stripped = line.strip()
                if stripped.endswith(';'):
                    lines_to_remove.add(line_idx)
                else:
                    # Multi-line: find the ending semicolon
                    j = line_idx
                    while j < len(lines):
                        if ';' in lines[j]:
                            for k in range(line_idx, j + 1):
                                lines_to_remove.add(k)
                            break
                        j += 1
                    else:
                        lines_to_remove.add(line_idx)
            
            elif 'unused capture' in msg:
                # Replace capture variable at col with _
                rest = line[col:]
                m2 = re.match(r'(\w+)', rest)
                if m2:
                    varname = m2.group(1)
                    new_line = line[:col] + '_' + line[col + len(varname):]
                    line_mods[line_idx] = new_line
            
            elif 'discard of unbounded counter' in msg:
                # The capture at col is an unbounded counter that's being discarded
                # We need to remove it entirely from the capture list
                # e.g., "for (items, 0..) |item, _i|" -> "for (items, 0..) |item, _|"
                # But Zig says "discard of unbounded counter; omit it instead"
                # So we need to remove ", _i" or ", i" from the capture
                rest = line[col:]
                m2 = re.match(r'_?\w*', rest)
                if m2:
                    capture_name = m2.group(0)
                    # We need to remove this capture and the preceding ", "
                    # Look backwards from col for ", "
                    before = line[:col]
                    if before.rstrip().endswith(','):
                        # Remove the ", capture_name"
                        comma_pos = before.rstrip().rfind(',')
                        # Also find the closing |
                        after_capture = line[col + len(capture_name):]
                        new_line = line[:comma_pos] + after_capture
                        line_mods[line_idx] = new_line
                    else:
                        # Just replace with nothing - remove the capture
                        new_line = line[:col] + line[col + len(capture_name):]
                        line_mods[line_idx] = new_line
            
            elif 'discard of error capture' in msg:
                # Similar to unbounded counter - remove the error capture
                rest = line[col:]
                m2 = re.match(r'_?\w*', rest)
                if m2:
                    capture_name = m2.group(0)
                    before = line[:col]
                    if before.rstrip().endswith(','):
                        comma_pos = before.rstrip().rfind(',')
                        after_capture = line[col + len(capture_name):]
                        new_line = line[:comma_pos] + after_capture
                        line_mods[line_idx] = new_line
                    else:
                        new_line = line[:col] + line[col + len(capture_name):]
                        line_mods[line_idx] = new_line
        
        if not lines_to_remove and not line_mods:
            print(f"  No fixable errors found, stopping.")
            break
        
        # Apply modifications
        for idx, new_content in line_mods.items():
            if idx not in lines_to_remove:
                lines[idx] = new_content
        
        # Remove lines (in reverse order to preserve indices)
        for idx in sorted(lines_to_remove, reverse=True):
            del lines[idx]
        
        fixes = len(lines_to_remove) + len(line_mods)
        total_fixes += fixes
        
        with open(filepath, 'w') as f:
            f.writelines(lines)
        
        print(f"    Removed {len(lines_to_remove)} lines, modified {len(line_mods)} lines")
    
    return total_fixes


def process_file(filepath: str) -> None:
    """Full pipeline: remove prints, clean up, fix compiler errors."""
    print(f"\n{'='*60}")
    print(f"Processing: {filepath}")
    print(f"{'='*60}")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_lines = len(content.split('\n'))
    prints_before = content.count('std.debug.print(')
    
    # Step 1: Remove debug prints
    content = remove_debug_prints(content)
    prints_after = content.count('std.debug.print(')
    
    # Step 2: Clean up empty blocks
    content = clean_empty_blocks(content)
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    after_removal_lines = len(content.split('\n'))
    print(f"  Debug prints: {prints_before} -> {prints_after}")
    print(f"  Lines: {original_lines} -> {after_removal_lines} (removed {original_lines - after_removal_lines})")
    
    if prints_after > 0:
        print(f"  WARNING: {prints_after} debug prints remain!")
    
    # Step 3: Fix compiler errors iteratively
    print(f"\n  Fixing compiler errors...")
    total_fixes = fix_compiler_errors(filepath)
    
    with open(filepath, 'r') as f:
        final_lines = len(f.read().split('\n'))
    
    print(f"\n  Final: {original_lines} -> {final_lines} lines (removed {original_lines - final_lines} total)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python remove_debug_prints.py <file1> [file2] ...")
        sys.exit(1)
    
    for filepath in sys.argv[1:]:
        process_file(filepath)
    
    # Final compilation check
    print(f"\n{'='*60}")
    print("Final compilation check...")
    print(f"{'='*60}")
    result = subprocess.run(['zig', 'build'], capture_output=True, text=True, timeout=300)
    output = (result.stdout + '\n' + result.stderr).strip()
    if result.returncode == 0:
        print("SUCCESS: Clean compilation!")
    else:
        error_count = output.count('error:')
        print(f"ERRORS REMAIN: {error_count} errors")
        # Show first 20 lines
        for line in output.split('\n')[:20]:
            print(f"  {line}")
