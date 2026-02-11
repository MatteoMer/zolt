#!/usr/bin/env python3
"""
Fix unused variable/capture errors from Zig compiler output.
- 'unused local constant' -> prefix variable name with _
- 'unused capture' -> replace capture name with _
"""

import re
import sys
import subprocess


def parse_errors(compiler_output: str, target_file: str):
    """Parse compiler errors for a specific file."""
    errors = []
    for line in compiler_output.split('\n'):
        if target_file not in line:
            continue
        if 'error: unused' not in line:
            continue
        
        # Format: file:line:col: error: unused local constant
        # or: file:line:col: error: unused capture
        m = re.match(r'^(.+?):(\d+):(\d+): error: (unused .+)$', line)
        if m:
            filepath, lineno, col, error_type = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
            errors.append({
                'file': filepath,
                'line': lineno,
                'col': col,
                'type': error_type,
            })
    
    # Deduplicate
    seen = set()
    unique = []
    for e in errors:
        key = (e['file'], e['line'], e['col'])
        if key not in seen:
            seen.add(key)
            unique.append(e)
    
    return unique


def fix_unused_in_file(filepath: str, errors: list):
    """Fix unused variables in a file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Sort errors by line number (descending) so we can modify without shifting
    errors_sorted = sorted(errors, key=lambda e: e['line'], reverse=True)
    
    fixes = 0
    for err in errors_sorted:
        line_idx = err['line'] - 1  # 0-based
        col = err['col'] - 1  # 0-based
        
        if line_idx >= len(lines):
            continue
        
        line = lines[line_idx]
        
        if 'unused local constant' in err['type'] or 'unused local variable' in err['type']:
            # Find the const/var name at the error column and prefix with _
            # Pattern: "const name = ..." or "var name = ..."
            # The col points to the start of the variable name
            # Find the variable name starting at col
            rest = line[col:]
            m = re.match(r'(\w+)', rest)
            if m:
                varname = m.group(1)
                new_line = line[:col] + '_' + line[col:]
                lines[line_idx] = new_line
                fixes += 1
        
        elif 'unused capture' in err['type']:
            # The col points to the capture variable in |...|
            # Replace the identifier at col with _
            rest = line[col:]
            m = re.match(r'(\w+)', rest)
            if m:
                varname = m.group(1)
                new_line = line[:col] + '_' + line[col + len(varname):]
                lines[line_idx] = new_line
                fixes += 1
    
    with open(filepath, 'w') as f:
        f.writelines(lines)
    
    return fixes


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_unused.py <filepath>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    # Run zig build and capture errors
    result = subprocess.run(
        ['zig', 'build'],
        capture_output=True,
        text=True,
        timeout=300,
    )
    
    output = result.stdout + result.stderr
    errors = parse_errors(output, filepath)
    
    if not errors:
        print(f"No unused errors found for {filepath}")
        return
    
    print(f"Found {len(errors)} unused errors in {filepath}")
    
    fixes = fix_unused_in_file(filepath, errors)
    print(f"Applied {fixes} fixes")


if __name__ == '__main__':
    main()
