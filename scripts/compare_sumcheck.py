#!/usr/bin/env python3
"""
Compare Jolt vs Zolt sumcheck verification logs side-by-side.
Shows all values in hex, both BE and LE for easy endianness debugging.
"""

import re
import sys

# ANSI colors
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'
DIM = '\033[2m'

def parse_zolt_bytes(line: str) -> list:
    """Extract byte array from Zolt log like { 32, 30, 11, ... }"""
    match = re.search(r'\{\s*([\d,\s]+)\s*\}', line)
    if match:
        nums = match.group(1).split(',')
        return [int(n.strip()) for n in nums if n.strip()]
    return []

def parse_jolt_bytes(line: str) -> list:
    """Extract byte array from Jolt log like [130, 56, 27, ...] or [e1, f2, ...]"""
    match = re.search(r'\[([\w\d,\s]+)\]', line)
    if match:
        nums = match.group(1).split(',')
        result = []
        for n in nums:
            n = n.strip()
            if n:
                try:
                    # Try decimal first
                    result.append(int(n))
                except ValueError:
                    # Try hex
                    try:
                        result.append(int(n, 16))
                    except ValueError:
                        pass
        return result
    return []

def parse_jolt_decimal(line: str) -> int:
    """Extract decimal number from Jolt log"""
    match = re.search(r'=\s*(\d{10,})', line)
    if match:
        return int(match.group(1))
    return None

def bytes_to_hex(b: list, prefix: str = "") -> str:
    """Convert byte list to hex string"""
    if not b:
        return "(none)"
    return prefix + ''.join(f'{x:02x}' for x in b)

def int_to_hex(n: int) -> str:
    """Convert int to 64-char hex (32 bytes)"""
    if n is None:
        return "(none)"
    return f'{n:064x}'

def reverse_bytes(b: list) -> list:
    """Reverse byte order"""
    return list(reversed(b))

def compare_value(name: str, zolt_bytes: list, jolt_bytes: list = None, jolt_decimal: int = None):
    """Compare and display a value from both systems"""
    print(f"\n{CYAN}{BOLD}--- {name} ---{RESET}")

    # Zolt value
    if zolt_bytes:
        zolt_be = bytes_to_hex(zolt_bytes)
        zolt_le = bytes_to_hex(reverse_bytes(zolt_bytes))
        zolt_int_be = int.from_bytes(bytes(zolt_bytes), 'big')
        zolt_int_le = int.from_bytes(bytes(zolt_bytes), 'little')

        print(f"{YELLOW}ZOLT:{RESET}")
        print(f"  raw bytes: {zolt_bytes[:8]}...{zolt_bytes[-4:]}" if len(zolt_bytes) > 12 else f"  raw bytes: {zolt_bytes}")
        print(f"  as-is hex: {zolt_be}")
        print(f"  reversed:  {zolt_le}")
        print(f"  {DIM}int(as-is/BE): {zolt_int_be}{RESET}")
        print(f"  {DIM}int(rev/LE):   {zolt_int_le}{RESET}")
    else:
        print(f"{YELLOW}ZOLT:{RESET} (not found)")

    # Jolt value
    if jolt_bytes:
        jolt_be = bytes_to_hex(jolt_bytes)
        jolt_le = bytes_to_hex(reverse_bytes(jolt_bytes))
        jolt_int_be = int.from_bytes(bytes(jolt_bytes), 'big')
        jolt_int_le = int.from_bytes(bytes(jolt_bytes), 'little')

        print(f"{BLUE}JOLT:{RESET}")
        print(f"  raw bytes: {jolt_bytes[:8]}...{jolt_bytes[-4:]}" if len(jolt_bytes) > 12 else f"  raw bytes: {jolt_bytes}")
        print(f"  as-is hex: {jolt_be}")
        print(f"  reversed:  {jolt_le}")
        print(f"  {DIM}int(as-is/BE): {jolt_int_be}{RESET}")
        print(f"  {DIM}int(rev/LE):   {jolt_int_le}{RESET}")
    elif jolt_decimal is not None:
        jolt_bytes_be = list(jolt_decimal.to_bytes(32, 'big'))
        jolt_bytes_le = list(jolt_decimal.to_bytes(32, 'little'))

        print(f"{BLUE}JOLT:{RESET}")
        print(f"  decimal:   {jolt_decimal}")
        print(f"  hex (BE):  {int_to_hex(jolt_decimal)}")
        print(f"  hex (LE):  {bytes_to_hex(jolt_bytes_le)}")
        print(f"  bytes BE:  {jolt_bytes_be[:8]}...{jolt_bytes_be[-4:]}")
        print(f"  bytes LE:  {jolt_bytes_le[:8]}...{jolt_bytes_le[-4:]}")
    else:
        print(f"{BLUE}JOLT:{RESET} (not found)")

    # Check for matches
    if zolt_bytes and jolt_bytes:
        if zolt_bytes == jolt_bytes:
            print(f"{GREEN}✓ MATCH (same byte order){RESET}")
            return True
        elif zolt_bytes == reverse_bytes(jolt_bytes):
            print(f"{GREEN}✓ MATCH (reversed/endianness){RESET}")
            return True
        else:
            print(f"{RED}✗ NO MATCH{RESET}")
            return False
    elif zolt_bytes and jolt_decimal is not None:
        zolt_int_le = int.from_bytes(bytes(zolt_bytes), 'little')
        zolt_int_be = int.from_bytes(bytes(zolt_bytes), 'big')
        if zolt_int_le == jolt_decimal:
            print(f"{GREEN}✓ MATCH (zolt LE = jolt){RESET}")
            return True
        elif zolt_int_be == jolt_decimal:
            print(f"{GREEN}✓ MATCH (zolt BE = jolt){RESET}")
            return True
        else:
            print(f"{RED}✗ NO MATCH{RESET}")
            return False
    return None

def extract_value(log: str, pattern: str, is_zolt: bool = True) -> tuple:
    """Extract value matching pattern, return (bytes, decimal)"""
    for line in log.split('\n'):
        if pattern in line:
            if is_zolt:
                b = parse_zolt_bytes(line)
                if b:
                    return b, None
            else:
                b = parse_jolt_bytes(line)
                if b:
                    return b, None
                d = parse_jolt_decimal(line)
                if d:
                    return None, d
    return None, None

def extract_preamble_value(log: str, pattern: str) -> str:
    """Extract a preamble value as string"""
    for line in log.split('\n'):
        if pattern in line:
            # Extract the value after the pattern
            match = re.search(pattern + r'[=:]\s*(.+)', line)
            if match:
                return match.group(1).strip()
            # Just return the whole line after pattern
            idx = line.find(pattern)
            if idx >= 0:
                return line[idx + len(pattern):].strip()
    return None

def compare_preamble(zolt_log: str, jolt_log: str):
    """Compare preamble values between Zolt and Jolt"""
    print(f"\n{BOLD}{'='*70}")
    print(f"  FIAT-SHAMIR PREAMBLE COMPARISON")
    print(f"{'='*70}{RESET}")

    preamble_fields = [
        ("max_input_size", "appendU64: max_input_size"),
        ("max_output_size", "appendU64: max_output_size"),
        ("memory_size", "appendU64: memory_size"),
        ("inputs.len", "appendBytes: inputs.len"),
        ("outputs.len", "appendBytes: outputs.len"),
        ("panic", "appendU64: panic"),
        ("ram_K", "appendU64: ram_K"),
        ("trace_length", "appendU64: trace_length"),
    ]

    all_match = True
    for name, pattern in preamble_fields:
        zolt_val = extract_preamble_value(zolt_log, pattern)
        jolt_val = extract_preamble_value(jolt_log, pattern)

        # Clean up values
        if zolt_val:
            zolt_val = zolt_val.split()[0] if zolt_val else None
        if jolt_val:
            jolt_val = jolt_val.split()[0] if jolt_val else None

        match = zolt_val == jolt_val
        if not match:
            all_match = False

        status = f"{GREEN}✓{RESET}" if match else f"{RED}✗{RESET}"
        print(f"  {status} {name:20}: ZOLT={zolt_val or '(none)':15} JOLT={jolt_val or '(none)':15}")

    # Special handling for inputs/outputs content
    print(f"\n{CYAN}--- Input/Output Content ---{RESET}")

    # Extract Zolt inputs
    zolt_inputs_match = re.search(r'\[ZOLT PREAMBLE\] appendBytes: inputs\.len=(\d+)', zolt_log)
    zolt_outputs_match = re.search(r'\[ZOLT PREAMBLE\] appendBytes: outputs\.len=(\d+)', zolt_log)

    # Extract Jolt inputs
    jolt_inputs_match = re.search(r'\[JOLT PREAMBLE\]   inputs=\[([^\]]*)\]', jolt_log)
    jolt_outputs_match = re.search(r'\[JOLT PREAMBLE\]   outputs=\[([^\]]*)\]', jolt_log)

    zolt_inputs_len = int(zolt_inputs_match.group(1)) if zolt_inputs_match else 0
    zolt_outputs_len = int(zolt_outputs_match.group(1)) if zolt_outputs_match else 0

    jolt_inputs = jolt_inputs_match.group(1) if jolt_inputs_match else ""
    jolt_outputs = jolt_outputs_match.group(1) if jolt_outputs_match else ""

    print(f"  {YELLOW}ZOLT inputs:{RESET}  len={zolt_inputs_len}, content=(empty)")
    print(f"  {BLUE}JOLT inputs:{RESET}  [{jolt_inputs}]")

    print(f"  {YELLOW}ZOLT outputs:{RESET} len={zolt_outputs_len}, content=(empty)")
    print(f"  {BLUE}JOLT outputs:{RESET} [{jolt_outputs}]")

    if zolt_inputs_len == 0 and jolt_inputs:
        print(f"\n  {RED}{BOLD}⚠ MISMATCH: Zolt has no inputs but Jolt has inputs!{RESET}")
        all_match = False
    if zolt_outputs_len == 0 and jolt_outputs:
        print(f"  {RED}{BOLD}⚠ MISMATCH: Zolt has no outputs but Jolt has outputs!{RESET}")
        all_match = False

    return all_match

def compare_commitments(zolt_log: str, jolt_log: str):
    """Compare commitment bytes between Zolt and Jolt"""
    print(f"\n{BOLD}{'='*70}")
    print(f"  COMMITMENT COMPARISON")
    print(f"{'='*70}{RESET}")

    # Extract Zolt GT commitments
    zolt_gt_pattern = r'\[ZOLT TRANSCRIPT\] appendGT:\s*\n.*?raw_bytes\[0\.\.16\]=\{\s*([^}]+)\}'
    zolt_gts = re.findall(r'\[ZOLT TRANSCRIPT\]   raw_bytes\[0\.\.16\]=\{\s*([^}]+)\}', zolt_log)

    # Extract Jolt commitments (from the appending)
    jolt_comm_pattern = r'\[JOLT\] Appending commitment (\d+): raw first 16 = \[([^\]]+)\]'
    jolt_comms = re.findall(jolt_comm_pattern, jolt_log)

    print(f"  Found {len(zolt_gts)} Zolt GT commitments, {len(jolt_comms)} Jolt commitments")

    for i, (zolt_gt, jolt_comm) in enumerate(zip(zolt_gts[:5], jolt_comms[:5])):
        print(f"\n  {CYAN}Commitment {i}:{RESET}")
        print(f"    {YELLOW}ZOLT first 16:{RESET} {zolt_gt}")
        print(f"    {BLUE}JOLT first 16:{RESET} {jolt_comm[1]}")

def main():
    zolt_file = sys.argv[1] if len(sys.argv) > 1 else '/tmp/zolt.log'
    jolt_file = sys.argv[2] if len(sys.argv) > 2 else '/tmp/jolt.log'

    with open(zolt_file, 'r') as f:
        zolt_log = f.read()
    with open(jolt_file, 'r') as f:
        jolt_log = f.read()

    print(f"{BOLD}{'='*70}")
    print(f"     ZOLT vs JOLT SUMCHECK COMPARISON (HEX + ENDIANNESS)")
    print(f"{'='*70}{RESET}")

    # ========== PREAMBLE ==========
    preamble_match = compare_preamble(zolt_log, jolt_log)

    # ========== COMMITMENTS ==========
    compare_commitments(zolt_log, jolt_log)

    # ========== INITIAL CLAIM ==========
    print(f"\n{BOLD}{'='*70}")
    print(f"  STAGE 1 - INITIAL CLAIM")
    print(f"{'='*70}{RESET}")

    zolt_init, _ = extract_value(zolt_log, "STAGE1_INITIAL: claim = ", True)
    _, jolt_init = extract_value(jolt_log, "STAGE1_INITIAL: claim = ", False)
    compare_value("Initial Claim", zolt_init, jolt_decimal=jolt_init)

    # ========== ROUND 0 ==========
    print(f"\n{BOLD}{'='*70}")
    print(f"  STAGE 1 - ROUND 0")
    print(f"{'='*70}{RESET}")

    # c0
    zolt_c0, _ = extract_value(zolt_log, "STAGE1_ROUND_0: c0 = ", True)
    jolt_c0, _ = extract_value(jolt_log, "STAGE1_ROUND_0: c0_bytes = ", False)
    compare_value("c0", zolt_c0, jolt_c0)

    # c2
    zolt_c2, _ = extract_value(zolt_log, "STAGE1_ROUND_0: c2 = ", True)
    jolt_c2, _ = extract_value(jolt_log, "STAGE1_ROUND_0: c2_bytes = ", False)
    compare_value("c2", zolt_c2, jolt_c2)

    # c3
    zolt_c3, _ = extract_value(zolt_log, "STAGE1_ROUND_0: c3 = ", True)
    jolt_c3, _ = extract_value(jolt_log, "STAGE1_ROUND_0: c3_bytes = ", False)
    compare_value("c3", zolt_c3, jolt_c3)

    # Challenge
    zolt_ch, _ = extract_value(zolt_log, "STAGE1_ROUND_0: challenge = ", True)
    jolt_ch, _ = extract_value(jolt_log, "STAGE1_ROUND_0: challenge_bytes = ", False)
    compare_value("Challenge", zolt_ch, jolt_ch)

    # ========== SUMMARY ==========
    print(f"\n{BOLD}{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}{RESET}")

    if not preamble_match:
        print(f"{RED}{BOLD}⚠ PREAMBLE MISMATCH - This is the ROOT CAUSE!{RESET}")
        print(f"  The Fiat-Shamir transcript is seeded with different values.")
        print(f"  This causes all challenges and claims to diverge.")
        print(f"\n  {YELLOW}FIX: Ensure Zolt uses the same inputs/outputs as Jolt.{RESET}")
        print(f"  The Jolt verifier loads I/O from /tmp/fib_io_device.bin")
        print(f"  but Zolt is proving with empty inputs/outputs.")
    else:
        # Check if initial claims match
        if zolt_init and jolt_init:
            zolt_int = int.from_bytes(bytes(zolt_init), 'little')
            if zolt_int == jolt_init:
                print(f"{GREEN}✓ Initial claims match (using LE interpretation){RESET}")
            else:
                zolt_int_be = int.from_bytes(bytes(zolt_init), 'big')
                if zolt_int_be == jolt_init:
                    print(f"{GREEN}✓ Initial claims match (using BE interpretation){RESET}")
                else:
                    print(f"{RED}✗ Initial claims DON'T match{RESET}")

    print(f"\n{DIM}Logs used: {zolt_file}, {jolt_file}{RESET}")

if __name__ == '__main__':
    main()
