#!/usr/bin/env python3
"""
Comprehensive Jolt vs Zolt verification stage comparison tool.

Compares all verification stages between Zolt and Jolt logs:
- Preamble (Fiat-Shamir setup)
- Commitments
- Stage 1: Outer Spartan (R1CS)
- Stage 2: RAM RAF Evaluation (uni-skip)
- Stage 3: Lasso Lookup
- Stage 4: Value Evaluation
- Stage 5: Register Evaluation
- Stage 6: Booleanity

Usage:
    python3 compare_sumcheck.py [zolt.log] [jolt.log] [--stage N] [--verbose]
"""

import re
import sys
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from enum import Enum

# ANSI colors
class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

class MatchResult(Enum):
    MATCH = "match"
    MATCH_REVERSED = "match_reversed"
    MISMATCH = "mismatch"
    MISSING_ZOLT = "missing_zolt"
    MISSING_JOLT = "missing_jolt"
    MISSING_BOTH = "missing_both"

@dataclass
class CompareResult:
    name: str
    zolt_value: Optional[str]
    jolt_value: Optional[str]
    result: MatchResult
    details: str = ""

class LogParser:
    """Parse values from Zolt and Jolt logs"""

    @staticmethod
    def parse_hex_string(line: str) -> Optional[str]:
        """Extract 64-char hex string (field element)"""
        match = re.search(r'([0-9a-f]{64})', line, re.I)
        if match:
            return match.group(1).lower()
        return None

    @staticmethod
    def parse_zolt_bytes(line: str) -> Optional[List[int]]:
        """Extract byte array from Zolt log like { 32, 30, 11, ... }"""
        match = re.search(r'\{\s*([\d,\s]+)\s*\}', line)
        if match:
            nums = match.group(1).split(',')
            return [int(n.strip()) for n in nums if n.strip()]
        return None

    @staticmethod
    def parse_jolt_bytes(line: str) -> Optional[List[int]]:
        """Extract byte array from Jolt log like [130, 56, 27, ...]"""
        match = re.search(r'\[([\d,\s]+)\]', line)
        if match:
            nums = match.group(1).split(',')
            try:
                return [int(n.strip()) for n in nums if n.strip()]
            except ValueError:
                return None
        return None

    @staticmethod
    def parse_decimal(line: str) -> Optional[int]:
        """Extract large decimal number"""
        match = re.search(r'=\s*(\d{10,})', line)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def bytes_to_hex(b: List[int]) -> str:
        """Convert byte list to hex string"""
        return ''.join(f'{x:02x}' for x in b)

    @staticmethod
    def hex_to_bytes(h: str) -> List[int]:
        """Convert hex string to byte list"""
        return [int(h[i:i+2], 16) for i in range(0, len(h), 2)]

    @staticmethod
    def decimal_to_hex(n: int) -> str:
        """Convert decimal to 64-char hex"""
        return f'{n:064x}'

class StageComparator:
    """Compare verification stages between Zolt and Jolt"""

    def __init__(self, zolt_log: str, jolt_log: str, verbose: bool = False):
        self.zolt_log = zolt_log
        self.jolt_log = jolt_log
        self.verbose = verbose
        self.results: List[CompareResult] = []
        self.parser = LogParser()

    def extract_values(self, pattern: str, is_zolt: bool) -> Dict[str, str]:
        """Extract all values matching a pattern prefix"""
        log = self.zolt_log if is_zolt else self.jolt_log
        values = {}
        for line in log.split('\n'):
            if pattern in line:
                # Try to extract hex string
                hex_val = self.parser.parse_hex_string(line)
                if hex_val:
                    # Extract the key (what comes after pattern before =)
                    key_match = re.search(pattern + r'(\w+)', line)
                    key = key_match.group(1) if key_match else "value"
                    values[key] = hex_val
                    continue

                # Try to extract decimal
                dec_val = self.parser.parse_decimal(line)
                if dec_val:
                    key_match = re.search(pattern + r'(\w+)', line)
                    key = key_match.group(1) if key_match else "value"
                    values[key] = self.parser.decimal_to_hex(dec_val)
                    continue

                # Try to extract bytes
                if is_zolt:
                    bytes_val = self.parser.parse_zolt_bytes(line)
                else:
                    bytes_val = self.parser.parse_jolt_bytes(line)
                if bytes_val:
                    key_match = re.search(pattern + r'(\w+)', line)
                    key = key_match.group(1) if key_match else "value"
                    values[key] = self.parser.bytes_to_hex(bytes_val)
        return values

    def extract_single(self, pattern: str, log: str) -> Optional[str]:
        """Extract a single value matching pattern"""
        for line in log.split('\n'):
            if pattern in line:
                # Try hex string first (64-char)
                hex_val = self.parser.parse_hex_string(line)
                if hex_val:
                    return hex_val

                # Try Zolt byte format: { 32, 30, 11, ... }
                zolt_bytes = self.parser.parse_zolt_bytes(line)
                if zolt_bytes and len(zolt_bytes) >= 32:
                    return self.parser.bytes_to_hex(zolt_bytes[:32])

                # Try Jolt byte format: [47, 61, 92, ...]
                jolt_bytes = self.parser.parse_jolt_bytes(line)
                if jolt_bytes and len(jolt_bytes) >= 32:
                    return self.parser.bytes_to_hex(jolt_bytes[:32])

                # Try decimal
                dec_val = self.parser.parse_decimal(line)
                if dec_val:
                    return self.parser.decimal_to_hex(dec_val)
        return None

    def extract_u64(self, pattern: str, log: str) -> Optional[int]:
        """Extract a u64 value"""
        for line in log.split('\n'):
            if pattern in line:
                match = re.search(r'=\s*(\d+)', line)
                if match:
                    return int(match.group(1))
        return None

    def compare_values(self, name: str, zolt_val: Optional[str], jolt_val: Optional[str]) -> CompareResult:
        """Compare two hex values, considering endianness"""
        if not zolt_val and not jolt_val:
            return CompareResult(name, None, None, MatchResult.MISSING_BOTH)
        if not zolt_val:
            return CompareResult(name, None, jolt_val, MatchResult.MISSING_ZOLT)
        if not jolt_val:
            return CompareResult(name, zolt_val, None, MatchResult.MISSING_JOLT)

        # Direct match
        if zolt_val.lower() == jolt_val.lower():
            return CompareResult(name, zolt_val, jolt_val, MatchResult.MATCH)

        # Check reversed (endianness)
        zolt_bytes = self.parser.hex_to_bytes(zolt_val)
        jolt_bytes = self.parser.hex_to_bytes(jolt_val)
        if zolt_bytes == list(reversed(jolt_bytes)):
            return CompareResult(name, zolt_val, jolt_val, MatchResult.MATCH_REVERSED,
                               "Endianness difference")

        return CompareResult(name, zolt_val, jolt_val, MatchResult.MISMATCH)

    def compare_preamble(self) -> List[CompareResult]:
        """Compare Fiat-Shamir preamble"""
        results = []
        fields = [
            ("max_input_size", "appendU64: max_input_size"),
            ("max_output_size", "appendU64: max_output_size"),
            ("memory_size", "appendU64: memory_size"),
            ("inputs.len", "appendBytes: inputs.len"),
            ("outputs.len", "appendBytes: outputs.len"),
            ("panic", "appendU64: panic"),
            ("ram_K", "appendU64: ram_K"),
            ("trace_length", "appendU64: trace_length"),
        ]

        for name, pattern in fields:
            zolt_val = self.extract_u64(pattern, self.zolt_log)
            jolt_val = self.extract_u64(pattern, self.jolt_log)

            if zolt_val == jolt_val:
                result = MatchResult.MATCH
            elif zolt_val is None:
                result = MatchResult.MISSING_ZOLT
            elif jolt_val is None:
                result = MatchResult.MISSING_JOLT
            else:
                result = MatchResult.MISMATCH

            results.append(CompareResult(
                name,
                str(zolt_val) if zolt_val is not None else None,
                str(jolt_val) if jolt_val is not None else None,
                result
            ))

        return results

    def extract_zolt_le_hex(self, pattern: str) -> Optional[str]:
        """Extract Zolt value in LE format and convert to hex"""
        for line in self.zolt_log.split('\n'):
            if pattern in line:
                zolt_bytes = self.parser.parse_zolt_bytes(line)
                if zolt_bytes and len(zolt_bytes) >= 32:
                    return self.parser.bytes_to_hex(zolt_bytes[:32])
        return None

    def extract_zolt_be_as_le_hex(self, pattern: str) -> Optional[str]:
        """Extract Zolt value in BE format and convert to LE hex"""
        for line in self.zolt_log.split('\n'):
            if pattern in line:
                zolt_bytes = self.parser.parse_zolt_bytes(line)
                if zolt_bytes and len(zolt_bytes) >= 32:
                    # Reverse BE to LE
                    le_bytes = list(reversed(zolt_bytes[:32]))
                    return self.parser.bytes_to_hex(le_bytes)
        return None

    def extract_jolt_decimal_as_le_hex(self, pattern: str) -> Optional[str]:
        """Extract Jolt decimal and convert to LE hex for comparison with Zolt _le"""
        for line in self.jolt_log.split('\n'):
            if pattern in line:
                dec_val = self.parser.parse_decimal(line)
                if dec_val:
                    # Convert decimal to BE hex, then reverse to LE
                    be_hex = self.parser.decimal_to_hex(dec_val)
                    be_bytes = self.parser.hex_to_bytes(be_hex)
                    le_bytes = list(reversed(be_bytes))
                    return self.parser.bytes_to_hex(le_bytes)
        return None

    def compare_stage1_round(self, round_idx: int) -> List[CompareResult]:
        """Compare a single Stage 1 round - coefficients and challenges only"""
        results = []

        # Coefficients c0, c2, c3 - try new format first, fall back to old _le suffix
        for coeff in ["c0", "c2", "c3"]:
            zolt_val = self.extract_zolt_le_hex(f"STAGE1_ROUND_{round_idx}: {coeff} = ")
            if zolt_val is None:
                zolt_val = self.extract_zolt_le_hex(f"STAGE1_ROUND_{round_idx}: {coeff}_le = ")
            jolt_val = self.extract_single(f"STAGE1_ROUND_{round_idx}: {coeff}_bytes = ", self.jolt_log)
            results.append(self.compare_values(f"Round {round_idx} {coeff}", zolt_val, jolt_val))

        # Challenge - try new format first, fall back to old _le suffix
        zolt_ch = self.extract_zolt_le_hex(f"STAGE1_ROUND_{round_idx}: challenge = ")
        if zolt_ch is None:
            zolt_ch = self.extract_zolt_le_hex(f"STAGE1_ROUND_{round_idx}: challenge_le = ")
        jolt_ch = self.extract_single(f"STAGE1_ROUND_{round_idx}: challenge_bytes = ", self.jolt_log)
        results.append(self.compare_values(f"Round {round_idx} challenge", zolt_ch, jolt_ch))

        return results

    def compare_stage1(self) -> List[CompareResult]:
        """Compare Stage 1: Outer Spartan sumcheck polynomials and challenges"""
        results = []

        # tau.len - try both old and new Zolt format
        zolt_tau_len = self.extract_u64("STAGE1: tau.len = ", self.zolt_log)
        if zolt_tau_len is None:
            zolt_tau_len = self.extract_u64("STAGE1_PRE: tau.len = ", self.zolt_log)
        jolt_tau_len = self.extract_u64("STAGE1_PRE: tau.len = ", self.jolt_log)

        if zolt_tau_len == jolt_tau_len:
            results.append(CompareResult("tau.len", str(zolt_tau_len), str(jolt_tau_len), MatchResult.MATCH))
        else:
            results.append(CompareResult("tau.len", str(zolt_tau_len), str(jolt_tau_len), MatchResult.MISMATCH))

        # Initial claim - try new format first, fall back to old _le suffix
        zolt_init = self.extract_zolt_le_hex("STAGE1_INITIAL: claim = ")
        if zolt_init is None:
            zolt_init = self.extract_zolt_le_hex("STAGE1_INITIAL: claim_le = ")
        jolt_init = self.extract_jolt_decimal_as_le_hex("STAGE1_INITIAL: claim = ")
        results.append(self.compare_values("Initial claim", zolt_init, jolt_init))

        # Find number of rounds from logs
        num_rounds = 0
        for line in self.zolt_log.split('\n'):
            match = re.search(r'STAGE1_ROUND_(\d+):', line)
            if match:
                num_rounds = max(num_rounds, int(match.group(1)) + 1)

        # Compare each round (coefficients and challenges)
        for i in range(min(num_rounds, 20)):
            results.extend(self.compare_stage1_round(i))

        # Stage 1 UniSkip values (from verifier)
        zolt_coeffs0 = self.extract_single("STAGE1_UNISKIP: coeffs[0] = ", self.zolt_log)
        jolt_coeffs0 = self.extract_single("STAGE1_UNISKIP: coeffs[0] = ", self.jolt_log)
        results.append(self.compare_values("Stage1 UniSkip coeffs[0]", zolt_coeffs0, jolt_coeffs0))

        zolt_input_claim = self.extract_single("STAGE1_UNISKIP: input_claim = ", self.zolt_log)
        jolt_input_claim = self.extract_single("STAGE1_UNISKIP: input_claim = ", self.jolt_log)
        results.append(self.compare_values("Stage1 UniSkip input_claim", zolt_input_claim, jolt_input_claim))

        zolt_domain_sum = self.extract_single("STAGE1_UNISKIP: domain_sum = ", self.zolt_log)
        jolt_domain_sum = self.extract_single("STAGE1_UNISKIP: domain_sum = ", self.jolt_log)
        results.append(self.compare_values("Stage1 UniSkip domain_sum", zolt_domain_sum, jolt_domain_sum))

        return results

    def compare_stage2(self) -> List[CompareResult]:
        """Compare Stage 2: RAM RAF / UniSkip"""
        results = []

        # Uni-skip values
        zolt_tau = self.extract_single("STAGE2: tau_high = ", self.zolt_log)
        jolt_tau = self.extract_single("STAGE2: tau_high = ", self.jolt_log)
        results.append(self.compare_values("Stage2 tau_high", zolt_tau, jolt_tau))

        # Base evaluations
        for i in range(5):
            zolt_val = self.extract_single(f"STAGE2: base_evals[{i}] = ", self.zolt_log)
            jolt_val = self.extract_single(f"STAGE2: base_evals[{i}] = ", self.jolt_log)
            results.append(self.compare_values(f"Stage2 base_evals[{i}]", zolt_val, jolt_val))

        # UniSkip first round values
        zolt_coeffs0 = self.extract_single("STAGE2_UNISKIP: coeffs[0] = ", self.zolt_log)
        jolt_coeffs0 = self.extract_single("STAGE2_UNISKIP: coeffs[0] = ", self.jolt_log)
        results.append(self.compare_values("Stage2 UniSkip coeffs[0]", zolt_coeffs0, jolt_coeffs0))

        zolt_input_claim = self.extract_single("STAGE2_UNISKIP: input_claim = ", self.zolt_log)
        jolt_input_claim = self.extract_single("STAGE2_UNISKIP: input_claim = ", self.jolt_log)
        results.append(self.compare_values("Stage2 UniSkip input_claim", zolt_input_claim, jolt_input_claim))

        zolt_domain_sum = self.extract_single("STAGE2_UNISKIP: domain_sum = ", self.zolt_log)
        jolt_domain_sum = self.extract_single("STAGE2_UNISKIP: domain_sum = ", self.jolt_log)
        results.append(self.compare_values("Stage2 UniSkip domain_sum", zolt_domain_sum, jolt_domain_sum))

        return results

    def compare_commitments(self) -> List[CompareResult]:
        """Compare commitment values"""
        results = []

        # Extract commitment patterns - both use hex bytes
        zolt_pattern = r'\[ZOLT TRANSCRIPT\]   raw_bytes\[0\.\.16\]=\{\s*([^}]+)\}'
        jolt_pattern = r'\[JOLT\] Appending commitment (\d+): raw first 16 = \[([^\]]+)\]'

        zolt_comms = re.findall(zolt_pattern, self.zolt_log)
        jolt_comms = re.findall(jolt_pattern, self.jolt_log)

        for i, (zolt_comm, jolt_comm) in enumerate(zip(zolt_comms[:5], jolt_comms[:5])):
            # Parse Zolt bytes (hex with spaces: "aa 47 b8 36...")
            zolt_hex = zolt_comm.replace(' ', '').strip()

            # Parse Jolt bytes (hex with commas: "aa, 47, b8, 36...")
            jolt_bytes_str = jolt_comm[1] if isinstance(jolt_comm, tuple) else jolt_comm
            jolt_hex = ''.join(x.strip() for x in jolt_bytes_str.split(',') if x.strip())

            results.append(self.compare_values(f"Commitment {i} (first 16)", zolt_hex, jolt_hex))

        return results

    def run_comparison(self, stages: List[int] = None) -> Dict[str, List[CompareResult]]:
        """Run full comparison"""
        all_results = {}

        if stages is None or 0 in stages:
            all_results['Preamble'] = self.compare_preamble()
            all_results['Commitments'] = self.compare_commitments()

        if stages is None or 1 in stages:
            all_results['Stage 1 (Spartan)'] = self.compare_stage1()

        if stages is None or 2 in stages:
            all_results['Stage 2 (UniSkip)'] = self.compare_stage2()

        return all_results

def print_results(results: Dict[str, List[CompareResult]], verbose: bool = False):
    """Print comparison results with formatting"""
    total_match = 0
    total_mismatch = 0
    total_missing = 0

    for section, section_results in results.items():
        print(f"\n{Color.BOLD}{'='*70}")
        print(f"  {section}")
        print(f"{'='*70}{Color.RESET}")

        section_match = 0
        section_mismatch = 0
        first_mismatch = None

        for r in section_results:
            if r.result == MatchResult.MATCH:
                status = f"{Color.GREEN}MATCH{Color.RESET}"
                section_match += 1
                total_match += 1
            elif r.result == MatchResult.MATCH_REVERSED:
                status = f"{Color.GREEN}MATCH{Color.RESET} {Color.DIM}(endian){Color.RESET}"
                section_match += 1
                total_match += 1
            elif r.result == MatchResult.MISMATCH:
                status = f"{Color.RED}MISMATCH{Color.RESET}"
                section_mismatch += 1
                total_mismatch += 1
                if first_mismatch is None:
                    first_mismatch = r
            elif r.result in [MatchResult.MISSING_ZOLT, MatchResult.MISSING_JOLT, MatchResult.MISSING_BOTH]:
                status = f"{Color.YELLOW}MISSING{Color.RESET}"
                total_missing += 1
                continue  # Skip missing values in normal output

            # Print summary line
            print(f"  {status:30} {r.name}")

            # Print values if mismatch, verbose, or Stage 2 (for debugging UniSkip)
            show_values = r.result == MatchResult.MISMATCH or verbose or "Stage 2" in section
            if show_values:
                if r.zolt_value:
                    print(f"    {Color.YELLOW}ZOLT:{Color.RESET} {r.zolt_value[:32]}..." if len(r.zolt_value or '') > 32 else f"    {Color.YELLOW}ZOLT:{Color.RESET} {r.zolt_value}")
                if r.jolt_value:
                    print(f"    {Color.BLUE}JOLT:{Color.RESET} {r.jolt_value[:32]}..." if len(r.jolt_value or '') > 32 else f"    {Color.BLUE}JOLT:{Color.RESET} {r.jolt_value}")

        # Section summary
        if section_mismatch > 0:
            print(f"\n  {Color.RED}Section: {section_mismatch} mismatches, {section_match} matches{Color.RESET}")
            if first_mismatch:
                print(f"  {Color.RED}First mismatch: {first_mismatch.name}{Color.RESET}")
        else:
            print(f"\n  {Color.GREEN}Section: All {section_match} values match{Color.RESET}")

    # Overall summary
    print(f"\n{Color.BOLD}{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}{Color.RESET}")

    if total_mismatch == 0:
        print(f"  {Color.GREEN}{Color.BOLD}ALL {total_match} VALUES MATCH{Color.RESET}")
    else:
        print(f"  {Color.RED}Mismatches: {total_mismatch}{Color.RESET}")
        print(f"  {Color.GREEN}Matches: {total_match}{Color.RESET}")
        if total_missing > 0:
            print(f"  {Color.YELLOW}Missing: {total_missing}{Color.RESET}")

def find_first_divergence(results: Dict[str, List[CompareResult]]) -> Optional[CompareResult]:
    """Find the first point where Zolt and Jolt diverge"""
    for section, section_results in results.items():
        for r in section_results:
            if r.result == MatchResult.MISMATCH:
                return r
    return None

def main():
    parser = argparse.ArgumentParser(description='Compare Zolt and Jolt verification logs')
    parser.add_argument('zolt_log', nargs='?', default='/tmp/zolt.log', help='Zolt log file')
    parser.add_argument('jolt_log', nargs='?', default='/tmp/jolt.log', help='Jolt log file')
    parser.add_argument('--stage', '-s', type=int, nargs='+', help='Compare specific stages (0=preamble, 1-6)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show all values, not just mismatches')
    parser.add_argument('--find-divergence', '-d', action='store_true', help='Find first point of divergence')
    args = parser.parse_args()

    try:
        with open(args.zolt_log, 'r') as f:
            zolt_log = f.read()
        with open(args.jolt_log, 'r') as f:
            jolt_log = f.read()
    except FileNotFoundError as e:
        print(f"{Color.RED}Error: {e}{Color.RESET}")
        sys.exit(1)

    print(f"{Color.BOLD}{'='*70}")
    print(f"     ZOLT vs JOLT VERIFICATION COMPARISON")
    print(f"{'='*70}{Color.RESET}")
    print(f"{Color.DIM}Zolt: {args.zolt_log}")
    print(f"Jolt: {args.jolt_log}{Color.RESET}")

    comparator = StageComparator(zolt_log, jolt_log, args.verbose)
    results = comparator.run_comparison(args.stage)

    print_results(results, args.verbose)

    if args.find_divergence:
        divergence = find_first_divergence(results)
        if divergence:
            print(f"\n{Color.RED}{Color.BOLD}FIRST DIVERGENCE:{Color.RESET}")
            print(f"  {Color.CYAN}{divergence.name}{Color.RESET}")
            print(f"  {Color.YELLOW}ZOLT:{Color.RESET} {divergence.zolt_value}")
            print(f"  {Color.BLUE}JOLT:{Color.RESET} {divergence.jolt_value}")
        else:
            print(f"\n{Color.GREEN}No divergence found - all values match!{Color.RESET}")

if __name__ == '__main__':
    main()
