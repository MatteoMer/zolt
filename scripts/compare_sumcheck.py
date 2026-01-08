#!/usr/bin/env python3
"""
Comprehensive Jolt vs Zolt verification stage comparison tool.

Compares all verification stages between Zolt and Jolt logs:
- Preamble (Fiat-Shamir setup)
- Commitments
- Stage 1: Outer Spartan (R1CS)
- Stage 2: RAM RAF Evaluation (uni-skip)
- Stage 3: Lasso Lookup (Shift, InstructionInput, RegistersClaimReduction)
- Stage 4: Value Evaluation
- Stage 5: Register Evaluation
- Stage 6: Booleanity

NEW: Transcript state tracking to catch state-dependent divergence.

Usage:
    python3 compare_sumcheck.py [zolt.log] [jolt.log] [--stage N] [--verbose]
    python3 compare_sumcheck.py --transcript  # Compare transcript state evolution

Stage 3 Debug Output Format:
    Zolt: [ZOLT] STAGE3_PRE: ... = { bytes }
    Jolt: [JOLT] STAGE3_PRE: ... = Field(0x...) or [n1, n2, ...]
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

@dataclass
class TranscriptOp:
    """A single transcript operation"""
    op_type: str  # "append_u64", "append_bytes", "append_message", "challenge"
    label: str
    value: str  # hex or numeric value
    state_before: str  # transcript state before op
    state_after: str  # transcript state after op
    round_num: int
    line_num: int

class TranscriptAnalyzer:
    """Analyze transcript state evolution to find divergence"""

    def __init__(self, zolt_log: str, jolt_log: str):
        self.zolt_log = zolt_log
        self.jolt_log = jolt_log
        self.zolt_ops: List[TranscriptOp] = []
        self.jolt_ops: List[TranscriptOp] = []

    def parse_zolt_transcript(self) -> List[TranscriptOp]:
        """Parse Zolt transcript operations"""
        ops = []
        lines = self.zolt_log.split('\n')

        # Patterns for Zolt transcript logs
        # [ZOLT TRANSCRIPT] appendU64: max_input_size = 1024, round=0, state_before={ aa bb cc ... }
        append_pattern = re.compile(
            r'\[ZOLT TRANSCRIPT\]\s+(\w+):\s+(\S+)\s*=\s*([^,]+),\s*round=(\d+),\s*state_before=\{([^}]*)\}'
        )
        state_after_pattern = re.compile(r'\[ZOLT TRANSCRIPT\]\s+state_after=\{([^}]*)\}')
        challenge_pattern = re.compile(
            r'\[ZOLT TRANSCRIPT\]\s+challenge:\s*(\S+)\s*=\s*\{([^}]*)\},\s*round=(\d+)'
        )

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for append operation
            match = append_pattern.search(line)
            if match:
                op_type = match.group(1)
                label = match.group(2)
                value = match.group(3).strip()
                round_num = int(match.group(4))
                state_before = match.group(5).replace(' ', '')

                # Look for state_after on next line
                state_after = ""
                if i + 1 < len(lines):
                    after_match = state_after_pattern.search(lines[i + 1])
                    if after_match:
                        state_after = after_match.group(1).replace(' ', '')
                        i += 1

                ops.append(TranscriptOp(
                    op_type=op_type,
                    label=label,
                    value=value,
                    state_before=state_before,
                    state_after=state_after,
                    round_num=round_num,
                    line_num=i
                ))

            # Check for challenge
            ch_match = challenge_pattern.search(line)
            if ch_match:
                label = ch_match.group(1)
                value = ch_match.group(2).replace(' ', '')
                round_num = int(ch_match.group(3))
                ops.append(TranscriptOp(
                    op_type="challenge",
                    label=label,
                    value=value,
                    state_before="",
                    state_after="",
                    round_num=round_num,
                    line_num=i
                ))

            i += 1

        return ops

    def parse_jolt_transcript(self) -> List[TranscriptOp]:
        """Parse Jolt transcript operations"""
        ops = []
        lines = self.jolt_log.split('\n')

        # Patterns for Jolt transcript logs
        # [JOLT TRANSCRIPT] append_u64: value=1024, round=0, state_before=[aa, bb, cc, ...]
        append_u64_pattern = re.compile(
            r'\[JOLT TRANSCRIPT\]\s+append_u64:\s+value=(\d+),\s*round=(\d+),\s*state_before=\[([^\]]*)\]'
        )
        append_bytes_pattern = re.compile(
            r'\[JOLT TRANSCRIPT\]\s+append_bytes:\s+len=(\d+),\s*round=(\d+),\s*state_before=\[([^\]]*)\]'
        )
        append_msg_pattern = re.compile(
            r'\[JOLT TRANSCRIPT\]\s+append_message:\s+(\S+),\s*round=(\d+),\s*state_before=\[([^\]]*)\]'
        )
        state_after_pattern = re.compile(r'\[JOLT TRANSCRIPT\]\s+state_after=\[([^\]]*)\]')
        challenge_pattern = re.compile(
            r'\[JOLT TRANSCRIPT\]\s+challenge:\s*(\S+)\s*=\s*\[([^\]]*)\],\s*round=(\d+)'
        )

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check append_u64
            match = append_u64_pattern.search(line)
            if match:
                value = match.group(1)
                round_num = int(match.group(2))
                state_before = self._normalize_jolt_state(match.group(3))

                state_after = ""
                if i + 1 < len(lines):
                    after_match = state_after_pattern.search(lines[i + 1])
                    if after_match:
                        state_after = self._normalize_jolt_state(after_match.group(1))
                        i += 1

                ops.append(TranscriptOp(
                    op_type="append_u64",
                    label="u64",
                    value=value,
                    state_before=state_before,
                    state_after=state_after,
                    round_num=round_num,
                    line_num=i
                ))

            # Check append_bytes
            match = append_bytes_pattern.search(line)
            if match:
                value = f"len={match.group(1)}"
                round_num = int(match.group(2))
                state_before = self._normalize_jolt_state(match.group(3))

                state_after = ""
                if i + 1 < len(lines):
                    after_match = state_after_pattern.search(lines[i + 1])
                    if after_match:
                        state_after = self._normalize_jolt_state(after_match.group(1))
                        i += 1

                ops.append(TranscriptOp(
                    op_type="append_bytes",
                    label="bytes",
                    value=value,
                    state_before=state_before,
                    state_after=state_after,
                    round_num=round_num,
                    line_num=i
                ))

            # Check append_message
            match = append_msg_pattern.search(line)
            if match:
                label = match.group(1)
                round_num = int(match.group(2))
                state_before = self._normalize_jolt_state(match.group(3))

                state_after = ""
                if i + 1 < len(lines):
                    after_match = state_after_pattern.search(lines[i + 1])
                    if after_match:
                        state_after = self._normalize_jolt_state(after_match.group(1))
                        i += 1

                ops.append(TranscriptOp(
                    op_type="append_message",
                    label=label,
                    value="",
                    state_before=state_before,
                    state_after=state_after,
                    round_num=round_num,
                    line_num=i
                ))

            # Check challenge
            ch_match = challenge_pattern.search(line)
            if ch_match:
                label = ch_match.group(1)
                value = self._normalize_jolt_state(ch_match.group(2))
                round_num = int(ch_match.group(3))
                ops.append(TranscriptOp(
                    op_type="challenge",
                    label=label,
                    value=value,
                    state_before="",
                    state_after="",
                    round_num=round_num,
                    line_num=i
                ))

            i += 1

        return ops

    def _normalize_jolt_state(self, state: str) -> str:
        """Convert Jolt state format [130, 56, 27] to hex"""
        try:
            nums = [int(x.strip()) for x in state.split(',') if x.strip()]
            return ''.join(f'{n:02x}' for n in nums)
        except:
            return state

    def find_state_divergence(self) -> Optional[Tuple[int, TranscriptOp, TranscriptOp, str]]:
        """Find the first point where transcript states diverge.
        Returns (index, zolt_op, jolt_op, reason) or None if no divergence."""

        self.zolt_ops = self.parse_zolt_transcript()
        self.jolt_ops = self.parse_jolt_transcript()

        # Compare state_after values in sequence
        min_len = min(len(self.zolt_ops), len(self.jolt_ops))

        for i in range(min_len):
            zolt_op = self.zolt_ops[i]
            jolt_op = self.jolt_ops[i]

            # Check if state_after matches (the cumulative state)
            if zolt_op.state_after and jolt_op.state_after:
                if zolt_op.state_after != jolt_op.state_after:
                    return (i, zolt_op, jolt_op, "state_after mismatch")

            # Check if operation types match
            if zolt_op.op_type != jolt_op.op_type:
                return (i, zolt_op, jolt_op, f"op_type mismatch: {zolt_op.op_type} vs {jolt_op.op_type}")

            # Check round numbers
            if zolt_op.round_num != jolt_op.round_num:
                return (i, zolt_op, jolt_op, f"round mismatch: {zolt_op.round_num} vs {jolt_op.round_num}")

        # Check if one log has more ops than the other
        if len(self.zolt_ops) != len(self.jolt_ops):
            return (min_len, None, None,
                    f"operation count mismatch: Zolt has {len(self.zolt_ops)}, Jolt has {len(self.jolt_ops)}")

        return None

    def print_ops_around(self, index: int, context: int = 3):
        """Print operations around a given index for debugging"""
        start = max(0, index - context)
        end = min(len(self.zolt_ops), len(self.jolt_ops), index + context + 1)

        print(f"\n{Color.BOLD}Transcript operations around divergence (index {index}):{Color.RESET}")
        print(f"\n{Color.YELLOW}ZOLT operations:{Color.RESET}")
        for i in range(start, min(end, len(self.zolt_ops))):
            op = self.zolt_ops[i]
            marker = " >>> " if i == index else "     "
            print(f"{marker}[{i}] {op.op_type}: {op.label}={op.value[:20] if op.value else ''}, round={op.round_num}")
            if op.state_after:
                print(f"          state_after: {op.state_after[:32]}...")

        print(f"\n{Color.BLUE}JOLT operations:{Color.RESET}")
        for i in range(start, min(end, len(self.jolt_ops))):
            op = self.jolt_ops[i]
            marker = " >>> " if i == index else "     "
            print(f"{marker}[{i}] {op.op_type}: {op.label}={op.value[:20] if op.value else ''}, round={op.round_num}")
            if op.state_after:
                print(f"          state_after: {op.state_after[:32]}...")

    def compare_stage_boundaries(self) -> List[Tuple[str, str, str, bool]]:
        """Compare transcript state at stage boundaries.
        Returns list of (stage_name, zolt_state, jolt_state, matches)"""
        results = []

        # Look for stage markers in both logs
        stage_markers = [
            ("STAGE1_PRE", "Stage 1 start"),
            ("STAGE1_FINAL", "Stage 1 end"),
            ("STAGE2_PRE", "Stage 2 start"),
            ("STAGE2_FINAL", "Stage 2 end"),
        ]

        for marker, name in stage_markers:
            zolt_state = self._extract_state_at_marker(self.zolt_log, marker, is_zolt=True)
            jolt_state = self._extract_state_at_marker(self.jolt_log, marker, is_zolt=False)
            matches = zolt_state == jolt_state if zolt_state and jolt_state else False
            results.append((name, zolt_state or "N/A", jolt_state or "N/A", matches))

        return results

    def _extract_state_at_marker(self, log: str, marker: str, is_zolt: bool) -> Optional[str]:
        """Extract transcript state at a given marker"""
        for line in log.split('\n'):
            if marker in line and "transcript_state" in line:
                if is_zolt:
                    match = re.search(r'transcript_state=\{([^}]*)\}', line)
                    if match:
                        return match.group(1).replace(' ', '')
                else:
                    match = re.search(r'transcript_state=\[([^\]]*)\]', line)
                    if match:
                        return self._normalize_jolt_state(match.group(1))
        return None

    def analyze_stage2_tau_high(self) -> Dict[str, any]:
        """Specifically analyze Stage 2 tau_high sampling point - common failure point"""
        result = {
            'zolt_pre_state': None,
            'jolt_pre_state': None,
            'zolt_tau_high': None,
            'jolt_tau_high': None,
            'zolt_r_cycle_len': None,
            'jolt_r_cycle_len': None,
            'states_match': False,
            'tau_high_match': False,
        }

        # Extract STAGE2_PRE transcript state (state before tau_high sampling)
        for line in self.zolt_log.split('\n'):
            if 'STAGE2_PRE' in line and 'transcript_state' in line:
                match = re.search(r'transcript_state=\{([^}]*)\}', line)
                if match:
                    result['zolt_pre_state'] = match.group(1).replace(' ', '')
            if 'STAGE2:' in line and 'r_cycle.len' in line:
                match = re.search(r'r_cycle\.len\s*=\s*(\d+)', line)
                if match:
                    result['zolt_r_cycle_len'] = int(match.group(1))
            if 'STAGE2:' in line and 'tau_high' in line:
                match = re.search(r'tau_high\s*=\s*\{([^}]*)\}', line)
                if match:
                    result['zolt_tau_high'] = match.group(1).replace(' ', '')

        for line in self.jolt_log.split('\n'):
            if 'STAGE2_PRE' in line and 'transcript_state' in line:
                match = re.search(r'transcript_state=\[([^\]]*)\]', line)
                if match:
                    result['jolt_pre_state'] = self._normalize_jolt_state(match.group(1))
            if 'STAGE2:' in line and 'r_cycle.len' in line:
                match = re.search(r'r_cycle\.len\s*=\s*(\d+)', line)
                if match:
                    result['jolt_r_cycle_len'] = int(match.group(1))
            if 'STAGE2:' in line and 'tau_high' in line:
                # Jolt might output as bytes or hex
                match = re.search(r'tau_high\s*=\s*\[([^\]]*)\]', line)
                if match:
                    result['jolt_tau_high'] = self._normalize_jolt_state(match.group(1))
                else:
                    match = re.search(r'tau_high\s*=\s*([0-9a-f]{64})', line, re.I)
                    if match:
                        result['jolt_tau_high'] = match.group(1).lower()

        # Check matches
        if result['zolt_pre_state'] and result['jolt_pre_state']:
            result['states_match'] = result['zolt_pre_state'] == result['jolt_pre_state']

        if result['zolt_tau_high'] and result['jolt_tau_high']:
            result['tau_high_match'] = result['zolt_tau_high'] == result['jolt_tau_high']

        return result

    def analyze_opening_claims(self) -> List[Tuple[str, str, str, bool]]:
        """Compare opening claims added during verification.
        Returns list of (claim_id, zolt_value, jolt_value, matches)"""
        results = []

        # Pattern for opening claims
        # [JOLT] Opening claim: SumcheckId::Outer => point=[...], eval=...
        zolt_claims = {}
        jolt_claims = {}

        for line in self.zolt_log.split('\n'):
            match = re.search(r'Opening claim:\s*(\S+)\s*=>\s*.*eval\s*=\s*\{([^}]*)\}', line)
            if match:
                claim_id = match.group(1)
                eval_val = match.group(2).replace(' ', '')
                zolt_claims[claim_id] = eval_val

        for line in self.jolt_log.split('\n'):
            match = re.search(r'Opening claim:\s*(\S+)\s*=>\s*.*eval\s*=\s*\[([^\]]*)\]', line)
            if match:
                claim_id = match.group(1)
                eval_val = self._normalize_jolt_state(match.group(2))
                jolt_claims[claim_id] = eval_val

        # Compare claims
        all_ids = set(zolt_claims.keys()) | set(jolt_claims.keys())
        for claim_id in sorted(all_ids):
            zolt_val = zolt_claims.get(claim_id, "N/A")
            jolt_val = jolt_claims.get(claim_id, "N/A")
            matches = zolt_val == jolt_val if zolt_val != "N/A" and jolt_val != "N/A" else False
            results.append((claim_id, zolt_val, jolt_val, matches))

        return results


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

    def compare_stage2_batched(self) -> List[CompareResult]:
        """Compare Stage 2 batched sumcheck - all 5 proofs"""
        results = []

        # Compare input claims for each of the 5 instances
        for i in range(5):
            zolt_val = self.extract_zolt_le_hex(f"STAGE2_PRE: input_claim[{i}] = ")
            jolt_val = self.extract_single(f"STAGE2_PRE: input_claim[{i}]_bytes = ", self.jolt_log)
            results.append(self.compare_values(f"Stage2 input_claim[{i}]", zolt_val, jolt_val))

            # Also compare num_rounds and degree
            zolt_rounds = self.extract_u64(f"STAGE2_PRE: num_rounds[{i}] = ", self.zolt_log)
            jolt_rounds = self.extract_u64(f"STAGE2_PRE: num_rounds[{i}] = ", self.jolt_log)
            if zolt_rounds == jolt_rounds:
                results.append(CompareResult(f"Stage2 num_rounds[{i}]", str(zolt_rounds), str(jolt_rounds), MatchResult.MATCH))
            else:
                results.append(CompareResult(f"Stage2 num_rounds[{i}]", str(zolt_rounds), str(jolt_rounds), MatchResult.MISMATCH))

        # Compare batching coefficients
        for i in range(5):
            zolt_val = self.extract_zolt_le_hex(f"STAGE2_PRE: batching_coeff[{i}] = ")
            jolt_val = self.extract_single(f"STAGE2_PRE: batching_coeff[{i}]_bytes = ", self.jolt_log)
            results.append(self.compare_values(f"Stage2 batching_coeff[{i}]", zolt_val, jolt_val))

        # Compare initial batched claim
        zolt_claim = self.extract_zolt_le_hex("STAGE2_INITIAL: batched_claim = ")
        jolt_claim = self.extract_single("STAGE2_INITIAL: batched_claim_bytes = ", self.jolt_log)
        results.append(self.compare_values("Stage2 initial claim", zolt_claim, jolt_claim))

        # Find number of rounds from logs
        num_rounds = self.find_max_round("STAGE2_ROUND_")

        # Compare rounds (limit to first 30 for sanity)
        for i in range(min(num_rounds, 30)):
            results.extend(self.compare_stage2_round(i))

        # Compare final output claim
        zolt_final = self.extract_zolt_le_hex("STAGE2_FINAL: output_claim = ")
        jolt_final = self.extract_single("STAGE2_FINAL: output_claim_bytes = ", self.jolt_log)
        results.append(self.compare_values("Stage2 final claim", zolt_final, jolt_final))

        return results

    def compare_stage2_round(self, round_idx: int) -> List[CompareResult]:
        """Compare Stage 2 round values"""
        results = []

        # Current claim
        zolt_claim = self.extract_zolt_le_hex(f"STAGE2_ROUND_{round_idx}: current_claim = ")
        jolt_claim = self.extract_single(f"STAGE2_ROUND_{round_idx}: current_claim_bytes = ", self.jolt_log)
        results.append(self.compare_values(f"Stage2 round {round_idx} claim", zolt_claim, jolt_claim))

        # Coefficients c0, c2, c3
        for coeff in ["c0", "c2", "c3"]:
            zolt_val = self.extract_zolt_le_hex(f"STAGE2_ROUND_{round_idx}: {coeff} = ")
            jolt_val = self.extract_single(f"STAGE2_ROUND_{round_idx}: {coeff}_bytes = ", self.jolt_log)
            results.append(self.compare_values(f"Stage2 round {round_idx} {coeff}", zolt_val, jolt_val))

        # Challenge
        zolt_ch = self.extract_zolt_le_hex(f"STAGE2_ROUND_{round_idx}: challenge = ")
        jolt_ch = self.extract_single(f"STAGE2_ROUND_{round_idx}: challenge_bytes = ", self.jolt_log)
        results.append(self.compare_values(f"Stage2 round {round_idx} challenge", zolt_ch, jolt_ch))

        # Next claim
        zolt_next = self.extract_zolt_le_hex(f"STAGE2_ROUND_{round_idx}: next_claim = ")
        jolt_next = self.extract_single(f"STAGE2_ROUND_{round_idx}: next_claim_bytes = ", self.jolt_log)
        results.append(self.compare_values(f"Stage2 round {round_idx} next_claim", zolt_next, jolt_next))

        return results

    def find_max_round(self, prefix: str) -> int:
        """Find the maximum round number from logs"""
        max_round = 0
        for line in self.zolt_log.split('\n') + self.jolt_log.split('\n'):
            match = re.search(prefix + r'(\d+):', line)
            if match:
                max_round = max(max_round, int(match.group(1)) + 1)
        return max_round

    def compare_stage3(self) -> List[CompareResult]:
        """Compare Stage 3: Batched Sumcheck (Shift, InstructionInput, Registers)"""
        results = []

        # Compare r_outer and r_product inputs
        for name in ["r_outer", "r_product"]:
            zolt_len = self.extract_u64(f"STAGE3_PRE: {name}.len = ", self.zolt_log)
            jolt_len = self.extract_u64(f"STAGE3_PRE: {name}.len = ", self.jolt_log)
            if zolt_len is not None or jolt_len is not None:
                match = MatchResult.MATCH if zolt_len == jolt_len else MatchResult.MISMATCH
                results.append(CompareResult(f"Stage3 {name}.len", str(zolt_len), str(jolt_len), match))

            zolt_val = self.extract_zolt_le_hex(f"STAGE3_PRE: {name}[0] = ")
            jolt_val = self.extract_jolt_debug_bytes(f"[JOLT] STAGE3_PRE: {name}[0] = ")
            if zolt_val or jolt_val:
                results.append(self.compare_values(f"Stage3 {name}[0]", zolt_val, jolt_val))

            zolt_val = self.extract_zolt_le_hex(f"STAGE3_PRE: {name}[last] = ")
            jolt_val = self.extract_jolt_debug_bytes(f"[JOLT] STAGE3_PRE: {name}[last] = ")
            if zolt_val or jolt_val:
                results.append(self.compare_values(f"Stage3 {name}[last]", zolt_val, jolt_val))

        # Compare gamma powers for ShiftSumcheck (5 values)
        for i in range(5):
            zolt_val = self.extract_zolt_le_hex(f"STAGE3_SHIFT: gamma_powers[{i}] = ")
            jolt_val = self.extract_single(f"STAGE3_SHIFT: gamma_powers[{i}]_bytes = ", self.jolt_log)
            # Try alternative Jolt format (Debug output)
            if not jolt_val:
                jolt_val = self.extract_jolt_debug_bytes(f"[JOLT_SHIFT] gamma_powers[{i}] = ")
            results.append(self.compare_values(f"Shift gamma[{i}]", zolt_val, jolt_val))

        # Compare InstructionInput gamma
        zolt_instr_gamma = self.extract_zolt_le_hex("STAGE3_INSTR: gamma = ")
        jolt_instr_gamma = self.extract_jolt_debug_bytes("[JOLT_INSTR] gamma = ")
        results.append(self.compare_values("Instr gamma", zolt_instr_gamma, jolt_instr_gamma))

        # Compare Registers gamma
        zolt_reg_gamma = self.extract_zolt_le_hex("STAGE3_REG: gamma = ")
        jolt_reg_gamma = self.extract_jolt_debug_bytes("[JOLT] STAGE3_REG: gamma = ")
        results.append(self.compare_values("Reg gamma", zolt_reg_gamma, jolt_reg_gamma))

        # Compare shift input claim individual components
        shift_components = [
            ("NextUnexpandedPC", "input_claim_next_unexpanded_pc"),
            ("NextPC", "input_claim_next_pc"),
            ("NextIsVirtual", "input_claim_next_is_virtual"),
            ("NextIsFirstInSequence", "input_claim_next_is_first_in_sequence"),
            ("NextIsNoop", "input_claim_next_is_noop"),
        ]
        for zolt_name, jolt_name in shift_components:
            zolt_val = self.extract_zolt_le_hex(f"STAGE3_SHIFT_INPUT: {zolt_name} = ")
            jolt_val = self.extract_jolt_debug_bytes(f"[JOLT_SHIFT] {jolt_name} = ")
            results.append(self.compare_values(f"Shift {zolt_name}", zolt_val, jolt_val))

        # Compare input claims for each instance
        for i, name in enumerate(["Shift", "InstrInput", "Registers"]):
            zolt_val = self.extract_zolt_le_hex(f"STAGE3_PRE: input_claim[{i}]")
            jolt_val = self.extract_single(f"STAGE3_PRE: input_claim[{i}]_bytes = ", self.jolt_log)
            if not jolt_val:
                jolt_val = self.extract_jolt_debug_bytes(f"[JOLT] STAGE3_PRE: input_claim[{i}] = ")
            results.append(self.compare_values(f"Stage3 input_claim[{i}] ({name})", zolt_val, jolt_val))

        # Compare batching coefficients
        for i in range(3):
            zolt_val = self.extract_zolt_le_hex(f"STAGE3_PRE: batching_coeff[{i}] = ")
            jolt_val = self.extract_single(f"STAGE3_PRE: batching_coeff[{i}]_bytes = ", self.jolt_log)
            results.append(self.compare_values(f"Stage3 batching_coeff[{i}]", zolt_val, jolt_val))

        # Compare initial batched claim
        zolt_claim = self.extract_zolt_le_hex("STAGE3_INITIAL: batched_claim = ")
        jolt_claim = self.extract_single("STAGE3_INITIAL: batched_claim_bytes = ", self.jolt_log)
        results.append(self.compare_values("Stage3 initial claim", zolt_claim, jolt_claim))

        # Compare MLE sample values
        mle_fields = [
            ("STAGE3_MLE: unexpanded_pc[0]", "MLE unexpanded_pc[0]"),
            ("STAGE3_MLE: pc[0]", "MLE pc[0]"),
            ("STAGE3_MLE: is_virtual[0]", "MLE is_virtual[0]"),
            ("STAGE3_MLE: is_noop[0]", "MLE is_noop[0]"),
            ("STAGE3_MLE: left_is_rs1[0]", "MLE left_is_rs1[0]"),
            ("STAGE3_MLE: rs1_value[0]", "MLE rs1_value[0]"),
            ("STAGE3_MLE: rd_write_value[0]", "MLE rd_write_value[0]"),
        ]
        for pattern, name in mle_fields:
            zolt_val = self.extract_zolt_le_hex(f"{pattern} = ")
            jolt_val = self.extract_jolt_debug_bytes(f"[JOLT] {pattern} = ")
            if zolt_val or jolt_val:
                results.append(self.compare_values(name, zolt_val, jolt_val))

        # Compare eq polynomial evaluations
        eq_fields = [
            ("STAGE3_EQ: eq_r_outer[0]", "eq_r_outer[0]"),
            ("STAGE3_EQ: eq_r_product[0]", "eq_r_product[0]"),
            ("STAGE3_EQ: eq_plus_one_outer[0]", "eq_plus_one_outer[0]"),
            ("STAGE3_EQ: eq_plus_one_product[0]", "eq_plus_one_product[0]"),
        ]
        for pattern, name in eq_fields:
            zolt_val = self.extract_zolt_le_hex(f"{pattern} = ")
            jolt_val = self.extract_jolt_debug_bytes(f"[JOLT] {pattern} = ")
            if zolt_val or jolt_val:
                results.append(self.compare_values(name, zolt_val, jolt_val))

        # Compare first 10 rounds (or all if fewer)
        num_rounds = min(self.find_max_round("STAGE3_ROUND_"), 10)
        for i in range(num_rounds):
            results.extend(self.compare_stage3_round(i))

        # Compare final output claim
        zolt_final = self.extract_zolt_le_hex("STAGE3_FINAL: output_claim = ")
        jolt_final = self.extract_jolt_debug_bytes("[JOLT] STAGE3_FINAL: output_claim = ")
        results.append(self.compare_values("Stage3 final claim", zolt_final, jolt_final))

        # Compare final individual claims
        for name in ["shift_claim", "instr_claim", "reg_claim"]:
            zolt_val = self.extract_zolt_le_hex(f"STAGE3_FINAL: {name} = ")
            jolt_val = self.extract_jolt_debug_bytes(f"[JOLT] STAGE3_FINAL: {name} = ")
            if zolt_val or jolt_val:
                results.append(self.compare_values(f"Stage3 final {name}", zolt_val, jolt_val))

        # Compare opening claims
        opening_fields = [
            ("STAGE3_OPENING: unexpanded_pc", "Opening unexpanded_pc"),
            ("STAGE3_OPENING: pc", "Opening pc"),
            ("STAGE3_OPENING: is_virtual", "Opening is_virtual"),
            ("STAGE3_OPENING: is_noop", "Opening is_noop"),
            ("STAGE3_OPENING: left_is_rs1", "Opening left_is_rs1"),
            ("STAGE3_OPENING: rs1_value", "Opening rs1_value"),
            ("STAGE3_OPENING: rd_write_value", "Opening rd_write_value"),
        ]
        for pattern, name in opening_fields:
            zolt_val = self.extract_zolt_le_hex(f"{pattern} = ")
            jolt_val = self.extract_jolt_debug_bytes(f"[JOLT] {pattern} = ")
            if zolt_val or jolt_val:
                results.append(self.compare_values(name, zolt_val, jolt_val))

        return results

    def compare_stage3_round(self, round_idx: int) -> List[CompareResult]:
        """Compare Stage 3 round values"""
        results = []

        # Current claim
        zolt_claim = self.extract_zolt_le_hex(f"STAGE3_ROUND_{round_idx}: current_claim = ")
        jolt_claim = self.extract_jolt_debug_bytes(f"[JOLT] STAGE3_ROUND_{round_idx}: current_claim = ")
        if not jolt_claim:
            jolt_claim = self.extract_single(f"STAGE3_ROUND_{round_idx}: current_claim_bytes = ", self.jolt_log)
        results.append(self.compare_values(f"Stage3 round {round_idx} claim", zolt_claim, jolt_claim))

        # Individual sumcheck claims
        for name in ["shift_claim", "instr_claim", "reg_claim"]:
            zolt_val = self.extract_zolt_le_hex(f"STAGE3_ROUND_{round_idx}: {name} = ")
            jolt_val = self.extract_jolt_debug_bytes(f"[JOLT] STAGE3_ROUND_{round_idx}: {name} = ")
            if zolt_val or jolt_val:
                results.append(self.compare_values(f"Stage3 round {round_idx} {name}", zolt_val, jolt_val))

        # Coefficients c0, c2, c3
        for coeff in ["c0", "c2", "c3"]:
            zolt_val = self.extract_zolt_le_hex(f"STAGE3_ROUND_{round_idx}: {coeff} = ")
            jolt_val = self.extract_single(f"STAGE3_ROUND_{round_idx}: {coeff}_bytes = ", self.jolt_log)
            if not jolt_val:
                jolt_val = self.extract_jolt_debug_bytes(f"[JOLT] STAGE3_ROUND_{round_idx}: {coeff} = ")
            results.append(self.compare_values(f"Stage3 round {round_idx} {coeff}", zolt_val, jolt_val))

        # Challenge
        zolt_ch = self.extract_zolt_le_hex(f"STAGE3_ROUND_{round_idx}: challenge = ")
        jolt_ch = self.extract_single(f"STAGE3_ROUND_{round_idx}: challenge_bytes = ", self.jolt_log)
        if not jolt_ch:
            jolt_ch = self.extract_jolt_debug_bytes(f"[JOLT] STAGE3_ROUND_{round_idx}: challenge = ")
        results.append(self.compare_values(f"Stage3 round {round_idx} challenge", zolt_ch, jolt_ch))

        # Next claim
        zolt_next = self.extract_zolt_le_hex(f"STAGE3_ROUND_{round_idx}: next_claim = ")
        jolt_next = self.extract_jolt_debug_bytes(f"[JOLT] STAGE3_ROUND_{round_idx}: next_claim = ")
        if zolt_next or jolt_next:
            results.append(self.compare_values(f"Stage3 round {round_idx} next_claim", zolt_next, jolt_next))

        return results

    def extract_jolt_debug_bytes(self, pattern: str) -> Optional[str]:
        """Extract bytes from Jolt Debug format: SomeField(0x..., ...)"""
        for line in self.jolt_log.split('\n'):
            if pattern in line:
                # Try to find ark-ff field debug format like: SomeField(0xabc123...)
                match = re.search(r'\(0x([0-9a-fA-F]+)', line)
                if match:
                    hex_val = match.group(1).lower()
                    # Pad to 64 chars if needed
                    if len(hex_val) < 64:
                        hex_val = hex_val.zfill(64)
                    # The 0x format is big-endian, convert to little-endian bytes for comparison
                    # Actually, let's return as-is and handle in comparison
                    return hex_val

                # Try byte array format [n1, n2, ...]
                bytes_val = self.parser.parse_jolt_bytes(line)
                if bytes_val and len(bytes_val) >= 32:
                    return self.parser.bytes_to_hex(bytes_val[:32])
        return None

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
            all_results['Stage 2 (Batched Sumcheck)'] = self.compare_stage2_batched()

        if stages is None or 3 in stages:
            all_results['Stage 3 (Lasso Lookup)'] = self.compare_stage3()

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

            # Print values if mismatch, verbose, or Stage 2/3 (for debugging)
            show_values = r.result == MatchResult.MISMATCH or verbose or "Stage 2" in section or "Stage 3" in section
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

def run_transcript_comparison(zolt_log: str, jolt_log: str, verbose: bool = False):
    """Run transcript state comparison to find state-dependent divergence"""
    print(f"\n{Color.BOLD}{'='*70}")
    print(f"  TRANSCRIPT STATE ANALYSIS")
    print(f"{'='*70}{Color.RESET}")

    analyzer = TranscriptAnalyzer(zolt_log, jolt_log)

    # First, compare stage boundaries
    print(f"\n{Color.BOLD}Stage Boundary States:{Color.RESET}")
    boundaries = analyzer.compare_stage_boundaries()
    for name, zolt_state, jolt_state, matches in boundaries:
        status = f"{Color.GREEN}MATCH{Color.RESET}" if matches else f"{Color.RED}MISMATCH{Color.RESET}"
        print(f"  {status:30} {name}")
        if not matches or verbose:
            print(f"    {Color.YELLOW}ZOLT:{Color.RESET} {zolt_state[:48]}..." if len(zolt_state) > 48 else f"    {Color.YELLOW}ZOLT:{Color.RESET} {zolt_state}")
            print(f"    {Color.BLUE}JOLT:{Color.RESET} {jolt_state[:48]}..." if len(jolt_state) > 48 else f"    {Color.BLUE}JOLT:{Color.RESET} {jolt_state}")

    # Find first divergence in transcript operations
    print(f"\n{Color.BOLD}Transcript Operation Analysis:{Color.RESET}")
    divergence = analyzer.find_state_divergence()

    if divergence:
        idx, zolt_op, jolt_op, reason = divergence
        print(f"\n{Color.RED}{Color.BOLD}TRANSCRIPT DIVERGENCE FOUND at operation {idx}:{Color.RESET}")
        print(f"  Reason: {Color.CYAN}{reason}{Color.RESET}")

        if zolt_op and jolt_op:
            print(f"\n  {Color.YELLOW}ZOLT operation:{Color.RESET}")
            print(f"    type: {zolt_op.op_type}, label: {zolt_op.label}")
            print(f"    value: {zolt_op.value[:40]}..." if len(zolt_op.value) > 40 else f"    value: {zolt_op.value}")
            print(f"    round: {zolt_op.round_num}")
            print(f"    state_after: {zolt_op.state_after[:48]}..." if len(zolt_op.state_after) > 48 else f"    state_after: {zolt_op.state_after}")

            print(f"\n  {Color.BLUE}JOLT operation:{Color.RESET}")
            print(f"    type: {jolt_op.op_type}, label: {jolt_op.label}")
            print(f"    value: {jolt_op.value[:40]}..." if len(jolt_op.value) > 40 else f"    value: {jolt_op.value}")
            print(f"    round: {jolt_op.round_num}")
            print(f"    state_after: {jolt_op.state_after[:48]}..." if len(jolt_op.state_after) > 48 else f"    state_after: {jolt_op.state_after}")

        # Print surrounding context
        analyzer.print_ops_around(idx, context=5)

        print(f"\n{Color.BOLD}What this means:{Color.RESET}")
        print(f"  The transcript state diverged at operation {idx}.")
        print(f"  Even if sumcheck values match, the verification will fail because")
        print(f"  challenges are derived from this cumulative state.")
        print(f"  Check what was appended differently at this point.")
    else:
        zolt_count = len(analyzer.zolt_ops)
        jolt_count = len(analyzer.jolt_ops)
        print(f"  {Color.GREEN}No transcript divergence detected!{Color.RESET}")
        print(f"  Parsed {zolt_count} Zolt ops and {jolt_count} Jolt ops")

        if zolt_count == 0 or jolt_count == 0:
            print(f"\n  {Color.YELLOW}WARNING: No transcript operations parsed.{Color.RESET}")
            print(f"  Make sure your logs include [ZOLT TRANSCRIPT] and [JOLT TRANSCRIPT] output.")
            print(f"  You may need to enable transcript debug logging in both implementations.")

    # Stage 2 tau_high specific analysis (common failure point)
    print(f"\n{Color.BOLD}Stage 2 tau_high Analysis (common failure point):{Color.RESET}")
    tau_high_info = analyzer.analyze_stage2_tau_high()

    if tau_high_info['zolt_pre_state'] or tau_high_info['jolt_pre_state']:
        state_status = f"{Color.GREEN}MATCH{Color.RESET}" if tau_high_info['states_match'] else f"{Color.RED}MISMATCH{Color.RESET}"
        print(f"  Pre-sampling state: {state_status}")
        if not tau_high_info['states_match'] or verbose:
            print(f"    {Color.YELLOW}ZOLT:{Color.RESET} {tau_high_info['zolt_pre_state'][:48] if tau_high_info['zolt_pre_state'] else 'N/A'}...")
            print(f"    {Color.BLUE}JOLT:{Color.RESET} {tau_high_info['jolt_pre_state'][:48] if tau_high_info['jolt_pre_state'] else 'N/A'}...")

        if tau_high_info['zolt_r_cycle_len'] or tau_high_info['jolt_r_cycle_len']:
            r_match = tau_high_info['zolt_r_cycle_len'] == tau_high_info['jolt_r_cycle_len']
            r_status = f"{Color.GREEN}MATCH{Color.RESET}" if r_match else f"{Color.RED}MISMATCH{Color.RESET}"
            print(f"  r_cycle.len: {r_status} (Zolt: {tau_high_info['zolt_r_cycle_len']}, Jolt: {tau_high_info['jolt_r_cycle_len']})")

        tau_status = f"{Color.GREEN}MATCH{Color.RESET}" if tau_high_info['tau_high_match'] else f"{Color.RED}MISMATCH{Color.RESET}"
        print(f"  Sampled tau_high: {tau_status}")
        if not tau_high_info['tau_high_match'] or verbose:
            print(f"    {Color.YELLOW}ZOLT:{Color.RESET} {tau_high_info['zolt_tau_high'][:48] if tau_high_info['zolt_tau_high'] else 'N/A'}...")
            print(f"    {Color.BLUE}JOLT:{Color.RESET} {tau_high_info['jolt_tau_high'][:48] if tau_high_info['jolt_tau_high'] else 'N/A'}...")

        if not tau_high_info['states_match'] and tau_high_info['zolt_pre_state'] and tau_high_info['jolt_pre_state']:
            print(f"\n  {Color.RED}{Color.BOLD}LIKELY ROOT CAUSE:{Color.RESET}")
            print(f"  Transcript state differs BEFORE Stage 2 tau_high sampling.")
            print(f"  This means Stage 1 or commitments left the transcript in different states.")
            print(f"  Check: commitment byte order, Stage 1 final values, r_cycle serialization.")
    else:
        print(f"  {Color.YELLOW}No Stage 2 tau_high debug output found in logs.{Color.RESET}")
        print(f"  Add STAGE2_PRE transcript_state logging to debug.")

    # Opening claims analysis
    print(f"\n{Color.BOLD}Opening Claims Analysis:{Color.RESET}")
    claims = analyzer.analyze_opening_claims()
    if claims:
        for claim_id, zolt_val, jolt_val, matches in claims:
            status = f"{Color.GREEN}MATCH{Color.RESET}" if matches else f"{Color.RED}MISMATCH{Color.RESET}"
            print(f"  {status:30} {claim_id}")
            if not matches or verbose:
                print(f"    {Color.YELLOW}ZOLT:{Color.RESET} {zolt_val[:32] if zolt_val and zolt_val != 'N/A' else zolt_val}...")
                print(f"    {Color.BLUE}JOLT:{Color.RESET} {jolt_val[:32] if jolt_val and jolt_val != 'N/A' else jolt_val}...")
    else:
        print(f"  {Color.DIM}No opening claims found in logs.{Color.RESET}")
        print(f"  Add 'Opening claim: <id> => eval=...' logging to track claims.")


def main():
    parser = argparse.ArgumentParser(description='Compare Zolt and Jolt verification logs')
    parser.add_argument('zolt_log', nargs='?', default='/tmp/zolt.log', help='Zolt log file')
    parser.add_argument('jolt_log', nargs='?', default='/tmp/jolt.log', help='Jolt log file')
    parser.add_argument('--stage', '-s', type=int, nargs='+', help='Compare specific stages (0=preamble, 1-6)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show all values, not just mismatches')
    parser.add_argument('--find-divergence', '-d', action='store_true', help='Find first point of divergence')
    parser.add_argument('--transcript', '-t', action='store_true',
                        help='Compare transcript state evolution (finds state-dependent bugs)')
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

    # Run transcript comparison if requested
    if args.transcript:
        run_transcript_comparison(zolt_log, jolt_log, args.verbose)

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

    # If all values match but user didn't use --transcript, suggest it
    all_match = all(
        r.result in [MatchResult.MATCH, MatchResult.MATCH_REVERSED, MatchResult.MISSING_BOTH,
                     MatchResult.MISSING_ZOLT, MatchResult.MISSING_JOLT]
        for section_results in results.values()
        for r in section_results
    )
    if all_match and not args.transcript:
        print(f"\n{Color.YELLOW}{Color.BOLD}TIP:{Color.RESET} All values match but verification still failing?")
        print(f"     Run with {Color.CYAN}--transcript{Color.RESET} to compare transcript state evolution.")
        print(f"     This can catch state-dependent bugs that value comparison misses.")

if __name__ == '__main__':
    main()
