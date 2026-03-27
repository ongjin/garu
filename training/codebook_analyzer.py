"""Codebook-based Korean morphological analyzer (prototype).

Usage:
    from codebook_analyzer import CodebookAnalyzer
    analyzer = CodebookAnalyzer.load("training/codebook_data")
    result = analyzer.analyze("학교에서 공부했다")
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional

POS_TAGS = [
    "NNG", "NNP", "NNB", "NR", "NP",
    "VV", "VA", "VX", "VCP", "VCN",
    "MAG", "MAJ", "MM", "IC",
    "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC",
    "EP", "EF", "EC", "ETN", "ETM",
    "XPN", "XSN", "XSV", "XSA", "XR",
    "SF", "SP", "SS", "SE", "SO", "SW", "SH", "SL", "SN",
]
POS_TO_IDX = {p: i for i, p in enumerate(POS_TAGS)}


class LatticeArc:
    """An arc in the morpheme lattice."""
    __slots__ = ['start', 'end', 'morphemes', 'cost']

    def __init__(self, start: int, end: int, morphemes: List[Tuple[str, str]], cost: float):
        self.start = start      # char start position
        self.end = end          # char end position
        self.morphemes = morphemes  # [(surface, POS), ...]
        self.cost = cost        # word cost


class CodebookAnalyzer:
    def __init__(self, suffix_codebook, content_dict, trigram_costs, bigram_costs, default_cost, word_freqs):
        self.suffix_codebook = suffix_codebook  # {surface: [{morphemes: [POS...], freq}]}
        self.content_dict = content_dict        # {surface: (POS, freq)}
        self.trigram_costs = trigram_costs       # {(p1,p2,p3): cost}
        self.bigram_costs = bigram_costs         # {(p1,p2): cost}
        self.default_cost = default_cost
        self.word_freqs = word_freqs            # {surface: freq} for word cost
        self.max_word_len = max((len(w) for w in content_dict), default=1)
        self.max_suffix_len = max((len(s) for s in suffix_codebook), default=1)

    @classmethod
    def load(cls, data_dir: str) -> 'CodebookAnalyzer':
        d = Path(data_dir)

        with open(d / "suffix_codebook.json") as f:
            suffix_codebook = json.load(f)

        content_dict = {}
        word_freqs = {}
        with open(d / "content_dict.txt") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    surface, pos, freq = parts[0], parts[1], int(parts[2])
                    content_dict[surface] = (pos, freq)
                    word_freqs[surface] = freq

        with open(d / "trigram_costs.json") as f:
            cost_data = json.load(f)

        trigram_costs = {}
        for key, cost in cost_data["trigram"].items():
            p1, p2, p3 = key.split(",")
            trigram_costs[(p1, p2, p3)] = cost

        bigram_costs = {}
        for key, cost in cost_data["bigram"].items():
            p1, p2 = key.split(",")
            bigram_costs[(p1, p2)] = cost

        default_cost = cost_data.get("default_cost", 15.0)

        return cls(suffix_codebook, content_dict, trigram_costs, bigram_costs, default_cost, word_freqs)

    def get_trigram_cost(self, p1: str, p2: str, p3: str) -> float:
        """Get trigram transition cost with bigram backoff."""
        key = (p1, p2, p3)
        if key in self.trigram_costs:
            return self.trigram_costs[key]
        # Backoff to bigram
        bg_key = (p2, p3)
        if bg_key in self.bigram_costs:
            return self.bigram_costs[bg_key] + 2.0  # penalty for backoff
        return self.default_cost

    def get_word_cost(self, surface: str, pos: str) -> float:
        """Cost of a word (lower = more common)."""
        freq = self.word_freqs.get(surface, 0)
        if freq > 0:
            return -math.log(freq + 1)  # +1 smoothing, negated so common = low cost
        return 10.0  # OOV penalty

    def classify_oov_char(self, ch: str) -> str:
        """Classify a single character for OOV handling."""
        if ch in '.!?':
            return 'SF'
        if ch in ',;:':
            return 'SP'
        if ch in '()[]{}""\'\'':
            return 'SS'
        if ch.isascii() and ch.isalpha():
            return 'SL'
        if ch.isdigit():
            return 'SN'
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            return 'NNG'  # Hangul → assume noun
        return 'SW'

    def build_lattice(self, text: str) -> dict:
        """Build morpheme lattice (DAG). Returns {position: [LatticeArc]}."""
        n = len(text)
        arcs_from = defaultdict(list)  # start_pos -> [LatticeArc]

        for i in range(n):
            ch = text[i]

            # Skip spaces — they are natural word boundaries
            if ch == ' ':
                continue

            # Strategy A: Content word match (try all lengths)
            for end in range(i + 1, min(i + self.max_word_len + 1, n + 1)):
                substr = text[i:end]
                if ' ' in substr:
                    break  # don't span across spaces
                if substr in self.content_dict:
                    pos, freq = self.content_dict[substr]
                    cost = self.get_word_cost(substr, pos)
                    morphemes = [(substr, pos)]
                    arcs_from[i].append(LatticeArc(i, end, morphemes, cost))

                    # After content word, try suffix patterns
                    for s_end in range(end + 1, min(end + self.max_suffix_len + 1, n + 1)):
                        suffix = text[end:s_end]
                        if ' ' in suffix:
                            break
                        if suffix in self.suffix_codebook:
                            for analysis in self.suffix_codebook[suffix]:
                                suffix_morphemes = [(suffix, p) for p in analysis["morphemes"]]
                                s_cost = -math.log(analysis["freq"] + 1)
                                combined = morphemes + suffix_morphemes
                                total_cost = cost + s_cost
                                arcs_from[i].append(LatticeArc(i, s_end, combined, total_cost))

            # Strategy B: Suffix-only match (standalone functional morphemes)
            for end in range(i + 1, min(i + self.max_suffix_len + 1, n + 1)):
                substr = text[i:end]
                if ' ' in substr:
                    break
                if substr in self.suffix_codebook:
                    for analysis in self.suffix_codebook[substr]:
                        morphemes = [(substr, p) for p in analysis["morphemes"]]
                        cost = -math.log(analysis["freq"] + 1)
                        arcs_from[i].append(LatticeArc(i, end, morphemes, cost))

            # OOV fallback: single character arc
            if not arcs_from[i]:
                pos = self.classify_oov_char(ch)
                arcs_from[i].append(LatticeArc(i, i + 1, [(ch, pos)], 10.0))

        # Merge consecutive OOV chars of same type
        # (This happens naturally during Viterbi — consecutive SL chars will be merged in post-processing)

        return arcs_from

    def viterbi(self, text: str, arcs_from: dict) -> List[Tuple[str, str]]:
        """Find lowest-cost path through lattice using trigram Viterbi."""
        n = len(text)
        if n == 0:
            return []

        # State: (position, prev_pos, prev_prev_pos) -> (cost, backpointer)
        # backpointer = (prev_position, prev_prev_pos_before, arc)
        INF = float('inf')
        BOS = "<BOS>"

        # dp[pos][(prev_pos, prev_prev_pos)] = (cost, backpointer)
        dp = [dict() for _ in range(n + 1)]
        dp[0][(BOS, BOS)] = (0.0, None)

        for i in range(n):
            if text[i] == ' ':
                # Pass through spaces
                for state, (cost, bp) in dp[i].items():
                    if i + 1 not in range(len(dp)):
                        continue
                    if state not in dp[i + 1] or dp[i + 1][state][0] > cost:
                        dp[i + 1][state] = (cost, bp if bp else (i, state, None))
                continue

            if i not in arcs_from:
                continue

            for arc in arcs_from[i]:
                for state, (prev_cost, prev_bp) in dp[i].items():
                    prev_pos, prev_prev_pos = state

                    # Compute transition cost for each morpheme in this arc
                    trans_cost = 0.0
                    pp, p = prev_prev_pos, prev_pos
                    for _, morph_pos in arc.morphemes:
                        trans_cost += self.get_trigram_cost(pp, p, morph_pos)
                        pp = p
                        p = morph_pos

                    total_cost = prev_cost + arc.cost + trans_cost
                    new_state = (p, pp)  # (last_pos, second_to_last_pos)

                    end = arc.end
                    if new_state not in dp[end] or dp[end][new_state][0] > total_cost:
                        dp[end][new_state] = (total_cost, (i, state, arc))

        # Find best final state
        best_cost = INF
        best_state = None
        for state, (cost, bp) in dp[n].items():
            if cost < best_cost:
                best_cost = cost
                best_state = state

        if best_state is None:
            # Fallback: character-by-character OOV
            return [(ch, self.classify_oov_char(ch)) for ch in text if ch != ' ']

        # Backtrack
        arcs_in_path = []
        pos = n
        state = best_state
        while dp[pos][state][1] is not None:
            prev_pos, prev_state, arc = dp[pos][state][1]
            if arc is not None:
                arcs_in_path.append(arc)
            pos = prev_pos
            state = prev_state

        arcs_in_path.reverse()

        # Collect morphemes
        result = []
        for arc in arcs_in_path:
            result.extend(arc.morphemes)

        return result

    def analyze(self, text: str) -> List[Tuple[str, str]]:
        """Analyze text into morphemes. Returns [(surface, POS), ...]."""
        if not text:
            return []
        arcs = self.build_lattice(text)
        return self.viterbi(text, arcs)
