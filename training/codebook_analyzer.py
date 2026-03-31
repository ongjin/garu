"""Codebook-based Korean morphological analyzer (prototype).

Usage:
    from codebook_analyzer import CodebookAnalyzer
    analyzer = CodebookAnalyzer.load("training/codebook_data")
    result = analyzer.analyze("학교에서 공부했다")
"""

import json
import math
import re
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

CONTENT_POS = {"NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "VX", "VCP", "VCN",
               "MAG", "MAJ", "MM", "IC", "XR"}
FUNC_POS = {"JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC",
            "EP", "EF", "EC", "ETN", "ETM", "XPN", "XSN", "XSV", "XSA"}

MORPHEME_PENALTY = 3.0
OOV_PENALTY = 15.0
LENGTH_BONUS = 2.5
SINGLE_CHAR_CONTENT_PENALTY = 4.0


def is_pure_functional(morphemes):
    """Check if all morphemes are functional POS tags.
    Handles both old format (list of POS strings) and new format (list of [form, pos] pairs)."""
    for m in morphemes:
        if isinstance(m, list) and len(m) == 2:
            if m[1] not in FUNC_POS:
                return False
        elif m not in FUNC_POS:
            return False
    return True


class LatticeArc:
    __slots__ = ['start', 'end', 'morphemes', 'cost']

    def __init__(self, start, end, morphemes, cost):
        self.start = start
        self.end = end
        self.morphemes = morphemes
        self.cost = cost


class CodebookAnalyzer:
    def __init__(self, suffix_codebook, content_dict, trigram_costs, bigram_costs, default_cost, word_freqs):
        self.suffix_codebook = suffix_codebook
        self.content_dict = content_dict
        self.trigram_costs = trigram_costs
        self.bigram_costs = bigram_costs
        self.default_cost = default_cost
        self.word_freqs = word_freqs
        self.max_freq = max(word_freqs.values()) if word_freqs else 1
        self.max_word_len = max((len(w) for w in content_dict), default=1)
        self.max_suffix_len = max((len(s) for s in suffix_codebook), default=1)
        self.max_suffix_freq = max(
            (a["freq"] for entries in suffix_codebook.values() for a in entries),
            default=1)

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

    def get_trigram_cost(self, p1, p2, p3):
        key = (p1, p2, p3)
        if key in self.trigram_costs:
            return self.trigram_costs[key]
        bg_key = (p2, p3)
        if bg_key in self.bigram_costs:
            return self.bigram_costs[bg_key] + 2.0
        return self.default_cost

    def get_word_cost(self, surface, pos):
        freq = self.word_freqs.get(surface, 0)
        if freq > 0:
            cost = math.log(self.max_freq / freq)
            char_len = len(surface)
            if char_len > 1:
                cost -= LENGTH_BONUS * (char_len - 1)
            elif char_len == 1 and pos in CONTENT_POS:
                cost += SINGLE_CHAR_CONTENT_PENALTY
            return cost
        return OOV_PENALTY

    def get_suffix_cost(self, surface, analysis):
        freq = analysis["freq"]
        morphemes = analysis["morphemes"]
        cost = math.log(self.max_suffix_freq / max(freq, 1))
        cost += MORPHEME_PENALTY * len(morphemes)
        char_len = len(surface)
        if char_len > 1:
            cost -= LENGTH_BONUS * (char_len - 1)
        return cost

    def classify_oov_char(self, ch):
        if ch in '.!?':
            return 'SF'
        if ch in ',;:/\u00B7':
            return 'SP'
        if ch == '\u2026':  # …
            return 'SE'
        if ch in '~-\u2013\u2212':  # ~, -, –, −
            return 'SO'
        if ch in '\u300C\u300D\u300E\u300F':
            return 'SS'  # CJK brackets only (「」『』)
        if ch in '()[]{}"\'\u2018\u2019\u201C\u201D\u2015\u2014<>\u3008\u3009\u300A\u300B':
            return 'SW'  # parentheses, brackets, quotes → SW (Kiwi convention)
        if ch.isascii() and ch.isalpha():
            return 'SL'
        if ch.isdigit():
            return 'SN'
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            return 'NNG'
        if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0xF900 <= code <= 0xFAFF:
            return 'NNG'  # Kiwi treats Hanja as NNG
        return 'SW'

    def _preprocess_ascii_runs(self, text):
        runs = []
        for m in re.finditer(r'[A-Za-z][A-Za-z.]*[A-Za-z]|[A-Za-z]|\d+', text):
            start, end = m.start(), m.end()
            surface = m.group()
            if surface[0].isdigit():
                runs.append((start, end, 'SN'))
            else:
                runs.append((start, end, 'SL'))
        return runs

    def build_lattice(self, text):
        n = len(text)
        arcs_from = defaultdict(list)

        # ASCII runs
        ascii_runs = self._preprocess_ascii_runs(text)
        ascii_interior = set()
        for start, end, pos_tag in ascii_runs:
            surface = text[start:end]
            arcs_from[start].append(LatticeArc(start, end, [(surface, pos_tag)], 1.0))
            for p in range(start + 1, end):
                ascii_interior.add(p)

        for i in range(n):
            ch = text[i]
            if ch == ' ' or i in ascii_interior:
                continue

            # Strategy A: Content word + optional suffix
            for end in range(min(i + self.max_word_len, n), i, -1):
                substr = text[i:end]
                if ' ' in substr:
                    continue
                if substr not in self.content_dict:
                    continue
                pos, freq = self.content_dict[substr]
                cost = self.get_word_cost(substr, pos)
                morphemes = [(substr, pos)]
                arcs_from[i].append(LatticeArc(i, end, morphemes, cost))

                # Try pure-functional suffixes after content word
                for s_end in range(end + 1, min(end + self.max_suffix_len + 1, n + 1)):
                    suffix = text[end:s_end]
                    if ' ' in suffix:
                        break
                    if suffix in self.suffix_codebook:
                        for analysis in self.suffix_codebook[suffix]:
                            if not is_pure_functional(analysis["morphemes"]):
                                continue
                            if isinstance(analysis["morphemes"][0], list):
                                suffix_morphemes = [(m[0], m[1]) for m in analysis["morphemes"]]
                            else:
                                suffix_morphemes = [(suffix, p) for p in analysis["morphemes"]]
                            s_cost = self.get_suffix_cost(suffix, analysis)
                            arcs_from[i].append(LatticeArc(
                                i, s_end, morphemes + suffix_morphemes, cost + s_cost))

            # Strategy B: Pure functional suffix standalone
            for end in range(i + 1, min(i + self.max_suffix_len + 1, n + 1)):
                substr = text[i:end]
                if ' ' in substr:
                    break
                if substr in self.suffix_codebook:
                    for analysis in self.suffix_codebook[substr]:
                        if not is_pure_functional(analysis["morphemes"]):
                            continue
                        if isinstance(analysis["morphemes"][0], list):
                            morphemes = [(m[0], m[1]) for m in analysis["morphemes"]]
                        else:
                            morphemes = [(substr, p) for p in analysis["morphemes"]]
                        cost = self.get_suffix_cost(substr, analysis)
                        arcs_from[i].append(LatticeArc(i, end, morphemes, cost))

            # Strategy C: Contracted forms (content+functional) from codebook
            for end in range(i + 1, min(i + self.max_suffix_len + 1, n + 1)):
                substr = text[i:end]
                if ' ' in substr:
                    break
                if substr in self.suffix_codebook:
                    for analysis in self.suffix_codebook[substr]:
                        if is_pure_functional(analysis["morphemes"]):
                            continue  # already handled in B
                        if not isinstance(analysis["morphemes"][0], list):
                            continue  # old format without surfaces, skip
                        morphemes = [(m[0], m[1]) for m in analysis["morphemes"]]
                        cost = self.get_suffix_cost(substr, analysis)
                        arcs_from[i].append(LatticeArc(i, end, morphemes, cost))

            # OOV fallback
            if not arcs_from[i]:
                pos = self.classify_oov_char(ch)
                arcs_from[i].append(LatticeArc(i, i + 1, [(ch, pos)], OOV_PENALTY))

        return arcs_from

    def viterbi(self, text, arcs_from):
        n = len(text)
        if n == 0:
            return []

        BOS = "<BOS>"
        dp = [dict() for _ in range(n + 1)]
        dp[0][(BOS, BOS)] = (0.0, None)

        for i in range(n):
            if not dp[i]:
                continue
            if text[i] == ' ':
                for state, (cost, bp) in dp[i].items():
                    if state not in dp[i + 1] or dp[i + 1][state][0] > cost:
                        dp[i + 1][state] = (cost, (i, state, None))
                continue
            if i not in arcs_from:
                continue

            for arc in arcs_from[i]:
                for state, (prev_cost, prev_bp) in dp[i].items():
                    prev_pos, prev_prev_pos = state
                    trans_cost = 0.0
                    pp, p = prev_prev_pos, prev_pos
                    for _, morph_pos in arc.morphemes:
                        trans_cost += self.get_trigram_cost(pp, p, morph_pos)
                        pp = p
                        p = morph_pos

                    total_cost = prev_cost + arc.cost + trans_cost
                    new_state = (p, pp)
                    end = arc.end
                    if new_state not in dp[end] or dp[end][new_state][0] > total_cost:
                        dp[end][new_state] = (total_cost, (i, state, arc))

        best_cost = float('inf')
        best_state = None
        for state, (cost, bp) in dp[n].items():
            if cost < best_cost:
                best_cost = cost
                best_state = state

        if best_state is None:
            return [(ch, self.classify_oov_char(ch)) for ch in text if ch != ' ']

        arcs_in_path = []
        pos = n
        state = best_state
        while dp[pos][state][1] is not None:
            prev_pos_val, prev_state, arc = dp[pos][state][1]
            if arc is not None:
                arcs_in_path.append(arc)
            pos = prev_pos_val
            state = prev_state
        arcs_in_path.reverse()

        raw_result = []
        for arc in arcs_in_path:
            raw_result.extend(arc.morphemes)

        # Merge consecutive single-char SL/SN/SW
        result = []
        for surface, pos in raw_result:
            if (result and pos == result[-1][1] and pos in ('SL', 'SN', 'SW')
                    and len(surface) == 1):
                result[-1] = (result[-1][0] + surface, pos)
            else:
                result.append((surface, pos))

        # Post-process: JKB → JC for 과/와 between nouns
        NOUN_POS = {'NNG', 'NNP', 'NNB', 'NR', 'NP', 'SN', 'SL'}
        NOUN_LIKE = {'NNG', 'NNP', 'NNB', 'NR', 'NP', 'SN', 'SL', 'MM', 'MAG'}
        for i in range(1, len(result)):
            if result[i][1] == 'JKB' and result[i][0] in ('과', '와', '이랑', '랑', '하고'):
                if result[i - 1][1] in NOUN_POS:
                    if i + 1 < len(result) and result[i + 1][1] in NOUN_LIKE:
                        result[i] = (result[i][0], 'JC')

        # Post-process: VX - auxiliary verbs after EC
        for i in range(1, len(result)):
            if result[i][1] == 'VV' and result[i - 1][1] == 'EC':
                form, prev_form = result[i][0], result[i - 1][0]
                if form in ('있', '없') and prev_form in ('고', '어', '아'):
                    result[i] = (form, 'VX')
                elif form == '하' and prev_form == '지':
                    result[i] = (form, 'VX')
                elif form in ('보', '주', '지', '오', '가', '내', '나', '버리', '놓', '두') and prev_form in ('어', '아'):
                    result[i] = (form, 'VX')

        # Post-process: XSV/XSA - 하/되 after NNG → XSV; 하/VA after NNG → XSA
        for i in range(1, len(result)):
            if result[i - 1][1] == 'NNG':
                if result[i][1] == 'VV' and result[i][0] in ('하', '되', '시키'):
                    result[i] = (result[i][0], 'XSV')
                elif result[i][1] == 'VA' and result[i][0] == '하':
                    result[i] = (result[i][0], 'XSA')

        # Post-process: NNB - dependency nouns after ETM
        NNB_AFTER_ETM = {'수', '것', '때', '데', '바', '번', '개', '시', '군', '줄',
                         '뿐', '채', '척', '듯', '리', '셈', '나름', '탓', '만큼',
                         '가지', '곳'}
        for i in range(1, len(result)):
            if result[i][1] in ('NNG', 'VV') and result[i - 1][1] == 'ETM':
                if result[i][0] in NNB_AFTER_ETM:
                    result[i] = (result[i][0], 'NNB')

        # Post-process: XSN - suffix nouns after NNG
        XSN_FORMS = {'성', '형', '적', '식', '계', '권', '자', '화', '률', '율',
                      '상', '장', '급', '제'}
        for i in range(1, len(result)):
            if result[i][1] == 'NNG' and len(result[i][0]) == 1:
                if result[i - 1][1] == 'NNG' and result[i][0] in XSN_FORMS:
                    result[i] = (result[i][0], 'XSN')

        # Post-process: XPN - prefix before NNG
        XPN_FORMS = {'비', '재', '초', '무', '탈', '반', '불'}
        for i in range(len(result) - 1):
            if result[i][1] == 'NNG' and len(result[i][0]) == 1:
                if result[i + 1][1] == 'NNG' and result[i][0] in XPN_FORMS:
                    result[i] = (result[i][0], 'XPN')

        # Post-process: MM - demonstratives before nouns
        NOUN_LIKE_PREV = {'NNG', 'NNP', 'NNB', 'NR', 'NP', 'SN', 'SL', 'XSN', 'XSA', 'XSV'}
        for i in range(len(result) - 1):
            if result[i + 1][1] not in ('NNG', 'NNP', 'NNB', 'NR', 'XR'):
                continue
            form, pos = result[i]
            # 이/그/저: only if NOT preceded by noun
            if form in ('이', '그', '저') and pos in ('NP', 'JKS'):
                preceded_by_noun = i > 0 and result[i - 1][1] in NOUN_LIKE_PREV
                if not preceded_by_noun:
                    result[i] = (form, 'MM')
            # 새/각/온/현/전: NNG → MM before nouns
            elif form in ('새', '각', '온', '현', '전') and pos == 'NNG':
                result[i] = (form, 'MM')

        # Post-process: JKC - complement marker before 되/아니
        for i in range(len(result) - 1):
            if result[i][1] == 'JKS' and result[i][0] in ('가', '이'):
                if result[i + 1][0] in ('되', '아니'):
                    result[i] = (result[i][0], 'JKC')

        return result

    def analyze(self, text):
        if not text:
            return []
        arcs = self.build_lattice(text)
        return self.viterbi(text, arcs)
