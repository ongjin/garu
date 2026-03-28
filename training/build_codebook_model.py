"""Build GMDL v3 codebook model from extracted data.

Reads codebook_data/ and produces models/codebook.gmdl in GMDL v3 binary format.

Usage:
    python training/build_codebook_model.py
"""
import gzip
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "training" / "codebook_data"
OUT_PATH = ROOT / "models" / "codebook.gmdl"

POS_TAGS = [
    "NNG", "NNP", "NNB", "NR", "NP",
    "VV", "VA", "VX", "VCP", "VCN",
    "MAG", "MAJ", "MM", "IC",
    "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC",
    "EP", "EF", "EC", "ETN", "ETM",
    "XPN", "XSN", "XSV", "XSA", "XR",
    "SF", "SP", "SS", "SE", "SO", "SW", "SH", "SL", "SN",
]
POS_TO_BYTE = {p: i for i, p in enumerate(POS_TAGS)}
NUM_POS = len(POS_TAGS)  # 42


def write_section(buf: bytearray, section_type: int, data: bytes):
    """Append a section header (type u8 + len u32) and data to buf."""
    buf.extend(struct.pack("B", section_type))
    buf.extend(struct.pack("<I", len(data)))
    buf.extend(data)


def pos_byte(tag: str) -> int:
    """Map POS tag string to byte. Falls back to NNP(1) for unknown tags."""
    return POS_TO_BYTE.get(tag, 1)


def build_content_dict_fst(dict_path: Path) -> tuple[bytes, int]:
    """Build Dict v2 (FST) format from content_dict.txt using the Rust build-dict tool.

    Reads content_dict.txt (word, pos_tag, freq), keeps highest-freq POS per word,
    shells out to `cargo run --release --bin build-dict` to produce FST binary.
    Returns (dict_bytes, max_freq).
    """
    # Parse content dict: keep highest-freq POS per word, filter by min freq
    MIN_CONTENT_FREQ = 7
    best = {}  # {word: (tag, freq)}
    max_freq = 0
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            word, tag, freq_str = parts[0], parts[1], parts[2]
            freq = int(freq_str)
            if freq < MIN_CONTENT_FREQ:
                continue
            # Skip punctuation/symbols in content dict — they should be handled by OOV classifier
            if len(word) == 1 and not ('\uAC00' <= word <= '\uD7A3') and not word.isalnum():
                continue
            if freq > max_freq:
                max_freq = freq
            if word not in best or freq > best[word][1]:
                best[word] = (tag, freq)

    if max_freq == 0:
        max_freq = 1

    # Wiki NNP disabled — experiments show removing wiki entries
    # improves F1 (+0.4%p) while halving model size (4.4MB → 2.0MB).
    # Wiki NNP entries collide with common words and suffix patterns.
    print(f"  Wiki NNP: disabled (improves F1 and reduces size)")

    # Remove content dict entries that conflict with suffix codebook.
    # This prevents low-freq nouns (배가/NNG, 긴/NNP, etc.) from blocking
    # correct morpheme analysis (배+가, 길+ᆫ).
    REMOVABLE_POS = {"NNG", "NNP", "NNB", "MM", "MAG", "IC"}
    FUNC_POS_SET = {"JKS","JKC","JKG","JKO","JKB","JKV","JKQ","JX","JC",
                    "EP","EF","EC","ETN","ETM","XPN","XSN","XSV","XSA"}
    codebook_path = DATA_DIR / "suffix_codebook.json"
    if codebook_path.exists():
        cb = json.load(open(codebook_path))
        removed = 0
        for word in list(best.keys()):
            if word not in cb:
                continue
            content_tag, content_freq = best[word]
            # Only remove nouns/adverbs — never remove pronouns, verbs, copulas
            if content_tag not in REMOVABLE_POS:
                continue
            # Check if ANY analysis has at least one functional morpheme
            max_func_freq = 0
            for a in cb[word]:
                has_func = False
                for m in a["morphemes"]:
                    mpos = m[1] if isinstance(m, list) else m
                    if mpos in FUNC_POS_SET:
                        has_func = True
                        break
                if has_func and a["freq"] > max_func_freq:
                    max_func_freq = a["freq"]
            # Remove if suffix is more common than content entry.
            # Aggressive for multi-char: these are almost always suffix patterns (e.g., 해서→하+어서)
            word_len = len(word)
            threshold = 2 if word_len >= 2 else 5
            if max_func_freq > content_freq * threshold:
                del best[word]
                removed += 1
        print(f"  Removed {removed} entries conflicting with suffix codebook")

    # Build multi-POS content dict: for ambiguous words, include top 2 POS
    # Load NIKL word→POS distribution for secondary POS
    nikl_word_pos = {}
    nikl_dir = Path.home() / "Downloads" / "NIKL_MP(v1.1)"
    NIKL_MAP_LOCAL = {'MMD':'MM','MMN':'MM','MMA':'MM','NA':'NNG','NAP':'NNG','NF':'NNG','NV':'VV'}
    POS_SET_LOCAL = set(POS_TAGS)
    def np_local(t):
        if t in POS_SET_LOCAL: return t
        if t in NIKL_MAP_LOCAL: return NIKL_MAP_LOCAL[t]
        b = t.split('-')[0]
        return b if b in POS_SET_LOCAL else 'SW'
    for fname in ["NXMP1902008040.json", "SXMP1902008031.json"]:
        npath = nikl_dir / fname
        if not npath.exists(): continue
        with open(npath) as nf:
            ndata = json.load(nf)
        for doc in ndata["document"]:
            if doc is None: continue
            for sent in (doc.get("sentence") or []):
                for m in (sent.get("morpheme") or []):
                    form = m.get("form", "").strip()
                    label = np_local(m.get("label", ""))
                    if form and label:
                        if form not in nikl_word_pos:
                            nikl_word_pos[form] = {}
                        nikl_word_pos[form][label] = nikl_word_pos[form].get(label, 0) + 1

    # Build input lines: word\tpos_byte\tfreq (allow duplicate words for 2nd POS)
    sorted_words = sorted(best.keys(), key=lambda w: w.encode("utf-8"))
    dual_count = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "dict_input.txt"
        output_path = Path(tmpdir) / "dict_output.bin"

        with open(input_path, "w", encoding="utf-8") as f:
            for word in sorted_words:
                tag, freq = best[word]
                pb = pos_byte(tag) if isinstance(tag, str) else tag
                f.write(f"{word}\t{pb}\t{freq}\n")

                # Add secondary POS from NIKL if significantly different
                if word in nikl_word_pos:
                    total = sum(nikl_word_pos[word].values())
                    if total >= 100:
                        for alt_tag, alt_count in sorted(nikl_word_pos[word].items(), key=lambda x: -x[1]):
                            if alt_tag == tag: continue
                            alt_pct = alt_count / total
                            if alt_pct >= 0.15:  # at least 15%
                                alt_freq = int(alt_pct * freq)  # scale to content dict range
                                alt_pb = pos_byte(alt_tag)
                                f.write(f"{word}\t{alt_pb}\t{alt_freq}\n")
                                dual_count += 1
                                break  # only 1 secondary
        print(f"  Dual-POS words: {dual_count}")

        # Run build-dict from repo root
        result = subprocess.run(
            ["cargo", "run", "--release", "--bin", "build-dict", "--",
             str(input_path), str(output_path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("build-dict stderr:", result.stderr)
            raise RuntimeError(f"build-dict failed with code {result.returncode}")

        # Print build-dict info
        for line in result.stderr.strip().splitlines():
            print(f"  [build-dict] {line}")

        dict_bytes = output_path.read_bytes()

    return dict_bytes, max_freq


MIN_SUFFIX_FREQ = 75


def decompose_hangul(ch):
    """Decompose Hangul syllable -> (lead, vowel, tail) indices."""
    code = ord(ch) - 0xAC00
    if code < 0 or code > 11171:
        return None
    return (code // (21 * 28), (code % (21 * 28)) // 28, code % 28)


def compose_hangul(lead, vowel, tail=0):
    """Compose Hangul syllable from (lead, vowel, tail) indices."""
    return chr(0xAC00 + lead * 21 * 28 + vowel * 28 + tail)


# Tail consonant indices
TAIL_NONE = 0; TAIL_BIEUP = 17; TAIL_DIGEUT = 7; TAIL_RIEUL = 8
TAIL_HIEUT = 27; TAIL_SIOT = 19
# Vowel indices
V_A = 0; V_AE = 1; V_EO = 4; V_E = 5; V_O = 8; V_WA = 9; V_WEO = 14
V_U = 13; V_EU = 18; V_I = 20
# Lead consonant indices (for reference)
L_RIEUL = 5

# Known irregular ㄷ stems
IRREG_DIGEUT_STEMS = {"걷", "듣", "묻", "싣", "깨닫", "일컫", "걷잡"}
# Known irregular ㅎ stems (VA only)
IRREG_HIEUT_STEMS = {"빨갛", "파랗", "노랗", "하얗", "까맣", "그렇", "이렇",
                      "저렇", "어떻", "아무렇", "누렇", "허옇", "시뻘겋",
                      "뻘겋", "벌겋", "발갛", "새빨갛", "시퍼렇", "퍼렇"}

# Suffixes to generate: (suffix_str, list_of_pos_tags)
# The first element of the suffix is what attaches directly to the stem
SUFFIX_COMBOS = [
    ("어", ["EC"]),
    ("어서", ["EC"]),
    ("었", ["EP"]),
    ("었다", ["EP", "EF"]),
    ("은", ["ETM"]),
    ("을", ["ETM"]),
    ("으니", ["EC"]),
    ("으면", ["EC"]),
]


def _make_suffix_morphemes(suffix_str, suffix_tags):
    """Break suffix string into morpheme pairs based on tags.

    For single-tag suffixes like ("어서", ["EC"]), return [["어서", "EC"]].
    For multi-tag like ("었다", ["EP", "EF"]), return [["었", "EP"], ["다", "EF"]].
    """
    if len(suffix_tags) == 1:
        return [[suffix_str, suffix_tags[0]]]
    # Multi-tag: first morpheme is first char, rest mapped to remaining tags
    # "었다" -> [["었","EP"],["다","EF"]]
    result = [[suffix_str[0], suffix_tags[0]]]
    remaining = suffix_str[1:]
    for i, tag in enumerate(suffix_tags[1:]):
        if i < len(remaining):
            result.append([remaining[i:] if i == len(suffix_tags) - 2 else remaining[i], tag])
    return result


def augment_irregular_conjugations(codebook: dict, content_dict_path: Path) -> dict:
    """Add irregular verb/adjective conjugation forms to the codebook.

    Reads stems from content_dict.txt, applies Korean irregular conjugation
    rules, and adds generated surface forms to the codebook.
    """
    # Read stems from content_dict
    stems = {}  # {stem_str: (pos, freq)}
    with open(content_dict_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            word, tag, freq_str = parts[0], parts[1], parts[2]
            if tag not in ("VA", "VV"):
                continue
            freq = int(freq_str)
            if word not in stems or freq > stems[word][1]:
                stems[word] = (tag, freq)

    added = 0

    def add_entry(surface, stem_form, stem_pos, suffix_str, suffix_tags, stem_freq):
        """Add a codebook entry or boost freq of existing matching analysis."""
        nonlocal added
        morphemes = [[stem_form, stem_pos]]
        morphemes.extend(_make_suffix_morphemes(suffix_str, suffix_tags))
        freq = max(stem_freq, 100)

        if surface not in codebook:
            codebook[surface] = [{"morphemes": morphemes, "freq": freq}]
            added += 1
        else:
            # Check if this exact analysis already exists
            new_key = tuple(tuple(m) for m in morphemes)
            found = False
            for a in codebook[surface]:
                existing_key = tuple(tuple(m) for m in a["morphemes"])
                if existing_key == new_key:
                    # Boost freq if our generated freq is higher
                    if freq > a["freq"]:
                        a["freq"] = freq
                    found = True
                    break
            if not found:
                codebook[surface].append({"morphemes": morphemes, "freq": freq})
                added += 1

    def is_bright_vowel(vowel_idx):
        """Check if vowel is 'bright' (양성모음: ㅏ, ㅗ, ㅘ)."""
        return vowel_idx in (V_A, V_O, V_WA)

    for stem, (pos, freq) in stems.items():
        if len(stem) < 1:
            continue
        last_ch = stem[-1]
        dec = decompose_hangul(last_ch)
        if dec is None:
            continue
        lead, vowel, tail = dec

        # === ㅂ불규칙 (VA only, stem ends in ㅂ) ===
        if pos == "VA" and tail == TAIL_BIEUP:
            # Remove ㅂ from last syllable
            stem_base = stem[:-1] + compose_hangul(lead, vowel, TAIL_NONE)
            stem_form = stem  # dictionary form of the stem

            for suffix_str, suffix_tags in SUFFIX_COMBOS:
                first_suffix_ch = suffix_str[0]
                if first_suffix_ch in ("어", "아"):
                    # ㅂ -> 우 + 어/아 -> 워/와
                    # Most ㅂ-irregular use 워 (ㅂ+어→워)
                    conj_char = "워"
                    rest = suffix_str[1:]
                    surface = stem_base + conj_char + rest
                    add_entry(surface, stem_form, pos, suffix_str, suffix_tags, freq)
                elif first_suffix_ch == "은":
                    # ㅂ -> 우 + ㄴ -> 운
                    surface = stem_base + "운"
                    if len(suffix_str) > 1:
                        surface += suffix_str[1:]
                    add_entry(surface, stem_form, pos, suffix_str, suffix_tags, freq)
                elif first_suffix_ch == "을":
                    # ㅂ -> 울
                    surface = stem_base + "울"
                    if len(suffix_str) > 1:
                        surface += suffix_str[1:]
                    add_entry(surface, stem_form, pos, suffix_str, suffix_tags, freq)
                elif first_suffix_ch == "으":
                    # ㅂ -> 우 + 으 contracts: 우면, 우니
                    rest = suffix_str[1:]
                    surface = stem_base + "우" + rest
                    add_entry(surface, stem_form, pos, suffix_str, suffix_tags, freq)

        # === ㄷ불규칙 (known irregular ㄷ stems) ===
        elif tail == TAIL_DIGEUT and stem in IRREG_DIGEUT_STEMS:
            # ㄷ -> ㄹ before vowel suffixes
            stem_with_rieul = stem[:-1] + compose_hangul(lead, vowel, TAIL_RIEUL)
            stem_form = stem

            for suffix_str, suffix_tags in SUFFIX_COMBOS:
                first_suffix_ch = suffix_str[0]
                if first_suffix_ch in ("어", "아"):
                    surface = stem_with_rieul + suffix_str
                    add_entry(surface, stem_form, pos, suffix_str, suffix_tags, freq)
                elif first_suffix_ch in ("은", "을"):
                    surface = stem_with_rieul + suffix_str
                    add_entry(surface, stem_form, pos, suffix_str, suffix_tags, freq)
                elif first_suffix_ch == "으":
                    # ㄹ tail + 으 -> ㄹ contracts: 걸으니 -> valid
                    surface = stem_with_rieul + suffix_str
                    add_entry(surface, stem_form, pos, suffix_str, suffix_tags, freq)

        # === 르불규칙 (stem ends in 르) ===
        elif len(stem) >= 2 and tail == TAIL_NONE and vowel == V_EU and lead == L_RIEUL:
            prev_ch = stem[-2]
            prev_dec = decompose_hangul(prev_ch)
            if prev_dec and prev_dec[2] == TAIL_NONE:
                # Add ㄹ to previous syllable's tail
                prev_with_rieul = compose_hangul(prev_dec[0], prev_dec[1], TAIL_RIEUL)
                stem_prefix = stem[:-2] + prev_with_rieul
                stem_form = stem

                # Determine 러 vs 라 based on previous vowel
                use_ra = is_bright_vowel(prev_dec[1])

                for suffix_str, suffix_tags in SUFFIX_COMBOS:
                    first_suffix_ch = suffix_str[0]
                    if first_suffix_ch in ("어", "아"):
                        conj_ch = "라" if use_ra else "러"
                        rest = suffix_str[1:]
                        surface = stem_prefix + conj_ch + rest
                        add_entry(surface, stem_form, pos, suffix_str, suffix_tags, freq)
                    elif first_suffix_ch == "었":
                        conj_ch = "랐" if use_ra else "렀"
                        rest = suffix_str[1:]
                        surface = stem_prefix + conj_ch + rest
                        add_entry(surface, stem_form, pos, suffix_str, suffix_tags, freq)
                    elif first_suffix_ch in ("은", "을", "으"):
                        # 르 + 은/을/으 -> regular: 흐르 + 는 etc.
                        # Actually 르 stems: 흐른 (흐르+ㄴ), treat as regular
                        surface = stem + suffix_str
                        add_entry(surface, stem_form, pos, suffix_str, suffix_tags, freq)

        # === ㅡ탈락 (stem vowel ㅡ, no tail) ===
        elif tail == TAIL_NONE and vowel == V_EU and lead != L_RIEUL:
            stem_form = stem
            # Determine 아 vs 어 based on previous syllable
            if len(stem) >= 2:
                prev_ch = stem[-2]
                prev_dec = decompose_hangul(prev_ch)
                use_a = prev_dec and is_bright_vowel(prev_dec[1])
            else:
                use_a = False

            for suffix_str, suffix_tags in SUFFIX_COMBOS:
                first_suffix_ch = suffix_str[0]
                if first_suffix_ch in ("어", "아"):
                    # ㅡ drops, lead consonant attaches to 아/어
                    attach_vowel = V_A if use_a else V_EO
                    new_ch = compose_hangul(lead, attach_vowel, TAIL_NONE)
                    rest = suffix_str[1:]
                    if len(stem) >= 2:
                        surface = stem[:-1] + new_ch + rest
                    else:
                        surface = new_ch + rest
                    add_entry(surface, stem_form, pos, suffix_str, suffix_tags, freq)
                elif first_suffix_ch == "었":
                    attach_vowel = V_A if use_a else V_EO
                    # 았/었 merges: lead + 아/어 + ㅆ tail
                    new_ch = compose_hangul(lead, attach_vowel, TAIL_NONE)
                    rest = suffix_str[1:]
                    if len(stem) >= 2:
                        surface = stem[:-1] + new_ch + suffix_str
                    else:
                        surface = new_ch + suffix_str
                    add_entry(surface, stem_form, pos, suffix_str, suffix_tags, freq)
                elif first_suffix_ch in ("은", "을", "으"):
                    surface = stem + suffix_str
                    add_entry(surface, stem_form, pos, suffix_str, suffix_tags, freq)

        # === ㅎ불규칙 (VA only, known stems) ===
        if pos == "VA" and tail == TAIL_HIEUT and stem in IRREG_HIEUT_STEMS:
            # ㅎ drops
            stem_no_hieut = stem[:-1] + compose_hangul(lead, vowel, TAIL_NONE)
            stem_form = stem

            for suffix_str, suffix_tags in SUFFIX_COMBOS:
                first_suffix_ch = suffix_str[0]
                if first_suffix_ch in ("어", "아"):
                    # ㅎ drops + 어/아 -> 애/에
                    # If vowel is ㅏ: 빨갛+어 -> 빨개 (ㅏ+ㅎ+어 -> 애)
                    # If vowel is ㅓ: 그렇+어 -> 그래... actually 그러+ㅎ+어 -> 그래
                    # ㅎ-irregular: vowel becomes ㅐ (if bright) or ㅐ (for 렇 type too)
                    new_vowel = V_AE
                    new_ch = compose_hangul(lead, new_vowel, TAIL_NONE)
                    rest = suffix_str[1:]
                    surface = stem[:-1] + new_ch + rest
                    add_entry(surface, stem_form, pos, suffix_str, suffix_tags, freq)
                elif first_suffix_ch == "었":
                    new_vowel = V_AE
                    # 빨개 + ㅆ -> not quite; 빨갛+었 -> 빨갰
                    new_ch = compose_hangul(lead, new_vowel, TAIL_NONE)
                    rest = suffix_str  # 었다 etc
                    surface = stem[:-1] + new_ch + rest
                    add_entry(surface, stem_form, pos, suffix_str, suffix_tags, freq)
                elif first_suffix_ch == "은":
                    # ㅎ + ㄴ -> ㄴ: 빨갛+은 -> 빨간
                    new_ch = compose_hangul(lead, vowel, 4)  # tail ㄴ=4
                    rest = suffix_str[1:]
                    surface = stem[:-1] + new_ch + rest
                    add_entry(surface, stem_form, pos, suffix_str, suffix_tags, freq)
                elif first_suffix_ch == "을":
                    # ㅎ + ㄹ -> ㄹ: 빨갛+을 -> 빨갈
                    new_ch = compose_hangul(lead, vowel, TAIL_RIEUL)
                    rest = suffix_str[1:]
                    surface = stem[:-1] + new_ch + rest
                    add_entry(surface, stem_form, pos, suffix_str, suffix_tags, freq)
                elif first_suffix_ch == "으":
                    # ㅎ drops before 으: 빨갛+으면 -> 빨가면?
                    # Actually ㅎ-irregular + 으면 -> ㅎ drops: 그렇+으면 -> 그러면
                    rest = suffix_str[1:]
                    surface = stem_no_hieut + rest
                    add_entry(surface, stem_form, pos, suffix_str, suffix_tags, freq)

    print(f"  Irregular conjugation augmentation: {added} new entries added")
    return codebook


def augment_contractions(codebook: dict) -> dict:
    """Add prefix entries derived from multi-char codebook entries.

    For each multi-char entry (e.g., 걸었다 → [걸,VV][었,EP][다,EF]):
    - Generate 1-char prefix: 걸 → [걸,VV][었,EP]  (if not in codebook)
    - Generate 2-char prefix: 걸었 → [걸,VV][었,EP]  (if not in codebook)

    Also adds hardcoded contractions for forms not derivable from data.
    """

    # Part A: derive 1-char and 2-char prefixes from multi-char entries
    derived = {}  # {surface: {morph_tuple: freq}}

    for surface, analyses in codebook.items():
        if len(surface) < 2:
            continue

        for a in analyses:
            morphs = a["morphemes"]
            if not isinstance(morphs[0], list):
                continue
            # First morpheme must be content POS
            if morphs[0][1] not in ("VV","VA","VX","VCP","VCN","NNG","NNP","NNB","NR","NP","MAG","MAJ","MM","IC","XR"):
                continue
            # Find prefix morphemes: all up to and including first EP
            prefix = []
            for m in morphs:
                prefix.append(tuple(m))
                if m[1] == "EP":
                    break
            if len(prefix) < 2 or prefix[-1][1] != "EP":
                continue

            key = tuple(prefix)
            # Generate 1-char prefix (from surface[0])
            p1 = surface[0]
            if '\uAC00' <= p1 <= '\uD7A3' and p1 not in codebook:
                if p1 not in derived:
                    derived[p1] = {}
                derived[p1][key] = derived[p1].get(key, 0) + a["freq"]
            # Generate 2-char prefix (from surface[:2]) for 3+ char entries
            if len(surface) >= 3:
                p2 = surface[:2]
                if p2 not in codebook:
                    if p2 not in derived:
                        derived[p2] = {}
                    derived[p2][key] = derived[p2].get(key, 0) + a["freq"]

    # Part B: hardcoded contraction table for entries not derivable from data
    # These are contracted syllables where stem + 었/았 merge into one syllable
    hardcoded = {
        "셨": [[["시", "EP"], ["었", "EP"]], 10000],   # honorific past
        "녔": [[["니", "VV"], ["었", "EP"]], 100],
        "렸": [[["리", "VV"], ["었", "EP"]], 100],
        "몄": [[["미", "VV"], ["었", "EP"]], 100],
        "텄": [[["터", "VV"], ["었", "EP"]], 100],
        "켰": [[["키", "VV"], ["었", "EP"]], 500],
        "빴": [[["빠", "VV"], ["았", "EP"]], 100],
        "꼈": [[["끼", "VV"], ["었", "EP"]], 100],
        "팠": [[["파", "VV"], ["았", "EP"]], 100],
        "잤": [[["자", "VV"], ["았", "EP"]], 500],
        "뤘": [[["루", "VV"], ["었", "EP"]], 50],
        "챘": [[["채", "VV"], ["었", "EP"]], 100],
        "맸": [[["매", "VV"], ["었", "EP"]], 100],
        "댔": [[["대", "VV"], ["었", "EP"]], 100],
        "랐": [[["라", "VV"], ["았", "EP"]], 100],
        "뺐": [[["빼", "VV"], ["었", "EP"]], 100],
        "샜": [[["새", "VV"], ["었", "EP"]], 50],
    }

    # Merge into codebook
    added = 0
    for prefix_surface, morph_freqs in derived.items():
        if prefix_surface not in codebook:
            entries = []
            for morph_tuple, freq in sorted(morph_freqs.items(), key=lambda x: -x[1]):
                if freq >= 10:  # minimum threshold
                    entries.append({"morphemes": [list(m) for m in morph_tuple], "freq": freq})
            if entries:
                codebook[prefix_surface] = entries
                added += 1

    for char, (morphs, freq) in hardcoded.items():
        if char not in codebook:
            codebook[char] = [{"morphemes": morphs, "freq": freq}]
            added += 1
        # If it exists but doesn't have this analysis, add it
        else:
            existing_keys = set()
            for a in codebook[char]:
                existing_keys.add(tuple(tuple(m) for m in a["morphemes"]))
            new_key = tuple(tuple(m) for m in morphs)
            if new_key not in existing_keys:
                codebook[char].append({"morphemes": morphs, "freq": freq})

    print(f"  Contraction augmentation: {added} new entries added")
    return codebook


def augment_jamo_suffixes(codebook: dict) -> dict:
    """Add standalone jamo-initial suffix patterns to the codebook.

    Korean morpheme analysis often requires splitting a syllable's tail consonant
    (받침) from the stem. For example, "온다" = 오/VV + ㄴ다/EF. These jamo-initial
    suffix patterns must exist in the codebook for correct analysis.
    """
    JAMO_SUFFIXES = {
        "ㄴ": [{"morphemes": [["ㄴ", "ETM"]], "freq": 1000000}],
        "ㄹ": [{"morphemes": [["ㄹ", "ETM"]], "freq": 500000}],
        "ㅂ니다": [{"morphemes": [["ㅂ니다", "EF"]], "freq": 100000}],
        "ㅂ니까": [{"morphemes": [["ㅂ니까", "EF"]], "freq": 50000}],
        "ㅂ시다": [{"morphemes": [["ㅂ시다", "EF"]], "freq": 10000}],
        "ㄴ다": [{"morphemes": [["ㄴ다", "EF"]], "freq": 200000}],
        "ㄴ다고": [{"morphemes": [["ㄴ다고", "EC"]], "freq": 50000}],
        "ㄴ다면": [{"morphemes": [["ㄴ다면", "EC"]], "freq": 20000}],
        "ㄴ다며": [{"morphemes": [["ㄴ다며", "EC"]], "freq": 10000}],
        "ㄴ다는": [{"morphemes": [["ㄴ다는", "ETM"]], "freq": 30000}],
        "ㄴ데": [{"morphemes": [["ㄴ데", "EC"]], "freq": 30000}],
        "ㄴ지": [{"morphemes": [["ㄴ지", "EC"]], "freq": 20000}],
        "ㄴ가": [{"morphemes": [["ㄴ가", "EF"]], "freq": 30000}],
        "ㄹ까": [{"morphemes": [["ㄹ까", "EF"]], "freq": 20000}],
        "ㄹ지": [{"morphemes": [["ㄹ지", "EC"]], "freq": 10000}],
        "ㄹ수록": [{"morphemes": [["ㄹ수록", "EC"]], "freq": 10000}],
        "ㄹ려고": [{"morphemes": [["ㄹ려고", "EC"]], "freq": 5000}],
        "ㅁ": [{"morphemes": [["ㅁ", "ETN"]], "freq": 50000}],
    }

    added = 0
    for surface, analyses in JAMO_SUFFIXES.items():
        if surface not in codebook:
            codebook[surface] = analyses
            added += len(analyses)
        else:
            # Add any missing analyses
            for a in analyses:
                new_key = tuple(tuple(m) for m in a["morphemes"])
                found = False
                for existing in codebook[surface]:
                    existing_key = tuple(tuple(m) for m in existing["morphemes"])
                    if existing_key == new_key:
                        # Boost freq if ours is higher
                        if a["freq"] > existing["freq"]:
                            existing["freq"] = a["freq"]
                        found = True
                        break
                if not found:
                    codebook[surface].append(a)
                    added += 1

    print(f"  Jamo suffix augmentation: {added} new entries added")
    return codebook


def build_suffix_codebook(codebook_path: Path, min_freq: int = MIN_SUFFIX_FREQ) -> tuple[bytes, int]:
    """Build Section 7: Suffix Codebook.

    Format:
      num_entries [u32]
      For each entry (sorted by surface UTF-8):
        surface_len [u16] + surface_bytes
        num_analyses [u16]
        For each analysis:
          freq [u32] + num_morphemes [u8]
          For each morpheme: form_len [u16] + form_bytes + pos_byte [u8]
    """
    with open(codebook_path, "r", encoding="utf-8") as f:
        codebook = json.load(f)

    # Augment with contraction entries
    codebook = augment_contractions(codebook)

    # Augment with irregular verb/adjective conjugation forms
    content_dict_path = DATA_DIR / "content_dict.txt"
    codebook = augment_irregular_conjugations(codebook, content_dict_path)

    # Augment with jamo-initial suffix patterns (ㄴ, ㄹ, ㅂ니다, etc.)
    codebook = augment_jamo_suffixes(codebook)

    # Count total entries before filtering
    total_before = sum(len(analyses) for analyses in codebook.values())

    # Filter analyses by freq >= min_freq, drop surfaces with no remaining analyses.
    # Exempt entries whose morphemes contain compatibility jamo (ㄱ-ㅎ, U+3131-U+314E):
    # these are high-value grammatical endings that should be kept at a lower threshold.
    JAMO_MIN_FREQ = 5

    def _has_jamo_morpheme(analysis):
        """Check if any morpheme form contains a jamo character.

        Covers both compatibility jamo (ㄱ-ㅎ, U+3131-U+314E) and
        Hangul Jamo trailing consonants (ᆨ-ᇂ, U+11A8-U+11C2).
        """
        for form, _tag in analysis["morphemes"]:
            if any(('\u3131' <= ch <= '\u314e') or ('\u11a8' <= ch <= '\u11c2') for ch in form):
                return True
        return False

    filtered = {}
    jamo_exempted = 0
    for surface, analyses in codebook.items():
        kept = []
        for a in analyses:
            if a["freq"] >= min_freq:
                kept.append(a)
            elif a["freq"] >= JAMO_MIN_FREQ and _has_jamo_morpheme(a):
                kept.append(a)
                jamo_exempted += 1
        if kept:
            filtered[surface] = kept

    total_after = sum(len(analyses) for analyses in filtered.values())
    print(f"  Suffix filter: {total_before:,} → {total_after:,} entries (freq >= {min_freq})")
    print(f"    Jamo-exempted entries: {jamo_exempted} (freq >= {JAMO_MIN_FREQ})")

    # Sort entries by surface UTF-8
    sorted_surfaces = sorted(filtered.keys(), key=lambda s: s.encode("utf-8"))

    max_suffix_freq = 0
    buf = bytearray()
    buf.extend(struct.pack("<I", len(sorted_surfaces)))

    for surface in sorted_surfaces:
        analyses = filtered[surface]
        surface_bytes = surface.encode("utf-8")
        buf.extend(struct.pack("<H", len(surface_bytes)))
        buf.extend(surface_bytes)
        buf.extend(struct.pack("<H", len(analyses)))

        for analysis in analyses:
            freq = analysis["freq"]
            if freq > max_suffix_freq:
                max_suffix_freq = freq
            morphemes = analysis["morphemes"]
            buf.extend(struct.pack("<I", freq))
            buf.extend(struct.pack("B", len(morphemes)))
            for form, tag in morphemes:
                form_bytes = form.encode("utf-8")
                buf.extend(struct.pack("<H", len(form_bytes)))
                buf.extend(form_bytes)
                buf.extend(struct.pack("B", pos_byte(tag)))

    return bytes(buf), max_suffix_freq


def build_trigram_costs(costs_path: Path) -> bytes:
    """Build Section 8: Sparse trigram cost table with u8 quantization.

    Format v2 (sparse bitmap + quantized values):
      num_pos [u32] = 42
      default_cost [f32]
      tg_min [f32], tg_max [f32]  — quantization range for trigrams
      bg_min [f32], bg_max [f32]  — quantization range for bigrams
      tg_bitmap [ceil(42^3/8) bytes] — 1 bit per trigram entry (1=present)
      tg_values [N_tg u8]         — quantized costs for present entries
      bg_bitmap [ceil(42^2/8) bytes]
      bg_values [N_bg u8]
    """
    with open(costs_path, "r", encoding="utf-8") as f:
        costs = json.load(f)

    default_cost = costs.get("default_cost", 15.0)
    trigrams = costs.get("trigram", {})
    bigrams = costs.get("bigram", {})

    tg_count = NUM_POS * NUM_POS * NUM_POS
    bg_count = NUM_POS * NUM_POS

    # Build dense arrays first
    tg_data = [0.0] * tg_count
    bg_data = [0.0] * bg_count

    for key, cost in bigrams.items():
        parts = key.split(",")
        if len(parts) != 2: continue
        i, j = POS_TO_BYTE.get(parts[0].strip()), POS_TO_BYTE.get(parts[1].strip())
        if i is not None and j is not None:
            bg_data[i * NUM_POS + j] = cost

    for key, cost in trigrams.items():
        parts = key.split(",")
        if len(parts) != 3: continue
        i, j, k = POS_TO_BYTE.get(parts[0].strip()), POS_TO_BYTE.get(parts[1].strip()), POS_TO_BYTE.get(parts[2].strip())
        if i is not None and j is not None and k is not None:
            tg_data[i * NUM_POS * NUM_POS + j * NUM_POS + k] = cost

    # Compute min/max for quantization (non-zero values only)
    tg_nz = [v for v in tg_data if v != 0.0]
    bg_nz = [v for v in bg_data if v != 0.0]
    tg_min = min(tg_nz) if tg_nz else 0.0
    tg_max = max(tg_nz) if tg_nz else 1.0
    bg_min = min(bg_nz) if bg_nz else 0.0
    bg_max = max(bg_nz) if bg_nz else 1.0

    def quantize_u8(val, vmin, vmax):
        if vmax == vmin: return 128
        return max(0, min(255, int((val - vmin) / (vmax - vmin) * 255 + 0.5)))

    # Build bitmaps and quantized values
    def build_sparse(data, count, vmin, vmax):
        bitmap = bytearray((count + 7) // 8)
        values = bytearray()
        for idx, v in enumerate(data):
            if v != 0.0:
                bitmap[idx // 8] |= (1 << (idx % 8))
                values.append(quantize_u8(v, vmin, vmax))
        return bytes(bitmap), bytes(values)

    tg_bitmap, tg_values = build_sparse(tg_data, tg_count, tg_min, tg_max)
    bg_bitmap, bg_values = build_sparse(bg_data, bg_count, bg_min, bg_max)

    buf = bytearray()
    buf.extend(struct.pack("<I", NUM_POS))
    buf.extend(struct.pack("<f", default_cost))
    buf.extend(struct.pack("<f", tg_min))
    buf.extend(struct.pack("<f", tg_max))
    buf.extend(struct.pack("<f", bg_min))
    buf.extend(struct.pack("<f", bg_max))
    buf.extend(tg_bitmap)
    buf.extend(tg_values)
    buf.extend(bg_bitmap)
    buf.extend(bg_values)

    n_tg = sum(1 for v in tg_data if v != 0.0)
    n_bg = sum(1 for v in bg_data if v != 0.0)
    print(f"  Sparse trigram: {n_tg} nonzero/{tg_count} total, {n_bg} nonzero/{bg_count} bigram")

    return bytes(buf)


def build_word_frequencies(max_freq: int, max_suffix_freq: int) -> bytes:
    """Build Section 9: Word frequency metadata."""
    buf = bytearray()
    buf.extend(struct.pack("<I", max_freq))
    buf.extend(struct.pack("<I", max_suffix_freq))
    return bytes(buf)


def build_ambiguity_table(content_dict_path: Path, max_freq: int) -> bytes:
    """Build Section 11: Word ambiguity table.

    For words that have multiple POS tags in NIKL data, store alternative POS
    entries so the lattice can consider them alongside the primary FST entry.

    Format:
      num_entries [u32]
      For each entry:
        word_len [u16]
        word_utf8 [word_len bytes]
        num_alts [u8]
        For each alt:
          pos_byte [u8]
          quantized_freq [u16]  (same scale as content dict FST)
    """
    import math
    from collections import defaultdict, Counter

    NIKL_DIR_LOCAL = Path.home() / "Downloads" / "NIKL_MP(v1.1)"
    POS_SET_LOCAL = set(POS_TAGS)
    NIKL_MAP_LOCAL = {'MMD':'MM','MMN':'MM','MMA':'MM','NA':'NNG','NAP':'NNG','NF':'NNG','NV':'VV'}

    def np_local(t):
        if t in POS_SET_LOCAL: return t
        if t in NIKL_MAP_LOCAL: return NIKL_MAP_LOCAL[t]
        b = t.split('-')[0]
        return b if b in POS_SET_LOCAL else 'SW'

    # Count word→POS from NIKL
    word_pos = defaultdict(Counter)
    for fname in ["NXMP1902008040.json", "SXMP1902008031.json"]:
        path = NIKL_DIR_LOCAL / fname
        if not path.exists(): continue
        with open(path) as f:
            data = json.load(f)
        for doc in data["document"]:
            if doc is None: continue
            for sent in (doc.get("sentence") or []):
                for m in (sent.get("morpheme") or []):
                    form = m.get("form", "").strip()
                    label = np_local(m.get("label", ""))
                    if form and label:
                        word_pos[form][label] += 1

    # Load primary POS from content dict
    primary_pos = {}
    with open(content_dict_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                primary_pos[parts[0]] = parts[1]

    # Build ambiguity entries: words where NIKL has significant alternative POS
    entries = []
    for word, pos_counts in word_pos.items():
        if word not in primary_pos: continue
        total = sum(pos_counts.values())
        if total < 1000: continue  # Only very high-frequency words

        prim = primary_pos[word]
        alts = []
        for pos, count in pos_counts.most_common(2):  # Top 2 only
            if pos == prim: continue
            pct = count / total
            if pct < 0.25: continue  # At least 25% (very strong alternative)
            # Scale: make alternative slightly more expensive than primary
            # Use 10% of proportional frequency — let trigram decide
            scaled_freq = int(pct * max_freq * 0.1)
            qfreq = max(1, min(65535, int(math.log(max(scaled_freq, 1)) / math.log(max_freq) * 65535)))
            alts.append((POS_TO_BYTE[pos], qfreq))

        if alts:
            entries.append((word, alts))

    # Encode
    buf = bytearray()
    buf.extend(struct.pack("<I", len(entries)))
    for word, alts in entries:
        word_bytes = word.encode("utf-8")
        buf.extend(struct.pack("<H", len(word_bytes)))
        buf.extend(word_bytes)
        buf.extend(struct.pack("B", len(alts)))
        for pos_b, qfreq in alts:
            buf.extend(struct.pack("B", pos_b))
            buf.extend(struct.pack("<H", qfreq))

    print(f"  Ambiguity table: {len(entries)} words, {sum(len(a) for _, a in entries)} alt entries")
    return bytes(buf)


def build_analyzer_params() -> bytes:
    """Build Section 10: Analyzer parameters."""
    buf = bytearray()
    buf.extend(struct.pack("<f", 0.25))  # morpheme_penalty (tuned)
    buf.extend(struct.pack("<f", 4.0))   # oov_penalty (tuned)
    buf.extend(struct.pack("<f", 1.5))   # length_bonus (tuned)
    buf.extend(struct.pack("<f", 3.5))   # single_char_content_penalty
    return bytes(buf)


def main():
    print("Building GMDL v3 codebook model...")
    print(f"  Input:  {DATA_DIR}")
    print(f"  Output: {OUT_PATH}")
    print()

    # Section 6: Content dict (Dict v2 FST format)
    print("Building content dict (Dict v2 FST)...")
    dict_data, max_freq = build_content_dict_fst(DATA_DIR / "content_dict.txt")
    print(f"  Content dict: {len(dict_data):,} bytes, max_freq={max_freq}")

    # Section 7: Suffix codebook
    print("Building suffix codebook...")
    codebook_data, max_suffix_freq = build_suffix_codebook(DATA_DIR / "suffix_codebook.json")
    print(f"  Suffix codebook: {len(codebook_data):,} bytes, max_suffix_freq={max_suffix_freq}")

    # Section 8: Trigram cost table
    print("Building trigram cost table...")
    trigram_data = build_trigram_costs(DATA_DIR / "trigram_costs.json")
    print(f"  Trigram costs: {len(trigram_data):,} bytes")

    # Section 9: Word frequencies
    freq_data = build_word_frequencies(max_freq, max_suffix_freq)

    # Section 10: Analyzer parameters
    params_data = build_analyzer_params()

    # Section 11: Word ambiguity table (disabled)
    ambig_data = b'\x00\x00\x00\x00'

    # Section 12: Word-bigram cost bonuses
    wbigram_path = DATA_DIR / "word_bigrams.bin"
    if wbigram_path.exists():
        wbigram_data = wbigram_path.read_bytes()
        n_entries = int.from_bytes(wbigram_data[:4], 'little')
        print(f"  Word bigrams: {n_entries} entries, {len(wbigram_data)} bytes")
    else:
        wbigram_data = b'\x00\x00\x00\x00'
        print(f"  Word bigrams: none")

    # Assemble GMDL v3
    print()
    print("Assembling GMDL v3...")
    buf = bytearray()
    buf.extend(b"GMDL")
    buf.extend(struct.pack("<I", 3))  # version 3

    write_section(buf, 6, dict_data)
    write_section(buf, 7, codebook_data)
    write_section(buf, 8, trigram_data)
    write_section(buf, 9, freq_data)
    write_section(buf, 10, params_data)
    write_section(buf, 11, ambig_data)
    write_section(buf, 12, wbigram_data)

    # Write output
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        f.write(buf)

    total = len(buf)
    print()
    print("=== Section Size Report ===")
    print(f"  Header:              8 bytes")
    print(f"  Section 6 (dict):    {len(dict_data):>10,} bytes  ({len(dict_data)/total*100:.1f}%)")
    print(f"  Section 7 (suffix):  {len(codebook_data):>10,} bytes  ({len(codebook_data)/total*100:.1f}%)")
    print(f"  Section 8 (trigram): {len(trigram_data):>10,} bytes  ({len(trigram_data)/total*100:.1f}%)")
    print(f"  Section 9 (freq):    {len(freq_data):>10,} bytes  ({len(freq_data)/total*100:.1f}%)")
    print(f"  Section 10 (params): {len(params_data):>10,} bytes  ({len(params_data)/total*100:.1f}%)")
    print(f"  Section headers:     {5 * 5:>10,} bytes")
    print(f"  ---")
    print(f"  Total:               {total:>10,} bytes  ({total/1024/1024:.2f} MB)")
    print()
    print(f"Output written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
