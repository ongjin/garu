"""Prepare syllable-level BIO+POS training data from NIKL MP.

Each sentence → sequence of (syllable, BIO-POS label).
Morpheme boundaries are mapped to syllable boundaries within each eojeol.

Example:
  어절: "먹었다"  gold: [먹/VV, 었/EP, 다/EF]
  → 먹/B-VV  었/B-EP  다/B-EF

  어절: "학교에서"  gold: [학교/NNG, 에서/JKB]
  → 학/B-NNG  교/I-NNG  에/B-JKB  서/I-JKB

Output: JSON lines with {syllables: [...], labels: [...]}
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

NIKL_DIR = Path.home() / "Downloads" / "NIKL_MP(v1.1)"
OUT_DIR = Path(__file__).parent

POS_TAGS = [
    "NNG", "NNP", "NNB", "NR", "NP",
    "VV", "VA", "VX", "VCP", "VCN",
    "MAG", "MAJ", "MM", "IC",
    "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC",
    "EP", "EF", "EC", "ETN", "ETM",
    "XPN", "XSN", "XSV", "XSA", "XR",
    "SF", "SP", "SS", "SE", "SO", "SW", "SH", "SL", "SN",
]
POS_SET = set(POS_TAGS)
NIKL_MAP = {'MMD': 'MM', 'MMN': 'MM', 'MMA': 'MM', 'NA': 'NNG', 'NAP': 'NNG', 'NF': 'NNG', 'NV': 'VV'}


def normalize_pos(t):
    if t in POS_SET: return t
    if t in NIKL_MAP: return NIKL_MAP[t]
    b = t.split('-')[0]
    return b if b in POS_SET else 'SW'


def morphemes_to_syllable_labels(eojeol_text, morphemes):
    """Map morpheme sequence to syllable-level BIO labels.

    Strategy: greedily align morpheme forms to eojeol syllables.
    If forms don't perfectly concatenate to surface (irregular conjugation etc.),
    fall back to character-level alignment.
    """
    syllables = list(eojeol_text)
    labels = ['O'] * len(syllables)

    # Try greedy alignment: concatenate morpheme forms and match to surface
    concat = ''.join(form for form, _ in morphemes)

    if concat == eojeol_text:
        # Perfect alignment — map directly
        pos_idx = 0
        for form, pos in morphemes:
            for j, ch in enumerate(form):
                if pos_idx < len(syllables):
                    tag = f"B-{pos}" if j == 0 else f"I-{pos}"
                    labels[pos_idx] = tag
                    pos_idx += 1
    else:
        # Imperfect alignment (irregular conjugation, contraction)
        # Use simple heuristic: spread morphemes proportionally
        total_chars = sum(len(form) for form, _ in morphemes)
        if total_chars == 0:
            return syllables, labels

        pos_idx = 0
        for form, pos in morphemes:
            # Assign proportional syllables
            n_syl = max(1, round(len(form) / total_chars * len(syllables)))
            if pos_idx >= len(syllables):
                break
            for j in range(min(n_syl, len(syllables) - pos_idx)):
                tag = f"B-{pos}" if j == 0 else f"I-{pos}"
                labels[pos_idx] = tag
                pos_idx += 1

        # Fill remaining with last POS
        if pos_idx < len(syllables) and morphemes:
            last_pos = morphemes[-1][1]
            for j in range(pos_idx, len(syllables)):
                labels[j] = f"I-{last_pos}"

    return syllables, labels


def main():
    all_sentences = []

    for fname in ["NXMP1902008040.json", "SXMP1902008031.json"]:
        path = NIKL_DIR / fname
        if not path.exists():
            print(f"Warning: {path} not found")
            continue
        data = json.load(open(path))

        for doc in data["document"]:
            if not doc:
                continue
            for sent in (doc.get("sentence") or []):
                text = sent.get("form", "")
                if not text or len(text) < 3 or len(text) > 150:
                    continue

                words = {w["id"]: w["form"] for w in (sent.get("word") or [])}
                morphemes = sent.get("morpheme") or []

                # Group morphemes by word_id
                word_morphs = defaultdict(list)
                for m in morphemes:
                    wid = m.get("word_id")
                    form = m.get("form", "").strip()
                    label = normalize_pos(m.get("label", ""))
                    if wid and form and label:
                        word_morphs[wid].append((form, label))

                # Build syllable sequence with labels
                sent_syllables = []
                sent_labels = []

                for wid in sorted(words.keys()):
                    wform = words[wid]
                    morphs = word_morphs.get(wid, [])

                    if not morphs or not wform.strip():
                        continue

                    syls, labs = morphemes_to_syllable_labels(wform, morphs)
                    # Add space token between eojeols
                    if sent_syllables:
                        sent_syllables.append(' ')
                        sent_labels.append('O')
                    sent_syllables.extend(syls)
                    sent_labels.extend(labs)

                if sent_syllables and len(sent_syllables) <= 200:
                    all_sentences.append({
                        "text": text,
                        "syllables": sent_syllables,
                        "labels": sent_labels,
                    })

    print(f"Total sentences: {len(all_sentences)}")

    # Build label vocabulary
    label_counter = Counter()
    for s in all_sentences:
        label_counter.update(s["labels"])
    labels_sorted = sorted(label_counter.keys())
    print(f"Label vocabulary: {len(labels_sorted)} labels")
    print(f"Top labels: {label_counter.most_common(15)}")

    # Build syllable vocabulary
    syl_counter = Counter()
    for s in all_sentences:
        syl_counter.update(s["syllables"])
    print(f"Unique syllables: {len(syl_counter)}")

    # Split train/val/test
    random.seed(42)
    random.shuffle(all_sentences)
    n = len(all_sentences)
    train = all_sentences[:int(n * 0.8)]
    val = all_sentences[int(n * 0.8):int(n * 0.9)]
    test = all_sentences[int(n * 0.9):]

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Save
    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = OUT_DIR / f"{name}.jsonl"
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"  {path}: {len(data)} sentences")

    # Save vocabularies
    syl_vocab = ['<PAD>', '<UNK>'] + [s for s, _ in syl_counter.most_common(3000)]
    label_vocab = ['O'] + [l for l in labels_sorted if l != 'O']
    vocab = {"syllables": syl_vocab, "labels": label_vocab}
    with open(OUT_DIR / "vocab.json", 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"  Syllable vocab: {len(syl_vocab)}, Label vocab: {len(label_vocab)}")


if __name__ == "__main__":
    main()
