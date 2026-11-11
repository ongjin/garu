"""Instagram sentence supervision for the first-stage sentence head (research history #83).

    python3 training/boundaries/sns_sentences.py prepare --out SNS_LAB
    python3 training/boundaries/sns_sentences.py train --base LARGE19 --sns SNS_LAB --out OUT
    python3 training/boundaries/sns_sentences.py apply --npz OUT/sentences.npz --ts js/src/boundary-data.ts

NIKL online posting corpus 2025 Instagram paragraphs carry sentence records; blog
posts do not. Document split: sha256(id)[:8] mod 10, 0-1 final test, 2-3 dev,
4-9 train. Training paragraphs whose sentence keys appear in dev/test are dropped.

The labels are punctuation-driven: 96% of boundaries follow terminal punctuation,
and unpunctuated sentence ends are mostly left unsplit. Training therefore keeps
only positions after terminal punctuation (optionally followed by laughter,
emoticons or closing brackets) and only the punctuated variant. Every other
Instagram position has zero loss weight, so unpunctuated decisions still come from
the 2021 news/spoken data. LARGE19 is the adopted 2021 sentence-head pool (see
README). The objective, features and int8 quantization match boundary.py train.
"""
import argparse
import base64
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boundary import FEATURES, hash_features, leak_key, metrics, pack_parts, source_sha, targets  # noqa: E402

CORPUS = 'nikl_online_2025/NIKL_Online Posting Materials Corpus 2025'
PUBLISHER = '인스타그램'
MAX_PARAGRAPH_CODEPOINTS = 400
DIMENSIONS = 1 << 19
TERMINAL = set('.!?…')
ORNAMENT_SYMBOLS = '~^;*♡♥★☆♪♬)]}"\'”’」』>'


def split_of(doc_id):
    bucket = int(hashlib.sha256(doc_id.encode()).hexdigest()[:8], 16) % 10
    return 'test' if bucket < 2 else 'dev' if bucket < 4 else 'train'


def is_ornament(char):
    value = ord(char)
    if 0x1100 <= value <= 0x11FF or 0x3131 <= value <= 0x318E or char in ORNAMENT_SYMBOLS:
        return True
    return value >= 0x2600 and not 0xAC00 <= value <= 0xD7A3 and not char.isalnum()


def region(text, boundary):
    """P: after terminal punctuation (+ornaments), O: after ornaments only, C: otherwise."""
    i = boundary
    while i > 0 and is_ornament(text[i - 1]):
        i -= 1
    if i > 0 and text[i - 1] in TERMINAL:
        return 'P'
    return 'O' if i < boundary else 'C'


def align(form, sentences):
    cursor, parts = 0, []
    for sentence in sentences:
        part = (sentence.get('form') or '').strip()
        at = form.find(part, cursor)
        if not part or at < 0 or form[cursor:at].strip():
            return None
        parts.append(part)
        cursor = at + len(part)
    return None if form[cursor:].strip() else parts


def prepare(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    rows = {'train': [], 'dev': [], 'test': []}
    stats = {'align_fail': 0, 'too_long': 0}
    files = sorted((Path(args.data_root) / CORPUS).glob('*.json'))
    for path in files:
        data = json.loads(path.read_text())
        for doc in data['document']:
            if doc['metadata'].get('publisher') != PUBLISHER:
                continue
            split = split_of(doc['id'])
            for para in doc.get('paragraph') or []:
                if not para.get('sentence'):
                    continue
                parts = align(para['form'], para['sentence'])
                if parts is None:
                    stats['align_fail'] += 1
                    continue
                if len(''.join(parts)) > MAX_PARAGRAPH_CODEPOINTS:
                    stats['too_long'] += 1
                    continue
                base = pack_parts(parts)
                for stripped in [False, True]:
                    row = pack_parts(parts, strip_terminal=stripped)
                    if stripped and row['text'] == base['text']:
                        continue
                    if len(row['text']) < 2:
                        continue
                    row.update({'source': 'SNS', 'document_id': doc['id'], 'paragraph_id': para['id'],
                                'sentence_eligible': True, 'part_count': len(parts),
                                'keys': [leak_key(part) for part in parts]})
                    rows[split].append(row)
    held = {key for split in ['dev', 'test'] for row in rows[split] for key in row['keys']}
    before = len(rows['train'])
    rows['train'] = [row for row in rows['train'] if not any(key in held for key in row['keys'])]
    stats['train_dropped_overlap'] = before - len(rows['train'])
    for split, records in rows.items():
        with (out / (split + '.jsonl')).open('w') as stream:
            for row in records:
                row.pop('keys')
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stats[split] = {'rows': len(records), 'boundaries': sum(len(r['sentences']) for r in records),
                        'sha256': source_sha(out / (split + '.jsonl'))}
    stats.update({'source_files': len(files), 'split': 'sha256(document id)[:8] mod 10: test0-1 dev2-3 train4-9',
                  'publisher': PUBLISHER, 'source_text_redistributed': False})
    (out / 'manifest.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(stats, ensure_ascii=False))


def load(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream]


def train(args):
    base, sns = Path(args.base), Path(args.sns)
    rows = {'train': load(base / 'train.jsonl'), 'dev': load(base / 'dev.jsonl')}
    rows['train'] += [row for row in load(sns / 'train.jsonl') if row['variant'] == 'punctuated']
    rows['dev'] += load(sns / 'dev.jsonl')
    matrices, spans = {}, {}
    for split in rows:
        matrices[split], spans[split] = hash_features(rows[split], DIMENSIONS, FEATURES)
    labels = targets(rows['train'], spans['train'], 'sentences')
    weights = np.zeros(len(labels))
    for row, (start, end) in zip(rows['train'], spans['train']):
        if not row['sentence_eligible']:
            continue
        if row['source'] == 'SX':
            weights[start:end] = args.speech_weight
        elif row['source'] == 'SNS':
            text = row['text']
            for k in range(1, len(text)):
                if region(text, k) == 'P':
                    weights[start + k - 1] = args.sns_weight
        else:
            weights[start:end] = 1.0
    mask = weights > 0
    matrix, labels, weights = matrices['train'][mask], labels[mask], weights[mask]
    total = weights.sum()
    initial = np.zeros(DIMENSIONS)
    initial[0] = np.log((labels.sum() + 1) / (len(labels) - labels.sum() + 1))
    started = time.monotonic()

    def objective(w):
        logits = matrix @ w
        loss = np.dot(weights, np.logaddexp(0, logits) - labels * logits) / total + args.l2 * np.dot(w[1:], w[1:]) / 2
        gradient = matrix.T @ ((expit(logits) - labels) * weights) / total
        gradient[1:] += args.l2 * w[1:]
        return loss, gradient

    result = minimize(objective, initial, method='L-BFGS-B', jac=True,
                      options={'maxiter': args.iterations, 'ftol': 1e-8, 'gtol': 1e-6, 'maxcor': 10})
    scale = float(np.max(np.abs(result.x)) / 127)
    quantized = np.clip(np.round(result.x / scale), -127, 127).astype(np.int8)
    scores = (matrices['dev'] @ quantized.astype(np.float64)) * scale

    def dev_metric(threshold, source, variant=None):
        for row in rows['dev']:
            row['_selected'] = (row['source'] == source and row['sentence_eligible']
                                and (variant is None or row['variant'] == variant))
        return metrics(rows['dev'], spans['dev'], scores, 'sentences', float(threshold), '_selected')

    sources = ['NX', 'SX', 'SNS']
    trials = [(sum(dev_metric(t, s)['f1'] for s in sources) / len(sources), float(t))
              for t in np.arange(-4.0, 4.01, 0.25)]
    selected = max(trials)[1]
    threshold = selected if args.threshold is None else args.threshold
    report = {'threshold': threshold, 'dev_macro_selected_threshold': selected, 'scale': scale,
              'iterations': int(result.nit), 'converged': bool(result.success),
              'seconds': time.monotonic() - started, 'sns_weight': args.sns_weight,
              'speech_weight': args.speech_weight, 'l2': args.l2,
              'training_positions': int(len(labels)), 'training_positive': int(labels.sum()),
              'weights_sha256': hashlib.sha256(quantized.tobytes()).hexdigest(),
              'datasets': {f'{name}/{split}': source_sha(folder / (split + '.jsonl'))
                           for name, folder in [('base', base), ('sns', sns)] for split in ['train', 'dev']},
              'dev': {}}
    for source in sources:
        for variant in [None, 'punctuated', 'no_terminal']:
            m = dev_metric(threshold, source, variant)
            report['dev'][f'{source}/{variant or "all"}'] = {k: m[k] for k in ['tp', 'fp', 'fn', 'f1']}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'sentences.npz', weights=quantized, scale=scale, threshold=threshold)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


def apply_head(args):
    """Replace only the sentence head (weights, scale, threshold) in a generated boundary-data.ts."""
    path = Path(args.ts)
    source = path.read_text()
    head = np.load(args.npz)
    weights = re.search(r'BOUNDARY_WEIGHTS: string = "([^"]+)"', source)
    raw = bytearray(base64.b64decode(weights.group(1), validate=True))
    metadata_match = re.search(r'export const BOUNDARY_MODEL = (.*) as const;', source)
    metadata = json.loads(metadata_match.group(1))
    sentences = metadata['heads']['sentences']
    start = sentences['order'] * metadata['dimensions']
    raw[start:start + metadata['dimensions']] = head['weights'].astype(np.int8).tobytes()
    sentences.update(scale=float(head['scale']), threshold=float(head['threshold']))
    source = source.replace(weights.group(1), base64.b64encode(bytes(raw)).decode())
    source = source.replace(metadata_match.group(1), json.dumps(metadata, separators=(',', ':')))
    path.write_text(source)
    print(json.dumps({'weights_sha256': hashlib.sha256(bytes(raw)).hexdigest(),
                      'sentence_head_sha256': hashlib.sha256(head['weights'].tobytes()).hexdigest()}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'train', 'apply'])
    parser.add_argument('--out')
    parser.add_argument('--npz')
    parser.add_argument('--ts')
    parser.add_argument('--data-root', default=str(Path.home() / 'workspace/data'))
    parser.add_argument('--base')
    parser.add_argument('--sns')
    parser.add_argument('--sns-weight', type=float, default=3.0)
    parser.add_argument('--speech-weight', type=float, default=10.0)
    parser.add_argument('--l2', type=float, default=1e-5)
    parser.add_argument('--iterations', type=int, default=160)
    parser.add_argument('--threshold', type=float, default=-0.75,
                        help='fixed decision threshold; -0.75 keeps the 2021 head value (#83)')
    parser.add_argument('--dev-threshold', action='store_true', help='use the dev macro-selected threshold')
    args = parser.parse_args()
    if args.dev_threshold:
        args.threshold = None
    {'prepare': prepare, 'train': train, 'apply': apply_head}[args.mode](args)


if __name__ == '__main__':
    main()
