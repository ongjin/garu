"""Train the masked-context 사/살 classifier with the retained NIKL 2021 train IDs."""
import argparse
import gzip
import hashlib
import json
import os
import re
import struct
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from context_cache import sections
from prepare import documents

PUNCTUATION = {'SS', 'SF', 'SP', 'SW', 'SE', 'SO'}
GRAMMAR = {'JKO', 'JKB', 'JKS', 'JKG', 'JKC', 'JKQ', 'JX', 'JC',
           'EC', 'EF', 'ETM', 'ETN', 'EP'}
CONTEXT_WIDTH = 4

def feature_hash(name):
    h = 2166136261
    for byte in name.encode('utf8'):
        h = ((h ^ byte) * 16777619) & 0xffffffff
    return h

def encode(bias, weights):
    scale = float(max(abs(w) for w in weights.values())) / 127
    pairs = sorted((feature_hash(name), round(float(weight) / scale))
                   for name, weight in weights.items())
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError('Two retained feature names have the same FNV-1a key')
    header = struct.pack('<ffI', bias, scale, len(pairs))
    return header + b''.join(struct.pack('<Ib', key, q) for key, q in pairs)

def training_rows(data):
    refs_path = data / 'korean_context_pos/20261020/train.refs.jsonl.gz'
    with gzip.open(refs_path, 'rt') as source:
        refs = {r['id']: r['text_sha256'] for r in map(json.loads, source)}
    matched = 0
    for path in sorted((data / 'nikl_mp_2021').glob('*.json')):
        for doc in documents(path):
            for sentence in (doc or {}).get('sentence', []):
                sid = sentence['id']
                if sid not in refs:
                    continue
                text = sentence['form']
                if hashlib.sha256(text.encode()).hexdigest() != refs[sid]:
                    raise ValueError(f'Text hash mismatch for {sid}')
                matched += 1
                by_word = defaultdict(list)
                for m in sentence.get('MP') or sentence['morpheme']:
                    if m['label'] not in PUNCTUATION:
                        by_word[m['word_id']].append((m['form'], m['label']))
                for word in sentence['word']:
                    morphs = by_word[word['id']]
                    if not morphs or morphs[0] not in {('사', 'VV'), ('살', 'VV')}:
                        continue
                    if '살게요' in word['form']:
                        continue
                    yield text, word['begin'], word['end'], morphs[0][0], doc['id']
    if matched != len(refs):
        raise ValueError(f'Matched {matched} of {len(refs)} retained IDs')

def window_features(text, begin, end, tokens):
    words = list(re.finditer(r'\S+', text))
    left = [w for w in words if w.end() <= begin][-CONTEXT_WIDTH:][::-1]
    right = [w for w in words if w.start() >= end][:CONTEXT_WIDTH]
    features = set()
    for side, neighbors in [('L', left), ('R', right)]:
        if not neighbors:
            features.add(side + ':BOS/EOS')
        for distance, word in enumerate(neighbors):
            parts = []
            for form, tag, b, e in tokens:
                if not (word.start() <= b < word.end() and e <= word.end()):
                    continue
                if tag == 'EF' and form in {'ㄹ게요', '을게요'}:
                    parts.extend([(form[:-1], tag), ('요', 'JX')])
                else:
                    parts.append((form, tag))
            noun = None
            for form, tag in parts:
                feature = form + '/' + tag
                if tag in GRAMMAR:
                    features.add(side + 'Gr:' + feature)
                    if distance == 0:
                        features.add(side + '1Gr:' + feature)
                elif tag[0] not in 'SJE':
                    features.update(('lex:' + feature, side + 'Lex:' + feature))
                    if distance == 0:
                        features.add(side + '1Lex:' + feature)
                    if tag.startswith('N'):
                        noun = form
            if distance == 0 and parts:
                tag = parts[-1][1]
                features.add(side + '1Tag:' + tag)
                suffix = re.sub(r'^[^가-힣]+|[^가-힣]+$', '', word.group())[-2:]
                if suffix:
                    features.add(side + '1Sfx:' + suffix)
                if noun and tag in GRAMMAR:
                    features.add(side + '1Arg:' + noun + '/' + tag)
    return dict.fromkeys(features, 1)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path.home() / 'workspace/data')
    parser.add_argument('--analyzer', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = list(training_rows(args.data))
    # Exclude the classifier being trained so context always comes from the base analyzer.
    parts = sections(args.base.read_bytes())
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory) / 'without-homograph.gmdl'
        base.write_bytes(b'GMDL' + struct.pack('<I', 3) + b''.join(
            struct.pack('<BI', kind, len(blob)) + blob for kind, blob in parts.items() if kind != 19
        ))
        source = Path(directory) / 'contexts.txt'
        with source.open('w', encoding='utf-8') as stream:
            stream.writelines(text + '\n' for text, *_ in rows)
        env = {**os.environ, 'GARU_MODEL': str(base), 'GARU_PROMISE_ENDINGS': 'standard'}
        data = subprocess.check_output([args.analyzer.resolve(), source, '--json-spans'],
                                       env=env, text=True).splitlines()
    if len(data) != len(rows):
        raise RuntimeError('Analyzer row count mismatch')
    features = [window_features(text, b, e, json.loads(output))
                for (text, b, e, _, _), output in zip(rows, data)]
    labels = [int(stem == '사') for _, _, _, stem, _ in rows]
    vectorizer = DictVectorizer()
    matrix = vectorizer.fit_transform(features)
    # liblinear only accepts signed int32 sparse indices.
    matrix.indices = matrix.indices.astype('int32')
    matrix.indptr = matrix.indptr.astype('int32')
    regression = LogisticRegression(C=1.0, solver='liblinear', random_state=0, max_iter=500)
    regression.fit(matrix, labels)
    weights = dict(zip(vectorizer.get_feature_names_out(), regression.coef_[0]))
    blob = encode(float(regression.intercept_[0]), weights)
    args.out.write_bytes(blob)
    print(json.dumps({'documents': len({row[4] for row in rows}), 'counts_by_stem':
                      dict(Counter(r[3] for r in rows)), 'features': len(weights),
                      'bytes': len(blob), 'section_sha256': hashlib.sha256(blob).hexdigest()},
                     ensure_ascii=False))

if __name__ == '__main__':
    main()
