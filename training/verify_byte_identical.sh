#!/usr/bin/env bash
# 디코더 변경이 출력을 바꾸지 않았는지 검증한다.
#   baseline : 현재 빌드의 출력을 기준으로 저장
#   check    : 현재 빌드의 출력을 기준과 비교 (다르면 exit 1)
#
# 코퍼스는 v15k 골드 9,000문장 + 구어 held-out(NIKL 2025 SX) 16,407문장.
# 산출물은 target/ 아래(gitignore)라 커밋되지 않는다.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-check}"
OUT=target/baseline
mkdir -p "$OUT"

export GARU_MODEL=js/models/base.gmdl
cargo build --release --example analyze_batch

if [ ! -s "$OUT/corpus_v15k.txt" ]; then
  python3 -c "
import json
with open('training/gold_testset/gold_testset.jsonl') as f, open('$OUT/corpus_v15k.txt','w') as o:
    for line in f:
        o.write(json.loads(line)['text'].replace('\n',' ')+'\n')
"
fi

if [ ! -s "$OUT/corpus_guueh.txt" ]; then
  python3 -c "
import sys
sys.path.insert(0, 'training')
try:
    from eval_nikl2025_guueh import load_sx
except Exception as e:
    print(f'[skip] 구어 코퍼스 로드 불가: {e}', file=sys.stderr); sys.exit(0)
rows = load_sx()   # list of (text, morphs)
with open('$OUT/corpus_guueh.txt','w') as o:
    for text, _ in rows:
        o.write(text.replace('\n',' ')+'\n')
" || true
fi

status=0
for c in v15k guueh; do
  src="$OUT/corpus_$c.txt"
  if [ ! -s "$src" ]; then echo "[skip] $c 코퍼스 없음"; continue; fi
  cur="$OUT/cur_$c.txt"; base="$OUT/base_$c.txt"
  ./target/release/examples/analyze_batch "$src" > "$cur"
  if [ "$MODE" = "baseline" ]; then
    cp "$cur" "$base"
    echo "[baseline] $c: $(grep -c '^---$' "$base") 문장, sha=$(shasum -a 256 "$base" | cut -c1-16)"
  else
    if [ ! -s "$base" ]; then echo "[FAIL] $c: 기준 없음 — 먼저 baseline 실행"; status=1; continue; fi
    if cmp -s "$cur" "$base"; then
      echo "[OK] $c: byte-identical ($(grep -c '^---$' "$cur") 문장)"
    else
      echo "[FAIL] $c: 출력 불일치 — 다른 줄 $(diff "$base" "$cur" | grep -c '^<' || true)개"
      diff "$base" "$cur" | head -20
      status=1
    fi
  fi
done
exit $status
