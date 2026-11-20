"""Apply the reviewed POS and segmentation changes to a separate benchmark copy."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    review = json.loads((HERE / 'corrections.json').read_text())
    source = HERE / review['source']
    original = source.read_bytes()
    assert hashlib.sha256(original).hexdigest() == review['source_sha256']
    records = [json.loads(line) for line in original.decode().splitlines()]
    for item in review['patches']:
        record = records[item['gold_line'] - 1]
        assert record['text'] == item['text']
        assert record['morphemes'][item['token_index']] == item['before']
        record['morphemes'][item['token_index']] = item['after']
    promises = review['promise_endings']['occurrences']
    for item in sorted(promises, key=lambda row: row['token_index'], reverse=True):
        record = records[item['gold_line'] - 1]
        i = item['token_index']
        assert record['text'] == item['text'] and record['morphemes'][i] == item['before']
        form, pos = item['before']
        assert form in ('ㄹ게요', 'ᆯ게요', '을게요') and pos == 'EF'
        record['morphemes'][i:i + 1] = [[form[:-1], 'EF'], ['요', 'JX']]
    target = HERE / 'gold_corrected.jsonl'
    target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    assert source.read_bytes() == original
    print(f'{len(review["patches"])} POS changes and {len(promises)} splits written to {target}')


if __name__ == '__main__':
    main()
