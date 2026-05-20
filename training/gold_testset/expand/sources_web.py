"""WebFetch 결과파일에서 raw 문장 로드.

WebFetch 실행은 Claude가 Phase 6에서 직접 수행하고
결과를 _candidates/<domain>_web_raw.txt에 저장한 것을 가정.
"""
import random
from pathlib import Path


def load_web_raw(path: str, n: int = None, seed: int = 42,
                 min_len: int = 5, max_len: int = 80) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. WebFetch로 먼저 채우세요.")

    candidates = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if min_len <= len(line) <= max_len:
                candidates.append(line)

    rng = random.Random(seed)
    unique = list(dict.fromkeys(candidates))
    rng.shuffle(unique)
    if n is None:
        return unique
    return unique[:n]


if __name__ == "__main__":
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
        f.write("이건 테스트 문장입니다.\n")
        f.write("두 번째 문장.\n")
        f.write("ab\n")
        f.write("이건 또 테스트 문장입니다.\n")
        mock_path = f.name
    sents = load_web_raw(mock_path, n=10)
    assert len(sents) == 3, f"길이 필터 적용 후 3개여야 함, got {len(sents)}"
    os.unlink(mock_path)
    print("OK")
