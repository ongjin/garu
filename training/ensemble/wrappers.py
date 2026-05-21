"""4분석기 wrapper + Garu wrapper. 공통 인터페이스: analyze(text) -> [(surface, pos)].

POS는 자동으로 세종 표준으로 정규화. 컴파운드 POS(예: EP+EF)는 split하지 않고 첫 태그만 사용
(합의 단위가 단일 형태소이므로 컴파운드 어미 분석 결과는 후보 풀에 영향 적음).

Garu는 합의 분모(4명)에서 빠지지만 dedup용으로 동일 인터페이스 제공.

스펙: .specs/2026-05-21-dict-expansion-design.md 섹션 2.1-2.4.
"""
import json, os, subprocess, sys, tempfile
from pos_normalize import normalize_pos, split_compound_pos


def _norm_token(surface: str, raw_pos: str, source: str) -> tuple[str, str]:
    """단일 토큰 (surface, raw_pos) → (surface, normalized_pos)."""
    first = split_compound_pos(raw_pos)[0]
    return (surface, normalize_pos(first, source=source))


class KiwiWrapper:
    def __init__(self):
        from kiwipiepy import Kiwi
        self._k = Kiwi()

    def analyze(self, text: str) -> list[tuple[str, str]]:
        r = self._k.analyze(text)
        if not r:
            return []
        return [
            _norm_token(m.form, m.tag.replace("-I", "").replace("-R", ""), "kiwi")
            for m in r[0][0]
        ]


class MecabWrapper:
    def __init__(self):
        import mecab
        self._m = mecab.MeCab()

    def analyze(self, text: str) -> list[tuple[str, str]]:
        return [_norm_token(form, tag, "mecab") for form, tag in self._m.pos(text)]


class KkmaWrapper:
    def __init__(self):
        from konlpy.tag import Kkma
        self._k = Kkma()

    def analyze(self, text: str) -> list[tuple[str, str]]:
        return [_norm_token(form, tag, "kkma") for form, tag in self._k.pos(text)]


class KomoranWrapper:
    def __init__(self):
        from konlpy.tag import Komoran
        self._k = Komoran()

    def analyze(self, text: str) -> list[tuple[str, str]]:
        return [_norm_token(form, tag, "komoran") for form, tag in self._k.pos(text)]


_GARU_BIN = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "target", "release", "examples", "analyze_batch",
)
_GARU_MODEL = os.path.join(
    os.path.dirname(__file__), "..", "..", "js", "models", "base.gmdl"
)


class GaruWrapper:
    """Garu wrapper. analyze_batch 바이너리 subprocess 호출 (dedup용).

    eval_f1.py와 동일 패턴: tempfile에 텍스트 쓰고 path 인자로 전달.
    """
    def analyze(self, text: str) -> list[tuple[str, str]]:
        if not os.path.exists(_GARU_BIN):
            raise FileNotFoundError(
                f"Garu binary not found at {_GARU_BIN}. "
                "Run `cargo build --release --examples` first."
            )
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(text + "\n")
            tmp_path = f.name
        try:
            proc = subprocess.run(
                [_GARU_BIN, tmp_path, "--json"],
                capture_output=True, text=True,
                env={**os.environ, "GARU_MODEL": _GARU_MODEL},
                timeout=30,
            )
        finally:
            os.unlink(tmp_path)
        if proc.returncode != 0:
            raise RuntimeError(f"Garu failed: {proc.stderr[:300]}")
        line = proc.stdout.strip().split("\n")[0]
        tokens = json.loads(line)
        return [_norm_token(t[0], t[1], "garu") for t in tokens]
