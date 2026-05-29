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
        try:
            r = self._k.analyze(text)
        except Exception:
            return []
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
        try:
            return [_norm_token(form, tag, "mecab") for form, tag in self._m.pos(text)]
        except Exception:
            return []


class KkmaWrapper:
    def __init__(self):
        from konlpy.tag import Kkma
        self._k = Kkma()

    def analyze(self, text: str) -> list[tuple[str, str]]:
        try:
            return [_norm_token(form, tag, "kkma") for form, tag in self._k.pos(text)]
        except Exception:
            return []


class KomoranWrapper:
    def __init__(self):
        from konlpy.tag import Komoran
        self._k = Komoran()

    def analyze(self, text: str) -> list[tuple[str, str]]:
        try:
            return [_norm_token(form, tag, "komoran") for form, tag in self._k.pos(text)]
        except Exception:
            return []


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

    `analyze_batch(texts)`는 한 subprocess 안에서 N개 텍스트를 한꺼번에 분석.
    Rust 바이너리가 입력 라인 수만큼 정확히 출력 라인을 내므로 1:1 매핑.
    어절별 호출 → 문장당 1회로 줄이는 cleansing 파이프라인 최적화에 사용.
    """
    def analyze_batch(self, texts: list[str]) -> list[list[tuple[str, str]]]:
        if not texts:
            return []
        if not os.path.exists(_GARU_BIN):
            raise FileNotFoundError(
                f"Garu binary not found at {_GARU_BIN}. "
                "Run `cargo build --release --examples` first."
            )
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            # 각 텍스트의 내부 개행은 단일 라인으로 압축 (Rust 쪽이 line-by-line 읽음).
            # trailing \n 포함 — Rust 쪽 trim().is_empty() 분기가 빈 줄을 []로 받아주므로 안전.
            for t in texts:
                f.write(t.replace("\n", " ") + "\n")
            tmp_path = f.name
        try:
            proc = subprocess.run(
                [_GARU_BIN, tmp_path, "--json"],
                capture_output=True, text=True,
                env={**os.environ, "GARU_MODEL": _GARU_MODEL},
                timeout=60,
            )
        finally:
            os.unlink(tmp_path)
        if proc.returncode != 0:
            raise RuntimeError(f"Garu failed: {proc.stderr[:300]}")
        # splitlines()는 trailing newline을 무시. 출력 라인 수 == 입력 라인 수 보장.
        lines = proc.stdout.splitlines()
        if len(lines) < len(texts):
            raise RuntimeError(
                f"Garu batch output line count mismatch: "
                f"expected {len(texts)}, got {len(lines)}"
            )
        results: list[list[tuple[str, str]]] = []
        for line in lines[:len(texts)]:
            tokens = json.loads(line)
            results.append([_norm_token(t[0], t[1], "garu") for t in tokens])
        return results

    def analyze(self, text: str) -> list[tuple[str, str]]:
        # 빈 문자열은 subprocess 우회 (Garu는 빈 입력에도 []만 내지만, 굳이 호출 안 함).
        if not text:
            return []
        return self.analyze_batch([text])[0]
