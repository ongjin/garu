"""simhash 기반 문장 dedup.

64bit simhash + hamming distance threshold. 단일 프로세스 메모리 보관.
스펙: ≥80% 유사도 dedup. hamming threshold=6은 64bit에서 ~90% 유사도에 해당.
threshold=3은 더 엄격.
"""
from simhash import Simhash


class SimhashDedup:
    def __init__(self, hamming_threshold: int = 6):
        self.threshold = hamming_threshold
        self._fingerprints: list[Simhash] = []

    def add(self, text: str) -> bool:
        """문장 추가. 이미 본 적 있으면 False, 처음이면 True."""
        sh = Simhash(text)
        for existing in self._fingerprints:
            if sh.distance(existing) <= self.threshold:
                return False
        self._fingerprints.append(sh)
        return True
