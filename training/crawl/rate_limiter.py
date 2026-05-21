"""도메인별 rate limiter. 호스트 단위 최소 간격 강제 (정중한 크롤링)."""
import time
from urllib.parse import urlparse


class RateLimiter:
    """sleep-based per-host rate limiter.

    스레드 안전 아님 (단일 프로세스 시범 크롤 전제).
    """

    def __init__(
        self,
        min_interval_sec: float = 1.0,
        per_host_overrides: dict[str, float] | None = None,
    ):
        self.default = min_interval_sec
        self.overrides = per_host_overrides or {}
        self._last_request: dict[str, float] = {}

    def _interval(self, host: str) -> float:
        return self.overrides.get(host, self.default)

    def wait(self, host_or_url: str) -> None:
        """호스트 단위로 다음 요청까지 대기. URL을 넘기면 host 추출."""
        host = self._extract_host(host_or_url)
        now = time.monotonic()
        last = self._last_request.get(host, 0.0)
        elapsed = now - last
        interval = self._interval(host)
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_request[host] = time.monotonic()

    @staticmethod
    def _extract_host(host_or_url: str) -> str:
        if "://" in host_or_url:
            return urlparse(host_or_url).netloc
        return host_or_url
