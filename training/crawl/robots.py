"""robots.txt 파싱 + 호스트 단위 메모리 캐시.

표준 라이브러리 urllib.robotparser 활용. fetch 실패 시 default allow.
"""
from typing import Callable
from urllib import request as urlrequest
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser


def _default_fetcher(url: str) -> str:
    with urlrequest.urlopen(url, timeout=10) as resp:
        return resp.read().decode("utf-8", errors="replace")


class RobotsCache:
    """호스트별 robots.txt 파서 캐시. 단일 프로세스 메모리 캐시."""

    def __init__(
        self,
        fetcher: Callable[[str], str] = _default_fetcher,
        user_agent: str = "garu-crawler/0.1",
    ):
        self._fetch = fetcher
        self._ua = user_agent
        self._cache: dict[str, RobotFileParser] = {}

    def is_allowed(self, url: str) -> bool:
        host = urlparse(url).netloc
        if host not in self._cache:
            self._cache[host] = self._load(host)
        parser = self._cache[host]
        if parser is None:
            return True
        return parser.can_fetch(self._ua, url)

    def _load(self, host: str):
        robots_url = f"http://{host}/robots.txt"
        try:
            text = self._fetch(robots_url)
        except Exception:
            return None  # fetch 실패 → default allow
        rp = RobotFileParser()
        rp.parse(text.splitlines())
        return rp
