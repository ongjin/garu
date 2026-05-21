"""robots.txt 파서 + 24h 메모리 캐시 테스트."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "crawl"))
from robots import RobotsCache

# 미니 robots.txt — 의존성 회피 위해 모의 응답 주입
ROBOTS_ALLOW_ALL = """
User-agent: *
Allow: /
"""

ROBOTS_DISALLOW_PRIVATE = """
User-agent: *
Disallow: /private/
Allow: /
"""

class _FakeFetcher:
    def __init__(self, text: str):
        self.text = text
        self.calls = 0
    def __call__(self, url: str) -> str:
        self.calls += 1
        return self.text


def test_allow_all():
    fetcher = _FakeFetcher(ROBOTS_ALLOW_ALL)
    rc = RobotsCache(fetcher=fetcher, user_agent="garu-crawler/0.1")
    assert rc.is_allowed("http://example.com/foo") is True

def test_disallow_path():
    fetcher = _FakeFetcher(ROBOTS_DISALLOW_PRIVATE)
    rc = RobotsCache(fetcher=fetcher, user_agent="garu-crawler/0.1")
    assert rc.is_allowed("http://example.com/private/secret") is False
    assert rc.is_allowed("http://example.com/public") is True

def test_cache_avoids_refetch():
    fetcher = _FakeFetcher(ROBOTS_ALLOW_ALL)
    rc = RobotsCache(fetcher=fetcher, user_agent="garu-crawler/0.1")
    rc.is_allowed("http://example.com/a")
    rc.is_allowed("http://example.com/b")
    rc.is_allowed("http://example.com/c")
    assert fetcher.calls == 1, f"should fetch robots.txt once per host, got {fetcher.calls}"

def test_fetch_failure_default_allow():
    """robots.txt 없거나 fetch 실패 시 기본은 allow (관행)."""
    def fail(url): raise RuntimeError("404")
    rc = RobotsCache(fetcher=fail, user_agent="garu-crawler/0.1")
    assert rc.is_allowed("http://nowhere.example.com/foo") is True

if __name__ == "__main__":
    test_allow_all(); print("allow_all OK")
    test_disallow_path(); print("disallow OK")
    test_cache_avoids_refetch(); print("cache OK")
    test_fetch_failure_default_allow(); print("fallback OK")
    print("ROBOTS OK")
