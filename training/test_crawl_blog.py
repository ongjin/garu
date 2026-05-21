"""Naver Blog Search 기반 블로그 크롤러 테스트."""
import sys, os
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "crawl"))
from crawl_blog import BlogCrawler

FAKE_BLOG_SEARCH_ITEMS = [
    {"title": "취미 이야기", "link": "http://blog.naver.com/x/123", "description": "오늘 등산을 다녀왔어요."},
    {"title": "맛집 후기", "link": "http://blog.naver.com/y/456", "description": "이 식당 정말 맛있어요!"},
]

FAKE_BLOG_HTML = """<html><body>
<div class="post-content">
<p>오늘 점심으로 김치찌개를 먹었어요. 정말 맛있었어요.</p>
<p>다음에 또 가고 싶은 식당이에요.</p>
</div>
</body></html>"""


def test_search_returns_items():
    naver_client = MagicMock()
    naver_client.search_blog.return_value = FAKE_BLOG_SEARCH_ITEMS
    crawler = BlogCrawler(naver_client=naver_client, rate_limiter=None, robots=None)
    items = crawler.search("취미", display=2)
    assert len(items) == 2

def test_fetch_blog_post_extracts_sentences():
    naver_client = MagicMock()
    rl = MagicMock()
    robots = MagicMock(); robots.is_allowed.return_value = True
    crawler = BlogCrawler(naver_client=naver_client, rate_limiter=rl, robots=robots)
    with patch("crawl_blog.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, text=FAKE_BLOG_HTML)
        sents = crawler.fetch_post_sentences("http://blog.naver.com/x/123")
    assert any("김치찌개" in s for s in sents)
    rl.wait.assert_called_once()

def test_post_search_descriptions_also_extracted():
    """검색 결과의 description 자체도 한국어 문장이면 후보 (전체 본문 크롤 실패 시 fallback)."""
    naver_client = MagicMock()
    naver_client.search_blog.return_value = FAKE_BLOG_SEARCH_ITEMS
    crawler = BlogCrawler(naver_client=naver_client, rate_limiter=None, robots=None)
    items = crawler.search("취미", display=2)
    desc_sents = crawler.descriptions_to_sentences(items)
    assert any("등산" in s for s in desc_sents)
    assert any("식당" in s for s in desc_sents)

if __name__ == "__main__":
    test_search_returns_items(); print("search OK")
    test_fetch_blog_post_extracts_sentences(); print("fetch OK")
    test_post_search_descriptions_also_extracted(); print("desc OK")
    print("BLOG CRAWLER OK")
