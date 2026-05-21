"""뉴스 RSS 크롤러 테스트 (mock feedparser)."""
import sys, os
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "crawl"))
from crawl_news import NewsRSSCrawler

FAKE_FEED = MagicMock(entries=[
    MagicMock(link="http://hani.co.kr/news/1", title="기사1", summary="요약1"),
    MagicMock(link="http://hani.co.kr/news/2", title="기사2", summary="요약2"),
])

FAKE_ARTICLE_HTML = """<html><body>
<article>
<p>이것은 정치 기사입니다. 정부는 새로운 정책을 발표했습니다.</p>
<p>김 의원은 강한 반대 의사를 표명했습니다.</p>
</article>
</body></html>"""

def test_parse_feed_returns_entries():
    crawler = NewsRSSCrawler(["http://hani.co.kr/rss"], rate_limiter=None, robots=None)
    with patch("crawl_news.feedparser.parse") as mock_parse:
        mock_parse.return_value = FAKE_FEED
        entries = crawler.list_articles()
    assert len(entries) == 2
    assert entries[0]["link"] == "http://hani.co.kr/news/1"

def test_fetch_article_extracts_sentences():
    rl = MagicMock(); robots = MagicMock()
    robots.is_allowed.return_value = True
    crawler = NewsRSSCrawler([], rate_limiter=rl, robots=robots)
    with patch("crawl_news.requests.get") as mock_get:
        mock_resp = MagicMock(status_code=200)
        mock_resp.text = FAKE_ARTICLE_HTML
        mock_get.return_value = mock_resp
        sents = crawler.fetch_article_sentences("http://hani.co.kr/news/1")
    assert len(sents) >= 1
    assert any("정부" in s for s in sents)
    rl.wait.assert_called_once()
    robots.is_allowed.assert_called_once()

def test_fetch_article_respects_robots_disallow():
    rl = MagicMock(); robots = MagicMock()
    robots.is_allowed.return_value = False
    crawler = NewsRSSCrawler([], rate_limiter=rl, robots=robots)
    with patch("crawl_news.requests.get") as mock_get:
        sents = crawler.fetch_article_sentences("http://hani.co.kr/news/blocked")
    assert sents == []
    mock_get.assert_not_called()  # robots disallow → 호출조차 안 함

if __name__ == "__main__":
    test_parse_feed_returns_entries(); print("feed OK")
    test_fetch_article_extracts_sentences(); print("article OK")
    test_fetch_article_respects_robots_disallow(); print("robots OK")
    print("NEWS CRAWLER OK")
