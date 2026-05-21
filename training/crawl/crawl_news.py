"""뉴스 RSS 크롤러. RSS feed 파싱 → 본문 fetch → 문장 추출.

rate_limiter와 robots을 dependency injection으로 주입 (테스트 용이).
"""
import feedparser
import requests

from text_clean import extract_sentences


class NewsRSSCrawler:
    USER_AGENT = "garu-crawler/0.1 (+https://garu.zerry.co.kr)"

    def __init__(self, feed_urls: list[str], rate_limiter, robots, timeout: float = 15.0):
        self.feeds = feed_urls
        self.rl = rate_limiter
        self.robots = robots
        self.timeout = timeout

    def list_articles(self) -> list[dict]:
        out = []
        for url in self.feeds:
            if self.rl is not None:
                self.rl.wait(url)
            d = feedparser.parse(url)
            for e in d.entries:
                out.append({
                    "link": getattr(e, "link", ""),
                    "title": getattr(e, "title", ""),
                    "summary": getattr(e, "summary", ""),
                })
        return out

    def fetch_article_sentences(self, url: str) -> list[str]:
        if self.robots is not None and not self.robots.is_allowed(url):
            return []
        if self.rl is not None:
            self.rl.wait(url)
        try:
            resp = requests.get(url, headers={"User-Agent": self.USER_AGENT},
                                timeout=self.timeout)
        except Exception:
            return []
        if resp.status_code != 200:
            return []
        return extract_sentences(resp.text)
