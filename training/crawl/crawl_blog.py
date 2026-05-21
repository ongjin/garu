"""Naver Blog Search 기반 블로그 본문 크롤러.

API 결과의 description은 자체 한국어 문장 → fallback source.
원본 블로그 페이지는 robots.txt 준수해서 fetch.
"""
import requests

from text_clean import extract_sentences, is_korean_sentence


class BlogCrawler:
    USER_AGENT = "garu-crawler/0.1 (+https://garu.zerry.co.kr)"

    def __init__(self, naver_client, rate_limiter, robots, timeout: float = 15.0):
        self.naver = naver_client
        self.rl = rate_limiter
        self.robots = robots
        self.timeout = timeout

    def search(self, query: str, display: int = 100, start: int = 1) -> list[dict]:
        return self.naver.search_blog(query, display=display, start=start)

    def search_paginated(self, query: str, total_results: int = 500,
                         display: int = 100) -> list[dict]:
        """Naver Blog 검색 페이지네이션. start=1, 101, 201, ... 까지."""
        all_items = []
        for start in range(1, total_results + 1, display):
            page = self.naver.search_blog(query, display=display, start=start)
            if not page:
                break
            all_items.extend(page)
            if len(page) < display:
                break  # 마지막 페이지
        return all_items

    def descriptions_to_sentences(self, items: list[dict]) -> list[str]:
        out = []
        for it in items:
            desc = it.get("description", "")
            for s in extract_sentences(desc):
                out.append(s)
            if is_korean_sentence(desc):
                out.append(desc)
        return list(dict.fromkeys(out))

    def fetch_post_sentences(self, url: str) -> list[str]:
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
