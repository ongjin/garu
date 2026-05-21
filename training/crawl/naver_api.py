"""Naver Search API client (https://developers.naver.com).

뉴스/블로그 검색 endpoint. HTML 태그 stripping. 10000 req/day quota.
"""
import re

import requests

_TAG_STRIP = re.compile(r"<[^>]+>")


def _strip_html_tags(s: str) -> str:
    return _TAG_STRIP.sub("", s)


class NaverSearchClient:
    BASE_URL = "https://openapi.naver.com/v1/search"

    def __init__(self, client_id: str, client_secret: str, timeout: float = 10.0):
        if not client_id or not client_secret:
            raise ValueError("client_id and client_secret are required")
        self.client_id = client_id
        self.client_secret = client_secret
        self.timeout = timeout

    def _request(self, endpoint: str, query: str, display: int, start: int) -> dict:
        url = f"{self.BASE_URL}/{endpoint}.json"
        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
        }
        params = {"query": query, "display": display, "start": start}
        resp = requests.get(url, headers=headers, params=params, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Naver API {endpoint} failed: HTTP {resp.status_code} {resp.text[:200]}"
            )
        return resp.json()

    def search_news(self, query: str, display: int = 100, start: int = 1) -> list[dict]:
        data = self._request("news", query, display, start)
        items = data.get("items", [])
        for it in items:
            it["title"] = _strip_html_tags(it.get("title", ""))
            it["description"] = _strip_html_tags(it.get("description", ""))
        return items

    def search_blog(self, query: str, display: int = 100, start: int = 1) -> list[dict]:
        data = self._request("blog", query, display, start)
        items = data.get("items", [])
        for it in items:
            it["title"] = _strip_html_tags(it.get("title", ""))
            it["description"] = _strip_html_tags(it.get("description", ""))
        return items
