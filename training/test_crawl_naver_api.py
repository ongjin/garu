"""Naver Search API client 테스트 (HTTP mock)."""
import sys, os
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "crawl"))
from naver_api import NaverSearchClient

# Naver API response shape (실제 형식과 동일)
FAKE_NEWS_RESPONSE = {
    "lastBuildDate": "Thu, 21 May 2026 14:00:00 +0900",
    "total": 100,
    "start": 1,
    "display": 2,
    "items": [
        {
            "title": "<b>경제</b> 회복세 둔화",
            "originallink": "https://hani.co.kr/news/123",
            "link": "https://news.naver.com/x",
            "description": "한국 경제는 다소 둔화되었다.",
            "pubDate": "Thu, 21 May 2026 13:00:00 +0900",
        },
        {
            "title": "신년 <b>인사</b>",
            "originallink": "https://chosun.com/news/456",
            "link": "https://news.naver.com/y",
            "description": "안녕하세요 새해 복 많이 받으세요.",
            "pubDate": "Thu, 21 May 2026 12:00:00 +0900",
        },
    ],
}


def test_search_news_returns_items():
    client = NaverSearchClient(client_id="testid", client_secret="testsecret")
    with patch("naver_api.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = FAKE_NEWS_RESPONSE
        mock_get.return_value = mock_resp
        items = client.search_news("경제", display=2)
    assert len(items) == 2
    assert items[0]["originallink"].startswith("https://hani")
    # HTML 태그 제거 확인 (title의 <b>...</b>)
    assert "<b>" not in items[0]["title"]

def test_search_news_passes_auth_headers():
    client = NaverSearchClient(client_id="myid", client_secret="mysecret")
    with patch("naver_api.requests.get") as mock_get:
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"items": []}
        mock_get.return_value = mock_resp
        client.search_news("foo")
        kwargs = mock_get.call_args.kwargs
        assert kwargs["headers"]["X-Naver-Client-Id"] == "myid"
        assert kwargs["headers"]["X-Naver-Client-Secret"] == "mysecret"

def test_search_blog_returns_items():
    client = NaverSearchClient(client_id="x", client_secret="y")
    with patch("naver_api.requests.get") as mock_get:
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"items": [{"title": "블로그 글", "link": "http://blog.naver.com/x"}]}
        mock_get.return_value = mock_resp
        items = client.search_blog("취미")
    assert len(items) == 1

def test_search_rejects_invalid_response():
    client = NaverSearchClient(client_id="x", client_secret="y")
    with patch("naver_api.requests.get") as mock_get:
        mock_resp = MagicMock(status_code=401)
        mock_resp.text = "unauthorized"
        mock_get.return_value = mock_resp
        try:
            client.search_news("test")
            assert False, "should have raised on 401"
        except RuntimeError as e:
            assert "401" in str(e) or "unauthorized" in str(e).lower()

if __name__ == "__main__":
    test_search_news_returns_items(); print("news OK")
    test_search_news_passes_auth_headers(); print("auth OK")
    test_search_blog_returns_items(); print("blog OK")
    test_search_rejects_invalid_response(); print("error OK")
    print("NAVER API OK")
