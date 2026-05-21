"""시범 크롤 orchestrator (news + blog, specialist 제외) — Expanded version.

각 도메인의 정해진 쿼리/소스를 돌면서 한국어 문장을 수집.
출력: training/crawl_data/<domain>/<date>/sentences.jsonl.gz (한 줄당 1 문장 JSON).

스펙: .specs/2026-05-21-dict-expansion-design.md 섹션 4.3.
"""
import argparse
import gzip
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from rate_limiter import RateLimiter
from robots import RobotsCache
from dedup import SimhashDedup
from naver_api import NaverSearchClient
from crawl_news import NewsRSSCrawler
from crawl_blog import BlogCrawler

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "training" / "crawl_data"
TODAY = datetime.now().strftime("%Y%m%d")


def _write_sents(domain: str, sents: list[dict]) -> Path:
    out_dir = DATA_ROOT / domain / TODAY
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "sentences.jsonl.gz"
    with gzip.open(out, "at", encoding="utf-8") as f:
        for s in sents:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return out


def crawl_news(config: dict, rl: RateLimiter, robots: RobotsCache,
               dedup: SimhashDedup) -> int:
    crawler = NewsRSSCrawler(config["rss_feeds"], rate_limiter=rl, robots=robots)
    articles = crawler.list_articles()
    print(f"[news] {len(articles)} articles from {len(config['rss_feeds'])} feeds")
    collected = 0
    for i, art in enumerate(articles[: config["target_articles"]]):
        sents = crawler.fetch_article_sentences(art["link"])
        batch = []
        for s in sents:
            if dedup.add(s):
                batch.append({"text": s, "source": art["link"]})
        if batch:
            _write_sents("news", batch)
            collected += len(batch)
        if (i + 1) % 25 == 0:
            print(f"  [news] processed {i+1} articles, {collected} unique sents so far")
        if collected >= config["target_sentences"]:
            break
    print(f"[news] collected {collected} unique sentences")
    return collected


def crawl_blog(config: dict, naver: NaverSearchClient, rl: RateLimiter,
               robots: RobotsCache, dedup: SimhashDedup) -> int:
    crawler = BlogCrawler(naver, rate_limiter=rl, robots=robots)
    collected = 0
    pages = config.get("pages_per_query", 5)
    display = config.get("display_per_page", 100)
    total_results = pages * display
    for q in config["queries"]:
        items = crawler.search_paginated(q, total_results=total_results, display=display)
        desc_sents = crawler.descriptions_to_sentences(items)
        batch = []
        for s in desc_sents:
            if dedup.add(s):
                batch.append({"text": s, "source": f"blog_desc:{q}"})
        if batch:
            _write_sents("blog", batch)
            collected += len(batch)
        print(f"  [blog] query='{q}' added {len(batch)} (total {collected})")
        if collected >= config["target_sentences"]:
            break
    print(f"[blog] collected {collected} unique sentences")
    return collected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["news", "blog", "all"], default="all")
    ap.add_argument("--config", default=str(Path(__file__).parent / "pilot_queries.json"))
    args = ap.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    rl = RateLimiter(min_interval_sec=1.0)
    robots = RobotsCache(user_agent="garu-crawler/0.1")
    dedup = SimhashDedup(hamming_threshold=16)

    naver_id = os.environ.get("NAVER_CLIENT_ID")
    naver_secret = os.environ.get("NAVER_CLIENT_SECRET")
    naver = NaverSearchClient(naver_id, naver_secret) if naver_id else None

    total = 0
    if args.domain in ("news", "all"):
        total += crawl_news(config["news"], rl, robots, dedup)
    if args.domain in ("blog", "all"):
        if naver is None:
            print("[blog] NAVER_CLIENT_ID unset — skipping")
        else:
            total += crawl_blog(config["blog"], naver, rl, robots, dedup)
    print(f"=== pilot crawl total: {total} unique sentences ===")


if __name__ == "__main__":
    main()
