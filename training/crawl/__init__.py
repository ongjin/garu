"""Garu 사전 확장 P1+P2: 크롤 인프라 + 후보 풀 생성.

서브모듈:
- rate_limiter: 도메인별 최소 간격 강제
- robots: robots.txt 파서 + 캐시
- dedup: simhash 기반 문장 중복 제거
- text_clean: trafilatura 본문 추출 + 한국어 문장 필터
- naver_api: Naver Search API (news, blog)
- crawl_news: 한국 언론사 RSS 크롤
- crawl_specialist: 국가법령정보센터 OpenAPI 등 전문 도메인
- crawl_blog: Naver 블로그 검색
- run_pilot: 도메인별 시범 크롤 orchestrator
- analyze_corpus: 5분석기 합의 점수 부여 → candidates_pool.jsonl

스펙: .specs/2026-05-21-dict-expansion-design.md 섹션 4.
"""
