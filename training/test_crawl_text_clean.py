"""HTML 본문 추출 + 한국어 문장 필터 테스트."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "crawl"))
from text_clean import extract_sentences, is_korean_sentence

HTML_SAMPLE = """<html><body>
<article>
<p>오늘 서울은 비가 옵니다. 우산 챙기세요!</p>
<p>The weather is rainy today.</p>
<p>123 456 789</p>
<p>안녕하세요</p>
</article>
</body></html>"""

def test_is_korean_sentence_korean_high_ratio():
    assert is_korean_sentence("오늘 서울은 비가 옵니다.") is True
    assert is_korean_sentence("안녕하세요 한국어 문장입니다") is True

def test_is_korean_sentence_rejects_english():
    assert is_korean_sentence("The weather is rainy today.") is False

def test_is_korean_sentence_rejects_too_short():
    assert is_korean_sentence("안녕") is False  # 5자 미만

def test_is_korean_sentence_rejects_numeric():
    assert is_korean_sentence("123 456 789") is False

def test_is_korean_sentence_rejects_too_long():
    long_text = "가나다라마바사아자차카타파하" * 20  # 260+ chars
    assert is_korean_sentence(long_text) is False

def test_extract_sentences_from_html():
    sents = extract_sentences(HTML_SAMPLE)
    # 한국어 + 길이 5-200 + 한글 비율 60%+ 만 통과
    assert any("오늘 서울" in s for s in sents), f"missing first KR sent: {sents}"
    assert any("우산" in s for s in sents), f"missing exclamation sent: {sents}"
    assert not any("weather" in s for s in sents)
    assert not any("123" in s for s in sents)
    assert not any("안녕하세요" == s for s in sents)  # 5자 미만

if __name__ == "__main__":
    test_is_korean_sentence_korean_high_ratio(); print("kr_high OK")
    test_is_korean_sentence_rejects_english(); print("en_reject OK")
    test_is_korean_sentence_rejects_too_short(); print("short_reject OK")
    test_is_korean_sentence_rejects_numeric(); print("num_reject OK")
    test_is_korean_sentence_rejects_too_long(); print("long_reject OK")
    test_extract_sentences_from_html(); print("html_extract OK")
    print("TEXT CLEAN OK")
