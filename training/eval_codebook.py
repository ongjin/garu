"""Evaluate codebook analyzer F1 score against Kiwi ground truth.

Usage:
    python training/eval_codebook.py
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.codebook_analyzer import CodebookAnalyzer
from kiwipiepy import Kiwi

POS_TAGS = [
    "NNG", "NNP", "NNB", "NR", "NP",
    "VV", "VA", "VX", "VCP", "VCN",
    "MAG", "MAJ", "MM", "IC",
    "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC",
    "EP", "EF", "EC", "ETN", "ETM",
    "XPN", "XSN", "XSV", "XSA", "XR",
    "SF", "SP", "SS", "SE", "SO", "SW", "SH", "SL", "SN",
]
POS_SET = set(POS_TAGS)


def normalize_pos(tag):
    if tag in POS_SET:
        return tag
    base = tag.split('-')[0]
    if base in POS_SET:
        return base
    for p, d in [('V', 'VV'), ('N', 'NNG'), ('J', 'JX'), ('E', 'EF'), ('X', 'XR')]:
        if tag.startswith(p):
            return d
    return 'SW'


test_sentences = [
    '나는 오늘 학교에서 한국어 형태소 분석기를 만들었다.',
    '서울특별시 강남구에 위치한 회사에 다니고 있습니다.',
    '인공지능 기술이 빠르게 발전하고 있다.',
    '이 프로그램은 Python과 JavaScript로 작성되었다.',
    '됐다. 이제 그만하자.',
    '그가 했던 일은 대단했다.',
    '어제 학교에 갔다가 집에 왔다.',
    'TypeScript는 JavaScript의 상위 집합이다.',
    'React와 Next.js를 사용하여 웹 애플리케이션을 개발했다.',
    '한국어는 교착어로 조사와 어미가 발달한 언어이다.',
    '정부는 내년 예산안을 국회에 제출했다.',
    '삼성전자는 새로운 반도체 공장을 건설할 계획이다.',
    '머신러닝 모델의 정확도를 높이기 위해 데이터를 전처리했다.',
    '프론트엔드 개발자와 백엔드 개발자의 협업이 중요하다.',
    '블록체인 기술은 금융 산업에 혁신을 가져올 수 있다.',
    '매일 한 시간씩 독서하는 습관을 기르고 싶다.',
    '이 라이브러리는 MIT 라이선스로 배포되고 있다.',
    '지하철 2호선은 서울의 주요 노선 중 하나이다.',
    '최근 환율이 급격하게 변동하면서 수출 기업들이 어려움을 겪고 있다.',
    '도커와 쿠버네티스를 활용한 마이크로서비스 아키텍처를 구축했다.',
]


def main():
    data_dir = Path(__file__).parent / "codebook_data"
    print("Loading codebook analyzer...")
    analyzer = CodebookAnalyzer.load(str(data_dir))
    print(f"  Suffix patterns: {len(analyzer.suffix_codebook)}")
    print(f"  Content words: {len(analyzer.content_dict)}")

    kw = Kiwi()

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    tm = 0
    tk = 0
    to_ = 0

    for s in test_sentences:
        kiwi_result = [(t.form, normalize_pos(t.tag)) for t in kw.tokenize(s)]
        codebook_result = analyzer.analyze(s)

        kr = set((f, t) for f, t in kiwi_result if f)
        mr = set((f, t) for f, t in codebook_result if f)

        for item in mr:
            if item in kr:
                tp[item[1]] += 1
            else:
                fp[item[1]] += 1
        for item in kr:
            if item not in mr:
                fn[item[1]] += 1

        tm += len(kr & mr)
        tk += len(kr)
        to_ += len(mr)

    P = tm / max(to_, 1)
    R = tm / max(tk, 1)
    F = 2 * P * R / max(P + R, 1e-10)

    print(f'\n{"=" * 60}')
    print(f'  Precision: {P:.1%}  Recall: {R:.1%}  F1: {F:.1%}')
    print(f'{"=" * 60}\n')

    print(f'{"POS":<8}{"Prec":>8}{"Rec":>8}{"F1":>8}{"TP":>6}{"FP":>6}{"FN":>6}')
    print('-' * 52)
    for tag in sorted(set(list(tp) + list(fp) + list(fn))):
        t, f_p, f_n = tp[tag], fp[tag], fn[tag]
        p = t / max(t + f_p, 1)
        r = t / max(t + f_n, 1)
        f = 2 * p * r / max(p + r, 1e-10)
        print(f'{tag:<8}{p:>8.1%}{r:>8.1%}{f:>8.1%}{t:>6}{f_p:>6}{f_n:>6}')

    print(f'\n틀린 예시:')
    for s in test_sentences[:10]:
        kiwi_result = [(t.form, normalize_pos(t.tag)) for t in kw.tokenize(s)]
        codebook_result = analyzer.analyze(s)
        krs = set((f, t) for f, t in kiwi_result if f)
        mrs = set((f, t) for f, t in codebook_result if f)
        if krs != mrs:
            print(f'\n  Input: {s}')
            print(f'  Kiwi:     {kiwi_result}')
            print(f'  Codebook: {codebook_result}')

    print(f'\n{"=" * 60}')
    if F >= 0.85:
        print(f'  PASS: F1 {F:.1%} >= 85% threshold. Proceed to Rust implementation.')
    else:
        print(f'  BELOW TARGET: F1 {F:.1%} < 85%. Analyze failures and iterate.')
    print(f'{"=" * 60}')


if __name__ == "__main__":
    main()
