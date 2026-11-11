import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { Garu } from '../src/node';
import { splitSentences } from '../src/sentences';
import type { SentenceAnalyzer } from '../src/sentences';
import { splitSentences as splitPunctuation } from '../src/normalize';

describe('optional sentence verifier entry', () => {
  let garu: Garu;
  let analyzer: SentenceAnalyzer;

  beforeAll(async () => {
    garu = await Garu.load();
    analyzer = garu;
  });

  afterAll(() => {
    garu?.destroy();
  });

  it('splits offline without loading a morphological analyzer', () => {
    expect(splitSentences('안녕하세요반갑습니다')).toEqual([
      { text: '안녕하세요', offset: 0 },
      { text: '반갑습니다', offset: 5 },
    ]);
    expect(splitSentences('  😀안녕하세요\t반갑습니다  ')).toEqual([
      { text: '😀안녕하세요', offset: 2 },
      { text: '반갑습니다', offset: 9 },
    ]);
    expect(splitSentences('“안녕하세요반갑습니다”')).toEqual([
      { text: '“안녕하세요반갑습니다”', offset: 0 },
    ]);
  });

  it('splits SNS exclamations and keeps trailing ornaments with the sentence', () => {
    expect(splitSentences('가방에 쏙! 파우치에 쏙! 음료보다 가벼워')).toEqual([
      { text: '가방에 쏙!', offset: 0 },
      { text: '파우치에 쏙!', offset: 7 },
      { text: '음료보다 가벼워', offset: 15 },
    ]);
    expect(splitSentences('오메가3는 필수인것 같아요~!')).toEqual([
      { text: '오메가3는 필수인것 같아요~!', offset: 0 },
    ]);
    expect(splitSentences('영어를 시작했어요 ㅎㅎ 영어교육의 명문과 함께 읽어볼게요')).toEqual([
      { text: '영어를 시작했어요 ㅎㅎ', offset: 0 },
      { text: '영어교육의 명문과 함께 읽어볼게요', offset: 13 },
    ]);
  });

  it('does not split inside a word of spaced input', () => {
    expect(splitSentences('새로운 컬러인 다크그레이, 실버 최초로 공개되었어요~')).toEqual([
      { text: '새로운 컬러인 다크그레이, 실버 최초로 공개되었어요~', offset: 0 },
    ]);
  });

  it('ends a sentence at every line break with codepoint offsets', () => {
    expect(splitSentences('맛있어요\n분위기도 좋아요')).toEqual([
      { text: '맛있어요', offset: 0 },
      { text: '분위기도 좋아요', offset: 5 },
    ]);
    expect(splitSentences('첫 줄입니다\r\n\n😀둘째 줄입니다')).toEqual([
      { text: '첫 줄입니다', offset: 0 },
      { text: '😀둘째 줄입니다', offset: 9 },
    ]);
  });

  it('uses the selected local model with a real Garu analyzer', () => {
    expect(splitSentences('안녕하세요반갑습니다', analyzer)).toEqual([
      { text: '안녕하세요', offset: 0 },
      { text: '반갑습니다', offset: 5 },
    ]);
    expect(splitPunctuation('안녕하세요반갑습니다')).toEqual([
      { text: '안녕하세요반갑습니다', offset: 0 },
    ]);
  });

  it('returns original text spans and codepoint offsets around emoji and whitespace', () => {
    const text = '  😀안녕하세요\t반갑습니다  ';
    expect(splitSentences(text, analyzer)).toEqual([
      { text: '😀안녕하세요', offset: 2 },
      { text: '반갑습니다', offset: 9 },
    ]);
  });

  it('preserves a Unicode URL while separating the following sentence', () => {
    expect(splitSentences('주소는 https://example.com/한글검색 입니다.다음입니다.', analyzer)).toEqual([
      { text: '주소는 https://example.com/한글검색 입니다.', offset: 0 },
      { text: '다음입니다.', offset: 33 },
    ]);
  });

  it('does not call the analyzer when no candidates remain', () => {
    const unused: SentenceAnalyzer = { analyze() { throw new Error('no candidates'); } };
    expect(splitSentences('', unused)).toEqual([{ text: '', offset: 0 }]);
    expect(splitSentences(' \n\t', unused)).toEqual([{ text: '', offset: 0 }]);
    expect(splitSentences('https://example.com/한글검색', unused)).toEqual([
      { text: 'https://example.com/한글검색', offset: 0 },
    ]);
    expect(splitSentences('“안녕하세요반갑습니다”', unused)).toEqual([
      { text: '“안녕하세요반갑습니다”', offset: 0 },
    ]);
  });
});
