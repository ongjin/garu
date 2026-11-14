import { describe, it, expect } from 'vitest';
import { splitSentences } from '../src/normalize';
import { restoreSpacing, splitUnspaced } from '../src/text';

describe('spacing restoration', () => {
  it('restores missing Korean word boundaries', () => {
    expect(restoreSpacing('나는밥을먹었다.')).toBe('나는 밥을 먹었다.');
  });

  it('uses the conservative default to avoid splitting a common noun', () => {
    const text = '오늘은날씨가좋아서산책을했다';
    expect(restoreSpacing(text).includes('날씨')).toBe(true);
    expect(restoreSpacing(text, { mode: 'balanced' }).replace(/\s/gu, '')).toBe(text);
  });

  it('keeps original whitespace and all non-whitespace characters', () => {
    const text = '  😀나는밥을먹었다.\t너는물을마셨다.\n';
    const restored = restoreSpacing(text);
    expect(restored.startsWith('  😀')).toBe(true);
    expect(restored.includes('\t')).toBe(true);
    expect(restored.endsWith('\n')).toBe(true);
    expect(restored.replace(/\s/gu, '')).toBe(text.replace(/\s/gu, ''));
  });

  it('preserves URLs, email addresses, and numeric notation', () => {
    for (const text of ['https://example.com/a?q=1.2', 'name@example.com', '3.14', '2026-10-20']) {
      expect(restoreSpacing(text)).toBe(text);
    }
    const restored = restoreSpacing('주소는 https://example.com/a?q=1.2 입니다.');
    expect(restored.includes('https://example.com/a?q=1.2')).toBe(true);
  });

  it('restores Korean text following an unspaced ASCII URL', () => {
    const restored = restoreSpacing('링크는https://example.com입니다오늘도밥을먹었다');
    expect(restored.includes('https://example.com')).toBe(true);
    expect(restored.endsWith('오늘도 밥을 먹었다')).toBe(true);
  });

  it('keeps a whitespace-delimited Unicode URL intact', () => {
    const text = '주소는 https://example.com/한글검색 입니다.';
    expect(restoreSpacing(text).includes('https://example.com/한글검색')).toBe(true);
  });

  it('retains long punctuation runs', () => {
    const punctuation = '.'.repeat(10000);
    expect(restoreSpacing('가' + punctuation).replace(/\s/gu, '')).toBe('가' + punctuation);
  });

  it('does not replace underscores or supplementary Unicode characters', () => {
    const text = '𠮷😀나는_밥을먹었다';
    expect(restoreSpacing(text).replace(/\s/gu, '')).toBe(text);
  });

  it('keeps empty and whitespace-only inputs', () => {
    expect(restoreSpacing('')).toBe('');
    expect(restoreSpacing(' \n\t')).toBe(' \n\t');
  });
});

describe('inferred sentence boundaries', () => {
  it('separates complete sentences without spaces or punctuation when requested', () => {
    expect(splitUnspaced('안녕하세요반갑습니다')).toEqual([
      { text: '안녕하세요', offset: 0 },
      { text: '반갑습니다', offset: 5 },
    ]);
  });

  it('keeps the existing punctuation-only default', () => {
    expect(splitSentences('안녕하세요반갑습니다')).toEqual([
      { text: '안녕하세요반갑습니다', offset: 0 },
    ]);
    expect(splitSentences('안녕하세요. 반갑습니다.')).toEqual([
      { text: '안녕하세요.', offset: 0 },
      { text: '반갑습니다.', offset: 7 },
    ]);
  });

  it('offsets the punctuation default past leading whitespace', () => {
    expect(splitSentences('  공백 문장.')).toEqual([{ text: '공백 문장.', offset: 2 }]);
    expect(splitSentences('\n첫 문장. 둘째 문장.')).toEqual([
      { text: '첫 문장.', offset: 1 },
      { text: '둘째 문장.', offset: 7 },
    ]);
    expect(splitSentences('\t탭 문장. 끝.')).toEqual([
      { text: '탭 문장.', offset: 1 },
      { text: '끝.', offset: 7 },
    ]);
  });

  it('returns original substrings with Unicode codepoint offsets', () => {
    const text = '  😀안녕하세요반갑습니다  ';
    const segments = splitUnspaced(text);
    expect(segments[0].offset).toBe(2);
    expect(segments.map(segment => segment.text).join('')).toBe(text.trim());
    for (const segment of segments) {
      expect([...text].slice(segment.offset, segment.offset + [...segment.text].length).join(''))
        .toBe(segment.text);
    }
  });
});
