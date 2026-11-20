import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { Garu } from '../src/node';
import type { AnalyzeResult } from '../src/core';

describe('decomposed modern Hangul', () => {
  let garu: Garu;

  beforeAll(async () => { garu = await Garu.load(); });
  afterAll(() => { garu?.destroy(); });

  it('finds the same nouns and morphemes in NFD text', () => {
    const sentence = '학교에서 공부했다';
    const decomposed = sentence.normalize('NFD');
    expect(garu.nouns(decomposed)).toEqual(['학교', '공부']);
    expect(garu.tokenize(decomposed)).toEqual(garu.tokenize(sentence));
  });

  it('preserves the source code-point span for every candidate', () => {
    const words = ['🙂', '하\u11a8교에서', '공부했다'.normalize('NFD'), 'ㅋㅋ'];
    const source = words.join(' ');
    const canonical = source.normalize('NFC');
    const withSource = (text: string, result: AnalyzeResult) => result.tokens.map(token => [
      token.text, token.pos, [...text].slice(token.start, token.end).join('').normalize('NFC'),
    ]);
    for (const topN of [1, 5]) {
      const expected = [garu.analyze(canonical, { topN })].flat();
      const actual = [garu.analyze(source, { topN })].flat();
      expect(actual.map(r => [r.score, withSource(source, r)]))
        .toEqual(expected.map(r => [r.score, withSource(canonical, r)]));
    }
  });

  it('looks up and replaces canonical user dictionary keys', () => {
    const word = '탄소중립';
    const decomposed = word.normalize('NFD');
    try {
      garu.addUserWord(decomposed, 'NNG');
      expect(garu.nouns(word + ' 목표')).toContain(word);
      garu.addUserWord(word, 'NNP');
      expect(garu.userWordCount()).toBe(1);
      const result = garu.analyze(decomposed + ' 목표') as AnalyzeResult;
      expect(result.tokens).toContainEqual(expect.objectContaining({
        text: word, pos: 'NNP', start: 0, end: [...decomposed].length,
      }));
    } finally {
      garu.clearUserWords();
    }
  });

  it('keeps output jamo formatting independent of input composition', async () => {
    const canonicalOutput = await Garu.load({ normalizeJamo: true });
    try {
      const input = '갈 사람 ㅋㅋ'.normalize('NFD');
      expect(garu.tokenize(input)).toContain('ㄹ');
      expect(canonicalOutput.tokenize(input)).toEqual(['가', '\u11af', '사람', 'ㅋ', 'ㅋ']);
    } finally {
      canonicalOutput.destroy();
    }
  });
});
