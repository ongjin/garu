import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { Garu } from '../src/node';

describe('runtime user dictionary', () => {
  let garu: Garu;

  beforeAll(async () => {
    garu = await Garu.load();
  });

  afterAll(() => {
    garu?.destroy();
  });

  const surfaces = (text: string) =>
    (garu.analyze(text) as { tokens: { text: string }[] }).tokens.map((t) => t.text);

  it('splits an unregistered compound', () => {
    expect(surfaces('탄소중립 목표')).not.toContain('탄소중립');
  });

  it('keeps a registered word whole', () => {
    garu.addUserWord('탄소중립', 'NNG');
    expect(surfaces('탄소중립 목표')).toContain('탄소중립');
    expect(garu.userWordCount()).toBe(1);
  });

  it('clearUserWords restores the built-in behaviour', () => {
    garu.clearUserWords();
    expect(garu.userWordCount()).toBe(0);
    expect(surfaces('탄소중립 목표')).not.toContain('탄소중립');
  });

  it('addUserWords registers a batch and nouns() picks them up', () => {
    garu.addUserWords([
      { surface: '전세사기', pos: 'NNG' },
      { surface: '반려동물등록제', pos: 'NNG' },
    ]);
    expect(garu.userWordCount()).toBe(2);
    expect(garu.nouns('전세사기 피해자')).toContain('전세사기');
    expect(surfaces('반려동물등록제 시행')).toContain('반려동물등록제');
    garu.clearUserWords();
  });

  it('rejects an unknown POS tag', () => {
    expect(() => garu.addUserWord('아무말', 'ZZZ' as never)).toThrow();
    expect(garu.userWordCount()).toBe(0);
  });
});
