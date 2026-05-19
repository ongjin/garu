import { describe, it, expect, beforeAll } from 'vitest';
import { Garu } from 'garu-ko';
import { createTokenizer, createProcessTerm, DEFAULT_POS } from '../src/index';

let garu: Garu;

beforeAll(async () => {
  garu = await Garu.load();
}, 30_000);

describe('createTokenizer (MiniSearch)', () => {
  it('returns a function', async () => {
    const tok = await createTokenizer({ garu });
    expect(typeof tok).toBe('function');
  });

  it('drops particles, keeps nouns', async () => {
    const tok = await createTokenizer({ garu });
    const tokens = tok('나는 학교에 갔다');
    expect(tokens).toContain('학교');
    expect(tokens).not.toContain('는');
    expect(tokens).not.toContain('에');
  });

  it('keeps verb and adjective stems', async () => {
    const tok = await createTokenizer({ garu });
    const tokens = tok('빠르게 달렸다');
    expect(tokens).toEqual(expect.arrayContaining(['빠르', '달리']));
  });

  it('lowercases foreign words', async () => {
    const tok = await createTokenizer({ garu });
    const tokens = tok('AI 기술');
    expect(tokens).toContain('ai');
    expect(tokens).toContain('기술');
  });

  it('returns [] for empty input', async () => {
    const tok = await createTokenizer({ garu });
    expect(tok('')).toEqual([]);
  });

  it('accepts a fieldName argument (per MiniSearch contract)', async () => {
    const tok = await createTokenizer({ garu });
    const tokens = tok('학교', 'title');
    expect(tokens).toContain('학교');
  });

  it('honors a custom posFilter', async () => {
    const tok = await createTokenizer({ garu, posFilter: ['NNG'] });
    const tokens = tok('빠르게 달렸다');
    expect(tokens.some((t) => ['빠르', '달리'].includes(t))).toBe(false);
  });

  it('drops stopwords (post-lowercase)', async () => {
    const tok = await createTokenizer({ garu, stopwords: ['것', '수'] });
    const tokens = tok('할 수 있는 것');
    expect(tokens).not.toContain('수');
    expect(tokens).not.toContain('것');
  });

  it('exposes DEFAULT_POS', () => {
    expect(DEFAULT_POS).toContain('NNG');
    expect(DEFAULT_POS).toContain('VV');
  });
});

describe('createProcessTerm', () => {
  it('lowercases', () => {
    const fn = createProcessTerm();
    expect(fn('AI')).toBe('ai');
  });

  it('drops stopwords by returning null', () => {
    const fn = createProcessTerm(['the', '것']);
    expect(fn('The')).toBeNull();
    expect(fn('것')).toBeNull();
    expect(fn('학교')).toBe('학교');
  });
});
