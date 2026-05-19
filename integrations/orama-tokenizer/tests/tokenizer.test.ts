import { describe, it, expect, beforeAll } from 'vitest';
import { Garu } from 'garu-ko';
import { createTokenizer, DEFAULT_POS } from '../src/index';

let garu: Garu;

beforeAll(async () => {
  garu = await Garu.load();
}, 30_000);

describe('createTokenizer (Orama)', () => {
  it('returns the shape Orama expects', async () => {
    const tok = await createTokenizer({ garu });
    expect(tok.language).toBe('korean');
    expect(tok.normalizationCache).toBeInstanceOf(Map);
    expect(typeof tok.tokenize).toBe('function');
  });

  it('drops particles and endings from a typical sentence', async () => {
    const tok = await createTokenizer({ garu });
    const tokens = tok.tokenize('나는 학교에 갔다');
    expect(tokens).toContain('학교');
    expect(tokens).not.toContain('는');
    expect(tokens).not.toContain('에');
    expect(tokens).not.toContain('다');
  });

  it('keeps verb and adjective stems', async () => {
    const tok = await createTokenizer({ garu });
    const tokens = tok.tokenize('아주 빠르게 달렸다');
    expect(tokens).toEqual(expect.arrayContaining(['빠르', '달리']));
  });

  it('keeps foreign words (SL) when present', async () => {
    const tok = await createTokenizer({ garu });
    const tokens = tok.tokenize('AI 기술이 발전했다');
    expect(tokens).toContain('ai');
    expect(tokens).toContain('기술');
    expect(tokens).toContain('발전');
  });

  it('returns [] for empty input', async () => {
    const tok = await createTokenizer({ garu });
    expect(tok.tokenize('')).toEqual([]);
  });

  it('honors a custom posFilter', async () => {
    const tok = await createTokenizer({ garu, posFilter: ['NNG'] });
    const tokens = tok.tokenize('아주 빠르게 달렸다');
    expect(tokens.some((t) => ['빠르', '달리'].includes(t))).toBe(false);
  });

  it('drops stopwords', async () => {
    const tok = await createTokenizer({
      garu,
      stopwords: ['것', '수'],
    });
    const tokens = tok.tokenize('할 수 있는 것');
    expect(tokens).not.toContain('수');
    expect(tokens).not.toContain('것');
  });

  it('exposes DEFAULT_POS', () => {
    expect(DEFAULT_POS).toContain('NNG');
    expect(DEFAULT_POS).toContain('VV');
  });
});
