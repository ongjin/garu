import { describe, it, expect, vi } from 'vitest';
import { Garu } from '../src/node';

describe('Garu.load (node)', () => {
  it('uses context to distinguish buying from living with the bundled model', async () => {
    const garu = await Garu.load();
    try {
      expect(garu.tokenize('이걸 살게요.')).toContain('사');
      expect(garu.tokenize('여기서 가족과 평생 살게요.')).toContain('살');
      garu.addUserWord('살게요', 'NNP', 1_000_000);
      expect(garu.tokenize('이걸 살게요.')).toContain('살');
      garu.addUserWord('살게요', 'NNP', 0xffff_ffff);
      expect(garu.nouns('이걸 살게요.')).toContain('살게요');
      garu.clearUserWords();
      expect(garu.tokenize('이걸 살게요.'.normalize('NFD'))).toContain('사');
    } finally {
      garu.destroy();
    }
  });

  it('defaults to standard promise endings and supports legacy output per instance', async () => {
    const standard = await Garu.load();
    const legacy = await Garu.load({ promiseEndings: 'legacy' });
    try {
      expect(standard.modelInfo().accuracy).toBe(0.9617);
      expect(legacy.modelInfo().accuracy).toBe(0.9618);
      expect(legacy.tokenize('제가 할게요')).toEqual(['제', '가', '하', 'ㄹ게요']);
      expect(standard.tokenize('제가 할게요')).toEqual(['제', '가', '하', 'ㄹ게', '요']);
      expect(standard.tokenize('누울게요')).toEqual(['눕', '을게', '요']);
    } finally {
      legacy.destroy();
      standard.destroy();
    }
  });

  it('initializes WASM without deprecated-parameter warnings', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const garu = await Garu.load();
      expect(garu.isLoaded()).toBe(true);
      expect(warn).not.toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });
});

describe('Garu API surface', () => {
  it('Garu class is defined', () => {
    expect(Garu).toBeDefined();
  });

  it('Garu.load is a static function', () => {
    expect(typeof Garu.load).toBe('function');
  });

  it('prototype has analyze method', () => {
    expect(typeof Garu.prototype.analyze).toBe('function');
  });

  it('prototype has tokenize method', () => {
    expect(typeof Garu.prototype.tokenize).toBe('function');
  });

  it('prototype has isLoaded method', () => {
    expect(typeof Garu.prototype.isLoaded).toBe('function');
  });

  it('prototype has modelInfo method', () => {
    expect(typeof Garu.prototype.modelInfo).toBe('function');
  });

  it('prototype has destroy method', () => {
    expect(typeof Garu.prototype.destroy).toBe('function');
  });
});
