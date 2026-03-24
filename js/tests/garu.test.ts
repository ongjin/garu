import { describe, it, expect } from 'vitest';
import { Garu } from '../src/index';

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
