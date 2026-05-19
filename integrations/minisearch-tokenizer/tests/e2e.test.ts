import { describe, it, expect, beforeAll } from 'vitest';
import MiniSearch from 'minisearch';
import { Garu } from 'garu-ko';
import { createTokenizer } from '../src/index';

let garu: Garu;

beforeAll(async () => {
  garu = await Garu.load();
}, 30_000);

describe('MiniSearch end-to-end with garu-minisearch-tokenizer', () => {
  it('matches inflected verbs through their stem', async () => {
    const tokenize = await createTokenizer({ garu });
    const ms = new MiniSearch({ fields: ['title'], tokenize });

    ms.addAll([
      { id: 1, title: '학교에서 점심을 먹었다' },
      { id: 2, title: '오늘 뭐 먹지' },
      { id: 3, title: '날씨가 좋다' },
    ]);

    const res = ms.search('먹다');
    expect(res.length).toBe(2);
  });

  it('matches noun across particle variants', async () => {
    const tokenize = await createTokenizer({ garu });
    const ms = new MiniSearch({ fields: ['title'], tokenize });

    ms.addAll([
      { id: 1, title: '학교에 갔다' },
      { id: 2, title: '학교를 좋아한다' },
      { id: 3, title: '집에 있다' },
    ]);

    const res = ms.search('학교');
    expect(res.length).toBe(2);
  });

  it('case-insensitive English mixed with Korean', async () => {
    const tokenize = await createTokenizer({ garu });
    const ms = new MiniSearch({ fields: ['title'], tokenize });

    ms.addAll([
      { id: 1, title: 'AI 기술의 발전' },
      { id: 2, title: '그냥 평범한 글' },
    ]);

    const res = ms.search('ai');
    expect(res.length).toBe(1);
  });
});
