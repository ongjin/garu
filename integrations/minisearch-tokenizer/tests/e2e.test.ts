import { describe, it, expect, beforeAll } from 'vitest';
import MiniSearch from 'minisearch';
import { Garu } from 'garu-ko';
import { createTokenizer } from '../src/index';

let garu: Garu;

beforeAll(async () => {
  garu = await Garu.load();
}, 30_000);

describe('MiniSearch end-to-end with garu-minisearch-tokenizer', () => {
  it('matches composed and decomposed Hangul in either direction', async () => {
    const tokenize = await createTokenizer({ garu });
    const ms = new MiniSearch({ fields: ['title'], tokenize });
    ms.addAll([
      { id: 'nfc', title: '학교에서 만났다' },
      { id: 'nfd', title: '학교를 찾았다'.normalize('NFD') },
      { id: 'other', title: '집에 갔다' },
    ]);
    for (const query of ['학교', '학교'.normalize('NFD')]) {
      expect(ms.search(query).map(hit => hit.id).sort()).toEqual(['nfc', 'nfd']);
    }
  });

  it('retrieves model numbers across Korean spacing and particles', async () => {
    const tokenize = await createTokenizer({ garu });
    const ms = new MiniSearch({ fields: ['title'], tokenize });
    ms.addAll([
      { id: 's24', title: '갤럭시S24를 샀다' },
      { id: 's25', title: '갤럭시 S25를 샀다' },
    ]);

    for (const [query, id] of [
      ['S24', 's24'], ['갤럭시 S24', 's24'],
      ['S25', 's25'], ['갤럭시S25', 's25'],
    ]) {
      expect(ms.search(query, { combineWith: 'AND' }).map(hit => hit.id)).toEqual([id]);
    }
    expect(ms.search('갤럭시S2')).toEqual([]);
  });

  it('keeps C, C++ and C# distinguishable in an exact search', async () => {
    const tokenize = await createTokenizer({ garu });
    const ms = new MiniSearch({ fields: ['title'], tokenize });
    ms.addAll([
      { id: 'c', title: 'C로 작성했다' },
      { id: 'cpp', title: 'C++로 작성했다' },
      { id: 'cs', title: 'C#으로 작성했다' },
    ]);
    for (const [query, id] of [['c', 'c'], ['c++', 'cpp'], ['c#', 'cs']]) {
      expect(ms.search(query).map(hit => hit.id)).toEqual([id]);
    }
  });

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
