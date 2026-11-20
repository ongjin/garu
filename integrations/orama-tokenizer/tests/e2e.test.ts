import { describe, it, expect, beforeAll } from 'vitest';
import { create, insert, search } from '@orama/orama';
import { Garu } from 'garu-ko';
import { createTokenizer } from '../src/index';

let garu: Garu;

beforeAll(async () => {
  garu = await Garu.load();
}, 30_000);

describe('Orama end-to-end with garu-orama-tokenizer', () => {
  it('matches composed and decomposed Hangul in either direction', async () => {
    const tokenizer = await createTokenizer({ garu });
    const db = create({ schema: { title: 'string' }, components: { tokenizer } });
    await insert(db, { id: 'nfc', title: '학교에서 만났다' });
    await insert(db, { id: 'nfd', title: '학교를 찾았다'.normalize('NFD') });
    await insert(db, { id: 'other', title: '집에 갔다' });
    for (const term of ['학교', '학교'.normalize('NFD')]) {
      const res = await search(db, { term });
      expect(res.hits.map(hit => hit.id).sort()).toEqual(['nfc', 'nfd']);
    }
  });

  it('retrieves model numbers across Korean spacing and particles', async () => {
    const tokenizer = await createTokenizer({ garu });
    const db = create({ schema: { title: 'string' }, components: { tokenizer } });
    await insert(db, { id: 's24', title: '갤럭시S24를 샀다' });
    await insert(db, { id: 's25', title: '갤럭시 S25를 샀다' });

    for (const [term, id] of [
      ['S24', 's24'], ['갤럭시 S24', 's24'],
      ['S25', 's25'], ['갤럭시S25', 's25'],
    ]) {
      const res = await search(db, { term, threshold: 0 });
      expect(res.hits.map(hit => hit.id)).toEqual([id]);
    }
    expect((await search(db, { term: '갤럭시S2' })).count).toBe(0);
  });

  it('indexes the distinct language names for Orama search', async () => {
    const tokenizer = await createTokenizer({ garu });
    const db = create({ schema: { title: 'string' }, components: { tokenizer } });
    await insert(db, { id: 'c', title: 'C로 작성했다' });
    await insert(db, { id: 'cpp', title: 'C++로 작성했다' });
    await insert(db, { id: 'cs', title: 'C#으로 작성했다' });
    // Orama prefix-matches by default. Use its case-sensitive exact mode for C.
    for (const [term, id] of [['C', 'c'], ['c++', 'cpp'], ['c#', 'cs']]) {
      const res = await search(db, { term, exact: term === 'C' });
      expect(res.hits.map(hit => hit.id)).toEqual([id]);
    }
  });

  it('matches inflected verbs through their stem', async () => {
    const tokenizer = await createTokenizer({ garu });
    const db = create({
      schema: { title: 'string' },
      components: { tokenizer },
    });

    await insert(db, { title: '학교에서 점심을 먹었다' });
    await insert(db, { title: '오늘 뭐 먹지' });
    await insert(db, { title: '날씨가 좋다' });

    const res = await search(db, { term: '먹다' });
    expect(res.count).toBe(2);
  });

  it('matches noun across particle variants', async () => {
    const tokenizer = await createTokenizer({ garu });
    const db = create({
      schema: { title: 'string' },
      components: { tokenizer },
    });

    await insert(db, { title: '학교에 갔다' });
    await insert(db, { title: '학교를 좋아한다' });
    await insert(db, { title: '집에 있다' });

    const res = await search(db, { term: '학교' });
    expect(res.count).toBe(2);
  });

  it('case-insensitive English mixed with Korean', async () => {
    const tokenizer = await createTokenizer({ garu });
    const db = create({
      schema: { title: 'string' },
      components: { tokenizer },
    });

    await insert(db, { title: 'AI 기술의 발전' });
    await insert(db, { title: '그냥 평범한 글' });

    const res = await search(db, { term: 'ai' });
    expect(res.count).toBe(1);
  });
});
