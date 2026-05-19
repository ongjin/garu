import { describe, it, expect, beforeAll } from 'vitest';
import { create, insert, search } from '@orama/orama';
import { Garu } from 'garu-ko';
import { createTokenizer } from '../src/index';

let garu: Garu;

beforeAll(async () => {
  garu = await Garu.load();
}, 30_000);

describe('Orama end-to-end with garu-orama-tokenizer', () => {
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
