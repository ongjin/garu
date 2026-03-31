/**
 * Text normalization utilities for Garu Korean morphological analyzer.
 * Pre-processes informal/non-standard Korean text before analysis.
 */

export interface NormalizeOptions {
  /** Collapse repeated jamo (ㅋㅋㅋ→ㅋㅋ). Default: true */
  jamo?: boolean;
  /** Expand 2-char jamo abbreviations (ㄱㅅ→감사). Default: false */
  jamoAbbrev?: boolean;
  /** Normalize common dialect stems (워디→어디). Default: false */
  dialect?: boolean;
}

const COMPAT_JAMO_RE = /([\u3131-\u3163])\1{2,}/g;

const JAMO_ABBREV_MAP: Record<string, string> = {
  'ㄱㅅ': '감사',
  'ㅊㅋ': '축하',
  'ㄴㄴ': '노노',
  'ㅂㅂ': '바이바이',
  'ㅈㅅ': '죄송',
  'ㄱㅊ': '괜찮',
  'ㅇㅈ': '인정',
  'ㅎㅇ': '하이',
  'ㄹㅇ': '리얼',
  'ㅇㅋ': '오케이',
  'ㅁㄹ': '모름',
  'ㅅㄱ': '수고',
  'ㅊㅊ': '추천',
  'ㅍㅌ': '파이팅',
  'ㄷㄷ': '덜덜',
};

const DIALECT_MAP: Record<string, string> = {
  '워디': '어디',
  '와케': '왜이렇게',
  '우예': '어떻게',
  '긍게': '그러니까',
  '워메': '어머',
  '가유': '가요',
  '허유': '해요',
};

export function normalizeText(text: string, opts: NormalizeOptions = {}): string {
  const { jamo = true, jamoAbbrev = false, dialect = false } = opts;

  let result = text;

  if (jamo) {
    result = result.replace(COMPAT_JAMO_RE, '$1$1');
  }

  if (jamoAbbrev) {
    for (const [abbrev, expansion] of Object.entries(JAMO_ABBREV_MAP)) {
      const re = new RegExp(`(?<![가-힣ㄱ-ㅣ])${abbrev}(?![가-힣ㄱ-ㅣ])`, 'g');
      result = result.replace(re, expansion);
    }
  }

  if (dialect) {
    for (const [dialectForm, standard] of Object.entries(DIALECT_MAP)) {
      result = result.replace(new RegExp(dialectForm, 'g'), standard);
    }
  }

  return result;
}

export interface Segment {
  text: string;
  /** Character offset of this segment in the original text */
  offset: number;
}

export function splitSentences(text: string): Segment[] {
  const segments: Segment[] = [];
  const re = /([.!?…]+)\s+(?=[가-힣A-Z"'([{])/g;

  let lastCharOffset = 0;
  let lastByteIdx = 0;
  let match: RegExpExecArray | null;

  while ((match = re.exec(text)) !== null) {
    const segText = text.slice(lastByteIdx, match.index + match[1].length).trim();
    if (segText.length > 0) {
      segments.push({ text: segText, offset: lastCharOffset });
    }
    const newStart = match.index + match[0].length;
    lastCharOffset += [...text.slice(lastByteIdx, newStart)].length;
    lastByteIdx = newStart;
  }

  const last = text.slice(lastByteIdx).trim();
  if (last.length > 0) {
    segments.push({ text: last, offset: lastCharOffset });
  }

  return segments.length > 0 ? segments : [{ text: text.trim(), offset: 0 }];
}
