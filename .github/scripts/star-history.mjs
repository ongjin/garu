// 누적 스타 차트(SVG) 자가 생성.
// 2026-06-30 GitHub이 /stargazers API·UI를 소유자/협력자 전용으로 제한해
// star-history.com 등 외부 임베드가 전부 깨짐 → 레포 자신의 GITHUB_TOKEN으로
// 데이터를 받아 정적 SVG를 만들고 star-history 브랜치에 커밋한다(워크플로 참조).
// 사용: GITHUB_TOKEN=... node .github/scripts/star-history.mjs <출력디렉토리>

import { writeFile } from 'node:fs/promises';
import path from 'node:path';

const TOKEN = process.env.GITHUB_TOKEN;
const REPO = process.env.GITHUB_REPOSITORY ?? 'ongjin/garu';
const OUT_DIR = process.argv[2] ?? '.';
const PER_PAGE = 100;

if (!TOKEN) {
  console.error('GITHUB_TOKEN 이 필요합니다 (stargazers API가 소유자 인증 전용).');
  process.exit(1);
}

async function fetchStarTimes() {
  const times = [];
  for (let page = 1; ; page++) {
    const res = await fetch(
      `https://api.github.com/repos/${REPO}/stargazers?per_page=${PER_PAGE}&page=${page}`,
      {
        headers: {
          accept: 'application/vnd.github.star+json',
          authorization: `Bearer ${TOKEN}`,
          'x-github-api-version': '2022-11-28',
        },
      },
    );
    if (!res.ok) throw new Error(`stargazers ${res.status}: ${await res.text()}`);

    const batch = await res.json();
    times.push(...batch.map((s) => Date.parse(s.starred_at)));
    if (batch.length < PER_PAGE) return times.sort((a, b) => a - b);
  }
}

// 1/2/5 × 10^n 눈금 간격
function niceStep(rough) {
  const pow = 10 ** Math.floor(Math.log10(rough));
  for (const m of [1, 2, 5, 10]) if (m * pow >= rough) return m * pow;
}

const ymd = (t) => new Date(t).toISOString().slice(0, 10);
const ym = (t) => new Date(t).toISOString().slice(0, 7);

const W = 840;
const H = 420;
const PAD = { top: 28, right: 64, bottom: 44, left: 56 };

const THEMES = {
  light: { surface: '#ffffff', accent: '#2563eb', ink: '#1f2328', muted: '#59636e', grid: '#d1d9e0' },
  dark: { surface: '#0d1117', accent: '#6690f0', ink: '#e6edf3', muted: '#9198a1', grid: '#3d444d' },
};

function renderSvg(points, theme) {
  const t = THEMES[theme];
  const [x0, x1] = [points[0].time, points[points.length - 1].time];
  const total = points[points.length - 1].count;

  const yStep = niceStep(total / 4);
  const yMax = Math.ceil(total / yStep) * yStep;
  const px = (time) => PAD.left + ((time - x0) / (x1 - x0)) * (W - PAD.left - PAD.right);
  const py = (count) => H - PAD.bottom - (count / yMax) * (H - PAD.top - PAD.bottom);

  const line = points.map((p, i) => `${i ? 'L' : 'M'}${px(p.time).toFixed(1)},${py(p.count).toFixed(1)}`).join('');
  const area = `${line}L${px(x1).toFixed(1)},${py(0)}L${px(x0).toFixed(1)},${py(0)}Z`;

  const yTicks = [];
  for (let v = 0; v <= yMax; v += yStep) {
    yTicks.push(
      `<line x1="${PAD.left}" y1="${py(v)}" x2="${W - PAD.right}" y2="${py(v)}" stroke="${t.grid}" stroke-width="1" opacity="0.5"/>`,
      `<text x="${PAD.left - 8}" y="${py(v) + 4}" text-anchor="end" fill="${t.muted}">${v}</text>`,
    );
  }

  const X_TICK_COUNT = 4;
  const xTicks = [];
  for (let i = 0; i <= X_TICK_COUNT; i++) {
    const time = x0 + ((x1 - x0) / X_TICK_COUNT) * i;
    xTicks.push(`<text x="${px(time).toFixed(1)}" y="${H - PAD.bottom + 20}" text-anchor="middle" fill="${t.muted}">${ym(time)}</text>`);
  }

  const [endX, endY] = [px(x1), py(total)];
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="GitHub stars over time: ${total}">
<rect width="${W}" height="${H}" fill="${t.surface}"/>
<g font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif" font-size="12">
${yTicks.join('\n')}
${xTicks.join('\n')}
<line x1="${PAD.left}" y1="${py(0)}" x2="${W - PAD.right}" y2="${py(0)}" stroke="${t.muted}" stroke-width="1"/>
<path d="${area}" fill="${t.accent}" opacity="0.08"/>
<path d="${line}" fill="none" stroke="${t.accent}" stroke-width="2" stroke-linejoin="round"/>
<circle cx="${endX}" cy="${endY}" r="4" fill="${t.accent}" stroke="${t.surface}" stroke-width="2"/>
<text x="${endX + 10}" y="${endY + 4}" fill="${t.ink}" font-size="13" font-weight="600">${total}</text>
<text x="${W - PAD.right}" y="${H - 10}" text-anchor="end" fill="${t.muted}" font-size="11">updated ${ymd(Date.now())}</text>
</g>
</svg>
`;
}

const times = await fetchStarTimes();
if (times.length === 0) {
  console.error('스타 0개 — 차트 생성 스킵');
  process.exit(0);
}

const points = times.map((time, i) => ({ time, count: i + 1 }));
points.push({ time: Date.now(), count: times.length });

for (const theme of Object.keys(THEMES)) {
  const file = path.join(OUT_DIR, `star-history-${theme}.svg`);
  await writeFile(file, renderSvg(points, theme));
  console.log(`${file} (${times.length} stars)`);
}
