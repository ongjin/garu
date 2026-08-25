// 누적 스타 차트(SVG) 자가 생성 — 손글씨(Excalidraw) 스타일.
// 2026-06-30 GitHub이 /stargazers API·UI를 소유자/협력자 전용으로 제한해
// star-history.com 등 외부 임베드가 전부 깨짐 → 레포 자신의 GITHUB_TOKEN으로
// 데이터를 받아 정적 SVG를 만들고 star-history 브랜치에 커밋한다(워크플로 참조).
// 웨이블은 시드 고정 PRNG라 데이터가 같으면 출력도 바이트 단위로 같다("updated" 날짜 제외).
// 폰트는 Patrick Hand(OFL) 서브셋(a-z, 0-9, "-. ")을 base64 임베드 — GitHub <img>
// 컨텍스트는 외부 리소스를 못 불러오므로 인라인만 동작한다. 라벨에 새 문자를 쓰면
// 서브셋(patrick-hand-subset.woff2)도 다시 떠야 한다.
// 사용: GITHUB_TOKEN=... node .github/scripts/star-history.mjs <출력디렉토리>

import { readFile, writeFile } from 'node:fs/promises';
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

function mulberry32(seed) {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const W = 840;
const H = 420;
const PAD = { top: 32, right: 84, bottom: 52, left: 64 };
const WOBBLE = 1.3; // 손그림 지터 진폭(px)
const SEG = 24; // 직선을 웨이블 주기로 쪼개는 간격(px)

const THEMES = {
  light: { surface: '#ffffff', accent: '#2563eb', ink: '#1f2328', muted: '#59636e' },
  dark: { surface: '#0d1117', accent: '#6690f0', ink: '#e6edf3', muted: '#9198a1' },
};

function renderSvg(points, theme, fontB64) {
  const t = THEMES[theme];
  const rand = mulberry32(20260330); // 첫 스타 날짜 시드 — 실행마다 동일한 웨이블
  const jitter = () => (rand() - 0.5) * 2 * WOBBLE;

  const [x0, x1] = [points[0].time, points[points.length - 1].time];
  const total = points[points.length - 1].count;
  const yStep = niceStep(total / 4);
  const yMax = Math.ceil(total / yStep) * yStep;
  const px = (time) => PAD.left + ((time - x0) / (x1 - x0)) * (W - PAD.left - PAD.right);
  const py = (count) => H - PAD.bottom - (count / yMax) * (H - PAD.top - PAD.bottom);

  // rough.js 풍 이중 스트로크: 같은 꼭짓점 열을 독립 지터로 두 번 그린다
  const wobblePath = (pts) =>
    pts.map((p, i) => `${i ? 'L' : 'M'}${(p[0] + jitter()).toFixed(1)},${(p[1] + jitter()).toFixed(1)}`).join('');
  const rough = (pts, stroke, w) =>
    `<path d="${wobblePath(pts)}" fill="none" stroke="${stroke}" stroke-width="${w}" stroke-linecap="round" stroke-linejoin="round"/>` +
    `<path d="${wobblePath(pts)}" fill="none" stroke="${stroke}" stroke-width="${w * 0.7}" opacity="0.5" stroke-linecap="round" stroke-linejoin="round"/>`;
  const seg = (a, b) => {
    const n = Math.max(2, Math.round(Math.hypot(b[0] - a[0], b[1] - a[1]) / SEG));
    return Array.from({ length: n + 1 }, (_, i) => [a[0] + ((b[0] - a[0]) / n) * i, a[1] + ((b[1] - a[1]) / n) * i]);
  };

  // 데이터 곡선: x축 6px 간격으로 리샘플(선형 보간) 후 웨이블
  const curve = [];
  let di = 0;
  for (let x = px(x0); x <= px(x1); x += 6) {
    while (di < points.length - 1 && px(points[di + 1].time) < x) di++;
    const [a, b] = [points[di], points[Math.min(di + 1, points.length - 1)]];
    const [ax, bx] = [px(a.time), px(b.time)];
    const c = bx === ax ? b.count : a.count + ((b.count - a.count) * (x - ax)) / (bx - ax);
    curve.push([x, py(c)]);
  }
  curve.push([px(x1), py(total)]);

  const axisY = py(0);
  const ticks = [];
  for (let v = 0; v <= yMax; v += yStep) {
    ticks.push(rough(seg([PAD.left - 6, py(v)], [PAD.left + 2, py(v)]), t.muted, 1.6));
    ticks.push(`<text x="${PAD.left - 12}" y="${py(v) + 6}" text-anchor="end" fill="${t.muted}">${v}</text>`);
  }
  const X_TICK_COUNT = 4;
  for (let i = 0; i <= X_TICK_COUNT; i++) {
    const time = x0 + ((x1 - x0) / X_TICK_COUNT) * i;
    ticks.push(rough(seg([px(time), axisY - 2], [px(time), axisY + 6]), t.muted, 1.6));
    ticks.push(`<text x="${px(time).toFixed(1)}" y="${axisY + 26}" text-anchor="middle" fill="${t.muted}">${ym(time)}</text>`);
  }

  // 끝점 별 마커 (5꼭지, 바깥 r9 / 안 r4)
  const [ex, ey] = [px(x1), py(total)];
  const starPts = Array.from({ length: 11 }, (_, i) => {
    const r = i % 2 ? 4 : 9;
    const a = -Math.PI / 2 + (Math.PI / 5) * i;
    return [ex + r * Math.cos(a) + jitter() * 0.5, ey + r * Math.sin(a) + jitter() * 0.5];
  });
  const starD = `${starPts.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join('')}Z`;

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="GitHub stars over time: ${total}">
<defs><style>
@font-face{font-family:'PatrickHand';src:url(data:font/woff2;base64,${fontB64}) format('woff2');}
text{font-family:'PatrickHand','Comic Sans MS','Segoe Print',cursive;}
</style></defs>
<rect width="${W}" height="${H}" fill="${t.surface}"/>
<g font-size="17">
${ticks.join('\n')}
${rough(seg([PAD.left, PAD.top - 8], [PAD.left, axisY]), t.muted, 1.6)}
${rough(seg([PAD.left, axisY], [W - PAD.right + 24, axisY]), t.muted, 1.6)}
${rough(curve, t.accent, 2.4)}
<path d="${starD}" fill="${t.accent}" stroke="${t.surface}" stroke-width="2"/>
<text x="${(ex + 16).toFixed(1)}" y="${(ey + 8).toFixed(1)}" fill="${t.ink}" font-size="27" transform="rotate(-4 ${ex + 16} ${ey + 8})">${total}</text>
<text x="${W - PAD.right + 24}" y="${H - 12}" text-anchor="end" fill="${t.muted}" font-size="14">updated ${ymd(Date.now())}</text>
</g>
</svg>
`;
}

const times = await fetchStarTimes();
if (times.length === 0) {
  console.error('스타 0개 — 차트 생성 스킵');
  process.exit(0);
}

const fontB64 = (await readFile(new URL('./patrick-hand-subset.woff2', import.meta.url))).toString('base64');
const points = times.map((time, i) => ({ time, count: i + 1 }));
points.push({ time: Date.now(), count: times.length });

for (const theme of Object.keys(THEMES)) {
  const file = path.join(OUT_DIR, `star-history-${theme}.svg`);
  await writeFile(file, renderSvg(points, theme, fontB64));
  console.log(`${file} (${times.length} stars)`);
}
