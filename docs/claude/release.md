> **언제 읽나**: garu-ko를 X.X.X로 npm 배포할 때, 또는 통합 패키지(orama/minisearch tokenizer)를 garu-ko 마이너 bump에 맞춰 동기화할 때. 순서를 절대 바꾸지 말 것.

# 배포 전 검증 + 수치 동기화 (0단계, 선행 필수)

릴리스 절차 시작 전에 **항상**:

```bash
# (a) Rust 테스트 + 골드 F1 측정 (회귀 확인 — 직전 배포 대비 떨어지면 중단)
cargo test
(cd training/gold_testset && python3 eval_f1.py --analyzers garu)   # norm=True 헤드라인
(cd training/gold_testset && python3 eval_f1.py --analyzers garu --no-norm)  # raw 참고
```

F1 또는 모델 크기가 바뀌었으면 **하드코딩된 수치를 전부 손으로 갱신**한다 (자동 연동 아님 — 한 곳만 고치면 불일치 남음). `modelInfo().size`는 로드된 실제 바이트라 동적이지만, `accuracy`와 README/CLAUDE 헤드라인은 수동:

| 위치 | 갱신할 값 |
|------|-----------|
| `js/src/core.ts` `modelInfo()` | `accuracy:` (F1, 0~1 소수) |
| `README.md` | F1 (2곳: 헤드라인 + 정확도 blockquote), 모델 크기 표 |
| `js/README.md` | F1 (헤드라인) |
| `CLAUDE.md` 프로젝트 개요 | F1, 모델 크기 |
| `js/CHANGELOG.md` 해당 X.X.X 섹션 | base.gmdl 바이트 수, F1 |

> canonical 값: **F1 93.7% (modelInfo `accuracy` 0.937)**, NIKL MP 93.7%. 위 위치들을 항상 이 값으로 동기화.

# 릴리스 절차 (X.X.X 배포 시 항상 풀세트)

한 단계라도 빠지면 WASM 버전 불일치 / npm-GitHub 불일치 / latest 표시 깨짐 사고가 난다.

```bash
# 1. Cargo 버전 (core + wasm 동시)
sed -i '' 's/^version = ".*"/version = "X.X.X"/' crates/garu-core/Cargo.toml crates/garu-wasm/Cargo.toml

# 2. WASM 리빌드 (env!("CARGO_PKG_VERSION") 새 버전 박힘)
wasm-pack build crates/garu-wasm --target web --out-dir ../../js/pkg

# 3. TypeScript 빌드
(cd js && npx tsc)

# 4. js/CHANGELOG.md 항목 추가 (## X.X.X 섹션)

# 5. npm version (자동으로 git tag vX.X.X 생성 — --no-git-tag-version 쓰지 말 것)
(cd js && npm version X.X.X)

# 6. 커밋 + push (코드 + 태그)
git add -A && git commit -m "feat: ..., bump to X.X.X"
git push origin main
git push origin vX.X.X

# 7. npm publish
(cd js && npm publish)

# 8. GitHub Release 생성 (CHANGELOG.md 해당 섹션 그대로, --latest 명시)
gh release create vX.X.X --title "vX.X.X" --latest --notes-file <(awk '/^## X.X.X$/{flag=1; next} /^## /{flag=0} flag' js/CHANGELOG.md)
```

주의:
- 코드블록은 `cat <<'EOF'` heredoc 안에서 백틱을 그대로 사용. 이스케이프(`\``) 금지 — single-quote heredoc은 `\` 를 문자 그대로 받아 화면에 노출됨.
- 옛 release를 일괄 추가할 때는 새 버전을 제외한 옛 버전들에 `--latest=false` 명시.

## 통합 패키지 동기화 (garu-ko가 마이너 이상 bump 시 반드시 같이)

`integrations/orama-tokenizer`, `integrations/minisearch-tokenizer`는 garu-ko를 dep로 끌어 씀. SemVer caret(`^x.y.z`)은 같은 마이너 안에서만 매치되므로, **garu-ko 마이너 이상이 올라가면 통합 패키지 dep도 함께 갱신 + minor bump + republish 해야 함.** 안 하면 통합 패키지 신규 사용자가 옛 garu-ko를 받아 모든 인가 픽스/진입점 분리 혜택을 못 받음.

릴리스 절차 8단계 끝난 뒤 이어서:

```bash
# 9. 두 통합 패키지의 garu-ko dep 동시 업데이트 (정규식으로 caret 부분만 교체)
sed -i '' 's/"garu-ko": "\^[0-9.]*"/"garu-ko": "^X.X.X"/' \
  integrations/orama-tokenizer/package.json \
  integrations/minisearch-tokenizer/package.json

# 10. 워크스페이스 lockfile 재계산
npm install

# 11. 통합 패키지 자체 minor bump (e.g. 0.1.0 → 0.2.0)
#     package.json의 "version" 필드 직접 수정 (npm version 명령은 워크스페이스에서
#     태그 생성 동작이 헷갈리므로 사용 안 함). garu-ko의 X.X.X와 별개 버전임에 주의.

# 12. 두 통합 패키지 클린 리빌드 + 테스트
(cd integrations/orama-tokenizer && rm -rf dist && npx tsc && npx vitest run)
(cd integrations/minisearch-tokenizer && rm -rf dist && npx tsc && npx vitest run)

# 13. 커밋 + push
git add integrations/*/package.json package-lock.json
git commit -m "chore: 통합 패키지 garu-ko dep 동기화 + Y.Y.Y bump"
git push origin main

# 14. npm publish 두 번
(cd integrations/orama-tokenizer && npm publish)
(cd integrations/minisearch-tokenizer && npm publish)
```

주의:
- 통합 패키지는 별도 버전 트랙(0.1, 0.2, …). garu-ko의 X.X.X와 일치시킬 필요 없음.
- npm 태그(`vX.X.X` 같은)는 garu-ko 본체만 만듦. 통합 패키지에는 별도 git 태그 안 만들어도 무방 (혹시 추적 필요하면 `garu-orama-tokenizer@Y.Y.Y` 형태로 수동 부여).
- patch 수준 bump(예: 0.6.11 → 0.6.12)에선 `^0.6.x`가 자동 매치되므로 통합 패키지 republish 불필요. **마이너 이상 bump일 때만** 본 절차 실행.
