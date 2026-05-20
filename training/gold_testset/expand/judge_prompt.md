당신은 한국어 형태소 분석 검증 작업을 합니다.

세종 42 태그셋:
NNG/NNP/NNB/NR/NP/VV/VA/VX/VCP/VCN/MAG/MAJ/MM/IC/JKS/JKC/JKG/JKO/JKB/JKV/JKQ/JX/JC/EP/EF/EC/ETN/ETM/XPN/XSN/XSV/XSA/XR/SF/SP/SS/SE/SO/SW/SH/SL/SN

자주 헷갈리는 패턴:
- `학생이다` = 학생/NNG + 이/VCP + 다/EF
- `학생이 간다` = 학생/NNG + 이/JKS + 가/VV + ㄴ다/EF
- `공부하다` = 공부/NNG + 하/XSV + 다/EF
- `일을 하다` = 일/NNG + 을/JKO + 하/VV + 다/EF
- `필요하다` = 필요/NNG + 하/XSA + 다/EF
- `먹는 밥` = 먹/VV + 는/ETM + 밥/NNG
- `나는 학생` = 나/NP + 는/JX + 학생/NNG
- `저공해` = 저/XPN + 공해/NNG
- `저 사람` = 저/MM + 사람/NNG

EP 정규화: '했' = 하/XSV + 았/EP, '였' = 이/VCP + 었/EP 분해형으로 통일.

작업: 아래 cases 배열을 받아 각 case에 대해 결정합니다.
- Garu가 맞으면: choice="garu"
- Kiwi가 맞으면: choice="kiwi"
- 둘 다 틀리면: choice="수정", 직접 morphemes 작성

응답 형식: 순수 JSON 배열. 다른 텍스트 일체 없이.
각 항목: `{"id": <int>, "choice": "garu"|"kiwi"|"수정", "morphemes": [["형태소","POS"], ...], "reason": "<한 줄>"}`

입력 cases (JSON):
<INPUT_CASES>
