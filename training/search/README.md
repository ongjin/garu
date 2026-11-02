# Section 18: 구어 입력의 탐색 전 아크 비용 보정

연구 이력 #64를 재구현한 것이다. 래티스 아크마다 품사열·길이·형태소·어절 경계 위치 feature의
선형 비용을 더한 뒤 기존 N-best Viterbi를 돌린다. v2 섹션은 문자 n-gram 구어/문어 판별기를
함께 담아 구어로 판별된 입력에만 보정한다. 2026-11-02 기본 모델에 채택했다(연구 이력 #79).

```sh
LAB=<prepare.py 산출물 폴더>   # training/ambiguity/prepare.py --out LAB --model js/models/base.gmdl
cargo build --release --example search_train --example analyze_batch
# 1. 정답 형태소열과 같은 아크 경로 찾기 → ep_norm으로 후처리 결과 확인
GARU_MODEL=js/models/base.gmdl target/release/examples/search_train gold $LAB/train.jsonl > $LAB/train.gold.jsonl
python3 training/search/verify_gold.py $LAB/train.jsonl $LAB/train.gold.jsonl > $LAB/train.verified.txt
# 2. 전체 경로 averaged perceptron (epoch마다 $LAB/w.e<k>.tsv)
GARU_MODEL=js/models/base.gmdl target/release/examples/search_train train $LAB/train.jsonl $LAB/train.verified.txt $LAB/w 3 42
# 3. 판별기와 함께 내보내기
python3 training/search/router.py $LAB/train.jsonl $LAB/dev.jsonl $LAB/router.npz --dim-log2 15
python3 training/search/export.py $LAB/w.e2.tsv js/models/base.gmdl out.gmdl --scale 0.5 --top 20000 \
    --router $LAB/router.npz --threshold 0 --blob training/codebook_data/search_cost.bin
python3 -m unittest discover -s training/search -p 'test_*.py'
```

epoch·scale·top은 prepare.py dev에서 고른다. 판별기 임계값 0은 dev로 고를 수 없어 미리 고정했다
(dev가 뉴스·구어뿐이고 둘 다 이득). 빌더는 `training/codebook_data/search_cost.bin`이 있으면
Section 18로 기록한다. 채택본은 epoch 2·scale 0.5·상위 20K·판별기 2^15 int8이다. 학습 문장은
`prepare.py` train(285,152)이고, 정답 경로 검증을 통과한 166,039문장만 쓴다. 학습용 `lab_*`
함수는 Section 18을 적용하지 않은 래티스를 쓰므로 배포 모델로 돌려도 같다. 다만 기존 섹션이
바뀐 모델로 다시 학습하면 결과도 달라진다.
