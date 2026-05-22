#!/usr/bin/env bash
# Phase α 가설 검증: v2 풀 (news+blog+wiki 50K)로 layer 재실행
# 순서: L2-NNG → L3 → L4 → L5 → Wiki-NNG-strict → (통과 시) Wiki-NNG-3of4

set -uo pipefail
cd "$(dirname "$0")/../.."

export GARU_POOL_PATH="$PWD/training/codebook_data/candidates_pool_v2.jsonl"
PY=/opt/homebrew/bin/python3.14

echo "=== POOL: $GARU_POOL_PATH ==="

run_layer() {
  local script=$1
  echo
  echo "======================================================"
  echo "### Running: $script ###"
  echo "======================================================"
  $PY "training/layer_gates/$script" 2>&1
  return $?
}

run_layer run_l2_news_nng.py
run_layer run_l3_guueh.py
run_layer run_l4_vv_va.py
run_layer run_l5_compound.py
run_layer run_wiki_nng_strict.py
strict_rc=$?

# Strict 결과만 보고 3/4 시도 여부 결정.
# run_layer 자체는 항상 exit 0 (게이트 fail은 stdout으로 보고).
# strict 통과 여부는 사용자가 stdout 확인 후 별도 결정 → 자동 ramp-up은 skip하고
# 항상 3/4도 실행 (fail이면 자동 revert).
run_layer run_wiki_nng_3of4.py

echo
echo "=== Phase α complete ==="
echo "Log: training/dict_expansion_log.tsv (최근 6 entries)"
tail -6 training/dict_expansion_log.tsv
