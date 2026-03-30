"""4개 분석기 F1 비교: Garu, Kiwi, Claude, Mecab vs Gold"""
import json, os, subprocess, sys
from kiwipiepy import Kiwi
import mecab

BASE = os.path.dirname(__file__)
ROOT = os.path.join(BASE, "..", "..")

def load_gold():
    records = []
    with open(os.path.join(BASE, "gold_testset.jsonl")) as f:
        for line in f:
            records.append(json.loads(line))
    return records

def run_garu(texts):
    txt_path = os.path.join(BASE, "_eval_input.txt")
    with open(txt_path, "w") as f:
        for t in texts:
            f.write(t + "\n")
    model = os.path.join(ROOT, "js/models/base.gmdl")
    binary = os.path.join(ROOT, "target/release/examples/analyze_batch")
    result = subprocess.run([binary, txt_path, "--json"],
                          capture_output=True, text=True,
                          env={**os.environ, "GARU_MODEL": model})
    os.remove(txt_path)
    return [json.loads(l) for l in result.stdout.strip().split("\n")]

def run_kiwi(texts):
    kw = Kiwi()
    results = []
    for t in texts:
        r = kw.analyze(t)
        tokens = [[m.form, m.tag.replace("-I","").replace("-R","")] for m in r[0][0]] if r else []
        results.append(tokens)
    return results

def run_mecab(texts):
    mc = mecab.MeCab()
    results = []
    for t in texts:
        tokens = []
        for form, tag in mc.pos(t):
            # Mecab uses + for compound tags like VCP+EC -> split
            if "+" in tag:
                # 복합태그는 첫 번째만 사용 (단순화)
                tokens.append([form, tag.split("+")[0]])
            else:
                tokens.append([form, tag])
        results.append(tokens)
    return results

def compute_f1(pred, gold):
    """Token-level F1 (form+POS pair matching)"""
    tp = fp = fn = 0
    for p_tokens, g_tokens in zip(pred, gold):
        p_set = {}
        for form, pos in p_tokens:
            key = (form, pos)
            p_set[key] = p_set.get(key, 0) + 1
        g_set = {}
        for form, pos in g_tokens:
            key = (form, pos)
            g_set[key] = g_set.get(key, 0) + 1
        all_keys = set(list(p_set.keys()) + list(g_set.keys()))
        for k in all_keys:
            pc = p_set.get(k, 0)
            gc = g_set.get(k, 0)
            matched = min(pc, gc)
            tp += matched
            fp += pc - matched
            fn += gc - matched
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return prec, rec, f1

def main():
    records = load_gold()
    texts = [r["text"] for r in records]
    gold = [r["morphemes"] for r in records]

    print("Running Garu...", flush=True)
    garu = run_garu(texts)
    print("Running Kiwi...", flush=True)
    kiwi = run_kiwi(texts)
    print("Running Mecab...", flush=True)
    mec = run_mecab(texts)
    # Claude는 이미 gold 구축에 사용됐으므로 저장된 결과 없음 -> skip

    print("\n=== F1 Score (vs Gold Testset, 5000 sentences) ===\n")
    print(f"{'Analyzer':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 42)
    for name, pred in [("Garu", garu), ("Kiwi", kiwi), ("Mecab", mec)]:
        p, r, f1 = compute_f1(pred, gold)
        print(f"{name:<10} {p:>10.4f} {r:>10.4f} {f1:>10.4f}")

    # Domain별 F1
    print("\n=== Domain별 Garu F1 ===\n")
    domains = {}
    for i, rec in enumerate(records):
        d = rec["domain"]
        if d not in domains:
            domains[d] = {"pred": [], "gold": []}
        domains[d]["pred"].append(garu[i])
        domains[d]["gold"].append(gold[i])
    for d in ["뉴스", "일상", "SNS", "기술", "문학", "엣지케이스"]:
        if d in domains:
            p, r, f1 = compute_f1(domains[d]["pred"], domains[d]["gold"])
            print(f"  {d:<10} F1={f1:.4f}")

if __name__ == "__main__":
    main()
