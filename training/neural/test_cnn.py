"""Test trained CNN+CRF model on ambiguous sentences."""
import json
import torch
from pathlib import Path
from train_model import TinyCNNCRF, collate_fn, SyllableDataset
from torch.utils.data import DataLoader

DATA_DIR = Path(__file__).parent
DEVICE = "cpu"


def load_model():
    vocab = json.load(open(DATA_DIR / "vocab.json"))
    info = json.load(open(DATA_DIR / "model_info.json"))
    syl2idx = {s: i for i, s in enumerate(vocab["syllables"])}
    idx2label = {i: l for i, l in enumerate(vocab["labels"])}

    model = TinyCNNCRF(info["vocab_size"], info["embed_dim"], info["num_labels"], info["hidden"])
    model.load_state_dict(torch.load(DATA_DIR / "model_best.pt", map_location="cpu", weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model, syl2idx, idx2label, info


def predict_sentence(model, syl2idx, idx2label, text):
    syllables = list(text)
    syl_ids = [syl2idx.get(s, 1) for s in syllables]
    x = torch.tensor([syl_ids], dtype=torch.long).to(DEVICE)
    mask = (x != 0)
    with torch.no_grad():
        preds = model.predict(x, mask)
    tags = [idx2label.get(p.item(), 'O') for p in preds[0, :len(syllables)]]
    return syllables, tags


def tags_to_morphemes(syllables, tags):
    morphemes = []
    current_form = ""
    current_pos = None
    for syl, tag in zip(syllables, tags):
        if syl == ' ':
            if current_form and current_pos:
                morphemes.append((current_form, current_pos))
                current_form = ""
                current_pos = None
            continue
        if tag.startswith('B-'):
            if current_form and current_pos:
                morphemes.append((current_form, current_pos))
            current_form = syl
            current_pos = tag[2:]
        elif tag.startswith('I-'):
            current_form += syl
        elif tag == 'O':
            if current_form and current_pos:
                morphemes.append((current_form, current_pos))
            current_form = syl
            current_pos = "SW"
        else:
            current_form += syl
    if current_form and current_pos:
        morphemes.append((current_form, current_pos))
    return morphemes


def main():
    print("Loading CNN+CRF model...")
    model, syl2idx, idx2label, info = load_model()
    print(f"  {info['n_params']:,} params, {info['size_int8_kb']}KB int8, val_acc={info['best_val_acc']}")

    test_cases = [
        ("나는 하늘을 나는 새를 보았다.", "두번째 나는=VV+ETM"),
        ("나는 밥을 먹었다.", "나는=NP+JX"),
        ("내가 한 일이 많다.", "한=하/VV+ㄴ/ETM"),
        ("한 사람이 왔다.", "한=MM"),
        ("먹고 있는 사람이 많다.", "있는=VX+ETM"),
        ("재미있는 영화를 봤다.", "재미있는=VA+ETM"),
        ("나는 나는 것이 무섭다.", "첫NP+JX, 둘째VV+ETM"),
        ("그는 그 사건을 기억한다.", "그=NP, 그=MM"),
        ("문제가 해결되었다.", "되=XSV"),
        ("밥이 되었다.", "되=VV"),
    ]

    print("\n" + "=" * 60)
    print("  CNN+CRF AMBIGUITY TEST")
    print("=" * 60)

    for text, desc in test_cases:
        syllables, tags = predict_sentence(model, syl2idx, idx2label, text)
        morphemes = tags_to_morphemes(syllables, tags)
        morphs_str = "  ".join(f"{f}/{p}" for f, p in morphemes)
        print(f"\n  [{desc}]")
        print(f"  입력: {text}")
        print(f"  CNN:  {morphs_str}")

    # F1 on test set
    print("\n" + "=" * 60)
    print("  CNN+CRF NIKL MP TEST SET F1")
    print("=" * 60)

    vocab = json.load(open(DATA_DIR / "vocab.json"))
    label2idx = {l: i for i, l in enumerate(vocab["labels"])}
    test_ds = SyllableDataset(DATA_DIR / "test.jsonl", syl2idx, label2idx)
    test_data = []
    with open(DATA_DIR / "test.jsonl", encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))

    loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)
    total_match = total_pred = total_gold = 0
    idx = 0

    with torch.no_grad():
        for syls, labs, lengths in loader:
            syls = syls.to(DEVICE)
            mask = (syls != 0)
            preds = model.predict(syls, mask)
            for b in range(syls.size(0)):
                if idx >= len(test_data): break
                length = lengths[b].item()
                pred_tags = [idx2label.get(preds[b, i].item(), 'O') for i in range(length)]
                gold_tags = test_data[idx]["labels"][:length]
                syllables_b = test_data[idx]["syllables"][:length]
                pred_m = set(tuple(m) for m in tags_to_morphemes(syllables_b, pred_tags))
                gold_m = set(tuple(m) for m in tags_to_morphemes(syllables_b, gold_tags))
                total_match += len(pred_m & gold_m)
                total_pred += len(pred_m)
                total_gold += len(gold_m)
                idx += 1

    P = total_match / max(total_pred, 1)
    R = total_match / max(total_gold, 1)
    F = 2 * P * R / max(P + R, 1e-10)
    print(f"  CNN-only: P={P:.4f}  R={R:.4f}  F1={F:.4f}  ({idx} sentences)")


if __name__ == "__main__":
    main()
