"""Morpheme-level F1 scoring for BiLSTM+CRF Korean morphological analyzer.

Reads JSONL test data, runs the trained model, and reports
precision, recall, and F1 at the morpheme level.
"""

import json
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import torch
from torch.utils.data import DataLoader

from model import BiLstmCrf
from preprocess import VOCAB_SIZE
from train import BIO_LABELS, LABEL_TO_IDX, NUM_TAGS, MorphDataset, collate_fn

# ---------------------------------------------------------------------------
# BIO span extraction
# ---------------------------------------------------------------------------


def extract_morphemes(
    tag_ids: List[int],
) -> Set[Tuple[int, int, str]]:
    """Extract morpheme spans from a BIO tag sequence.

    Returns a set of (start, end, pos_tag) tuples.
    A morpheme span starts at B-TAG and continues through I-TAG.
    """
    morphemes: Set[Tuple[int, int, str]] = set()
    current_start = -1
    current_tag = ""

    for idx, tid in enumerate(tag_ids):
        label = BIO_LABELS[tid] if tid < len(BIO_LABELS) else "O"

        if label.startswith("B-"):
            # Close previous span
            if current_start >= 0:
                morphemes.add((current_start, idx, current_tag))
            current_start = idx
            current_tag = label[2:]
        elif label.startswith("I-"):
            tag = label[2:]
            if current_start >= 0 and tag == current_tag:
                # Continue current span
                pass
            else:
                # Mismatched I-tag: close previous and start new
                if current_start >= 0:
                    morphemes.add((current_start, idx, current_tag))
                current_start = idx
                current_tag = tag
        else:
            # O tag
            if current_start >= 0:
                morphemes.add((current_start, idx, current_tag))
                current_start = -1
                current_tag = ""

    # Close last span
    if current_start >= 0:
        morphemes.add((current_start, len(tag_ids), current_tag))

    return morphemes


# ---------------------------------------------------------------------------
# Main scoring routine
# ---------------------------------------------------------------------------


def run_scoring(
    checkpoint_path: str,
    config_path: str,
    data_path: str,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Score the model on test data and report morpheme-level F1.

    Args:
        checkpoint_path: Path to best_model.pt.
        config_path: Path to config.json.
        data_path: Path to test JSONL data.
        batch_size: Batch size for inference.

    Returns:
        Dictionary with precision, recall, f1 scores.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    vocab_size = config["vocab_size"]
    embed_dim = config["embed_dim"]
    hidden_size = config["hidden_size"]
    num_tags = config["num_tags"]
    num_layers = config["num_layers"]
    dropout = config.get("dropout", 0.3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BiLstmCrf(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        num_tags=num_tags,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.train(False)

    dataset = MorphDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    total_tp = 0
    total_pred = 0
    total_gold = 0

    # Per-POS counts
    per_pos_tp: Dict[str, int] = defaultdict(int)
    per_pos_pred: Dict[str, int] = defaultdict(int)
    per_pos_gold: Dict[str, int] = defaultdict(int)

    with torch.no_grad():
        for ids, tags, mask in loader:
            ids = ids.to(device)
            tags = tags.to(device)
            mask = mask.to(device)

            emissions = model.forward_emissions(ids)
            pred_sequences = model.decode(emissions, mask)

            lengths = mask.long().sum(dim=1).tolist()
            gold_tags_batch = tags.cpu().tolist()

            for b in range(len(pred_sequences)):
                length = lengths[b]
                pred_tags = pred_sequences[b][:length]
                gold_tags_seq = gold_tags_batch[b][:length]

                pred_spans = extract_morphemes(pred_tags)
                gold_spans = extract_morphemes(gold_tags_seq)

                tp = pred_spans & gold_spans
                total_tp += len(tp)
                total_pred += len(pred_spans)
                total_gold += len(gold_spans)

                for start, end, pos in tp:
                    per_pos_tp[pos] += 1
                for start, end, pos in pred_spans:
                    per_pos_pred[pos] += 1
                for start, end, pos in gold_spans:
                    per_pos_gold[pos] += 1

    # Overall metrics
    precision = total_tp / max(total_pred, 1)
    recall = total_tp / max(total_gold, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    print(f"{'='*60}")
    print(f"Morpheme-level Scoring Results")
    print(f"{'='*60}")
    print(f"  Total predictions: {total_pred}")
    print(f"  Total gold:        {total_gold}")
    print(f"  True positives:    {total_tp}")
    print(f"{'='*60}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"{'='*60}")

    # Per-POS breakdown
    print(f"\n{'POS':<8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Pred':>8} {'Gold':>8} {'TP':>8}")
    print("-" * 56)
    all_pos = sorted(set(list(per_pos_pred.keys()) + list(per_pos_gold.keys())))
    for pos in all_pos:
        tp = per_pos_tp.get(pos, 0)
        pred = per_pos_pred.get(pos, 0)
        gold = per_pos_gold.get(pos, 0)
        p = tp / max(pred, 1)
        r = tp / max(gold, 1)
        f = 2 * p * r / max(p + r, 1e-10)
        print(f"{pos:<8} {p:>8.4f} {r:>8.4f} {f:>8.4f} {pred:>8d} {gold:>8d} {tp:>8d}")

    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <best_model.pt> <config.json> <test_data.jsonl>")
        sys.exit(1)
    run_scoring(sys.argv[1], sys.argv[2], sys.argv[3])
