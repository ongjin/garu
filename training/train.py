"""Training script for BiLSTM+CRF Korean morphological analyzer.

Reads JSONL data produced by preprocess.py, trains the model, and saves
the best checkpoint along with a config.json.
"""

import json
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from model import BiLstmCrf
from preprocess import POS_TAGS, VOCAB_SIZE

# ---------------------------------------------------------------------------
# BIO label set: B-TAG / I-TAG for each POS, plus O
# ---------------------------------------------------------------------------

BIO_LABELS: List[str] = []
for _tag in POS_TAGS:
    BIO_LABELS.append(f"B-{_tag}")
    BIO_LABELS.append(f"I-{_tag}")
BIO_LABELS.append("O")

LABEL_TO_IDX: Dict[str, int] = {lbl: idx for idx, lbl in enumerate(BIO_LABELS)}
NUM_TAGS: int = len(BIO_LABELS)  # 39*2 + 1 = 79

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MorphDataset(Dataset):
    """Dataset reading JSONL files produced by preprocess.py.

    Each line: {"ids": [int, ...], "labels": ["B-NNG", ...]}
    """

    def __init__(self, path: str) -> None:
        self.samples: List[Tuple[List[int], List[int]]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                ids = obj["ids"]
                labels = [LABEL_TO_IDX.get(lbl, LABEL_TO_IDX["O"]) for lbl in obj["labels"]]
                self.samples.append((ids, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Collate function: pads to max length in batch
# ---------------------------------------------------------------------------


def collate_fn(
    batch: List[Tuple[List[int], List[int]]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad sequences and create mask.

    Returns:
        ids:   LongTensor [batch, max_len]
        tags:  LongTensor [batch, max_len]
        mask:  BoolTensor [batch, max_len]
    """
    ids_list, tags_list = zip(*batch)
    max_len = max(len(s) for s in ids_list)

    padded_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_tags = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, (ids, tags) in enumerate(zip(ids_list, tags_list)):
        length = len(ids)
        padded_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
        padded_tags[i, :length] = torch.tensor(tags, dtype=torch.long)
        mask[i, :length] = True

    return padded_ids, padded_tags, mask


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    data_path: str,
    output_dir: str = "checkpoints",
    embed_dim: int = 64,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    batch_size: int = 32,
    epochs: int = 30,
    lr: float = 1e-3,
    clip: float = 5.0,
    split_ratio: float = 0.9,
    seed: int = 42,
) -> None:
    """Train the BiLSTM+CRF model.

    Args:
        data_path: Path to JSONL training data.
        output_dir: Directory to save model checkpoint and config.
        embed_dim: Embedding dimension.
        hidden_size: LSTM hidden size per direction.
        num_layers: Number of stacked BiLSTM layers.
        dropout: Dropout rate.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        lr: Learning rate for Adam.
        clip: Gradient clipping max norm.
        split_ratio: Fraction of data for training (rest is validation).
        seed: Random seed.
    """
    torch.manual_seed(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    dataset = MorphDataset(data_path)
    print(f"Loaded {len(dataset)} samples")

    # 90/10 split
    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Model
    model = BiLstmCrf(
        vocab_size=VOCAB_SIZE,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        num_tags=NUM_TAGS,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0
        t0 = time.time()

        for ids, tags, mask in train_loader:
            ids = ids.to(device)
            tags = tags.to(device)
            mask = mask.to(device)

            emissions = model.forward_emissions(ids)
            loss = model.crf_loss(emissions, tags, mask)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1

        avg_train = total_train_loss / max(num_train_batches, 1)
        train_time = time.time() - t0

        # --- Validate ---
        model.set_to_test_mode = False  # use torch eval mode below
        model.train(False)
        total_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for ids, tags, mask in val_loader:
                ids = ids.to(device)
                tags = tags.to(device)
                mask = mask.to(device)

                emissions = model.forward_emissions(ids)
                loss = model.crf_loss(emissions, tags, mask)
                total_val_loss += loss.item()
                num_val_batches += 1

        avg_val = total_val_loss / max(num_val_batches, 1)

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={avg_train:.4f}  "
            f"val_loss={avg_val:.4f}  "
            f"time={train_time:.1f}s"
        )

        # Save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            ckpt_path = os.path.join(output_dir, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved best model (epoch {epoch})")

    # Save config
    config = {
        "vocab_size": VOCAB_SIZE,
        "embed_dim": embed_dim,
        "hidden_size": hidden_size,
        "num_tags": NUM_TAGS,
        "num_layers": num_layers,
        "dropout": dropout,
        "bio_labels": BIO_LABELS,
        "pos_tags": POS_TAGS,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"Config saved to {config_path}")
    print(f"Best model at epoch {best_epoch} with val_loss={best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <data.jsonl> [output_dir]")
        sys.exit(1)

    data_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "checkpoints"
    train(data_path, output_dir)
