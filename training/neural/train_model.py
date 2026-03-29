"""Train a tiny 1D CNN + CRF model for Korean syllable-level morpheme tagging.

Architecture:
  - Syllable embedding: 3002 x 32 dim
  - 1D CNN: [kernel3 x 64, kernel5 x 64] -> concat -> 128
  - Dropout -> Linear -> 81 labels
  - CRF layer for sequence decoding

Target: <500KB int8, <1ms per sentence in WASM
"""
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

DATA_DIR = Path(__file__).parent
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")


class SyllableDataset(Dataset):
    def __init__(self, path, syl2idx, label2idx, max_len=150):
        self.data = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                syls = item["syllables"][:max_len]
                labs = item["labels"][:max_len]
                syl_ids = [syl2idx.get(s, 1) for s in syls]
                lab_ids = [label2idx.get(l, 0) for l in labs]
                if syl_ids:
                    self.data.append((
                        torch.tensor(syl_ids, dtype=torch.long),
                        torch.tensor(lab_ids, dtype=torch.long),
                    ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    syls, labs = zip(*batch)
    lengths = torch.tensor([len(s) for s in syls])
    syls_padded = pad_sequence(syls, batch_first=True, padding_value=0)
    labs_padded = pad_sequence(labs, batch_first=True, padding_value=0)
    return syls_padded, labs_padded, lengths


class CRFLayer(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward_score(self, emissions, tags, mask):
        batch_size, seq_len, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        for i in range(1, seq_len):
            cur_mask = mask[:, i].float()
            trans = self.transitions[tags[:, i - 1], tags[:, i]]
            emit = emissions[:, i].gather(1, tags[:, i:i+1]).squeeze(1)
            score = score + (trans + emit) * cur_mask
        last_idx = mask.sum(dim=1).long() - 1
        last_tags = tags.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        return score

    def partition(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        alpha = self.start_transitions + emissions[:, 0]
        for i in range(1, seq_len):
            cur_mask = mask[:, i].float().unsqueeze(1)
            emit = emissions[:, i].unsqueeze(1)
            trans = self.transitions.unsqueeze(0)
            next_alpha = torch.logsumexp(alpha.unsqueeze(2) + trans + emit, dim=1)
            alpha = next_alpha * cur_mask + alpha * (1 - cur_mask)
        return torch.logsumexp(alpha + self.end_transitions, dim=1)

    def loss(self, emissions, tags, mask):
        nll = self.partition(emissions, mask) - self.forward_score(emissions, tags, mask)
        return nll.mean()

    def decode(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        viterbi = self.start_transitions + emissions[:, 0]
        backpointers = []
        for i in range(1, seq_len):
            cur_mask = mask[:, i].float().unsqueeze(1)
            scores = viterbi.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, i].unsqueeze(1)
            max_scores, bp = scores.max(dim=1)
            viterbi = max_scores * cur_mask + viterbi * (1 - cur_mask)
            backpointers.append(bp)
        viterbi = viterbi + self.end_transitions
        best_last = viterbi.argmax(dim=1)
        best_paths = [best_last]
        for bp in reversed(backpointers):
            best_last = bp.gather(1, best_last.unsqueeze(1)).squeeze(1)
            best_paths.append(best_last)
        best_paths.reverse()
        return torch.stack(best_paths, dim=1)


class TinyCNNCRF(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_labels, hidden=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(embed_dim, hidden, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, hidden, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden * 2, num_labels)
        self.crf = CRFLayer(num_labels)

    def get_emissions(self, x):
        emb = self.embedding(x)
        emb_t = emb.transpose(1, 2)
        c3 = torch.relu(self.conv3(emb_t))
        c5 = torch.relu(self.conv5(emb_t))
        cat = torch.cat([c3, c5], dim=1).transpose(1, 2)
        cat = self.dropout(cat)
        return self.fc(cat)

    def loss(self, x, tags, mask):
        emissions = self.get_emissions(x)
        return self.crf.loss(emissions, tags, mask)

    def predict(self, x, mask):
        emissions = self.get_emissions(x)
        return self.crf.decode(emissions, mask)


def compute_accuracy(model, loader):
    model.eval()
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for syls, labs, lengths in loader:
            syls, labs, lengths = syls.to(DEVICE), labs.to(DEVICE), lengths.to(DEVICE)
            mask = (syls != 0)
            preds = model.predict(syls, mask)
            for b in range(syls.size(0)):
                length = lengths[b].item()
                total_correct += (preds[b, :length] == labs[b, :length]).sum().item()
                total_count += length
    return total_correct / max(total_count, 1)


def main():
    vocab = json.load(open(DATA_DIR / "vocab.json"))
    syl2idx = {s: i for i, s in enumerate(vocab["syllables"])}
    label2idx = {l: i for i, l in enumerate(vocab["labels"])}

    vocab_size = len(vocab["syllables"])
    num_labels = len(vocab["labels"])
    embed_dim = 32
    hidden = 64

    print(f"Vocab: {vocab_size} syllables, {num_labels} labels")
    print(f"Model: embed={embed_dim}, hidden={hidden}")

    train_ds = SyllableDataset(DATA_DIR / "train.jsonl", syl2idx, label2idx)
    val_ds = SyllableDataset(DATA_DIR / "val.jsonl", syl2idx, label2idx)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = TinyCNNCRF(vocab_size, embed_dim, num_labels, hidden).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    size_fp32 = n_params * 4 / 1024
    size_int8 = n_params / 1024
    print(f"Parameters: {n_params:,} ({size_fp32:.0f}KB fp32, {size_int8:.0f}KB int8)")

    optimizer = optim.Adam(model.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_acc = 0
    for epoch in range(12):
        model.train()
        total_loss = 0
        n_batches = 0
        t0 = time.time()

        for syls, labs, lengths in train_loader:
            syls, labs, lengths = syls.to(DEVICE), labs.to(DEVICE), lengths.to(DEVICE)
            mask = (syls != 0)
            loss = model.loss(syls, labs, mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        val_acc = compute_accuracy(model, val_loader)
        elapsed = time.time() - t0
        scheduler.step(-val_acc)
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}, time={elapsed:.0f}s, lr={lr:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), DATA_DIR / "model_best.pt")
            print(f"  New best: {best_acc:.4f}")

    print(f"\nBest val accuracy: {best_acc:.4f}")

    info = {
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "hidden": hidden,
        "num_labels": num_labels,
        "n_params": n_params,
        "size_fp32_kb": round(size_fp32),
        "size_int8_kb": round(size_int8),
        "best_val_acc": round(best_acc, 4),
    }
    with open(DATA_DIR / "model_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Model info: {json.dumps(info)}")


if __name__ == "__main__":
    main()
