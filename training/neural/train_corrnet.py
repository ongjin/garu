"""Train CorrNet - boosting error correction model for CNN2.

Input: syllable(3002,24) + CNN2 predicted label(81,16) + confidence(1) = 41 dims
Architecture: Conv1d(41, 48, k=7) -> ReLU -> FC(48, 81)
"""
import json, struct, gzip, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

DATA_DIR = Path(__file__).parent
ROOT = DATA_DIR.parent.parent
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class CorrNet(nn.Module):
    def __init__(self, syl_vocab=3002, num_labels=81, syl_dim=24, label_dim=16, h=48):
        super().__init__()
        self.syl_emb = nn.Embedding(syl_vocab, syl_dim, padding_idx=0)
        self.label_emb = nn.Embedding(num_labels, label_dim)
        self.conv = nn.Conv1d(syl_dim + label_dim + 1, h, 7, padding=3)
        self.fc = nn.Linear(h, num_labels)
        self.drop = nn.Dropout(0.3)

    def forward(self, syl_ids, cnn2_label_ids, cnn2_confidence):
        se = self.syl_emb(syl_ids)
        le = self.label_emb(cnn2_label_ids)
        cf = cnn2_confidence.unsqueeze(-1)
        x = torch.cat([se, le, cf], -1).transpose(1, 2)
        x = torch.relu(self.conv(x)).transpose(1, 2)
        return self.fc(self.drop(x))


def load_data():
    cache = DATA_DIR / "moe_cache"
    V = json.load(open(DATA_DIR / "vocab.json"))
    s2i = {s: i for i, s in enumerate(V["syllables"])}

    datasets = {}
    for split in ["train", "val"]:
        preds = torch.load(cache / f"{split}_preds.pt", weights_only=True)
        confidence = torch.load(cache / f"{split}_confidence.pt", weights_only=True)
        labels = torch.load(cache / f"{split}_labels.pt", weights_only=True)

        syl_ids_list = []
        with open(DATA_DIR / f"{split}.jsonl") as f:
            for line in f:
                it = json.loads(line)
                sy = it["syllables"][:150]
                si = [s2i.get(s, 1) for s in sy]
                if si:
                    syl_ids_list.append(torch.tensor(si, dtype=torch.long))

        datasets[split] = {
            "syl_ids": syl_ids_list,
            "preds": preds,
            "confidence": confidence,
            "labels": labels,
        }

    return datasets, len(V["labels"])


def train_corrnet(datasets, num_labels, epochs=15, lr=0.002):
    print("\n[Training CorrNet]")
    model = CorrNet(3002, num_labels, 24, 16, 48).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {params:,} ({params // 1024}KB int8)")

    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    train = datasets["train"]
    val = datasets["val"]
    batch_size = 256
    best_net = -999999

    for epoch in range(epochs):
        model.train()
        total_loss = n_batches = 0
        indices = list(range(len(train["syl_ids"])))
        random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            syls = [train["syl_ids"][i] for i in batch_idx]
            preds = [train["preds"][i] for i in batch_idx]
            confs = [train["confidence"][i] for i in batch_idx]
            labs = [train["labels"][i] for i in batch_idx]

            s_pad = pad_sequence(syls, batch_first=True, padding_value=0).to(DEVICE)
            p_pad = pad_sequence(preds, batch_first=True, padding_value=0).to(DEVICE)
            c_pad = pad_sequence(confs, batch_first=True, padding_value=0).to(DEVICE)
            l_pad = pad_sequence(labs, batch_first=True, padding_value=-1).to(DEVICE)

            logits = model(s_pad, p_pad, c_pad)
            loss_flat = criterion(logits.view(-1, num_labels), l_pad.view(-1))
            loss_flat = loss_flat.view(s_pad.size(0), -1)

            is_error = (p_pad != l_pad).float() * (l_pad != -1).float()
            weight = 1.0 + 2.0 * is_error
            valid = (l_pad != -1).float()
            weighted_loss = (loss_flat * weight * valid).sum() / valid.sum().clamp(min=1)

            optimizer.zero_grad()
            weighted_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += weighted_loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        correct = total = cnn2_correct = 0
        corrnet_fixes = corrnet_breaks = 0

        with torch.no_grad():
            for start in range(0, len(val["syl_ids"]), batch_size):
                syls = val["syl_ids"][start:start + batch_size]
                preds_v = val["preds"][start:start + batch_size]
                confs_v = val["confidence"][start:start + batch_size]
                labs_v = val["labels"][start:start + batch_size]

                s_pad = pad_sequence(syls, batch_first=True, padding_value=0).to(DEVICE)
                p_pad = pad_sequence(preds_v, batch_first=True, padding_value=0).to(DEVICE)
                c_pad = pad_sequence(confs_v, batch_first=True, padding_value=0).to(DEVICE)
                l_pad = pad_sequence(labs_v, batch_first=True, padding_value=-1).to(DEVICE)

                logits = model(s_pad, p_pad, c_pad)
                corrnet_preds = logits.argmax(-1)

                mask = (l_pad != -1)
                correct += ((corrnet_preds == l_pad) & mask).sum().item()
                total += mask.sum().item()
                cnn2_correct += ((p_pad == l_pad) & mask).sum().item()

                cnn2_wrong = (p_pad != l_pad) & mask
                cnn2_right = (p_pad == l_pad) & mask
                corrnet_fixes += ((corrnet_preds == l_pad) & cnn2_wrong).sum().item()
                corrnet_breaks += ((corrnet_preds != l_pad) & cnn2_right).sum().item()

        acc = correct / max(total, 1)
        cnn2_acc = cnn2_correct / max(total, 1)
        net = corrnet_fixes - corrnet_breaks

        if net > best_net:
            best_net = net
            torch.save(model.state_dict(), DATA_DIR / "moe_cache" / "corrnet.pt")

        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"  E{epoch+1}: loss={total_loss/n_batches:.4f} "
                  f"acc={acc:.4f}(cnn2={cnn2_acc:.4f}) "
                  f"fixes={corrnet_fixes} breaks={corrnet_breaks} net={net:+d}")

    model.load_state_dict(
        torch.load(DATA_DIR / "moe_cache" / "corrnet.pt", map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"  Best net improvement: {best_net:+d}")
    return model


def export_corrnet(model, num_labels):
    print("\n[Exporting CorrNet]")

    def quantize_tensor(t):
        t_flat = t.detach().cpu().float().flatten()
        max_val = t_flat.abs().max().item()
        scale = max_val / 127.0 if max_val > 0 else 1.0
        quantized = (t_flat / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized.numpy().tobytes(), scale

    buf = bytearray()
    buf.extend(b"COR1")
    buf.append(1)
    buf.extend(struct.pack('<HHH', 24, 16, 48))
    buf.extend(struct.pack('<HH', 3002, num_labels))

    state = model.state_dict()

    data, scale = quantize_tensor(state['syl_emb.weight'])
    buf.extend(struct.pack('<f', scale))
    buf.extend(data)

    data, scale = quantize_tensor(state['label_emb.weight'])
    buf.extend(struct.pack('<f', scale))
    buf.extend(data)

    data, scale = quantize_tensor(state['conv.weight'])
    buf.extend(struct.pack('<f', scale))
    buf.extend(data)
    for v in state['conv.bias'].cpu().numpy():
        buf.extend(struct.pack('<f', float(v)))

    data, scale = quantize_tensor(state['fc.weight'])
    buf.extend(struct.pack('<f', scale))
    buf.extend(data)
    for v in state['fc.bias'].cpu().numpy():
        buf.extend(struct.pack('<f', float(v)))

    raw_size = len(buf)
    compressed = gzip.compress(bytes(buf), compresslevel=9)

    out_path = ROOT / "js" / "models" / "corrnet.bin"
    with open(out_path, 'wb') as f:
        f.write(compressed)

    print(f"  Raw: {raw_size:,} bytes ({raw_size/1024:.1f} KB)")
    print(f"  Gzip: {len(compressed):,} bytes ({len(compressed)/1024:.1f} KB)")


def main():
    print("=" * 60)
    print("  CorrNet (Boosting Error Correction)")
    print("=" * 60)

    datasets, num_labels = load_data()
    print(f"  Train: {len(datasets['train']['syl_ids'])}, Val: {len(datasets['val']['syl_ids'])}")

    model = train_corrnet(datasets, num_labels, epochs=15)
    export_corrnet(model, num_labels)

    params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total: {params:,} params ({params//1024}KB int8)")


if __name__ == "__main__":
    main()
