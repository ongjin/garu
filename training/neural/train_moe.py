"""Train MoE (Mixture of Experts) CNN system.

Architecture:
  - Base CNN2 (frozen, 192-dim hidden states as input)
  - 4 specialized experts (all output 81 BIO-POS labels):
    E1: NounSeg — Dilated Conv (d=1,2,4), k=3, 48ch
    E2: POSDisamb — BiConv (forward k=5 + backward k=5), 48ch each
    E3: VerbAux — Wide Conv k=9, 64ch
    E4: CharMorph — Jamo decomposition + Conv k=3,5
  - Router: FC(192 -> 4) selects expert per position
  - Confidence gate: positions with CNN2 confidence > threshold use CNN2 directly

Training phases:
  1. Extract CNN2 hidden states for all data
  2. Train each expert on error-filtered data
  3. Train router
  4. Export binary
"""
import json, os, sys, time, struct, gzip, random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

DATA_DIR = Path(__file__).parent
ROOT = DATA_DIR.parent.parent
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Base CNN2 (from experiment_all.py)
# ---------------------------------------------------------------------------

class CNN2(nn.Module):
    def __init__(self, vs, ed, nl, h=96):
        super().__init__()
        self.emb = nn.Embedding(vs, ed, padding_idx=0)
        self.c3 = nn.Conv1d(ed, h, 3, padding=1)
        self.c5 = nn.Conv1d(ed, h, 5, padding=2)
        self.c9 = nn.Conv1d(ed, h, 9, padding=4)
        self.bn = nn.BatchNorm1d(h * 3)
        self.c2a = nn.Conv1d(h * 3, h, 3, padding=1)
        self.c2b = nn.Conv1d(h * 3, h, 7, padding=3)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(h * 2, nl)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        l1 = torch.cat([torch.relu(self.c3(e)), torch.relu(self.c5(e)), torch.relu(self.c9(e))], 1)
        l1 = self.bn(l1)
        l2 = torch.cat([torch.relu(self.c2a(l1)), torch.relu(self.c2b(l1))], 1).transpose(1, 2)
        return self.fc(self.drop(l2))

    def extract_hidden(self, x):
        e = self.emb(x).transpose(1, 2)
        l1 = torch.cat([torch.relu(self.c3(e)), torch.relu(self.c5(e)), torch.relu(self.c9(e))], 1)
        l1 = self.bn(l1)
        l2 = torch.cat([torch.relu(self.c2a(l1)), torch.relu(self.c2b(l1))], 1).transpose(1, 2)
        return l2  # [batch, seq_len, 192]

    def predict_with_confidence(self, x):
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values
        preds = logits.argmax(dim=-1)
        return preds, confidence, logits


# ---------------------------------------------------------------------------
# Expert architectures (input: 192-dim CNN2 hidden, output: 81 labels)
# ---------------------------------------------------------------------------

class ExpertNounSeg(nn.Module):
    """E1: Dilated convolutions for compound noun boundary detection."""
    def __init__(self, in_dim=192, nl=81, h=48):
        super().__init__()
        self.d1 = nn.Conv1d(in_dim, h, 3, padding=1, dilation=1)
        self.d2 = nn.Conv1d(in_dim, h, 3, padding=2, dilation=2)
        self.d4 = nn.Conv1d(in_dim, h, 3, padding=4, dilation=4)
        self.fc = nn.Linear(h * 3, nl)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        xt = x.transpose(1, 2)
        d = torch.cat([
            torch.relu(self.d1(xt)),
            torch.relu(self.d2(xt)),
            torch.relu(self.d4(xt)),
        ], 1).transpose(1, 2)
        return self.fc(self.drop(d))


class ExpertPOSDisamb(nn.Module):
    """E2: Bidirectional conv for POS disambiguation."""
    def __init__(self, in_dim=192, nl=81, h=48):
        super().__init__()
        self.fwd = nn.Conv1d(in_dim, h, 5, padding=2)
        self.bwd = nn.Conv1d(in_dim, h, 5, padding=2)
        self.fc = nn.Linear(h * 2, nl)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        xt = x.transpose(1, 2)
        f_out = torch.relu(self.fwd(xt))
        b_out = torch.relu(self.bwd(torch.flip(xt, [2])))
        b_out = torch.flip(b_out, [2])
        combined = torch.cat([f_out, b_out], 1).transpose(1, 2)
        return self.fc(self.drop(combined))


class ExpertVerbAux(nn.Module):
    """E3: Wide conv for auxiliary verb / ending boundaries."""
    def __init__(self, in_dim=192, nl=81, h=64):
        super().__init__()
        self.wide = nn.Conv1d(in_dim, h, 9, padding=4)
        self.narrow = nn.Conv1d(in_dim, h, 3, padding=1)
        self.fc = nn.Linear(h * 2, nl)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        xt = x.transpose(1, 2)
        w = torch.relu(self.wide(xt))
        n = torch.relu(self.narrow(xt))
        combined = torch.cat([w, n], 1).transpose(1, 2)
        return self.fc(self.drop(combined))


NUM_JAMO = 70  # 19 cho + 21 jung + 28 jong + PAD + UNK


def decompose_syllable(ch):
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        offset = code - 0xAC00
        cho = offset // (21 * 28)
        jung = (offset % (21 * 28)) // 28
        jong = offset % 28
        return (cho + 2, jung + 21, jong + 42 if jong > 0 else 0)
    return (1, 1, 0)


class ExpertCharMorph(nn.Module):
    """E4: Jamo-level conv for OOV/technical term analysis."""
    def __init__(self, in_dim=192, nl=81, jamo_dim=24, h=48):
        super().__init__()
        self.jamo_emb = nn.Embedding(NUM_JAMO, jamo_dim, padding_idx=0)
        self.jamo_conv1 = nn.Conv1d(jamo_dim * 3, h, 3, padding=1)
        self.jamo_conv2 = nn.Conv1d(h, h, 5, padding=2)
        self.context_fc = nn.Linear(in_dim, h)
        self.fc = nn.Linear(h * 2, nl)
        self.drop = nn.Dropout(0.3)

    def forward(self, x, jamo_ids=None):
        batch, seq, _ = x.shape
        if jamo_ids is not None:
            je = self.jamo_emb(jamo_ids)
            je = je.view(batch, seq, -1).transpose(1, 2)
            j1 = torch.relu(self.jamo_conv1(je))
            j2 = torch.relu(self.jamo_conv2(j1)).transpose(1, 2)
        else:
            j2 = torch.zeros(batch, seq, 48, device=x.device)
        ctx = torch.relu(self.context_fc(x))
        combined = torch.cat([j2, ctx], -1)
        return self.fc(self.drop(combined))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class Router(nn.Module):
    def __init__(self, in_dim=192, num_experts=4, hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_experts)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Phase 1: Extract CNN2 hidden states
# ---------------------------------------------------------------------------

def extract_cnn2_features():
    print("\n[Phase 1] Extracting CNN2 hidden states...")
    V = json.load(open(DATA_DIR / "vocab.json"))
    s2i = {s: i for i, s in enumerate(V["syllables"])}
    l2i = {l: i for i, l in enumerate(V["labels"])}
    vs, nl = len(V["syllables"]), len(V["labels"])

    cnn2 = CNN2(vs, 48, nl, 96).to(DEVICE)
    cnn2.load_state_dict(torch.load(DATA_DIR / "cnn2_baseline.pt", map_location=DEVICE, weights_only=True))
    cnn2.eval()

    for split in ["train", "val"]:
        print(f"  Processing {split}...")
        data = []
        with open(DATA_DIR / f"{split}.jsonl") as f:
            for line in f:
                it = json.loads(line)
                sy, la = it["syllables"][:150], it["labels"][:150]
                si = [s2i.get(s, 1) for s in sy]
                li = [l2i.get(l, 0) for l in la]
                if si:
                    data.append((torch.tensor(si, dtype=torch.long),
                                 torch.tensor(li, dtype=torch.long), sy))

        all_hidden, all_labels, all_jamo = [], [], []
        all_confidence, all_preds = [], []
        batch_size = 256

        with torch.no_grad():
            for start in range(0, len(data), batch_size):
                batch = data[start:start + batch_size]
                seqs = [b[0] for b in batch]
                labs = [b[1] for b in batch]
                syls = [b[2] for b in batch]

                x_pad = pad_sequence(seqs, batch_first=True, padding_value=0).to(DEVICE)
                hidden = cnn2.extract_hidden(x_pad)
                preds, confidence, _ = cnn2.predict_with_confidence(x_pad)

                for i in range(len(seqs)):
                    slen = len(seqs[i])
                    all_hidden.append(hidden[i, :slen].cpu())
                    all_labels.append(labs[i])
                    all_confidence.append(confidence[i, :slen].cpu())
                    all_preds.append(preds[i, :slen].cpu())

                    jamo = []
                    for ch in syls[i]:
                        if len(ch) == 1:
                            cho, jung, jong = decompose_syllable(ch)
                            jamo.append([cho, jung, jong])
                        else:
                            jamo.append([1, 1, 0])
                    all_jamo.append(torch.tensor(jamo, dtype=torch.long))

                if (start // batch_size) % 100 == 0:
                    print(f"    {start}/{len(data)}")

        out_dir = DATA_DIR / "moe_cache"
        out_dir.mkdir(exist_ok=True)
        torch.save(all_hidden, out_dir / f"{split}_hidden.pt")
        torch.save(all_labels, out_dir / f"{split}_labels.pt")
        torch.save(all_jamo, out_dir / f"{split}_jamo.pt")
        torch.save(all_confidence, out_dir / f"{split}_confidence.pt")
        torch.save(all_preds, out_dir / f"{split}_preds.pt")
        print(f"    Saved {len(all_hidden)} samples to {out_dir}")


# ---------------------------------------------------------------------------
# Error mask generation
# ---------------------------------------------------------------------------

NOUN_LABELS = {"B-NNG", "I-NNG", "B-NNP", "I-NNP", "B-NNB", "I-NNB",
               "B-XSN", "I-XSN", "B-XPN", "I-XPN", "B-NR", "I-NR"}
VERB_LABELS = {"B-VV", "I-VV", "B-VA", "I-VA", "B-VX", "I-VX",
               "B-VCP", "I-VCP", "B-VCN", "I-VCN",
               "B-EP", "B-EF", "B-EC", "B-ETM", "B-ETN",
               "I-EP", "I-EF", "I-EC", "I-ETM", "I-ETN"}
POS_CONFUSABLE = {"B-NNG", "B-NNP", "B-MAG", "B-VV", "B-VA", "B-VX",
                  "B-EF", "B-EC", "I-NNG", "I-NNP", "B-XSV", "B-XSA", "B-MM"}


def get_error_masks(preds, labels, confidence, category):
    V = json.load(open(DATA_DIR / "vocab.json"))
    l2i = {l: i for i, l in enumerate(V["labels"])}

    if category == "noun":
        target_indices = {l2i[l] for l in NOUN_LABELS if l in l2i}
    elif category == "verb":
        target_indices = {l2i[l] for l in VERB_LABELS if l in l2i}
    elif category == "pos":
        target_indices = {l2i[l] for l in POS_CONFUSABLE if l in l2i}
    else:
        target_indices = set(range(len(V["labels"])))

    masks = []
    for pred, label, conf in zip(preds, labels, confidence):
        mask = torch.zeros(len(pred), dtype=torch.bool)
        for i in range(len(pred)):
            if label[i] == -1:
                continue
            is_error = (pred[i] != label[i]).item()
            is_low_conf = (conf[i] < 0.85).item()
            is_target = label[i].item() in target_indices or pred[i].item() in target_indices
            mask[i] = (is_error or is_low_conf) and is_target
        masks.append(mask)
    return masks


# ---------------------------------------------------------------------------
# Phase 2: Train experts
# ---------------------------------------------------------------------------

def train_expert(expert_model, name, train_hidden, train_labels, train_jamo,
                 val_hidden, val_labels, val_jamo,
                 error_masks_train, error_masks_val,
                 epochs=12, lr=0.002, use_jamo=False):
    print(f"\n  Training {name}...")
    expert_model = expert_model.to(DEVICE)
    params = sum(p.numel() for p in expert_model.parameters())
    print(f"    Params: {params:,} ({params // 1024}KB int8)")

    V = json.load(open(DATA_DIR / "vocab.json"))
    nl = len(V["labels"])
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
    optimizer = optim.Adam(expert_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    best_acc = 0
    batch_size = 256

    for epoch in range(epochs):
        expert_model.train()
        total_loss = n_batches = 0
        indices = list(range(len(train_hidden)))
        random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            hs = [train_hidden[i] for i in batch_idx]
            ls = [train_labels[i] for i in batch_idx]
            ms = [error_masks_train[i] for i in batch_idx]

            h_pad = pad_sequence(hs, batch_first=True, padding_value=0).to(DEVICE)
            l_pad = pad_sequence(ls, batch_first=True, padding_value=-1).to(DEVICE)
            m_pad = pad_sequence(ms, batch_first=True, padding_value=False).to(DEVICE).float()

            if use_jamo:
                js = [train_jamo[i] for i in batch_idx]
                j_pad = pad_sequence(js, batch_first=True, padding_value=0).to(DEVICE)
                logits = expert_model(h_pad, j_pad)
            else:
                logits = expert_model(h_pad)

            loss_flat = criterion(logits.view(-1, nl), l_pad.view(-1))
            loss_flat = loss_flat.view(h_pad.size(0), -1)

            weight = 1.0 + 2.0 * m_pad
            valid = (l_pad != -1).float()
            weighted_loss = (loss_flat * weight * valid).sum() / valid.sum().clamp(min=1)

            optimizer.zero_grad()
            weighted_loss.backward()
            nn.utils.clip_grad_norm_(expert_model.parameters(), 5.0)
            optimizer.step()
            total_loss += weighted_loss.item()
            n_batches += 1

        scheduler.step()

        expert_model.eval()
        correct_err = total_err = correct_all = total_all = 0
        with torch.no_grad():
            for start in range(0, len(val_hidden), batch_size):
                hs = val_hidden[start:start + batch_size]
                ls = val_labels[start:start + batch_size]
                ms = error_masks_val[start:start + batch_size]

                h_pad = pad_sequence(hs, batch_first=True, padding_value=0).to(DEVICE)
                l_pad = pad_sequence(ls, batch_first=True, padding_value=-1).to(DEVICE)
                m_pad = pad_sequence(ms, batch_first=True, padding_value=False).to(DEVICE)

                if use_jamo:
                    js = val_jamo[start:start + batch_size]
                    j_pad = pad_sequence(js, batch_first=True, padding_value=0).to(DEVICE)
                    logits = expert_model(h_pad, j_pad)
                else:
                    logits = expert_model(h_pad)

                preds = logits.argmax(-1)
                valid = (l_pad != -1)
                correct_all += ((preds == l_pad) & valid).sum().item()
                total_all += valid.sum().item()
                correct_err += ((preds == l_pad) & valid & m_pad).sum().item()
                total_err += (valid & m_pad).sum().item()

        acc_all = correct_all / max(total_all, 1)
        acc_err = correct_err / max(total_err, 1)

        if acc_all > best_acc:
            best_acc = acc_all
            torch.save(expert_model.state_dict(), DATA_DIR / "moe_cache" / f"{name}.pt")

        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"    E{epoch+1}: loss={total_loss/n_batches:.4f} acc_all={acc_all:.4f} acc_err={acc_err:.4f}")

    expert_model.load_state_dict(
        torch.load(DATA_DIR / "moe_cache" / f"{name}.pt", map_location=DEVICE, weights_only=True))
    expert_model.eval()
    print(f"    Best acc: {best_acc:.4f}")
    return expert_model


# ---------------------------------------------------------------------------
# Phase 3: Train router
# ---------------------------------------------------------------------------

def train_router(router, experts, train_hidden, train_labels, train_jamo,
                 val_hidden, val_labels, val_jamo,
                 train_confidence, val_confidence, epochs=8):
    print("\n[Phase 3] Training router...")
    router = router.to(DEVICE)

    for exp in experts:
        exp.eval()

    def get_router_labels(hiddens, labels, jamos, confidence):
        router_labels = []
        batch_size = 256
        with torch.no_grad():
            for start in range(0, len(hiddens), batch_size):
                hs = hiddens[start:start + batch_size]
                ls = labels[start:start + batch_size]
                js = jamos[start:start + batch_size]
                cs = confidence[start:start + batch_size]

                h_pad = pad_sequence(hs, batch_first=True, padding_value=0).to(DEVICE)
                l_pad = pad_sequence(ls, batch_first=True, padding_value=-1).to(DEVICE)
                j_pad = pad_sequence(js, batch_first=True, padding_value=0).to(DEVICE)

                expert_preds = []
                for ei, exp in enumerate(experts):
                    if ei == 3:
                        logits = exp(h_pad, j_pad)
                    else:
                        logits = exp(h_pad)
                    expert_preds.append(logits.argmax(-1))

                for b in range(len(hs)):
                    slen = len(hs[b])
                    rl = torch.full((slen,), -1, dtype=torch.long)
                    for pos in range(slen):
                        if ls[b][pos] == -1:
                            continue
                        if cs[b][pos].item() >= 0.90:
                            continue
                        gold = l_pad[b, pos].item()
                        for ei in range(4):
                            if expert_preds[ei][b, pos].item() == gold:
                                rl[pos] = ei
                                break
                    router_labels.append(rl)
        return router_labels

    print("  Generating router targets...")
    train_router_labels = get_router_labels(train_hidden, train_labels, train_jamo, train_confidence)
    val_router_labels = get_router_labels(val_hidden, val_labels, val_jamo, val_confidence)

    dist = Counter()
    for rl in train_router_labels:
        for v in rl.tolist():
            if v >= 0:
                dist[v] += 1
    print(f"  Router label dist: {dict(sorted(dist.items()))}")

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(router.parameters(), lr=0.003)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    batch_size = 512
    best_acc = 0

    for epoch in range(epochs):
        router.train()
        total_loss = n_batches = 0
        indices = list(range(len(train_hidden)))
        random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            hs = [train_hidden[i] for i in batch_idx]
            rls = [train_router_labels[i] for i in batch_idx]

            h_pad = pad_sequence(hs, batch_first=True, padding_value=0).to(DEVICE)
            rl_pad = pad_sequence(rls, batch_first=True, padding_value=-1).to(DEVICE)

            logits = router(h_pad)
            loss = criterion(logits.view(-1, 4), rl_pad.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        router.eval()
        correct = total = 0
        with torch.no_grad():
            for start in range(0, len(val_hidden), batch_size):
                hs = val_hidden[start:start + batch_size]
                rls = val_router_labels[start:start + batch_size]

                h_pad = pad_sequence(hs, batch_first=True, padding_value=0).to(DEVICE)
                rl_pad = pad_sequence(rls, batch_first=True, padding_value=-1).to(DEVICE)

                logits = router(h_pad)
                preds = logits.argmax(-1)
                valid = (rl_pad != -1)
                correct += ((preds == rl_pad) & valid).sum().item()
                total += valid.sum().item()

        acc = correct / max(total, 1)
        if acc > best_acc:
            best_acc = acc
            torch.save(router.state_dict(), DATA_DIR / "moe_cache" / "router.pt")

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"    E{epoch+1}: loss={total_loss/max(n_batches,1):.4f} acc={acc:.4f}")

    router.load_state_dict(
        torch.load(DATA_DIR / "moe_cache" / "router.pt", map_location=DEVICE, weights_only=True))
    router.eval()
    print(f"  Router best acc: {best_acc:.4f}")
    return router


# ---------------------------------------------------------------------------
# Phase 4: Export binary
# ---------------------------------------------------------------------------

def export_moe_binary(experts, router, confidence_threshold=0.85):
    print("\n[Phase 4] Exporting MoE binary...")

    def quantize_tensor(t):
        t_flat = t.detach().cpu().float().flatten()
        max_val = t_flat.abs().max().item()
        scale = max_val / 127.0 if max_val > 0 else 1.0
        quantized = (t_flat / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized.numpy().tobytes(), scale

    buf = bytearray()
    buf.extend(b"MOE1")
    buf.append(1)   # version
    buf.append(4)   # num_experts
    buf.extend(struct.pack('<f', confidence_threshold))

    # Router: fc1 (16, 192) + bias(16) + fc2 (4, 16) + bias(4)
    r_fc1_w, r_fc1_s = quantize_tensor(router.fc1.weight)
    r_fc1_b = router.fc1.bias.detach().cpu().numpy()
    r_fc2_w, r_fc2_s = quantize_tensor(router.fc2.weight)
    r_fc2_b = router.fc2.bias.detach().cpu().numpy()

    buf.extend(struct.pack('<f', r_fc1_s))
    buf.extend(r_fc1_w)
    for v in r_fc1_b:
        buf.extend(struct.pack('<f', float(v)))
    buf.extend(struct.pack('<f', r_fc2_s))
    buf.extend(r_fc2_w)
    for v in r_fc2_b:
        buf.extend(struct.pack('<f', float(v)))

    for name, exp in [("NounSeg", experts[0]), ("POSDisamb", experts[1]),
                       ("VerbAux", experts[2]), ("CharMorph", experts[3])]:
        state = exp.state_dict()
        tensors = [(k, v) for k, v in state.items()]
        buf.extend(struct.pack('<H', len(tensors)))

        for key, tensor in tensors:
            key_bytes = key.encode('utf-8')
            buf.extend(struct.pack('<H', len(key_bytes)))
            buf.extend(key_bytes)

            shape = tensor.shape
            buf.append(len(shape))
            for d in shape:
                buf.extend(struct.pack('<I', d))

            if tensor.dtype == torch.float32 and tensor.numel() > 16:
                data, scale = quantize_tensor(tensor)
                buf.append(1)  # quantized
                buf.extend(struct.pack('<f', scale))
                buf.extend(data)
            else:
                buf.append(0)  # f32
                for v in tensor.flatten():
                    buf.extend(struct.pack('<f', float(v)))

    raw_size = len(buf)
    compressed = gzip.compress(bytes(buf), compresslevel=9)

    out_path = ROOT / "js" / "models" / "moe.bin"
    with open(out_path, 'wb') as f:
        f.write(compressed)

    print(f"  Raw: {raw_size:,} bytes ({raw_size/1024:.1f} KB)")
    print(f"  Gzip: {len(compressed):,} bytes ({len(compressed)/1024:.1f} KB)")
    print(f"  Saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  MoE (Mixture of Experts) CNN Training")
    print("=" * 70)

    cache_dir = DATA_DIR / "moe_cache"
    cache_dir.mkdir(exist_ok=True)

    # Phase 1
    if not (cache_dir / "train_hidden.pt").exists():
        extract_cnn2_features()
    else:
        print("\n[Phase 1] Using cached CNN2 features")

    print("  Loading cached data...")
    train_hidden = torch.load(cache_dir / "train_hidden.pt", weights_only=True)
    train_labels = torch.load(cache_dir / "train_labels.pt", weights_only=True)
    train_jamo = torch.load(cache_dir / "train_jamo.pt", weights_only=True)
    train_confidence = torch.load(cache_dir / "train_confidence.pt", weights_only=True)
    train_preds = torch.load(cache_dir / "train_preds.pt", weights_only=True)

    val_hidden = torch.load(cache_dir / "val_hidden.pt", weights_only=True)
    val_labels = torch.load(cache_dir / "val_labels.pt", weights_only=True)
    val_jamo = torch.load(cache_dir / "val_jamo.pt", weights_only=True)
    val_confidence = torch.load(cache_dir / "val_confidence.pt", weights_only=True)
    val_preds = torch.load(cache_dir / "val_preds.pt", weights_only=True)

    print(f"  Train: {len(train_hidden)}, Val: {len(val_hidden)}")

    V = json.load(open(DATA_DIR / "vocab.json"))
    nl = len(V["labels"])

    # Phase 2: Train experts
    print("\n[Phase 2] Training 4 experts...")

    noun_masks_tr = get_error_masks(train_preds, train_labels, train_confidence, "noun")
    noun_masks_va = get_error_masks(val_preds, val_labels, val_confidence, "noun")
    verb_masks_tr = get_error_masks(train_preds, train_labels, train_confidence, "verb")
    verb_masks_va = get_error_masks(val_preds, val_labels, val_confidence, "verb")
    pos_masks_tr = get_error_masks(train_preds, train_labels, train_confidence, "pos")
    pos_masks_va = get_error_masks(val_preds, val_labels, val_confidence, "pos")
    all_masks_tr = get_error_masks(train_preds, train_labels, train_confidence, "all")
    all_masks_va = get_error_masks(val_preds, val_labels, val_confidence, "all")

    e1 = train_expert(ExpertNounSeg(192, nl, 48), "expert_noun",
                       train_hidden, train_labels, train_jamo,
                       val_hidden, val_labels, val_jamo,
                       noun_masks_tr, noun_masks_va, epochs=12)

    e2 = train_expert(ExpertPOSDisamb(192, nl, 48), "expert_pos",
                       train_hidden, train_labels, train_jamo,
                       val_hidden, val_labels, val_jamo,
                       pos_masks_tr, pos_masks_va, epochs=12)

    e3 = train_expert(ExpertVerbAux(192, nl, 64), "expert_verb",
                       train_hidden, train_labels, train_jamo,
                       val_hidden, val_labels, val_jamo,
                       verb_masks_tr, verb_masks_va, epochs=12)

    e4 = train_expert(ExpertCharMorph(192, nl, 24, 48), "expert_char",
                       train_hidden, train_labels, train_jamo,
                       val_hidden, val_labels, val_jamo,
                       all_masks_tr, all_masks_va, epochs=12, use_jamo=True)

    experts = [e1, e2, e3, e4]

    # Phase 3
    router = Router(192, 4, 16)
    router = train_router(router, experts,
                           train_hidden, train_labels, train_jamo,
                           val_hidden, val_labels, val_jamo,
                           train_confidence, val_confidence, epochs=8)

    # Phase 4
    export_moe_binary(experts, router)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for name, exp in zip(["E1:NounSeg", "E2:POSDisamb", "E3:VerbAux", "E4:CharMorph"], experts):
        p = sum(pp.numel() for pp in exp.parameters())
        print(f"  {name}: {p:,} params ({p//1024}KB int8)")
    rp = sum(p.numel() for p in router.parameters())
    print(f"  Router: {rp:,} params ({rp//1024}KB int8)")
    total = sum(sum(p.numel() for p in e.parameters()) for e in experts) + rp
    print(f"  Total MoE: {total:,} params ({total//1024}KB int8)")


if __name__ == "__main__":
    main()
