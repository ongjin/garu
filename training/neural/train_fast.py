"""Fast CNN model (no CRF) for quick experiment."""
import json, time, torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

DATA_DIR = Path(__file__).parent
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

class DS(Dataset):
    def __init__(self, path, s2i, l2i):
        self.data = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                it = json.loads(line)
                sy, la = it["syllables"][:150], it["labels"][:150]
                si = [s2i.get(s, 1) for s in sy]
                li = [l2i.get(l, 0) for l in la]
                if si: self.data.append((torch.tensor(si,dtype=torch.long), torch.tensor(li,dtype=torch.long)))
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def collate(batch):
    s, l = zip(*batch)
    le = torch.tensor([len(x) for x in s])
    return pad_sequence(s,True,0), pad_sequence(l,True,-1), le

class Model(nn.Module):
    def __init__(self, vs, ed, nl, h=96):
        super().__init__()
        self.emb = nn.Embedding(vs, ed, padding_idx=0)
        self.c3 = nn.Conv1d(ed, h, 3, padding=1)
        self.c5 = nn.Conv1d(ed, h, 5, padding=2)
        self.c7 = nn.Conv1d(ed, h, 7, padding=3)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(h*3, nl)
    def forward(self, x):
        e = self.emb(x).transpose(1,2)
        c = torch.cat([torch.relu(self.c3(e)), torch.relu(self.c5(e)), torch.relu(self.c7(e))], 1).transpose(1,2)
        return self.fc(self.drop(c))

def main():
    V = json.load(open(DATA_DIR/"vocab.json"))
    s2i = {s:i for i,s in enumerate(V["syllables"])}
    l2i = {l:i for i,l in enumerate(V["labels"])}
    i2l = {i:l for l,i in l2i.items()}
    vs, nl, ed, h = len(V["syllables"]), len(V["labels"]), 48, 96

    tr = DS(DATA_DIR/"train.jsonl", s2i, l2i)
    va = DS(DATA_DIR/"val.jsonl", s2i, l2i)
    print(f"V={vs} L={nl} E={ed} H={h} Train={len(tr)} Val={len(va)}")

    tl = DataLoader(tr, 256, True, collate_fn=collate, num_workers=0)
    vl = DataLoader(va, 512, False, collate_fn=collate, num_workers=0)

    m = Model(vs, ed, nl, h).to(DEVICE)
    np_ = sum(p.numel() for p in m.parameters())
    print(f"Params: {np_:,} ({np_*4//1024}KB fp32, {np_//1024}KB int8)")

    crit = nn.CrossEntropyLoss(ignore_index=-1)
    opt = optim.Adam(m.parameters(), lr=0.003)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, 10)
    best = 0

    for ep in range(10):
        m.train(); tl_sum, nb = 0, 0; t0 = time.time()
        for s, l, le in tl:
            s, l = s.to(DEVICE), l.to(DEVICE)
            lo = crit(m(s).view(-1, nl), l.view(-1))
            opt.zero_grad(); lo.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
            tl_sum += lo.item(); nb += 1
        sch.step()
        m.eval(); cor = tot = 0
        with torch.no_grad():
            for s, l, le in vl:
                s, l = s.to(DEVICE), l.to(DEVICE)
                p = m(s).argmax(-1); mk = (l != -1)
                cor += ((p==l)&mk).sum().item(); tot += mk.sum().item()
        acc = cor/max(tot,1)
        print(f"E{ep+1}: loss={tl_sum/nb:.4f} acc={acc:.4f} t={time.time()-t0:.0f}s")
        if acc > best:
            best = acc; torch.save(m.state_dict(), DATA_DIR/"model_best.pt")
            print(f"  best={best:.4f}")

    m.load_state_dict(torch.load(DATA_DIR/"model_best.pt", map_location=DEVICE, weights_only=True))
    m.eval()

    tests = [
        "나는 하늘을 나는 새를 보았다.",
        "나는 밥을 먹었다.",
        "내가 한 일이 많다.",
        "한 사람이 왔다.",
        "먹고 있는 사람이 많다.",
        "나는 나는 것이 무섭다.",
        "그는 그 사건을 기억한다.",
    ]
    print(f"\n{'='*60}\n  AMBIGUITY TEST\n{'='*60}")
    for text in tests:
        sy = list(text); si = [s2i.get(s,1) for s in sy]
        x = torch.tensor([si], dtype=torch.long).to(DEVICE)
        with torch.no_grad(): tg = m(x).argmax(-1)
        tags = [i2l.get(tg[0,i].item(),'O') for i in range(len(sy))]
        mo = []; cf, cp = "", None
        for c, t in zip(sy, tags):
            if c==' ':
                if cf and cp: mo.append(f"{cf}/{cp}"); cf, cp = "", None
                continue
            if t[:2]=='B-':
                if cf and cp: mo.append(f"{cf}/{cp}")
                cf, cp = c, t[2:]
            elif t[:2]=='I-': cf += c
            else:
                if cf and cp: mo.append(f"{cf}/{cp}")
                cf, cp = c, "SW"
        if cf and cp: mo.append(f"{cf}/{cp}")
        print(f"  {text}\n  -> {' + '.join(mo)}\n")

    info = {"vocab_size":vs,"embed_dim":ed,"hidden":h,"num_labels":nl,
            "n_params":np_,"size_fp32_kb":np_*4//1024,"size_int8_kb":np_//1024,"best_val_acc":round(best,4)}
    json.dump(info, open(DATA_DIR/"model_info.json",'w'), indent=2)
    print(f"Done: {json.dumps(info)}")

if __name__=="__main__": main()
