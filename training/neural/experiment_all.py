"""Run ALL disambiguation experiments. CNN ensemble + pair table + 2-layer CNN."""
import json, os, random, subprocess, tempfile, time
import torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = Path(__file__).parent
NIKL_DIR = Path.home() / "Downloads" / "NIKL_MP(v1.1)"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

POS_TAGS = ["NNG","NNP","NNB","NR","NP","VV","VA","VX","VCP","VCN","MAG","MAJ","MM","IC",
    "JKS","JKC","JKG","JKO","JKB","JKV","JKQ","JX","JC","EP","EF","EC","ETN","ETM",
    "XPN","XSN","XSV","XSA","XR","SF","SP","SS","SE","SO","SW","SH","SL","SN"]
POS_SET = set(POS_TAGS)
NM = {'MMD':'MM','MMN':'MM','MMA':'MM','NA':'NNG','NAP':'NNG','NF':'NNG','NV':'VV'}
def np_(t):
    if t in POS_SET: return t
    if t in NM: return NM[t]
    b = t.split('-')[0]
    return b if b in POS_SET else 'SW'

def load_nikl(n=2000):
    ss = []
    for fn in ["NXMP1902008040.json","SXMP1902008031.json"]:
        p = NIKL_DIR/fn
        if not p.exists(): continue
        d = json.load(open(p))
        for doc in d["document"]:
            for s in doc["sentence"]:
                t = s["form"]
                if not t or len(t)<5 or len(t)>200: continue
                ms = [(m["form"],np_(m["label"])) for m in s["morpheme"] if m["form"].strip()]
                if ms: ss.append((t,ms))
    random.seed(42); return random.sample(ss, min(n,len(ss)))

def run_garu(ss, ex="analyze_batch_v2"):
    with tempfile.NamedTemporaryFile(mode='w',suffix='.txt',delete=False,encoding='utf-8') as f:
        for t,_ in ss: f.write(t+'\n')
        ip = f.name
    r = subprocess.run(["cargo","run","--release","--example",ex,"--",ip],cwd=str(ROOT),capture_output=True,text=True,timeout=600)
    os.unlink(ip)
    if r.returncode!=0: return None
    res, cur = [], []
    for l in r.stdout.strip().split('\n'):
        l = l.strip()
        if l=='---': res.append(cur); cur=[]
        elif l=='[]': res.append([])
        elif '\t' in l: f,p=l.split('\t',1); cur.append((f,p))
    if cur: res.append(cur)
    return res

def f1(preds, golds):
    tm=tp=tg=0
    for i in range(min(len(preds),len(golds))):
        ps=set((f,t) for f,t in preds[i] if f.strip())
        gs=set((f,t) for f,t in golds[i][1] if f.strip())
        tm+=len(ps&gs); tp+=len(ps); tg+=len(gs)
    P=tm/max(tp,1); R=tm/max(tg,1)
    return 2*P*R/max(P+R,1e-10)

# --- CNN ---
class DS(Dataset):
    def __init__(self, path, s2i, l2i):
        self.data = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                it = json.loads(line); sy,la=it["syllables"][:150],it["labels"][:150]
                si=[s2i.get(s,1) for s in sy]; li=[l2i.get(l,0) for l in la]
                if si: self.data.append((torch.tensor(si,dtype=torch.long),torch.tensor(li,dtype=torch.long)))
    def __len__(self): return len(self.data)
    def __getitem__(self,i): return self.data[i]
def coll(b):
    s,l=zip(*b); return pad_sequence(s,True,0),pad_sequence(l,True,-1),torch.tensor([len(x) for x in s])

class CNN1(nn.Module):
    def __init__(s,vs,ed,nl,h=96):
        super().__init__(); s.emb=nn.Embedding(vs,ed,padding_idx=0)
        s.c3=nn.Conv1d(ed,h,3,padding=1); s.c5=nn.Conv1d(ed,h,5,padding=2); s.c7=nn.Conv1d(ed,h,7,padding=3)
        s.drop=nn.Dropout(0.3); s.fc=nn.Linear(h*3,nl)
    def forward(s,x):
        e=s.emb(x).transpose(1,2)
        c=torch.cat([torch.relu(s.c3(e)),torch.relu(s.c5(e)),torch.relu(s.c7(e))],1).transpose(1,2)
        return s.fc(s.drop(c))

class CNN2(nn.Module):
    def __init__(s,vs,ed,nl,h=96):
        super().__init__(); s.emb=nn.Embedding(vs,ed,padding_idx=0)
        s.c3=nn.Conv1d(ed,h,3,padding=1); s.c5=nn.Conv1d(ed,h,5,padding=2); s.c9=nn.Conv1d(ed,h,9,padding=4)
        s.bn=nn.BatchNorm1d(h*3)
        s.c2a=nn.Conv1d(h*3,h,3,padding=1); s.c2b=nn.Conv1d(h*3,h,7,padding=3)
        s.drop=nn.Dropout(0.3); s.fc=nn.Linear(h*2,nl)
    def forward(s,x):
        e=s.emb(x).transpose(1,2)
        l1=torch.cat([torch.relu(s.c3(e)),torch.relu(s.c5(e)),torch.relu(s.c9(e))],1)
        l1=s.bn(l1); l2=torch.cat([torch.relu(s.c2a(l1)),torch.relu(s.c2b(l1))],1).transpose(1,2)
        return s.fc(s.drop(l2))

def train_cnn(cls, name, ep=8):
    V=json.load(open(DATA_DIR/"vocab.json")); s2i={s:i for i,s in enumerate(V["syllables"])}
    l2i={l:i for i,l in enumerate(V["labels"])}; i2l={i:l for l,i in l2i.items()}
    vs,nl=len(V["syllables"]),len(V["labels"])
    tr=DS(DATA_DIR/"train.jsonl",s2i,l2i); va=DS(DATA_DIR/"val.jsonl",s2i,l2i)
    tl=DataLoader(tr,256,True,collate_fn=coll,num_workers=0)
    vl=DataLoader(va,512,False,collate_fn=coll,num_workers=0)
    m=cls(vs,48,nl,96).to(DEVICE); np_=sum(p.numel() for p in m.parameters())
    print(f"  [{name}] params={np_:,} ({np_//1024}KB int8)")
    cr=nn.CrossEntropyLoss(ignore_index=-1); op=optim.Adam(m.parameters(),lr=0.003)
    sc=optim.lr_scheduler.CosineAnnealingLR(op,ep); best=0
    for e in range(ep):
        m.train(); tls,nb=0,0
        for s,l,le in tl:
            s,l=s.to(DEVICE),l.to(DEVICE); lo=cr(m(s).view(-1,nl),l.view(-1))
            op.zero_grad(); lo.backward(); nn.utils.clip_grad_norm_(m.parameters(),5.0); op.step()
            tls+=lo.item(); nb+=1
        sc.step(); m.eval(); cor=tot=0
        with torch.no_grad():
            for s,l,le in vl:
                s,l=s.to(DEVICE),l.to(DEVICE); p=m(s).argmax(-1); mk=(l!=-1)
                cor+=((p==l)&mk).sum().item(); tot+=mk.sum().item()
        acc=cor/max(tot,1)
        if acc>best: best=acc; torch.save(m.state_dict(),DATA_DIR/f"{name}.pt")
        print(f"    E{e+1}: acc={acc:.4f}")
    m.load_state_dict(torch.load(DATA_DIR/f"{name}.pt",map_location=DEVICE,weights_only=True)); m.eval()
    return m,s2i,i2l

def cnn_pred(m,s2i,i2l,text):
    sy=list(text); si=[s2i.get(s,1) for s in sy]
    x=torch.tensor([si],dtype=torch.long).to(DEVICE)
    with torch.no_grad(): tg=m(x).argmax(-1)
    tags=[i2l.get(tg[0,i].item(),'O') for i in range(len(sy))]
    mo=[]; cf,cp="",None
    for c,t in zip(sy,tags):
        if c==' ':
            if cf and cp: mo.append((cf,cp)); cf,cp="",None
            continue
        if t[:2]=='B-':
            if cf and cp: mo.append((cf,cp))
            cf,cp=c,t[2:]
        elif t[:2]=='I-': cf+=c
        else:
            if cf and cp: mo.append((cf,cp))
            cf,cp=c,"SW"
    if cf and cp: mo.append((cf,cp))
    return mo

def cnn_pred_all(m,s2i,i2l,sents):
    return [cnn_pred(m,s2i,i2l,t) for t,_ in sents]

def main():
    print("="*70+"\n  COMPREHENSIVE EXPERIMENT\n"+"="*70)
    sents=load_nikl(2000); print(f"NIKL: {len(sents)} sents")

    # Baselines
    print("\n[Garu v1]"); v1=run_garu(sents,"analyze_batch"); v1f=f1(v1,sents); print(f"  F1={v1f:.4f}")
    print("\n[Garu v2]"); v2=run_garu(sents,"analyze_batch_v2"); v2f=f1(v2,sents); print(f"  F1={v2f:.4f}")

    # CNN1 (already trained)
    V=json.load(open(DATA_DIR/"vocab.json"))
    s2i={s:i for i,s in enumerate(V["syllables"])}; l2i={l:i for i,l in enumerate(V["labels"])}
    i2l={i:l for l,i in l2i.items()}
    m1=CNN1(len(V["syllables"]),48,len(V["labels"]),96).to(DEVICE)
    m1.load_state_dict(torch.load(DATA_DIR/"model_best.pt",map_location=DEVICE,weights_only=True)); m1.eval()
    c1p=cnn_pred_all(m1,s2i,i2l,sents); c1f=f1(c1p,sents)
    print(f"\n[CNN1 standalone] F1={c1f:.4f}")

    # Train CNN2
    print("\n[Training CNN2 (2-layer)]")
    m2,s2i2,i2l2=train_cnn(CNN2,"cnn2",8)
    c2p=cnn_pred_all(m2,s2i2,i2l2,sents); c2f=f1(c2p,sents)
    print(f"  CNN2 standalone F1={c2f:.4f}")

    # Ensembles
    # A) CNN overrides Garu when disagree
    ea=[list(c1p[i]) for i in range(len(sents))]
    eaf=f1(ea,sents)
    print(f"\n[Ensemble A: CNN1 only] F1={eaf:.4f}")

    # B) Garu base, CNN overrides POS only (same form)
    eb=[]
    for i in range(min(len(v2),len(c1p))):
        gm=dict(v2[i]); cm=dict(c1p[i])
        res=[]
        for gf,gp in v2[i]:
            if gf in cm and cm[gf]!=gp:
                res.append((gf,cm[gf]))
            else:
                res.append((gf,gp))
        eb.append(res)
    ebf=f1(eb,sents)
    print(f"[Ensemble B: Garu+CNN1 POS swap] F1={ebf:.4f}")

    # C) Majority vote: v1, v2, CNN1
    ec=[]
    for i in range(min(len(v1),len(v2),len(c1p))):
        s1=set((f,t) for f,t in v1[i]); s2=set((f,t) for f,t in v2[i]); sc=set((f,t) for f,t in c1p[i])
        # Items in at least 2 of 3
        all_items = s1|s2|sc
        voted = [(f,t) for f,t in all_items if sum([((f,t) in s) for s in [s1,s2,sc]])>=2]
        ec.append(voted)
    ecf=f1(ec,sents)
    print(f"[Ensemble C: majority vote v1+v2+CNN1] F1={ecf:.4f}")

    # D) CNN2 override
    ed=[list(c2p[i]) for i in range(len(sents))]
    edf=f1(ed,sents)
    print(f"[Ensemble D: CNN2 only] F1={edf:.4f}")

    # E) v2 + CNN2 POS swap
    ee=[]
    for i in range(min(len(v2),len(c2p))):
        gm=dict(v2[i]); cm=dict(c2p[i])
        res=[]
        for gf,gp in v2[i]:
            if gf in cm and cm[gf]!=gp: res.append((gf,cm[gf]))
            else: res.append((gf,gp))
        ee.append(res)
    eef=f1(ee,sents)
    print(f"[Ensemble E: Garu+CNN2 POS swap] F1={eef:.4f}")

    # Ambiguity test
    print("\n"+"="*70+"\n  AMBIGUITY TEST\n"+"="*70)
    tests=["나는 하늘을 나는 새를 보았다.","나는 밥을 먹었다.","내가 한 일이 많다.",
           "한 사람이 왔다.","먹고 있는 사람이 많다.","나는 나는 것이 무섭다.","그는 그 사건을 기억한다."]
    for t in tests:
        print(f"\n  {t}")
        print(f"    CNN1: {' + '.join(f'{f}/{p}' for f,p in cnn_pred(m1,s2i,i2l,t))}")
        print(f"    CNN2: {' + '.join(f'{f}/{p}' for f,p in cnn_pred(m2,s2i2,i2l2,t))}")

    # Summary
    print("\n"+"="*70+"\n  FINAL RANKING\n"+"="*70)
    all_res = [
        ("Garu v1 (baseline)",v1f),("Garu v2 (sent Viterbi)",v2f),
        ("CNN1 standalone",c1f),("CNN2 standalone (2-layer)",c2f),
        ("Ens A: CNN1 only",eaf),("Ens B: Garu+CNN1 POS swap",ebf),
        ("Ens C: majority vote",ecf),("Ens D: CNN2 only",edf),
        ("Ens E: Garu+CNN2 POS swap",eef),
    ]
    for name,score in sorted(all_res,key=lambda x:-x[1]):
        d=(score-v1f)*100
        print(f"  {score:.4f} ({d:+.2f}%p)  {name}")

if __name__=="__main__": main()
