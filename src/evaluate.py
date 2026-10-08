import torch
import time
import json
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

def ev_md(md, dl, d, cn):
    md.eval()
    y = []
    p = []
    st = time.time()
    with torch.no_grad():
        for bx, by in dl:
            bx = bx.to(d)
            pr = md(bx)
            _, pd = pr.max(1)
            y.extend(by.tolist())
            p.extend(pd.cpu().tolist())
    et = time.time() - st
    it = et / len(y)
    
    a = accuracy_score(y, p)
    mp, mr, mf, _ = precision_recall_fscore_support(y, p, average='macro')
    cp, cr, cf, _ = precision_recall_fscore_support(y, p, average=None)
    cm = confusion_matrix(y, p)
    ca = cm.diagonal() / cm.sum(axis=1)
    
    r = {
        "accuracy": a,
        "macro_precision": mp,
        "macro_recall": mr,
        "macro_f1": mf,
        "per_class_precision": cp.tolist(),
        "per_class_recall": cr.tolist(),
        "per_class_f1": cf.tolist(),
        "per_class_accuracy": ca.tolist(),
        "confusion_matrix": cm.tolist(),
        "inference_time_per_image": it
    }
    return r, y, p

def er_an(y, p, cn):
    ms = []
    for i, (ty, tp) in enumerate(zip(y, p)):
        if ty != tp:
            ms.append((i, ty, tp))
    return ms

def pc(md):
    return sum(p.numel() for p in md.parameters() if p.requires_grad)

def pc_tb(md):
    tb = []
    for n, m in md.named_children():
        tb.append({"layer": n, "params": pc(m)})
    return tb
