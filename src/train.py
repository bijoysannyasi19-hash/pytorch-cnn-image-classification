import torch
import time
import csv
import json

def tr_ep(md, dl, op, cr, d):
    md.train()
    el = 0.0
    c = 0
    t = 0
    for bx, by in dl:
        bx, by = bx.to(d), by.to(d)
        op.zero_grad()
        pr = md(bx)
        ls = cr(pr, by)
        ls.backward()
        op.step()
        el += ls.item() * bx.size(0)
        _, pd = pr.max(1)
        c += pd.eq(by).sum().item()
        t += by.size(0)
    return el / t, c / t

def vl_ep(md, dl, cr, d):
    md.eval()
    el = 0.0
    c = 0
    t = 0
    with torch.no_grad():
        for bx, by in dl:
            bx, by = bx.to(d), by.to(d)
            pr = md(bx)
            ls = cr(pr, by)
            el += ls.item() * bx.size(0)
            _, pd = pr.max(1)
            c += pd.eq(by).sum().item()
            t += by.size(0)
    return el / t, c / t

def tr_md(md, tl, vl, op, cr, sch, n, mn, d, pt=5):
    bm = float('inf')
    pc = 0
    h = []
    st = time.time()
    for ep in range(n):
        ts = time.time()
        trl, tra = tr_ep(md, tl, op, cr, d)
        vll, vla = vl_ep(md, vl, cr, d)
        if sch is not None:
            sch.step(vll)
        lr = op.param_groups[0]['lr']
        tt = time.time() - ts
        h.append([ep, trl, vll, tra, vla, lr, tt])
        if vll < bm:
            bm = vll
            pc = 0
            torch.save(md.state_dict(), f"models/{mn}.pt")
        else:
            pc += 1
            if pc >= pt:
                break
    with open(f"results/history_{mn}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr", "time"])
        w.writerows(h)
    return bm, time.time() - st
