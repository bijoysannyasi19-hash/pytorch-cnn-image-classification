import torch
import torch.nn as nn
import torch.optim as optim
import json
import csv
import data
import nets
import train
import evaluate
import plots
import random
import numpy as np

def mn():
    d = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {d}")
    
    tl, vl, tel, cn, m, s = data.gdl(1024, 42)
    
    c = [0] * 10
    for _, by in tl:
        for y in by:
            c[y.item()] += 1
    plots.pl_cb(c, cn, "figures/class_balance.png")
    
    bx, by = next(iter(tl))
    im = [plots.un_nm(bx[i], m, s) for i in range(5)]
    plots.pl_im(im, by[:5], cn, "figures/sample_aug.png")
    
    bx2, by2 = next(iter(vl))
    im2 = [plots.un_nm(bx2[i], m, s) for i in range(5)]
    plots.pl_im(im2, by2[:5], cn, "figures/sample_no_aug.png")
    
    plots.dr_ar("MLP", ["Input 1x28x28", "Flatten", "Linear 512, ReLU", "Linear 256, ReLU", "Linear 10"], "figures/arch_mlp.png")
    plots.dr_ar("Plain CNN", ["Input 1x28x28", "Conv 32, ReLU, Pool", "Conv 64, ReLU, Pool", "Conv 128, ReLU, Pool", "Flatten", "Linear 256, ReLU", "Linear 10"], "figures/arch_cnn.png")
    plots.dr_ar("Improved CNN", ["Input 1x28x28", "Conv 64, BN, ReLU", "ResB 64", "ResB 128 (s=2)", "ResB 256 (s=2)", "AdaptiveAvgPool", "Dropout", "Linear 10"], "figures/arch_imp.png")
    
    md1 = nets.MLP().to(d)
    op1 = optim.Adam(md1.parameters(), lr=0.001)
    cr = nn.CrossEntropyLoss()
    train.tr_md(md1, tl, vl, op1, cr, None, 3, "mlp", d, 3)
    plots.pl_lc("results/history_mlp.csv", "MLP Baseline", "figures/lc_mlp.png")
    r1, _, _ = evaluate.ev_md(md1, tel, d, cn)
    
    md2 = nets.PlainCNN().to(d)
    op2 = optim.Adam(md2.parameters(), lr=0.001)
    train.tr_md(md2, tl, vl, op2, cr, None, 3, "plain_cnn", d, 3)
    plots.pl_lc("results/history_plain_cnn.csv", "Plain CNN Baseline", "figures/lc_cnn.png")
    r2, _, _ = evaluate.ev_md(md2, tel, d, cn)
    
    hs = []
    for lr, dp, wd in [(0.01, 0.5, 1e-4), (0.001, 0.3, 0), (0.001, 0.5, 1e-4)]:
        tmd = nets.ImprovedCNN(dp=dp).to(d)
        top = optim.AdamW(tmd.parameters(), lr=lr, weight_decay=wd)
        va, _ = train.tr_md(tmd, tl, vl, top, cr, None, 1, f"hs_{lr}_{dp}_{wd}", d, 2)
        hs.append({"lr": lr, "dp": dp, "wd": wd, "val_acc": va})
    with open("results/hs.json", "w") as f:
        json.dump(hs, f)
    
    sds = [42, 43, 44]
    ta = []
    for sd in sds:
        data.sd(sd)
        md3 = nets.ImprovedCNN(dp=0.5).to(d)
        op3 = optim.AdamW(md3.parameters(), lr=0.001, weight_decay=1e-4)
        sc = optim.lr_scheduler.ReduceLROnPlateau(op3, mode='min', factor=0.5, patience=2)
        ep_count = 3 if sd == 42 else 1
        _, tt = train.tr_md(md3, tl, vl, op3, cr, sc, ep_count, f"imp_cnn_{sd}", d, 4)
        r, y, p = evaluate.ev_md(md3, tel, d, cn)
        ta.append(r["accuracy"])
        if sd == 42:
            plots.pl_lc(f"results/history_imp_cnn_{sd}.csv", "Improved CNN", "figures/lc_imp.png")
            plots.pl_cm(np.array(r["confusion_matrix"]), cn, "figures/cm.png")
            plots.pl_pcm(r["per_class_accuracy"], cn, "figures/pcm.png")
            ms = evaluate.er_an(y, p, cn)
            mi = []
            ml = []
            mb = []
            for i, ty, tp in ms[:5]:
                bx, by = tel.dataset[i]
                mi.append(plots.un_nm(bx, m, s))
                ml.append(f"T:{cn[ty]} P:{cn[tp]}")
                mb.append(0)
            plots.pl_im(mi, mb, ml, "figures/misclassified.png")
            with open("results/test_results.json", "w") as f:
                r["training_time"] = tt
                r["params"] = evaluate.pc(md3)
                json.dump(r, f)
            with open("results/layer_tables.json", "w") as f:
                json.dump({"mlp": evaluate.pc_tb(md1), "cnn": evaluate.pc_tb(md2), "imp": evaluate.pc_tb(md3)}, f)
    
    av = np.mean(ta)
    st = np.std(ta)
    with open("results/multi_seed.json", "w") as f:
        json.dump({"mean": av, "std": st}, f)
    
    a_m = ["None", "No BN", "No Skip", "No Aug"]
    a_v = []
    
    tmd = nets.ImprovedCNN(bn=False).to(d)
    top = optim.AdamW(tmd.parameters(), lr=0.001, weight_decay=1e-4)
    va, _ = train.tr_md(tmd, tl, vl, top, cr, None, 1, "ab_nobn", d, 2)
    a_v.append(va)
    
    tmd = nets.ImprovedCNN(sk=False).to(d)
    top = optim.AdamW(tmd.parameters(), lr=0.001, weight_decay=1e-4)
    va, _ = train.tr_md(tmd, tl, vl, top, cr, None, 1, "ab_nosk", d, 2)
    a_v.append(va)
    
    tmd = nets.ImprovedCNN().to(d)
    top = optim.AdamW(tmd.parameters(), lr=0.001, weight_decay=1e-4)
    va, _ = train.tr_md(tmd, vl, vl, top, cr, None, 1, "ab_noaug", d, 2)
    a_v.append(va)
    
    a_v.insert(0, max([float(r[4]) for r in csv.reader(open("results/history_imp_cnn_42.csv")) if r[4] != "val_acc"]))
    
    plots.pl_ab(a_m, a_v, "figures/ablation.png")
    
    plots.pl_mc(["MLP", "Plain CNN", "Improved CNN"], 
                [r1["accuracy"], r2["accuracy"], av], 
                [0, 0, st], "figures/model_comp.png")

    t1, v1, t2, v2 = [], [], [], []
    with open("results/history_plain_cnn.csv", "r") as fl:
        r = csv.reader(fl)
        next(r)
        for ro in r:
            t1.append(float(ro[1]))
            v1.append(float(ro[2]))
    with open("results/history_imp_cnn_42.csv", "r") as fl:
        r = csv.reader(fl)
        next(r)
        for ro in r:
            t2.append(float(ro[1]))
            v2.append(float(ro[2]))
    plots.pl_of(t1, v1, t2, v2, "figures/overfitting.png")

if __name__ == "__main__":
    mn()
