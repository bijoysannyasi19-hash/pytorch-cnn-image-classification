import matplotlib.pyplot as plt
import numpy as np
import csv

def un_nm(img, m, s):
    i = img.clone()
    for c in range(1):
        i[c] = i[c] * s[c] + m[c]
    return i.clamp(0, 1)

def pl_im(imgs, lbs, cn, f):
    fg, ax = plt.subplots(1, len(imgs), figsize=(12, 3))
    for i, a in enumerate(ax):
        a.imshow(imgs[i].squeeze(), cmap="gray")
        a.set_title(cn[lbs[i]])
        a.axis("off")
    plt.tight_layout()
    plt.savefig(f, dpi=200)
    plt.close()

def pl_cb(c, cn, f):
    plt.figure(figsize=(8, 4))
    plt.bar(cn, c)
    plt.title("Class Balance")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f, dpi=200)
    plt.close()

def pl_lc(hs, t, f):
    ep = []
    tl = []
    vl = []
    ta = []
    va = []
    with open(hs, "r") as fl:
        r = csv.reader(fl)
        next(r)
        for ro in r:
            ep.append(int(ro[0]))
            tl.append(float(ro[1]))
            vl.append(float(ro[2]))
            ta.append(float(ro[3]))
            va.append(float(ro[4]))
    fg, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(ep, tl, label="Train Loss")
    a1.plot(ep, vl, label="Val Loss")
    a1.set_title(f"{t} - Loss")
    a1.set_xlabel("Epoch")
    a1.set_ylabel("Loss")
    a1.legend()
    a2.plot(ep, ta, label="Train Acc")
    a2.plot(ep, va, label="Val Acc")
    a2.set_title(f"{t} - Accuracy")
    a2.set_xlabel("Epoch")
    a2.set_ylabel("Accuracy")
    a2.legend()
    plt.tight_layout()
    plt.savefig(f, dpi=200)
    plt.close()

def pl_cm(cm, cn, f):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tm = np.arange(len(cn))
    plt.xticks(tm, cn, rotation=45)
    plt.yticks(tm, cn)
    for i in range(len(cn)):
        for j in range(len(cn)):
            plt.text(j, i, str(int(cm[i][j])), ha="center", va="center", color="white" if cm[i][j] > cm.max()/2 else "black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f, dpi=200)
    plt.close()

def pl_pcm(ca, cn, f):
    plt.figure(figsize=(8, 4))
    plt.bar(cn, ca)
    plt.title("Per-Class Accuracy")
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f, dpi=200)
    plt.close()

def pl_ab(nm, vl, f):
    plt.figure(figsize=(8, 4))
    plt.bar(nm, vl)
    plt.title("Ablation Study Results")
    plt.xlabel("Component Removed")
    plt.ylabel("Validation Accuracy")
    plt.tight_layout()
    plt.savefig(f, dpi=200)
    plt.close()

def pl_mc(nm, ac, st, f):
    plt.figure(figsize=(8, 4))
    plt.bar(nm, ac, yerr=st, capsize=5)
    plt.title("Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Test Accuracy")
    plt.tight_layout()
    plt.savefig(f, dpi=200)
    plt.close()

def dr_ar(n, l, f):
    plt.figure(figsize=(3, len(l)))
    for i, lx in enumerate(l):
        plt.text(0.5, 1 - (i + 0.5) / len(l), lx, ha="center", va="center", bbox=dict(boxstyle="round", fc="white", ec="black"))
        if i < len(l) - 1:
            plt.annotate("", xy=(0.5, 1 - (i + 0.8) / len(l)), xytext=(0.5, 1 - (i + 0.7) / len(l)), arrowprops=dict(arrowstyle="->"))
    plt.title(n)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f, dpi=200)
    plt.close()

def pl_of(t1, v1, t2, v2, f):
    fg, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(t1, label="Train")
    a1.plot(v1, label="Val")
    a1.set_title("Before (Overfitting)")
    a1.legend()
    a2.plot(t2, label="Train")
    a2.plot(v2, label="Val")
    a2.set_title("After (Regularization)")
    a2.legend()
    plt.tight_layout()
    plt.savefig(f, dpi=200)
    plt.close()
