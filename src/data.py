import torch
import torchvision
import numpy as np

def sd(val):
    torch.manual_seed(val)
    np.random.seed(val)
    torch.backends.cudnn.deterministic = True

def ms():
    ts = torchvision.transforms.ToTensor()
    ds = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=ts)
    ld = torch.utils.data.DataLoader(ds, batch_size=5000, shuffle=False)
    m = torch.zeros(1)
    s = torch.zeros(1)
    for bx, by in ld:
        m += bx.mean([0, 2, 3])
        s += (bx ** 2).mean([0, 2, 3])
    m /= len(ld)
    s = torch.sqrt((s / len(ld)) - m ** 2)
    return m.tolist(), s.tolist()

def gdl(bs, sg=42):
    print("Setting seed...")
    sd(sg)
    print("Calculating mean/std...")
    m, s = ms()
    print("Mean/std done:", m, s)
    t1 = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(28, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(m, s)
    ])
    t2 = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(m, s)
    ])
    print("Loading datasets...")
    d1 = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=t1)
    d2 = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=t2)
    print("Splitting dataset...")
    g = torch.Generator().manual_seed(42)
    ix, iy = torch.utils.data.random_split(range(60000), [45000, 15000], generator=g)
    tr = torch.utils.data.Subset(d1, ix.indices[:45000])
    vl = torch.utils.data.Subset(d2, iy.indices[:5000])
    te = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=t2)
    tl = torch.utils.data.DataLoader(tr, batch_size=bs, shuffle=True)
    vll = torch.utils.data.DataLoader(vl, batch_size=bs, shuffle=False)
    tel = torch.utils.data.DataLoader(te, batch_size=bs, shuffle=False)
    cn = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    return tl, vll, tel, cn, m, s
