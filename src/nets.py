import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.m(x)

class PlainCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 3 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.m(x)

class ResB(nn.Module):
    def __init__(self, ci, co, s=1, bn=True, sk=True):
        super().__init__()
        self.c1 = nn.Conv2d(ci, co, 3, stride=s, padding=1, bias=not bn)
        self.b1 = nn.BatchNorm2d(co) if bn else nn.Identity()
        self.c2 = nn.Conv2d(co, co, 3, stride=1, padding=1, bias=not bn)
        self.b2 = nn.BatchNorm2d(co) if bn else nn.Identity()
        self.sk = sk
        self.s = nn.Sequential()
        if sk and (s != 1 or ci != co):
            self.s = nn.Sequential(
                nn.Conv2d(ci, co, 1, stride=s, bias=not bn),
                nn.BatchNorm2d(co) if bn else nn.Identity()
            )
    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)))
        o = self.b2(self.c2(o))
        if self.sk:
            o += self.s(x)
        return F.relu(o)

class ImprovedCNN(nn.Module):
    def __init__(self, dp=0.5, bn=True, sk=True):
        super().__init__()
        self.m = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=not bn),
            nn.BatchNorm2d(16) if bn else nn.Identity(),
            nn.ReLU(),
            ResB(16, 16, 1, bn, sk),
            ResB(16, 32, 2, bn, sk),
            ResB(32, 64, 2, bn, sk),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dp) if dp > 0 else nn.Identity(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.m(x)
