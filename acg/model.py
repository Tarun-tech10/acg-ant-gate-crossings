"""Small fully-convolutional hourglass trained from randomly initialised weights."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class CenterNet(nn.Module):
    """Full-resolution centre heatmap plus a sub-pixel offset field."""

    def __init__(self, cin=3, width=32, depth=4):
        super().__init__()
        chs = [width * (2 ** i) for i in range(depth + 1)]
        self.down = nn.ModuleList()
        prev = cin
        for c in chs[:-1]:
            self.down.append(conv_block(prev, c))
            prev = c
        self.bottom = conv_block(prev, chs[-1])
        self.up_reduce = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.up_reduce.append(nn.Conv2d(chs[i + 1], chs[i], 1))
            self.up_conv.append(conv_block(chs[i] * 2, chs[i]))
        self.hm_head = nn.Sequential(
            nn.Conv2d(chs[0], chs[0], 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(chs[0], 1, 1),
        )
        self.off_head = nn.Sequential(
            nn.Conv2d(chs[0], chs[0], 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(chs[0], 2, 1),
        )
        self.hm_head[-1].bias.data.fill_(-4.0)

    def forward(self, x):
        skips = []
        for blk in self.down:
            x = blk(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottom(x)
        for red, blk, skip in zip(self.up_reduce, self.up_conv, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = red(x)
            x = blk(torch.cat([x, skip], 1))
        return self.hm_head(x), self.off_head(x)
