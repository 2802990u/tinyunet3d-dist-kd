# student_unet3d.py
# Ultra-lightweight 3D U-Net student (TinyUNet3D).
# Depthwise-separable convs + trilinear upsampling to keep params < 1M.
# The paper uses base=64 (channel widths 64/128/256/512, 0.957M parameters).
# BraTS I/O: in_chans=4 (t2w,t2f,t1n,t1c), out_chans=4.
import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConv3d(nn.Module):
    """Depthwise-separable 3D conv + InstanceNorm + LeakyReLU."""
    def __init__(self, cin, cout):
        super().__init__()
        self.dw = nn.Conv3d(cin, cin, kernel_size=3, padding=1, groups=cin, bias=False)
        self.pw = nn.Conv3d(cin, cout, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm3d(cout, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.pw(self.dw(x))))


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.a = DSConv3d(cin, cout)
        self.b = DSConv3d(cout, cout)

    def forward(self, x):
        return self.b(self.a(x))


def up(x, ref):
    """Trilinear upsample x to ref's spatial size (cheap, no params)."""
    return F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)


class TinyUNet3D(nn.Module):
    """
    4-level encoder/decoder. Parameter count by base (verify with `python student_unet3d.py`):
    base=32 -> 0.262M, base=48 -> 0.556M, base=64 -> 0.957M (the paper setting).
    """
    def __init__(self, in_chans=4, out_chans=4, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]     # base=64: 64, 128, 256, 512
        self.pool = nn.MaxPool3d(2)

        # encoder
        self.enc0 = Block(in_chans, c[0])
        self.enc1 = Block(c[0], c[1])
        self.enc2 = Block(c[1], c[2])
        self.bott = Block(c[2], c[3])

        # decoder (trilinear up + concat skip + block)
        self.dec2 = Block(c[3] + c[2], c[2])
        self.dec1 = Block(c[2] + c[1], c[1])
        self.dec0 = Block(c[1] + c[0], c[0])

        self.head = nn.Conv3d(c[0], out_chans, kernel_size=1)

    def forward(self, x):
        e0 = self.enc0(x)                 # full res
        e1 = self.enc1(self.pool(e0))     # /2
        e2 = self.enc2(self.pool(e1))     # /4
        b  = self.bott(self.pool(e2))     # /8

        d2 = self.dec2(torch.cat([up(b, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([up(d2, e1), e1], dim=1))
        d0 = self.dec0(torch.cat([up(d1, e0), e0], dim=1))
        return self.head(d0)              # (B, out_chans, D, H, W) logits


if __name__ == "__main__":
    for base in (32, 48, 64):
        m = TinyUNet3D(in_chans=4, out_chans=4, base=base)
        n = sum(p.numel() for p in m.parameters())
        print(f"base={base:>2}  params={n:>9,}  ({n/1e6:.3f}M)")
    # shape sanity check on a 128^3 patch
    m = TinyUNet3D(base=64).eval()
    with torch.no_grad():
        y = m(torch.randn(1, 4, 128, 128, 128))
    print("output shape:", tuple(y.shape))
