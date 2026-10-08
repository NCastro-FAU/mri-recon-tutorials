"""06 - Deep-learning reconstruction: a small unrolled network (MoDL-style).

    repeat K times:   z = x + CNN(x)                      (learned denoiser, shared weights)
                      x = argmin ||M F x - y||^2 + mu ||x - z||^2   (data consistency, closed form)

For single-coil Cartesian data the data-consistency step is exact in k-space:
    F x = (M y + mu F z) / (M + mu)

The network is trained on random-ellipse images (no real data needed) and
tested on the Shepp-Logan phantom, which it has never seen. Training takes
2-4 minutes on an Apple-silicon GPU (MPS), a CUDA GPU or a recent CPU.

Reference: Aggarwal, Mani, Jacob, IEEE TMI 38:394 (2019).
"""
import time

import numpy as np
import torch
import torch.nn as nn

from mrutils import mask_variable_density, nrmse, shepp_logan

N, R, K = 128, 3, 5
dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


def fft2c(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def ifft2c(k):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def random_ellipses(batch, n=N, rng=np.random.default_rng(1)):
    """Training images: 5-15 random ellipses with random intensity and a smooth phase."""
    y, x = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    out = np.zeros((batch, n, n), np.complex64)
    for b in range(batch):
        img = np.zeros((n, n))
        for _ in range(rng.integers(5, 16)):
            a, bb = rng.uniform(.05, .7, 2); x0, y0 = rng.uniform(-.5, .5, 2); p = rng.uniform(0, np.pi)
            xr = (x - x0) * np.cos(p) + (y - y0) * np.sin(p); yr = -(x - x0) * np.sin(p) + (y - y0) * np.cos(p)
            img[(xr / a) ** 2 + (yr / bb) ** 2 <= 1] += rng.uniform(-.3, .8)
        img = np.clip(img, 0, None); img /= max(img.max(), 1e-6)
        out[b] = img * np.exp(1j * rng.uniform(-1, 1) * (x * rng.uniform(-1, 1) + y * rng.uniform(-1, 1)))
    return torch.from_numpy(out)


class Denoiser(nn.Module):
    def __init__(self, ch=48, layers=5):
        super().__init__()
        mods = [nn.Conv2d(2, ch, 3, padding=1), nn.ReLU()]
        for _ in range(layers - 2):
            mods += [nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU()]
        mods += [nn.Conv2d(ch, 2, 3, padding=1)]
        self.net = nn.Sequential(*mods)

    def forward(self, x):                                  # complex (B, n, n) -> complex
        xr = torch.stack([x.real, x.imag], 1)
        out = xr + self.net(xr)                            # residual learning
        return torch.complex(out[:, 0], out[:, 1])


class Unrolled(nn.Module):
    def __init__(self, k=K):
        super().__init__()
        self.k, self.dn = k, Denoiser()
        self.log_mu = nn.Parameter(torch.tensor(0.0))      # learned DC weight

    def forward(self, y, mask):
        x = ifft2c(y)
        mu = self.log_mu.exp()
        for _ in range(self.k):
            z = self.dn(x)
            x = ifft2c((mask * y + mu * fft2c(z)) / (mask + mu))
        return x


def train(steps=1000, batch=8):
    mask = torch.from_numpy(mask_variable_density(N, R, acs=16).astype(np.float32)).to(dev)
    model = Unrolled().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for step in range(steps):
        x = random_ellipses(batch).to(dev)
        y = mask * fft2c(x)
        y = y + mask * 0.003 * torch.randn_like(y)
        loss = (model(y, mask) - x).abs().mean()           # L1 on the complex image
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"  step {step:4d}  loss {loss.item():.4f}  ({time.time() - t0:.0f} s)")
    return model, mask


if __name__ == "__main__":
    print(f"device: {dev}")
    model, mask = train()
    model.eval()
    gt = torch.from_numpy(shepp_logan(N).astype(np.complex64))[None].to(dev)
    y = mask * fft2c(gt)
    with torch.no_grad():
        zf, rec = ifft2c(y), model(y, mask)
    gt, zf, rec = (t[0].cpu().numpy() for t in (gt, zf, rec))
    print(f"Shepp-Logan, R={R}:  zero-filled NRMSE {nrmse(zf, gt):.3f}   unrolled net {nrmse(rec, gt):.3f}")
    torch.save(model.state_dict(), "unrolled_net.pt")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
    for a, im, t in zip(ax, [gt, mask.cpu().numpy(), zf, rec],
                        ["truth (never seen in training)", f"mask R={R}", "zero-filled", f"unrolled net (K={K})"]):
        a.imshow(np.abs(im), cmap="gray", vmax=1); a.set_title(t, fontsize=10); a.axis("off")
    plt.tight_layout(); plt.savefig("figures/06_dl_unrolled.png", dpi=110)
    print("saved figures/06_dl_unrolled.png")
