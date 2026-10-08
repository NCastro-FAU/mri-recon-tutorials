"""03 - PICS: parallel imaging + compressed sensing.

    x* = argmin_x  1/2 || M F S x - y ||^2  +  lambda || W x ||_1

S: ESPIRiT coil maps, F: FFT, M: sampling mask, W: orthonormal Haar wavelet.
Solved with FISTA (gradient step on the data term, soft-threshold in the
wavelet domain). Compared with CG-SENSE (lambda = 0) on a random
2D variable-density mask, where SENSE alone shows noise and incoherent aliasing.

The Haar transform is not shift invariant, so each iteration uses a random
circular shift ("cycle spinning"); this removes the blocky look of plain Haar.
"""
import importlib

import numpy as np

from mrutils import add_noise, coil_maps, fft2c, ifft2c, mask_poisson_like, nrmse, shepp_logan, show

espirit = importlib.import_module("02_espirit").espirit
N, NC, R = 128, 8, 6


# ---------- operators
def A(x, maps, mask):
    return mask * fft2c(maps * x)


def AH(y, maps, mask):
    return np.sum(maps.conj() * ifft2c(mask * y), 0)


def haar(x, levels=4):
    """Orthonormal 2D Haar transform (in place layout, like pywt's coefficient image)."""
    x = x.copy(); n = x.shape[0]
    for _ in range(levels):
        a = x[:n, :n]
        lo, hi = (a[0::2] + a[1::2]) / np.sqrt(2), (a[0::2] - a[1::2]) / np.sqrt(2)
        a = np.concatenate([lo, hi], 0)
        lo, hi = (a[:, 0::2] + a[:, 1::2]) / np.sqrt(2), (a[:, 0::2] - a[:, 1::2]) / np.sqrt(2)
        x[:n, :n] = np.concatenate([lo, hi], 1)
        n //= 2
    return x


def ihaar(x, levels=4):
    x = x.copy(); n = x.shape[0] >> (levels - 1)
    for _ in range(levels):
        a = x[:n, :n]
        h = n // 2
        lo, hi = a[:, :h], a[:, h:]
        b = np.empty_like(a); b[:, 0::2] = (lo + hi) / np.sqrt(2); b[:, 1::2] = (lo - hi) / np.sqrt(2)
        lo, hi = b[:h], b[h:]
        c = np.empty_like(b); c[0::2] = (lo + hi) / np.sqrt(2); c[1::2] = (lo - hi) / np.sqrt(2)
        x[:n, :n] = c
        n *= 2
    return x


def soft(z, t):
    mag = np.abs(z)
    return z * np.maximum(mag - t, 0) / np.maximum(mag, 1e-12)


# ---------- solvers
def cg_sense(y, maps, mask, iters=30):
    """Conjugate gradient on the normal equations A^H A x = A^H y."""
    b = AH(y, maps, mask); x = np.zeros_like(b); r = b.copy(); p = r.copy(); rr = np.vdot(r, r)
    for _ in range(iters):
        Ap = AH(A(p, maps, mask), maps, mask)
        alpha = rr / np.vdot(p, Ap)
        x += alpha * p; r -= alpha * Ap
        rr_new = np.vdot(r, r); p = r + rr_new / rr * p; rr = rr_new
    return x


def fista(y, maps, mask, lam, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    x = AH(y, maps, mask); z = x.copy(); t = 1.0
    step = 1.0                                     # ||A||^2 <= 1: orthonormal FFT, normalised maps
    for _ in range(iters):
        grad = AH(A(z, maps, mask) - y, maps, mask)
        sy, sx = rng.integers(0, 16, 2)                 # cycle spinning
        v = np.roll(z - step * grad, (sy, sx), (0, 1))
        x_new = np.roll(ihaar(soft(haar(v), lam * step)), (-sy, -sx), (0, 1))
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = x_new + (t - 1) / t_new * (x_new - x)
        x, t = x_new, t_new
    return x


if __name__ == "__main__":
    img = shepp_logan(N)
    sens = coil_maps(N, NC)
    k = add_noise(fft2c(sens * img), 0.01)
    mask = mask_poisson_like(N, R, acs=20)
    y = k * mask
    maps, _ = espirit(y, acs=20)
    print(f"acceleration {mask.size / mask.sum():.1f}")

    x_zf = AH(y, maps, mask)
    x_sense = cg_sense(y, maps, mask)
    x_pics = fista(y, maps, mask, lam=0.002)
    for name, x in [("zero-filled", x_zf), ("CG-SENSE", x_sense), ("PICS L1-wavelet", x_pics)]:
        print(f"  {name:16s} NRMSE {nrmse(x, img):.3f}")
    show([img, mask, x_zf, x_sense, x_pics],
         ["truth", f"2D VD mask R={mask.size / mask.sum():.1f}", "zero-filled", "CG-SENSE", "PICS (L1 wavelet)"],
         "figures/03_pics.png", vmax=1)
