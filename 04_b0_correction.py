"""04 - B0 (off-resonance) correction for a spiral readout.

With an off-resonance map f(r) [Hz] the signal is

    s(t) = sum_r  m(r) exp(-i 2 pi k(t).r) exp(-i 2 pi f(r) t)

A long spiral readout turns the extra phase into a blur around every pixel
with |f| > 0. Three reconstructions:

  1. ignore f                          (blurred)
  2. conjugate phase recon (CPR)       adjoint of the full model, with density compensation
  3. model-based CG                    solve the full model iteratively (exact)

The encoding matrix is built explicitly (64 x 64 image), so nothing is
approximated; in practice the same model is applied with a NUFFT and time
segmentation (a few L FFTs, each with exp(-i 2 pi f t_l)).
"""
import numpy as np

from mrutils import nrmse, shepp_logan

N, FOV = 64, 0.22                  # matrix, m
SHOTS, T_RO, DWELL = 4, 12e-3, 8e-6


def spiral(n=N, fov=FOV, shots=SHOTS, t_ro=T_RO, dwell=DWELL):
    """Archimedean spiral with roughly constant k-space speed: theta ~ sqrt(t)."""
    t = np.arange(int(t_ro / dwell)) * dwell
    kmax = n / (2 * fov)
    theta_max = 2 * np.pi * kmax * fov / shots
    theta = theta_max * np.sqrt(t / t_ro)
    k = [kmax * theta / theta_max * np.exp(1j * (theta + 2 * np.pi * s / shots)) for s in range(shots)]
    return np.concatenate(k), np.tile(t, shots)


def field_map(n=N):
    y, x = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    return 250 * np.exp(-((x - .3) ** 2 + (y + .2) ** 2) / .08) \
        - 180 * np.exp(-((x + .35) ** 2 + (y - .35) ** 2) / .05) + 40 * y   # Hz, like air-tissue interfaces at 3 T


def encoding(k, t, fmap, n=N, fov=FOV):
    r = (np.arange(n) - n / 2) / n * fov
    yy, xx = np.meshgrid(r, r, indexing="ij")
    phase = np.outer(k.real, xx.ravel()) + np.outer(k.imag, yy.ravel())
    E = np.exp(-2j * np.pi * phase).astype(np.complex64)
    if fmap is not None:
        E *= np.exp(-2j * np.pi * np.outer(t, fmap.ravel())).astype(np.complex64)
    return E


def dcf(k, shots=SHOTS):
    """Meyer's analytic spiral weights |g| |sin(angle(g) - angle(k))|."""
    kk = k.reshape(shots, -1)
    g = np.gradient(kk, axis=1)
    return (np.abs(g) * np.abs(np.sin(np.angle(g) - np.angle(kk)))).ravel()


def cg(E, s, iters=30, lam=1e-3):
    """CG on (E^H E + lam I) x = E^H s."""
    b = E.conj().T @ s
    x = np.zeros_like(b); r = b.copy(); p = r.copy(); rr = np.vdot(r, r)
    scale = np.linalg.norm(E @ b) ** 2 / np.linalg.norm(b) ** 2       # ~ largest eigenvalue
    for _ in range(iters):
        Ap = E.conj().T @ (E @ p) + lam * scale * p
        a = rr / np.vdot(p, Ap)
        x += a * p; r -= a * Ap
        rr_new = np.vdot(r, r); p = r + rr_new / rr * p; rr = rr_new
    return x


if __name__ == "__main__":
    img = shepp_logan(N)
    fmap = field_map(N)
    k, t = spiral()
    print(f"{len(k)} samples, readout {T_RO * 1e3:.0f} ms per shot, field map {fmap.min():.0f}..{fmap.max():.0f} Hz")

    E_true = encoding(k, t, fmap)
    rng = np.random.default_rng(0)
    s = E_true @ img.ravel().astype(np.complex64)
    s += 0.002 * np.abs(s).max() * (rng.standard_normal(len(s)) + 1j * rng.standard_normal(len(s)))

    E_0 = encoding(k, t, None)
    w = dcf(k)
    x_none = cg(E_0, s).reshape(N, N)
    x_cpr = (E_true.conj().T @ (w * s)).reshape(N, N)
    x_mb = cg(E_true, s).reshape(N, N)
    for name, x in [("no correction (CG)", x_none), ("conjugate phase", x_cpr), ("model-based CG", x_mb)]:
        print(f"  {name:20s} NRMSE {nrmse(x, img):.3f}")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.9))
    ax[0].imshow(img, cmap="gray", vmax=1.1); ax[0].set_title("truth")
    h = ax[1].imshow(fmap, cmap="RdBu_r", vmin=-250, vmax=250); ax[1].set_title("field map [Hz]")
    plt.colorbar(h, ax=ax[1], fraction=0.046)
    for a, x, ttl in zip(ax[2:], [x_none, x_cpr, x_mb], ["no B0 correction", "conjugate phase", "model-based CG"]):
        x = np.abs(x); x *= np.vdot(x, img) / np.vdot(x, x)          # same scale as the truth
        a.imshow(x, cmap="gray", vmin=0, vmax=1.1); a.set_title(ttl)
    for a in ax:
        a.axis("off")
    plt.tight_layout(); plt.savefig("figures/04_b0_correction.png", dpi=110)
    print("saved figures/04_b0_correction.png")
