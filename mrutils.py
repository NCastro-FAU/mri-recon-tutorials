"""Small helpers shared by the tutorials: phantom, coils, FFTs, masks, metrics."""
import numpy as np


def fft2c(x, axes=(-2, -1)):
    """Centred, orthonormal 2D FFT (image -> k-space)."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=axes), axes=axes, norm="ortho"), axes=axes)


def ifft2c(k, axes=(-2, -1)):
    """Centred, orthonormal 2D inverse FFT (k-space -> image)."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, axes=axes), axes=axes, norm="ortho"), axes=axes)


def shepp_logan(n=128):
    """Modified Shepp-Logan phantom (Toft's intensities), values in [0, 1]."""
    ellipses = [  # intensity, a, b, x0, y0, phi
        (1.0, .69, .92, 0, 0, 0), (-.8, .6624, .874, 0, -.0184, 0),
        (-.2, .11, .31, .22, 0, -18), (-.2, .16, .41, -.22, 0, 18),
        (.1, .21, .25, 0, .35, 0), (.1, .046, .046, 0, .1, 0), (.1, .046, .046, 0, -.1, 0),
        (.1, .046, .023, -.08, -.605, 0), (.1, .023, .023, 0, -.606, 0), (.1, .023, .046, .06, -.605, 0)]
    y, x = np.mgrid[1:-1:n * 1j, -1:1:n * 1j]
    img = np.zeros((n, n))
    for val, a, b, x0, y0, phi in ellipses:
        p = np.deg2rad(phi)
        xr = (x - x0) * np.cos(p) + (y - y0) * np.sin(p)
        yr = -(x - x0) * np.sin(p) + (y - y0) * np.cos(p)
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1] += val
    return img


def coil_maps(n=128, nc=8, radius=1.3, seed=0):
    """Smooth, normalised birdcage-like coil sensitivities (nc, n, n), complex."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    maps = []
    for c in range(nc):
        ang = 2 * np.pi * c / nc
        cx, cy = radius * np.cos(ang), radius * np.sin(ang)
        mag = 1 / np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + 0.3)
        phase = np.exp(1j * (ang + 0.6 * (x * np.cos(ang) + y * np.sin(ang)) + rng.uniform(0, 0.3)))
        maps.append(mag * phase)
    maps = np.array(maps)
    return maps / np.sqrt(np.sum(np.abs(maps) ** 2, axis=0))


def add_noise(k, sigma, seed=1):
    rng = np.random.default_rng(seed)
    return k + sigma * (rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape)) / np.sqrt(2)


def mask_uniform(n, R, acs=24):
    """1D uniform undersampling along the phase-encode (row) direction + ACS block."""
    m = np.zeros((n, n), bool)
    m[::R] = True
    m[n // 2 - acs // 2: n // 2 + acs // 2] = True
    return m


def mask_variable_density(n, R, acs=16, seed=0):
    """1D variable-density random mask along phase encoding (for compressed sensing / DL)."""
    rng = np.random.default_rng(seed)
    ky = np.abs(np.arange(n) - n / 2) / (n / 2)
    centre = np.zeros(n, bool); centre[n // 2 - acs // 2: n // 2 + acs // 2] = True
    target = n / R - acs                                  # random lines outside the centre
    pdf = (1 - ky) ** 2 * ~centre
    p = pdf
    for _ in range(50):                                   # rescale until the clipped pdf hits the target
        p = np.clip(pdf * target / pdf.sum(), 0, 1)
        if abs(p.sum() - target) < 0.5:
            break
        pdf = p
    rows = (rng.random(n) < p) | centre
    return np.repeat(rows[:, None], n, axis=1)


def mask_poisson_like(n, R, acs=20, seed=0):
    """2D variable-density random mask (think ky-kz of a 3D scan) with a full centre."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    pdf = (1 - np.clip(np.hypot(x, y) / np.sqrt(2), 0, 1)) ** 4
    target = n * n / R
    for _ in range(50):                                  # scale the pdf to hit the target R
        p = np.clip(pdf * target / pdf.sum(), 0, 1)
        if abs(p.sum() - target) < 1:
            break
        pdf = p
    m = rng.random((n, n)) < p
    m[n // 2 - acs // 2: n // 2 + acs // 2, n // 2 - acs // 2: n // 2 + acs // 2] = True
    return m


def rss(x, axis=0):
    return np.sqrt(np.sum(np.abs(x) ** 2, axis=axis))


def nrmse(x, ref):
    x, ref = np.abs(x), np.abs(ref)
    x = x * np.vdot(x, ref).real / np.vdot(x, x).real      # best scaling
    return np.linalg.norm(x - ref) / np.linalg.norm(ref)


def show(images, titles, fname, vmax=None, cmap="gray"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(images), figsize=(3.2 * len(images), 3.4))
    for a, im, t in zip(np.atleast_1d(ax), images, titles):
        a.imshow(np.abs(im), cmap=cmap, vmax=vmax); a.set_title(t, fontsize=10); a.axis("off")
    plt.tight_layout(); plt.savefig(fname, dpi=110); plt.close()
    print("saved", fname)
