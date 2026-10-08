"""02 - ESPIRiT: coil sensitivity maps from the k-space centre.

1. Slide a k x k window over the calibration region of all coils and stack the
   patches as rows of a (calibration) matrix.
2. Its dominant right-singular vectors span the "signal subspace": every valid
   multi-coil k-space patch lies in it. These vectors are k-space kernels.
3. Moved to image space, the kernels give at every pixel a small nc x nc matrix
   whose eigenvector with eigenvalue ~1 is the coil sensitivity at that pixel.
   Where no eigenvalue is close to 1 there is no signal -> the maps are masked.

Reference: Uecker et al., MRM 71:990 (2014).
"""
import numpy as np

from mrutils import add_noise, coil_maps, fft2c, ifft2c, rss, shepp_logan, show

N, NC, ACS = 128, 8, 24


def espirit(kspace, acs=24, k=6, thresh=0.02, crop=0.95):
    nc, ny, nx = kspace.shape
    calib = kspace[:, ny // 2 - acs // 2: ny // 2 + acs // 2, nx // 2 - acs // 2: nx // 2 + acs // 2]
    # 1. calibration matrix: one row per k x k patch, all coils
    patches = np.lib.stride_tricks.sliding_window_view(calib, (k, k), axis=(1, 2))   # (nc, a, a, k, k)
    A = patches.transpose(1, 2, 0, 3, 4).reshape(-1, nc * k * k)
    # 2. signal subspace
    _, s, vh = np.linalg.svd(A, full_matrices=False)
    vh = vh[s > thresh * s[0]]
    kernels = vh.reshape(-1, nc, k, k)
    print(f"  calibration matrix {A.shape}, kept {len(kernels)} of {len(s)} singular vectors")
    # 3. image-space kernels: zero-pad to the image size and inverse FFT.
    #    The rows of A are patches (a correlation, not a convolution), so the
    #    image-space eigenvectors come out conjugated: take conj at the end.
    kpad = np.zeros((len(kernels), nc, ny, nx), complex)
    y0, x0 = ny // 2 - k // 2, nx // 2 - k // 2
    kpad[:, :, y0:y0 + k, x0:x0 + k] = kernels
    img_k = ifft2c(kpad) * np.sqrt(ny * nx / (k * k))                # (r, nc, ny, nx)
    # per pixel: G = K^H K (nc x nc), eigen-decomposition
    G = np.einsum("rcyx,rdyx->yxcd", img_k.conj(), img_k)
    w, v = np.linalg.eigh(G)                                            # ascending eigenvalues
    maps = v[..., -1].transpose(2, 0, 1).conj()                         # (nc, ny, nx)
    eig = w[..., -1]
    maps = maps * np.exp(-1j * np.angle(maps[0]))                       # coil 0 sets the phase
    return maps * (eig > crop), eig


def sense_combine(coil_imgs, maps):
    """Optimal (SENSE-1) coil combination: sum_c conj(S_c) x_c / sum_c |S_c|^2."""
    return np.sum(maps.conj() * coil_imgs, 0) / np.maximum(np.sum(np.abs(maps) ** 2, 0), 1e-8)


if __name__ == "__main__":
    img = shepp_logan(N)
    sens = coil_maps(N, NC)
    k = add_noise(fft2c(sens * img), 0.005)
    maps, eig = espirit(k, ACS)

    coil_imgs = ifft2c(k)
    combined = sense_combine(coil_imgs, maps)
    # compare with the truth (up to a common phase per pixel)
    true_rel = sens * np.exp(-1j * np.angle(sens[0]))
    inside = img > 0
    err = np.linalg.norm((maps - true_rel)[:, inside]) / np.linalg.norm(true_rel[:, inside])
    print(f"relative error of the ESPIRiT maps inside the object: {err:.3f}")
    show([rss(coil_imgs), eig, *np.abs(maps[:3]), np.angle(maps[1]) * (eig > 0.95), combined],
         ["RSS", "eigenvalue map", "|map| coil 0", "|map| coil 1", "|map| coil 2", "phase coil 1", "SENSE-1 combination"],
         "figures/02_espirit.png")
