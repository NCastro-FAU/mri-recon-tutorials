"""01 - GRAPPA: fill in the missing k-space lines from their neighbours.

Every missing point is a linear combination of the acquired points around it,
across all coils. The weights are learned on the fully sampled centre (ACS)
and are the same everywhere in k-space (the coil sensitivities are smooth, so
their k-space is a small convolution kernel).

Kernel here: 2 acquired rows (above/below) x 3 columns x all coils -> 1 point
per coil, one set of weights per offset 1..R-1.
"""
import numpy as np

from mrutils import add_noise, coil_maps, fft2c, ifft2c, mask_uniform, nrmse, rss, shepp_logan, show

N, NC, R, ACS = 128, 8, 3, 24


def grappa(kspace, mask_rows, R, acs, kx=3, lam=1e-3):
    nc, ny, nx = kspace.shape
    calib = kspace[:, ny // 2 - acs // 2: ny // 2 + acs // 2]
    out = kspace.copy()
    hx = kx // 2
    pad = np.pad(kspace, ((0, 0), (R, R), (hx, hx)))
    for off in range(1, R):
        # --- calibration: sources at rows -off and R-off, target at row 0
        src, tgt = [], []
        for y in range(off, acs - (R - off)):
            for x in range(hx, nx - hx):
                s = np.concatenate([calib[:, y - off, x - hx:x + hx + 1].ravel(),
                                    calib[:, y + R - off, x - hx:x + hx + 1].ravel()])
                src.append(s); tgt.append(calib[:, y, x])
        A, B = np.array(src), np.array(tgt)
        # Tikhonov-regularised least squares: W = (A^H A + lam I)^-1 A^H B
        AhA = A.conj().T @ A
        W = np.linalg.solve(AhA + lam * np.trace(AhA).real / len(AhA) * np.eye(len(AhA)), A.conj().T @ B)
        # --- synthesis: every missing row whose two neighbours were acquired
        for y in range(ny):
            if mask_rows[y]:
                continue
            ya, yb = y - off, y + R - off
            if not (0 <= ya < ny and 0 <= yb < ny and mask_rows[ya] and mask_rows[yb]):
                continue
            # all columns at once: sliding windows over x
            wa = np.lib.stride_tricks.sliding_window_view(pad[:, ya + R], kx, axis=-1)  # (nc, nx, kx)
            wb = np.lib.stride_tricks.sliding_window_view(pad[:, yb + R], kx, axis=-1)
            S = np.concatenate([wa.transpose(1, 0, 2).reshape(nx, -1),
                                wb.transpose(1, 0, 2).reshape(nx, -1)], axis=1)
            out[:, y] = (S @ W).T
    return out


if __name__ == "__main__":
    img = shepp_logan(N)
    sens = coil_maps(N, NC)
    k_full = add_noise(fft2c(sens * img), 0.01)
    mask = mask_uniform(N, R, ACS)
    k_us = k_full * mask

    zero_filled = rss(ifft2c(k_us))
    k_grappa = grappa(k_us, mask[:, 0], R, ACS)
    recon = rss(ifft2c(k_grappa))
    ref = rss(ifft2c(k_full))
    print(f"NRMSE zero-filled {nrmse(zero_filled, ref):.3f}  GRAPPA {nrmse(recon, ref):.3f}")
    show([ref, mask, zero_filled, recon, 5 * np.abs(recon - ref)],
         ["fully sampled (RSS)", f"mask R={R} + {ACS} ACS", "zero-filled", "GRAPPA", "|error| x5"],
         "figures/01_grappa.png")
