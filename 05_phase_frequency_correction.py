"""05 - Phase and frequency correction.

Part A - EPI Nyquist (N/2) ghost.
    EPI reads every other k-space line with the opposite gradient polarity. A
    small gradient delay or eddy current shifts odd and even echoes in
    opposite directions, which in the hybrid (x, ky) space is a linear phase
    phi0 + phi1 * x on every reversed line -> a ghost shifted by FOV/2.
    Fix: three navigator echoes without phase encoding (+, -, +), compare the
    two polarities in hybrid space, fit phi0 + phi1 x, remove it.

Part B - frequency drift during a Cartesian GRE.
    Gradient heating and breathing move B0 by a few Hz during the scan. Each
    line then carries an extra phase 2 pi df(line) t over its readout:
    a global phase at TE (-> ghosting along phase encoding) and a shift along
    readout. Fix: a short FID navigator before every readout gives df(line)
    from its phase; demodulate every sample with exp(-i 2 pi df t).
"""
import numpy as np

from mrutils import fft2c, ifft2c, nrmse, shepp_logan

N = 128
rng = np.random.default_rng(0)
x = (np.arange(N) - N / 2) / N                    # readout position in FOV units


# ------------------------------------------------------------------ part A
def epi_ghost_demo(phi0=0.6, phi1=6.0):
    img = shepp_logan(N)
    k = fft2c(img)
    hybrid = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(k, axes=1), axis=1, norm="ortho"), axes=1)  # (ky, x)
    err = np.exp(1j * (phi0 + phi1 * x))
    hybrid_bad = hybrid.copy(); hybrid_bad[1::2] *= err             # reversed lines
    # navigators: the ky = 0 line read with + and - polarity (the 2 "+" echoes averaged)
    nav_pos = hybrid[N // 2] + 0.002 * rng.standard_normal(N)
    nav_neg = hybrid[N // 2] * err + 0.002 * rng.standard_normal(N)
    # weighted linear fit of the phase difference
    d = nav_neg * nav_pos.conj()
    w = np.abs(d) > 0.1 * np.abs(d).max()
    slope = np.angle(np.sum(d[1:] * d[:-1].conj() * (w[1:] & w[:-1]))) / (x[1] - x[0])   # robust to wraps
    offset = np.angle(np.sum(d[w] * np.exp(-1j * slope * x[w])))
    hybrid_fix = hybrid_bad.copy(); hybrid_fix[1::2] *= np.exp(-1j * (offset + slope * x))
    to_img = lambda h: np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(h, axes=0), axis=0, norm="ortho"), axes=0)
    print(f"A  true phi0 {phi0:.3f} phi1 {phi1:.3f}  |  estimated {offset:.3f} {slope:.3f}")
    return img, to_img(hybrid_bad), to_img(hybrid_fix)


# ------------------------------------------------------------------ part B
def drift_demo(te=5e-3, dwell=10e-6, t_nav=2e-3):
    img = shepp_logan(N)
    k = fft2c(img)
    lines = np.arange(N)
    df = 25 * lines / N + 10 * np.sin(2 * np.pi * lines / 40)       # Hz: linear drift + breathing
    t = te + (np.arange(N) - N / 2) * dwell                            # sample times in a line
    data = k * np.exp(2j * np.pi * df[:, None] * t[None, :])
    data += 0.002 * (rng.standard_normal(data.shape) + 1j * rng.standard_normal(data.shape))
    # FID navigator: same off-resonance, sampled at t_nav after the pulse
    nav = np.sum(img) * np.exp(2j * np.pi * df * t_nav) * np.exp(1j * 0.3)   # 0.3 rad: unknown receiver phase
    nav += 0.01 * np.abs(nav[0]) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))   # 1 % noise
    df_est = np.angle(nav * nav[0].conj()) / (2 * np.pi * t_nav)       # relative to the first line
    fixed = data * np.exp(-2j * np.pi * df_est[:, None] * t[None, :])
    print(f"B  frequency estimate error: rms {np.sqrt(np.mean((df_est - (df - df[0])) ** 2)):.2f} Hz")
    return img, ifft2c(data), ifft2c(fixed), df - df[0], df_est


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    a_ref, a_bad, a_fix = epi_ghost_demo()
    b_ref, b_bad, b_fix, df, df_est = drift_demo()
    for name, im, ref in [("A ghosted", a_bad, a_ref), ("A corrected", a_fix, a_ref),
                          ("B drifting", b_bad, b_ref), ("B corrected", b_fix, b_ref)]:
        print(f"   {name:12s} NRMSE {nrmse(im, ref):.3f}")
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.9))
    kw = dict(cmap="gray", vmax=0.6)
    ax[0].imshow(np.abs(a_bad), **kw); ax[0].set_title("EPI: N/2 ghost")
    ax[1].imshow(np.abs(a_fix), **kw); ax[1].set_title("EPI: navigator corrected")
    ax[2].imshow(np.abs(b_bad), **kw); ax[2].set_title("GRE: B0 drift")
    ax[3].imshow(np.abs(b_fix), **kw); ax[3].set_title("GRE: FID-navigator corrected")
    for a in ax[:4]:
        a.axis("off")
    ax[4].plot(df, label="true"); ax[4].plot(df_est, ".", ms=3, label="navigator")
    ax[4].set_xlabel("phase-encode line"); ax[4].set_ylabel("frequency offset [Hz]"); ax[4].legend()
    plt.tight_layout(); plt.savefig("figures/05_phase_frequency.png", dpi=110)
    print("saved figures/05_phase_frequency.png")
