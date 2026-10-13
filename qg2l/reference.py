"""Reference numbers computed with numpy from the *exported* control data.

The JavaScript unit tests (tests/js) recompute the same quantities in the browser code and
compare them with these values, so that the in-browser analysis is validated against an
independent implementation, on the very data the website serves.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .analysis import ke_spectrum, shell_matrix
from .config import Config
from .export import Writer, read_array
from .model import QG2L
from .stability import most_unstable_eigenmode


def _moments(x: np.ndarray) -> dict[str, float]:
    x = x.ravel()
    mu, sd = x.mean(), x.std()
    z = (x - mu) / sd
    return {"mean": float(mu), "std": float(sd), "skew": float((z**3).mean()),
            "kurt": float((z**4).mean())}


def make_reference(cfg: Config, w: Writer, manifest: dict[str, Any]) -> dict[str, Any]:
    p = cfg.params
    m = QG2L(p)
    psi = read_array(w.outdir, manifest["control"]["psi"])            # (nt, 2, ny, nx) float64
    nt = psi.shape[0]
    S = shell_matrix(m)
    ph = m.to_spec(psi)                                               # all modes, no dealiasing
    n2 = float(p.nx * p.ny) ** 2

    def mean_sq(fh):
        return float(np.sum(m.wk * np.abs(fh) ** 2) / n2)

    def energetics(h):
        ke1 = 0.5 * mean_sq(np.sqrt(m.K2) * h[0])
        ke2 = 0.5 * mean_sq(np.sqrt(m.K2) * h[1])
        tau = h[0] - h[1]
        ape = 0.5 * p.G * mean_sq(tau)
        vt = float(np.sum(m.wk * np.real(np.conj(m.ikx * h[0]) * tau)) / n2)
        return {"KE1": ke1, "KE2": ke2, "APE": ape, "heat_flux": vt,
                "C_B": p.G * p.dU * vt, "D_E": (1 - p.delta) * p.r_ek * 2 * ke2}

    idx = sorted({0, nt // 4, nt // 2, nt - 1})
    snaps = []
    for i in idx:
        rec = {"index": i, **energetics(ph[i])}
        rec["spec_ke1"] = ke_spectrum(m, ph[i][None], S)[0, 0, :12].tolist()
        zeta = m.to_phys(-m.K2 * ph[i, 0])
        rec["zeta1_rms"] = float(np.sqrt((zeta**2).mean()))
        rec["psi1_rms"] = float(np.sqrt((psi[i, 0] ** 2).mean()))
        snaps.append(rec)

    u = m.to_phys(-m.iky * ph)                                        # (nt, 2, ny, nx)
    ubar = u.mean(-1)                                                 # (nt, 2, ny)
    zeta1 = m.to_phys(-m.K2 * ph[:, 0])
    x = ubar.reshape(nt, -1)
    xa = x - x.mean(0)
    cov = xa.T @ xa / nt
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    hist_edges = np.linspace(-4, 4, 17)
    zs = (zeta1 - zeta1.mean()) / zeta1.std()
    hist = np.histogram(zs.ravel(), bins=hist_edges)[0] / zs.size
    tmean_u1 = ubar[:, 0].mean(0)
    lin = []
    for mx in (1, 3, 5, 8):
        for my in (0, 2, 4):
            kx, ky = 2 * np.pi * mx / p.Lx, 2 * np.pi * my / p.Ly
            lam, vec = most_unstable_eigenmode(p, kx, ky)
            ratio = vec[1] / vec[0]
            lin.append({"mx": mx, "my": my, "kx": kx, "ky": ky, "sigma": float(lam.real),
                        "c": float(-lam.imag / kx), "ratio": float(abs(ratio)),
                        "lag": float(np.angle(ratio))})
    return {
        "linear_modes": lin,
        "snapshots": snaps,
        "zonal_mean_u1_timemean": tmean_u1[:: max(1, p.ny // 16)].tolist(),
        "eof_zonal_wind_variance_fractions": (ev[:4] / ev.sum()).tolist(),
        "eof_zonal_wind_total_variance": float(ev.sum()),
        "zeta1_moments": _moments(zeta1),
        "zeta1_hist_fraction": hist.tolist(),
        "zeta1_hist_edges": hist_edges.tolist(),
        "z1_moments": _moments(psi[:, 0] * p.f0 / p.g),
    }
