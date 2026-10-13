"""Diagnostics shared by the experiments and the export.  The browser re-implements the
same definitions in JavaScript (docs/js/physics.js) and is tested against these."""
from __future__ import annotations

import numpy as np

from .model import QG2L


def shell_matrix(m: QG2L) -> np.ndarray:
    """(nbins, ny*nkx) 0/1 matrix summing rfft2 modes into integer wavenumber shells.

    Shell n holds all modes with round(|k| / dk) == n, dk = 2 pi / Lx.  Shells 1..nx/2.
    """
    p = m.p
    dk = 2 * np.pi / p.Lx
    shell = np.rint(np.sqrt(m.K2) / dk).astype(int)
    nb = p.nx // 2
    S = np.zeros((nb, shell.size))
    flat = shell.ravel()
    for n in range(1, nb + 1):
        S[n - 1, flat == n] = 1.0
    return S


def ke_spectrum(m: QG2L, psi_hat: np.ndarray, S: np.ndarray | None = None) -> np.ndarray:
    """Kinetic-energy spectrum per layer, shape (..., 2, nbins); sums to the layer KE
    (except modes beyond shell nx/2, which are dealiased away)."""
    if S is None:
        S = shell_matrix(m)
    n2 = float(m.p.nx * m.p.ny) ** 2
    e = 0.5 * m.K2 * np.abs(psi_hat) ** 2 * m.wk / n2
    return e.reshape(e.shape[:-2] + (-1,)) @ S.T


def spectrum_of_difference(m: QG2L, dq_hat: np.ndarray, S: np.ndarray | None = None) -> np.ndarray:
    """Spectrum of the *difference* total energy (kinetic + available potential), summed over
    layers with the model's energy weights; shape (..., nbins)."""
    p = m.p
    if S is None:
        S = shell_matrix(m)
    ph = m.invert(dq_hat)
    n2 = float(p.nx * p.ny) ** 2
    ke = 0.5 * m.K2 * m.wk / n2 * (p.delta * np.abs(ph[..., 0, :, :]) ** 2
                                   + (1 - p.delta) * np.abs(ph[..., 1, :, :]) ** 2)
    ape = 0.5 * p.G * m.wk / n2 * np.abs(ph[..., 0, :, :] - ph[..., 1, :, :]) ** 2
    e = ke + ape
    return e.reshape(e.shape[:-2] + (-1,)) @ S.T


def velocities(m: QG2L, psi_hat: np.ndarray):
    """u = -dpsi/dy, v = dpsi/dx (physical space)."""
    return m.to_phys(-m.iky * psi_hat), m.to_phys(m.ikx * psi_hat)


class MeanAccumulator:
    """Time-mean statistics over a sequence of states (used by the parameter sweep)."""

    def __init__(self, m: QG2L):
        self.m = m
        self.S = shell_matrix(m)
        self.n = 0
        p = m.p
        self.u_zm = np.zeros((2, p.ny))
        self.eke_y = np.zeros((2, p.ny))
        self.uv_y = np.zeros((2, p.ny))
        self.vt_y = np.zeros(p.ny)
        self.spec = np.zeros((2, p.nx // 2))
        self.spec_tau = np.zeros(p.nx // 2)
        self.scal = {k: 0.0 for k in ("EKE1", "EKE2", "ZKE1", "ZKE2", "v_rms1", "u_rms1",
                                       "tau_rms", "z_rms1")}

    def add(self, qh: np.ndarray) -> None:
        m, p = self.m, self.m.p
        ph = m.invert(qh)
        u, v = velocities(m, ph)                       # (2, ny, nx)
        psi = m.to_phys(ph)
        tau = psi[0] - psi[1]
        ubar = u.mean(-1)
        up, vp = u - ubar[..., None], v
        taup = tau - tau.mean(-1, keepdims=True)
        self.u_zm += ubar
        self.eke_y += 0.5 * (up**2 + vp**2).mean(-1)
        self.uv_y += (up * vp).mean(-1)
        self.vt_y += (vp[0] * taup).mean(-1)
        self.spec += ke_spectrum(m, ph, self.S)
        d = ph[0] - ph[1]
        n2 = float(p.nx * p.ny) ** 2
        self.spec_tau += (0.5 * p.G * m.wk * np.abs(d) ** 2 / n2).ravel() @ self.S.T
        self.scal["EKE1"] += 0.5 * (up[0] ** 2 + vp[0] ** 2).mean()
        self.scal["EKE2"] += 0.5 * (up[1] ** 2 + vp[1] ** 2).mean()
        self.scal["ZKE1"] += 0.5 * (ubar[0] ** 2).mean()
        self.scal["ZKE2"] += 0.5 * (ubar[1] ** 2).mean()
        self.scal["v_rms1"] += float(np.sqrt((vp[0] ** 2).mean()))
        self.scal["u_rms1"] += float(np.sqrt((up[0] ** 2).mean()))
        self.scal["tau_rms"] += float(np.sqrt((taup**2).mean()))
        self.scal["z_rms1"] += float(p.f0 / p.g * np.sqrt(((psi[0] - psi[0].mean()) ** 2).mean()))
        self.n += 1

    def result(self) -> dict:
        n = max(self.n, 1)
        p = self.m.p
        out = {
            "u1_total": (self.u_zm[0] / n + p.U1).tolist(),
            "u2_total": (self.u_zm[1] / n + p.U2).tolist(),
            "eke1_y": (self.eke_y[0] / n).tolist(),
            "eke2_y": (self.eke_y[1] / n).tolist(),
            "uv1_y": (self.uv_y[0] / n).tolist(),
            "uv2_y": (self.uv_y[1] / n).tolist(),
            "vt_y": (self.vt_y / n).tolist(),
            "spec_ke1": (self.spec[0] / n).tolist(),
            "spec_ke2": (self.spec[1] / n).tolist(),
            "spec_ape": (self.spec_tau / n).tolist(),
        }
        out.update({k: v / n for k, v in self.scal.items()})
        return out
