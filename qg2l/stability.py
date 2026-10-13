"""Linear stability of the two-layer basic state (normal modes exp(i(kx x + ky y - omega t)))."""
from __future__ import annotations

import numpy as np

from .model import Params


def eigenvalues(p: Params, kx: float, ky: float) -> np.ndarray:
    """Complex eigenvalues lambda of  d(psi_hat)/dt = lambda psi_hat  for one wavenumber.

    Re(lambda) is the growth rate (s^-1); -Im(lambda)/kx is the phase speed.
    """
    K2 = kx**2 + ky**2
    F1, F2 = p.F1, p.F2
    A = np.array([[-K2 - F1, F1], [F2, -K2 - F2]], dtype=complex)
    U = np.diag([p.U1, p.U2]).astype(complex)
    B = np.diag([p.beta1, p.beta2]).astype(complex)
    fric = np.diag([0.0, p.r_ek * K2]).astype(complex)
    M = -1j * kx * (U @ A + B) + fric
    return np.linalg.eigvals(np.linalg.solve(A, M))


def growth_rate(p: Params, kx: float, ky: float) -> float:
    return float(np.max(eigenvalues(p, kx, ky).real))


def most_unstable_eigenmode(p: Params, kx: float, ky: float):
    """Eigenvalue and (psi1, psi2) eigenvector of the fastest-growing normal mode."""
    K2 = kx**2 + ky**2
    F1, F2 = p.F1, p.F2
    A = np.array([[-K2 - F1, F1], [F2, -K2 - F2]], dtype=complex)
    U = np.diag([p.U1, p.U2]).astype(complex)
    B = np.diag([p.beta1, p.beta2]).astype(complex)
    fric = np.diag([0.0, p.r_ek * K2]).astype(complex)
    M = -1j * kx * (U @ A + B) + fric
    w, v = np.linalg.eig(np.linalg.solve(A, M))
    i = int(np.argmax(w.real))
    return w[i], v[:, i]


def growth_map(p: Params, nk: int = 120, kmax_factor: float = 3.0):
    """Growth rate (s^-1) on a regular (kx, ky) grid; returns kx, ky, sigma."""
    kd = np.sqrt(p.kd2)
    kx = np.linspace(kd / nk, kmax_factor * kd, nk)
    ky = np.linspace(-kmax_factor * kd, kmax_factor * kd, 2 * nk + 1)
    sig = np.zeros((ky.size, kx.size))
    for j, l in enumerate(ky):
        for i, k in enumerate(kx):
            sig[j, i] = growth_rate(p, k, l)
    return kx, ky, sig


def fastest_mode(p: Params, nk: int = 1500):
    """Most unstable zonal wavenumber (ky = 0): returns (k, sigma, e-folding time in days)."""
    kd = np.sqrt(p.kd2)
    ks = np.linspace(kd / nk, 3 * kd, nk * 3)
    sig = np.array([growth_rate(p, k, 0.0) for k in ks])
    i = int(np.argmax(sig))
    s = float(sig[i])
    return float(ks[i]), s, (1.0 / s / 86400.0 if s > 0 else float("inf"))


def phillips_growth_rate(U: float, F: float, k: float) -> float:
    """Analytic growth rate for beta = 0, r = 0, equal layers (F1 = F2 = F), U1 = -U2 = U, ky = 0.

        sigma = U k sqrt((2F - k^2) / (2F + k^2))      for k^2 < 2F
    """
    k2 = k * k
    if k2 >= 2 * F:
        return 0.0
    return U * k * np.sqrt((2 * F - k2) / (2 * F + k2))
