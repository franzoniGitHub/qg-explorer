"""Synthetic observing network and a serial ensemble square-root filter (EnSRF).

The state vector is the grid-point streamfunction of both layers, flattened to
(2 * ny * nx).  Stations sit at random off-grid positions; the observation operator is a
bilinear interpolation (psi) or a centred finite difference of it (winds).

Update: Whitaker & Hamill (2002) serial EnSRF with Gaspari & Cohn (1999) localisation.
"""
from __future__ import annotations

import numpy as np

from .model import Params

OBS_VARS = ("psi1", "psi2", "u1", "v1", "u2", "v2")


def gaspari_cohn(dist: np.ndarray, c: float) -> np.ndarray:
    """Gaspari-Cohn 5th-order compact correlation function; zero for dist >= 2c."""
    z = np.asarray(dist, dtype=float) / c
    out = np.zeros_like(z)
    m1 = z <= 1.0
    m2 = (z > 1.0) & (z < 2.0)
    z1, z2 = z[m1], z[m2]
    out[m1] = -0.25 * z1**5 + 0.5 * z1**4 + 0.625 * z1**3 - (5.0 / 3.0) * z1**2 + 1.0
    out[m2] = (z2**5 / 12.0 - 0.5 * z2**4 + 0.625 * z2**3 + (5.0 / 3.0) * z2**2
               - 5.0 * z2 + 4.0 - 2.0 / (3.0 * z2))
    return out


def random_stations(p: Params, n: int, seed: int) -> np.ndarray:
    """n station positions (x, y) in metres; nested: the first k stations of any n are the same."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(n, 2)) * np.array([p.Lx, p.Ly])


class ObsOperator:
    """Linear observation operator H: state vector (2*ny*nx) -> one value per station."""

    def __init__(self, p: Params, pos: np.ndarray, var: str):
        if var not in OBS_VARS:
            raise ValueError(f"unknown observation variable '{var}', choose from {OBS_VARS}")
        self.var = var
        nx, ny = p.nx, p.ny
        layer = int(var[-1]) - 1
        kind = var[:-1]
        fx = pos[:, 0] / p.dx - 0.5
        fy = pos[:, 1] / p.dy - 0.5
        i0 = np.floor(fx).astype(int)
        j0 = np.floor(fy).astype(int)
        ax, ay = fx - i0, fy - j0
        base = [(0, 0, (1 - ay) * (1 - ax)), (0, 1, (1 - ay) * ax),
                (1, 0, ay * (1 - ax)), (1, 1, ay * ax)]
        idx, wts = [], []
        for dj, di, w in base:
            j, i = j0 + dj, i0 + di
            if kind == "psi":
                stencil = [(0, 0, 1.0)]
            elif kind == "u":                       # u = -dpsi/dy
                stencil = [(1, 0, -0.5 / p.dy), (-1, 0, 0.5 / p.dy)]
            else:                                   # v = dpsi/dx
                stencil = [(0, 1, 0.5 / p.dx), (0, -1, -0.5 / p.dx)]
            for sj, si, sw in stencil:
                idx.append(layer * ny * nx + ((j + sj) % ny) * nx + ((i + si) % nx))
                wts.append(w * sw)
        self.idx = np.stack(idx, axis=1)            # (nobs, K)
        self.wts = np.stack(wts, axis=1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x: (..., 2*ny*nx) -> (..., nobs)"""
        return np.sum(x[..., self.idx] * self.wts, axis=-1)


class Network:
    """A set of stations, each observing every variable in `variables`."""

    def __init__(self, p: Params, pos: np.ndarray, variables: list[str],
                 sigma_psi: float, sigma_wind: float, loc_radius: float):
        self.p = p
        self.pos = pos
        self.variables = list(variables)
        self.ops = [ObsOperator(p, pos, v) for v in variables]
        self.sigma = np.array([sigma_psi if v.startswith("psi") else sigma_wind for v in variables])
        n = len(pos)
        # Gaspari-Cohn weights of each station over the (ny, nx) grid, periodic distance
        x = (np.arange(p.nx) + 0.5) * p.dx
        y = (np.arange(p.ny) + 0.5) * p.dy
        self.loc = np.empty((n, p.ny * p.nx))
        c = loc_radius / 2.0                        # weights vanish at 2c = loc_radius
        for s in range(n):
            dx = np.abs(x - pos[s, 0])
            dx = np.minimum(dx, p.Lx - dx)
            dy = np.abs(y - pos[s, 1])
            dy = np.minimum(dy, p.Ly - dy)
            d = np.sqrt(dy[:, None] ** 2 + dx[None, :] ** 2)
            self.loc[s] = gaspari_cohn(d, c).ravel()

    @property
    def nobs(self) -> int:
        return len(self.pos) * len(self.variables)

    def observe(self, x: np.ndarray) -> np.ndarray:
        """x (..., nstate) -> (..., nvar, nstations)"""
        return np.stack([op(x) for op in self.ops], axis=-2)

    def perturbed_obs(self, x_truth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        y = self.observe(x_truth)
        return y + self.sigma[:, None] * rng.standard_normal(y.shape)


def ensrf_update(X: np.ndarray, net: Network, y: np.ndarray, inflation: float = 1.0):
    """Serial EnSRF analysis.  X: (N, nstate) ensemble (not modified); y: (nvar, nstations).

    Returns the analysis ensemble.  Multiplicative inflation of the prior anomalies is local:
    the factor is ``1 + (inflation - 1) * w(x)`` with w the largest localisation weight of any
    station.  Where nothing is observed the spread is left to the model; a global factor would
    inflate it every cycle without anything to pull it back, and a sparse network then drives
    the ensemble far beyond the climatological variance.
    """
    N = X.shape[0]
    xm = X.mean(axis=0)
    w = net.loc.max(axis=0) if len(net.pos) else np.zeros(X.shape[1] // 2)
    factor = 1.0 + (inflation - 1.0) * np.concatenate([w, w])
    Xp = (X - xm) * factor
    for iv, op in enumerate(net.ops):
        R = net.sigma[iv] ** 2
        for s in range(len(net.pos)):
            idx, w = op.idx[s], op.wts[s]
            yp = Xp[:, idx] @ w                         # (N,)
            ym = float(xm[idx] @ w)
            var = float(yp @ yp) / (N - 1)
            if var <= 0.0:
                continue
            cov = (Xp.T @ yp) / (N - 1)                 # (nstate,)
            loc = net.loc[s]
            K = np.concatenate([loc, loc]) * cov / (var + R)
            xm = xm + K * (y[iv, s] - ym)
            alpha = 1.0 / (1.0 + np.sqrt(R / (var + R)))
            Xp = Xp - alpha * np.outer(yp, K)
    return xm + Xp
