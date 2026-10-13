"""Two-layer quasi-geostrophic (QG) model on a doubly periodic beta-plane.

This is a Phillips-type model: two layers of constant density, a prescribed
uniform mean flow in each layer (hence a uniform vertical shear), the beta
effect, linear Ekman drag on the lower layer, and scale-selective damping.

Prognostic variable
-------------------
The potential-vorticity (PV) anomaly ``q_i`` of each layer (i = 1 upper, 2 lower).

    q1 = lap(psi1) + F1 (psi2 - psi1)
    q2 = lap(psi2) + F2 (psi1 - psi2)

    dq1/dt + J(psi1, q1) + U1 dq1/dx + beta1 dpsi1/dx                  = D
    dq2/dt + J(psi2, q2) + U2 dq2/dx + beta2 dpsi2/dx = -r lap(psi2) + D

with the *background* PV gradients

    beta1 = beta + F1 (U1 - U2)       beta2 = beta - F2 (U1 - U2)

so that the shear ``U1 - U2`` is what makes the basic state baroclinically
unstable once ``beta2 < 0`` (the Charney-Stern / Phillips criterion).

Numerics
--------
Pseudo-spectral in x and y, 2/3-rule (circular) dealiasing, classical RK4 in
time, and an exponential hyper-viscous filter applied once per step.  Every
function accepts arrays with arbitrary leading "batch" dimensions, so an
ensemble of N members is just an array of shape (N, 2, ny, nkx) and is
integrated with one call.

Units are SI throughout (m, s); ``psi`` is in m^2 s^-1.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

DAY = 86400.0

# scipy.fft is ~2x faster than numpy.fft for batches and can use threads; it is optional.
try:  # pragma: no cover - depends on the environment
    import scipy.fft as _sfft

    def _rfft2(f, workers):
        return _sfft.rfft2(f, axes=(-2, -1), workers=workers)

    def _irfft2(fh, s, workers):
        return _sfft.irfft2(fh, s=s, axes=(-2, -1), workers=workers)
except ImportError:  # pragma: no cover
    def _rfft2(f, workers):
        return np.fft.rfft2(f, axes=(-2, -1))

    def _irfft2(fh, s, workers):
        return np.fft.irfft2(fh, s=s, axes=(-2, -1))


@dataclass
class Params:
    """Physical and numerical parameters.  All lengths in m, times in s."""

    # grid / domain
    nx: int = 64
    ny: int = 64
    Lx: float = 1.2e7
    Ly: float = 1.2e7
    # physics
    f0: float = 1.0e-4           # Coriolis parameter (s^-1)
    beta: float = 1.6e-11        # df/dy (m^-1 s^-1)
    Ld: float = 4.5e5            # internal deformation radius, Ld^-2 = F1 + F2 (m)
    delta: float = 0.5           # H1 / (H1 + H2): upper-layer depth fraction
    U1: float = 25.0             # upper-layer mean zonal wind (m/s)
    U2: float = 5.0              # lower-layer mean zonal wind (m/s)
    r_ek: float = 1.0 / (5.0 * DAY)   # lower-layer linear (Ekman) drag (s^-1)
    g: float = 9.81              # only used to convert psi to geopotential height
    # numerics
    dt: float = 1200.0           # time step (s)
    filter_tau: float = 6.0 * 3600.0  # hyper-viscous damping time at the truncation scale (s)

    # ---- derived quantities -------------------------------------------------
    @property
    def kd2(self) -> float:
        return 1.0 / self.Ld**2

    @property
    def F1(self) -> float:
        return self.kd2 * (1.0 - self.delta)

    @property
    def F2(self) -> float:
        return self.kd2 * self.delta

    @property
    def G(self) -> float:
        """Energy weight of the interface term:  delta*F1 == (1-delta)*F2."""
        return self.kd2 * self.delta * (1.0 - self.delta)

    @property
    def dU(self) -> float:
        return self.U1 - self.U2

    @property
    def beta1(self) -> float:
        return self.beta + self.F1 * self.dU

    @property
    def beta2(self) -> float:
        return self.beta - self.F2 * self.dU

    @property
    def dU_crit(self) -> float:
        """Long-wave critical shear, beta / F2 (Phillips 1954)."""
        return self.beta / self.F2

    @property
    def dx(self) -> float:
        return self.Lx / self.nx

    @property
    def dy(self) -> float:
        return self.Ly / self.ny

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Params":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise KeyError(f"unknown model parameter(s): {sorted(unknown)}")
        return cls(**d)

    def replace(self, **kw: Any) -> "Params":
        d = self.to_dict()
        d.update(kw)
        return Params.from_dict(d)


class QG2L:
    """Spectral two-layer QG model.  State = rfft2 of the PV anomaly, shape (..., 2, ny, nkx)."""

    def __init__(self, p: Params, workers: int = 1):
        self.p = p
        self.workers = workers          # FFT threads (only used when scipy is installed)
        nx, ny = p.nx, p.ny
        self.nkx = nx // 2 + 1
        kx = 2 * np.pi / p.Lx * np.fft.rfftfreq(nx, 1.0 / nx)       # (nkx,)
        ky = 2 * np.pi / p.Ly * np.fft.fftfreq(ny, 1.0 / ny)        # (ny,)
        self.kx = kx[None, :] * np.ones((ny, 1))
        self.ky = ky[:, None] * np.ones((1, self.nkx))
        self.K2 = self.kx**2 + self.ky**2
        self.ikx = 1j * self.kx
        self.iky = 1j * self.ky

        # Inverse of the 2x2 PV-inversion matrix, per wavenumber.
        F1, F2 = p.F1, p.F2
        with np.errstate(divide="ignore", invalid="ignore"):
            det = self.K2 * (self.K2 + p.kd2)
            self.a11 = -(self.K2 + F2) / det
            self.a12 = -F1 / det
            self.a21 = -F2 / det
            self.a22 = -(self.K2 + F1) / det
        for a in (self.a11, self.a12, self.a21, self.a22):
            a[0, 0] = 0.0                      # no domain-mean streamfunction

        # normalised wavenumber radius: 1 at the Nyquist wavenumber
        kxn, kyn = np.pi / p.dx, np.pi / p.dy
        kappa = np.sqrt((self.kx / kxn) ** 2 + (self.ky / kyn) ** 2)
        self.kappa = kappa
        self.mask = (kappa <= 2.0 / 3.0).astype(float)          # 2/3 dealiasing
        # scale-selective damping (del^8): e-folding time `filter_tau` at the truncation
        # wavenumber (kappa = 2/3), negligible at synoptic scales.
        self.damp_rate = (kappa / (2.0 / 3.0)) ** 8 / p.filter_tau      # s^-1
        self.filt = self.mask * np.exp(-self.damp_rate * p.dt)
        self.filt[0, 0] = 0.0                                   # keep the domain-mean PV exactly 0

        # rfft2 Parseval weights: columns 0 (and Nyquist if nx even) count once.
        w = np.full(self.nkx, 2.0)
        w[0] = 1.0
        if nx % 2 == 0:
            w[-1] = 1.0
        self.wk = w[None, :] * np.ones((ny, 1))

        # steady linear coefficients
        self._U = np.array([p.U1, p.U2])[:, None, None]
        self._beta = np.array([p.beta1, p.beta2])[:, None, None]

    # ---- transforms -----------------------------------------------------------
    def to_phys(self, fh: np.ndarray) -> np.ndarray:
        return _irfft2(fh, (self.p.ny, self.p.nx), self.workers)

    def to_spec(self, f: np.ndarray) -> np.ndarray:
        return _rfft2(f, self.workers)

    # ---- PV inversion ---------------------------------------------------------
    def invert(self, qh: np.ndarray) -> np.ndarray:
        """psi_hat from q_hat."""
        q1, q2 = qh[..., 0, :, :], qh[..., 1, :, :]
        out = np.empty_like(qh)
        out[..., 0, :, :] = self.a11 * q1 + self.a12 * q2
        out[..., 1, :, :] = self.a21 * q1 + self.a22 * q2
        return out

    def pv_from_psi(self, ph: np.ndarray) -> np.ndarray:
        p = self.p
        p1, p2 = ph[..., 0, :, :], ph[..., 1, :, :]
        out = np.empty_like(ph)
        out[..., 0, :, :] = -(self.K2 + p.F1) * p1 + p.F1 * p2
        out[..., 1, :, :] = p.F2 * p1 - (self.K2 + p.F2) * p2
        return out

    # ---- dynamics -------------------------------------------------------------
    def rhs(self, qh: np.ndarray) -> np.ndarray:
        """Tendency dq_hat/dt of all non-filter terms."""
        p = self.p
        ph = self.invert(qh)
        # one batched inverse transform for psi_x, psi_y, q_x, q_y
        stack = np.stack([self.ikx * ph, self.iky * ph, self.ikx * qh, self.iky * qh])
        ph_x, ph_y, qh_x, qh_y = self.to_phys(stack)
        jac = ph_x * qh_y - ph_y * qh_x                         # J(psi, q)
        jh = self.to_spec(jac) * self.mask
        tend = -jh - self.ikx * (self._U * qh + self._beta * ph)
        tend[..., 1, :, :] += p.r_ek * self.K2 * ph[..., 1, :, :]   # -r lap(psi2)
        return tend

    def step(self, qh: np.ndarray) -> np.ndarray:
        dt = self.p.dt
        k1 = self.rhs(qh)
        k2 = self.rhs(qh + 0.5 * dt * k1)
        k3 = self.rhs(qh + 0.5 * dt * k2)
        k4 = self.rhs(qh + dt * k3)
        return (qh + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)) * self.filt

    def run(self, qh: np.ndarray, nsteps: int) -> np.ndarray:
        for _ in range(nsteps):
            qh = self.step(qh)
        return qh

    def steps_for(self, seconds: float) -> int:
        n = int(round(seconds / self.p.dt))
        if abs(n * self.p.dt - seconds) > 1e-6 * max(seconds, 1.0):
            raise ValueError(f"{seconds} s is not a multiple of dt = {self.p.dt} s")
        return n

    # ---- initial conditions ---------------------------------------------------
    def random_state(self, rng: np.random.Generator, amp: float = 1e5,
                     kpeak_wavelengths: float = 4000e3, batch: tuple[int, ...] = ()) -> np.ndarray:
        """Smooth random PV field with ~`amp` m^2/s streamfunction amplitude."""
        shape = batch + (2, self.p.ny, self.p.nx)
        white = rng.standard_normal(shape)
        wh = self.to_spec(white)
        k0 = 2 * np.pi / kpeak_wavelengths
        shape_fn = np.exp(-self.K2 / (2 * k0**2)) * (self.K2 > 0)
        ph = wh * shape_fn
        ph = ph * self.mask
        ph[..., 0, 0] = 0.0
        psi = self.to_phys(ph)
        psi *= amp / psi.std()
        return self.pv_from_psi(self.to_spec(psi)) * self.mask

    def perturbation(self, rng: np.random.Generator, amp: float, n: int,
                     wavelength: float = 4000e3) -> np.ndarray:
        """n independent smooth psi perturbations (std `amp` m^2/s), returned as PV_hat."""
        return self.random_state(rng, amp=amp, kpeak_wavelengths=wavelength, batch=(n,))

    # ---- diagnostics (all domain means, energy per unit mass: m^2 s^-2) --------
    def _msq(self, fh: np.ndarray) -> np.ndarray:
        """Domain mean of f^2 from rfft2 coefficients (Parseval)."""
        n2 = float(self.p.nx * self.p.ny) ** 2
        return np.sum(self.wk * np.abs(fh) ** 2, axis=(-2, -1)) / n2

    def _mcross(self, ah: np.ndarray, bh: np.ndarray) -> np.ndarray:
        n2 = float(self.p.nx * self.p.ny) ** 2
        return np.sum(self.wk * np.real(np.conj(ah) * bh), axis=(-2, -1)) / n2

    def energetics(self, qh: np.ndarray) -> dict[str, np.ndarray]:
        """Energy content and the terms of the perturbation energy budget.

        E = delta*KE1 + (1-delta)*KE2 + APE,   KE_i = <|grad psi_i|^2>/2,
        APE = G <(psi1-psi2)^2>/2,  G = delta F1.

        dE/dt = C_B - D_E - D_h  with
            C_B = G (U1-U2) <v tau>        (conversion of mean-flow APE, tau = psi1-psi2)
            D_E = (1-delta) r <|grad psi2|^2>   (Ekman drag)
            D_h = hyper-viscous (numerical) dissipation
        """
        p = self.p
        ph = self.invert(qh)
        p1, p2 = ph[..., 0, :, :], ph[..., 1, :, :]
        k2 = self.K2
        ke1 = 0.5 * self._mcross(np.sqrt(k2) * p1, np.sqrt(k2) * p1)
        ke2 = 0.5 * self._mcross(np.sqrt(k2) * p2, np.sqrt(k2) * p2)
        tau = p1 - p2
        ape = 0.5 * p.G * self._msq(tau)
        # <v tau> with v1 = d psi1 / dx
        vt = self._mcross(self.ikx * p1, tau)
        c_b = p.G * p.dU * vt
        d_e = (1.0 - p.delta) * p.r_ek * 2.0 * ke2
        lam = self.damp_rate
        d_h = -(p.delta * self._mcross(p1, lam * qh[..., 0, :, :])
                + (1.0 - p.delta) * self._mcross(p2, lam * qh[..., 1, :, :]))
        ke = p.delta * ke1 + (1.0 - p.delta) * ke2
        return {"KE1": ke1, "KE2": ke2, "KE": ke, "APE": ape, "E": ke + ape,
                "heat_flux": vt, "C_B": c_b, "D_E": d_e, "D_h": d_h}

    def psi(self, qh: np.ndarray) -> np.ndarray:
        return self.to_phys(self.invert(qh))

    def q_from_psi(self, psi: np.ndarray) -> np.ndarray:
        return self.pv_from_psi(self.to_spec(psi)) * self.mask

    @property
    def x(self) -> np.ndarray:
        return (np.arange(self.p.nx) + 0.5) * self.p.dx

    @property
    def y(self) -> np.ndarray:
        return (np.arange(self.p.ny) + 0.5) * self.p.dy
