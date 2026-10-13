"""Validation of the two-layer QG model against theory and its own conservation laws."""
import numpy as np
import pytest

from qg2l import DAY, QG2L, Params
from qg2l.stability import (growth_rate, most_unstable_eigenmode,
                            phillips_growth_rate)


# --------------------------------------------------------------------------- linear theory
def test_phillips_analytic_growth_rate():
    """beta = 0, equal layers, no drag: sigma = U k sqrt((2F-k^2)/(2F+k^2))."""
    p = Params(beta=0.0, r_ek=0.0, delta=0.5, U1=10.0, U2=-10.0)
    F = p.F1
    for f in (0.1, 0.3, 0.5, 0.7, 0.9):
        k = f * np.sqrt(2 * F)
        assert growth_rate(p, k, 0.0) == pytest.approx(phillips_growth_rate(10.0, F, k), rel=1e-9)


def test_critical_shear_is_beta_over_F2():
    """Onset of instability at (U1-U2) = beta/F2 (Phillips 1954), scanned over wavenumbers."""
    p = Params(r_ek=0.0)
    kd = np.sqrt(p.kd2)

    def threshold(k):
        lo, hi = 0.2 * p.dU_crit, 50 * p.dU_crit
        if growth_rate(p.replace(U1=p.U2 + hi), k, 0.0) <= 1e-12:
            return np.inf
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if growth_rate(p.replace(U1=p.U2 + mid), k, 0.0) > 1e-12:
                hi = mid
            else:
                lo = mid
        return hi

    th = min(threshold(k) for k in np.linspace(0.3, 1.1, 33) * kd)
    assert th == pytest.approx(p.dU_crit, rel=5e-3)


def test_numerical_model_reproduces_linear_normal_mode():
    """Integrate a single unstable normal mode and compare growth rate and phase speed."""
    p = Params(Ld=6e5, Lx=1.6e7, Ly=1.6e7, U1=20.5, U2=5.0)
    m = QG2L(p)
    mx = 3                                           # zonal wavenumber 3 -> 5333 km
    kx, ky = 2 * np.pi * mx / p.Lx, 0.0
    lam, vec = most_unstable_eigenmode(p, kx, ky)
    assert lam.real > 0
    X = m.x[None, :] * np.ones((p.ny, 1))
    amp = 1e2
    psi0 = np.stack([np.real(vec[i] * np.exp(1j * kx * X)) for i in range(2)])
    psi0 *= amp / np.abs(psi0).max()
    q = m.q_from_psi(psi0)

    def mode_amp(qh):
        ph = m.invert(qh)
        return np.abs(ph[0, 0, mx])                  # |psi1_hat| at (ky=0, kx=mx)

    n = m.steps_for(4 * DAY)
    a0 = mode_amp(q)
    q = m.run(q, n)
    a1 = mode_amp(q)
    sigma_num = np.log(a1 / a0) / (4 * DAY)
    assert sigma_num == pytest.approx(lam.real, rel=2e-3)
    # phase speed
    ph = m.invert(q)[0, 0, mx]
    ph0 = m.invert(m.q_from_psi(psi0))[0, 0, mx]
    dphi = np.angle(ph / ph0)                        # exp(-i omega t) -> phase = -omega t
    omega_num = -dphi / (4 * DAY)
    omega_th = -lam.imag
    assert omega_num == pytest.approx(omega_th, rel=2e-2)


# --------------------------------------------------------------------------- conservation
@pytest.fixture(scope="module")
def nonlinear_state():
    p = Params()
    m = QG2L(p)
    q = m.random_state(np.random.default_rng(5), amp=3e6)
    return m, q


def test_jacobian_conserves_energy_and_enstrophy(nonlinear_state):
    m, q = nonlinear_state
    p = m.p
    ph = m.invert(q)
    stack = np.stack([m.ikx * ph, m.iky * ph, m.ikx * q, m.iky * q])
    px, py, qx, qy = m.to_phys(stack)
    jh = m.to_spec(px * qy - py * qx) * m.mask
    w = [p.delta, 1 - p.delta]
    e_rate = sum(w[i] * m._mcross(ph[i], jh[i]) for i in range(2))
    z_rate = sum(w[i] * m._mcross(q[i], jh[i]) for i in range(2))
    scale_e = m.energetics(q)["E"] * 1e-5
    assert abs(e_rate) < 1e-12 * max(scale_e, 1e-30) + 1e-18
    assert abs(z_rate) < 1e-20


def test_energy_budget_tendency_closes_exactly(nonlinear_state):
    """dE/dt computed from the model tendency equals C_B - D_E (analytic budget)."""
    m, q = nonlinear_state
    p = m.p
    ph = m.invert(q)
    tend = m.rhs(q)
    w = [p.delta, 1 - p.delta]
    dEdt = sum(-w[i] * m._mcross(ph[i], tend[i]) for i in range(2))
    e = m.energetics(q)
    assert dEdt == pytest.approx(e["C_B"] - e["D_E"], rel=1e-9)


def test_energy_budget_over_a_step(nonlinear_state):
    """Finite-difference dE/dt over one RK4+filter step matches C_B - D_E - D_h."""
    m, q = nonlinear_state
    q = m.run(q, 100)
    e0 = m.energetics(q)
    q1 = m.step(q)
    e1 = m.energetics(q1)
    dEdt = (e1["E"] - e0["E"]) / m.p.dt
    rhs = 0.5 * ((e0["C_B"] - e0["D_E"] - e0["D_h"]) + (e1["C_B"] - e1["D_E"] - e1["D_h"]))
    assert dEdt == pytest.approx(rhs, rel=2e-2, abs=1e-9)


def test_domain_mean_pv_is_zero_and_state_real(nonlinear_state):
    m, q = nonlinear_state
    q = m.run(q, 50)
    assert np.abs(q[..., 0, 0]).max() == 0.0
    back = m.to_spec(m.to_phys(q))
    assert np.allclose(back * m.mask, q, atol=1e-18 + 1e-12 * np.abs(q).max())


# --------------------------------------------------------------------------- numerics
def test_rk4_time_step_convergence():
    p = Params()
    q0 = QG2L(p).random_state(np.random.default_rng(7), amp=2e6)
    ref = QG2L(p.replace(dt=150.0))
    qr = ref.run(q0, ref.steps_for(DAY))
    errs = []
    for dt in (1200.0, 600.0):
        m = QG2L(p.replace(dt=dt))
        errs.append(np.abs(m.psi(m.run(q0, m.steps_for(DAY))) - m.psi(qr)).max())
    assert errs[0] < 1e-2 * np.abs(ref.psi(qr)).max()
    assert errs[1] < errs[0]                      # error decreases with dt


def test_batched_ensemble_equals_individual_runs():
    p = Params()
    m = QG2L(p)
    rng = np.random.default_rng(11)
    qb = m.random_state(rng, amp=1e6, batch=(3,))
    out_b = m.run(qb, 40)
    for i in range(3):
        assert np.allclose(m.run(qb[i], 40), out_b[i], rtol=1e-9, atol=1e-20)


def test_subcritical_flow_decays_and_supercritical_grows():
    base = Params(Ld=6e5, Lx=1.6e7, Ly=1.6e7)
    for ratio, grows in ((0.8, False), (1.4, True)):
        p = base.replace(U1=base.U2 + ratio * base.dU_crit)
        m = QG2L(p)
        q = m.random_state(np.random.default_rng(2), amp=1e4)
        e0 = m.energetics(q)["E"]
        q = m.run(q, m.steps_for(60 * DAY))
        e1 = m.energetics(q)["E"]
        assert (e1 > 20 * e0) == grows


def test_parameter_validation():
    with pytest.raises(KeyError):
        Params.from_dict({"nonsense": 1})
    p = Params(delta=0.3, Ld=5e5)
    assert p.F1 + p.F2 == pytest.approx(1 / p.Ld**2)
    assert p.delta * p.F1 == pytest.approx((1 - p.delta) * p.F2)
