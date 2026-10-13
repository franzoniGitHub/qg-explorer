"""Observation operators, the EnSRF, the binary export format and config validation."""
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from qg2l import Params
from qg2l.config import Config
from qg2l.export import Writer, read_array, round_floats
from qg2l.obs import Network, ObsOperator, ensrf_update, gaspari_cohn, random_stations

ROOT = Path(__file__).resolve().parents[1]


def small_params():
    return Params(nx=32, ny=32)


# ------------------------------------------------------------------------------ localisation
def test_gaspari_cohn_shape():
    c = 1.0
    assert gaspari_cohn(np.array([0.0]), c)[0] == pytest.approx(1.0)
    assert gaspari_cohn(np.array([2.0, 3.0]), c).tolist() == [0.0, 0.0]
    z = np.linspace(0, 2, 401)
    g = gaspari_cohn(z, c)
    assert np.all(np.diff(g) <= 1e-12), "monotone decreasing"
    below, above = gaspari_cohn(np.array([1 - 1e-9]), c)[0], gaspari_cohn(np.array([1 + 1e-9]), c)[0]
    assert below == pytest.approx(above, abs=1e-6)
    assert below == pytest.approx(5.0 / 24.0, abs=1e-6)


def test_stations_are_nested():
    p = small_params()
    a, b = random_stations(p, 10, 7), random_stations(p, 25, 7)
    np.testing.assert_array_equal(a, b[:10])
    assert a.min() >= 0 and a[:, 0].max() < p.Lx and a[:, 1].max() < p.Ly


# --------------------------------------------------------------------- observation operator
def test_operator_exact_at_cell_centres_and_constants():
    p = small_params()
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2 * p.ny * p.nx)
    pos = np.array([[(5 + 0.5) * p.dx, (9 + 0.5) * p.dy]])
    h1, h2 = ObsOperator(p, pos, "psi1"), ObsOperator(p, pos, "psi2")
    assert h1(x)[0] == pytest.approx(x[9 * p.nx + 5])
    assert h2(x)[0] == pytest.approx(x[p.ny * p.nx + 9 * p.nx + 5])
    ones = np.ones_like(x)
    pos = random_stations(p, 20, 1)
    np.testing.assert_allclose(ObsOperator(p, pos, "psi1")(ones), 1.0)


def test_wind_operators_on_a_smooth_field():
    p = small_params()
    x = (np.arange(p.nx) + 0.5) * p.dx
    y = (np.arange(p.ny) + 0.5) * p.dy
    X, Y = np.meshgrid(x, y)
    kx, ky = 2 * np.pi / p.Lx, 2 * np.pi / p.Ly
    psi = np.sin(kx * X) * np.cos(ky * Y)
    state = np.concatenate([psi.ravel(), np.zeros(psi.size)])
    pos = random_stations(p, 30, 3)
    u_true = psi_dy = -(-ky * np.sin(kx * pos[:, 0]) * np.sin(ky * pos[:, 1]))
    v_true = kx * np.cos(kx * pos[:, 0]) * np.cos(ky * pos[:, 1])
    np.testing.assert_allclose(ObsOperator(p, pos, "u1")(state), u_true, atol=0.03 * np.abs(u_true).max())
    np.testing.assert_allclose(ObsOperator(p, pos, "v1")(state), v_true, atol=0.03 * np.abs(v_true).max())
    with pytest.raises(ValueError):
        ObsOperator(p, pos, "temperature")


# --------------------------------------------------------------------------------------- EnSRF
def _net(p, pos, sigma):
    return Network(p, pos, ["psi1"], sigma, 1.0, loc_radius=1e9)


def test_ensrf_matches_scalar_kalman_filter_at_the_station():
    p = small_params()
    rng = np.random.default_rng(5)
    N = 200
    X = rng.standard_normal((N, 2 * p.ny * p.nx)) * 3.0 + 1.0
    pos = np.array([[(7 + 0.5) * p.dx, (3 + 0.5) * p.dy]])
    sigma = 2.0
    net = _net(p, pos, sigma)
    idx = 3 * p.nx + 7
    prior_mean, prior_var = X[:, idx].mean(), X[:, idx].var(ddof=1)
    y = np.array([[prior_mean + 4.0]])
    Xa = ensrf_update(X, net, y)
    K = prior_var / (prior_var + sigma**2)
    assert Xa[:, idx].mean() == pytest.approx(prior_mean + K * 4.0, rel=1e-9)
    assert Xa[:, idx].var(ddof=1) == pytest.approx(prior_var * sigma**2 / (prior_var + sigma**2), rel=1e-9)
    # a point that is uncorrelated by construction moves much less than the observed one
    far = 20 * p.nx + 20
    assert abs(Xa[:, far].mean() - X[:, far].mean()) < 0.5 * abs(Xa[:, idx].mean() - prior_mean)


def test_ensrf_localisation_and_input_untouched():
    p = small_params()
    rng = np.random.default_rng(6)
    X = rng.standard_normal((30, 2 * p.ny * p.nx))
    X0 = X.copy()
    pos = np.array([[4.5 * p.dx, 4.5 * p.dy]])
    net = Network(p, pos, ["psi1"], 1.0, 1.0, loc_radius=4 * p.dx)
    Xa = ensrf_update(X, net, np.array([[5.0]]))
    np.testing.assert_array_equal(X, X0)
    d = np.abs(Xa.mean(0) - X.mean(0)).reshape(2, p.ny, p.nx)
    assert d[0, 4, 4] > 0
    assert d[0, 20, 20] < 1e-12, "beyond the localisation radius nothing changes"
    assert np.all(d[1, 20:, 20:] < 1e-12)


def test_inflation_is_local():
    p = small_params()
    rng = np.random.default_rng(9)
    X = rng.standard_normal((30, 2 * p.ny * p.nx))
    pos = np.array([[4.5 * p.dx, 4.5 * p.dy]])
    net = Network(p, pos, ["psi1"], 1.0, 1.0, loc_radius=4 * p.dx)
    # an observation equal to the prior mean changes the mean nowhere and only inflates
    y = np.array([[float(net.observe(X.mean(0))[0, 0])]])
    Xa = ensrf_update(X, net, y, inflation=1.5)
    sd0, sd1 = X.std(0, ddof=1).reshape(2, p.ny, p.nx), Xa.std(0, ddof=1).reshape(2, p.ny, p.nx)
    np.testing.assert_allclose(sd1[:, 20:, 20:], sd0[:, 20:, 20:], rtol=1e-12)    # nothing observed there
    assert sd1[1, 4, 4] > 1.3 * sd0[1, 4, 4]                                       # inflated near the station


def test_more_observations_reduce_error():
    p = small_params()
    rng = np.random.default_rng(8)
    truth = rng.standard_normal(2 * p.ny * p.nx)
    # smooth the truth so that interpolation is meaningful
    from numpy.fft import irfft2, rfft2
    def smooth(a):
        f = rfft2(a.reshape(p.ny, p.nx))
        ky = np.fft.fftfreq(p.ny)[:, None] * p.ny
        kx = np.fft.rfftfreq(p.nx)[None, :] * p.nx
        f *= np.exp(-(kx**2 + ky**2) / 8.0)
        return irfft2(f, s=(p.ny, p.nx)).ravel()
    truth = np.concatenate([smooth(truth[: p.ny * p.nx]), smooth(truth[p.ny * p.nx:])])
    truth /= truth.std()
    X = np.stack([np.concatenate([smooth(rng.standard_normal(p.ny * p.nx)), smooth(rng.standard_normal(p.ny * p.nx))]) for _ in range(40)])
    X /= truth.std()
    pos_all = random_stations(p, 60, 2)
    errs = []
    for n in (0, 10, 60):
        if n == 0:
            Xa = X
        else:
            net = Network(p, pos_all[:n], ["psi1", "psi2"], 0.1, 0.1, loc_radius=12 * p.dx)
            y = net.observe(truth)
            Xa = ensrf_update(X, net, y)
        errs.append(np.sqrt(((Xa.mean(0) - truth) ** 2).mean()))
    assert errs[0] > errs[1] > errs[2]


# ---------------------------------------------------------------------------------- export
def test_int16_roundtrip_per_axis_scale(tmp_path):
    w = Writer(tmp_path)
    rng = np.random.default_rng(1)
    a = np.stack([rng.standard_normal((4, 6, 5)) * 1e6, rng.standard_normal((4, 6, 5)) * 3e3], axis=1)   # (4, 2, 6, 5)
    d = w.array("a", a, "int16", scale_axis=1)
    assert d["dtype"] == "int16" and d["shape"] == [4, 2, 6, 5] and len(d["scale"]) == 2
    b = read_array(tmp_path, d)
    for L in range(2):
        assert np.abs(b[:, L] - a[:, L]).max() <= np.abs(a[:, L]).max() / 30000.0
    assert (tmp_path / d["file"]).stat().st_size == a.size * 2


def test_int16_scalar_scale_and_float32(tmp_path):
    w = Writer(tmp_path)
    a = np.linspace(-5, 5, 101).reshape(101, 1)
    d = w.array("s", a, "int16")
    assert np.abs(read_array(tmp_path, d) - a).max() < 5 / 30000.0 + 1e-12
    d2 = w.array("f", a, "float32")
    np.testing.assert_array_equal(read_array(tmp_path, d2), a.astype(np.float32).astype(np.float64))
    with pytest.raises(ValueError):
        w.array("bad", a, "float64")


def test_json_rounding_and_nan(tmp_path):
    obj = {"a": np.float64(1.23456789012), "b": [np.float32(0.1), 2], "c": {"d": float("nan")}}
    r = round_floats(obj, 4)
    assert r["a"] == 1.235   # significant digits
    assert isinstance(r["b"][1], int)
    w = Writer(tmp_path)
    name = w.json("x.json", {"v": [1.0000000001, 2.5]})
    assert json.loads((tmp_path / name).read_text()) == {"v": [1.0, 2.5]}


# ---------------------------------------------------------------------------------- config
def test_shipped_configs_load():
    for name in ("default.toml", "quick.toml"):
        cfg = Config.load(ROOT / "config" / name)
        assert cfg.params.dU / cfg.params.dU_crit > 1.0, "the shipped basic state is supercritical"


def _raw(**over):
    raw = tomllib.loads((ROOT / "config" / "quick.toml").read_text())
    for k, v in over.items():
        sect, key = k.split("__")
        raw[sect][key] = v
    return raw


def test_config_validation():
    Config(_raw())
    with pytest.raises(ValueError, match="powers of two"):
        Config(_raw(model__nx=48))
    with pytest.raises(ValueError, match="multiple of the model time step"):
        Config(_raw(control__output_hours=0.4))
    with pytest.raises(ValueError, match="must fit inside"):
        Config(_raw(assimilation__start_day=80))
    with pytest.raises(ValueError, match="primary"):
        Config(_raw(assimilation__primary=77))
    raw = _raw()
    raw["control"]["spin_up"] = 3
    with pytest.raises(KeyError, match="spin_up"):
        Config(raw)
