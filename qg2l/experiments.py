"""The four experiments behind the web explorer.

control      long, equilibrated run: the dataset the "Fields", "Waves", "Statistics",
             "Mean flow" and "Energetics" tabs analyse in the browser.
sweep        climate response: time-mean statistics for a range of shear and drag values.
ensemble     twin predictability experiments (error growth, upscale error cascade).
assimilation synthetic station network + EnSRF (observing-system experiment).

Each stage writes its arrays through `Writer` and returns a manifest fragment.
"""
from __future__ import annotations

import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import obs as obsmod
from .analysis import MeanAccumulator, ke_spectrum, shell_matrix, spectrum_of_difference
from .config import Config, apply_override
from .export import Writer
from .model import DAY, QG2L, Params

Log = Callable[[str], None]


def z_to_psi(p: Params, z):
    """Geopotential-height anomaly (m) -> streamfunction (m^2/s)."""
    return np.asarray(z) * p.g / p.f0


def psi_to_z(p: Params, psi):
    return np.asarray(psi) * p.f0 / p.g


def _cfl(m: QG2L, q: np.ndarray) -> float:
    """Advective CFL number of the full flow (mean wind + perturbation) for RK4 (limit ~2.8)."""
    p = m.p
    ph = m.invert(q)
    u = m.to_phys(-m.iky * ph) + np.array([p.U1, p.U2])[:, None, None]
    v = m.to_phys(m.ikx * ph)
    kmax = (2.0 / 3.0) * np.pi
    return float(np.max(np.abs(u) * kmax / p.dx + np.abs(v) * kmax / p.dy) * p.dt)


def _check_cfl(m: QG2L, q: np.ndarray, where: str) -> None:
    c = _cfl(m, q)
    if not np.isfinite(c) or c > 2.6:
        raise RuntimeError(f"CFL number {c:.2f} in {where}: the run is (about to become) numerically "
                           f"unstable. Reduce dt (model.dt) or the shear.")


# ======================================================================================
# control run
# ======================================================================================
def run_control(cfg: Config, w: Writer, work: Path, log: Log, workers: int = 1) -> dict[str, Any]:
    p, c = cfg.params, cfg.control
    m = QG2L(p, workers=workers)
    rng = np.random.default_rng(int(c["seed"]))
    q = m.random_state(rng, amp=1e4)
    spd = m.steps_for(DAY)
    nspin = int(round(c["spinup_days"]))
    spin_E = []
    t0 = time.time()
    for d in range(nspin):
        q = m.run(q, spd)
        spin_E.append(float(m.energetics(q)["E"]))
        if d % 50 == 49:
            _check_cfl(m, q, f"control spin-up, day {d + 1}")
            log(f"  control spin-up day {d + 1}/{nspin}  E = {spin_E[-1]:.1f} m2/s2")
    out_steps = m.steps_for(c["output_hours"] * 3600.0)
    nt = int(round(c["length_days"] * 24.0 / c["output_hours"]))
    psi = np.empty((nt, 2, p.ny, p.nx), dtype=np.float32)
    keys = ("KE1", "KE2", "APE", "E", "heat_flux", "C_B", "D_E", "D_h")
    ser = {k: np.empty(nt) for k in keys}
    np.save(work / "control_q0.npy", q)
    for k in range(nt):
        psi[k] = m.psi(q)
        e = m.energetics(q)
        for key in keys:
            ser[key][k] = float(e[key])
        q = m.run(q, out_steps)
        if k % 100 == 99:
            _check_cfl(m, q, f"control run, output {k + 1}")
            log(f"  control run {k + 1}/{nt} snapshots  ({time.time() - t0:.0f} s)")
    np.save(work / "control_psi.npy", psi)
    desc = w.array("control_psi", psi, "int16", scale_axis=1)
    return {"control": {
        "psi": desc, "start_days": 0.0, "step_hours": float(c["output_hours"]), "n": nt,
        "spinup_days": nspin, "seed": int(c["seed"]),
        "spinup_energy": spin_E,
        "model_energetics": {k: ser[k].tolist() for k in keys},
    }}


def load_control(work: Path) -> tuple[np.ndarray, np.ndarray]:
    return np.load(work / "control_psi.npy"), np.load(work / "control_q0.npy")


# ======================================================================================
# parameter sweep
# ======================================================================================
def _sweep_case(args: tuple) -> dict[str, Any]:
    pdict, q0, spin_days, len_days, out_hours, meta = args
    p = Params.from_dict(pdict)
    m = QG2L(p, workers=1)
    q = q0.copy()
    acc = MeanAccumulator(m)
    keys = ("E", "KE1", "KE2", "APE", "C_B", "D_E", "D_h")
    series = {k: [] for k in keys}
    t_days = []
    nspin = int(round(spin_days))
    n_halved = 0

    def adapt(m: QG2L, q: np.ndarray) -> QG2L:
        """Strongly supercritical cases develop fast jets: halve dt when the CFL number is high."""
        nonlocal n_halved, acc
        if _cfl(m, q) > 1.8 and n_halved < 3:
            n_halved += 1
            m = QG2L(m.p.replace(dt=m.p.dt / 2.0), workers=1)
            acc = MeanAccumulator(m)
        return m

    for d in range(nspin):
        m = adapt(m, q)
        q = m.run(q, m.steps_for(DAY))
        e = m.energetics(q)
        for k in keys:
            series[k].append(float(e[k]))
        t_days.append(d + 1.0)
        _check_cfl(m, q, f"sweep case {meta['label']}, day {d + 1}")
    nout = int(round(len_days * 24.0 / out_hours))
    for k in range(nout):
        m = adapt(m, q)
        q = m.run(q, m.steps_for(out_hours * 3600.0))
        acc.add(q)
        e = m.energetics(q)
        for key in keys:
            series[key].append(float(e[key]))
        t_days.append(nspin + (k + 1) * out_hours / 24.0)
    res = acc.result()
    mean_e = {k: float(np.mean(series[k][-nout:])) for k in keys}
    return {"meta": meta, "params": {"U1": p.U1, "U2": p.U2, "dU": p.dU, "dU_ratio": p.dU / p.dU_crit,
                                      "r_ek_days": 1.0 / p.r_ek / DAY, "dt": m.p.dt},
            "t_days": t_days, "series": series, "t_avg_start_index": nspin,
            "mean": {**mean_e, **res}, "psi_final": m.psi(q).astype(np.float32)}


def run_sweep(cfg: Config, w: Writer, work: Path, log: Log, workers: int) -> dict[str, Any]:
    s = cfg.sweep
    p = cfg.params
    _, q0 = load_control(work)
    jobs = []
    for ai, axis in enumerate(s["axis"]):
        name = axis["parameter"]
        for v in axis["values"]:
            pc = apply_override(p, name, v)
            meta = {"axis": ai, "parameter": name, "value": float(v), "label": f"{name} = {v:g}"}
            jobs.append((pc.to_dict(), q0, s["spinup_days"], s["length_days"], s["output_hours"], meta))
    log(f"  sweep: {len(jobs)} cases on {workers} worker(s)")
    t0 = time.time()
    if workers > 1:
        with Pool(workers) as pool:
            results = []
            for i, r in enumerate(pool.imap(_sweep_case, jobs)):
                results.append(r)
                log(f"  sweep case {i + 1}/{len(jobs)} done ({time.time() - t0:.0f} s)")
    else:
        results = []
        for i, j in enumerate(jobs):
            results.append(_sweep_case(j))
            log(f"  sweep case {i + 1}/{len(jobs)} done ({time.time() - t0:.0f} s)")
    finals = np.stack([r.pop("psi_final") for r in results])
    desc = w.array("sweep_final_psi", finals, "int16", scale_axis=1)
    cases = []
    for r in results:
        cases.append({"meta": r["meta"], "params": r["params"], "t_days": r["t_days"],
                      "series": r["series"], "t_avg_start_index": r["t_avg_start_index"],
                      "mean": r["mean"]})
    axes = [{"parameter": a["parameter"], "values": [float(v) for v in a["values"]]} for a in s["axis"]]
    w.json("sweep.json", {"axes": axes, "cases": cases}, digits=5)
    return {"sweep": {"file": "sweep.json", "final_psi": desc,
                      "spinup_days": s["spinup_days"], "average_days": s["length_days"],
                      "output_hours": s["output_hours"],
                      "dU_crit": p.dU_crit}}


# ======================================================================================
# ensemble predictability
# ======================================================================================
def _ensemble_group(args: tuple) -> dict[str, Any]:
    (pdict, truth, group, nmem, seed, stride, nout, out_hours, stamps, save_fields) = args
    p = Params.from_dict(pdict)
    m = QG2L(p, workers=1)
    S = shell_matrix(m)
    rng = np.random.default_rng(seed)
    amp = float(z_to_psi(p, group["amp_z"]))
    q_truth = m.q_from_psi(truth[0].astype(np.float64))
    # Exchangeable (calibrated) ensemble: the truth is one more draw from the same error
    # distribution as the members.  The ensemble is centred on the "analysis" truth - eps_t and
    # member i starts at analysis + eps_i, i.e. truth + eps_i - eps_t.
    eps = m.perturbation(rng, amp, nmem + 1, wavelength=group["wavelength_km"] * 1e3)
    q = q_truth + eps[1:] - eps[0]
    out_steps = m.steps_for(out_hours * 3600.0)
    per_day = int(round(24.0 / out_hours))
    nb = p.nx // 2
    dte = np.empty((nmem, nout + 1))
    rms1 = np.empty((nmem, nout + 1))
    rms2 = np.empty((nmem, nout + 1))
    spread1 = np.empty(nout + 1)
    spread2 = np.empty(nout + 1)
    rmse_mean1 = np.empty(nout + 1)
    rmse_mean2 = np.empty(nout + 1)
    err_spec = np.empty((nout + 1, nb))
    nd = nout // per_day + 1
    f_truth = np.empty((nd, 2, p.ny, p.nx), np.float32)
    f_mean = np.empty_like(f_truth)
    f_spread = np.empty_like(f_truth)
    f_stamps = np.empty((stamps, nd, p.ny, p.nx), np.float32)
    for t in range(nout + 1):
        pt = truth[t * stride].astype(np.float64)
        qt = m.q_from_psi(pt)
        dq = q - qt
        dte[:, t] = m.energetics(dq)["E"]
        ps = m.psi(q)                                           # (N, 2, ny, nx)
        for L, (rm, sp) in enumerate(((rms1, spread1), (rms2, spread2))):
            err = ps[:, L] - pt[L]
            rm[:, t] = psi_to_z(p, np.sqrt((err**2).mean(axis=(-2, -1))))
        mean = ps.mean(axis=0)
        std = ps.std(axis=0, ddof=1)
        rmse_mean1[t] = psi_to_z(p, np.sqrt(((mean[0] - pt[0]) ** 2).mean()))
        rmse_mean2[t] = psi_to_z(p, np.sqrt(((mean[1] - pt[1]) ** 2).mean()))
        spread1[t] = psi_to_z(p, np.sqrt((std[0] ** 2).mean()))
        spread2[t] = psi_to_z(p, np.sqrt((std[1] ** 2).mean()))
        err_spec[t] = spectrum_of_difference(m, dq, S).mean(axis=0)
        if save_fields and t % per_day == 0:
            d = t // per_day
            f_truth[d], f_mean[d], f_spread[d] = pt, mean, std
            f_stamps[:, d] = ps[:stamps, 0]
        if t < nout:
            q = m.run(q, out_steps)
    res: dict[str, Any] = {
        "name": group["name"], "amp_z": group["amp_z"], "wavelength_km": group["wavelength_km"],
        "dte": dte, "rms_z1": rms1, "rms_z2": rms2, "spread_z1": spread1, "spread_z2": spread2,
        "rmse_mean_z1": rmse_mean1, "rmse_mean_z2": rmse_mean2, "err_spectrum": err_spec,
    }
    if save_fields:
        res.update(f_truth=f_truth, f_mean=f_mean, f_spread=f_spread, f_stamps=f_stamps)
    return res


def run_ensemble(cfg: Config, w: Writer, work: Path, log: Log, workers: int) -> dict[str, Any]:
    e = cfg.ensemble
    p = cfg.params
    psi, _ = load_control(work)
    ctrl_h = cfg.control["output_hours"]
    i0 = int(round(e["start_day"] * 24.0 / ctrl_h))
    stride = int(round(e["output_hours"] / ctrl_h))
    nout = int(round(e["lead_days"] * 24.0 / e["output_hours"]))
    truth = psi[i0: i0 + nout * stride + 1]
    groups = e["group"]
    if not groups:
        groups = [{"name": "5 m, 2000 km", "amp_z": 5.0, "wavelength_km": 2000.0}]
    jobs = [(p.to_dict(), truth, g, int(e["members"]), int(e["seed"]) + gi, stride, nout,
             e["output_hours"], int(e["stamps"]), gi == 0) for gi, g in enumerate(groups)]
    log(f"  ensemble: {len(jobs)} group(s) x {e['members']} members x {e['lead_days']:g} days")
    t0 = time.time()
    if workers > 1 and len(jobs) > 1:
        with Pool(min(workers, len(jobs))) as pool:
            results = pool.map(_ensemble_group, jobs)
    else:
        results = [_ensemble_group(j) for j in jobs]
    log(f"  ensemble done ({time.time() - t0:.0f} s)")
    first = results[0]
    fields = {
        "truth": w.array("ens_truth_psi", first.pop("f_truth"), "int16", scale_axis=1),
        "mean": w.array("ens_mean_psi", first.pop("f_mean"), "int16", scale_axis=1),
        "spread": w.array("ens_spread_psi", first.pop("f_spread"), "int16", scale_axis=1),
        "stamps": w.array("ens_stamps_psi1", first.pop("f_stamps"), "int16"),
    }
    out_groups = []
    for r in results:
        out_groups.append({k: (v if isinstance(v, (str, float, int)) else np.asarray(v))
                           for k, v in r.items()})
    w.json("ensemble.json", {"groups": out_groups, "lead_hours": e["output_hours"]}, digits=5)
    clim = float(psi_to_z(p, psi[:, 0].std()))
    return {"ensemble": {"file": "ensemble.json", "fields": fields,
                         "start_day": e["start_day"], "members": int(e["members"]),
                         "lead_days": e["lead_days"], "output_hours": e["output_hours"],
                         "field_step_hours": 24.0, "stamps": int(e["stamps"]),
                         "climatological_std_z1": clim}}


# ======================================================================================
# observing system experiment
# ======================================================================================
def _rmse_spread(p: Params, mean: np.ndarray, std: np.ndarray, truth: np.ndarray):
    """Domain RMSE of the ensemble mean and RMS ensemble spread, per layer, in metres of Z'."""
    rm = [float(psi_to_z(p, np.sqrt(((mean[i] - truth[i]) ** 2).mean()))) for i in range(2)]
    sp = [float(psi_to_z(p, np.sqrt((std[i] ** 2).mean()))) for i in range(2)]
    return rm, sp


def _assim_task(args: tuple) -> dict[str, Any]:
    (pdict, kind, nst, a, truth, init_psi, positions, snap_cycles) = args
    p = Params.from_dict(pdict)
    m = QG2L(p, workers=1)
    N = int(a["members"])
    nc = truth.shape[0] - 1
    cyc_steps = m.steps_for(a["cycle_hours"] * 3600.0)
    sigma_psi = float(z_to_psi(p, a["obs_error_z"]))
    net = None
    if kind == "da":
        net = obsmod.Network(p, positions[:nst], a["variables"], sigma_psi,
                             float(a["obs_error_wind"]), a["localization_km"] * 1e3)
    q = m.q_from_psi(init_psi.astype(np.float64))
    nvar = len(a["variables"])
    out = {k: np.zeros((2, nc + 1)) for k in ("bg_rmse", "bg_spread", "an_rmse", "an_spread")}
    primary = kind == "da" and nst == a["primary"]
    snaps = {k: [] for k in ("truth", "background", "analysis", "spread")}
    st = {k: np.zeros((nvar, nc, nst)) for k in ("truth", "obs", "background", "analysis", "bg_spread")} \
        if primary else None
    nmax = len(positions)
    t0 = time.time()
    for k in range(nc + 1):
        if k > 0:
            q = m.run(q, cyc_steps)
        ps = m.psi(q)
        if not np.isfinite(ps).all():
            raise RuntimeError(f"assimilation run '{kind}' ({nst} stations) became non-finite at cycle {k}")
        mean, std = ps.mean(0), ps.std(0, ddof=1)
        tr = truth[k].astype(np.float64)
        rm, sp = _rmse_spread(p, mean, std, tr)
        out["bg_rmse"][:, k], out["bg_spread"][:, k] = rm, sp
        an_mean, an_std = mean, std
        if kind == "da" and k > 0:
            X = ps.reshape(N, -1)
            xt = tr.reshape(-1)
            rng = np.random.default_rng(int(a["seed"]) * 100003 + k)
            noise = rng.standard_normal((nvar, nmax))[:, :nst] * net.sigma[:, None]
            y = net.observe(xt) + noise
            Xa = obsmod.ensrf_update(X, net, y, float(a["inflation"]))
            psa = Xa.reshape(N, 2, p.ny, p.nx)
            an_mean, an_std = psa.mean(0), psa.std(0, ddof=1)
            q = m.q_from_psi(psa)
            if primary:
                hb = net.observe(X)
                ha = net.observe(Xa)
                st["truth"][:, k - 1] = net.observe(xt)
                st["obs"][:, k - 1] = y
                st["background"][:, k - 1] = hb.mean(0)
                st["analysis"][:, k - 1] = ha.mean(0)
                st["bg_spread"][:, k - 1] = hb.std(0, ddof=1)
        rm, sp = _rmse_spread(p, an_mean, an_std, tr)
        out["an_rmse"][:, k], out["an_spread"][:, k] = rm, sp
        if primary and k in snap_cycles:
            snaps["truth"].append(tr)
            snaps["background"].append(mean)
            snaps["analysis"].append(an_mean)
            snaps["spread"].append(an_std)
    res: dict[str, Any] = {"kind": kind, "stations": nst, "series": out,
                           "seconds": time.time() - t0}
    if primary:
        res["snapshots"] = {k: np.stack(v).astype(np.float32) for k, v in snaps.items()}
        res["station_series"] = st
    return res


def run_assimilation(cfg: Config, w: Writer, work: Path, log: Log, workers: int) -> dict[str, Any]:
    a = cfg.assim
    p = cfg.params
    psi, _ = load_control(work)
    ctrl_h = cfg.control["output_hours"]
    stride = int(round(a["cycle_hours"] / ctrl_h))
    i0 = int(round(a["start_day"] * 24.0 / ctrl_h))
    nc = int(round(a["days"] * 24.0 / a["cycle_hours"]))
    truth = psi[i0: i0 + nc * stride + 1: stride]
    rng = np.random.default_rng(int(a["seed"]))
    # initial ensemble: random states of the control run ("climatology"), excluding the truth window
    pool_idx = np.setdiff1d(np.arange(psi.shape[0]), np.arange(i0 - 4, i0 + nc * stride + 5))
    pick = rng.choice(pool_idx, size=int(a["members"]), replace=False)
    init = psi[np.sort(pick)]
    nmax = max(a["stations"])
    positions = obsmod.random_stations(p, nmax, int(a["seed"]))
    day_snaps = [d for d in (0, 2, 5, 10, 20, 40, a["days"]) if d <= a["days"]]
    snap_cycles = sorted({int(round(d * 24.0 / a["cycle_hours"])) for d in day_snaps})
    jobs = [(p.to_dict(), "free", 0, a, truth, init, positions, snap_cycles)]
    for n in a["stations"]:
        jobs.append((p.to_dict(), "da", int(n), a, truth, init, positions, snap_cycles))
    log(f"  assimilation: {len(jobs)} runs ({a['members']} members, {nc} cycles) on {workers} worker(s)")
    t0 = time.time()
    if workers > 1:
        with Pool(min(workers, len(jobs))) as pool:
            results = []
            for i, r in enumerate(pool.imap(_assim_task, jobs)):
                results.append(r)
                log(f"  assimilation run {i + 1}/{len(jobs)} done ({time.time() - t0:.0f} s)")
    else:
        results = []
        for i, j in enumerate(jobs):
            results.append(_assim_task(j))
            log(f"  assimilation run {i + 1}/{len(jobs)} done ({time.time() - t0:.0f} s)")
    free = next(r for r in results if r["kind"] == "free")
    nets = [r for r in results if r["kind"] == "da"]
    prim = next(r for r in nets if r["stations"] == a["primary"])
    spin = int(a["spinup_cycles"])
    clim_z = float(psi_to_z(p, psi[:, 0].std()))
    runs = []
    for r in nets:
        s = r["series"]
        runs.append({"stations": r["stations"], "series": s,
                     "score_an": [float(np.mean(s["an_rmse"][L, spin:])) for L in range(2)],
                     "score_bg": [float(np.mean(s["bg_rmse"][L, spin:])) for L in range(2)],
                     "score_spread_bg": [float(np.mean(s["bg_spread"][L, spin:])) for L in range(2)]})
    fs = free["series"]
    w.json("assimilation.json", {
        "free": {"series": fs, "score": [float(np.mean(fs["bg_rmse"][L, spin:])) for L in range(2)]},
        "runs": runs,
        "positions_km": (positions / 1e3).tolist(),
        "variables": a["variables"],
        "obs_units": ["m" if v.startswith("psi") else "m/s" for v in a["variables"]],
        "obs_error": [a["obs_error_z"] if v.startswith("psi") else a["obs_error_wind"]
                      for v in a["variables"]],
        "series_names": ["truth", "obs", "background", "analysis", "bg_spread"],
    }, digits=5)
    snap_arr = np.stack([prim["snapshots"][k] for k in ("truth", "background", "analysis", "spread")], axis=1)
    sf = w.array("assim_snapshots_psi", snap_arr, "int16")
    ss = np.stack([prim["station_series"][k]
                   for k in ("truth", "obs", "background", "analysis", "bg_spread")])
    # stations series: convert psi-type variables to metres of Z'
    for iv, v in enumerate(a["variables"]):
        if v.startswith("psi"):
            ss[:, iv] = psi_to_z(p, ss[:, iv])
    sdesc = w.array("assim_station_series", ss, "float32")
    return {"assimilation": {
        "file": "assimilation.json", "snapshots": sf, "station_series": sdesc,
        "snapshot_names": ["truth", "background", "analysis", "spread"],
        "snapshot_cycles": snap_cycles, "cycle_hours": a["cycle_hours"], "cycles": nc,
        "members": int(a["members"]), "start_day": a["start_day"], "primary": int(a["primary"]),
        "spinup_cycles": spin, "inflation": a["inflation"], "localization_km": a["localization_km"],
        "climatological_std_z1": clim_z,
    }}
