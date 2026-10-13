"""End-to-end: a tiny configuration runs through every stage and writes a loadable dataset."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from qg2l.config import Config
from qg2l.export import read_array
from qg2l.pipeline import run_pipeline, update_index

TINY = {
    "title": "tiny", "description": "smoke test",
    "model": {"nx": 16, "ny": 16, "Lx": 1.6e7, "Ly": 1.6e7, "Ld": 6.0e5, "delta": 0.5, "U1": 20.5, "U2": 5.0,
              "r_ek_days": 3.0, "dt": 1800.0, "filter_tau_hours": 6.0},
    "control": {"seed": 1, "spinup_days": 10, "length_days": 12, "output_hours": 12},
    "sweep": {"enabled": True, "spinup_days": 3, "length_days": 3, "output_hours": 24,
              "axis": [{"parameter": "shear_ratio", "values": [0.8, 1.4]}]},
    "ensemble": {"enabled": True, "start_day": 2, "members": 4, "lead_days": 4, "output_hours": 12, "stamps": 2,
                 "group": [{"name": "g", "amp_z": 5.0, "wavelength_km": 4000}]},
    "assimilation": {"enabled": True, "start_day": 3, "days": 4, "cycle_hours": 12, "members": 6,
                     "spinup_cycles": 2, "stations": [4, 8], "primary": 8, "localization_km": 4000},
}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("data") / "tiny"
    manifest = run_pipeline(Config(TINY), out, workers=1, log=lambda s: None)
    return out, manifest


def test_manifest_and_files(dataset):
    out, mf = dataset
    assert mf["format"] == "qg2l-dataset" and mf["version"] == 1
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk["title"] == "tiny"
    for key in ("model", "grid", "control", "sweep", "ensemble", "assimilation", "reference"):
        assert key in on_disk, key
    c = on_disk["control"]
    assert c["n"] == 24 and c["psi"]["shape"] == [24, 2, 16, 16]
    assert (out / c["psi"]["file"]).stat().st_size == 24 * 2 * 16 * 16 * 2


def test_control_data_are_sane(dataset):
    out, mf = dataset
    psi = read_array(out, mf["control"]["psi"])
    assert np.isfinite(psi).all() and psi.std() > 0
    E = np.array(mf["control"]["model_energetics"]["E"])
    assert E.shape == (24,) and np.all(E > 0)


def test_experiment_products(dataset):
    out, mf = dataset
    sw = json.loads((out / mf["sweep"]["file"]).read_text())
    assert [c["meta"]["value"] for c in sw["cases"]] == [0.8, 1.4]
    assert sw["cases"][1]["mean"]["E"] > sw["cases"][0]["mean"]["E"], "stronger shear gives more energy"
    ens = json.loads((out / mf["ensemble"]["file"]).read_text())
    g = ens["groups"][0]
    assert len(g["dte"]) == 4 and len(g["spread_z1"]) == 9
    assert g["rms_z1"][0][-1] > g["rms_z1"][0][0], "errors grow"
    asm = json.loads((out / mf["assimilation"]["file"]).read_text())
    assert [r["stations"] for r in asm["runs"]] == [4, 8]
    assert asm["runs"][1]["score_an"][0] < asm["free"]["score"][0], "observations help"


def test_cache_and_index(dataset, tmp_path):
    out, _ = dataset
    logs = []
    run_pipeline(Config(TINY), out, workers=1, log=logs.append)
    assert sum("up to date" in s for s in logs) == 4
    update_index(out.parent, log=lambda s: None, force=True)
    idx = json.loads((out.parent / "index.json").read_text())
    assert idx["datasets"][0]["id"] == "tiny"


def test_cli_info_and_bad_config(dataset, tmp_path):
    out, _ = dataset
    r = subprocess.run([sys.executable, "-m", "qg2l", "info", "config/default.toml"], capture_output=True, text=True,
                       cwd=Path(__file__).resolve().parents[1])
    assert r.returncode == 0 and "supercriticality" in r.stdout
    bad = tmp_path / "bad.toml"
    bad.write_text('[model]\nnx = 48\n')
    r = subprocess.run([sys.executable, "-m", "qg2l", "run", str(bad), "--out", str(tmp_path / "o")],
                       capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert r.returncode != 0 and "powers of two" in (r.stderr + r.stdout)


@pytest.mark.parametrize("variables", [["psi1"], ["u1", "v1"]])
def test_assimilation_with_other_observed_variables(tmp_path, variables):
    import copy
    cfg = copy.deepcopy(TINY)
    cfg["sweep"]["enabled"] = False
    cfg["ensemble"]["enabled"] = False
    cfg["assimilation"].update(variables=variables, stations=[8], primary=8, obs_error_wind=1.0)
    out = tmp_path / "d"
    mf = run_pipeline(Config(cfg), out, workers=1, log=lambda s: None)
    asm = json.loads((out / mf["assimilation"]["file"]).read_text())
    assert asm["variables"] == variables
    assert mf["assimilation"]["station_series"]["shape"][1] == len(variables)
    assert np.isfinite(asm["runs"][0]["score_an"]).all()
