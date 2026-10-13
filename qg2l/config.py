"""Configuration files (TOML) -> model parameters and experiment settings."""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .model import DAY, Params

# convenience keys accepted in [model]; they are converted to Params fields
_CONVENIENCE = ("r_ek_days", "filter_tau_hours", "shear_ratio")

DEFAULTS: dict[str, Any] = {
    "title": "Two-layer QG model",
    "description": "",
    "control": {"seed": 1, "spinup_days": 300.0, "length_days": 360.0, "output_hours": 12.0},
    "sweep": {"enabled": False, "spinup_days": 150.0, "length_days": 150.0, "output_hours": 24.0,
              "axis": []},
    "ensemble": {"enabled": False, "start_day": 60.0, "members": 20, "lead_days": 30.0,
                 "output_hours": 12.0, "seed": 3, "stamps": 6, "group": []},
    "assimilation": {"enabled": False, "start_day": 120.0, "days": 60.0, "cycle_hours": 12.0,
                     "members": 20, "spinup_cycles": 20, "obs_error_z": 10.0,
                     "obs_error_wind": 2.0,
                     "variables": ["psi1", "psi2"], "inflation": 1.1,
                     "localization_km": 3000.0, "stations": [100], "primary": 100, "seed": 4},
}


def model_params(model_cfg: dict[str, Any]) -> Params:
    """Build Params from the [model] table, resolving the convenience keys."""
    cfg = dict(model_cfg)
    r_days = cfg.pop("r_ek_days", None)
    tau_h = cfg.pop("filter_tau_hours", None)
    ratio = cfg.pop("shear_ratio", None)
    if r_days is not None:
        cfg["r_ek"] = 1.0 / (float(r_days) * DAY)
    if tau_h is not None:
        cfg["filter_tau"] = float(tau_h) * 3600.0
    p = Params.from_dict(cfg)
    if ratio is not None:
        p = p.replace(U1=p.U2 + float(ratio) * p.dU_crit)
    return p


def apply_override(p: Params, name: str, value: float) -> Params:
    """Apply one swept parameter (a Params field or a convenience key)."""
    if name == "shear_ratio":
        return p.replace(U1=p.U2 + float(value) * p.dU_crit)
    if name == "r_ek_days":
        return p.replace(r_ek=1.0 / (float(value) * DAY))
    if name == "filter_tau_hours":
        return p.replace(filter_tau=float(value) * 3600.0)
    return p.replace(**{name: value})


def _merge(base: dict[str, Any], new: dict[str, Any], path: str = "") -> dict[str, Any]:
    out = dict(base)
    for k, v in new.items():
        if k not in base:
            raise KeyError(f"unknown config key '{path}{k}'")
        if isinstance(base[k], dict) and isinstance(v, dict):
            out[k] = _merge(base[k], v, path + k + ".")
        else:
            out[k] = v
    return out


class Config:
    def __init__(self, raw: dict[str, Any]):
        raw = dict(raw)
        model = raw.pop("model", {})
        self.raw = _merge(DEFAULTS, raw)
        self.raw["model"] = model
        self.params = model_params(model)
        self.title: str = self.raw["title"]
        self.description: str = self.raw["description"]
        self.control: dict[str, Any] = self.raw["control"]
        self.sweep: dict[str, Any] = self.raw["sweep"]
        self.ensemble: dict[str, Any] = self.raw["ensemble"]
        self.assim: dict[str, Any] = self.raw["assimilation"]
        self._check()

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "rb") as f:
            return cls(tomllib.load(f))

    def _check(self) -> None:
        p = self.params
        n = p.nx
        if n & (n - 1) or p.ny & (p.ny - 1):
            raise ValueError("nx and ny must be powers of two (the web explorer uses a radix-2 FFT)")
        if self.control["output_hours"] * 3600 % p.dt:
            raise ValueError("control.output_hours must be a multiple of the model time step")
        for key in ("output_hours",):
            if self.sweep["output_hours"] * 3600 % p.dt:
                raise ValueError("sweep.output_hours must be a multiple of the time step")
        if self.assim["enabled"]:
            if self.assim["cycle_hours"] % self.control["output_hours"]:
                raise ValueError("assimilation.cycle_hours must be a multiple of control.output_hours")
            if self.assim["primary"] not in self.assim["stations"]:
                raise ValueError("assimilation.primary must be one of assimilation.stations")
            horizon = self.assim["start_day"] + self.assim["days"]
            if horizon > self.control["length_days"]:
                raise ValueError("the assimilation window must fit inside the control run")
        if self.ensemble["enabled"]:
            horizon = self.ensemble["start_day"] + self.ensemble["lead_days"]
            if horizon > self.control["length_days"]:
                raise ValueError("the ensemble forecasts must fit inside the control run")
            if self.ensemble["output_hours"] % self.control["output_hours"]:
                raise ValueError("ensemble.output_hours must be a multiple of control.output_hours")
