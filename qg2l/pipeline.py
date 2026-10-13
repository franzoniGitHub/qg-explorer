"""Run the experiments and assemble the dataset (manifest.json + binary arrays)."""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import __version__
from . import experiments as ex
from .config import Config
from .export import FORMAT, VERSION, Writer
from .model import DAY, QG2L

STAGES = ("control", "sweep", "ensemble", "assimilation")
# bump a stage's tag when the physics or that experiment changes in a way that invalidates its cache
CODE_TAGS = {"control": "1", "sweep": "1", "ensemble": "1", "assimilation": "2"}


def _stage_cfg(cfg: Config, stage: str) -> dict[str, Any]:
    return {"control": cfg.control, "sweep": cfg.sweep, "ensemble": cfg.ensemble,
            "assimilation": cfg.assim}[stage]


def _hash(cfg: Config, stage: str) -> str:
    blob = {"model": cfg.params.to_dict(), "stage": _stage_cfg(cfg, stage), "tag": CODE_TAGS[stage]}
    if stage != "control":                       # downstream stages depend on the control run
        blob["control"] = cfg.control
    return hashlib.sha1(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:16]


def derived_model(cfg: Config) -> dict[str, Any]:
    p = cfg.params
    d = p.to_dict()
    d.update({"kd2": p.kd2, "F1": p.F1, "F2": p.F2, "G": p.G, "beta1": p.beta1, "beta2": p.beta2,
              "dU": p.dU, "dU_crit": p.dU_crit, "dU_ratio": p.dU / p.dU_crit,
              "dx": p.dx, "dy": p.dy, "r_ek_days": 1.0 / p.r_ek / DAY if p.r_ek > 0 else None})
    return d


def run_pipeline(cfg: Config, outdir: str | Path, workdir: str | Path | None = None,
                 stages: list[str] | None = None, workers: int | None = None,
                 force: bool = False, log: Callable[[str], None] = print) -> dict[str, Any]:
    outdir = Path(outdir)
    work = Path(workdir) if workdir else outdir.parent / f".work-{outdir.name}"
    work.mkdir(parents=True, exist_ok=True)
    w = Writer(outdir)
    workers = workers or max(1, (os.cpu_count() or 1))
    enabled = ["control"] + [s for s in STAGES[1:] if _stage_cfg(cfg, s).get("enabled")]
    wanted = enabled if not stages else ["control"] + [s for s in STAGES[1:] if s in stages]
    for s in wanted:
        if s not in enabled:
            log(f"stage '{s}' is disabled in the config; skipping")
    wanted = [s for s in wanted if s in enabled]

    fragments: dict[str, Any] = {}
    t_all = time.time()
    for stage in wanted:
        h = _hash(cfg, stage)
        frag_path = work / f"{stage}.fragment.json"
        if not force and frag_path.exists():
            saved = json.loads(frag_path.read_text())
            if saved.get("_hash") == h:
                log(f"[{stage}] up to date (cached) - use --force to recompute")
                fragments.update({k: v for k, v in saved.items() if k != "_hash"})
                continue
        log(f"[{stage}] running ...")
        t0 = time.time()
        if stage == "control":
            frag = ex.run_control(cfg, w, work, log, workers=min(2, workers))
        elif stage == "sweep":
            frag = ex.run_sweep(cfg, w, work, log, workers)
        elif stage == "ensemble":
            frag = ex.run_ensemble(cfg, w, work, log, workers)
        else:
            frag = ex.run_assimilation(cfg, w, work, log, workers)
        log(f"[{stage}] done in {time.time() - t0:.0f} s")
        frag_path.write_text(json.dumps({**frag, "_hash": h}))
        fragments.update(frag)

    p = cfg.params
    manifest: dict[str, Any] = {
        "format": FORMAT, "version": VERSION,
        "title": cfg.title, "description": cfg.description,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": f"qg2l {__version__}",
        "model": derived_model(cfg),
        "grid": {"nx": p.nx, "ny": p.ny, "Lx": p.Lx, "Ly": p.Ly},
        "config": {**{k: v for k, v in cfg.raw.items() if k != "model"}, "model": cfg.raw["model"]},
    }
    manifest.update(fragments)
    from .reference import make_reference
    manifest["reference"] = make_reference(cfg, w, manifest)
    (outdir / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":"), default=str))
    update_index(outdir.parent, log)
    size = sum(f.stat().st_size for f in outdir.iterdir() if f.is_file()) / 1e6
    log(f"dataset written to {outdir}  ({size:.1f} MB, {time.time() - t_all:.0f} s total)")
    return manifest


def update_index(datadir: Path, log: Callable[[str], None] = print, force: bool = False) -> None:
    """Write <datadir>/index.json listing every dataset folder below it (used by the web page)."""
    idx = datadir / "index.json"
    if not force and not (idx.exists() or datadir.name == "data"):
        return
    items = []
    for sub in sorted(datadir.iterdir()):
        mf = sub / "manifest.json"
        if sub.is_dir() and mf.exists():
            try:
                m = json.loads(mf.read_text())
            except json.JSONDecodeError:
                continue
            size = sum(f.stat().st_size for f in sub.iterdir() if f.is_file()) / 1e6
            items.append({"id": sub.name, "path": sub.name + "/", "title": m.get("title", sub.name),
                          "description": m.get("description", ""), "created": m.get("created", ""),
                          "size_mb": round(size, 1)})
    idx.write_text(json.dumps({"datasets": items}, indent=1))
    log(f"index updated: {idx}  ({len(items)} dataset(s))")
