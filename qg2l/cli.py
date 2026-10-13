"""Command line interface:  python -m qg2l <command> ..."""
from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import sys
import webbrowser
from pathlib import Path

from .config import Config
from .pipeline import STAGES, derived_model, run_pipeline, update_index
from .stability import fastest_mode


def _log(msg: str) -> None:
    print(msg, flush=True)


def cmd_run(a: argparse.Namespace) -> int:
    cfg = Config.load(a.config)
    stages = a.stages.split(",") if a.stages else None
    if stages:
        bad = [s for s in stages if s not in STAGES]
        if bad:
            sys.exit(f"unknown stage(s) {bad}; choose from {list(STAGES)}")
    run_pipeline(cfg, a.out, workdir=a.workdir, stages=stages, workers=a.workers,
                 force=a.force, log=_log)
    return 0


def cmd_info(a: argparse.Namespace) -> int:
    cfg = Config.load(a.config)
    d = derived_model(cfg)
    p = cfg.params
    k, s, tau = fastest_mode(p)
    print(f"{cfg.title}")
    print(f"  grid                      {p.nx} x {p.ny}, dx = {p.dx / 1e3:.0f} km, domain {p.Lx / 1e3:.0f} x {p.Ly / 1e3:.0f} km")
    print(f"  deformation radius Ld     {p.Ld / 1e3:.0f} km     (F1 = {d['F1']:.3g}, F2 = {d['F2']:.3g} m^-2)")
    print(f"  shear U1 - U2             {p.dU:.2f} m/s")
    print(f"  critical shear beta/F2    {d['dU_crit']:.2f} m/s   ->  supercriticality {d['dU_ratio']:.2f}")
    print(f"  fastest linear mode       wavelength {2 * 3.14159265 / k / 1e3:.0f} km, e-folding {tau:.1f} days (zonal waves)")
    print(f"  Ekman spin-down time      {d['r_ek_days']:.1f} days")
    print(f"  time step                 {p.dt:.0f} s")
    return 0


def cmd_index(a: argparse.Namespace) -> int:
    update_index(Path(a.datadir), _log, force=True)
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    root = Path(a.dir).resolve()
    if not (root / "index.html").exists():
        sys.exit(f"{root} has no index.html - run from the repository root or pass --dir docs")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", a.port), handler) as httpd:
        url = f"http://127.0.0.1:{a.port}/"
        print(f"serving {root} at {url}   (Ctrl-C to stop)")
        if not a.no_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m qg2l", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the experiments and write a web dataset")
    r.add_argument("config", help="TOML configuration file, e.g. config/default.toml")
    r.add_argument("--out", required=True, help="output folder, e.g. docs/data/default")
    r.add_argument("--workers", type=int, default=None, help="processes for the parallel stages")
    r.add_argument("--stages", default=None, help=f"comma-separated subset of {list(STAGES)}")
    r.add_argument("--workdir", default=None, help="scratch folder for restart files (default: next to --out)")
    r.add_argument("--force", action="store_true", help="recompute even if cached results exist")
    r.set_defaults(fn=cmd_run)

    i = sub.add_parser("info", help="print the derived parameters of a configuration")
    i.add_argument("config")
    i.set_defaults(fn=cmd_info)

    x = sub.add_parser("index", help="(re)write index.json for a folder of datasets")
    x.add_argument("datadir")
    x.set_defaults(fn=cmd_index)

    s = sub.add_parser("serve", help="serve the website locally")
    s.add_argument("--dir", default="docs")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(fn=cmd_serve)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
