"""Writing the web dataset: raw little-endian binary arrays plus JSON descriptors.

Descriptor of one array (what the browser needs to read it)::

    {"file": "control_psi.i16", "dtype": "int16", "shape": [720, 2, 64, 64],
     "scale": [s1, s2], "scale_axis": 1}        # value = int16 * scale[index along axis 1]

``float32`` arrays carry no scale.  Small arrays may also be written inline in the JSON.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = "qg2l-dataset"
VERSION = 1
_EXT = {"int16": ".i16", "float32": ".f32"}


def round_floats(obj: Any, digits: int = 6) -> Any:
    """Round floats in nested lists/dicts to `digits` significant digits (keeps JSON small)."""
    if isinstance(obj, float):
        return float(f"{obj:.{digits}g}")
    if isinstance(obj, (np.floating,)):
        return float(f"{float(obj):.{digits}g}")
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return round_floats(obj.tolist(), digits)
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, digits) for v in obj]
    if isinstance(obj, dict):
        return {k: round_floats(v, digits) for k, v in obj.items()}
    return obj


class Writer:
    def __init__(self, outdir: str | Path):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

    def array(self, name: str, arr: np.ndarray, dtype: str = "float32",
              scale_axis: int | None = None) -> dict[str, Any]:
        """Write `arr` to <name>.<ext> and return its descriptor."""
        if dtype not in _EXT:
            raise ValueError(f"dtype must be one of {sorted(_EXT)}, not {dtype!r}")
        arr = np.asarray(arr)
        desc: dict[str, Any] = {"file": name + _EXT[dtype], "dtype": dtype, "shape": list(arr.shape)}
        if dtype == "int16":
            if scale_axis is None:
                amax = float(np.max(np.abs(arr))) or 1.0
                scale: Any = float(f"{amax / 30000.0:.6g}") or 1.0
                q = np.rint(arr / scale)
            else:
                other = tuple(i for i in range(arr.ndim) if i != scale_axis)
                amax = np.max(np.abs(arr), axis=other)
                scale = [float(f"{(a if a > 0 else 1.0) / 30000.0:.6g}") for a in amax]
                shape = [1] * arr.ndim
                shape[scale_axis] = -1
                q = np.rint(arr / np.array(scale).reshape(shape))
                desc["scale_axis"] = scale_axis
            data = np.clip(q, -32767, 32767).astype("<i2")
            desc["scale"] = scale
        elif dtype == "float32":
            data = arr.astype("<f4")
        else:
            raise ValueError(dtype)
        (self.outdir / desc["file"]).write_bytes(data.tobytes())
        return desc

    def json(self, name: str, obj: Any, digits: int = 6) -> str:
        path = self.outdir / name
        path.write_text(json.dumps(round_floats(obj, digits), separators=(",", ":")))
        return name


def read_array(outdir: str | Path, desc: dict[str, Any]) -> np.ndarray:
    """Inverse of Writer.array (used by tests and the reference generator)."""
    raw = np.fromfile(Path(outdir) / desc["file"], dtype="<i2" if desc["dtype"] == "int16" else "<f4")
    arr = raw.reshape(desc["shape"]).astype(np.float64)
    if desc["dtype"] == "int16":
        s = desc["scale"]
        if isinstance(s, list):
            shape = [1] * arr.ndim
            shape[desc["scale_axis"]] = -1
            arr = arr * np.array(s).reshape(shape)
        else:
            arr = arr * s
    return arr
