"""Export twin-fitted corrector pages into a new estimator file.

    python tools/build_twin.py --base estimator_k3_v59_corr400.py \
        --params experiments/<dir>/params_best.npz --out estimator_k3_v62_twin.py

HANDOFF 2026-09-25 Step 2. The twin (`tools/twin/twin.py`) trains the shipped
file's own corrector arrays through the exact chain; this writes them back. ONLY
the `_CORR_B64 = {...}` block changes -- the same base64-of-float32 encoding the
file already decodes (exact: base64 of the raw bytes round-trips bit for bit), so
FLOPs and op counts are unchanged by construction, and the diff against the base
is one data block. Arrays absent from the params file keep the base's bytes.
"""
from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ORDER = ("fmu", "fisd", "W1", "b1", "W2", "b2", "W3", "b3")


def _decode_block(src: str) -> dict:
    m = re.search(r"^_CORR_B64 = \{\n(.*?)^\}\n", src, flags=re.S | re.M)
    if not m:
        raise SystemExit("no _CORR_B64 block")
    body = m.group(1)
    out = {}
    for name, blob, shape in re.findall(
            r'^    "(\w+)": \(\n((?:        "[^"]*"\n)+)    , \(([\d, ]+)\)\),\n',
            body, flags=re.M):
        b64 = "".join(re.findall(r'"([^"]*)"', blob))
        shp = tuple(int(x) for x in shape.replace(" ", "").split(",") if x)
        out[name] = np.frombuffer(base64.b64decode(b64), dtype=np.float32).reshape(shp)
    if set(out) != set(ORDER):
        raise SystemExit(f"parsed {sorted(out)}, expected {sorted(ORDER)}")
    return out, m.span()


def _encode(name: str, a: np.ndarray) -> str:
    a = np.ascontiguousarray(a, dtype=np.float32)
    b64 = base64.b64encode(a.tobytes()).decode()
    lines = "".join(f'        "{b64[i:i + 96]}"\n' for i in range(0, len(b64), 96))
    shape = ", ".join(str(d) for d in a.shape) + ("," if a.ndim == 1 else "")
    return f'    "{name}": (\n{lines}    , ({shape})),\n'


def build(base: Path, params: Path) -> str:
    src = base.read_text(encoding="utf-8")
    arrays, (lo, hi) = _decode_block(src)
    new = dict(np.load(params))
    for k, v in new.items():
        if k not in arrays:
            raise SystemExit(f"params has {k!r}, which the base file does not carry")
        if v.shape != arrays[k].shape:
            raise SystemExit(f"{k}: shape {v.shape} != base {arrays[k].shape}")
        arrays[k] = v
    block = "_CORR_B64 = {\n" + "".join(_encode(k, arrays[k]) for k in ORDER) + "}\n"
    return src[:lo] + block + src[hi:]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="estimator_k3_v59_corr400.py")
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true",
                    help="re-encode the base's own arrays and require identical bytes")
    a = ap.parse_args(argv)
    base = ROOT / a.base if not Path(a.base).is_absolute() else Path(a.base)
    if a.selftest:
        src = base.read_text(encoding="utf-8")
        arrays, (lo, hi) = _decode_block(src)
        block = "_CORR_B64 = {\n" + "".join(_encode(k, arrays[k]) for k in ORDER) + "}\n"
        same = src[lo:hi] == block
        print("selftest: re-encoded block identical to base:", same)
        return 0 if same else 1
    out = build(base, Path(a.params))
    Path(a.out).write_text(out, encoding="utf-8", newline="\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
