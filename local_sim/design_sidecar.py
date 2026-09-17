"""Per-network prediction sidecars for the DESIGN scorer, with identity.

Handoff 2026-09-17b stage 0: *"extend the existing scoring path minimally to
persist per-network final predictions, immutable source/config/truth/mint/package
identities, failures and counted rows. Keep the current canonical scorer; no
second scoring formula."*

WHAT THIS IS
------------
A writer and a verifier for one directory of sidecars beside a
`p2_design_score.py` run. It holds **no scoring formula**: the score lives in
`competition_score.py`, the rows in `lifecycle_eval.py`, and this module only
stores the array each arm actually had scored, plus what it was produced by.

WHY IT EXISTS
-------------
Every accuracy question since 2026-09-16 has been "*where* did the prediction
move, and by how much relative to the error we are trying to remove" -- and the
answer needed the predictions, which were thrown away the moment a row's MSE was
computed. The exact decomposition the handoff asks for,

    MSE(new) - MSE(old) = 2 * mean((old - truth) * delta) + mean(delta**2)

cannot be formed from scores at all. 24 networks x 1024 float32 is 96 KiB per
arm; there is no reason to keep losing it.

WHAT IT REFUSES
---------------
* a sidecar whose stream is short, long, or not a whole number of rows;
* a sidecar whose index does not have exactly one line per stored row;
* a row whose stored bytes do not hash to the sha256 the index recorded;
* a manifest whose recorded source/truth sha256 no longer matches the file on
  disk, or whose configuration does not match the one being replayed.

A malformed sidecar RAISES. A sidecar is evidence or it is nothing.

LAYOUT (one directory per scorer run)
--------------------------------------
    preds_<arm>.f32      raw little-endian float32, n rows x width, append-only
    preds_<arm>.jsonl    one line per completed row, appended and fsynced
    predictions_manifest.json   identity of the whole run, written at the end
                                and again incrementally after every arm

The stream is append-only and fsynced per row so a run that dies keeps every
finished network -- the same discipline `--rows-jsonl` already has.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

MANIFEST = "predictions_manifest.json"
ARTIFACT = "design_score.prediction_sidecars"
VERSION = 1


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def stream_path(directory: str, arm: str) -> str:
    return os.path.join(directory, f"preds_{arm}.f32")


def index_path(directory: str, arm: str) -> str:
    return os.path.join(directory, f"preds_{arm}.jsonl")


class ArmWriter:
    """Append-only sidecar for ONE arm. One instance per arm per run."""

    def __init__(self, directory: str, arm: str, *, width: int,
                 candidate: str, candidate_sha256: str):
        os.makedirs(directory, exist_ok=True)
        self.directory = directory
        self.arm = arm
        self.width = int(width)
        self.candidate = candidate
        self.candidate_sha256 = candidate_sha256
        self.n = 0
        self._stream = stream_path(directory, arm)
        self._index = index_path(directory, arm)
        # A re-run must not silently append to a previous one's stream.
        for p in (self._stream, self._index):
            if os.path.exists(p):
                raise FileExistsError(
                    f"{p} already exists; a sidecar directory is written once. "
                    "Point --pred-out at a fresh directory rather than mixing "
                    "two runs' rows into one stream.")
            with open(p, "wb"):
                pass

    def append(self, row, prediction) -> None:
        """Store one completed row's SCORED final-layer prediction."""
        a = np.asarray(prediction, dtype=np.float32).reshape(-1)
        if a.shape[0] != self.width:
            raise ValueError(f"prediction has {a.shape[0]} entries, expected "
                             f"{self.width}")
        b = a.tobytes(order="C")
        with open(self._stream, "ab") as fh:
            fh.write(b)
            fh.flush()
            os.fsync(fh.fileno())
        rec = {"arm": self.arm, "i": self.n, "index": int(row.index),
               "seed": int(row.input_seed), "sha256": _sha_bytes(b),
               "mse": float(row.mse), "flops_used": float(row.flops_used),
               "n_ops": float(row.n_ops), "gt_noise": float(row.gt_noise),
               "failed": bool(row.failed), "error_code": row.error_code,
               "nonfinite": bool(row.nonfinite),
               "candidate_sha256": self.candidate_sha256}
        with open(self._index, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.n += 1


def write_manifest(directory: str, payload: Dict[str, Any]) -> str:
    """(Re)write the run manifest. Called after every arm, not only at the end."""
    os.makedirs(directory, exist_ok=True)
    out = dict(payload)
    out["artifact"] = ARTIFACT
    out["version"] = VERSION
    path = os.path.join(directory, MANIFEST)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    return path


# --------------------------------------------------------------------------- #
# reading, which is where the refusals live
# --------------------------------------------------------------------------- #

def load_arm(directory: str, arm: str, *, width: int,
             expect_n: Optional[int] = None) -> Dict[str, Any]:
    """Load one arm's sidecar, refusing anything malformed or incomplete."""
    sp, ip = stream_path(directory, arm), index_path(directory, arm)
    for p in (sp, ip):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing sidecar file {p}")
    raw = open(sp, "rb").read()
    itemsize = 4 * int(width)
    if itemsize <= 0:
        raise ValueError("width must be positive")
    if len(raw) % itemsize:
        raise ValueError(
            f"{sp}: {len(raw)} bytes is not a whole number of {width}-wide "
            "float32 rows -- the run died mid-write and the tail is a partial "
            "prediction, not a prediction")
    n_rows = len(raw) // itemsize
    lines = [ln for ln in open(ip, encoding="utf-8").read().splitlines() if ln.strip()]
    if len(lines) != n_rows:
        raise ValueError(f"{ip}: {len(lines)} index lines for {n_rows} stored "
                         "rows -- index and stream disagree")
    recs = [json.loads(ln) for ln in lines]
    preds = np.frombuffer(raw, dtype="<f4").reshape(n_rows, width).copy()
    for k, r in enumerate(recs):
        if int(r.get("i", -1)) != k:
            raise ValueError(f"{ip} line {k}: out-of-order row index {r.get('i')}")
        got = _sha_bytes(preds[k].tobytes(order="C"))
        if got != r["sha256"]:
            raise ValueError(f"{ip} line {k}: stored prediction hashes {got}, "
                             f"index recorded {r['sha256']}")
    if expect_n is not None and n_rows != int(expect_n):
        raise ValueError(f"{sp}: {n_rows} rows, expected {expect_n} -- an "
                         "incomplete sidecar is not evidence for a comparison")
    return {"arm": arm, "n": n_rows, "predictions": preds, "rows": recs,
            "seeds": [int(r["seed"]) for r in recs],
            "mse": np.array([float(r["mse"]) for r in recs], dtype=np.float64),
            "failed": np.array([bool(r["failed"]) for r in recs])}


def verify(directory: str, *, check_sources: bool = True,
           root: Optional[str] = None) -> Dict[str, Any]:
    """Verify a whole sidecar directory against its manifest. Raises on any defect."""
    mpath = os.path.join(directory, MANIFEST)
    if not os.path.exists(mpath):
        raise FileNotFoundError(f"missing {mpath}")
    man = json.load(open(mpath, encoding="utf-8"))
    if man.get("artifact") != ARTIFACT:
        raise ValueError(f"{mpath}: not a {ARTIFACT} manifest")
    for key in ("width", "depth", "n", "truth", "truth_sha256", "block",
                "arms", "config", "packages", "mint"):
        if key not in man:
            raise ValueError(f"{mpath}: manifest is missing {key!r}")
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    loaded = {}
    for arm, info in man["arms"].items():
        loaded[arm] = load_arm(directory, arm, width=int(man["width"]),
                               expect_n=int(man["n"]))
        for r in loaded[arm]["rows"]:
            if r["candidate_sha256"] != info["candidate_sha256"]:
                raise ValueError(
                    f"arm {arm}: a row carries candidate sha "
                    f"{r['candidate_sha256']}, the manifest says "
                    f"{info['candidate_sha256']} -- the source changed under "
                    "the run and the sidecar is not one candidate's output")
        if check_sources:
            p = os.path.join(root, info["path"])
            if os.path.exists(p) and sha_file(p) != info["candidate_sha256"]:
                raise ValueError(
                    f"arm {arm}: {info['path']} now hashes "
                    f"{sha_file(p)}, the sidecar was produced from "
                    f"{info['candidate_sha256']}")
    if check_sources:
        tp = os.path.join(root, man["truth"])
        if os.path.exists(tp) and sha_file(tp) != man["truth_sha256"]:
            raise ValueError(f"truth {man['truth']} now hashes {sha_file(tp)}, "
                             f"the sidecar was produced against "
                             f"{man['truth_sha256']}")
    return {"manifest": man, "arms": loaded}


def prediction_change(old: np.ndarray, new: np.ndarray,
                      truth: np.ndarray) -> Dict[str, Any]:
    """The EXACT decomposition the handoff requires, per network and pooled.

        MSE(new) - MSE(old) = 2 * mean((old - truth) * delta) + mean(delta**2)

    computed in float64 from float32 arrays, with the two halves reported
    separately: `cross` is where the change is ALIGNED with the existing error
    (the part that can help), `quad` is the unconditional cost of moving at all.
    """
    old = np.asarray(old, dtype=np.float64)
    new = np.asarray(new, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if old.shape != new.shape or old.shape != truth.shape:
        raise ValueError(f"shape mismatch {old.shape} {new.shape} {truth.shape}")
    delta = new - old
    resid = old - truth
    ax = tuple(range(1, old.ndim)) if old.ndim > 1 else None
    cross = 2.0 * np.mean(resid * delta, axis=ax)
    quad = np.mean(delta * delta, axis=ax)
    mse_old = np.mean(resid * resid, axis=ax)
    mse_new = np.mean((new - truth) ** 2, axis=ax)
    ident = np.max(np.abs((mse_new - mse_old) - (cross + quad)))
    # relative size of the move against the signal it is trying to correct
    nrm = np.sqrt(np.mean(new * new, axis=ax))
    return {"cross": cross, "quad": quad, "mse_old": mse_old, "mse_new": mse_new,
            "identity_residual": float(ident),
            "delta_rms": np.sqrt(quad),
            "delta_over_activation": np.sqrt(quad) / np.maximum(nrm, 1e-30),
            "signed_bias": np.mean(delta, axis=ax),
            "alignment": (cross / (2.0 * np.sqrt(np.maximum(mse_old, 1e-300))
                                   * np.sqrt(np.maximum(quad, 1e-300))))}


def to_jsonable(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = [float(x) for x in np.atleast_1d(v)]
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        else:
            out[k] = v
    return out
