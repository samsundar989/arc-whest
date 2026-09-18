"""Bounded P12 DESIGN own-fit driver; dry-run is label-free.

The candidate schedule is fixed here before any fit read: baseline, then two
coordinate passes in order theta_fam, s_lam, t_lam, s_feed. Each coordinate
evaluates its negative and positive finite step at the current accepted point;
the best penalized training score (including the current point) is retained.
Pass 1 steps are (0.25, 0.15, 0.20, 0.15), pass 2 halves them. After 17
training evaluations, the frozen default and selected point are evaluated on
the last 12 seeds. Total: 19 candidate evaluations, 228 full MLP runs, <=40.
No early stopping or data-dependent extra evaluations. Ties keep the incumbent.

The dimensionless training objective is mean official-precision adjusted score
divided by the frozen default PAIR+Q mean on the same first 12 networks, plus
1e-3 * ||x-(1,1,0,1)||^2. The default is evaluated and durably frozen before
any nondefault evaluation. Validation is never consulted during fitting.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
TRUTH = ROOT / "experiments/2026-09-17b_p12/truth_P12.npz"
TRUTH_SHA = "92a3228edee16080795c4a1ea0124b8c7bd1f7f7ecf7c84ad078edc4eca756b2"
P12_START = 40917012000
SEEDS = tuple(P12_START+i for i in range(24))
NAMES = ("theta_fam", "s_lam", "t_lam", "s_feed")
X0 = (1.0, 1.0, 0.0, 1.0)
BOUNDS = ((0., 1.5), (.5, 1.5), (-.5, .5), (.5, 1.5))
STEPS = ((.25, .15, .20, .15), (.125, .075, .10, .075))
RIDGE = 1e-3
MAX_EVALS = 40
FIT_N = VALID_N = 12
TOTAL_EVALS = 1 + 2*len(NAMES)*len(STEPS) + 2


def nearby(x, coordinate, direction, step):
    v = list(x)
    lo, hi = BOUNDS[coordinate]
    v[coordinate] = min(hi, max(lo, v[coordinate] + direction*step))
    return tuple(v)


def objective(score, baseline, x):
    if not (math.isfinite(score) and math.isfinite(baseline) and baseline > 0):
        raise ValueError("invalid official-precision fit score or baseline")
    return score/baseline + RIDGE*sum((a-b)**2 for a,b in zip(x,X0))


def plan():
    return {"tier":"DESIGN", "block":"P12", "ordered_seeds":list(SEEDS),
            "fit_indices":list(range(FIT_N)),
            "validation_indices":list(range(FIT_N,FIT_N+VALID_N)),
            "parameters":list(NAMES), "default":list(X0),
            "bounds":[list(x) for x in BOUNDS],
            "coordinate_order":list(NAMES),
            "passes":[{"step":dict(zip(NAMES,step)),
                       "direction_order":[-1,1]} for step in STEPS],
            "tie_rule":"retain incumbent", "early_stop":False,
            "objective":"fit mean official adjusted / frozen default fit mean + 0.001 squared distance",
            "ridge":RIDGE,"max_evaluations":MAX_EVALS,
            "scheduled_evaluations":TOTAL_EVALS,"full_network_runs":FIT_N*TOTAL_EVALS,
            "validation_rule":"matched default and selected points, last 12 only after fit freeze; never refit",
            "workerq_request":{"ram_gib":12,"vram_gib":0,"cpus":4},
            "workerq_line":"workerq submit --project arc-whest --ram 12 --vram 0 --cpus 4 -- uv run python local_sim/s15_own_fit_driver.py --execute --out-dir <absolute-path> --ts <registered-utc>",
            "optimization_review":"Reuse one truth load and the existing grader lifecycle; stream each row and atomic evaluation checkpoint. Do not cache 12 large MLP weight sets or bypass metered estimator work. Each candidate is priced on all 12 fit networks; matched default and selected validation are priced on all 12 held-out networks. 19 evaluations are below the 40 cap. Source stays per-candidate frozen; no automatic DECIDE reuse."}


def coordinate_fit(evaluate):
    """Pure deterministic scheduler; callback returns one official score."""
    history=[]
    x=X0
    base=evaluate("fit",x,len(history)); history.append(("fit",x,base))
    incumbent=objective(base,base,x)
    for steps in STEPS:
        for j,step in enumerate(steps):
            best_x,best_obj=x,incumbent
            for direction in (-1,1):
                trial=nearby(x,j,direction,step)
                score=evaluate("fit",trial,len(history)); history.append(("fit",trial,score))
                value=objective(score,base,trial)
                if value < best_obj:
                    best_x,best_obj=trial,value
            x,incumbent=best_x,best_obj
    held_base=evaluate("validation",X0,len(history)); history.append(("validation",X0,held_base))
    held=evaluate("validation",x,len(history)); history.append(("validation",x,held))
    assert len(history)==TOTAL_EVALS<=MAX_EVALS
    return x,base,held_base,held,history


def _atomic_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    with open(tmp,"w",encoding="utf-8") as fh:
        json.dump(data,fh,indent=2,sort_keys=True,default=str)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp,path)


def design_spec():
    """P12 comparison spec with explicit, auditable host-seconds deviations."""
    import lifecycle_eval as le
    return le.LifecycleSpec.for_round(
        "v2-phase2", residual_wall_time_limit_s=None,
        wall_time_limit_s=None, predict_timeout_s=1e9,
        setup_timeout_s=1e9)


def execute(out_dir, ts):
    import numpy as np
    import competition_score as cs
    import lifecycle_eval as le
    import s14_own_fit_controls as gen
    import truth_reads as tr

    out_dir=Path(out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("output directory must be empty; no automatic fit recovery")
    if not ts:
        raise ValueError("registered UTC read timestamp required")
    if hashlib.sha256(TRUTH.read_bytes()).hexdigest()!=TRUTH_SHA:
        raise ValueError("P12 truth hash mismatch")
    with tr.reserve("P12",str(TRUTH),consumer="local_sim/s15_own_fit_driver.py",
                    purpose="registered 12/12 own PAIR+Q global fit",
                    ts=ts,instrument=str(Path(__file__).resolve()),
                    result_artifact=str(out_dir/"result.json")) as truth_path:
        with np.load(truth_path) as z:
            if tuple(int(s) for s in z["seeds"])!=SEEDS:
                raise ValueError("P12 ordered network identities differ")
            if int(z["width"])!=1024 or int(z["depth"])!=16 or int(z["n_samples"])!=2**24:
                raise ValueError("P12 shape or precision differs")
            truth=(z["means_r1"][:,-1,:].astype(np.float64)
                   +z["means_r2"][:,-1,:].astype(np.float64))/2
            noise=z["gt_noise"].astype(np.float64)/2
    # Match the P12 DESIGN scorer: host-local seconds would turn sound methods
    # into zero-scored failures on this slower machine. Full round deviations
    # remain visible in each evaluation's provenance; release is a separate
    # official-package subprocess and Linux memory gate.
    spec=design_spec()
    manifest=plan(); manifest.update({"truth_sha256":TRUTH_SHA,
                                     "base_source_sha256":hashlib.sha256(gen.BASE.read_bytes()).hexdigest(),
                                     "driver_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                     "generator_sha256":hashlib.sha256(Path(gen.__file__).read_bytes()).hexdigest(),
                                     "dependency_sha256":{
                                         name:hashlib.sha256((ROOT/"local_sim"/name).read_bytes()).hexdigest()
                                         for name in ("competition_score.py","lifecycle_eval.py",
                                                      "mint_provenance.py","official_config.py",
                                                      "truth_reads.py")},
                                     "numerical_environment":{
                                         "python":sys.version,"numpy":np.__version__,
                                         "platform":platform.platform(),
                                         "executable":sys.executable,
                                         "threads":{k:os.environ.get(k) for k in
                                                    ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
                                                     "MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")}},
                                     "local_seconds_caps":"disabled for P12 DESIGN; not release evidence",
                                     "status":"running","evaluations":[]})
    _atomic_json(out_dir/"manifest.json",manifest)
    frozen_baseline=[None]

    def run(split,x,index):
        if index>=MAX_EVALS: raise ValueError("evaluation cap exceeded")
        name=f"eval_{index:02d}_{split}"
        source=out_dir/(name+".py")
        sha=gen.emit(source,**dict(zip(NAMES,x)))
        selected=range(0,12) if split=="fit" else range(12,24)
        ids=[SEEDS[i] for i in selected]
        yy=truth[list(selected)]
        vv=noise[list(selected)]
        rows_path=out_dir/(name+".jsonl")
        def sink(row):
            with open(rows_path,"a",encoding="utf-8") as fh:
                fh.write(json.dumps(dataclasses.asdict(row),sort_keys=True)+"\n")
                fh.flush();os.fsync(fh.fileno())
        rows,prov=le.evaluate(str(source),ids,yy,spec=spec,gt_noise=list(vv),
                           evidence_sink=sink)
        if len(rows)!=12 or [r.input_seed for r in rows]!=ids:
            raise ValueError("incomplete or mismatched fit evaluation")
        if any(r.failed for r in rows):
            raise ValueError("failed candidate; stop fit, preserve row evidence")
        official=[dataclasses.replace(r.to_row(),mse=r.mse-r.gt_noise+
                                      r.gt_noise*(2*2**24/1e9)) for r in rows]
        score=cs.score_suite(official,lam=0.).objective
        observed=le.score(rows,spec=spec).objective
        rec={"index":index,"split":split,"parameters":dict(zip(NAMES,x)),
             "candidate":source.name,"sha256":sha,"rows":rows_path.name,
             "rows_hash":le.rows_hash(rows),
             "rows_file_sha256":hashlib.sha256(rows_path.read_bytes()).hexdigest(),
             "provenance":prov,"n":len(rows),"n_failed":0,
             "observed_score":observed,"official_precision_score":score,
             "mean_flops":sum(r.flops_used for r in rows)/len(rows),
             "gt_noise_mean":float(vv.mean())}
        manifest["evaluations"].append(rec)
        if index==0:
            frozen_baseline[0]=score
            manifest["frozen_default_fit_score"]=score
        elif split=="fit":
            rec["penalized_fit_objective"]=objective(score,frozen_baseline[0],x)
        _atomic_json(out_dir/"manifest.json",manifest)
        return score

    x,base,held_base,held,history=coordinate_fit(run)
    manifest.update({"status":"complete","selected_parameters":dict(zip(NAMES,x)),
                     "frozen_default_fit_score":base,
                     "heldout_default_official_score":held_base,
                     "heldout_selected_official_score":held,
                     "evaluations_used":len(history)})
    _atomic_json(out_dir/"manifest.json",manifest)
    _atomic_json(out_dir/"result.json",manifest)
    return manifest


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dry-run",action="store_true")
    ap.add_argument("--execute",action="store_true")
    ap.add_argument("--out-dir",type=Path)
    ap.add_argument("--ts",default="")
    a=ap.parse_args()
    if a.execute==a.dry_run:
        ap.error("choose exactly one of --dry-run or --execute")
    if a.dry_run:
        print(json.dumps(plan(),indent=2))
        return
    if a.out_dir is None:
        ap.error("--execute requires --out-dir")
    execute(a.out_dir,a.ts)


if __name__=="__main__":
    main()
